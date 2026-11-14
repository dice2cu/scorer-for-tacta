"""Export the trained detector for the web app's own inference engine (no runtime dependencies).

BatchNorm is folded into the convolutions; the network becomes a short list of ops:
  conv   (any kernel/stride/dilation; groups 1 or depthwise), optional residual add, optional ReLU
  upadd  out = a + nearest_upsample_2x(b)
Writes src/data/model/tactanet.json (graph + metadata) and tactanet.bin (float32 weights).
Also writes a parity test case (input + PyTorch output) to out/test-cases/parity/ so the TypeScript
engine can be checked against PyTorch: a crop of the first photo in training-data/games/, or random
noise when there is none.

Usage, from the repo root (tooling.export runs this):
  tooling/.venv/Scripts/python -m tooling.detector.export_web [--ckpt <checkpoint>]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.detector.data import MEAN, STD, normalize
from tooling.detector.model import DECKS, HEADS, ICON_SCALE, N_OUT, STRIDE, SUITS, load_model

OUT_DIR = os.path.join(REPO_ROOT, "src", "data", "model")
PARITY_DIR = paths.PARITY_CASE


class Exporter:
    def __init__(self):
        self.ops, self.chunks, self.offset, self.n = [], [], 0, 0

    def _blob(self, arr):
        arr = np.ascontiguousarray(arr, np.float32).ravel()
        off = self.offset
        self.chunks.append(arr)
        self.offset += arr.size
        return off

    def tensor(self):
        self.n += 1
        return f"t{self.n}"

    def conv(self, src, mod, res=None, relu=None):
        """mod: conv_bn Sequential (Conv2d, BatchNorm2d[, ReLU]) or a bare Conv2d."""
        if isinstance(mod, nn.Conv2d):
            conv, bn, act = mod, None, False
        else:
            conv, bn, act = mod[0], mod[1], len(mod) > 2
        W = conv.weight.detach().double().cpu().numpy()
        b = conv.bias.detach().double().cpu().numpy() if conv.bias is not None else np.zeros(W.shape[0])
        if bn is not None:
            s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().cpu().numpy()
            W = W * s[:, None, None, None]
            b = (b - bn.running_mean.detach().double().cpu().numpy()) * s + bn.bias.detach().double().cpu().numpy()
        g = conv.groups
        assert g == 1 or (g == conv.in_channels == conv.out_channels), "only dense or depthwise convs"
        dst = self.tensor()
        self.ops.append(dict(op="conv", src=src, dst=dst, res=res, relu=bool(act if relu is None else relu),
                             cin=conv.in_channels, cout=conv.out_channels, k=conv.kernel_size[0],
                             stride=conv.stride[0], pad=conv.padding[0], dil=conv.dilation[0],
                             depthwise=g > 1, w=self._blob(W), b=self._blob(b)))
        return dst

    def dwblock(self, src, blk):
        a = self.conv(src, blk.body[0])
        a = self.conv(a, blk.body[1])
        # last conv has no BN-ReLU of its own; the block adds the residual, then ReLU
        return self.conv(a, blk.body[2], res=src if blk.res else None, relu=True)

    def upadd(self, a, b):
        dst = self.tensor()
        self.ops.append(dict(op="upadd", a=a, b=b, dst=dst))
        return dst


def export(model):
    E = Exporter()
    x = E.conv("input", model.stem)
    for blk in model.s4:
        x = E.dwblock(x, blk)
    f4 = x
    for blk in model.s8:
        x = E.dwblock(x, blk)
    f8 = x
    for blk in model.s16:
        x = E.dwblock(x, blk)
    f16 = x
    u8 = E.dwblock(E.upadd(f8, E.conv(f16, model.lat8)), model.fuse8)
    u4 = E.dwblock(E.upadd(f4, E.conv(u8, model.lat4)), model.fuse4)
    h = E.conv(u4, model.head[0])
    h = E.conv(h, model.head[1])
    out = E.conv(h, model.head[2])
    weights = np.concatenate(E.chunks).astype(np.float32)
    meta = dict(format="tactanet-v1", input="input", output=out, input_channels=3, input_multiple=16,
                mean=MEAN.tolist(), std=STD.tolist(), stride=STRIDE, n_out=N_OUT,
                heads=[[n, c] for n, c in HEADS], decks=DECKS, suits=SUITS, icon_scale=ICON_SCALE,
                target_card_width=100.0, weights_count=int(weights.size), ops=E.ops)
    return meta, weights


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=paths.MODEL)
    ap.add_argument("--parity-photo", default=None, help="default: the first photo in training-data/games/")
    args = ap.parse_args(argv)
    model = load_model(args.ckpt)
    meta, weights = export(model)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tactanet.json"), "w") as f:
        json.dump(meta, f, indent=1)
    weights.tofile(os.path.join(OUT_DIR, "tactanet.bin"))
    print(f"{len(meta['ops'])} ops, {weights.size:,} weights ({weights.nbytes / 1e6:.2f} MB) -> {OUT_DIR}")

    # parity case: a crop of a real photo (odd, non-square size multiple of 16)
    photo = args.parity_photo or next(iter(paths.photos(paths.GAMES)), None)
    if photo:
        img = cv2.imread(photo)
        s = max(400 / img.shape[0], 512 / img.shape[1], min(1.0, 1024 / max(img.shape[:2])))
        rgb = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., ::-1]
        cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
        rgb = np.ascontiguousarray(rgb[cy - 200:cy + 200, cx - 256:cx + 256])
    else:
        rgb = np.random.default_rng(1).integers(0, 256, (400, 512, 3), dtype=np.uint8)
    x = torch.from_numpy(normalize(rgb))[None]
    with torch.no_grad():
        y = model(x)[0].numpy()
    os.makedirs(PARITY_DIR, exist_ok=True)
    rgb.tofile(os.path.join(PARITY_DIR, "input_rgb.u8"))
    y.astype(np.float32).tofile(os.path.join(PARITY_DIR, "output.f32"))
    with open(os.path.join(PARITY_DIR, "case.json"), "w") as f:
        json.dump(dict(width=rgb.shape[1], height=rgb.shape[0], out_shape=list(y.shape)), f)
    print(f"parity case {rgb.shape[1]}x{rgb.shape[0]} -> {PARITY_DIR}")


if __name__ == "__main__":
    main()
