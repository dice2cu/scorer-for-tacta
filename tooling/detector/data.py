"""Training data: random crops of synthetic boards -> CenterNet-style targets."""
import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.detector.model import DECKS, ICON_SCALE, STRIDE, SUITS

LIB = json.load(open(os.path.join(REPO_ROOT, "src", "data", "cards", "card_library.json")))
DOT_R = LIB["dot_radius_over_width"]
MEAN = np.array([0.45, 0.42, 0.40], np.float32)
STD = np.array([0.28, 0.27, 0.27], np.float32)


def normalize(rgb_uint8):
    """HxWx3 uint8 RGB -> 3xHxW float32 (same function is mirrored in the app)."""
    x = rgb_uint8.astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def gaussian(hm, cx, cy, sigma):
    """Splat a peak centred on the object's CELL (value exactly 1 there, as CenterNet's focal loss
    requires for its positives); the sub-cell position is learned by the offset head."""
    h, w = hm.shape
    gx, gy = int(cx), int(cy)
    r = int(3 * sigma + 1)
    x0, x1 = max(0, gx - r), min(w, gx + r + 1)
    y0, y1 = max(0, gy - r), min(h, gy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2))
    np.maximum(hm[y0:y1, x0:x1], g, out=hm[y0:y1, x0:x1])


def rot90_points(pts, k, w, h):
    """Coordinates after np.rot90(img, k) (counter-clockwise) of an image w x h."""
    pts = np.array(pts, np.float64).reshape(-1, 2)
    for _ in range(k % 4):
        pts = np.stack([pts[:, 1], (w - 1) - pts[:, 0]], axis=1)
        w, h = h, w
    return pts


class BoardCrops(Dataset):
    def __init__(self, split="train", crop=384, crops_per_board=2, augment=True):
        # split: one name, or several joined with "+" (e.g. "train+train2")
        self.files = sorted(f for s in split.split("+")
                            for f in glob.glob(os.path.join(paths.BOARDS, s, "*.json")))
        self.crop, self.cpb, self.augment = crop, crops_per_board, augment

    def __len__(self):
        return len(self.files) * self.cpb

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.augment else i)
        path = self.files[i // self.cpb]
        gt = json.load(open(path))
        img = cv2.imread(path[:-5] + ".jpg")[..., ::-1]  # RGB
        H, W = img.shape[:2]
        S = self.crop
        x0 = int(rng.integers(0, W - S + 1))
        y0 = int(rng.integers(0, H - S + 1))
        img = img[y0:y0 + S, x0:x0 + S]
        off = np.array([x0, y0], np.float64)
        k = int(rng.integers(4)) if self.augment else 0
        img = np.ascontiguousarray(np.rot90(img, k))

        if self.augment:
            f = img.astype(np.float32)
            f = f * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)            # contrast/brightness
            gray = f.mean(axis=2, keepdims=True)
            f = gray + (f - gray) * rng.uniform(0.8, 1.2)                      # saturation (no hue shift)
            img = np.clip(f, 0, 255).astype(np.uint8)

        G = S // STRIDE
        t = {
            "badge_hm": np.zeros((4, G, G), np.float32),
            "dot_hm": np.zeros((1, G, G), np.float32),
            "part_hm": np.zeros((1, G, G), np.float32),  # partly covered dots
            "badge_reg": np.zeros((4, G, G), np.float32),   # off x, off y, icon dx, icon dy
            "badge_mask": np.zeros((G, G), np.float32),     # cells with regression targets
            "cls_mask": np.zeros((G, G), np.float32),       # cells with value/side/deck targets
            "value": np.zeros((G, G), np.int64),
            "side": np.zeros((G, G), np.float32),
            "deck": np.zeros((G, G), np.int64),
            "dot_reg": np.zeros((2, G, G), np.float32),
            "dot_mask": np.zeros((G, G), np.float32),
            "dot_deck": np.zeros((G, G), np.int64),
        }
        for c in gt["cards"]:
            b = rot90_points(np.array(c["badge"]) - off, k, S, S)[0]
            ic = rot90_points(np.array(c["icon"]) - off, k, S, S)[0]
            if not (0 <= b[0] < S and 0 <= b[1] < S):
                continue
            suit = 3 if c["design"] == "start" else SUITS.index(c["design"].split("-")[0])
            cx, cy = b / STRIDE
            sigma = max(0.7, 0.03 * c["card_w"] / STRIDE)
            gaussian(t["badge_hm"][suit], cx, cy, sigma)
            gx, gy = int(cx), int(cy)
            t["badge_reg"][:, gy, gx] = [cx - gx, cy - gy, (ic[0] - b[0]) / ICON_SCALE, (ic[1] - b[1]) / ICON_SCALE]
            t["badge_mask"][gy, gx] = 1
            if suit < 3:
                ys, xs = slice(max(0, gy - 1), gy + 2), slice(max(0, gx - 1), gx + 2)
                t["cls_mask"][ys, xs] = 1
                t["value"][ys, xs] = int(c["design"].split("-")[1]) - 1
                t["side"][ys, xs] = 1.0 if c["side"] == "back" else 0.0
                t["deck"][ys, xs] = DECKS.index(c["deck"])
        card_w_of = {c["idx"]: c["card_w"] for c in gt["cards"] if "idx" in c}
        for d in gt["dots"]:
            partial = d.get("partial", False)
            if not d["visible"] and not partial:
                continue
            p = rot90_points(np.array(d["xy"]) - off, k, S, S)[0]
            if not (0 <= p[0] < S and 0 <= p[1] < S):
                continue
            cx, cy = p / STRIDE
            card_w = card_w_of.get(d.get("card"), gt["px_per_card"])
            hm = t["part_hm"] if partial else t["dot_hm"]
            gaussian(hm[0], cx, cy, max(0.6, DOT_R * card_w / STRIDE / 1.5))
            gx, gy = int(cx), int(cy)
            t["dot_reg"][:, gy, gx] = [cx - gx, cy - gy]
            t["dot_mask"][gy, gx] = 1
            t["dot_deck"][gy, gx] = DECKS.index(d.get("colour", d["deck"]))  # the colour shown (see board.py recolour)
        x = torch.from_numpy(normalize(img))
        return x, {k_: torch.from_numpy(v) for k_, v in t.items()}
