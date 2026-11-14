"""Train TactaNet on synthetic boards.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.train --epochs 40
Checkpoints: <--out>/last.pt and best.pt (tooling.train runs this with out/runs/<name>/)
"""
import argparse
import math
import os
import time

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.detector.data import BoardCrops
from tooling.detector.model import TactaNet, head_slices, count_params

SL = head_slices()


def focal(logits, target):
    """CenterNet focal loss on heatmap logits, normalised by the number of peaks."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.999).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def masked_l1(pred, target, mask):
    m = mask.unsqueeze(1)
    return (F.l1_loss(pred * m, target * m, reduction="sum") / m.sum().clamp(min=1))


def masked_ce(logits, target, mask):
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def compute_loss(out, t):
    out = out.float()
    L = {}
    L["badge_hm"] = focal(out[:, SL["badge_hm"]], t["badge_hm"])
    L["dot_hm"] = focal(out[:, SL["dot_hm"]], t["dot_hm"])
    L["dot_part"] = focal(out[:, SL["dot_part"]], t["part_hm"])
    reg = torch.cat([out[:, SL["badge_off"]].sigmoid(), out[:, SL["icon"]]], dim=1)
    L["badge_reg"] = masked_l1(reg, t["badge_reg"], t["badge_mask"])
    L["value"] = 0.5 * masked_ce(out[:, SL["value"]], t["value"], t["cls_mask"])
    side_l = F.binary_cross_entropy_with_logits(out[:, SL["side"]][:, 0], t["side"], reduction="none")
    L["side"] = 0.5 * (side_l * t["cls_mask"]).sum() / t["cls_mask"].sum().clamp(min=1)
    L["deck"] = 0.5 * masked_ce(out[:, SL["deck"]], t["deck"], t["cls_mask"])
    L["dot_reg"] = masked_l1(out[:, SL["dot_off"]].sigmoid(), t["dot_reg"], t["dot_mask"])
    L["dot_deck"] = 0.5 * masked_ce(out[:, SL["dot_deck"]], t["dot_deck"], t["dot_mask"])
    return sum(L.values()), L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="start from these weights (fine-tune); new head channels are kept fresh")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(paths.RUNS, "manual"), help="folder for checkpoints")
    args = ap.parse_args(argv)
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BoardCrops(args.train_split, augment=True)
    val_ds = BoardCrops(args.val_split, crops_per_board=1, augment=False)
    print(f"train crops/epoch {len(train_ds)}, val crops {len(val_ds)}, device {dev}")
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, persistent_workers=True)

    model = TactaNet().to(dev)
    print(f"params {count_params(model):,}")
    if args.init:
        own = model.state_dict()
        src = torch.load(args.init, map_location=dev)["model"]
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
            elif k in own and own[k].shape[1:] == v.shape[1:] and own[k].shape[0] > v.shape[0]:
                own[k][: v.shape[0]] = v   # output layer grew: copy the existing channels
                print(f"init: {k} {tuple(v.shape)} -> {tuple(own[k].shape)} (new channels fresh)")
        model.load_state_dict(own)
        print(f"initialised from {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    start, best = 0, float("inf")
    if args.resume and os.path.exists(os.path.join(OUT, "last.pt")):
        ck = torch.load(os.path.join(OUT, "last.pt"), map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start}")

    for epoch in range(start, args.epochs):
        model.train()
        t0, agg, n = time.time(), {}, 0
        for x, t in tl:
            x = x.to(dev, non_blocking=True)
            t = {k: v.to(dev, non_blocking=True) for k, v in t.items()}
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model(x)
            loss, parts = compute_loss(out, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + float(v)
            n += 1
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for x, t in vl:
                x = x.to(dev)
                t = {k: v.to(dev) for k, v in t.items()}
                loss, _ = compute_loss(model(x), t)
                vloss += float(loss); vn += 1
        vloss /= max(vn, 1)
        parts = " ".join(f"{k} {v / n:.3f}" for k, v in agg.items())
        print(f"epoch {epoch:3d}  {time.time() - t0:5.0f}s  lr {sched.get_last_lr()[0]:.2e}  val {vloss:.4f}  | {parts}",
              flush=True)
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch, best=best)
        if vloss < best:
            best = vloss
            ck["best"] = best
            torch.save(ck, os.path.join(OUT, "best.pt"))
        torch.save(ck, os.path.join(OUT, "last.pt"))
    print(f"done; best val {best:.4f}")


if __name__ == "__main__":
    main()
