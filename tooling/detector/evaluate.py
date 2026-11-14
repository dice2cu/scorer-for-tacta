"""Evaluate a trained TactaNet on held-out synthetic boards (exact ground truth).

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.evaluate [--ckpt <checkpoint>] [--n 300]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.detector.decode import decode, run_model
from tooling.detector.model import DECKS, load_model

LIB = json.load(open(os.path.join(REPO_ROOT, "src", "data", "cards", "card_library.json")))
DOT_R = LIB["dot_radius_over_width"]


def match(pred, gt, max_d):
    """Greedy nearest matching. Returns list of (pred_i, gt_j)."""
    if not pred or not gt:
        return []
    P = np.array(pred, float)
    G = np.array(gt, float)
    d = np.linalg.norm(P[:, None] - G[None], axis=2)
    pairs, used_p, used_g = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[i, j] > max_d:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); pairs.append((i, j))
    return pairs


def normalize_scale(img_bgr, gt, target=100.0, jitter=0.0, rng=None):
    """Rescale like the app does (cards ~target px wide), using the known scale times an optional
    random error (simulating the app's scale estimate). Ground truth is transformed to match."""
    s = target / gt["px_per_card"]
    if jitter and rng is not None:
        s *= float(np.exp(rng.uniform(-jitter, jitter)))
    img = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    H, W = (h + 15) // 16 * 16, (w + 15) // 16 * 16
    img = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_REPLICATE)
    g = json.loads(json.dumps(gt))
    for c in g["cards"]:
        c["badge"] = [v * s for v in c["badge"]]; c["icon"] = [v * s for v in c["icon"]]
        c["card_w"] *= s; c["corners"] = [[v * s for v in p] for p in c["corners"]]
    for d in g["dots"]:
        d["xy"] = [v * s for v in d["xy"]]
    g["px_per_card"] *= s; g["width"], g["height"] = W, H
    g["scale"] = s * gt.get("scale", 1.0)  # model px per original image px
    return img, g


def evaluate_board(model, img_bgr, gt, device, margin=8, photo=None):
    """photo: the original-resolution image (the app reads deck icons from it); None = no icons."""
    rgb = img_bgr[..., ::-1].copy()
    out = run_model(model, rgb, device)
    badges, dots = decode(out)
    if photo is not None:
        from tooling.detector.icons import read_icons
        read_icons(photo, badges, 1 / gt.get("scale", 1.0))
    h, w = rgb.shape[:2]
    inside = lambda p: margin <= p[0] < w - margin and margin <= p[1] < h - margin
    ppc = gt["px_per_card"]
    st = {}

    gb = [c for c in gt["cards"] if inside(c["badge"])]
    pb = [b for b in badges if inside((b["x"], b["y"]))]
    pairs = match([(b["x"], b["y"]) for b in pb], [c["badge"] for c in gb], 0.08 * ppc)
    st["badge_tp"], st["badge_fn"], st["badge_fp"] = len(pairs), len(gb) - len(pairs), len(pb) - len(pairs)
    st["suit_ok"] = st["value_ok"] = st["side_ok"] = st["deck_ok"] = st["n_cls"] = st["icon_ok"] = st["icon_n"] = 0
    st["icon_err"] = []
    for i, j in pairs:
        b, c = pb[i], gb[j]
        suit = "start" if c["design"] == "start" else c["design"].split("-")[0]
        st["suit_ok"] += b["suit"] == suit
        if suit != "start":
            st["n_cls"] += 1
            st["value_ok"] += b["value"] == int(c["design"].split("-")[1])
            st["side_ok"] += b["side"] == c["side"]
            st["deck_ok"] += b["deck"] == c["deck"]
            if b.get("icon_scores") is not None:
                st["icon_n"] += 1
                st["icon_ok"] += DECKS[int(np.argmax(b["icon_scores"]))] == c["deck"]
        v_p = np.subtract(b["icon"], (b["x"], b["y"]))
        v_g = np.subtract(c["icon"], c["badge"])
        st["icon_err"].append(abs(np.degrees(np.arctan2(np.cross(v_p, v_g), np.dot(v_p, v_g)))))

    gd = [d for d in gt["dots"] if d["visible"] and inside(d["xy"])]
    pd = [d for d in dots if inside((d["x"], d["y"])) and not d.get("partial")]
    # partly covered dots (raw detector): found vs ground truth
    gpart = [d for d in gt["dots"] if d.get("partial") and inside(d["xy"])]
    ppart = [d for d in dots if inside((d["x"], d["y"])) and d.get("partial")]
    pairs_p = match([(d["x"], d["y"]) for d in ppart], [d["xy"] for d in gpart], 1.5 * DOT_R * ppc)
    st["part_tp"], st["part_fn"], st["part_fp"] = len(pairs_p), len(gpart) - len(pairs_p), len(ppart) - len(pairs_p)
    pairs = match([(d["x"], d["y"]) for d in pd], [d["xy"] for d in gd], 1.2 * DOT_R * ppc)
    st["dot_tp"], st["dot_fn"], st["dot_fp"] = len(pairs), len(gd) - len(pairs), len(pd) - len(pairs)
    st["dot_deck_ok"] = sum(pd[i]["deck"] == gd[j]["deck"] for i, j in pairs)

    gc = {k: 0 for k in DECKS}
    pc = {k: 0 for k in DECKS}
    for d in gd:
        gc[d["deck"]] += 1
    for d in pd:
        pc[d["deck"]] += 1
    st["count_abs_err"] = sum(abs(gc[k] - pc[k]) for k in DECKS)
    st["count_exact"] = int(st["count_abs_err"] == 0)
    st["dots_total"] = len(gd)

    # full pipeline (what the app reports): only dots on detected cards count. Ground truth for
    # this: visible dots in the image whose card's badge is also in the image.
    from tooling.detector.postprocess import score
    import copy
    res = score(copy.deepcopy(badges), copy.deepcopy(dots), (w, h))
    in_cards = {c["idx"] for c in gt["cards"] if inside(c["badge"])}
    gp = {k: 0 for k in DECKS}
    for d in gt["dots"]:
        if d["visible"] and inside(d["xy"]) and d["card"] in in_cards:
            gp[d["deck"]] += 1
    st["pipe_abs_err"] = sum(abs(gp[k] - res["counts"].get(k, 0)) for k in DECKS)
    st["pipe_exact"] = int(st["pipe_abs_err"] == 0)
    st["pipe_review"] = len(res["review"])
    st["pipe_partial"] = len(res["partial"])
    return st, badges, dots


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=paths.MODEL)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cpu", help="cpu (default; safe while training) or cuda")
    ap.add_argument("--native", action="store_true", help="don't rescale to ~100 px per card")
    ap.add_argument("--scale-jitter", type=float, default=0.15, help="simulated scale-estimate error (log)")
    ap.add_argument("--save-cache", default=None, help="pickle the decoded outputs (for tune_policy.py)")
    ap.add_argument("--no-icons", action="store_true", help="don't read deck icons (colour only, as before v8)")
    args = ap.parse_args(argv)
    dev = args.device
    model = load_model(args.ckpt, dev)
    files = sorted(glob.glob(os.path.join(paths.BOARDS, args.split, "*.json")))[:args.n]
    tot = {}
    icon_errs = []
    rng = np.random.default_rng(0)
    cache = []
    for f in files:
        img, gt = cv2.imread(f[:-5] + ".jpg"), json.load(open(f))
        photo = None if args.no_icons else img
        if not args.native:
            img, gt = normalize_scale(img, gt, jitter=args.scale_jitter, rng=rng)
        st, badges, dots = evaluate_board(model, img, gt, dev, photo=photo)
        if args.save_cache:
            cache.append(dict(name=os.path.basename(f)[:-5], size=(img.shape[1], img.shape[0]), gt=gt, badges=badges, dots=dots))
        icon_errs += st.pop("icon_err")
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
    n = len(files)
    br = tot["badge_tp"] / max(1, tot["badge_tp"] + tot["badge_fn"])
    bp = tot["badge_tp"] / max(1, tot["badge_tp"] + tot["badge_fp"])
    dr = tot["dot_tp"] / max(1, tot["dot_tp"] + tot["dot_fn"])
    dp = tot["dot_tp"] / max(1, tot["dot_tp"] + tot["dot_fp"])
    print(f"boards {n}")
    print(f"badges: recall {br:.4f}  precision {bp:.4f}  (missed {tot['badge_fn']}, false {tot['badge_fp']})")
    print(f"  suit {tot['suit_ok'] / max(1, tot['badge_tp']):.4f}  value {tot['value_ok'] / max(1, tot['n_cls']):.4f}"
          f"  side {tot['side_ok'] / max(1, tot['n_cls']):.4f}  deck {tot['deck_ok'] / max(1, tot['n_cls']):.4f}"
          + (f"  deck icons {tot['icon_ok'] / max(1, tot['icon_n']):.4f} (read {tot['icon_n']}/{tot['n_cls']})" if tot["icon_n"] else "")
          + (f"  icon angle err median {np.median(icon_errs):.1f} deg, p99 {np.percentile(icon_errs, 99):.1f} deg"
             if icon_errs else ""))
    print(f"dots:   recall {dr:.4f}  precision {dp:.4f}  (missed {tot['dot_fn']}, false {tot['dot_fp']} of {tot['dots_total']})"
          f"  owner colour {tot['dot_deck_ok'] / max(1, tot['dot_tp']):.4f}")
    print(f"counts: boards with every colour exact {tot['count_exact']}/{n}  "
          f"mean abs error per board {tot['count_abs_err'] / n:.2f} dots")
    pr = tot["part_tp"] / max(1, tot["part_tp"] + tot["part_fn"])
    pp = tot["part_tp"] / max(1, tot["part_tp"] + tot["part_fp"])
    print(f"partly covered dots: recall {pr:.3f}  precision {pp:.3f}  (of {tot['part_tp'] + tot['part_fn']}; false {tot['part_fp']})")
    if args.save_cache:
        import pickle
        with open(args.save_cache, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"cached decoded outputs -> {args.save_cache}")
    print(f"PIPELINE (detector + card geometry): boards with every colour exact {tot['pipe_exact']}/{n}  "
          f"mean abs error per board {tot['pipe_abs_err'] / n:.2f} dots  (sent to review: {tot['pipe_review']}, flagged partial: {tot['pipe_partial']})")
    return dict(boards=n, exact=tot["pipe_exact"], mean_abs_err=tot["pipe_abs_err"] / max(1, n))


if __name__ == "__main__":
    main()
