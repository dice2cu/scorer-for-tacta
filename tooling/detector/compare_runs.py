"""Compare two cached evaluation runs (evaluate.py --save-cache) board by board: pipeline count
errors, and how many visible ground-truth dots were decoded as "partly covered" (lost counts).

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.compare_runs A.pkl B.pkl
"""
import argparse
import copy
import pickle
from collections import Counter

import numpy as np

from tooling.detector import postprocess as P
from tooling.detector.evaluate import DOT_R


def board_stats(item):
    gt = item["gt"]
    w, h = item["size"]
    inside = lambda p: 8 <= p[0] < w - 8 and 8 <= p[1] < h - 8
    in_cards = {c["idx"] for c in gt["cards"] if inside(c["badge"])}
    gp = Counter(d["deck"] for d in gt["dots"] if d["visible"] and inside(d["xy"]) and d["card"] in in_cards)
    res = P.score(copy.deepcopy(item["badges"]), copy.deepcopy(item["dots"]), item["size"])
    err = sum(abs(res["counts"].get(k, 0) - gp.get(k, 0)) for k in set(gp) | set(res["counts"]))
    vis = np.array([d["xy"] for d in gt["dots"] if d["visible"] and inside(d["xy"])] or np.zeros((0, 2)))
    lost = 0
    for d in item["dots"]:
        if d.get("partial") and len(vis):
            if np.min(np.hypot(vis[:, 0] - d["x"], vis[:, 1] - d["y"])) < 1.2 * DOT_R * gt["px_per_card"]:
                lost += 1
    signed = sum(res["counts"].get(k, 0) - gp.get(k, 0) for k in set(gp) | set(res["counts"]))
    return err, lost, signed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    A, B = pickle.load(open(args.a, "rb")), pickle.load(open(args.b, "rb"))
    sa = [board_stats(x) for x in A]
    sb = [board_stats(x) for x in B]
    for name, s in (("A", sa), ("B", sb)):
        e = np.array([x[0] for x in s])
        print(f"{name}: exact {(e == 0).sum()}/{len(e)}  mean abs err {e.mean():.2f}  "
              f"visible dots decoded as partial: {sum(x[1] for x in s)}  net count bias {sum(x[2] for x in s):+d}")
    worse = [(i, a, b) for i, (a, b) in enumerate(zip(sa, sb)) if b[0] > a[0]]
    better = [(i, a, b) for i, (a, b) in enumerate(zip(sa, sb)) if b[0] < a[0]]
    print(f"B worse on {len(worse)} boards, better on {len(better)}")
    for i, a, b in sorted(worse, key=lambda t: t[1][0] - t[2][0])[:12]:
        print(f"  {A[i]['name']}: err {a[0]} -> {b[0]} (signed {a[2]:+d} -> {b[2]:+d}), real dots decoded partial in B: {b[1]}")


if __name__ == "__main__":
    main()
