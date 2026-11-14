"""Re-score cached detector outputs (evaluate.py --save-cache, e.g. out/runs/<run>/test.pkl) under different counting
policies, in parallel. Lets the classical post-processing be tuned without re-running the model.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.tune_policy --cache out/runs/<run>/test.pkl
"""
import copy
import os
import pickle
from collections import Counter
from multiprocessing import Pool

import numpy as np

VARIANTS = None  # set by --variants (python expression: list of (name, {param: value}))


def run_one(args):
    item, params = args
    from tooling.detector import postprocess as P, perspective as PS
    for k, v in params.items():
        mod, _, name = k.rpartition(".")
        setattr(PS if mod == "perspective" else P, name, v)
    res = P.score(copy.deepcopy(item["badges"]), copy.deepcopy(item["dots"]), item["size"])
    gt = item["gt"]
    w, h = item["size"]
    inside = lambda p: 8 <= p[0] < w - 8 and 8 <= p[1] < h - 8
    in_cards = {c["idx"] for c in gt["cards"] if inside(c["badge"])}
    gp = Counter(d["deck"] for d in gt["dots"] if d["visible"] and inside(d["xy"]) and d["card"] in in_cards)
    err = sum(abs(res["counts"].get(k, 0) - gp.get(k, 0)) for k in set(gp) | set(res["counts"]))
    return err


def evaluate(params, cache, pool):
    errs = pool.map(run_one, [(it, params) for it in cache])
    errs = np.array(errs)
    return int((errs == 0).sum()), float(errs.mean()), Counter(errs.tolist())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--variants", default=None, help="python list of (name, {param: value}); perspective.X sets perspective.py constants")
    args = ap.parse_args()
    cache = pickle.load(open(args.cache, "rb"))
    variants = eval(args.variants) if args.variants else [
        ("current settings", {}),
        ("explained >= 0.30", {"MIN_SCORE_EXPLAINED": 0.30}),
        ("explained >= 0.20", {"MIN_SCORE_EXPLAINED": 0.20}),
        ("explained >= 0.20, on-card >= 0.40", {"MIN_SCORE_EXPLAINED": 0.20, "MIN_SCORE_ON_CARD": 0.40}),
    ]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for name, params in variants:
            exact, mean, hist = evaluate(params, cache, pool)
            big = sum(v for k, v in hist.items() if k > 3)
            print(f"{name:40s} exact {exact}/{len(cache)}  mean abs err {mean:.2f}  boards off by >3: {big}", flush=True)


if __name__ == "__main__":
    main()
