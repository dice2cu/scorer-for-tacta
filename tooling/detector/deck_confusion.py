"""Card colour confusion of the full pipeline on a cached evaluation run (evaluate.py --save-cache):
for every scored card matched to a ground-truth card, true deck vs the deck the pipeline used.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.deck_confusion out/runs/<run>/test.pkl
"""
import copy
import pickle
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from tooling.detector import postprocess as P
from tooling.detector.model import DECKS


def one(item):
    res = P.score(copy.deepcopy(item["badges"]), copy.deepcopy(item["dots"]), item["size"])
    gb = np.array([c["badge"] for c in item["gt"]["cards"]])
    pairs = Counter()
    for c in res["cards"]:
        b = item["badges"][c["badge"]]
        j = int(np.argmin(np.hypot(gb[:, 0] - b["x"], gb[:, 1] - b["y"])))
        if np.hypot(*(gb[j] - (b["x"], b["y"]))) < 0.1 * c["width"]:
            pairs[(item["gt"]["cards"][j]["deck"], c["deck"], b["deck"])] += 1
    return pairs


def main():
    cache = pickle.load(open(sys.argv[1], "rb"))
    tot = Counter()
    with Pool(14) as pool:
        for p in pool.map(one, cache):
            tot.update(p)
    n = sum(tot.values())
    wrong_final = sum(v for (g, f, _), v in tot.items() if g != f)
    wrong_badge = sum(v for (g, _, b), v in tot.items() if g != b)
    print(f"cards {n}: badge colour wrong {wrong_badge} ({wrong_badge / n:.2%}), after dot vote wrong {wrong_final} ({wrong_final / n:.2%})")
    conf = Counter()
    for (g, f, _), v in tot.items():
        if g != f:
            conf[(g, f)] += v
    for (g, f), v in conf.most_common(12):
        print(f"  true {g:7s} -> used {f:7s}: {v}")


if __name__ == "__main__":
    main()
