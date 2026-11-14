"""Write test cases that let the TypeScript app be checked against this Python pipeline.

In out/test-cases/scoring/:
fullboard.json  decoded detector output of cached synthetic test boards (evaluate.py
                --save-cache), Python's scoring result and the ground-truth counts.
photos.json     for real photos: the model input size, Python's decoded output and
                scoring result (same steps as the app, photo.py).

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.make_web_cases --cache <evaluate.py cache> [photo ...]
Without photos, the first photo in training-data/games/counts.json is used. tooling.export runs this.
"""
import argparse
import copy
import json
import os
import pickle
from collections import Counter

import cv2

from tooling import paths
from tooling.detector import postprocess as P
from tooling.detector.model import load_model
from tooling.detector.photo import analyze

OUT = paths.SCORING_CASES


def summary(res):
    return dict(counts=res["counts"], cards_per_deck=res["cards_per_deck"], scale_factor=res["scale_factor"],
                perspective_fit=res["perspective_fit"], n_review=len(res["review"]), n_partial=len(res["partial"]),
                designs=sorted(f"{c['deck']} {c['design']} {c['side']}" for c in res["cards"]))


def plain(o):
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def default_photos():
    if not os.path.exists(paths.COUNTS):
        return []
    names = list(json.load(open(paths.COUNTS)))
    return [os.path.join(paths.GAMES, names[0])] if names else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="*")
    ap.add_argument("--cache", required=True, help="evaluate.py --save-cache output to turn into synthetic cases")
    ap.add_argument("--ckpt", default=paths.MODEL)
    args = ap.parse_args(argv)
    args.photos = args.photos or default_photos()
    os.makedirs(OUT, exist_ok=True)

    cases = []
    for it in pickle.load(open(args.cache, "rb")):
        res = P.score(copy.deepcopy(it["badges"]), copy.deepcopy(it["dots"]), it["size"])
        gt = Counter(d["deck"] for d in it["gt"]["dots"] if d["visible"])
        cases.append(dict(name=it["name"], size=list(it["size"]), badges=plain(it["badges"]), dots=plain(it["dots"]),
                          python=plain(summary(res)), gt=dict(gt)))
    with open(os.path.join(OUT, "fullboard.json"), "w") as f:
        json.dump(cases, f)
    print(f"{len(cases)} synthetic cases")

    model = load_model(args.ckpt)
    photos = []
    for path in args.photos:
        a = analyze(model, cv2.imread(path))
        badges, dots = a["decoded"]
        photos.append(dict(name=os.path.basename(path), card_w=a["card_w"], size=list(a["size"]),
                           badges=plain(badges), dots=plain(dots), python=plain(summary(a["result"]))))
        print(f"{os.path.basename(path)}: card width {a['card_w']:.0f}px, counts {a['result']['counts']}")
    with open(os.path.join(OUT, "photos.json"), "w") as f:
        json.dump(photos, f)


if __name__ == "__main__":
    main()
