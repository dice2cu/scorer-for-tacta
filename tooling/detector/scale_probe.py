"""Probe how detection/fit quality varies with the input scale on real photos (to design the app's
automatic scale search). Prints, per tried card width: fitted card width, cards, dots and the
fraction of confident dots that sit on a printed dot position.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.scale_probe photo [photo ...]
"""
import argparse
import copy
import os

import cv2
import numpy as np
import torch

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.detector import postprocess as P
from tooling.detector.decode import decode, run_model
from tooling.detector.model import load_model


def run(model, img, s):
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    H, W = (h + 15) // 16 * 16, (w + 15) // 16 * 16
    small = cv2.copyMakeBorder(small, 0, H - h, 0, W - w, cv2.BORDER_REPLICATE)
    badges, dots = decode(run_model(model, small[..., ::-1].copy()))
    res = P.score(copy.deepcopy(badges), dots, image_size=(W, H))
    conf = [d for d in dots if d["score"] >= 0.5]
    expl = sum(1 for d in conf if d.get("explained") and d.get("owner") is not None)
    widths = [c["width"] for c in res["cards"]]
    return dict(fit_w=float(np.median(widths)) if widths else 0, cards=len(res["cards"]), dots=sum(res["counts"].values()),
                expl=expl / max(len(conf), 1), conf=len(conf), counts=res["counts"],
                bscore=float(np.mean([b["score"] for b in badges if b["suit"] != "start"] or [0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    args = ap.parse_args()
    model = load_model(paths.MODEL)
    for p in args.photos:
        img = cv2.imread(p)
        long = max(img.shape[:2])
        print(os.path.basename(p), img.shape[1::-1])
        for target_long in [512, 640, 768, 896, 1024, 1280, 1536, 1856]:
            s = target_long / long
            r = run(model, img, s)
            print(f"  long {target_long:5d}: fitted card {r['fit_w']:6.1f}px (photo {r['fit_w'] / s:6.0f})  cards {r['cards']:3d}  "
                  f"dots {r['dots']:3d}  conf dots {r['conf']:3d}  on-template {r['expl']:.2f}  badge score {r['bscore']:.2f}  {r['counts']}")


if __name__ == "__main__":
    main()
