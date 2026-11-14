"""Check the app's deck-icon reading (src/pipeline/icons.ts) against icons.py on real photos.

app-e2e (node tests/browser/app-e2e.mjs) saves each photo's detections, including the icon
scores the browser read, in out/browser/<photo>_cards.json. This re-reads every icon in Python
from the same photo and compares. Expected: identical calls, scores within ~1e-6 (both sample the
photo with the same float bilinear arithmetic; only JPEG decoding could differ).

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.check_icons
"""
import copy
import json
import os
import sys

import cv2
import numpy as np

from tooling import paths
from tooling.detector.icons import read_icons


def main():
    n = same = 0
    worst = 0.0
    for path in paths.photos(paths.GAMES) + paths.photos(paths.DECKS):
        stem = os.path.splitext(os.path.basename(path))[0]
        f = os.path.join(paths.OUT, "browser", f"{stem}_cards.json")
        if not os.path.exists(f):
            continue
        J = json.load(open(f))
        if "decoded" not in J:
            continue
        ts = J["decoded"]["badges"]
        py = read_icons(cv2.imread(path), copy.deepcopy(ts), 1 / J["scale"])
        bad = 0
        for a, b in zip(ts, py):
            if a["suit"] == "start":
                continue
            n += 1
            sa, sb = a.get("icon_scores"), b.get("icon_scores")
            if (sa is None) != (sb is None):
                bad += 1
                continue
            if sa is None:
                same += 1
                continue
            same += int(np.argmax(sa) == np.argmax(sb))
            bad += int(np.argmax(sa) != np.argmax(sb))
            worst = max(worst, float(np.max(np.abs(np.subtract(sa, sb)))))
        print(f"{stem:32s} {len(ts):3d} badges  {'ok' if not bad else f'{bad} DIFFERENT'}")
    print(f"same call {same}/{n}, largest score difference {worst:.2e}")
    sys.exit(0 if same == n else 1)


if __name__ == "__main__":
    main()
