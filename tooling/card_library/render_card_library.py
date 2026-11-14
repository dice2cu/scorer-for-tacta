"""Render src/data/cards/card_library.json (+ masks) as a review sheet: every design, front and back.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.card_library.render_card_library
"""
import argparse
import json
import os

import cv2
import numpy as np

from tooling import paths
from tooling.paths import REPO_ROOT

COLORS = {0: (25, 25, 25), 1: (60, 150, 60), 2: (235, 235, 235)}  # black face, deck color, white ink


def render(card, lib, data_dir, w=300, mirror=False):
    h = int(round(w / lib["aspect_w_over_h"]))
    mask = cv2.imread(os.path.join(data_dir, "card_masks", f"{card['id']}.png"), cv2.IMREAD_GRAYSCALE)
    cls = np.round(mask / 127).astype(np.uint8)
    if mirror:
        cls = cls[:, ::-1]
    img = np.zeros((*cls.shape, 3), np.uint8)
    for c, col in COLORS.items():
        img[cls == c] = col
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    img = (img * 0.55).astype(np.uint8)  # dim the mask so the overlay stands out

    def px(p):
        x = 1 - p["x"] if mirror else p["x"]
        return int(round(x * w)), int(round(p["y"] * h))

    r = int(round(lib["dot_radius_over_width"] * w))
    for d in card["dots"]:
        cv2.circle(img, px(d), r, (255, 255, 255), -1)
        cv2.circle(img, px(d), r + 4, (0, 220, 255), 2)
    b = card["badge"]
    bx, by = px(b)
    bw, bh = int(b["w"] * w / 2), int(b["h"] * h / 2)
    cv2.rectangle(img, (bx - bw, by - bh), (bx + bw, by + bh), (255, 200, 0), 2)
    if card.get("icon"):
        cv2.drawMarker(img, px(card["icon"]), (255, 0, 255), cv2.MARKER_DIAMOND, 14, 2)
    label = f"{card['id']} {'back' if mirror else 'front'}"
    cv2.putText(img, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(len(card["dots"])), (w - 28, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(80, 80, 80))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "src", "data", "cards"))
    ap.add_argument("--out", default=os.path.join(paths.OUT, "card-library-preview.png"))
    args = ap.parse_args()
    lib = json.load(open(os.path.join(args.data, "card_library.json")))
    cards = lib["cards"]
    rows = []
    for mirror in (False, True):
        for s in range(3):
            rows.append(np.hstack([render(c, lib, args.data, mirror=mirror) for c in cards[s * 6:(s + 1) * 6]]))
    sheet = np.vstack(rows)
    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out}  ({len(cards)} designs; rows 1-3 front, rows 4-6 back)")


if __name__ == "__main__":
    main()
