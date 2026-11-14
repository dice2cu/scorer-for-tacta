"""Extract recolorable card textures ("materials") from the deck photos in training-data/decks/.

Each rectified card photo is decomposed per pixel into soft layers over the black face:
    pixel ~= K + a_color * (C - K) + a_white * (Wt - K)
where K = face black, C = this deck's color, Wt = white ink (all measured per card).
Re-rendering with a different C recolors the card while keeping the real print's glow,
blur and line weights. Icons (which differ per deck) are erased and re-pasted per color from
the app's deck icon templates (src/data/cards/icons.json).

Each photo shows one deck (18 cards, 3 rows) or two decks (36 cards, 6 rows) in a grid: rows
circle/square/triangle, columns value 1..6. Cards cut off by the photo's edge are skipped.

Outputs (out/card-assets/):
  cards/<design>__<side>__<k>.png  2-channel layers (a_color, a_white) as 8-bit
  icons/<deck>.png                  icon alpha (white ink), canonical size
  index.json

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.synth.card_assets
"""
import json
import os

import cv2
import numpy as np

from tooling import paths
from tooling.card_library.build_card_library import (
    REPO_ROOT, SUITS, find_cards, rectify, sort_grid, find_dots)

LIB = json.load(open(os.path.join(REPO_ROOT, "src", "data", "cards", "card_library.json")))
ASPECT = LIB["aspect_w_over_h"]
TEX_W = 360
TEX_H = int(round(TEX_W / ASPECT))
OUT = paths.CARD_ASSETS
DECKS = ["blue", "green", "orange", "pink", "purple", "red"]

# icon box (normalized, front orientation) - below the badge, clear of it for every suit
ICON = LIB["cards"][0]["icon"]
ICON_BOX = (ICON["x"] - 0.06, ICON["y"] - 0.032, ICON["x"] + 0.06, ICON["y"] + 0.032)

ICONS_JSON = os.path.join(REPO_ROOT, "src", "data", "cards", "icons.json")


def decompose(card):
    """Soft (a_color, a_white) layers + the measured K, C, Wt colors (RGB 0..1)."""
    rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    chroma = mx - mn
    K = np.median(rgb[mx < np.percentile(mx, 30)], axis=0)
    C = np.median(rgb[chroma > np.percentile(chroma, 90)], axis=0)
    wmask = (mn > np.percentile(mn, 97)) & (chroma < np.percentile(chroma, 50))
    Wt = np.median(rgb[wmask], axis=0) if wmask.sum() > 50 else np.array([0.95, 0.95, 0.95])
    A = np.stack([C - K, Wt - K], axis=1)          # 3x2
    pinv = np.linalg.pinv(A)                       # 2x3
    a = (rgb - K) @ pinv.T                         # HxWx2
    a = np.clip(a, 0, 1)
    s = a.sum(axis=2, keepdims=True)
    a = np.where(s > 1, a / np.maximum(s, 1e-6), a)
    return a, K, C, Wt


def orientation(card, lib_card):
    """'front' or 'back' by matching detected dots to the library (mirror for back)."""
    H, W = card.shape[:2]
    d = find_dots(card)
    ref = [(p["x"] * W, p["y"] * H) for p in lib_card["dots"]]
    if len(d) != len(ref):
        return None

    def cost(pts):
        return np.mean([min(np.hypot(a[0] - b[0], a[1] - b[1]) for b in pts) for a in ref])
    c_front = cost([(x, y) for x, y, _ in d])
    c_back = cost([(W - 1 - x, y) for x, y, _ in d])
    if min(c_front, c_back) > 0.04 * W:
        return None
    if abs(c_front - c_back) < 0.02 * W:
        return "symmetric"
    return "front" if c_front < c_back else "back"


def erase_icon(a):
    """Remove the deck icon from the layers (it is re-pasted per deck when rendering).
    The icon box sits on the black face, so clearing both layers there restores plain face."""
    H, W = a.shape[:2]
    x0, y0, x1, y1 = ICON_BOX
    a[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 0
    return a


def load_icons():
    """Icon alpha per deck (uint8 white ink), from the app's icon templates."""
    J = json.load(open(ICONS_JSON))
    assert J["tex_w"] == TEX_W
    return {d: np.array(e["data"], np.uint8).reshape(e["h"], e["w"]) for d, e in J["icons"].items()}


def main():
    os.makedirs(os.path.join(OUT, "cards"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "icons"), exist_ok=True)
    lib = {c["id"]: c for c in LIB["cards"]}
    index = {"tex_w": TEX_W, "tex_h": TEX_H, "icon_box": ICON_BOX, "cards": [], "icons": {}}
    counts = {}
    photos = paths.photos(paths.DECKS)
    if not photos:
        raise SystemExit(f"no photos in {paths.DECKS}")
    for full in photos:
        path = os.path.basename(full)
        img = cv2.imread(full)
        n = None
        for n_try in (36, 18):
            try:
                quads = find_cards(img, expected=n_try)
                n = n_try
                break
            except SystemExit:
                continue
        if n is None:
            print(f"  skip {path}: couldn't find 18 or 36 cards")
            continue
        quads = sort_grid(quads, rows=n // 6)
        ih, iw = img.shape[:2]
        for i, q in enumerate(quads):
            if (q[:, 0].min() < 10 or q[:, 1].min() < 10 or q[:, 0].max() > iw - 11 or q[:, 1].max() > ih - 11):
                continue  # cut off at photo border
            cid = f"{SUITS[(i // 6) % 3]}-{i % 6 + 1}"
            card = rectify(img, q, TEX_W, TEX_H)
            side = orientation(card, lib[cid])
            if side is None:
                print(f"  skip {cid} in {path}: orientation unclear")
                continue
            a, K, C, Wt = decompose(card)
            a = erase_icon(a)
            k = counts.get((cid, side), 0)
            counts[(cid, side)] = k + 1
            name = f"{cid}__{side}__{k}.png"
            layers = np.dstack([a[..., 0], a[..., 1], np.zeros_like(a[..., 0])])
            cv2.imwrite(os.path.join(OUT, "cards", name), (layers * 255).round().astype(np.uint8))
            index["cards"].append(dict(file=name, design=cid, side=side, source=path,
                                       K=K.round(4).tolist(), C=C.round(4).tolist(), Wt=Wt.round(4).tolist()))
    for deck, icon in load_icons().items():
        cv2.imwrite(os.path.join(OUT, "icons", f"{deck}.png"), icon)
        index["icons"][deck] = f"{deck}.png"
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    by_side = {}
    for c in index["cards"]:
        by_side[c["side"]] = by_side.get(c["side"], 0) + 1
    print(f"wrote {len(index['cards'])} card textures {by_side} and {len(index['icons'])} icons to {OUT}")


if __name__ == "__main__":
    main()
