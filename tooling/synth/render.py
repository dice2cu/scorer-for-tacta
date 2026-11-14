"""Render recolored card images from the extracted layer textures (see card_assets.py)."""
import json
import os

import cv2
import numpy as np

from tooling import paths
from tooling.paths import REPO_ROOT

ASSETS = paths.CARD_ASSETS
DECKS = ["blue", "green", "orange", "pink", "purple", "red"]

# Base deck colors (sRGB 0..1 as a phone photo shows them, white ink -> 0.96, black face -> 0.05),
# measured from photos of the real cards (training-data/decks/); jittered per render.
# Red and pink are close: printed red is crimson (almost no green or blue), pink differs mainly
# in its blue. The official art's orange-red made the model call real red cards pink.
DECK_RGB = {
    "blue": (0.40, 0.68, 0.92),
    "green": (0.50, 0.70, 0.26),
    "orange": (0.91, 0.53, 0.10),
    "pink": (0.82, 0.10, 0.40),
    "purple": (0.56, 0.35, 0.73),
    "red": (0.80, 0.06, 0.08),
}


class Assets:
    def __init__(self):
        self.index = json.load(open(os.path.join(ASSETS, "index.json")))
        self.tex_w, self.tex_h = self.index["tex_w"], self.index["tex_h"]
        self.icon_box = self.index["icon_box"]
        self.layers = {}
        self.by_design = {}
        for c in self.index["cards"]:
            # card_assets wrote channels (a_color, a_white, 0) via cv2, which reads back in the
            # same channel order: [..., 0] = a_color, [..., 1] = a_white
            im = cv2.imread(os.path.join(ASSETS, "cards", c["file"]))
            self.layers[c["file"]] = im[..., :2].astype(np.float32) / 255
            self.by_design.setdefault(c["design"], []).append(c)
        self.icons = {d: cv2.imread(os.path.join(ASSETS, "icons", f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                      for d, f in self.index["icons"].items()}

    def textures(self, design, side):
        """Texture entries usable for this design+side ('symmetric' fit both sides)."""
        return [c for c in self.by_design[design] if c["side"] in (side, "symmetric")]


def render_card(assets, entry, deck_rgb, face_rgb=(0.05, 0.05, 0.055), white_rgb=(0.96, 0.96, 0.94), icon=None):
    """RGB float image (tex_h x tex_w x 3) of a card: layers composited with the given colors."""
    a = assets.layers[entry["file"]].copy()
    if icon is not None:
        H, W = a.shape[:2]
        x0, y0, x1, y1 = assets.icon_box
        box = (int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W))
        ic = cv2.resize(icon, (box[3] - box[2], box[1] - box[0]), interpolation=cv2.INTER_LINEAR)
        a[box[0]:box[1], box[2]:box[3], 1] = np.maximum(a[box[0]:box[1], box[2]:box[3], 1], ic)
    K = np.array(face_rgb, np.float32)
    C = np.array(deck_rgb, np.float32)
    Wt = np.array(white_rgb, np.float32)
    img = K + a[..., :1] * (C - K) + a[..., 1:2] * (Wt - K)
    return np.clip(img, 0, 1)


def card_alpha(w, h, radius_frac=0.045):
    """Rounded-rectangle alpha mask for a card of size w x h."""
    m = np.zeros((h, w), np.uint8)
    r = int(round(radius_frac * w))
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255


def preview(out_path=os.path.join(paths.OUT, "synth-preview-recolour.png")):
    assets = Assets()
    rows = []
    for design, side in (("square-3", "front"), ("square-3", "back"), ("triangle-5", "front")):
        entry = assets.textures(design, side)[0]
        tiles = [render_card(assets, entry, DECK_RGB[d], icon=assets.icons[d]) for d in DECKS]
        rows.append(np.hstack([cv2.copyMakeBorder((t * 255).astype(np.uint8), 4, 4, 4, 4,
                                                  cv2.BORDER_CONSTANT, value=(60, 60, 60)) for t in tiles]))
    sheet = np.vstack(rows)[..., ::-1]
    cv2.imwrite(out_path, sheet)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    preview()
