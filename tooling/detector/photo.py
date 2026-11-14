"""Photo -> scores with the same steps as the app (src/pipeline/analyze.ts): a first pass at a fixed
size finds and measures the cards, the photo is re-run at the size the detector was trained for
(~100 px per card), then each card's deck icon is read from the full-resolution photo and the
cards are scored. Only the resizing differs slightly (cv2 here, a canvas in the browser).
"""
import copy

import cv2
import numpy as np

from tooling.detector import postprocess as P
from tooling.detector.decode import decode, run_model
from tooling.detector.icons import read_icons

TARGET_CARD_W = 100.0
FIRST_PASS_SIZES = [1024, 1536, 704]  # long side, tried in order until cards are found
CLOSE_ENOUGH = 0.08  # reuse a pass if its card width is within 8% of the target
MIN_CARDS = 6
MAX_LONG_SIDE = 2560
MAX_UPSCALE = 2


def _round(x):
    return int(np.floor(x + 0.5))  # as Math.round


def _run_at(model, img, scale, device):
    h, w = img.shape[:2]
    w0, h0 = max(16, _round(w * scale)), max(16, _round(h * scale))
    small = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    W, H = (w0 + 15) // 16 * 16, (h0 + 15) // 16 * 16
    small = cv2.copyMakeBorder(small, 0, H - h0, 0, W - w0, cv2.BORDER_REPLICATE)
    badges, dots = decode(run_model(model, small[..., ::-1].copy(), device))
    decoded = (copy.deepcopy(badges), copy.deepcopy(dots))
    res = P.score(badges, dots, image_size=(W, H))
    widths = [c["width"] for c in res["cards"]]
    card_w = float(np.median(widths)) / scale if len(widths) >= MIN_CARDS else None
    return dict(scale=scale, size=(W, H), decoded=decoded, n_cards=len(res["cards"]), card_w=card_w)


def analyze(model, img, device="cpu"):
    """img: BGR photo. Returns dict(result=<postprocess.score result>, decoded=(badges, dots) as the
    scoring got them (detector output plus icon readings), badges and dots as scoring annotated
    them, size (model input), scale (model px per photo px), card_w (photo px))."""
    long_side = max(img.shape[:2])
    first = None
    for size in FIRST_PASS_SIZES:
        scale = min(1.0, size / long_side)
        r = _run_at(model, img, scale, device)
        if first is None or r["n_cards"] > first["n_cards"]:
            first = r
        if r["card_w"] is not None or scale >= 1:
            break
    final = first
    for _ in range(2):
        if final["card_w"] is None:
            break
        scale = min(TARGET_CARD_W / final["card_w"], MAX_UPSCALE, MAX_LONG_SIDE / long_side)
        if abs(scale / final["scale"] - 1) <= CLOSE_ENOUGH:
            break
        nxt = _run_at(model, img, scale, device)
        if nxt["card_w"] is None and nxt["n_cards"] < final["n_cards"]:
            break
        final = nxt
    badges, dots = copy.deepcopy(final["decoded"])
    read_icons(img, badges, 1 / final["scale"])
    decoded = copy.deepcopy((badges, dots))
    res = P.score(badges, dots, image_size=final["size"])
    return dict(result=res, decoded=decoded, badges=badges, dots=dots, size=final["size"], scale=final["scale"],
                card_w=final["card_w"] or TARGET_CARD_W / final["scale"])
