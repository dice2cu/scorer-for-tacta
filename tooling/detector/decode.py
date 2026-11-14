"""Decode TactaNet output maps into badges (cards) and dots. Mirrors what the app will do."""
import numpy as np
import torch
import torch.nn.functional as F

from tooling.detector.model import DECKS, ICON_SCALE, STRIDE, SUITS, head_slices

SL = head_slices()


def peaks(hm, thr):
    """hm: CxHxW probabilities -> list of (c, y, x, score) local maxima above thr."""
    t = torch.from_numpy(hm)[None]
    mx = F.max_pool2d(t, 3, stride=1, padding=1)[0].numpy()
    keep = (hm == mx) & (hm > thr)
    c, y, x = np.nonzero(keep)
    return list(zip(c, y, x, hm[c, y, x]))


def softmax(v, axis=0):
    e = np.exp(v - v.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


PARTIAL_THR = 0.3
PARTIAL_MERGE = 1.5  # cells: a full-dot and a partial-dot peak this close are one dot


def decode(out, badge_thr=0.3, dot_thr=0.2, partial_thr=PARTIAL_THR):  # weak dots are judged later, in context
    """out: N_OUT x H/4 x W/4 raw logits (numpy). Returns (badges, dots) in input-pixel coords.
    Partly covered dots are included in dots with partial=True (never counted)."""
    sig = lambda a: 1 / (1 + np.exp(-a))
    bh = sig(out[SL["badge_hm"]])
    # one badge per location regardless of suit channel
    merged = bh.max(axis=0, keepdims=True)
    badges = []
    for _, y, x, s in peaks(merged, badge_thr):
        suit = int(np.argmax(bh[:, y, x]))
        ox, oy = sig(out[SL["badge_off"], y, x])
        bx, by = (x + ox) * STRIDE, (y + oy) * STRIDE
        dx, dy = out[SL["icon"], y, x] * ICON_SCALE
        pv = softmax(out[SL["value"], y, x])
        pd = softmax(out[SL["deck"], y, x])
        badges.append(dict(
            x=float(bx), y=float(by), score=float(s), suit=SUITS[suit],
            suit_p=float(bh[suit, y, x] / max(bh[:, y, x].sum(), 1e-6)),
            value=int(np.argmax(pv)) + 1, value_p=float(pv.max()), value_probs=pv.tolist(),
            side="back" if sig(out[SL["side"], y, x][0]) > 0.5 else "front",
            deck=DECKS[int(np.argmax(pd))], deck_p=float(pd.max()), deck_probs=pd.tolist(),
            icon=(float(bx + dx), float(by + dy))))
    def dot(y, x, s, partial):
        ox, oy = sig(out[SL["dot_off"], y, x])
        pd = softmax(out[SL["dot_deck"], y, x])
        d = dict(x=float((x + ox) * STRIDE), y=float((y + oy) * STRIDE), score=float(s),
                 deck=DECKS[int(np.argmax(pd))], deck_p=float(pd.max()), deck_probs=pd.tolist())
        if partial:
            d["partial"] = True
        return d

    full = [(y, x, s) for _, y, x, s in peaks(sig(out[SL["dot_hm"]]), dot_thr)]
    part = [(y, x, s) for _, y, x, s in peaks(sig(out[SL["dot_part"]]), partial_thr)] if "dot_part" in SL and out.shape[0] > SL["dot_part"].start else []
    # one dot per place: where a full and a partial peak coincide, the stronger reading wins
    drop_full, drop_part = set(), set()
    for i, (py, px, ps) in enumerate(part):
        for j, (fy, fx, fs) in enumerate(full):
            if (fy - py) ** 2 + (fx - px) ** 2 <= PARTIAL_MERGE ** 2:
                (drop_full if ps > fs else drop_part).add(j if ps > fs else i)
    dots = [dot(y, x, s, False) for j, (y, x, s) in enumerate(full) if j not in drop_full]
    dots += [dot(y, x, s, True) for i, (y, x, s) in enumerate(part) if i not in drop_part]
    return badges, dots


def run_model(model, rgb, device="cpu"):
    """rgb: HxWx3 uint8 (H, W divisible by 16). Returns raw output maps as numpy."""
    from tooling.detector.data import normalize
    x = torch.from_numpy(normalize(rgb))[None].to(device)
    with torch.no_grad():
        return model(x)[0].float().cpu().numpy()
