"""Read each card's colour-accessibility icon (the shape below the badge) from the full-resolution photo.

Every deck has its own icon (blue concave square, green trefoil, orange two bars, pink upright
diamond, purple fan, red chevron). The detector's deck reading leans on colour, which phones and
lighting shift (red vs pink especially); the icon's shape doesn't change with white balance, so it
is the primary deck evidence. Classical: no learned weights.

  1. Cut a card-aligned patch around the detector's icon point, at template scale (TEX_W px per
     card width), from the photo (bilinear).
  2. Whiteness = min(R, G, B), stretched between two percentiles of the patch.
  3. Find the icon: an icon-sized white blob (a closing merges bars/lobes) near the predicted spot.
  4. Template-match the 6 icons around it (3 scales; orange also mirrored): normalised correlation
     of whiteness plus that of its x/y gradients (edges separate a blurred two-bar icon from a
     square, a chevron's notch from a fan's round top).

Every step is plain arithmetic so src/pipeline/icons.ts can mirror it; only the patch resampling
differs between Python (cv2.warpAffine) and the browser (canvas drawImage).
Templates: src/data/cards/icons.json (in source control; the training boards' icons come from it too).
"""
import json
import os

import cv2
import numpy as np

from tooling.paths import REPO_ROOT
from tooling.detector.model import DECKS

TEX_W = 360               # patch scale: px per card width (templates were cut at this scale)
MARGIN = 45               # search margin around the template box (px at TEX_W)
CROP_R = 5                # template-match shift range around the found blob (px)
SCALES = (0.9, 1.0, 1.1)  # (trying only the best decks' other scales lost pink diamonds)
COARSE_STEP = 2           # shift search: every 2nd shift, then refined around the best (speed)
GRAD_WEIGHT = 1.0         # weight of the gradient correlation (intensity correlation = 1)
THRESHOLDS = (0.5, 0.65, 0.8, 0.35)
CLOSE_SIZES = (5, 9)
BLOB_W = (6, 40)
BLOB_H = (10, 34)
BLOB_AREA = (70, 750)
MAX_DX = 18               # blob centre across the badge->icon axis (px)
DY_RANGE = (-22, 38)      # ... and along it, from the expected icon centre (px)
ICON_CY_OFFSET = -4       # the icon's centre sits this far above the template box centre (px)
P_LO, P_HI = 0.20, 0.995  # whiteness stretch percentiles
# Below this card width (photo px) the icon is too small to read (~10 px); the deck then comes from
# colour alone. Real phone photos so far: 207-620 px; synthetic test boards: 35-95 px (no icons).
MIN_CARD_PX = 150

LIB = json.load(open(os.path.join(REPO_ROOT, "src", "data", "cards", "card_library.json")))
ASPECT = LIB["aspect_w_over_h"]
ICON = LIB["cards"][0]["icon"]
BADGE_Y = {c["suit"]: c["badge"]["y"] for c in LIB["cards"]}
ICONS_JSON = os.path.join(REPO_ROOT, "src", "data", "cards", "icons.json")


# ---------- templates ----------

def resize_bilinear(t, s):
    """Scale an image by s (bilinear, pixel centres aligned, edges clamped)."""
    h, w = t.shape
    H, W = int(np.floor(h * s + 0.5)), int(np.floor(w * s + 0.5))
    ys = np.clip((np.arange(H) + 0.5) / s - 0.5, 0, h - 1)
    xs = np.clip((np.arange(W) + 0.5) / s - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = t[y0][:, x0] * (1 - fx) + t[y0][:, x1] * fx
    b = t[y1][:, x0] * (1 - fx) + t[y1][:, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


def sobel(a):
    """3x3 Sobel x and y, edges replicated."""
    p = np.pad(a, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx.astype(np.float32), gy.astype(np.float32)


_TEMPLATES = None


def load_templates():
    """deck -> list of (scale, t, gx, gy) variants (scales; orange also mirrored: card backs)."""
    global _TEMPLATES
    if _TEMPLATES is None:
        J = json.load(open(ICONS_JSON))
        _TEMPLATES = {}
        for d in DECKS:
            e = J["icons"][d]
            base = np.asarray(e["data"], np.float32).reshape(e["h"], e["w"]) / 255
            vs = []
            for t0 in ([base, base[:, ::-1]] if d == "orange" else [base]):
                for s in SCALES:
                    t = resize_bilinear(np.ascontiguousarray(t0), s)
                    vs.append((s, t, *sobel(t)))
            _TEMPLATES[d] = vs
    return _TEMPLATES


def template_box():
    e = json.load(open(ICONS_JSON))["icons"]["red"]
    return e["w"], e["h"]


# ---------- geometry ----------

def patch_affine(badge, k):
    """2x3 matrix mapping patch px -> photo px for this badge (k = photo px per badge-coordinate px),
    and the patch size. The patch centre is the detector's icon point."""
    tw, th = template_box()
    b = np.array([badge["x"], badge["y"]], float) * k
    i = np.array(badge["icon"], float) * k
    d = i - b
    L = float(np.hypot(*d))
    if L < 1e-6:
        return None, None
    v = d / L
    u = np.array([v[1], -v[0]])
    width = L / ((ICON["y"] - BADGE_Y[badge["suit"]]) / ASPECT)
    if width < MIN_CARD_PX:
        return None, None
    s = width / TEX_W
    pw, ph = tw + 2 * MARGIN, th + 2 * MARGIN
    A = np.array([[s * u[0], s * v[0], i[0] - pw / 2 * s * u[0] - ph / 2 * s * v[0]],
                  [s * u[1], s * v[1], i[1] - pw / 2 * s * u[1] - ph / 2 * s * v[1]]])
    return A, (pw, ph)


def cut_patch(img_bgr, A, size):
    """RGB float patch (0..1): plain float bilinear sampling (pixel centres at integers), each
    neighbour outside the photo = black. (Not cv2.warpAffine: it quantises positions to 1/32 px;
    this is easy to mirror exactly in TypeScript.)"""
    pw, ph = size
    px, py = np.meshgrid(np.arange(pw, dtype=np.float64), np.arange(ph, dtype=np.float64))
    sx = A[0, 0] * px + A[0, 1] * py + A[0, 2]
    sy = A[1, 0] * px + A[1, 1] * py + A[1, 2]
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    H, W = img_bgr.shape[:2]
    rgb = img_bgr[..., ::-1]

    def at(yy, xx):
        ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        return rgb[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)].astype(np.float64) * ok[..., None]
    out = (at(y0, x0) * ((1 - fx) * (1 - fy)) + at(y0, x0 + 1) * (fx * (1 - fy))
           + at(y0 + 1, x0) * ((1 - fx) * fy) + at(y0 + 1, x0 + 1) * (fx * fy))
    return (out / 255).astype(np.float32)


# ---------- reading ----------

def percentile(a, q):
    s = np.sort(a.ravel())
    return float(s[int(np.floor(q * (s.size - 1)))])


def whiteness(rgb):
    mn = rgb.min(axis=2)
    lo, hi = percentile(mn, P_LO), percentile(mn, P_HI)
    return np.clip((mn - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)


def disk(n):
    """Offsets of cv2.getStructuringElement(MORPH_ELLIPSE, (n, n))."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    c = n // 2
    return [(int(y) - c, int(x) - c) for y, x in zip(*np.nonzero(k))]


DISKS = {n: disk(n) for n in CLOSE_SIZES}


def close(m, n):
    """Binary closing (dilate, then erode) with an n x n ellipse; outside the patch is neutral."""
    H, W = m.shape
    offs = DISKS[n]
    c = n // 2
    p = np.zeros((H + 2 * c, W + 2 * c), bool)
    p[c:c + H, c:c + W] = m
    dil = np.zeros((H, W), bool)
    for dy, dx in offs:
        dil |= p[c + dy:c + dy + H, c + dx:c + dx + W]
    p = np.ones((H + 2 * c, W + 2 * c), bool)
    p[c:c + H, c:c + W] = dil
    ero = np.ones((H, W), bool)
    for dy, dx in offs:
        ero &= p[c + dy:c + dy + H, c + dx:c + dx + W]
    return ero


def locate(w):
    """Centre (x, y) of the icon blob in patch px, or None."""
    ph, pw = w.shape
    ex, ey = pw / 2, ph / 2 + ICON_CY_OFFSET
    for n in CLOSE_SIZES:
        for thr in THRESHOLDS:
            m = close(w > thr, n)
            cnt, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            best, best_d = None, np.inf
            for j in range(1, cnt):
                x, y, bw, bh, a = (int(v) for v in st[j])
                if not (BLOB_W[0] <= bw <= BLOB_W[1] and BLOB_H[0] <= bh <= BLOB_H[1] and BLOB_AREA[0] <= a <= BLOB_AREA[1]):
                    continue
                cx, cy = x + bw / 2, y + bh / 2
                dx, dy = cx - ex, cy - ey
                if abs(dx) > MAX_DX or not DY_RANGE[0] <= dy <= DY_RANGE[1]:
                    continue
                dd = dx * dx + (0.6 * dy) ** 2
                if dd < best_d:
                    best, best_d = (cx, cy - ICON_CY_OFFSET), dd
            if best is not None:
                return best
    return None


def ncc(img, t, centred):
    """Normalised correlation of t at every position inside img (valid region)."""
    from numpy.lib.stride_tricks import sliding_window_view
    img, t = np.asarray(img, np.float64), np.asarray(t, np.float64)  # float64: the TS port sums in doubles
    win = sliding_window_view(img, t.shape)  # (Y, X, h, w)
    if centred:
        tz = t - t.mean()
        num = np.einsum("yxhw,hw->yx", win, tz)
        wsum = win.sum(axis=(2, 3))
        wsq = np.einsum("yxhw,yxhw->yx", win, win)
        den = np.sqrt(np.maximum(wsq - wsum * wsum / t.size, 0) * float((tz * tz).sum()))
    else:
        num = np.einsum("yxhw,hw->yx", win, t)
        den = np.sqrt(np.einsum("yxhw,yxhw->yx", win, win) * float((t * t).sum()))
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def read_patch(rgb):
    """Icon scores (one per deck, in DECKS order) from a patch cut by patch_affine/cut_patch.
    None if no icon-like blob was found."""
    w = whiteness(rgb)
    ctr = locate(w)
    if ctr is None:
        return None
    tw, th = template_box()
    pad = th + CROP_R
    wp = np.pad(w, pad)
    cx, cy = int(np.floor(ctr[0] + 0.5)) + pad, int(np.floor(ctr[1] + 0.5)) + pad
    crop = wp[cy - th // 2 - CROP_R:cy + th // 2 + CROP_R + 1, cx - tw // 2 - CROP_R:cx + tw // 2 + CROP_R + 1]
    gx, gy = sobel(crop)

    def match(t, tx, ty):
        """Best correlation over shifts: every COARSE_STEP-th shift, then all shifts next to the best."""
        if t.shape[0] > crop.shape[0] or t.shape[1] > crop.shape[1]:
            return -np.inf
        R = ncc(crop, t, True) + GRAD_WEIGHT * 0.5 * (ncc(gx, tx, False) + ncc(gy, ty, False))
        S = COARSE_STEP
        C = R[::S, ::S]
        j = int(np.argmax(C))
        y, x = (j // C.shape[1]) * S, (j % C.shape[1]) * S
        return float(R[max(y - S + 1, 0):y + S, max(x - S + 1, 0):x + S].max())

    T = load_templates()
    return [max(match(t, tx, ty) for _, t, tx, ty in T[d]) for d in DECKS]


def read_icons(img_bgr, badges, k):
    """Adds badge["icon_scores"] (or None) to every non-start badge. k: photo px per badge px."""
    for b in badges:
        if b["suit"] == "start":
            continue
        A, size = patch_affine(b, k)
        b["icon_scores"] = None if A is None else read_patch(cut_patch(img_bgr, A, size))
    return badges

