"""Build the Tacta card-design library from photos of a full deck laid out in a grid.

Input photos (e.g. in training-data/decks/, not under source control): one or more photos, each with 18 cards
(3 rows: circle, square, triangle x 6 columns: values 1..6) or 36 cards (two decks, 6 rows).
Cards may show front or back; the FIRST photo defines the front orientation.

Output:
  - src/data/cards/card_library.json : card geometry in normalized card coordinates (front
        orientation, badge upright, icon below badge). x,y in [0,1], origin top-left.
  - src/data/cards/card_masks/<id>.png : low-res layout mask (0=black face, 127=color, 254=white)
  - out/card-library/ : rectified cards + overlays for visual review

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.card_library.build_card_library
      training-data/decks/{pink_1,green_1,green_pink,pink_2,green_2}.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np

from tooling import paths
from tooling.paths import REPO_ROOT

SUITS = ["circle", "square", "triangle"]
CANON_W = 600  # rectified card width in px (height derived from measured aspect ratio)
MASK_W = 120   # stored layout-mask width


# ---------------------------------------------------------------- card finding

def sat_threshold(hsv, v_min=70, lo=40, hi=110):
    """Saturation level separating deck color from black/white ink, picked per image (Otsu).

    Deck colors photograph anywhere from vivid to muted depending on light and white balance,
    so a fixed threshold fails. Among reasonably bright pixels saturation is bimodal:
    white ink (low) vs deck color (high)."""
    s = hsv[..., 1][hsv[..., 2] > v_min]
    if s.size < 100:
        return 90
    t, _ = cv2.threshold(s.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(np.clip(t, lo, hi))


def saturated_mask(img):
    """Pixels of any deck color. A photo can hold several decks (vivid pink + muted green), whose
    saturations Otsu may split between, so cap the cutoff below even muted deck colors."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    t = sat_threshold(hsv, hi=80)
    return ((hsv[..., 1] > t) & (hsv[..., 2] > 70)).astype(np.uint8) * 255


def line_intersection(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    t = np.linalg.solve(a, b)
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])


def order_corners(pts):
    """Return corners ordered TL, TR, BR, BL (for a roughly upright card)."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coords starting near -pi (left)
    # rotate so first is top-left (min x+y)
    k = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -k, axis=0)


def refine_corners(contour, approx):
    """Fit a line to each side of the (rounded-corner) card outline; corners = intersections."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    quad = order_corners(approx.reshape(-1, 2))
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = q - p
        L = np.linalg.norm(d)
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = pts - p
        along = rel @ u
        perp = np.abs(rel @ n)
        sel = pts[(along > 0.15 * L) & (along < 0.85 * L) & (perp < 0.04 * L)]
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((vx, vy, x0, y0))
    corners = [line_intersection(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    return order_corners(corners)


def expand_quad(quad, d):
    """Move each side of a convex quad outward by d pixels."""
    c = quad.mean(axis=0)
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        u = (q - p) / np.linalg.norm(q - p)
        n = np.array([-u[1], u[0]])
        if n @ (p - c) < 0:
            n = -n
        p2 = p + n * d
        lines.append((u[0], u[1], p2[0], p2[1]))
    return order_corners([line_intersection(lines[(i - 1) % 4], lines[i]) for i in range(4)])


def card_contours(img, expected):
    """Card outer contours; erodes progressively until touching cards separate."""
    sat = saturated_mask(img)
    # remove speckle (e.g. reddish wood grain) that could bridge into card outlines; card
    # borders are ~3% of card width thick, so they survive this easily
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Fill each outline into a solid card shape. If an outline has a small break (glare),
    # its contour fill fails, so card-sized blobs are filled by their convex hull instead.
    cnts, _ = cv2.findContours(sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull_areas = np.array([cv2.contourArea(cv2.convexHull(c)) for c in cnts])
    card_area = np.median(np.sort(hull_areas)[::-1][:expected])
    solid = np.zeros_like(sat)
    for c, ha in zip(cnts, hull_areas):
        if 0.7 * card_area < ha < 1.3 * card_area:
            cv2.drawContours(solid, [cv2.convexHull(c)], -1, 255, cv2.FILLED)
        else:
            cv2.drawContours(solid, [c], -1, 255, cv2.FILLED)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Erode until every card is its own core (touching cards separate), then give every
    # solid pixel to its nearest core so each card regains its full outline.
    for r in range(0, 200, 4):
        m = cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) if r else solid
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        areas = stats[1:, cv2.CC_STAT_AREA]
        top = np.sort(areas)[::-1][:expected]
        keep = [i + 1 for i, a in enumerate(areas) if a > 0.5 * np.median(top)]
        ka = areas[[k - 1 for k in keep]]
        if len(keep) == expected and ka.max() < 1.4 * np.median(ka):
            break
    else:
        raise SystemExit(f"could not separate {expected} cards")

    cores = np.zeros(lab.shape, np.int32)
    for j, k in enumerate(keep, start=1):
        cores[lab == k] = j
    # distance to nearest core pixel, with that core's label
    _, nearest = cv2.distanceTransformWithLabels((cores == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    zero_y, zero_x = np.nonzero(cores)  # label index -> pixel order matches row-major nonzero scan
    pix_label = np.zeros(nearest.max() + 1, np.int32)
    pix_label[1:len(zero_y) + 1] = cores[zero_y, zero_x]
    seg = np.where(solid > 0, pix_label[nearest], 0)

    out = []
    for j in range(1, expected + 1):
        cm = (seg == j).astype(np.uint8) * 255
        cs, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out.append(max(cs, key=cv2.contourArea))
    return out, r


def find_cards(img, expected=18):
    cnts, r = card_contours(img, expected)
    cards = []
    for c in cnts:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = None
        for eps in np.linspace(0.01, 0.08, 15):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                approx = ap
                break
        if approx is None:
            continue
        cards.append(refine_corners(c, approx))
    print(f"  cards separated at erosion radius {r}px")
    return cards


def sort_grid(cards, rows=3, cols=6):
    """Sort card corner sets into reading order (row-major)."""
    centers = np.array([c.mean(axis=0) for c in cards])
    order = np.argsort(centers[:, 1])
    grid = []
    for r in range(rows):
        idx = order[r * cols:(r + 1) * cols]
        idx = idx[np.argsort(centers[idx, 0])]
        grid.extend(idx.tolist())
    return [cards[i] for i in grid]


def card_aspect(cards):
    ratios = []
    for q in cards:
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        ratios.append(w / h)
    return float(np.median(ratios)), ratios


def rectify(img, quad, w, h):
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- feature extraction (rectified card)

def classify_pixels(card):
    """3-level map: 0 = black face, 1 = colored (deck color), 2 = white ink."""
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    t = sat_threshold(hsv, v_min=90)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    # White ink brightness varies with exposure (a dim card's dots can be V~145), but it is always
    # far brighter than the black face. Use half the brightness of the card's brightest gray ink.
    gray = v[s < t]
    v_white = max(80, 0.5 * np.percentile(gray, 99.5)) if gray.size else 150
    cls = np.zeros(s.shape, np.uint8)
    cls[(v > 0.6 * v_white) & (s >= t)] = 1
    cls[(v > v_white) & (s < t)] = 2
    return cls


def find_dots(card):
    """White, round, dot-sized blobs. Returns list of (x, y, radius) in pixels."""
    # Printed dots are ~6% of card width across; lines/digit strokes ~2.3%, icon ~3%.
    # An opening with a disc of ~4% of width erases everything except dots.
    cls = classify_pixels(card)
    white = (cls == 2).astype(np.uint8) * 255
    H, W = white.shape
    k = int(round(W * 0.04)) | 1
    core = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(core)
    dots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        r = np.sqrt(area / np.pi)
        if not (0.02 * W < r < 0.045 * W) or not (0.7 < w / h < 1.4):
            continue
        comp = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * cv2.contourArea(c) / (per * per)
        if circ < 0.8:
            continue
        # A printed dot is an isolated disc: a thin ring just outside it holds no white ink.
        # Line junctions that survive the opening have white lines running out of them.
        cx, cy = cents[i]
        if ring_white_fraction(white, cx, cy, 1.25 * r, 1.45 * r) > 0.15:  # dots <= 0.09, junctions >= 0.22
            continue
        # The value badge + icon sit in the card center where no design has a dot; a glowing badge
        # outline can otherwise pass as a dot. (Verified: no library dot inside this zone.)
        if abs(cx - 0.5 * W) < 0.15 * W and 0.33 * H < cy < 0.62 * H:
            continue
        dots.append((float(cx), float(cy), float(r)))
    return dots


def ring_white_fraction(white, cx, cy, r_in, r_out):
    H, W = white.shape
    x0, x1 = int(max(0, cx - r_out - 1)), int(min(W, cx + r_out + 2))
    y0, y1 = int(max(0, cy - r_out - 1)), int(min(H, cy + r_out + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    ring = (d >= r_in) & (d <= r_out)
    if ring.sum() == 0:
        return 1.0
    return float((white[y0:y1, x0:x1][ring] > 0).mean())


def find_badge_and_icon(card):
    """Badge = colored blob nearest the card center; icon = white blob just below it."""
    cls = classify_pixels(card)
    H, W = cls.shape
    colored = (cls == 1).astype(np.uint8)

    def pick(mask):
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
        best, bd = None, 1e9
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # badge-sized and roughly square (bold digits can split the fill into flat slivers)
            if area < (0.03 * W) ** 2 or not (0.08 * W < w < 0.3 * W) or not (0.55 < h / w < 1.8):
                continue
            d = np.hypot(cents[i][0] - 0.5 * W, cents[i][1] - 0.5 * H)
            if d < 0.15 * W and d < bd:
                best, bd = i, d
        if best is None:
            return None
        x, y, w, h, _ = stats[best]
        return dict(x=float(cents[best][0]), y=float(cents[best][1]), w=float(w), h=float(h))

    badge = pick(colored)
    if badge is None:  # digit split the fill: close across the digit strokes and retry
        k = int(round(0.035 * W)) | 1
        badge = pick(cv2.morphologyEx(colored, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
    if badge is None:
        return None, None
    bx, by, h = badge["x"], badge["y"], badge["h"]

    white = (cls == 2).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(white)
    icon, bd = None, 1e9
    for i in range(1, n):
        cx, cy = cents[i]
        if stats[i][4] < 15 or abs(cx - bx) > 0.06 * W:
            continue
        if not (by + 0.5 * h < cy < by + 0.5 * h + 0.12 * H):
            continue
        d = cy - by
        if d < bd:
            icon, bd = (float(cx), float(cy)), d
    return badge, icon


def layout_map(card, mask_w=MASK_W):
    H, W = card.shape[:2]
    small = cv2.resize(card, (mask_w, int(round(mask_w * H / W))), interpolation=cv2.INTER_AREA)
    return classify_pixels(small)


def match_dots(a, b):
    """Greedy nearest matching; returns list of distances (or None if counts differ)."""
    if len(a) != len(b):
        return None
    a, b = np.array(a)[:, :2], np.array(b)[:, :2]
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    used, dist = set(), []
    for i in np.argsort(d.min(axis=1)):
        j = min((j for j in range(len(b)) if j not in used), key=lambda j: d[i, j])
        used.add(j)
        dist.append(d[i, j])
    return dist


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+",
                    help="layout photos: 18 cards (3 rows) or 36 cards (6 rows), rows ordered "
                         "circle/square/triangle, columns value 1..6. The FIRST photo defines the "
                         "front orientation; other photos may show fronts or backs.")
    ap.add_argument("--debug", default=os.path.join(paths.OUT, "card-library"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "src", "data", "cards"))
    args = ap.parse_args()
    os.makedirs(args.debug, exist_ok=True)

    # ---- find + sort cards in every photo
    photos = []
    aspects = []
    for p_idx, path in enumerate(args.photos):
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        name = os.path.splitext(os.path.basename(path))[0]
        n_cards = None
        for n in (36, 18):
            try:
                cards = find_cards(img, expected=n)
                n_cards = n
                break
            except SystemExit:
                continue
        if n_cards is None:
            raise SystemExit(f"{path}: could not find 18 or 36 cards")
        cards = sort_grid(cards, rows=n_cards // 6)
        a, ratios = card_aspect(cards)
        print(f"[{p_idx}] {path}: {n_cards} cards, aspect median={a:.4f} "
              f"(min {min(ratios):.4f}, max {max(ratios):.4f})")
        aspects.extend(ratios)
        photos.append((name, img, cards))
        vis = img.copy()
        for i, q in enumerate(cards):
            cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 255), 4)
            cv2.putText(vis, str(i), tuple(q[0].astype(int) + [15, 50]), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        cv2.imwrite(os.path.join(args.debug, f"found_{p_idx}_{name}.jpg"), cv2.resize(vis, None, fx=0.4, fy=0.4))

    aspect = float(np.median(aspects))
    W, H = CANON_W, int(round(CANON_W / aspect))
    print(f"canonical card size {W}x{H} (aspect {aspect:.4f})\n")

    # ---- rectify all instances; group by card id
    instances = {f"{s}-{v}": [] for s in SUITS for v in range(1, 7)}  # id -> [(photo_idx, img)]
    skipped = []
    for p_idx, (name, img, cards) in enumerate(photos):
        ih, iw = img.shape[:2]
        margin = 0.004 * max(ih, iw)
        for i, q in enumerate(cards):
            cid = f"{SUITS[(i // 6) % 3]}-{i % 6 + 1}"
            # a card touching the photo border may be cut off; its outline can't be trusted
            if (q[:, 0].min() < margin or q[:, 1].min() < margin or
                    q[:, 0].max() > iw - 1 - margin or q[:, 1].max() > ih - 1 - margin):
                skipped.append(f"{cid} in photo {p_idx} (touches photo border)")
                continue
            instances[cid].append((p_idx, rectify(img, q, W, H)))
    for s in skipped:
        print(f"skipping {s}")

    library, problems = [], []
    sheets = {p: [None] * len(photos[p][2]) for p in range(len(photos))}
    sheet_pos = {p: 0 for p in range(len(photos))}
    all_r = []
    for cid, inst in instances.items():
        suit, value = cid.split("-")[0], int(cid.split("-")[1])
        ref_p, ref = inst[0]
        ref_map = layout_map(ref)
        ref_dots = find_dots(ref)

        def dot_cost(dots):
            """Mean distance (fraction of width) from each reference dot to the nearest dot."""
            if len(dots) != len(ref_dots) or not dots:
                return None
            return float(np.mean([min(np.hypot(a[0] - b[0], a[1] - b[1]) for b in dots)
                                  for a in ref_dots])) / W

        # Orientation of each other instance relative to the reference: front (as-is) or back
        # (mirrored). Dot positions decide when usable; otherwise the coarse layout mask does.
        oriented, notes = [(ref_p, ref, "front")], []
        for p_idx, im in inst[1:]:
            d = find_dots(im)
            c_plain = dot_cost(d)
            c_mirror = dot_cost([(W - 1 - x, y, r) for x, y, r in d])
            if c_plain is not None and abs(c_plain - c_mirror) > 0.02:
                side = "back" if c_mirror < c_plain else "front"
                why = f"dots {min(c_plain, c_mirror) * 100:.1f}%"
            else:
                plain = float((layout_map(im) == ref_map).mean())
                mirror = float((layout_map(cv2.flip(im, 1)) == ref_map).mean())
                side = "back" if mirror > plain + 0.03 else "front" if plain > mirror + 0.03 else "symmetric"
                why = f"layout {max(plain, mirror):.2f}"
            oriented.append((p_idx, cv2.flip(im, 1) if side == "back" else im, side))
            notes.append(f"{p_idx}:{side}({why})")

        # Dots per instance. Instances with the wrong count are reported and skipped; the rest are
        # matched to the first good instance and combined with a per-dot median (robust to one
        # instance whose corners were fitted slightly off).
        inst_dots = [(p_idx, side, find_dots(im)) for p_idx, im, side in oriented]
        counts = [len(d) for _, _, d in inst_dots]
        good = [(p, s, d) for p, s, d in inst_dots if len(d) == value]
        for p, s, d in inst_dots:
            if len(d) != value:
                problems.append(f"{cid}: photo {p} ({s}) has {len(d)} dots, expected {value}")
        if not good:
            raise SystemExit(f"{cid}: no photo has the expected {value} dots")
        anchor = good[0][2]
        per_dot = [[] for _ in anchor]  # per_dot[k] = [(photo, x, y, r), ...]
        for p, s, d in good:
            for k, a in enumerate(anchor):
                j = int(np.argmin([np.hypot(a[0] - e[0], a[1] - e[1]) for e in d]))
                per_dot[k].append((p, *d[j]))
        mean_dots, spread, worst = [], 0.0, None
        for group in per_dot:
            g = np.array([g[1:] for g in group])
            m = np.median(g, axis=0)
            mean_dots.append(m)
            all_r.extend(g[:, 2].tolist())
            dev = np.hypot(g[:, 0] - m[0], g[:, 1] - m[1])
            if dev.max() > spread:
                spread, worst = float(dev.max()), group[int(dev.argmax())][0]

        # badge + icon: median over instances where both were found (front orientation; the
        # badge sits on the centerline, so mirrored backs measure the same position)
        bi = [find_badge_and_icon(im) for _, im, _ in oriented]
        with_icon = [(b, i) for b, i in bi if b is not None and i is not None]
        if not with_icon:
            raise SystemExit(f"{cid}: badge+icon not found in any photo")
        badge = {k: float(np.median([b[k] for b, _ in with_icon])) for k in ("x", "y", "w", "h")}
        icon = (float(np.median([i[0] for _, i in with_icon])), float(np.median([i[1] for _, i in with_icon])))

        # layout mask: per-pixel majority vote over all oriented instances
        maps = np.stack([layout_map(im) for _, im, _ in oriented])
        votes = np.stack([(maps == c).sum(axis=0) for c in range(3)])
        mask = votes.argmax(axis=0).astype(np.uint8)
        agree = float((maps == mask[None]).mean())

        print(f"{cid:11s} dots per photo={counts}  "
              f"max dot deviation from median={spread / W * 100:.2f}% of width (photo {worst})  "
              f"layout agreement={agree:.3f}  orientation {' '.join(notes)}")

        library.append(dict(
            id=cid, suit=suit, value=value,
            dots=[dict(x=round(float(x) / W, 4), y=round(float(y) / H, 4))
                  for x, y, _ in sorted(mean_dots, key=lambda d: (d[1], d[0]))],
            badge=dict(x=round(badge["x"] / W, 4), y=round(badge["y"] / H, 4),
                       w=round(badge["w"] / W, 4), h=round(badge["h"] / H, 4)),
            icon=None if icon is None else dict(x=round(icon[0] / W, 4), y=round(icon[1] / H, 4)),
            _mask=mask,
        ))

        # review overlays: each instance (front-oriented) with its detected dots + the mean dots
        for p_idx, im, side in oriented:
            vis = im.copy()
            for x, y, r in find_dots(im):
                cv2.circle(vis, (int(x), int(y)), int(r * 1.7), (0, 255, 255), 3)
            for x, y, _ in mean_dots:
                cv2.drawMarker(vis, (int(x), int(y)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(vis, f"{cid} {side}", (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            sheets[p_idx][sheet_pos[p_idx]] = vis
            sheet_pos[p_idx] += 1

    for p_idx, tiles in sheets.items():
        tiles = [t if t is not None else np.zeros((H, W, 3), np.uint8) for t in tiles]  # skipped
        n = len(tiles)
        rows = [np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(n // 6)]
        cv2.imwrite(os.path.join(args.debug, f"sheet_{p_idx}_{photos[p_idx][0]}.jpg"),
                    cv2.resize(np.vstack(rows), None, fx=0.4, fy=0.4))

    # Badge geometry is part of the print design: one shape+position per suit, icon identical on
    # all cards. Consolidate to suit/global medians and flag any card that disagrees.
    for suit in SUITS:
        group = [c for c in library if c["suit"] == suit]
        med = {k: round(float(np.median([c["badge"][k] for c in group])), 4) for k in ("x", "y", "w", "h")}
        for c in group:
            b = c["badge"]
            if abs(b["x"] - med["x"]) > 0.02 or abs(b["y"] - med["y"]) > 0.02 or \
                    abs(b["w"] / med["w"] - 1) > 0.15 or abs(b["h"] / med["h"] - 1) > 0.15:
                problems.append(f"{c['id']}: measured badge {b} disagrees with {suit} median {med}; using median")
            c["badge"] = dict(med)
    icon_med = {k: round(float(np.median([c["icon"][k] for c in library])), 4) for k in ("x", "y")}
    for c in library:
        if abs(c["icon"]["x"] - icon_med["x"]) > 0.02 or abs(c["icon"]["y"] - icon_med["y"]) > 0.02:
            problems.append(f"{c['id']}: measured icon {c['icon']} disagrees with median {icon_med}; using median")
        c["icon"] = dict(icon_med)

    os.makedirs(os.path.join(args.out, "card_masks"), exist_ok=True)
    for card in library:
        cv2.imwrite(os.path.join(args.out, "card_masks", f"{card['id']}.png"),
                    (card.pop("_mask") * 127).astype(np.uint8))
    out = dict(
        version=1,
        description="Tacta card designs in FRONT orientation (badge upright, icon below badge). "
                    "Coordinates are normalized to card width/height, origin top-left. "
                    "Card backs are the left-right mirror (x -> 1-x) of the front layout. "
                    "Masks: 0=black face, 127=deck color, 254=white ink.",
        source_photos=[os.path.basename(p) for p in args.photos],
        aspect_w_over_h=round(aspect, 4),
        dot_radius_over_width=round(float(np.median(all_r)) / W, 4),
        mask_width=MASK_W,
        cards=library,
    )
    with open(os.path.join(args.out, "card_library.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'card_library.json')}")
    print("PROBLEMS:" if problems else "no problems found")
    for p in problems:
        print("  - " + p)


if __name__ == "__main__":
    main()
