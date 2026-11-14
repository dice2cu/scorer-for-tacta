"""Synthetic Tacta boards with exact ground truth.

Geometry is simulated in "board units" (1 unit = 1 card width) and follows the real rules closely
enough for training:
  - cards connect to an existing card at a relative rotation that is a multiple of 45 degrees
    (edge shapes and corner triangles), with partial overlap
  - no card may cover another card's badge/icon area (the card centre is never covered)
  - legal play never partially covers a dot, but sloppy placement does: on some boards a few
    placements cut a dot (15-85% left visible, never ambiguous slivers); those dots are labelled
    "partial" (not counted, flagged). All other placements cover dots wholly or not at all.
Then a tilted pinhole camera (scale, rotation, up to ~58 degrees from overhead) maps the board
into the image, cards are rendered from real recolored textures, and photometric effects are
applied (lighting gradients, white balance and mixed-colour light, glare, shadows, blur, noise, JPEG).

Ground truth per image: every card (design, side, deck, badge/icon image position, local card
width in px) and every dot (position, owner card, visible fraction, visible/partial flags).
"""
import json
import os

import cv2
import numpy as np

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.synth.render import Assets, DECKS, DECK_RGB, render_card, card_alpha

LIB = json.load(open(os.path.join(REPO_ROOT, "src", "data", "cards", "card_library.json")))
ASPECT = LIB["aspect_w_over_h"]
CARD_H = 1 / ASPECT                       # card height in board units
DESIGNS = {c["id"]: c for c in LIB["cards"]}
DOT_R = LIB["dot_radius_over_width"]
BADGE_ZONE = (0.36, 0.40, 0.64, 0.67)     # normalized (x0, y0, x1, y1): badge + icon, never covered


# ---------------------------------------------------------------- board geometry

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class Card:
    def __init__(self, idx, design, side, deck, center, theta):
        self.idx, self.design, self.side, self.deck = idx, design, side, deck
        self.center, self.theta = np.asarray(center, float), float(theta)

    def to_board(self, xn, yn):
        """normalized card coords (front-orientation library coords, x/y in [0,1]) -> board."""
        if self.side == "back":
            xn = 1 - np.asarray(xn)
        local = np.stack([np.asarray(xn) - 0.5, (np.asarray(yn) - 0.5) * CARD_H], axis=-1)
        return local @ rot(self.theta).T + self.center

    def polygon(self):
        return self.to_board(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))

    def badge_zone(self):
        x0, y0, x1, y1 = BADGE_ZONE
        return self.to_board(np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]))

    def dots(self):
        if self.design == "start":
            return np.zeros((0, 2))
        d = DESIGNS[self.design]["dots"]
        return self.to_board(np.array([p["x"] for p in d]), np.array([p["y"] for p in d]))

    def badge(self):
        if self.design == "start":
            return self.to_board(0.5, 0.5)
        b = DESIGNS[self.design]["badge"]
        return self.to_board(b["x"], b["y"])

    def icon(self):
        if self.design == "start":
            return self.to_board(0.5, 0.6)
        i = DESIGNS[self.design]["icon"]
        return self.to_board(i["x"], i["y"])


def poly_overlap(a, b):
    area, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(area)


def points_in_poly(pts, poly):
    """Boolean per point: inside convex polygon (counter- or clockwise)."""
    pts = np.atleast_2d(pts)
    signs = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        signs.append((q[0] - p[0]) * (pts[:, 1] - p[1]) - (q[1] - p[1]) * (pts[:, 0] - p[0]))
    signs = np.stack(signs, axis=1)
    return np.all(signs >= 0, axis=1) | np.all(signs <= 0, axis=1)


DOT_RING = np.stack([np.cos(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
                     np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))], axis=1)


def _disc_samples(n=400):
    """Uniform points in the unit disc (fixed), to measure how much of a dot is covered."""
    g = np.random.default_rng(12345)
    r, a = np.sqrt(g.uniform(0, 1, n)), g.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


DISC = _disc_samples()
FULL, HIDDEN = 0.97, 0.03          # visible fraction: >= FULL counts as a whole dot, <= HIDDEN as covered
PARTIAL_RANGE = (0.15, 0.85)       # partially covered dots in the training data stay clearly partial


def dot_visible_fraction(p, covers):
    """Fraction of the dot disc at board point p not inside any of the cover polygons."""
    pts = p + DOT_R * DISC
    hidden = np.zeros(len(pts), bool)
    for poly in covers:
        hidden |= points_in_poly(pts, poly)
    return 1 - hidden.mean()


def layout_board(rng, partial_p=None):
    """Place cards following the rules. Returns list of Card in z-order (bottom first).
    partial_p: chance that a placement which would cut dots is allowed anyway (sloppy play)."""
    if partial_p is None:
        partial_p = float(rng.choice([0.0, 0.0, 0.15, 0.4, 1.0]))
    max_partial = int(rng.integers(1, 5))
    n_partial = 0
    n_decks = int(rng.choice([2, 3, 3, 4, 4, 5, 6]))
    decks = list(rng.choice(DECKS, size=n_decks, replace=False))
    per_deck = 18 if rng.random() < 0.6 else int(rng.integers(5, 18))
    pool = []
    for d in decks:
        designs = list(DESIGNS)
        rng.shuffle(designs)
        for des in designs[:per_deck]:
            pool.append((d, des))
    rng.shuffle(pool)

    cards = [Card(0, "start", "front", "white", (0, 0), rng.uniform(0, 2 * np.pi))]
    for deck, design in pool:
        side = "front" if rng.random() < 0.5 else "back"
        placed = False
        for attempt in range(250):
            if rng.random() < 0.04:  # no legal move: placed on its own, touching nothing
                ref = cards[int(rng.integers(len(cards)))]
                center = ref.center + rng.uniform(-3, 3, 2)
                theta = rng.uniform(0, 2 * np.pi)
                need_touch = False
            else:
                ref = cards[int(rng.integers(len(cards)))]
                theta = ref.theta + np.deg2rad(45 * int(rng.integers(8)) + rng.normal(0, 2.0))
                off = np.array([rng.uniform(-1.15, 1.15), rng.uniform(-1.45, 1.45)])
                center = ref.center + rot(ref.theta) @ off
                need_touch = True
            new = Card(len(cards), design, side, deck, center, theta)
            poly = new.polygon()
            card_area = CARD_H
            if need_touch:
                ov = poly_overlap(poly, ref.polygon())
                if not (0.03 * card_area < ov < 0.42 * card_area):
                    continue
            # never cover a badge/icon zone; never partially cover a still-visible dot (except
            # occasional sloppy placements, see partial_p)
            ok, cut = True, []
            for c in cards:
                if poly_overlap(poly, c.polygon()) <= 0:
                    continue
                if poly_overlap(poly, c.badge_zone()) > 0:
                    ok = False
                    break
                for p in c.dots():
                    ring = p + DOT_R * 1.05 * DOT_RING
                    inside = points_in_poly(np.vstack([p, ring]), poly)
                    if inside.any() and not inside.all():
                        cut.append((c.idx, p))
            if not ok:
                continue
            if cut:
                if n_partial >= max_partial or rng.random() >= partial_p:
                    continue
                covers = lambda i: [q.polygon() for q in cards[i + 1:]]
                after = [dot_visible_fraction(p, covers(i) + [poly]) for i, p in cut]
                before = [dot_visible_fraction(p, covers(i)) for i, p in cut]
                # every cut dot must end up hidden or clearly partial; at least one newly partial
                if not all(f <= HIDDEN or PARTIAL_RANGE[0] <= f <= PARTIAL_RANGE[1] for f in after):
                    continue
                new_partial = sum(1 for f, b in zip(after, before) if f > HIDDEN and b >= FULL)
                if new_partial == 0:
                    continue
                n_partial += new_partial
            if not need_touch and any(poly_overlap(poly, c.polygon()) > 0 for c in cards):
                continue
            cards.append(new)
            placed = True
            break
        # a card that cannot be placed legally is simply not played (rare)
    return cards


def visibility(cards):
    """Per card, per dot: visible fraction of the dot (not covered by any later card)."""
    vis = []
    polys = [c.polygon() for c in cards]
    probe = np.vstack([[0, 0], DOT_RING]) * DOT_R * 1.05
    for i, c in enumerate(cards):
        fr = []
        for p in c.dots():
            cand = [q for q in polys[i + 1:] if points_in_poly(p + probe, q).any()]
            fr.append(dot_visible_fraction(p, cand) if cand else 1.0)
        vis.append(np.array(fr, float))
    return vis


# ---------------------------------------------------------------- textures

_START = None


def start_texture(tex_w, tex_h):
    """Procedural white-line starting card (approximation of the real one)."""
    global _START
    if _START is not None and _START.shape[:2] == (tex_h, tex_w):
        return _START
    img = np.zeros((tex_h, tex_w, 3), np.float32) + 0.05
    w = max(2, tex_w // 90)
    col = (0.93, 0.93, 0.93)
    m = tex_w // 22
    cv2.rectangle(img, (m, m), (tex_w - m, tex_h - m), col, w, cv2.LINE_AA)
    for fx in (0.25, 0.5, 0.75):
        cv2.line(img, (int(fx * tex_w), m), (int(fx * tex_w), tex_h - m), col, w, cv2.LINE_AA)
    for fy in (0.25, 0.5, 0.75):
        cv2.line(img, (m, int(fy * tex_h)), (tex_w - m, int(fy * tex_h)), col, w, cv2.LINE_AA)
    for (x0, y0, x1, y1) in ((0, 0, .5, .25), (.5, 0, 1, .25), (0, .75, .5, 1), (.5, .75, 1, 1),
                             (0, .25, .25, .5), (.75, .5, 1, .75)):
        cv2.line(img, (int(x0 * tex_w), int(y0 * tex_h)), (int(x1 * tex_w), int(y1 * tex_h)), col, w, cv2.LINE_AA)
        cv2.line(img, (int(x1 * tex_w), int(y0 * tex_h)), (int(x0 * tex_w), int(y1 * tex_h)), col, w, cv2.LINE_AA)
    s = tex_w // 9
    cx, cy = tex_w // 2, tex_h // 2
    cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), (0.05, 0.05, 0.05), -1)
    cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), col, w, cv2.LINE_AA)
    cv2.line(img, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), col, w + 1, cv2.LINE_AA)
    cv2.line(img, (cx + s // 2, cy - s // 2), (cx - s // 2, cy + s // 2), col, w + 1, cv2.LINE_AA)
    _START = img
    return img


def mirrored_back(assets, design, rng):
    """Back texture synthesized from a front one (layout mirrored, badge+icon region not)."""
    entry = assets.textures(design, "front")[int(rng.integers(len(assets.textures(design, "front"))))]
    a = assets.layers[entry["file"]]
    m = a[:, ::-1].copy()
    H, W = a.shape[:2]
    x0, y0, x1, y1 = 0.38, 0.40, 0.62, 0.67
    m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = a[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    key = f"__mirror__{entry['file']}"
    assets.layers[key] = m
    return dict(file=key, design=design, side="back")


def card_image(assets, card, rng, deck_rgb):
    if card.design == "start":
        return start_texture(assets.tex_w, assets.tex_h)
    options = assets.textures(card.design, card.side)
    if not options or (card.side == "back" and rng.random() < 0.15):
        entry = mirrored_back(assets, card.design, rng) if card.side == "back" else options[0]
    else:
        entry = options[int(rng.integers(len(options)))]
    face = np.clip(np.array([0.05, 0.05, 0.055]) * rng.uniform(0.6, 1.6), 0, 1)
    white = np.clip(np.array([0.96, 0.96, 0.94]) * rng.uniform(0.85, 1.0), 0, 1)
    return render_card(assets, entry, deck_rgb, face_rgb=face, white_rgb=white, icon=assets.icons[card.deck])


# ---------------------------------------------------------------- backgrounds

def streaks(rng, h, w, angle=None):
    """Fine wood grain: short sharp streaks and pores along one direction, >= 0, sparse.
    Under glare they glint as small bright marks (what a sharp detector mistakes for dots)."""
    ang = rng.uniform(0, np.pi) if angle is None else angle
    L = int(rng.integers(5, 25))
    kern = np.zeros((L, L), np.float32)
    kern[L // 2] = 1
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), np.degrees(ang), 1)
    kern = cv2.warpAffine(kern, M, (L, L))
    z = cv2.filter2D(rng.normal(0, 1, (h, w)).astype(np.float32), -1, kern / max(kern.sum(), 1e-6))
    z = z / (z.std() + 1e-6)
    return np.maximum(z - rng.uniform(0.8, 1.8), 0) ** rng.uniform(1.0, 2.0), ang


def table_texture(rng, h, w):
    kind = rng.choice(["dark_wood", "light_wood", "solid", "mat", "white", "noise"],
                      p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    if kind in ("dark_wood", "light_wood"):
        base = np.array([0.10, 0.07, 0.05]) if kind == "dark_wood" else np.array([0.62, 0.42, 0.22])
        base = base * rng.uniform(0.6, 1.4)
        ang = rng.uniform(0, np.pi)
        n = cv2.resize(rng.normal(0, 1, (h // 8 + 1, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        grain = cv2.resize(rng.normal(0, 1, (h, 24)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:  # broad grain along a random direction
            M = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(ang), 1.5)
            grain = cv2.warpAffine(grain, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        tex = base[None, None] * (1 + 0.15 * n[..., None] + 0.12 * grain[..., None])
        if rng.random() < 0.7:  # fine grain: light streaks and pores
            s, _ = streaks(rng, h, w, ang + rng.normal(0, 0.05))
            tex = tex + rng.uniform(0.02, 0.12) * s[..., None]
    elif kind == "white":
        tex = np.ones((h, w, 3)) * rng.uniform(0.75, 0.98)
    elif kind == "mat":
        base = rng.uniform(0.05, 0.6, 3)
        tex = base[None, None] + rng.normal(0, 0.03, (h, w, 1))
    elif kind == "noise":
        small = rng.uniform(0, 1, (h // 16 + 1, w // 16 + 1, 3)).astype(np.float32)
        tex = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * rng.uniform(0.2, 1.0)
    else:
        tex = np.ones((h, w, 3)) * rng.uniform(0.02, 0.5, 3)
    return np.clip(tex, 0, 1).astype(np.float32)


def _fabric(rng, h, w, px):
    """Textures found around a table: checkered rug, carpet, denim/cloth, dark trousers."""
    kind = rng.choice(["checker", "carpet", "denim", "dark"])
    if kind == "checker":
        cell = rng.uniform(1.0, 4.0) * px
        cols = rng.uniform(0.05, 0.95, (int(rng.integers(2, 5)), 3))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = rng.uniform(-0.3, 0.3)
        u = (xx * np.cos(ang) + yy * np.sin(ang)) / cell
        v = (-xx * np.sin(ang) + yy * np.cos(ang)) / cell
        idx = (np.floor(u).astype(int) * 7 + np.floor(v).astype(int) * 3) % len(cols)
        tex = cols[idx]
    elif kind == "carpet":
        tex = np.ones((h, w, 3)) * rng.uniform(0.2, 0.7, 3)
    elif kind == "denim":
        tex = np.ones((h, w, 3)) * (np.array([0.25, 0.35, 0.55]) * rng.uniform(0.4, 1.3))
    else:
        tex = np.ones((h, w, 3)) * rng.uniform(0.03, 0.2)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(0.5, 2))
    tex = tex * (1 + rng.uniform(0.05, 0.25) * grain[..., None])
    # folds / creases: broad soft brightness waves
    n = cv2.resize(rng.normal(0, 1, (6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    tex = tex * (1 + rng.uniform(0, 0.3) * n[..., None])
    return np.clip(tex, 0, 1).astype(np.float32)


def edge_scene(rng, img, card_mask, px):
    """What lies past the table edge along one side of the photo: floor, rug, someone's legs and
    shoes. The table edge is a soft curve; it stays clear of the cards."""
    h, w = img.shape[:2]
    side = int(rng.integers(0, 4))
    L = w if side in (0, 1) else h        # length of the photo edge
    D = h if side in (0, 1) else w        # depth into the photo
    for _ in range(4):
        depth = rng.uniform(0.08, 0.35) * D
        t = np.linspace(0, 1, L)
        prof = depth * (1 + rng.uniform(-0.4, 0.4) * (t - 0.5) + rng.uniform(-0.5, 0.5) * (t - 0.5) ** 2)
        d = np.arange(D)[:, None]
        m = (d < prof[None, :]).astype(np.float32)   # D x L, row 0 at the photo edge
        if side == 1:
            m = m[::-1]
        if side in (2, 3):
            m = m.T
            if side == 3:
                m = m[:, ::-1]
        if not (m > 0.5)[card_mask > 0].any():
            break
    else:
        return img
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.7, 2.5))
    tex = _fabric(rng, h, w, px)
    # legs + shoes coming in from the edge (dark trousers, light or dark shoe tip)
    for _ in range(int(rng.choice([0, 1, 2]))):
        pts = np.argwhere(m > 0.9)
        if not len(pts):
            break
        y, x = pts[rng.integers(len(pts))]
        leg = np.zeros((h, w), np.float32)
        ax = (rng.uniform(0.5, 1.4) * px, rng.uniform(1.5, 4) * px)
        cv2.ellipse(leg, ((float(x), float(y)), ax, rng.uniform(0, 180)), 1.0, -1)
        leg = cv2.GaussianBlur(leg, (0, 0), 2)[..., None]
        tex = tex * (1 - leg) + _fabric(rng, h, w, px) * 0.5 * leg
        shoe = np.zeros((h, w), np.float32)
        cv2.ellipse(shoe, ((float(x + rng.normal(0, px)), float(y + rng.normal(0, px))),
                           (rng.uniform(0.3, 0.8) * px, rng.uniform(0.6, 1.4) * px), rng.uniform(0, 180)), 1.0, -1)
        shoe = cv2.GaussianBlur(shoe, (0, 0), 1.5)[..., None]
        tex = tex * (1 - shoe) + rng.choice([0.85, 0.15]) * rng.uniform(0.8, 1.1, 3) * shoe
    m = m[..., None]
    return img * (1 - m) + tex * m


def gadget(rng, img, card_mask, px):
    """A dark rounded object (game controller, phone, remote) with small light buttons and rings:
    dot-like and badge-like shapes that are not on any card."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    ax = (rng.uniform(0.4, 1.5) * px, rng.uniform(0.3, 1.0) * px)
    ang = rng.uniform(0, 180)
    body = np.zeros((h, w), np.float32)
    cv2.ellipse(body, ((cx, cy), ax, ang), 1.0, -1)
    if rng.random() < 0.5:
        cv2.ellipse(body, ((cx + rng.normal(0, 0.3 * px), cy + rng.normal(0, 0.3 * px)), ax, ang + 30), 1.0, -1)
    if (body > 0.01)[card_mask > 0].any():
        return img
    body = cv2.GaussianBlur(body, (0, 0), 1.2)[..., None]
    col = rng.uniform(0.02, 0.15) * rng.uniform(0.8, 1.2, 3)
    if rng.random() < 0.2:
        col = rng.uniform(0.6, 0.95, 3)  # light object (white phone, paper, shoe)
    out = img * (1 - body) + col * body
    deco = np.zeros((h, w), np.float32)
    r_dot = 0.035 * px
    for _ in range(int(rng.integers(1, 6))):
        x, y = cx + rng.normal(0, 0.25 * ax[0]), cy + rng.normal(0, 0.25 * ax[1])
        r = r_dot * rng.uniform(0.6, 2.5)
        if rng.random() < 0.5:
            cv2.circle(deco, (int(x * 4), int(y * 4)), int(r * 4), 1.0, -1, cv2.LINE_AA, shift=2)
        else:
            cv2.circle(deco, (int(x * 4), int(y * 4)), int(r * 4), 1.0, max(1, int(r * 0.4)), cv2.LINE_AA, shift=2)
    deco = (deco * body[..., 0])[..., None]
    return out * (1 - deco) + rng.uniform(0.6, 1.0) * deco


# ---------------------------------------------------------------- camera + photometry

def camera(rng, cards, out_w, out_h, px_per_card, jitter=True):
    """Homography board -> image: the board plane seen by a tilted pinhole camera (phone-like
    focal length), scaled so a card at the image centre is ~px_per_card wide."""
    pts = np.vstack([c.polygon() for c in cards])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2 + (rng.normal(0, 0.15, 2) * (hi - lo) if jitter else 0)
    phi = rng.uniform(0, 2 * np.pi)
    S = np.eye(3)
    S[:2, :2] = rot(phi) * px_per_card
    S[:2, 2] = -(rot(phi) * px_per_card) @ centre      # board -> px, centred on the image centre
    # tilt: mostly near overhead, sometimes a steep view across the table
    u = rng.random()
    tilt = np.deg2rad(rng.uniform(0, 15) if u < 0.5 else rng.uniform(15, 35) if u < 0.8 else rng.uniform(35, 58))
    axis = rng.uniform(0, 2 * np.pi)                     # direction the camera leans
    f = rng.uniform(0.6, 1.1) * max(out_w, out_h)
    a = np.array([np.cos(axis), np.sin(axis), 0.0])      # rotation axis, in the board plane
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(tilt) * Kx + (1 - np.cos(tilt)) * Kx @ Kx
    # plane point (x, y) [centred px] -> camera R @ (x, y, 0) + (0, 0, f) -> pinhole projection
    K = np.array([[f, 0, out_w / 2], [0, f, out_h / 2], [0, 0, 1]])
    P = K @ np.column_stack([R[:, 0], R[:, 1], [0, 0, f]]) / f
    # small extra keystone (lens/handheld)
    src = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32)
    dst = src + rng.normal(0, 0.01, src.shape).astype(np.float32) * np.array([out_w, out_h], np.float32)
    J = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return J @ P @ S


def apply_h(H, pts):
    pts = np.atleast_2d(pts)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def local_scale(H, p):
    """Image-space card width (px) at board point p."""
    a = apply_h(H, np.array([p, p + [1, 0], p + [0, 1]]))
    return float((np.linalg.norm(a[1] - a[0]) + np.linalg.norm(a[2] - a[0])) / 2)


def photometric(rng, img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # smooth illumination field + colour cast
    gx, gy = rng.normal(0, 0.35, 2)
    illum = 1 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    illum *= rng.uniform(0.6, 1.25)
    # Colour cast. Phones auto white-balance: in real photos the white ink is within ~12% of
    # neutral (mostly warm). Wider casts make printed red and pink indistinguishable (a warm cast
    # turns pink into red), so the range stays close to what phones produce.
    cast = rng.normal(1, 0.035, 3).astype(np.float32)
    if rng.random() < 0.6:  # white balance: warm lamps .. cool daylight, plus green/magenta tint
        t, g = rng.uniform(-0.12, 0.18), rng.normal(0, 0.03)
        cast = cast * np.array([1 + t, 1 + g, 1 - t], np.float32)
    cast = cast / cast.mean()
    img = img * illum[..., None] * cast[None, None]
    if rng.random() < 0.25:  # mixed light: a second colour temperature across the table
        t2 = rng.uniform(-0.12, 0.12)
        c2 = np.array([1 + t2, 1, 1 - t2], np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        ramp = np.clip(0.5 + ((xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)) * rng.uniform(1, 2.5), 0, 1)
        img = img * (1 + (c2[None, None] - 1) * ramp[..., None])
    # shadows
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(0.05, 0.4) * w, rng.uniform(0.05, 0.4) * h
        g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        img = img * (1 - rng.uniform(0.2, 0.6) * g[..., None])
    # glare: specular wash-out (brightens, desaturates, flattens contrast)
    # Strong veils (an overhead lamp reflected in the cards) lift black to ~0.8 of white; the
    # surface's fine grain then glints (streaky highlights).
    n_glare = int(rng.choice([0, 0, 1, 1, 2, 3]))
    for _ in range(n_glare):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        big = rng.random() < 0.3
        sx, sy = (rng.uniform(0.12, 0.45, 2) if big else rng.uniform(0.04, 0.3, 2)) * np.array([w, h])
        ang = rng.uniform(0, np.pi)
        X = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        Y = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        g = np.exp(-((X / sx) ** 2 + (Y / sy) ** 2) ** rng.uniform(0.6, 1.4))
        strength = rng.uniform(0.3, 1.15)
        tex = 1 + 0.08 * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
        if rng.random() < 0.6:
            tex = tex + rng.uniform(0.3, 1.5) * streaks(rng, h, w)[0]
        gcol = np.array([1.0, rng.uniform(0.95, 1.0), rng.uniform(0.88, 1.0)], np.float32)
        a = (strength * g * tex)[..., None]
        gray = img.mean(axis=2, keepdims=True)
        img = img * (1 - 0.5 * a) + gray * 0.5 * a * 0.3 + a * gcol * 0.75
    img = np.clip(img, 0, 1)
    # blur, noise, gamma. Most real photos are downscaled from 12 MP and so look sharp at the
    # model's scale; a minority are soft (small web images, focus, motion).
    soft = rng.random() < 0.15
    if soft or rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3) if soft else rng.uniform(0.2, 0.7))
    if rng.random() < (0.15 if soft else 0.05):
        k = int(rng.integers(3, 8))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kern = cv2.warpAffine(kern, M, (k, k))
        img = cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))
    img = img + rng.normal(0, rng.uniform(0, 0.03 if soft else 0.015), img.shape).astype(np.float32)
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- full sample

def make_sample(rng, assets, out_w=1280, out_h=960, px_per_card=None, frame_all=False, recolour=0.0):
    """frame_all: the whole board is in the picture (like a real end-of-game photo)."""
    cards = layout_board(rng)
    vis = visibility(cards)
    if px_per_card is None:
        px_per_card = float(np.exp(rng.uniform(np.log(70), np.log(150))))
    if frame_all:
        px_per_card = float(rng.uniform(85, 120))
        state = rng.bit_generator.state
        for _ in range(40):
            rng.bit_generator.state = state  # same camera draw, only the scale changes
            H = camera(rng, cards, out_w, out_h, px_per_card, jitter=False)
            q = apply_h(H, np.vstack([c.polygon() for c in cards]))
            m = 0.03 * min(out_w, out_h)
            if q[:, 0].min() > m and q[:, 1].min() > m and q[:, 0].max() < out_w - m and q[:, 1].max() < out_h - m:
                break
            px_per_card *= 0.93
    else:
        H = camera(rng, cards, out_w, out_h, px_per_card)

    # table in board space, warped (so its texture follows the perspective)
    img = table_texture(rng, out_h, out_w)
    tw, th = assets.tex_w, assets.tex_h
    # print/camera variation: relative, plus a little absolute so near-zero channels vary too
    deck_rgb = {d: np.clip(np.array(DECK_RGB[d]) * rng.normal(1, 0.08, 3) + rng.normal(0, 0.025, 3), 0, 1)
                for d in DECKS}
    alpha_tex = card_alpha(tw, th)
    # card texture px -> normalized card (x, y) -> board -> image
    # Icon-first colour reading: with probability `recolour` a card is printed in (or most of the
    # way towards) another deck's colour but keeps its own icon and label, so colour alone is not
    # a reliable cue and the badge head must read the icon. (Training only; test sets use 0.) Its
    # dots are labelled with the colour actually shown: a dot can't see the icon.
    shown = {}
    for c in cards:
        rgb = deck_rgb.get(c.deck, (1, 1, 1))
        shown[c.idx] = c.deck
        if recolour > 0 and c.deck in deck_rgb and rng.random() < recolour:  # (no rng draw at 0: old sets regenerate identically)
            other = str(rng.choice([d for d in DECKS if d != c.deck]))
            t = rng.uniform(0.6, 1.0)
            rgb = (1 - t) * np.asarray(rgb) + t * np.asarray(deck_rgb[other])
            shown[c.idx] = other
        tex = card_image(assets, c, rng, rgb)
        side = c.side
        c.side = "front"  # physical corners: no mirroring for the texture mapping
        corners_img = apply_h(H, c.polygon()).astype(np.float32)
        c.side = side
        # Area-downsample the texture to about its size in the image first (a phone photo is
        # downscaled from a sharp high-res image; warping a 3-5x larger texture with bilinear
        # sampling instead aliases fine lines and makes dots look smaller and less white).
        on_img = max(np.linalg.norm(corners_img[1] - corners_img[0]), np.linalg.norm(corners_img[3] - corners_img[0]) / CARD_H)
        k = min(1.0, 1.25 * on_img / tw)
        tex_s, alpha_s = tex, alpha_tex
        if k < 0.95:
            sz = (max(8, int(round(tw * k))), max(8, int(round(th * k))))
            tex_s = cv2.resize(tex, sz, interpolation=cv2.INTER_AREA)
            alpha_s = cv2.resize(alpha_tex, sz, interpolation=cv2.INTER_AREA)
        sw, sh_ = tex_s.shape[1], tex_s.shape[0]
        # texture is already in the card's own orientation (backs are back textures),
        # so map texture corners to the card's physical corners without mirroring
        corners_tex = np.array([[0, 0], [sw, 0], [sw, sh_], [0, sh_]], np.float32)
        # work only inside the card's bounding box (plus shadow margin)
        pad = int(px_per_card * 0.15) + 4
        x0 = max(0, int(np.floor(corners_img[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(corners_img[:, 1].min())) - pad)
        x1 = min(out_w, int(np.ceil(corners_img[:, 0].max())) + pad)
        y1 = min(out_h, int(np.ceil(corners_img[:, 1].max())) + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        M = cv2.getPerspectiveTransform(corners_tex, (corners_img - [x0, y0]).astype(np.float32))
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpPerspective(tex_s, M, size, flags=cv2.INTER_LINEAR)
        a = cv2.warpPerspective(alpha_s, M, size, flags=cv2.INTER_LINEAR)
        roi = img[y0:y1, x0:x1]
        # soft contact shadow
        sh = cv2.GaussianBlur(a, (0, 0), max(1.0, px_per_card * 0.03))
        roi = roi * (1 - 0.35 * sh[..., None])
        img[y0:y1, x0:x1] = roi * (1 - a[..., None]) + warped * a[..., None]

    # clutter: miniature card art (game box), coloured rectangles, skin-toned blobs.
    # Clutter never overlaps cards (ground truth would no longer match what is visible).
    card_mask = np.zeros((out_h, out_w), np.uint8)
    for c in cards:
        cv2.fillConvexPoly(card_mask, apply_h(H, c.polygon()).astype(np.int32), 1)
    card_mask = cv2.dilate(card_mask, np.ones((9, 9), np.uint8))
    for _ in range(int(rng.integers(0, 4))):
        kind = rng.random()
        cx, cy = rng.uniform(0, out_w), rng.uniform(0, out_h)
        if kind < 0.4:
            mini = Card(-1, str(rng.choice(list(DESIGNS))), "front", str(rng.choice(DECKS)), (0, 0), 0)
            tex = card_image(assets, mini, rng, deck_rgb[mini.deck])
            s = rng.uniform(0.2, 0.45) * px_per_card / tw
            M = cv2.getRotationMatrix2D((tw / 2, th / 2), rng.uniform(0, 360), s)
            M[:, 2] += (cx - tw / 2, cy - th / 2)
            w_ = cv2.warpAffine(tex, M, (out_w, out_h))
            a_ = cv2.warpAffine(alpha_tex, M, (out_w, out_h))[..., None]
            if (a_[..., 0] > 0.01)[card_mask > 0].any():
                continue
            img = img * (1 - a_) + w_ * a_
        else:
            col = rng.uniform(0, 1, 3) if kind < 0.75 else np.array([0.85, 0.62, 0.5]) * rng.uniform(0.7, 1.1)
            box = cv2.boxPoints(((cx, cy), (rng.uniform(0.3, 2.5) * px_per_card, rng.uniform(0.3, 2.5) * px_per_card),
                                 rng.uniform(0, 180))).astype(np.int32)
            m = np.zeros((out_h, out_w), np.float32)
            if kind < 0.75:
                cv2.fillConvexPoly(m, box, 1.0)
            else:
                cv2.ellipse(m, ((cx, cy), (rng.uniform(0.6, 2) * px_per_card, rng.uniform(1.5, 4) * px_per_card),
                                rng.uniform(0, 180)), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]
            if (m[..., 0] > 0.01)[card_mask > 0].any():
                continue
            img = img * (1 - m) + col[None, None] * m
    # negatives seen in real photos: rug/floor/legs beyond the table edge, gadgets with buttons
    if rng.random() < 0.4:
        img = edge_scene(rng, img, card_mask, px_per_card)
    for _ in range(int(rng.choice([0, 0, 1, 1, 2]))):
        img = gadget(rng, img, card_mask, px_per_card)

    img = photometric(rng, img)

    # ground truth in image coordinates
    gt_cards, gt_dots = [], []
    for c, v in zip(cards, vis):
        b = apply_h(H, c.badge())[0]
        ic = apply_h(H, c.icon())[0]
        gt_cards.append(dict(idx=c.idx, design=c.design, side=c.side, deck=c.deck,
                             badge=b.round(2).tolist(), icon=ic.round(2).tolist(),
                             card_w=round(local_scale(H, c.center), 2),
                             corners=apply_h(H, c.polygon()).round(1).tolist()))
        for p, fr in zip(apply_h(H, c.dots()) if len(c.dots()) else [], v):
            gt_dots.append(dict(xy=p.round(2).tolist(), card=c.idx, deck=c.deck, colour=shown.get(c.idx, c.deck),
                                frac=round(float(fr), 3),
                                visible=bool(fr >= FULL), partial=bool(HIDDEN < fr < FULL)))
    gt = dict(width=out_w, height=out_h, px_per_card=px_per_card, cards=gt_cards, dots=gt_dots)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)[..., ::-1], gt


def draw_gt(img, gt):
    vis = img.copy()
    for c in gt["cards"]:
        cv2.polylines(vis, [np.array(c["corners"], np.int32)], True, (90, 90, 90), 1)
        b, i = np.array(c["badge"]), np.array(c["icon"])
        cv2.arrowedLine(vis, tuple(b.astype(int)), tuple(i.astype(int)), (255, 255, 255), 2, tipLength=0.4)
        cv2.putText(vis, f"{c['design']}{'b' if c['side'] == 'back' else ''}", tuple((b + [6, -6]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    for d in gt["dots"]:
        p = tuple(np.array(d["xy"]).astype(int))
        cv2.circle(vis, p, 6, (0, 255, 0) if d["visible"] else (0, 200, 255) if d.get("partial") else (0, 0, 255), 2 if d.get("partial") else 1)
    return vis


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    assets = Assets()
    out = os.path.join(paths.OUT, "synth-preview")
    os.makedirs(out, exist_ok=True)
    for k in range(4):
        img, gt = make_sample(rng, assets)
        cv2.imwrite(os.path.join(out, f"board_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(out, f"board_{k}_gt.jpg"), draw_gt(img, gt), [cv2.IMWRITE_JPEG_QUALITY, 92])
        nv = sum(d["visible"] for d in gt["dots"])
        npart = sum(d.get("partial", False) for d in gt["dots"])
        print(f"board {k}: {len(gt['cards'])} cards, {len(gt['dots'])} dots, {nv} visible, {npart} partial")
