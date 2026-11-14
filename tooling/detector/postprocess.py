"""Classical post-processing of detector output: card poses from badges, template dot positions,
consistency checks, and final per-deck scores. (The app will port this to TypeScript.)

Card pose from a badge: the icon sits a fixed distance "below" the badge in card coordinates, so
the badge->icon vector gives both the card's down direction and its local width in pixels.
"""
import json
import os

from collections import Counter
from itertools import permutations

import numpy as np

from tooling.card_library.build_card_library import REPO_ROOT
from tooling.detector.model import DECKS

LIB = json.load(open(os.path.join(REPO_ROOT, "src", "data", "cards", "card_library.json")))
ASPECT = LIB["aspect_w_over_h"]
DESIGNS = {c["id"]: c for c in LIB["cards"]}
BADGE_BY_SUIT = {c["suit"]: c["badge"] for c in LIB["cards"]}
ICON = LIB["cards"][0]["icon"]
DOT_R = LIB["dot_radius_over_width"]
EXPLAIN_TOL = 0.06        # card widths, with the whole-board perspective fit
EXPLAIN_TOL_NO_FIT = 0.15  # card widths, per-card poses only (ignore perspective shear)


def icon_distance(suit):
    """badge->icon distance in card widths."""
    return (ICON["y"] - BADGE_BY_SUIT[suit]["y"]) / ASPECT


def card_pose(badge, min_len=1e-3):
    """Returns dict(center_badge, u (right), v (down), width_px) or None for the start card.
    min_len: shortest usable badge->icon vector (image px; ~0 in rectified units)."""
    if badge["suit"] == "start":
        return None
    b = np.array([badge["x"], badge["y"]])
    d = np.array(badge["icon"]) - b
    L = np.linalg.norm(d)
    if L < min_len or L < 1e-12:
        return None
    v = d / L
    u = np.array([v[1], -v[0]])  # card "right" = card "down" rotated -90 deg (image y points down)
    width = L / icon_distance(badge["suit"])
    return dict(b=b, u=u, v=v, width=float(width))


def to_image(pose, badge, xn, yn):
    """Normalized front-orientation card coords -> image px for this (possibly back-side) card."""
    bl = BADGE_BY_SUIT[badge["suit"]]
    bx = bl["x"]
    if badge["side"] == "back":
        xn, bx = 1 - np.asarray(xn), 1 - bx
    dx = (np.asarray(xn) - bx) * pose["width"]
    dy = (np.asarray(yn) - bl["y"]) / ASPECT * pose["width"]
    return pose["b"] + np.outer(dx, pose["u"]) + np.outer(dy, pose["v"])


def card_corners(pose, badge):
    return to_image(pose, badge, np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))


def refine_pose(pose, badge, design, dots_xy, max_frac=0.09):
    """Refit the card's similarity transform to its badge, icon and the detected dots that sit near
    its template dot positions (corrects perspective foreshortening away from the badge).
    Returns (pose, matched template-dot indices)."""
    import cv2
    tx = np.array([p["x"] for p in design["dots"]])
    ty = np.array([p["y"] for p in design["dots"]])
    pts = to_image(pose, badge, tx, ty)
    matched = {}
    if len(dots_xy):
        for k, p in enumerate(pts):
            dist = np.hypot(dots_xy[:, 0] - p[0], dots_xy[:, 1] - p[1])
            j = int(np.argmin(dist))
            if dist[j] < max_frac * pose["width"]:
                matched[k] = j
    if not matched:
        return pose, matched
    # source: card-local units (x right, y down, origin at badge); destination: image px
    bl = BADGE_BY_SUIT[badge["suit"]]
    bx = 1 - bl["x"] if badge["side"] == "back" else bl["x"]
    xs = (1 - tx) if badge["side"] == "back" else tx
    ix = (1 - ICON["x"]) if badge["side"] == "back" else ICON["x"]
    src = [(0.0, 0.0), (ix - bx, icon_distance(badge["suit"]))]
    dst = [tuple(pose["b"]), tuple(badge["icon"])]
    for k, j in matched.items():
        src.append((xs[k] - bx, (ty[k] - bl["y"]) / ASPECT))
        dst.append(tuple(dots_xy[j]))
    M, _ = cv2.estimateAffinePartial2D(np.array(src, np.float32), np.array(dst, np.float32), method=cv2.LMEDS)
    if M is None:
        return pose, matched
    s = float(np.hypot(M[0, 0], M[1, 0]))
    u = M[:, 0] / s
    v = M[:, 1] / s
    # keep handedness consistent with the original pose (similarity: v = u rotated +90 deg)
    return dict(b=M[:, 2].copy(), u=u, v=v, width=s), matched


def resolve_duplicates(badges):
    """Each deck holds exactly one of each design. If two cards of one deck claim the same design,
    the less confident one is moved to its best still-unused value of the same suit."""
    taken = {}
    order = sorted([b for b in badges if b["suit"] != "start"], key=lambda b: -b["value_p"])
    fixes = []
    for b in order:
        key = (b["deck"], b["suit"])
        used = taken.setdefault(key, set())
        if b["value"] in used:
            probs = b.get("value_probs")
            if probs is not None:
                for v in np.argsort(probs)[::-1] + 1:
                    if v not in used:
                        fixes.append((b["deck"], b["suit"], b["value"], int(v)))
                        b["value"] = int(v)
                        break
        used.add(b["value"])
    return fixes


def card_template_points(b, pts_norm, G=None, Gi=None, rot=0.0, icon_len=None):
    """Image positions of normalized card points; through the rectified plane when G is known.

    rot: extra rotation (radians) of the badge->icon direction; icon_len: override the
    badge->icon length (rectified units when G is given, px otherwise). Used to let detected dots
    correct small errors in the model's icon vector."""
    xs, ys = [p[0] for p in pts_norm], [p[1] for p in pts_norm]

    def adjust(bx, by, ix, iy):
        v = np.array([ix - bx, iy - by])
        L = np.hypot(*v)
        if L < 1e-9:
            return ix, iy
        c, s = np.cos(rot), np.sin(rot)
        v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        if icon_len is not None:
            v = v / L * icon_len
        return bx + v[0], by + v[1]

    if G is None:
        ix, iy = adjust(b["x"], b["y"], *b["icon"])
        bb = dict(b, icon=[ix, iy])
        pose = card_pose(bb)
        return to_image(pose, bb, xs, ys), pose["width"]
    from tooling.detector import perspective as PS
    br, ir = PS.apply(G, [(b["x"], b["y"]), b["icon"]])
    ix, iy = adjust(br[0], br[1], ir[0], ir[1])
    rb = dict(b, x=br[0], y=br[1], icon=[ix, iy])
    pts = PS.apply(Gi, to_image(card_pose(rb, min_len=0), rb, xs, ys))
    width = card_pose(b)["width"]
    if icon_len is not None:  # image-space width scales with the corrected length
        width *= icon_len / max(np.hypot(*(ir - br)), 1e-9)
    return pts, width


def count_hits(pts, width, dots_xy, tol_frac, weights=None):
    """(hits within tolerance, soft alignment score). The soft score peaks at exact alignment, so a
    slightly mis-scaled/rotated template can't tie with the true one. weights (per detected dot)
    down-weight dots already explained by other cards ("explaining away")."""
    if not len(dots_xy) or not len(pts):
        return 0, 0.0
    D = np.hypot(dots_xy[None, :, 0] - pts[:, None, 0], dots_xy[None, :, 1] - pts[:, None, 1])
    sigma = 0.5 * tol_frac * width
    g = np.exp(-0.5 * (D / sigma) ** 2)
    if weights is not None:
        g = g * weights[None, :]
    return int((D.min(axis=1) < tol_frac * width).sum()), float(g.max(axis=1).sum())


def template_gauss(pts, width, dots_xy, tol_frac):
    """Per detected dot: how well this template explains it (0..1)."""
    if not len(dots_xy) or not len(pts):
        return np.zeros(len(dots_xy))
    D = np.hypot(dots_xy[None, :, 0] - pts[:, None, 0], dots_xy[None, :, 1] - pts[:, None, 1])
    return np.exp(-0.5 * (D / (0.5 * tol_frac * width)) ** 2).max(axis=0)


ROT_SEARCH = np.deg2rad(np.arange(-20, 21, 2))
VALUE_PRIOR = 2.5  # weight of log p_model(value); a value change needs clearly more matching dots


def fit_card_options(b, G, Gi, dots_xy, tol_frac, icon_len, weights=None):
    """For each value 1..6: the best (score, side, rotation) for this card. Score = soft count of
    template dots landing on detected dots + the model's value probability as a prior."""
    probs = b.get("value_probs")
    options = {}
    for value in range(1, 7):
        p = probs[value - 1] if probs is not None else (1.0 if value == b["value"] else 1e-3)
        prior = VALUE_PRIOR * np.log(max(p, 1e-3))
        tpl = [(q["x"], q["y"]) for q in DESIGNS[f"{b['suit']}-{value}"]["dots"]]
        best = None
        for side in ("front", "back"):
            bb = dict(b, value=value, side=side)
            for rot in ROT_SEARCH:
                pts, w = card_template_points(bb, tpl, G, Gi, rot=rot, icon_len=icon_len(b))
                _, soft = count_hits(pts, w, dots_xy, tol_frac, weights)
                # tie-breaks: the model's side, no rotation
                sc = soft + prior - 0.01 * (side != b["side"]) - 0.002 * abs(rot)
                if best is None or sc > best[0]:
                    best = (sc, side, rot)
        options[value] = best
    return options


def assign_values(real, options):
    """Each deck has exactly one card per design: within each (deck, badge shape) group pick the
    value assignment with the best total score (all permutations; groups have <= 6 cards).
    Returns {id(badge): (value, side, rot)}."""
    from itertools import permutations
    fits, groups = {}, {}
    for b in real:
        groups.setdefault((b["deck"], b["suit"]), []).append(b)
    for grp in groups.values():
        if len(grp) > 6:  # more cards than designs (a deck colour misread): no uniqueness possible
            for b in grp:
                v = max(options[id(b)], key=lambda v: options[id(b)][v][0])
                fits[id(b)] = (v,) + options[id(b)][v][1:]
            continue
        best_total, best_perm = None, None
        for perm in permutations(range(1, 7), len(grp)):
            tot = sum(options[id(b)][v][0] for b, v in zip(grp, perm))
            if best_total is None or tot > best_total:
                best_total, best_perm = tot, perm
        for b, v in zip(grp, best_perm):
            fits[id(b)] = (v,) + options[id(b)][v][1:]
    return fits


def point_in_quad(p, quad, grow=0.0):
    """Point inside a convex quad, optionally grown about its centre by a fraction."""
    q = np.asarray(quad, float)
    c = q.mean(axis=0)
    q = c + (q - c) * (1 + grow)
    sgn = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        sgn.append((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    return all(v >= 0 for v in sgn) or all(v <= 0 for v in sgn)


def quad_distance(p, quad):
    """Distance from a point to a convex quad (0 inside)."""
    if point_in_quad(p, quad):
        return 0.0
    best = np.inf
    for i in range(4):
        a, b = np.asarray(quad[i], float), np.asarray(quad[(i + 1) % 4], float)
        ab, ap = b - a, np.asarray(p, float) - a
        t = np.clip(ap @ ab / max(ab @ ab, 1e-12), 0, 1)
        best = min(best, float(np.hypot(*(ap - t * ab))))
    return best


REVIEW_REACH = 1.0  # card widths from the nearest found card

SIZE_TOL = 2.0  # rejects only objects < half or > double the board's card size (box art is ~1/3); per-card icon lengths are noisy
SIZE_REF_SCORE = 0.6  # badges this confident set the table's card size (real cards ~0.7-0.95, junk mostly < 0.6)


def fit_perspective(cards_in, image_size):
    from tooling.detector import perspective as PS
    G = PS.fit([(b["x"], b["y"]) for b in cards_in], [b["icon"] for b in cards_in],
               [icon_distance(b["suit"]) for b in cards_in], image_size)
    return G


def rectified_sizes(G, cards_in):
    from tooling.detector import perspective as PS
    out = []
    for b in cards_in:
        br, ir = PS.apply(G, [(b["x"], b["y"]), b["icon"]])
        out.append(np.hypot(*(ir - br)) / icon_distance(b["suit"]))
    return np.array(out)


# Minimum detector confidence for a dot to count, by where it sits (from score distributions of
# real vs false dots on whole-board synthetic tests): on a printed dot position, merely on a card,
# or off every card (never counted; confident ones are shown for review).
MIN_SCORE_EXPLAINED = 0.2   # a dot exactly on a printed position is strong evidence even if faint (glare)
MIN_SCORE_ON_CARD = 0.5
MIN_SCORE_REVIEW = 0.5
MIN_BADGE_SCORE_NO_DOTS = 0.55
MIN_DECK_P_NO_DOTS = 0.6
# A card more than ISOLATED_GAP card widths from every other card needs a believable reading.
# Clutter seen (photos + synthetic): badge score 0.34-0.46, or score 0.6 with both value and
# colour unsure (0.43 / 0.71). Real lone cards (synthetic, hard light): score >= 0.52, and never
# both probabilities low. Layout photos have gaps up to ~0.16 card widths.
DOT_VOTE_WEIGHT = 1.0  # weight of each dot's colour reading in its card's colour vote (badge = 1)
# Red vs pink: weight of the dots' say between these two look-alike colours (their say on
# red-or-pink vs other colours is unchanged). In real photos the badge tells them apart well but
# a dot (white disc, thin colour rim) does not: at full weight, dots outvoted confident red badges
# (red/purple game: 13 red dots counted pink). At 0.25 that game is exact; synthetic boards with
# both decks lose ~1% of exact boards (their dots are more reliable than real ones). 0 = badge only.
DOT_RED_PINK_WEIGHT = 0.25
I_RED, I_PINK = DECKS.index("red"), DECKS.index("pink")
LOOKALIKE_PAIR = ("red", "pink")
DECK_CARDS = 18
MOVE_MAX_GAP = 6.0  # log-odds: the badge's reading of a card's new deck vs its current one
MIN_DECK_IN_PLAY = 4  # fewer cards of a colour are clutter or misreadings, not a deck in play
# Deck icons (the shape under the badge, read classically from the photo: icons.py) are the primary
# deck evidence; the detector's colour reading of the badge is secondary (white balance and light
# shift colours, red vs pink especially). A card's deck log-odds = ICON_BETA * (icon score - best
# icon score) + COLOUR_WEIGHT * log(colour probability). Real photos: correct icon reads have a
# score margin of ~0.44 (median), 1% below 0.09; the one wrong read had 0.05.
ICON_BETA = 15.0
COLOUR_WEIGHT = 0.3
ISOLATED_GAP = 0.4
MIN_BADGE_SCORE_ISOLATED = 0.5
UNSURE_VALUE_P = 0.6
UNSURE_DECK_P = 0.8


def is_weak_reading(b):
    return b["score"] < MIN_BADGE_SCORE_ISOLATED or (b.get("value_p", 1) < UNSURE_VALUE_P
                                                     and b.get("deck_p", 1) < UNSURE_DECK_P)


def quad_gap(a, b):
    """0 if two quads overlap, else the smallest corner-to-quad distance either way."""
    if any(point_in_quad(p, b) for p in a) or any(point_in_quad(p, a) for p in b):
        return 0.0
    return min(min(quad_distance(p, b) for p in a), min(quad_distance(p, a) for p in b))


def fuse_icon(b):
    """Combine a badge's icon reading (icon_scores) with its colour reading (deck_probs, kept as
    colour_probs): deck_probs / deck / deck_p become the combined belief."""
    s = b.get("icon_scores")
    if s is None or b.get("deck_probs") is None or "colour_probs" in b:
        return
    lp = [COLOUR_WEIGHT * float(np.log(min(max(float(p), 1e-4), 1.0))) for p in b["deck_probs"]]
    top = max(float(v) for v in s)
    lp = [a + ICON_BETA * (float(v) - top) for a, v in zip(lp, s)]
    m = max(lp)
    e = [float(np.exp(a - m)) for a in lp]
    z = sum(e)
    b["colour_probs"] = b["deck_probs"]
    b["deck_probs"] = [v / z for v in e]
    i = int(np.argmax(b["deck_probs"]))
    b["deck"], b["deck_p"] = DECKS[i], b["deck_probs"][i]


def score(badges, dots, image_size=None):
    """Final result: per-deck visible dot counts + per-card details + warnings.

    Policy: a dot is counted only if it sits on a printed dot position of a detected card; its owner
    is that card (when several cards have a printed dot there, i.e. stacked dots, the one whose deck
    matches the dot's colour). Dots matching no card are returned for review, not counted.
    Partly covered dots (decoded with partial=True) are never counted or used for fitting; those
    on a found card are returned in "partial" so the app can show them."""
    warnings = []
    for b in badges:
        fuse_icon(b)
    partial = [d for d in dots if d.get("partial")]
    dots = [d for d in dots if not d.get("partial")]
    starts = [b for b in badges if b["suit"] == "start"]
    for b in starts:
        if b is not max(starts, key=lambda t: t["score"]):
            b["rejected"] = "extra start card"
    real = [b for b in badges if b["suit"] != "start" and card_pose(b) is not None]

    # whole-board perspective, then reject "cards" of the wrong physical size (box art, clutter)
    G = Gi = None
    if image_size is not None and len(real) >= 6:
        # The table's card size comes from confident badges when there are enough: busy
        # surroundings (carpet, clothes, box art) can yield more weak junk badges than real cards.
        conf = [b for b in real if b["score"] >= SIZE_REF_SCORE]
        ref = conf if len(conf) >= 6 else real
        G = fit_perspective(ref, image_size)
        if G is not None:
            sz = rectified_sizes(G, real)
            med = np.median(rectified_sizes(G, ref))
            bad = np.abs(np.log(sz / med)) > np.log(SIZE_TOL)
            for b, is_bad in zip(real, bad):
                b["rejected"] = "size" if is_bad else None
            if bad.any():
                warnings.append(f"ignored {int(bad.sum())} badge(s) whose size doesn't fit the table "
                                f"(e.g. game box art or other objects)")
                real = [b for b, is_bad in zip(real, bad) if not is_bad]
                G = fit_perspective(real, image_size) if len(real) >= 6 else None
            Gi = np.linalg.inv(G) if G is not None else None
    tol_frac = EXPLAIN_TOL if G is not None else EXPLAIN_TOL_NO_FIT
    dots_xy = np.array([(d["x"], d["y"]) for d in dots]) if dots else np.zeros((0, 2))

    # Card size: every card has the same physical size, so use one board-wide badge->icon length
    # (rectified units if the perspective fit exists, else px), times a global factor k chosen so
    # that the most detected dots land on template dots (the model's icon lengths can be biased).
    if G is not None:
        base = float(np.median(rectified_sizes(G, real))) if real else 1.0
    else:
        base = float(np.median([np.hypot(*np.subtract(b["icon"], (b["x"], b["y"]))) / icon_distance(b["suit"])
                                for b in real])) if real else 1.0
    best_k, best_hits = 1.0, -1
    for k in np.arange(0.75, 1.3001, 0.025):
        tot = 0
        for b in real:
            tpl = [(q["x"], q["y"]) for q in DESIGNS[f"{b['suit']}-{b['value']}"]["dots"]]
            pts, w = card_template_points(b, tpl, G, Gi, icon_len=k * base * icon_distance(b["suit"]))
            tot += count_hits(pts, w, dots_xy, tol_frac)[1]
        if tot > best_hits:
            best_k, best_hits = float(k), tot
    icon_len = lambda b: best_k * base * icon_distance(b["suit"])

    # Fit value/side/rotation per card. Pass 1: independently. Pass 2: dots that OTHER cards'
    # templates already explain count much less, so a card can't win by borrowing neighbours' dots.
    model_value = {id(b): b["value"] for b in real}
    opts = {id(b): fit_card_options(b, G, Gi, dots_xy, tol_frac, icon_len) for b in real}
    fits = assign_values(real, opts)

    def explain_map(b, fit):
        value, side, rot = fit
        tpl = [(q["x"], q["y"]) for q in DESIGNS[f"{b['suit']}-{value}"]["dots"]]
        pts, w = card_template_points(dict(b, value=value, side=side), tpl, G, Gi, rot=rot, icon_len=icon_len(b))
        return template_gauss(pts, w, dots_xy, tol_frac)

    if len(dots_xy) and len(real) > 1:
        maps = {id(b): explain_map(b, fits[id(b)]) for b in real}
        M = np.stack([maps[id(b)] for b in real])
        for i_b, b in enumerate(real):
            others = np.delete(M, i_b, axis=0).max(axis=0)
            opts[id(b)] = fit_card_options(b, G, Gi, dots_xy, tol_frac, icon_len, weights=1 - 0.8 * others)
        fits = assign_values(real, opts)

    cards, expected = [], []  # expected: (x, y, card_idx)
    changed = 0
    for i, b in enumerate(badges):
        if b not in real:
            continue
        value, side, rot = fits[id(b)]
        changed += value != model_value[id(b)]
        b["value"], b["side"], b["rot_fix"] = value, side, float(rot)
        design = DESIGNS[f"{b['suit']}-{value}"]
        tpl = [(q["x"], q["y"]) for q in design["dots"]]
        pts, width = card_template_points(b, tpl, G, Gi, rot=rot, icon_len=icon_len(b))
        corners, _ = card_template_points(b, [(0, 0), (1, 0), (1, 1), (0, 1)], G, Gi, rot=rot, icon_len=icon_len(b))
        cards.append(dict(badge=i, deck=b["deck"], design=design["id"], side=side,
                          corners=corners.tolist(), dots=pts.tolist(), width=width, visible=0))
        for q in pts:
            expected.append((q[0], q[1], len(cards) - 1))
    if changed:
        warnings.append(f"{changed} card value(s) corrected from dot positions")

    E = np.array([(x, y) for x, y, _ in expected]) if expected else np.zeros((0, 2))
    counts = {d: 0 for d in DECKS}
    review, colour_conflicts, on_card_only = [], 0, 0
    for d in dots:
        d["explained"], d["owner"] = False, None
        if len(E):
            dist = np.hypot(E[:, 0] - d["x"], E[:, 1] - d["y"])
            j = int(np.argmin(dist))
            tol = tol_frac * cards[expected[j][2]]["width"]
            if dist[j] < tol:
                near = [expected[k][2] for k in np.nonzero(dist < tol)[0]]
                same = [c for c in near if cards[c]["deck"] == d["deck"]]
                if not same and d.get("deck_p", 0) >= 0.8:
                    # Stacked dots: the dot's colour says it belongs to a different card lying on
                    # top here whose template is slightly off. Prefer a same-colour card the dot is on.
                    same = [ci for ci, c in enumerate(cards) if c["deck"] == d["deck"]
                            and point_in_quad((d["x"], d["y"]), c["corners"], grow=0.04)]
                if not same and d.get("deck_probs") is not None:
                    # no card of the dot's colour: the card whose colour the dot's reading favours
                    # (a pink-looking dot on a red card stacked over a purple one is red's)
                    near = sorted(near, key=lambda c: -d["deck_probs"][DECKS.index(cards[c]["deck"])])
                owner = (same or near)[0]
                if not same:
                    colour_conflicts += 1
                if d["score"] >= MIN_SCORE_EXPLAINED:
                    d["explained"], d["owner"] = True, owner
                else:
                    d["explained"], d["ignored"] = True, True
        if d.get("ignored"):
            continue
        if not d["explained"]:
            # Not on a printed dot position (usually a slightly-off card pose). If it lies on a
            # detected card, count it for that card (lower confidence); otherwise send to review.
            inside = [ci for ci, c in enumerate(cards) if point_in_quad((d["x"], d["y"]), c["corners"], grow=0.04)]
            if inside and d["score"] < MIN_SCORE_ON_CARD:
                d["ignored"] = True
            elif inside:
                same = [ci for ci in inside if cards[ci]["deck"] == d["deck"]]
                owner = (same or inside)[0]
                d["owner"], d["on_card_only"] = owner, True
                on_card_only += 1
            elif d["score"] >= MIN_SCORE_REVIEW:
                review.append(d)
            else:
                d["ignored"] = True
    # A card's deck is a vote: the badge's colour reading once, plus each of its dots' colour
    # readings (all dots on a card share its colour). Then every dot counts for its card's deck.
    # With a deck icon read, the icon leads and all the card's colour readings (badge and dots) are
    # one piece of evidence: one white-balance error shifts them all alike, so their mean counts,
    # at COLOUR_WEIGHT.
    lps = []
    for ci, c in enumerate(cards):
        b = badges[c["badge"]]
        lp = np.log(np.clip(np.asarray(b.get("colour_probs") or b.get("deck_probs") or
                                       [1.0 if dk == c["deck"] else 1e-3 for dk in DECKS]), 1e-4, 1))
        n_parts = 1
        for d in dots:
            if d.get("owner") == ci and d.get("deck_probs") is not None:
                lq = np.log(np.clip(np.asarray(d["deck_probs"], dtype=float), 1e-4, 1))
                if DOT_RED_PINK_WEIGHT < 1:  # blend each of red/pink towards "red or pink"
                    g = np.log(np.clip(d["deck_probs"][I_RED] + d["deck_probs"][I_PINK], 1e-4, 1))
                    lq[[I_RED, I_PINK]] = (1 - DOT_RED_PINK_WEIGHT) * g + DOT_RED_PINK_WEIGHT * lq[[I_RED, I_PINK]]
                lp = lp + DOT_VOTE_WEIGHT * lq
                n_parts += 1
        if b.get("icon_scores") is not None:
            s = np.asarray(b["icon_scores"], dtype=float)
            lp = ICON_BETA * (s - s.max()) + COLOUR_WEIGHT * lp / n_parts
        voted = DECKS[int(np.argmax(lp))]
        if voted in ("red", "pink") and abs(lp[I_RED] - lp[I_PINK]) < 1e-6:  # tie: the badge's raw reading picks
            bp = badges[c["badge"]].get("deck_probs")
            if bp is not None and bp[I_RED] != bp[I_PINK]:
                voted = "red" if bp[I_RED] > bp[I_PINK] else "pink"
            elif c["deck"] in ("red", "pink"):
                voted = c["deck"]
        if voted != c["deck"]:
            warnings.append(f"card {c['design']} read as {c['deck']} but its dots say {voted}; using {voted}")
            c["deck"] = voted
        lps.append(lp)
    # Each deck holds each design once, and the design (badge shape + number) doesn't depend on
    # colour. When two cards of one deck share a design, the colour reading of one is wrong: give
    # that design's cards the best one-per-deck assignment over the decks in play (by the vote).
    # Skipped when a design has more cards than there are decks (then a number was misread). A card
    # only moves to a deck its badge gave some chance (synthetic boards: moves against a confident
    # badge reading were mostly wrong; the duplicate was a misread number instead).
    def badge_lp(ci, dk):
        bp = badges[cards[ci]["badge"]].get("deck_probs")
        return float(np.log(min(max(float(bp[DECKS.index(dk)]), 1e-4), 1.0))) if bp is not None else (0.0 if dk == cards[ci]["deck"] else -np.inf)

    nd = Counter(c["deck"] for c in cards)
    in_play = [dk for dk in DECKS if nd[dk] >= MIN_DECK_IN_PLAY]
    groups = {}
    for ci, c in enumerate(cards):
        if c["deck"] in in_play:
            groups.setdefault(c["design"], []).append(ci)
    for des, grp in groups.items():
        if len(grp) < 2 or len(grp) > len(in_play) or len({cards[ci]["deck"] for ci in grp}) == len(grp):
            continue
        best, best_s = None, -np.inf
        for perm in permutations(in_play, len(grp)):
            if any(dk != cards[ci]["deck"] and badge_lp(ci, cards[ci]["deck"]) - badge_lp(ci, dk) >= MOVE_MAX_GAP
                   for ci, dk in zip(grp, perm)):
                continue
            s = sum(lps[ci][DECKS.index(dk)] for ci, dk in zip(grp, perm))
            if s > best_s:
                best, best_s = perm, s
        for ci, dk in zip(grp, best or [cards[ci]["deck"] for ci in grp]):
            if dk != cards[ci]["deck"]:
                warnings.append(f"card {des} read as {cards[ci]['deck']}, but {cards[ci]['deck']} {des} is "
                                f"already on the table (each deck has one); using {dk}")
                cards[ci]["deck"] = dk
    seen = Counter((c["deck"], c["design"]) for c in cards)
    for (dk, des), n in seen.items():
        if n > 1:
            warnings.append(f"{dk} {des} detected {n} times (each deck has one)")
    for d in dots:
        if d.get("owner") is not None:
            cards[d["owner"]]["visible"] += 1
            counts[cards[d["owner"]]["deck"]] += 1
    # A "card" with no counted dots, a weak badge score and an unsure colour is almost certainly
    # not a card (clothing, clutter). Real cards with every dot covered keep a confident badge.
    weak = [ci for ci, c in enumerate(cards) if c["visible"] == 0 and
            (badges[c["badge"]]["score"] < MIN_BADGE_SCORE_NO_DOTS or badges[c["badge"]].get("deck_p", 1) < MIN_DECK_P_NO_DOTS)]
    if weak:
        warnings.append(f"ignored {len(weak)} weak badge-like detection(s) with no dots")
        for ci in weak:
            cards[ci]["ignored"] = True
            badges[cards[ci]["badge"]]["rejected"] = "weak"
    # Cards connect to the board; a card standing alone is legal (no legal move) but rare. A lone
    # "card" with a weak or unsure reading is clutter: a shoe, a shirt, a controller, box art.
    # Checked again after the few-cards rule below: box art can hold several phantom "cards" that
    # keep each other company until their colour is dropped.
    def drop_lone():
        lone = [ci for ci, c in enumerate(cards) if not c.get("ignored") and is_weak_reading(badges[c["badge"]])
                and all(quad_gap(c["corners"], o["corners"]) > ISOLATED_GAP * c["width"]
                        for oi, o in enumerate(cards) if oi != ci and not o.get("ignored"))]
        if lone:
            lost = sum(cards[ci]["visible"] for ci in lone)
            warnings.append(f"ignored {len(lone)} weak detection(s) away from the other cards ({lost} dots)")
            for ci in lone:
                c = cards[ci]
                c["ignored"] = True
                counts[c["deck"]] -= c["visible"]
                badges[c["badge"]]["rejected"] = "lone"
            for d in dots:
                if d.get("owner") is not None and d["owner"] in lone:
                    d["owner"], d["ignored"] = None, True

    drop_lone()
    # A deck in play always has several cards. A colour with only a few "cards" while other decks
    # have many is clutter (e.g. the card art on the game box): don't count it, but say so.
    live = Counter(c["deck"] for c in cards if not c.get("ignored"))
    # Red and pink look alike. When together they fit in one deck (18 cards) and one of them is
    # much smaller, they are one deck read partly wrong: count the smaller as the larger. (With
    # both decks in play there are ~18 of each.)
    small, large = sorted(LOOKALIKE_PAIR, key=lambda dk: (live.get(dk, 0), dk))
    ns, nl = live.get(small, 0), live.get(large, 0)
    if ns > 0 and nl > 3 and (ns <= 3 or (2 * ns <= nl and ns + nl <= DECK_CARDS + 1)):
        moved = sum(c["visible"] for c in cards if c["deck"] == small and not c.get("ignored"))
        warnings.append(f"{ns} {small} card(s) ({moved} dots) counted as {large}: {small} and {large} "
                        f"look alike, and together they make one deck")
        for c in cards:
            if c["deck"] == small and not c.get("ignored"):
                c["deck"] = large
        counts[small] -= moved
        counts[large] += moved
        live[large] += ns
        del live[small]
    if live and max(live.values()) >= 10:
        for dk, n in list(live.items()):
            if n < MIN_DECK_IN_PLAY:
                lost = sum(c["visible"] for c in cards if c["deck"] == dk and not c.get("ignored"))
                warnings.append(f"ignored {n} {dk} card(s) ({lost} dots): too few for a deck in play")
                for c in cards:
                    if c["deck"] == dk:
                        c["ignored"] = True
                counts[dk] -= lost
                for d in dots:  # e.g. box art: not shown for review either
                    if d.get("owner") is not None and cards[d["owner"]]["deck"] == dk:
                        d["owner"], d["ignored"] = None, True
        drop_lone()
    cards_kept = [c for c in cards if not c.get("ignored")]
    # Possible dots worth a player's look are on the board: within a card width of a found card.
    # (Off-board ones are clutter: clothing, the game box, other objects.)
    near = [d for d in review if any(quad_distance((d["x"], d["y"]), c["corners"]) <= REVIEW_REACH * c["width"]
                                     for c in cards_kept)]
    near_ids = {id(d) for d in near}
    for d in review:
        if id(d) not in near_ids:
            d["ignored"] = True
    review = near
    remap = {ci: k for k, ci in enumerate(ci for ci, c in enumerate(cards) if not c.get("ignored"))}
    if on_card_only:
        warnings.append(f"{on_card_only} dot(s) on a card but off its expected dot positions; counted (lower confidence)")
    if review:
        warnings.append(f"{len(review)} possible dot(s) not on any detected card were NOT counted - please check")
    if colour_conflicts:
        warnings.append(f"{colour_conflicts} dot(s) whose colour disagreed with their card; counted for the card")
    per_deck_cards = {dk: sum(1 for c in cards_kept if c["deck"] == dk) for dk in DECKS}
    for dk, n in per_deck_cards.items():
        if n > 18:
            warnings.append(f"{dk}: {n} cards detected (a deck has 18)")
        elif 0 < n < 18:
            warnings.append(f"{dk}: {n} cards detected (18 expected unless a card was removed)")
    # Partly covered dots: a misaligned card cuts them. Not counted (house rule); shown if on a card.
    partial = [d for d in partial if any(point_in_quad((d["x"], d["y"]), c["corners"], grow=0.04) for c in cards_kept)]
    if partial:
        warnings.append(f"{len(partial)} partly covered dot(s) (a card edge crosses them) were NOT counted")
    for d in dots:  # owner indices refer to the returned (kept) card list
        if d.get("owner") is not None:
            d["owner"] = remap.get(d["owner"])
    return dict(perspective_fit=G is not None,
                counts={k: v for k, v in counts.items() if v or per_deck_cards[k]},
                cards_per_deck={k: v for k, v in per_deck_cards.items() if v},
                cards=cards_kept, review=review, partial=partial, warnings=warnings, scale_factor=best_k)
