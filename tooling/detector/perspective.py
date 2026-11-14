"""Whole-board perspective from badge->icon vectors.

All cards lie on one plane, and every card's badge->icon vector has the same physical length
(per badge shape). Find the projective + affine correction G (image -> rectified plane, up to a
similarity) under which all those lengths agree. Four parameters: the vanishing line (p1, p2)
and an affine shear/aspect (sh, log k). Solved with Nelder-Mead (easy to port to TypeScript).
"""
import numpy as np

P_LIMIT = 2.0   # bound on the vanishing-line parameters (normalized by the image diagonal)
# Prior towards no shear / aspect change (weight on sh^2 + logk^2, added to the cost). When the
# badge->icon vectors point in only one or two directions (all cards square to each other, e.g. a
# layout photo), shear and aspect don't change their relative lengths, so the data can't pin them
# down and an unregularised fit wanders off to a large shear.
SHEAR_PRIOR = 0.003
STARTS = ([0, 0, 0, 0], [0.3, 0, 0, 0], [-0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, -0.3, 0, 0])


def make_G(params, cx, cy, f):
    p1, p2, sh, lk = params
    N = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])   # normalize image coords
    P = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])
    A = np.array([[1, sh, 0], [0, np.exp(lk), 0], [0, 0, 1]])
    return A @ P @ N


def apply(G, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[pts, np.ones(len(pts))] @ G.T
    return q[:, :2] / q[:, 2:3]


def nelder_mead(fun, x0, step=0.05, iters=600, tol=1e-10):
    n = len(x0)
    simplex = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [fun(x) for x in simplex]
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(simplex[:-1], axis=0)
        xr = c + (c - simplex[-1]); fr = fun(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - simplex[-1]); fe = fun(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (simplex[-1] - c); fc = fun(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex[1:]]
                vals = [vals[0]] + [fun(s) for s in simplex[1:]]
    return simplex[int(np.argmin(vals))]


def fit(badges, icons, unit_lengths, image_size):
    """badges, icons: Nx2 image px; unit_lengths: physical badge->icon length per card (card widths).
    Returns G (3x3, image -> rectified plane) or None if too few cards."""
    badges, icons = np.asarray(badges, float), np.asarray(icons, float)
    if len(badges) < 6:
        return None
    w, h = image_size
    cx, cy, f = w / 2, h / 2, float(np.hypot(w, h))
    unit = np.asarray(unit_lengths, float)

    def cost(params):
        if abs(params[0]) > P_LIMIT or abs(params[1]) > P_LIMIT or abs(params[2]) > 1 or abs(params[3]) > 1:
            return 1e6
        G = make_G(params, cx, cy, f)
        L = np.linalg.norm(apply(G, icons) - apply(G, badges), axis=1) / unit
        if np.any(~np.isfinite(L)) or np.any(L <= 0):
            return 1e6
        r = np.log(L) - np.median(np.log(L))
        return float(np.mean(np.minimum(r ** 2, 0.04))   # robust (caps outliers)
                     + SHEAR_PRIOR * (params[2] ** 2 + params[3] ** 2))

    best = None
    for start in STARTS:
        x = nelder_mead(cost, start)
        c = cost(x)
        if best is None or c < best[0]:
            best = (c, x)
    return make_G(best[1], cx, cy, f)
