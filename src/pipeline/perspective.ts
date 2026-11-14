// Whole-board perspective from badge->icon vectors. Port of tooling/detector/perspective.py.
//
// All cards lie on one plane and every card's badge->icon vector has the same physical length
// (per badge shape). Find the projective + affine correction G (image -> rectified plane, up to a
// similarity) under which all those lengths agree: vanishing line (p1, p2), shear, log aspect.

export type Mat3 = number[]; // row-major 3x3
export type XY = [number, number];

export function mul3(a: Mat3, b: Mat3): Mat3 {
  const r = new Array<number>(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return r;
}

export function inv3(m: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

export function applyPt(G: Mat3, x: number, y: number): XY {
  const w = G[6] * x + G[7] * y + G[8];
  return [(G[0] * x + G[1] * y + G[2]) / w, (G[3] * x + G[4] * y + G[5]) / w];
}

export function makeG(params: number[], cx: number, cy: number, f: number): Mat3 {
  const [p1, p2, sh, lk] = params;
  const N = [1 / f, 0, -cx / f, 0, 1 / f, -cy / f, 0, 0, 1];
  const P = [1, 0, 0, 0, 1, 0, p1, p2, 1];
  const A = [1, sh, 0, 0, Math.exp(lk), 0, 0, 0, 1];
  return mul3(mul3(A, P), N);
}

export function nelderMead(fun: (x: number[]) => number, x0: number[], step = 0.05, iters = 600, tol = 1e-10): number[] {
  const n = x0.length;
  let simplex = [x0.slice()];
  for (let i = 0; i < n; i++) simplex.push(x0.map((v, j) => v + (i === j ? step : 0)));
  let vals = simplex.map(fun);
  const comb = (a: number[], b: number[], t: number) => a.map((v, j) => v + t * (b[j] - v)); // a + t*(b-a)
  for (let it = 0; it < iters; it++) {
    const order = vals.map((_, i) => i).sort((i, j) => vals[i] - vals[j]);
    simplex = order.map((i) => simplex[i]);
    vals = order.map((i) => vals[i]);
    if (Math.abs(vals[n] - vals[0]) < tol) break;
    const c = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += simplex[i][j];
    for (let j = 0; j < n; j++) c[j] /= n;
    const worst = simplex[n];
    const xr = comb(c, worst, -1), fr = fun(xr);
    if (fr < vals[0]) {
      const xe = comb(c, worst, -2), fe = fun(xe);
      if (fe < fr) { simplex[n] = xe; vals[n] = fe; } else { simplex[n] = xr; vals[n] = fr; }
    } else if (fr < vals[n - 1]) {
      simplex[n] = xr; vals[n] = fr;
    } else {
      const xc = comb(c, worst, 0.5), fc = fun(xc);
      if (fc < vals[n]) { simplex[n] = xc; vals[n] = fc; } else {
        simplex = [simplex[0], ...simplex.slice(1).map((s) => comb(simplex[0], s, 0.5))];
        vals = [vals[0], ...simplex.slice(1).map(fun)];
      }
    }
  }
  let best = 0;
  for (let i = 1; i < vals.length; i++) if (vals[i] < vals[best]) best = i;
  return simplex[best];
}

export function median(v: number[]): number {
  const s = v.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Prior towards no shear / aspect change (weight on sh^2 + logk^2). When all cards are square to
// each other (e.g. a layout photo) shear and aspect don't change the relative badge->icon lengths,
// so the data can't pin them down and an unregularised fit wanders off to a large shear.
const SHEAR_PRIOR = 0.003;

/** badges, icons: image px; unitLengths: physical badge->icon length per card (card widths).
 *  Returns G (image -> rectified plane) or null if too few cards. */
export function fitPerspective(badges: XY[], icons: XY[], unitLengths: number[], imageSize: [number, number]): Mat3 | null {
  if (badges.length < 6) return null;
  const [w, h] = imageSize;
  const cx = w / 2, cy = h / 2, f = Math.hypot(w, h);
  const logL = new Array<number>(badges.length);
  const cost = (p: number[]) => {
    if (Math.abs(p[0]) > 2 || Math.abs(p[1]) > 2 || Math.abs(p[2]) > 1 || Math.abs(p[3]) > 1) return 1e6;
    const G = makeG(p, cx, cy, f);
    for (let i = 0; i < badges.length; i++) {
      const a = applyPt(G, badges[i][0], badges[i][1]), b = applyPt(G, icons[i][0], icons[i][1]);
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]) / unitLengths[i];
      if (!Number.isFinite(L) || L <= 0) return 1e6;
      logL[i] = Math.log(L);
    }
    const med = median(logL);
    let s = 0;
    for (const l of logL) s += Math.min((l - med) ** 2, 0.04); // robust (caps outliers)
    return s / logL.length + SHEAR_PRIOR * (p[2] * p[2] + p[3] * p[3]);
  };
  let best: [number, number[]] | null = null;
  for (const start of [[0, 0, 0, 0], [0.3, 0, 0, 0], [-0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, -0.3, 0, 0]]) {
    const x = nelderMead(cost, start);
    const c = cost(x);
    if (best === null || c < best[0]) best = [c, x];
  }
  return makeG(best![1], cx, cy, f);
}
