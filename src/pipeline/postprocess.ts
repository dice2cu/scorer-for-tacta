// Classical post-processing of detector output: card poses from badges, template dot positions,
// consistency checks, and final per-deck scores. Port of tooling/detector/postprocess.py
// (keep the two in step; tests/postprocess.test.mjs compares them on the same inputs).
//
// Card pose from a badge: the icon sits a fixed distance "below" the badge in card coordinates, so
// the badge->icon vector gives both the card's down direction and its local width in pixels.
import type { Badge, Deck, Dot } from "./decode.ts";
import type { Library, Suit } from "./library.ts";
import { iconDistance } from "./library.ts";
import type { Mat3, XY } from "./perspective.ts";
import { applyPt, fitPerspective, inv3, median } from "./perspective.ts";

export const DECKS: Deck[] = ["blue", "green", "orange", "pink", "purple", "red"];

export const EXPLAIN_TOL = 0.06; // card widths, with the whole-board perspective fit
export const EXPLAIN_TOL_NO_FIT = 0.15; // card widths, per-card poses only
export const SIZE_TOL = 2.0; // rejects objects < half or > double the board's card size (box art)
export const SIZE_REF_SCORE = 0.6; // badges this confident set the table's card size (real ~0.7-0.95, junk mostly < 0.6)
const ROT_SEARCH = Array.from({ length: 21 }, (_, i) => ((-20 + 2 * i) * Math.PI) / 180);
const VALUE_PRIOR = 2.5; // weight of log p_model(value); a value change needs clearly more matching dots
// global card-size factor k searched over K_MIN, K_MIN + K_STEP, ... (Python: np.arange(0.75, 1.3001, 0.025))
const K_MIN = 0.75, K_STEP = 0.025, K_STEPS = 23;
const K_MAX = K_MIN + (K_STEPS - 1) * K_STEP;
// Minimum detector confidence for a dot to count, by where it sits: on a printed dot position,
// merely on a card, or off every card (never counted; confident ones are shown for review).
export const MIN_SCORE_EXPLAINED = 0.2;
export const MIN_SCORE_ON_CARD = 0.5;
export const MIN_SCORE_REVIEW = 0.5;
const MIN_BADGE_SCORE_NO_DOTS = 0.55;
const MIN_DECK_P_NO_DOTS = 0.6;
// A card more than ISOLATED_GAP card widths from every other card needs a believable reading.
// Clutter seen (photos + synthetic): badge score 0.34-0.46, or score 0.6 with both value and
// colour unsure (0.43 / 0.71). Real lone cards (synthetic, hard light): score >= 0.52, and never
// both probabilities low. Layout photos have gaps up to ~0.16 card widths.
const ISOLATED_GAP = 0.4;
// Red vs pink: weight of the dots' say between these two look-alike colours (their say on
// red-or-pink vs other colours is unchanged). In real photos the badge tells them apart well but
// a dot (white disc, thin colour rim) does not: at full weight, dots outvoted confident red badges
// (red/purple game: 13 red dots counted pink). At 0.25 that game is exact; synthetic boards with
// both decks lose ~1% of exact boards (their dots are more reliable than real ones). 0 = badge only.
const DOT_RED_PINK_WEIGHT = 0.25;
const I_RED = 5, I_PINK = 3; // indices in DECKS
const LOOKALIKE_PAIR: [Deck, Deck] = ["red", "pink"];
const DECK_CARDS = 18;
const MOVE_MAX_GAP = 6.0; // log-odds: the badge's reading of a card's new deck vs its current one
const MIN_DECK_IN_PLAY = 4; // fewer cards of a colour are clutter or misreadings, not a deck in play
// Deck icons (the shape under the badge, read classically from the photo: icons.ts) are the primary
// deck evidence; the detector's colour reading of the badge is secondary (white balance and light
// shift colours, red vs pink especially). A card's deck log-odds = ICON_BETA * (icon score - best
// icon score) + COLOUR_WEIGHT * log(colour probability). Real photos: correct icon reads have a
// score margin of ~0.44 (median), 1% below 0.09; the one wrong read had 0.05.
const ICON_BETA = 15.0;
const COLOUR_WEIGHT = 0.3;
const MIN_BADGE_SCORE_ISOLATED = 0.5;
const UNSURE_VALUE_P = 0.6;
const UNSURE_DECK_P = 0.8;

export interface Card {
  badge: number; // index into the badges list
  deck: Deck;
  design: string;
  side: "front" | "back";
  corners: XY[];
  dots: XY[]; // printed dot positions (covered or not)
  width: number; // px
  visible: number; // counted dots
  ignored?: boolean;
}

export interface ScoreResult {
  perspective_fit: boolean;
  counts: Partial<Record<Deck, number>>;
  cards_per_deck: Partial<Record<Deck, number>>;
  cards: Card[];
  review: Dot[];
  partial: Dot[]; // partly covered dots on found cards (not counted)
  warnings: string[];
  scale_factor: number;
}

interface Geom {
  x: number;
  y: number;
  icon: [number, number];
  suit: Suit;
  side: "front" | "back";
}

interface Pose {
  b: XY;
  u: XY;
  v: XY;
  width: number;
}

/** minLen: shortest usable badge->icon vector (image px; ~0 in rectified units). */
function cardPose(lib: Library, g: Geom, minLen = 1e-3): Pose | null {
  const dx = g.icon[0] - g.x, dy = g.icon[1] - g.y;
  const L = Math.hypot(dx, dy);
  if (L < minLen || L < 1e-12) return null;
  const v: XY = [dx / L, dy / L];
  return { b: [g.x, g.y], u: [v[1], -v[0]], v, width: L / iconDistance(lib, g.suit) };
}

function toImage(lib: Library, pose: Pose, g: Geom, xs: number[], ys: number[]): XY[] {
  const bl = lib.badgeBySuit.get(g.suit)!;
  const back = g.side === "back";
  const bx = back ? 1 - bl.x : bl.x;
  return xs.map((x0, i) => {
    const xn = back ? 1 - x0 : x0;
    const dx = (xn - bx) * pose.width;
    const dy = ((ys[i] - bl.y) / lib.aspect) * pose.width;
    return [pose.b[0] + dx * pose.u[0] + dy * pose.v[0], pose.b[1] + dx * pose.u[1] + dy * pose.v[1]];
  });
}

interface Ctx {
  lib: Library;
  G: Mat3 | null;
  Gi: Mat3 | null;
  tol: number;
  dx: Float64Array;
  dy: Float64Array;
}

/** Image positions of normalized card points; through the rectified plane when G is known.
 *  rot: extra rotation of the badge->icon direction; iconLen: override the badge->icon length
 *  (rectified units when G is given, px otherwise). */
function templatePoints(ctx: Ctx, g: Geom, xs: number[], ys: number[], rot = 0, iconLen: number | null = null): { pts: XY[]; width: number } {
  const adjust = (bx: number, by: number, ix: number, iy: number): [number, number] => {
    let vx = ix - bx, vy = iy - by;
    const L = Math.hypot(vx, vy);
    if (L < 1e-9) return [ix, iy];
    const c = Math.cos(rot), s = Math.sin(rot);
    [vx, vy] = [c * vx - s * vy, s * vx + c * vy];
    if (iconLen !== null) { vx = (vx / L) * iconLen; vy = (vy / L) * iconLen; }
    return [bx + vx, by + vy];
  };
  const { lib, G, Gi } = ctx;
  if (G === null || Gi === null) {
    const bb = { ...g, icon: adjust(g.x, g.y, g.icon[0], g.icon[1]) };
    const pose = cardPose(lib, bb)!;
    return { pts: toImage(lib, pose, bb, xs, ys), width: pose.width };
  }
  const br = applyPt(G, g.x, g.y), ir = applyPt(G, g.icon[0], g.icon[1]);
  const rb = { ...g, x: br[0], y: br[1], icon: adjust(br[0], br[1], ir[0], ir[1]) };
  const pts = toImage(lib, cardPose(lib, rb, 0)!, rb, xs, ys).map((p) => applyPt(Gi, p[0], p[1]));
  let width = cardPose(lib, g)!.width;
  if (iconLen !== null) width *= iconLen / Math.max(Math.hypot(ir[0] - br[0], ir[1] - br[1]), 1e-9);
  return { pts, width };
}

/** Soft alignment score: per template point, the best Gaussian match among the candidate dots. */
function softScore(ctx: Ctx, pts: XY[], width: number, cand: number[], weights: Float64Array | null): number {
  if (!cand.length || !pts.length) return 0;
  const sigma = 0.5 * ctx.tol * width;
  const k = -0.5 / (sigma * sigma);
  let total = 0;
  for (const [px, py] of pts) {
    let best = 0;
    for (const j of cand) {
      const ddx = ctx.dx[j] - px, ddy = ctx.dy[j] - py;
      let gv = Math.exp(k * (ddx * ddx + ddy * ddy));
      if (weights) gv *= weights[j];
      if (gv > best) best = gv;
    }
    total += best;
  }
  return total;
}

/** Per detected dot: how well this template explains it (0..1). */
function templateGauss(ctx: Ctx, pts: XY[], width: number, cand: number[], out: Float64Array) {
  const sigma = 0.5 * ctx.tol * width;
  const k = -0.5 / (sigma * sigma);
  for (const j of cand) {
    let best = 0;
    for (const [px, py] of pts) {
      const ddx = ctx.dx[j] - px, ddy = ctx.dy[j] - py;
      best = Math.max(best, Math.exp(k * (ddx * ddx + ddy * ddy)));
    }
    out[j] = best;
  }
}

function designPoints(lib: Library, suit: Suit, value: number): { xs: number[]; ys: number[] } {
  const d = lib.designs.get(`${suit}-${value}`)!;
  return { xs: d.dots.map((p) => p.x), ys: d.dots.map((p) => p.y) };
}

type Option = [score: number, side: "front" | "back", rot: number];

function fitCardOptions(ctx: Ctx, b: Badge, cand: number[], iconLen: number, weights: Float64Array | null): Option[] {
  const options: Option[] = [];
  for (let value = 1; value <= 6; value++) {
    const prior = VALUE_PRIOR * Math.log(Math.max(b.value_probs[value - 1], 1e-3));
    const { xs, ys } = designPoints(ctx.lib, b.suit as Suit, value);
    let best: Option | null = null;
    for (const side of ["front", "back"] as const) {
      const g: Geom = { x: b.x, y: b.y, icon: b.icon, suit: b.suit as Suit, side };
      for (const rot of ROT_SEARCH) {
        const { pts, width } = templatePoints(ctx, g, xs, ys, rot, iconLen);
        const soft = softScore(ctx, pts, width, cand, weights);
        // tie-breaks: the model's side, no rotation
        const sc = soft + prior - 0.01 * (side !== b.side ? 1 : 0) - 0.002 * Math.abs(rot);
        if (best === null || sc > best[0]) best = [sc, side, rot];
      }
    }
    options.push(best!);
  }
  return options; // index value-1
}

/** r-permutations of 0..n-1, in the order of Python's itertools.permutations. */
function* permutations(n: number, r: number): Generator<number[]> {
  const used = new Array<boolean>(n).fill(false);
  const cur: number[] = [];
  function* rec(): Generator<number[]> {
    if (cur.length === r) { yield cur.slice(); return; }
    for (let v = 0; v < n; v++) {
      if (used[v]) continue;
      used[v] = true; cur.push(v);
      yield* rec();
      cur.pop(); used[v] = false;
    }
  }
  yield* rec();
}

/** Each deck has exactly one card per design: within each (deck, badge shape) group pick the
 *  value assignment with the best total score (all permutations; groups have <= 6 cards). */
function assignValues(real: Badge[], options: Map<Badge, Option[]>): Map<Badge, [number, "front" | "back", number]> {
  const fits = new Map<Badge, [number, "front" | "back", number]>();
  const groups = new Map<string, Badge[]>();
  for (const b of real) {
    const key = `${b.deck}|${b.suit}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(b);
  }
  for (const grp of groups.values()) {
    if (grp.length > 6) { // more cards than designs (a deck colour misread): no uniqueness possible
      for (const b of grp) {
        const o = options.get(b)!;
        let v = 0;
        for (let i = 1; i < 6; i++) if (o[i][0] > o[v][0]) v = i;
        fits.set(b, [v + 1, o[v][1], o[v][2]]);
      }
      continue;
    }
    let bestTotal: number | null = null, bestPerm: number[] = [];
    const opts = grp.map((b) => options.get(b)!);
    for (const perm of permutations(6, grp.length)) {
      let tot = 0;
      for (let i = 0; i < perm.length; i++) tot += opts[i][perm[i]][0];
      if (bestTotal === null || tot > bestTotal) { bestTotal = tot; bestPerm = perm; }
    }
    grp.forEach((b, i) => {
      const o = opts[i][bestPerm[i]];
      fits.set(b, [bestPerm[i] + 1, o[1], o[2]]);
    });
  }
  return fits;
}

/** Point inside a convex quad, optionally grown about its centre by a fraction. */
export function pointInQuad(p: XY, quad: XY[], grow = 0): boolean {
  const cx = (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0]) / 4;
  const cy = (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1]) / 4;
  const q = quad.map(([x, y]) => [cx + (x - cx) * (1 + grow), cy + (y - cy) * (1 + grow)]);
  let pos = true, neg = true;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4];
    const v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (v < 0) pos = false;
    if (v > 0) neg = false;
  }
  return pos || neg;
}

/** Distance from a point to a convex quad (0 inside). */
export function quadDistance(p: XY, quad: XY[]): number {
  if (pointInQuad(p, quad)) return 0;
  let best = Infinity;
  for (let i = 0; i < 4; i++) {
    const a = quad[i], b = quad[(i + 1) % 4];
    const abx = b[0] - a[0], aby = b[1] - a[1], apx = p[0] - a[0], apy = p[1] - a[1];
    const t = Math.min(1, Math.max(0, (apx * abx + apy * aby) / Math.max(abx * abx + aby * aby, 1e-12)));
    best = Math.min(best, Math.hypot(apx - t * abx, apy - t * aby));
  }
  return best;
}

/** 0 if two quads overlap, else the smallest corner-to-quad distance either way. */
function quadGap(a: XY[], b: XY[]): number {
  if (a.some((p) => pointInQuad(p, b)) || b.some((p) => pointInQuad(p, a))) return 0;
  return Math.min(...a.map((p) => quadDistance(p, b)), ...b.map((p) => quadDistance(p, a)));
}

function isWeakReading(b: Badge): boolean {
  return b.score < MIN_BADGE_SCORE_ISOLATED || ((b.value_p ?? 1) < UNSURE_VALUE_P && (b.deck_p ?? 1) < UNSURE_DECK_P);
}

const REVIEW_REACH = 1.0; // card widths from the nearest found card

function geom(b: Badge): Geom {
  return { x: b.x, y: b.y, icon: b.icon, suit: b.suit as Suit, side: b.side };
}

function rectifiedSizes(lib: Library, G: Mat3, cards: Badge[]): number[] {
  return cards.map((b) => {
    const br = applyPt(G, b.x, b.y), ir = applyPt(G, b.icon[0], b.icon[1]);
    return Math.hypot(ir[0] - br[0], ir[1] - br[1]) / iconDistance(lib, b.suit as Suit);
  });
}

function fitBoard(lib: Library, cards: Badge[], imageSize: [number, number]): Mat3 | null {
  return fitPerspective(cards.map((b) => [b.x, b.y]), cards.map((b) => b.icon), cards.map((b) => iconDistance(lib, b.suit as Suit)), imageSize);
}

/** Final result: per-deck visible dot counts + per-card details + warnings.
 *
 *  Policy: a dot is counted only if it sits on a printed dot position of a detected card; its owner
 *  is that card (when several cards have a printed dot there, i.e. stacked dots, the one whose deck
 *  matches the dot's colour). Dots matching no card are returned for review, not counted.
 *  Partly covered dots (decoded with partial = true) are never counted or used for fitting; those
 *  on a found card are returned in `partial` so the app can show them. */
/** Combine a badge's icon reading (icon_scores) with its colour reading (deck_probs, kept as
 *  colour_probs): deck_probs / deck / deck_p become the combined belief. */
export function fuseIcon(b: Badge): void {
  const s = b.icon_scores;
  if (s == null || b.colour_probs !== undefined) return;
  const top = Math.max(...s);
  const lp = b.deck_probs.map((p, i) => COLOUR_WEIGHT * Math.log(Math.min(Math.max(p, 1e-4), 1)) + ICON_BETA * (s[i] - top));
  const m = Math.max(...lp);
  const e = lp.map((a) => Math.exp(a - m));
  const z = e.reduce((a, v) => a + v, 0);
  b.colour_probs = b.deck_probs;
  b.deck_probs = e.map((v) => v / z);
  let i = 0;
  b.deck_probs.forEach((p, k) => { if (p > b.deck_probs[i]) i = k; });
  b.deck = DECKS[i];
  b.deck_p = b.deck_probs[i];
}

export function score(lib: Library, badges: Badge[], allDots: Dot[], imageSize: [number, number] | null = null): ScoreResult {
  const warnings: string[] = [];
  for (const b of badges) fuseIcon(b);
  let partial = allDots.filter((d) => d.partial);
  const dots = allDots.filter((d) => !d.partial);
  const starts = badges.filter((b) => b.suit === "start");
  if (starts.length) {
    let top = starts[0];
    for (const b of starts) if (b.score > top.score) top = b;
    for (const b of starts) if (b !== top) b.rejected = "extra start card";
  }
  let real = badges.filter((b) => b.suit !== "start" && cardPose(lib, geom(b)) !== null);

  // whole-board perspective, then reject "cards" of the wrong physical size (box art, clutter)
  let G: Mat3 | null = null, Gi: Mat3 | null = null;
  if (imageSize !== null && real.length >= 6) {
    // The table's card size comes from confident badges when there are enough: busy
    // surroundings (carpet, clothes, box art) can yield more weak junk badges than real cards.
    const conf = real.filter((b) => b.score >= SIZE_REF_SCORE);
    const ref = conf.length >= 6 ? conf : real;
    G = fitBoard(lib, ref, imageSize);
    if (G !== null) {
      const sz = rectifiedSizes(lib, G, real);
      const med = median(rectifiedSizes(lib, G, ref));
      const bad = sz.map((s) => Math.abs(Math.log(s / med)) > Math.log(SIZE_TOL));
      real.forEach((b, i) => (b.rejected = bad[i] ? "size" : null));
      const nBad = bad.filter(Boolean).length;
      if (nBad) {
        warnings.push(`ignored ${nBad} badge(s) whose size doesn't fit the table (e.g. game box art or other objects)`);
        real = real.filter((_, i) => !bad[i]);
        G = real.length >= 6 ? fitBoard(lib, real, imageSize) : null;
      }
      Gi = G !== null ? inv3(G) : null;
    }
  }
  const tol = G !== null ? EXPLAIN_TOL : EXPLAIN_TOL_NO_FIT;
  const ctx: Ctx = { lib, G, Gi, tol, dx: Float64Array.from(dots, (d) => d.x), dy: Float64Array.from(dots, (d) => d.y) };

  // Card size: every card has the same physical size, so use one board-wide badge->icon length
  // (rectified units if the perspective fit exists, else px), times a global factor k chosen so
  // that the most detected dots land on template dots (the model's icon lengths can be biased).
  let base = 1;
  if (real.length) {
    base = G !== null ? median(rectifiedSizes(lib, G, real))
      : median(real.map((b) => Math.hypot(b.icon[0] - b.x, b.icon[1] - b.y) / iconDistance(lib, b.suit as Suit)));
  }

  // Candidate dots per card: everything within a generous radius of its badge, sized by the larger
  // of the card's own size and its biggest possible template (board-wide size at the largest k,
  // which can be over twice a badge's own size estimate). Template points never reach that far,
  // so dots outside contribute nothing to any score (as in Python, which scores all dots).
  const cand = new Map<Badge, number[]>();
  for (const b of real) {
    const own = cardPose(lib, geom(b))!.width;
    const tpl = templatePoints(ctx, geom(b), [], [], 0, K_MAX * base * iconDistance(lib, b.suit as Suit)).width;
    const R = 3 * Math.max(own, tpl);
    const list: number[] = [];
    dots.forEach((d, j) => { if (Math.hypot(d.x - b.x, d.y - b.y) < R) list.push(j); });
    cand.set(b, list);
  }

  let bestK = 1, bestHits = -1;
  for (let i = 0; i < K_STEPS; i++) {
    const k = K_MIN + i * K_STEP;
    let tot = 0;
    for (const b of real) {
      const { xs, ys } = designPoints(lib, b.suit as Suit, b.value);
      const { pts, width } = templatePoints(ctx, geom(b), xs, ys, 0, k * base * iconDistance(lib, b.suit as Suit));
      tot += softScore(ctx, pts, width, cand.get(b)!, null);
    }
    if (tot > bestHits) { bestK = k; bestHits = tot; }
  }
  const iconLen = (b: Badge) => bestK * base * iconDistance(lib, b.suit as Suit);

  // Fit value/side/rotation per card. Pass 1: independently. Pass 2: dots that OTHER cards'
  // templates already explain count much less, so a card can't win by borrowing neighbours' dots.
  const modelValue = new Map(real.map((b) => [b, b.value]));
  const opts = new Map(real.map((b) => [b, fitCardOptions(ctx, b, cand.get(b)!, iconLen(b), null)]));
  let fits = assignValues(real, opts);

  if (dots.length && real.length > 1) {
    const maps = real.map((b) => {
      const [value, side, rot] = fits.get(b)!;
      const { xs, ys } = designPoints(lib, b.suit as Suit, value);
      const { pts, width } = templatePoints(ctx, { ...geom(b), side }, xs, ys, rot, iconLen(b));
      const m = new Float64Array(dots.length);
      templateGauss(ctx, pts, width, cand.get(b)!, m);
      return m;
    });
    // best and second-best explanation per dot, so "max over the other cards" is O(1)
    const best1 = new Float64Array(dots.length), best2 = new Float64Array(dots.length), who = new Int32Array(dots.length).fill(-1);
    maps.forEach((m, i) => {
      for (let j = 0; j < dots.length; j++) {
        if (m[j] > best1[j]) { best2[j] = best1[j]; best1[j] = m[j]; who[j] = i; }
        else if (m[j] > best2[j]) best2[j] = m[j];
      }
    });
    real.forEach((b, i) => {
      const w = new Float64Array(dots.length);
      for (let j = 0; j < dots.length; j++) w[j] = 1 - 0.8 * (who[j] === i ? best2[j] : best1[j]);
      opts.set(b, fitCardOptions(ctx, b, cand.get(b)!, iconLen(b), w));
    });
    fits = assignValues(real, opts);
  }

  const cards: Card[] = [];
  const expected: [number, number, number][] = []; // x, y, card index
  const realSet = new Set(real);
  let changed = 0;
  badges.forEach((b, i) => {
    if (!realSet.has(b)) return;
    const [value, side, rot] = fits.get(b)!;
    if (value !== modelValue.get(b)) changed++;
    b.value = value; b.side = side; b.rot_fix = rot;
    const design = lib.designs.get(`${b.suit}-${value}`)!;
    const { xs, ys } = designPoints(lib, b.suit as Suit, value);
    const { pts, width } = templatePoints(ctx, geom(b), xs, ys, rot, iconLen(b));
    const corners = templatePoints(ctx, geom(b), [0, 1, 1, 0], [0, 0, 1, 1], rot, iconLen(b)).pts;
    cards.push({ badge: i, deck: b.deck, design: design.id, side, corners, dots: pts, width, visible: 0 });
    for (const q of pts) expected.push([q[0], q[1], cards.length - 1]);
  });
  if (changed) warnings.push(`${changed} card value(s) corrected from dot positions`);

  const counts = Object.fromEntries(DECKS.map((d) => [d, 0])) as Record<Deck, number>;
  const review: Dot[] = [];
  let colourConflicts = 0, onCardOnly = 0;
  for (const d of dots) {
    d.explained = false; d.owner = null;
    if (expected.length) {
      let j = 0, dj = Infinity;
      const dist = expected.map(([x, y]) => Math.hypot(x - d.x, y - d.y));
      dist.forEach((v, k) => { if (v < dj) { dj = v; j = k; } });
      const t = tol * cards[expected[j][2]].width;
      if (dj < t) {
        const near = expected.filter((_, k) => dist[k] < t).map((e) => e[2]);
        let same = near.filter((c) => cards[c].deck === d.deck);
        if (!same.length && (d.deck_p ?? 0) >= 0.8) {
          // Stacked dots: the dot's colour says it belongs to a different card lying on top here
          // whose template is slightly off. Prefer a same-colour card the dot is on.
          same = cards.flatMap((c, ci) => (c.deck === d.deck && pointInQuad([d.x, d.y], c.corners, 0.04) ? [ci] : []));
        }
        // no card of the dot's colour: the card whose colour the dot's reading favours (a
        // pink-looking dot on a red card stacked over a purple one is red's); stable sort as Python
        const ranked = same.length ? same : [...near].sort((a, b) =>
          d.deck_probs[DECKS.indexOf(cards[b].deck)] - d.deck_probs[DECKS.indexOf(cards[a].deck)]);
        const owner = ranked[0];
        if (!same.length) colourConflicts++;
        if (d.score >= MIN_SCORE_EXPLAINED) { d.explained = true; d.owner = owner; }
        else { d.explained = true; d.ignored = true; }
      }
    }
    if (d.ignored) continue;
    if (!d.explained) {
      // Not on a printed dot position (usually a slightly-off card pose). If it lies on a
      // detected card, count it for that card (lower confidence); otherwise send to review.
      const inside = cards.flatMap((c, ci) => (pointInQuad([d.x, d.y], c.corners, 0.04) ? [ci] : []));
      if (inside.length && d.score < MIN_SCORE_ON_CARD) d.ignored = true;
      else if (inside.length) {
        const same = inside.filter((ci) => cards[ci].deck === d.deck);
        d.owner = (same.length ? same : inside)[0];
        d.on_card_only = true;
        onCardOnly++;
      } else if (d.score >= MIN_SCORE_REVIEW) review.push(d);
      else d.ignored = true;
    }
  }
  // A card's deck is a vote: the badge's colour reading once, plus each of its dots' colour
  // readings (all dots on a card share its colour). Then every dot counts for its card's deck.
  // With a deck icon read, the icon leads and all the card's colour readings (badge and dots) are
  // one piece of evidence: one white-balance error shifts them all alike, so their mean counts,
  // at COLOUR_WEIGHT.
  const clipLog = (p: number) => Math.log(Math.min(Math.max(p, 1e-4), 1));
  const lps: number[][] = [];
  cards.forEach((c, ci) => {
    const b = badges[c.badge];
    let lp = (b.colour_probs ?? b.deck_probs).map(clipLog);
    let parts = 1;
    for (const d of dots) {
      if (d.owner !== ci || !d.deck_probs) continue;
      const lq = d.deck_probs.map(clipLog);
      if (DOT_RED_PINK_WEIGHT < 1) { // blend each of red/pink towards "red or pink"
        const g = clipLog(d.deck_probs[I_RED] + d.deck_probs[I_PINK]);
        for (const i of [I_RED, I_PINK]) lq[i] = (1 - DOT_RED_PINK_WEIGHT) * g + DOT_RED_PINK_WEIGHT * lq[i];
      }
      lq.forEach((v, k) => (lp[k] += v));
      parts++;
    }
    const s = b.icon_scores;
    if (s != null) {
      const top = Math.max(...s);
      lp = lp.map((v, k) => ICON_BETA * (s[k] - top) + COLOUR_WEIGHT * v / parts);
    }
    let k = 0;
    for (let i = 1; i < lp.length; i++) if (lp[i] > lp[k]) k = i;
    let voted = DECKS[k];
    if ((voted === "red" || voted === "pink") && Math.abs(lp[I_RED] - lp[I_PINK]) < 1e-6) { // tie: the badge's raw reading picks
      const bp = badges[c.badge].deck_probs;
      if (bp[I_RED] !== bp[I_PINK]) voted = bp[I_RED] > bp[I_PINK] ? "red" : "pink";
      else if (c.deck === "red" || c.deck === "pink") voted = c.deck;
    }
    if (voted !== c.deck) {
      warnings.push(`card ${c.design} read as ${c.deck} but its dots say ${voted}; using ${voted}`);
      c.deck = voted;
    }
    lps.push(lp);
  });
  // Each deck holds each design once, and the design (badge shape + number) doesn't depend on
  // colour. When two cards of one deck share a design, the colour reading of one is wrong: give
  // that design's cards the best one-per-deck assignment over the decks in play (by the vote).
  // Skipped when a design has more cards than there are decks (then a number was misread). A card
  // only moves to a deck its badge gave some chance (synthetic boards: moves against a confident
  // badge reading were mostly wrong; the duplicate was a misread number instead).
  {
    const badgeLp = (ci: number, dk: Deck) => clipLog(badges[cards[ci].badge].deck_probs[DECKS.indexOf(dk)]);
    const nd = new Map<Deck, number>();
    for (const c of cards) nd.set(c.deck, (nd.get(c.deck) ?? 0) + 1);
    const inPlay = DECKS.filter((dk) => (nd.get(dk) ?? 0) >= MIN_DECK_IN_PLAY);
    const groups = new Map<string, number[]>();
    cards.forEach((c, ci) => {
      if (!inPlay.includes(c.deck)) return;
      if (!groups.has(c.design)) groups.set(c.design, []);
      groups.get(c.design)!.push(ci);
    });
    for (const [des, grp] of groups) {
      if (grp.length < 2 || grp.length > inPlay.length || new Set(grp.map((ci) => cards[ci].deck)).size === grp.length) continue;
      let best: Deck[] = grp.map((ci) => cards[ci].deck), bestS = -Infinity;
      for (const perm of permutations(inPlay.length, grp.length)) {
        if (grp.some((ci, k) => inPlay[perm[k]] !== cards[ci].deck &&
          badgeLp(ci, cards[ci].deck) - badgeLp(ci, inPlay[perm[k]]) >= MOVE_MAX_GAP)) continue;
        const s = grp.reduce((acc, ci, k) => acc + lps[ci][DECKS.indexOf(inPlay[perm[k]])], 0);
        if (s > bestS) { best = perm.map((i) => inPlay[i]); bestS = s; }
      }
      grp.forEach((ci, k) => {
        if (best[k] !== cards[ci].deck) {
          warnings.push(`card ${des} read as ${cards[ci].deck}, but ${cards[ci].deck} ${des} is already on the table (each deck has one); using ${best[k]}`);
          cards[ci].deck = best[k];
        }
      });
    }
  }
  const seen = new Map<string, number>();
  for (const c of cards) seen.set(`${c.deck} ${c.design}`, (seen.get(`${c.deck} ${c.design}`) ?? 0) + 1);
  for (const [key, n] of seen) if (n > 1) warnings.push(`${key} detected ${n} times (each deck has one)`);
  for (const d of dots) {
    if (d.owner !== null && d.owner !== undefined) {
      cards[d.owner].visible++;
      counts[cards[d.owner].deck]++;
    }
  }
  // A "card" with no counted dots, a weak badge score and an unsure colour is almost certainly
  // not a card (clothing, clutter). Real cards with every dot covered keep a confident badge.
  const weak = cards.flatMap((c, ci) => {
    const b = badges[c.badge];
    return c.visible === 0 && (b.score < MIN_BADGE_SCORE_NO_DOTS || (b.deck_p ?? 1) < MIN_DECK_P_NO_DOTS) ? [ci] : [];
  });
  if (weak.length) {
    warnings.push(`ignored ${weak.length} weak badge-like detection(s) with no dots`);
    for (const ci of weak) { cards[ci].ignored = true; badges[cards[ci].badge].rejected = "weak"; }
  }
  // Cards connect to the board; a card standing alone is legal (no legal move) but rare. A lone
  // "card" with a weak or unsure reading is clutter: a shoe, a shirt, a controller, box art.
  // Checked again after the few-cards rule below: box art can hold several phantom "cards" that
  // keep each other company until their colour is dropped.
  const dropLone = () => {
    const lone = cards.flatMap((c, ci) =>
      !c.ignored && isWeakReading(badges[c.badge]) &&
      cards.every((o, oi) => oi === ci || o.ignored || quadGap(c.corners, o.corners) > ISOLATED_GAP * c.width) ? [ci] : []);
    if (lone.length) {
      const lost = lone.reduce((s, ci) => s + cards[ci].visible, 0);
      warnings.push(`ignored ${lone.length} weak detection(s) away from the other cards (${lost} dots)`);
      for (const ci of lone) {
        const c = cards[ci];
        c.ignored = true;
        counts[c.deck] -= c.visible;
        badges[c.badge].rejected = "lone";
      }
      for (const d of dots) {
        if (d.owner !== null && d.owner !== undefined && lone.includes(d.owner)) { d.owner = null; d.ignored = true; }
      }
    }
  };
  dropLone();
  // A deck in play always has several cards. A colour with only a few "cards" while other decks
  // have many is clutter (e.g. the card art on the game box): don't count it, but say so.
  const live = new Map<Deck, number>();
  for (const c of cards) if (!c.ignored) live.set(c.deck, (live.get(c.deck) ?? 0) + 1);
  // Red and pink look alike. When together they fit in one deck (18 cards) and one of them is
  // much smaller, they are one deck read partly wrong: count the smaller as the larger. (With
  // both decks in play there are ~18 of each.)
  {
    const [a, b] = LOOKALIKE_PAIR;
    const na = live.get(a) ?? 0, nb = live.get(b) ?? 0;
    const [small, large] = na < nb || (na === nb && a < b) ? [a, b] : [b, a];
    const ns = Math.min(na, nb), nl = Math.max(na, nb);
    if (ns > 0 && nl > 3 && (ns <= 3 || (2 * ns <= nl && ns + nl <= DECK_CARDS + 1))) {
      const moved = cards.reduce((s, c) => s + (c.deck === small && !c.ignored ? c.visible : 0), 0);
      warnings.push(`${ns} ${small} card(s) (${moved} dots) counted as ${large}: ${small} and ${large} look alike, and together they make one deck`);
      for (const c of cards) if (c.deck === small && !c.ignored) c.deck = large;
      counts[small] -= moved;
      counts[large] += moved;
      live.set(large, nl + ns);
      live.delete(small);
    }
  }
  if (live.size && Math.max(...live.values()) >= 10) {
    for (const [dk, n] of [...live]) {
      if (n >= MIN_DECK_IN_PLAY) continue;
      const lost = cards.reduce((s, c) => s + (c.deck === dk && !c.ignored ? c.visible : 0), 0);
      warnings.push(`ignored ${n} ${dk} card(s) (${lost} dots): too few for a deck in play`);
      for (const c of cards) if (c.deck === dk) c.ignored = true;
      counts[dk] -= lost;
      for (const d of dots) { // e.g. box art: not shown for review either
        if (d.owner !== null && d.owner !== undefined && cards[d.owner].deck === dk) { d.owner = null; d.ignored = true; }
      }
    }
    dropLone();
  }
  const remap = new Map<number, number>();
  const kept: Card[] = [];
  cards.forEach((c, ci) => { if (!c.ignored) { remap.set(ci, kept.length); kept.push(c); } });
  // Possible dots worth a player's look are on the board: within a card width of a found card.
  // (Off-board ones are clutter: clothing, the game box, other objects.)
  const near = review.filter((d) => kept.some((c) => quadDistance([d.x, d.y], c.corners) <= REVIEW_REACH * c.width));
  for (const d of review) if (!near.includes(d)) d.ignored = true;
  review.splice(0, review.length, ...near);
  if (onCardOnly) warnings.push(`${onCardOnly} dot(s) on a card but off its expected dot positions; counted (lower confidence)`);
  if (review.length) warnings.push(`${review.length} possible dot(s) not on any detected card were NOT counted - please check`);
  if (colourConflicts) warnings.push(`${colourConflicts} dot(s) whose colour disagreed with their card; counted for the card`);
  const perDeck = Object.fromEntries(DECKS.map((dk) => [dk, kept.filter((c) => c.deck === dk).length])) as Record<Deck, number>;
  for (const dk of DECKS) {
    const n = perDeck[dk];
    if (n > 18) warnings.push(`${dk}: ${n} cards detected (a deck has 18)`);
    else if (n > 0 && n < 18) warnings.push(`${dk}: ${n} cards detected (18 expected unless a card was removed)`);
  }
  // Partly covered dots: a misaligned card cuts them. Not counted (house rule); shown if on a card.
  partial = partial.filter((d) => kept.some((c) => pointInQuad([d.x, d.y], c.corners, 0.04)));
  if (partial.length) warnings.push(`${partial.length} partly covered dot(s) (a card edge crosses them) were NOT counted`);
  for (const d of dots) if (d.owner !== null && d.owner !== undefined) d.owner = remap.get(d.owner) ?? null;
  return {
    partial,
    perspective_fit: G !== null,
    counts: Object.fromEntries(DECKS.filter((dk) => counts[dk] || perDeck[dk]).map((dk) => [dk, counts[dk]])),
    cards_per_deck: Object.fromEntries(DECKS.filter((dk) => perDeck[dk]).map((dk) => [dk, perDeck[dk]])),
    cards: kept, review, warnings, scale_factor: bestK,
  };
}
