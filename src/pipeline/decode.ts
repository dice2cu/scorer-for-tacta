// Detector output maps -> badges (cards) and dots. Port of tooling/detector/decode.py.
import type { ModelMeta, Tensor } from "./model.ts";

export type Deck = "blue" | "green" | "orange" | "pink" | "purple" | "red";
export type BadgeSuit = "circle" | "square" | "triangle" | "start";

export interface Badge {
  x: number;
  y: number;
  score: number;
  suit: BadgeSuit;
  suit_p: number;
  value: number;
  value_p: number;
  value_probs: number[];
  side: "front" | "back";
  deck: Deck;
  deck_p: number;
  deck_probs: number[];
  icon: [number, number];
  /** deck icon match scores (DECKS order) read from the photo (icons.ts); null = not found */
  icon_scores?: number[] | null;
  /** the detector's colour reading, kept when deck_probs became the icon + colour belief */
  colour_probs?: number[];
  rejected?: string | null;
  rot_fix?: number;
}

export interface Dot {
  x: number;
  y: number;
  score: number;
  deck: Deck;
  deck_p: number;
  deck_probs: number[];
  // set by scoring
  explained?: boolean;
  owner?: number | null;
  ignored?: boolean;
  on_card_only?: boolean;
  partial?: boolean;
}

export const BADGE_THR = 0.3;
export const DOT_THR = 0.2; // weak dots are judged later, in context

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

function softmax(v: number[]): number[] {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function argmax(v: ArrayLike<number>): number {
  let k = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i;
  return k;
}

/** Local maxima (3x3) of one probability map above thr, in row-major order. */
function peaks(map: Float32Array, h: number, w: number, thr: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = map[y * w + x];
      if (!(v > thr)) continue;
      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          if (map[yy * w + xx] > v) { isMax = false; break; }
        }
      }
      if (isMax) out.push([y, x, v]);
    }
  }
  return out;
}

export const PARTIAL_THR = 0.3;
const PARTIAL_MERGE = 1.5; // cells: a full-dot and a partial-dot peak this close are one dot

/** Partly covered dots are included in dots with partial = true (never counted). */
export function decode(meta: ModelMeta, out: Tensor, badgeThr = BADGE_THR, dotThr = DOT_THR, partialThr = PARTIAL_THR): { badges: Badge[]; dots: Dot[] } {
  const { h, w, data } = out;
  const n = h * w;
  const slices: Record<string, number> = {};
  let k = 0;
  for (const [name, c] of meta.heads) { slices[name] = k; k += c; }
  const at = (ch: number, y: number, x: number) => data[ch * n + y * w + x];
  const vec = (start: number, len: number, y: number, x: number) => Array.from({ length: len }, (_, i) => at(start + i, y, x));
  const S = meta.stride;

  // badges: one per location regardless of suit channel
  const bh = slices.badge_hm;
  const merged = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let c = 0; c < 4; c++) m = Math.max(m, sigmoid(data[(bh + c) * n + i]));
    merged[i] = m;
  }
  const badges: Badge[] = [];
  for (const [y, x, s] of peaks(merged, h, w, badgeThr)) {
    const suitP = vec(bh, 4, y, x).map(sigmoid);
    const suit = argmax(suitP);
    const ox = sigmoid(at(slices.badge_off, y, x)), oy = sigmoid(at(slices.badge_off + 1, y, x));
    const bx = (x + ox) * S, by = (y + oy) * S;
    const dx = at(slices.icon, y, x) * meta.icon_scale, dy = at(slices.icon + 1, y, x) * meta.icon_scale;
    const pv = softmax(vec(slices.value, 6, y, x));
    const pd = softmax(vec(slices.deck, 6, y, x));
    badges.push({
      x: bx, y: by, score: s, suit: meta.suits[suit] as BadgeSuit,
      suit_p: suitP[suit] / Math.max(suitP.reduce((a, b) => a + b, 0), 1e-6),
      value: argmax(pv) + 1, value_p: Math.max(...pv), value_probs: pv,
      side: sigmoid(at(slices.side, y, x)) > 0.5 ? "back" : "front",
      deck: meta.decks[argmax(pd)] as Deck, deck_p: Math.max(...pd), deck_probs: pd,
      icon: [bx + dx, by + dy],
    });
  }

  const probMap = (ch: number) => {
    const m = new Float32Array(n);
    for (let i = 0; i < n; i++) m[i] = sigmoid(data[ch * n + i]);
    return m;
  };
  const dot = (y: number, x: number, s: number, partial: boolean): Dot => {
    const ox = sigmoid(at(slices.dot_off, y, x)), oy = sigmoid(at(slices.dot_off + 1, y, x));
    const pd = softmax(vec(slices.dot_deck, 6, y, x));
    const d: Dot = { x: (x + ox) * S, y: (y + oy) * S, score: s, deck: meta.decks[argmax(pd)] as Deck, deck_p: Math.max(...pd), deck_probs: pd };
    if (partial) d.partial = true;
    return d;
  };
  const full = peaks(probMap(slices.dot_hm), h, w, dotThr);
  const part = slices.dot_part !== undefined ? peaks(probMap(slices.dot_part), h, w, partialThr) : [];
  // one dot per place: where a full and a partial peak coincide, the stronger reading wins
  const dropFull = new Set<number>(), dropPart = new Set<number>();
  part.forEach(([py, px, ps], i) => {
    full.forEach(([fy, fx, fs], j) => {
      if ((fy - py) ** 2 + (fx - px) ** 2 <= PARTIAL_MERGE ** 2) {
        if (ps > fs) dropFull.add(j); else dropPart.add(i);
      }
    });
  });
  const dots: Dot[] = [
    ...full.filter((_, j) => !dropFull.has(j)).map(([y, x, s]) => dot(y, x, s, false)),
    ...part.filter((_, i) => !dropPart.has(i)).map(([y, x, s]) => dot(y, x, s, true)),
  ];
  return { badges, dots };
}
