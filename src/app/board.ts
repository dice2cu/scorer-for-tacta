// The scored board in photo coordinates, plus the player's corrections (tap a dot to toggle it,
// tap a missed dot on a card to add it).
import type { Analysis } from "../pipeline/analyze.ts";
import type { Deck } from "../pipeline/decode.ts";
import type { XY } from "../pipeline/perspective.ts";
import { DECKS, pointInQuad } from "../pipeline/postprocess.ts";

export interface CardMark {
  deck: Deck;
  design: string;
  side: "front" | "back";
  corners: XY[];
  printed: XY[]; // printed dot positions
  width: number;
}

export interface DotMark {
  x: number;
  y: number;
  deck: Deck;
  card: number | null;
  /** how the app judged it: counted, shown for review (not counted), partly covered by a card
   *  (not counted), or added by the player */
  source: "counted" | "review" | "partial" | "manual";
  counted: boolean;
}

export interface Board {
  cards: CardMark[];
  dots: DotMark[];
  dotRadius: number; // photo px
  edits: number;
}

export function makeBoard(a: Analysis, dotRadiusOverWidth: number): Board {
  const s = 1 / a.scale;
  const p = ([x, y]: XY): XY => [x * s, y * s];
  const cards: CardMark[] = a.result.cards.map((c) => ({
    deck: c.deck, design: c.design, side: c.side, corners: c.corners.map(p), printed: c.dots.map(p), width: c.width * s,
  }));
  const dots: DotMark[] = [];
  for (const d of a.dots) {
    if (d.owner !== null && d.owner !== undefined) {
      dots.push({ x: d.x * s, y: d.y * s, deck: cards[d.owner].deck, card: d.owner, source: "counted", counted: true });
    }
  }
  for (const d of a.result.review) dots.push({ x: d.x * s, y: d.y * s, deck: d.deck, card: null, source: "review", counted: false });
  for (const d of a.result.partial) dots.push({ x: d.x * s, y: d.y * s, deck: d.deck, card: null, source: "partial", counted: false });
  return { cards, dots, dotRadius: dotRadiusOverWidth * a.cardWidth, edits: 0 };
}

export function counts(b: Board): Record<Deck, number> {
  const c = Object.fromEntries(DECKS.map((d) => [d, 0])) as Record<Deck, number>;
  for (const d of b.dots) if (d.counted) c[d.deck]++;
  return c;
}

export function cardsPerDeck(b: Board): Record<Deck, number> {
  const c = Object.fromEntries(DECKS.map((d) => [d, 0])) as Record<Deck, number>;
  for (const card of b.cards) c[card.deck]++;
  return c;
}

/** The card a point on the photo belongs to: among cards containing it, the one with a printed
 *  dot closest to the point (overlapping cards: the visible one has its dot there). */
function cardAt(b: Board, p: XY): number | null {
  let best: number | null = null, bestD = Infinity;
  b.cards.forEach((c, i) => {
    if (!pointInQuad(p, c.corners)) return;
    for (const q of c.printed) {
      const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (!c.printed.length && best === null) best = i;
  });
  return best;
}

/** Handle a tap at photo coordinates. Returns what happened (for a short message), or null. */
export function tap(b: Board, p: XY, slop: number): string | null {
  const R = Math.max(2.2 * b.dotRadius, slop);
  let hit = -1, hitD = Infinity;
  b.dots.forEach((d, i) => {
    const dist = Math.hypot(d.x - p[0], d.y - p[1]);
    if (dist < R && dist < hitD) { hit = i; hitD = dist; }
  });
  if (hit >= 0) {
    const d = b.dots[hit];
    if (d.source === "manual") {
      b.dots.splice(hit, 1);
      b.edits--;
      return "Removed the dot you added";
    }
    d.counted = !d.counted;
    b.edits += d.counted === (d.source === "counted") ? -1 : 1;
    if (d.counted && (d.source === "review" || d.source === "partial")) {
      const c = cardAt(b, [d.x, d.y]);
      if (c !== null) { d.card = c; d.deck = b.cards[c].deck; }
    }
    return d.counted ? "Counting this dot" : "Not counting this dot";
  }
  const c = cardAt(b, p);
  if (c === null) return null;
  b.dots.push({ x: p[0], y: p[1], deck: b.cards[c].deck, card: c, source: "manual", counted: true });
  b.edits++;
  return "Added a dot";
}

export function resetEdits(b: Board) {
  b.dots = b.dots.filter((d) => d.source !== "manual");
  for (const d of b.dots) d.counted = d.source === "counted";
  b.edits = 0;
}
