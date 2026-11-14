// The 18 Tacta card designs (built by tooling/card_library/build_card_library.py).
// Coordinates are normalized to the card in its front orientation: x in [0,1] left->right,
// y in [0,1] top->bottom (badge/icon near the middle).

import type { IconReader } from "./icons.ts";

export type Suit = "circle" | "square" | "triangle";

export interface Pt {
  x: number;
  y: number;
}

export interface CardDesign {
  id: string; // e.g. "circle-3"
  suit: Suit;
  value: number;
  dots: Pt[];
  badge: Pt & { w: number; h: number };
  icon: Pt;
}

export interface CardLibraryJson {
  version: number;
  aspect_w_over_h: number;
  dot_radius_over_width: number;
  cards: CardDesign[];
}

export interface Library {
  aspect: number; // card width / height
  dotRadius: number; // dot radius / card width
  designs: Map<string, CardDesign>;
  badgeBySuit: Map<Suit, CardDesign["badge"]>;
  icon: Pt;
  /** deck icon templates (icons.ts); without them decks come from colour alone */
  icons?: IconReader;
}

export function makeLibrary(json: CardLibraryJson): Library {
  const designs = new Map(json.cards.map((c) => [c.id, c]));
  const badgeBySuit = new Map<Suit, CardDesign["badge"]>();
  for (const c of json.cards) badgeBySuit.set(c.suit, c.badge); // last of each suit, as in Python
  return { aspect: json.aspect_w_over_h, dotRadius: json.dot_radius_over_width, designs, badgeBySuit, icon: json.cards[0].icon };
}

/** badge -> icon distance in card widths. */
export function iconDistance(lib: Library, suit: Suit): number {
  return (lib.icon.y - lib.badgeBySuit.get(suit)!.y) / lib.aspect;
}
