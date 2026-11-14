import type { Deck } from "../pipeline/decode.ts";

/** Display names, colours and symbols. (The model calls the yellow deck "orange".) symbol: SVG path
 *  (24x24 box) of the deck's colour-accessibility symbol, drawn by hand after the cards' shapes. */
export const DECK_INFO: Record<Deck, { name: string; color: string; symbol: string }> = {
  // concave square
  blue: { name: "Blue", color: "#2f7fdc", symbol: "M3.6 3.6Q12 7.6 20.4 3.6Q20.1 12 20.4 20.4Q12 16.4 3.6 20.4Q3.9 12 3.6 3.6Z" },
  // three petals
  green: { name: "Green", color: "#2ea44f", symbol: "M12 14.4C16.4 11.2 16.4 5.2 12 3.6C7.6 5.2 7.6 11.2 12 14.4ZM12 14.4C13.7 19.6 19.2 21.3 22 17.6C21.8 12.9 16.3 11.2 12 14.4ZM12 14.4C7.7 11.2 2.2 12.9 2 17.6C4.8 21.3 10.3 19.6 12 14.4Z" },
  // two tilted bars
  orange: { name: "Yellow", color: "#f0a020", symbol: "M3.2 4.3L9.6 3.5L11.6 19.7L5.2 20.5ZM12.4 4.3L18.8 3.5L20.8 19.7L14.4 20.5Z" },
  // tall six-sided diamond
  pink: { name: "Pink", color: "#e8559f", symbol: "M12 2.3L16.9 6.2V17.8L12 21.7L7.1 17.8V6.2Z" },
  // fan: a dome over concave sides that meet in a point
  purple: { name: "Purple", color: "#8b5cd0", symbol: "M2.5 11.6C3.1 6 7.1 3.6 12 3.6C16.9 3.6 20.9 6 21.5 11.6Q13 16.4 12.8 20.4H11.2Q11 16.4 2.5 11.6Z" },
  // chevron
  red: { name: "Red", color: "#dc3b3b", symbol: "M2.5 8.8L6.5 4L12 12L17.5 4L21.5 8.8L12 20Z" },
};

/** Black or white, whichever contrasts more with the colour (WCAG relative luminance). */
function inkFor(hex: string): string {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  // (lum + 0.05) / 0.05 vs 1.05 / (lum + 0.05): equal at lum ~0.179
  return (lum + 0.05) ** 2 > 1.05 * 0.05 ? "#000" : "#fff";
}

/** The deck's colour swatch with its symbol on it (decorative: the name is shown next to it). */
export function deckSwatch(deck: Deck): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const { color, symbol } = DECK_INFO[deck];
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("swatch");
  svg.style.background = color;
  svg.style.color = inkFor(color);
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", symbol);
  svg.append(path);
  return svg;
}
