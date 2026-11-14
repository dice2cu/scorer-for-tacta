// Photo -> scores. Runs the detector at a scale where cards are ~100 px wide (what it was trained
// for), decodes, and scores. The scale is found automatically: a first pass at a fixed size finds
// the cards and measures them; a second pass runs at the right scale unless the first was close.
// Finally each card's deck icon is read from the full-resolution photo and the cards are scored
// with it (icons.ts).
import type { Badge, Dot } from "./decode.ts";
import { decode } from "./decode.ts";
import { patchAffine, patchRegion, readPatch } from "./icons.ts";
import type { Library } from "./library.ts";
import type { Model, Tensor } from "./model.ts";
import { normalizeInput } from "./model.ts";
import type { ScoreResult } from "./postprocess.ts";
import { score } from "./postprocess.ts";
import { median } from "./perspective.ts";

export interface Engine {
  readonly name: string;
  run(input: Tensor): Tensor;
}

export interface Rgba {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** The photo, resizable to any size (the browser does this with a canvas). */
export interface PhotoSource {
  width: number;
  height: number;
  resize(width: number, height: number): Rgba | Promise<Rgba>;
  /** The photo's own pixels in a rectangle (no resampling; outside the photo = black). Without it,
   *  the deck icons aren't read and decks come from colour alone. */
  region?(x: number, y: number, width: number, height: number): Rgba | Promise<Rgba>;
}

export interface Analysis {
  /** model pixels per photo pixel; divide model coordinates by this to get photo coordinates */
  scale: number;
  inputSize: [number, number];
  cardWidth: number; // in photo px
  badges: Badge[];
  dots: Dot[];
  /** the final pass's detector output before scoring (for tests and debugging) */
  decoded: { badges: Badge[]; dots: Dot[] };
  result: ScoreResult;
  passes: { scale: number; size: [number, number]; badges: number; cardWidth: number | null; ms: number }[];
  /** deck icons read from the photo (null: not available, decks from colour alone) */
  icons: { read: number; ms: number } | null;
  engine: string;
}

const FIRST_PASS_SIZES = [1024, 1536, 704]; // long side, tried in order until cards are found
const CLOSE_ENOUGH = 0.08; // reuse the first pass if its card width is within 8% of the target
const MIN_CARDS = 6;
// Limits for the rescaled image: tiny-looking cards (e.g. a drawing of many cards) must not ask for
// an image too big for a phone's memory or time budget, and upscaling beyond 2x adds no detail.
const MAX_LONG_SIDE = 2560;
const MAX_UPSCALE = 2;

/** Pad to a multiple of m by repeating the last row/column (as cv2.BORDER_REPLICATE). */
function padTo(img: Rgba, m: number): Rgba {
  const W = Math.ceil(img.width / m) * m, H = Math.ceil(img.height / m) * m;
  if (W === img.width && H === img.height) return img;
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(y, img.height - 1);
    out.set(img.data.subarray(sy * img.width * 4, (sy + 1) * img.width * 4), y * W * 4);
    const last = (sy * img.width + img.width - 1) * 4;
    for (let x = img.width; x < W; x++) out.set(img.data.subarray(last, last + 4), (y * W + x) * 4);
  }
  return { width: W, height: H, data: out };
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

async function runAt(engine: Engine, model: Model, lib: Library, photo: PhotoSource, scale: number) {
  const t0 = performance.now();
  const w = Math.max(16, Math.round(photo.width * scale)), h = Math.max(16, Math.round(photo.height * scale));
  const img = padTo(await photo.resize(w, h), model.meta.input_multiple);
  const out = engine.run(normalizeInput(model.meta, img.data, img.width, img.height));
  const { badges, dots } = decode(model.meta, out);
  const decoded = { badges: structuredClone(badges), dots: structuredClone(dots) }; // score() annotates its inputs
  await tick();
  const result = score(lib, badges, dots, [img.width, img.height]);
  const widths = result.cards.map((c) => c.width);
  const cardWidth = widths.length >= MIN_CARDS ? median(widths) / scale : null;
  return { badges, dots, decoded, result, cardWidth, size: [img.width, img.height] as [number, number], ms: performance.now() - t0, scale };
}

export async function analyze(engine: Engine, model: Model, lib: Library, photo: PhotoSource,
  progress: (msg: string) => void = () => {}): Promise<Analysis> {
  const target = model.meta.target_card_width;
  const passes: Analysis["passes"] = [];
  const log = (r: Awaited<ReturnType<typeof runAt>>) =>
    passes.push({ scale: r.scale, size: r.size, badges: r.badges.length, cardWidth: r.cardWidth, ms: Math.round(r.ms) });

  progress("Finding the cards…");
  await tick();
  let first: Awaited<ReturnType<typeof runAt>> | null = null;
  for (const long of FIRST_PASS_SIZES) {
    const scale = Math.min(1, long / Math.max(photo.width, photo.height));
    const r = await runAt(engine, model, lib, photo, scale);
    log(r);
    if (first === null || r.result.cards.length > first.result.cards.length) first = r;
    if (r.cardWidth !== null || scale >= 1) break;
  }
  // Rescale until the cards measure ~target px (the first pass's size estimate is rougher when its
  // cards are far from the training range; one extra pass corrects that).
  let final = first!;
  for (let i = 0; i < 2 && final.cardWidth !== null; i++) {
    const scale = Math.min(target / final.cardWidth, MAX_UPSCALE, MAX_LONG_SIDE / Math.max(photo.width, photo.height));
    if (Math.abs(scale / final.scale - 1) <= CLOSE_ENOUGH) break;
    progress("Counting dots…");
    await tick();
    const next = await runAt(engine, model, lib, photo, scale);
    log(next);
    if (next.cardWidth === null && final.cardWidth !== null && next.result.cards.length < final.result.cards.length) break;
    final = next;
  }
  // Each card's deck icon, read from the photo itself at full resolution, then the final scoring.
  let { badges, dots, result } = final;
  let icons: Analysis["icons"] = null;
  if (lib.icons && photo.region) {
    progress("Reading the deck symbols…");
    await tick();
    const t0 = performance.now();
    let read = 0;
    for (const b of final.decoded.badges) {
      const p = patchAffine(lib, lib.icons, b, 1 / final.scale);
      if (!p) continue;
      const [rx, ry, rw, rh] = patchRegion(p.m, p.w, p.h);
      const region = await photo.region(rx, ry, rw, rh);
      b.icon_scores = readPatch(lib.icons, p.m, p.w, p.h, region, rx, ry, photo.width, photo.height);
      if (b.icon_scores) read++;
    }
    badges = structuredClone(final.decoded.badges);
    dots = structuredClone(final.decoded.dots);
    result = score(lib, badges, dots, final.size);
    icons = { read, ms: Math.round(performance.now() - t0) };
  }
  return {
    scale: final.scale, inputSize: final.size, cardWidth: final.cardWidth ?? target / final.scale,
    badges, dots, decoded: final.decoded, result, passes, icons, engine: engine.name,
  };
}
