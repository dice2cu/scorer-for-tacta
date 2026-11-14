// Reads each card's colour-accessibility icon (the shape below the badge) from the full-resolution
// photo: blue concave square, green trefoil, orange two bars, pink upright diamond, purple fan, red
// chevron. The icon's shape doesn't change with white balance or light, so it is the primary deck
// evidence (the detector's colour reading is secondary; see fuseIcon in postprocess.ts).
// Port of tooling/detector/icons.py; only the patch resampling differs (canvas vs cv2.warpAffine).
import type { Rgba } from "./analyze.ts";
import type { Badge, Deck } from "./decode.ts";
import type { Library } from "./library.ts";

const DECKS: Deck[] = ["blue", "green", "orange", "pink", "purple", "red"];
const MARGIN = 45; // search margin around the template box (px at tex_w per card width)
const CROP_R = 5; // template-match shift range around the found blob (px)
const SCALES = [0.9, 1.0, 1.1];
const COARSE_STEP = 2; // shift search: every 2nd shift, then refined around the best (speed)
const GRAD_WEIGHT = 1.0; // weight of the gradient correlation (intensity correlation = 1)
const THRESHOLDS = [0.5, 0.65, 0.8, 0.35];
const CLOSE_SIZES = [5, 9];
const BLOB_W = [6, 40], BLOB_H = [10, 34], BLOB_AREA = [70, 750];
const MAX_DX = 18; // blob centre across the badge->icon axis (px)
const DY_RANGE = [-22, 38]; // ... and along it, from the expected icon centre (px)
const ICON_CY_OFFSET = -4; // the icon's centre sits this far above the template box centre (px)
const P_LO = 0.2, P_HI = 0.995; // whiteness stretch percentiles
// Below this card width (photo px) the icon is too small to read (~10 px); the deck then comes from
// colour alone. Real phone photos so far: 207-620 px.
const MIN_CARD_PX = 150;

export interface IconsJson {
  tex_w: number;
  icons: Record<Deck, { w: number; h: number; data: number[] }>;
  disks: Record<string, [number, number][]>;
}

interface Img { w: number; h: number; d: Float64Array }
interface Variant { t: Img; gx: Img; gy: Img; tz: Float64Array; tzz: number; gxx: number; gyy: number }

export interface IconReader {
  texW: number;
  tw: number; // template box (px)
  th: number;
  variants: Map<Deck, Variant[]>;
  disks: Map<number, [number, number][]>;
}

/** 2x3 affine [a, b, c, d, e, f]: photo = (a*px + b*py + c, d*px + e*py + f), pixel centres at integers. */
export type Affine = [number, number, number, number, number, number];

/** Scale an image by s (bilinear, pixel centres aligned, edges clamped). */
function resizeBilinear(t: Img, s: number): Img {
  const H = Math.floor(t.h * s + 0.5), W = Math.floor(t.w * s + 0.5);
  const d = new Float64Array(W * H);
  const clamp = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  for (let y = 0; y < H; y++) {
    const sy = clamp((y + 0.5) / s - 0.5, t.h - 1), y0 = Math.floor(sy), y1 = Math.min(y0 + 1, t.h - 1), fy = sy - y0;
    for (let x = 0; x < W; x++) {
      const sx = clamp((x + 0.5) / s - 0.5, t.w - 1), x0 = Math.floor(sx), x1 = Math.min(x0 + 1, t.w - 1), fx = sx - x0;
      const a = t.d[y0 * t.w + x0] * (1 - fx) + t.d[y0 * t.w + x1] * fx;
      const b = t.d[y1 * t.w + x0] * (1 - fx) + t.d[y1 * t.w + x1] * fx;
      d[y * W + x] = Math.fround(a * (1 - fy) + b * fy); // Python keeps templates in float32
    }
  }
  return { w: W, h: H, d };
}

/** 3x3 Sobel x and y, edges replicated. */
function sobel(a: Img): [Img, Img] {
  const { w, h } = a;
  const gx = new Float64Array(w * h), gy = new Float64Array(w * h);
  const at = (x: number, y: number) => a.d[Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      gx[y * w + x] = Math.fround((at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1)));
      gy[y * w + x] = Math.fround((at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1)));
    }
  }
  return [{ w, h, d: gx }, { w, h, d: gy }];
}

export function makeIconReader(json: IconsJson): IconReader {
  const variants = new Map<Deck, Variant[]>();
  for (const deck of DECKS) {
    const e = json.icons[deck];
    const base: Img = { w: e.w, h: e.h, d: Float64Array.from(e.data, (v) => Math.fround(v / 255)) };
    const bases = [base];
    if (deck === "orange") { // card backs show it mirrored
      const m = new Float64Array(e.w * e.h);
      for (let y = 0; y < e.h; y++) for (let x = 0; x < e.w; x++) m[y * e.w + x] = base.d[y * e.w + e.w - 1 - x];
      bases.push({ w: e.w, h: e.h, d: m });
    }
    const vs: Variant[] = [];
    for (const b of bases) {
      for (const s of SCALES) {
        const t = resizeBilinear(b, s);
        const [gx, gy] = sobel(t);
        let mean = 0;
        for (const v of t.d) mean += v;
        mean /= t.d.length;
        const tz = t.d.map((v) => v - mean);
        let tzz = 0, gxx = 0, gyy = 0;
        for (let i = 0; i < tz.length; i++) { tzz += tz[i] * tz[i]; gxx += gx.d[i] * gx.d[i]; gyy += gy.d[i] * gy.d[i]; }
        vs.push({ t, gx, gy, tz, tzz, gxx, gyy });
      }
    }
    variants.set(deck, vs);
  }
  const red = json.icons.red;
  const disks = new Map(Object.entries(json.disks).map(([k, v]) => [Number(k), v] as [number, [number, number][]]));
  return { texW: json.tex_w, tw: red.w, th: red.h, variants, disks };
}

/** Where to cut a badge's icon patch: patch px -> photo px, and the patch size. k: photo px per
 *  badge-coordinate px. The patch centre is the detector's icon point; null for start cards. */
export function patchAffine(lib: Library, reader: IconReader, b: Badge, k: number): { m: Affine; w: number; h: number } | null {
  if (b.suit === "start") return null;
  const bx = b.x * k, by = b.y * k, ix = b.icon[0] * k, iy = b.icon[1] * k;
  const L = Math.hypot(ix - bx, iy - by);
  if (L < 1e-6) return null;
  const v = [(ix - bx) / L, (iy - by) / L], u = [v[1], -v[0]];
  const width = L / ((lib.icon.y - lib.badgeBySuit.get(b.suit)!.y) / lib.aspect);
  if (width < MIN_CARD_PX) return null;
  const s = width / reader.texW;
  const pw = reader.tw + 2 * MARGIN, ph = reader.th + 2 * MARGIN;
  return {
    m: [s * u[0], s * v[0], ix - pw / 2 * s * u[0] - ph / 2 * s * v[0],
        s * u[1], s * v[1], iy - pw / 2 * s * u[1] - ph / 2 * s * v[1]],
    w: pw, h: ph,
  };
}

/** The photo pixels a patch needs: [x, y, w, h] (may extend past the photo; outside = black). */
export function patchRegion(m: Affine, w: number, h: number): [number, number, number, number] {
  const xs: number[] = [], ys: number[] = [];
  for (const [px, py] of [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]) {
    xs.push(m[0] * px + m[1] * py + m[2]);
    ys.push(m[3] * px + m[4] * py + m[5]);
  }
  const x0 = Math.floor(Math.min(...xs)) - 1, y0 = Math.floor(Math.min(...ys)) - 1;
  return [x0, y0, Math.floor(Math.max(...xs)) + 3 - x0, Math.floor(Math.max(...ys)) + 3 - y0];
}

/** Min-channel (0..1) of the patch: plain float bilinear sampling of `region` (photo pixels from
 *  (rx, ry)), each neighbour outside the photo black. Python: cut_patch + rgb.min(axis=2). */
function sampleMin(region: Rgba, rx: number, ry: number, m: Affine, w: number, h: number, photoW: number, photoH: number): Float64Array {
  const out = new Float64Array(w * h);
  const at = (x: number, y: number, c: number) =>
    x >= 0 && x < photoW && y >= 0 && y < photoH ? region.data[4 * ((y - ry) * region.width + (x - rx)) + c] : 0;
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const sx = m[0] * px + m[1] * py + m[2], sy = m[3] * px + m[4] * py + m[5];
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      let mn = Infinity;
      for (let c = 0; c < 3; c++) {
        const v = at(x0, y0, c) * ((1 - fx) * (1 - fy)) + at(x0 + 1, y0, c) * (fx * (1 - fy))
          + at(x0, y0 + 1, c) * ((1 - fx) * fy) + at(x0 + 1, y0 + 1, c) * (fx * fy);
        mn = Math.min(mn, Math.fround(v / 255));
      }
      out[py * w + px] = mn;
    }
  }
  return out;
}

function whiteness(mn: Float64Array, w: number, h: number): Img {
  const n = w * h;
  const s = Float64Array.from(mn).sort();
  const lo = s[Math.floor(P_LO * (n - 1))], hi = s[Math.floor(P_HI * (n - 1))];
  const den = Math.max(hi - lo, 1e-3);
  const d = mn.map((v) => Math.fround(Math.min(Math.max((v - lo) / den, 0), 1)));
  return { w, h, d };
}

/** Binary closing (dilate, then erode) with the given offsets; outside the patch is neutral. */
function close(m: Uint8Array, w: number, h: number, offs: [number, number][]): Uint8Array {
  const dil = new Uint8Array(w * h), ero = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (const [dy, dx] of offs) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < h && xx >= 0 && xx < w && m[yy * w + xx]) { dil[y * w + x] = 1; break; }
      }
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let all = 1;
      for (const [dy, dx] of offs) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < h && xx >= 0 && xx < w && !dil[yy * w + xx]) { all = 0; break; }
      }
      ero[y * w + x] = all;
    }
  }
  return ero;
}

/** 8-connected components in raster order of their first pixel: [x, y, w, h, area] each. */
function components(m: Uint8Array, w: number, h: number): number[][] {
  const lab = new Int32Array(w * h);
  const out: number[][] = [];
  const stack: number[] = [];
  for (let start = 0; start < w * h; start++) {
    if (!m[start] || lab[start]) continue;
    const id = out.length + 1;
    let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
    lab[start] = id;
    stack.push(start);
    while (stack.length) {
      const p = stack.pop()!, x = p % w, y = (p - x) / w;
      area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
          const q = yy * w + xx;
          if (m[q] && !lab[q]) { lab[q] = id; stack.push(q); }
        }
      }
    }
    out.push([x0, y0, x1 - x0 + 1, y1 - y0 + 1, area]);
  }
  return out;
}

/** Centre (x, y) of the icon blob in patch px, or null. */
function locate(reader: IconReader, wimg: Img): [number, number] | null {
  const { w, h } = wimg;
  const ex = w / 2, ey = h / 2 + ICON_CY_OFFSET;
  for (const n of CLOSE_SIZES) {
    for (const thr of THRESHOLDS) {
      const m = new Uint8Array(w * h);
      for (let i = 0; i < m.length; i++) m[i] = wimg.d[i] > thr ? 1 : 0;
      let best: [number, number] | null = null, bestD = Infinity;
      for (const [x, y, bw, bh, a] of components(close(m, w, h, reader.disks.get(n)!), w, h)) {
        if (!(BLOB_W[0] <= bw && bw <= BLOB_W[1] && BLOB_H[0] <= bh && bh <= BLOB_H[1] && BLOB_AREA[0] <= a && a <= BLOB_AREA[1])) continue;
        const cx = x + bw / 2, cy = y + bh / 2;
        const dx = cx - ex, dy = cy - ey;
        if (Math.abs(dx) > MAX_DX || !(DY_RANGE[0] <= dy && dy <= DY_RANGE[1])) continue;
        const dd = dx * dx + (0.6 * dy) ** 2;
        if (dd < bestD) { best = [cx, cy - ICON_CY_OFFSET]; bestD = dd; }
      }
      if (best) return best;
    }
  }
  return null;
}

/** Summed-area table ((w+1) x (h+1)) of f(values). */
function integral(a: Img, f: (v: number) => number): Float64Array {
  const W = a.w + 1, S = new Float64Array(W * (a.h + 1));
  for (let y = 0; y < a.h; y++) {
    let row = 0;
    for (let x = 0; x < a.w; x++) {
      row += f(a.d[y * a.w + x]);
      S[(y + 1) * W + x + 1] = S[y * W + x + 1] + row;
    }
  }
  return S;
}

interface CropSums { w: number; s: Float64Array; ss: Float64Array; gxx: Float64Array; gyy: Float64Array }
const boxSum = (S: Float64Array, W: number, x: number, y: number, w: number, h: number) =>
  S[(y + h) * W + x + w] - S[y * W + x + w] - S[(y + h) * W + x] + S[y * W + x];

/** Best normalised correlation of a template variant inside the crop: every COARSE_STEP-th shift,
 *  then all shifts next to the best of those (Python: match() in read_patch). */
function bestMatch(crop: Img, cgx: Img, cgy: Img, sums: CropSums, v: Variant): number {
  const { w: tw, h: th } = v.t;
  const n = tw * th;
  const tz = v.tz, tgx = v.gx.d, tgy = v.gy.d, cd = crop.d, gxd = cgx.d, gyd = cgy.d, cw = crop.w;
  const maxY = crop.h - th, maxX = crop.w - tw, S = COARSE_STEP;
  const corr = (x: number, y: number): number => {
    let num = 0, nx = 0, ny = 0;
    for (let j = 0; j < th; j++) {
      const r = (y + j) * cw + x, q = j * tw;
      for (let i = 0; i < tw; i++) {
        num += cd[r + i] * tz[q + i];
        nx += gxd[r + i] * tgx[q + i];
        ny += gyd[r + i] * tgy[q + i];
      }
    }
    const ws = boxSum(sums.s, sums.w, x, y, tw, th), wsq = boxSum(sums.ss, sums.w, x, y, tw, th);
    const xx = boxSum(sums.gxx, sums.w, x, y, tw, th), yy = boxSum(sums.gyy, sums.w, x, y, tw, th);
    const den = Math.sqrt(Math.max(wsq - ws * ws / n, 0) * v.tzz);
    const dx = Math.sqrt(xx * v.gxx), dy = Math.sqrt(yy * v.gyy);
    return (den > 1e-9 ? num / den : 0) + GRAD_WEIGHT * 0.5 * ((dx > 1e-9 ? nx / dx : 0) + (dy > 1e-9 ? ny / dy : 0));
  };
  let bx = 0, by = 0, coarse = -Infinity;
  for (let y = 0; y <= maxY; y += S) {
    for (let x = 0; x <= maxX; x += S) {
      const R = corr(x, y);
      if (R > coarse) { coarse = R; bx = x; by = y; }
    }
  }
  let best = -Infinity;
  for (let y = Math.max(by - S + 1, 0); y <= Math.min(by + S - 1, maxY); y++) {
    for (let x = Math.max(bx - S + 1, 0); x <= Math.min(bx + S - 1, maxX); x++) best = Math.max(best, corr(x, y));
  }
  return best;
}

/** Icon scores (one per deck, in DECKS order) for the patch placed by patchAffine (m, w, h), sampled
 *  from `region` (the photo pixels of patchRegion, starting at photo pixel (rx, ry)); null if no
 *  icon-like blob was found. */
export function readPatch(reader: IconReader, m: Affine, w: number, h: number, region: Rgba, rx: number, ry: number,
  photoW: number, photoH: number): number[] | null {
  const wimg = whiteness(sampleMin(region, rx, ry, m, w, h, photoW, photoH), w, h);
  const ctr = locate(reader, wimg);
  if (!ctr) return null;
  const { tw, th } = reader;
  const cx = Math.floor(ctr[0] + 0.5), cy = Math.floor(ctr[1] + 0.5);
  const y0 = cy - Math.floor(th / 2) - CROP_R, x0 = cx - Math.floor(tw / 2) - CROP_R;
  const ch = 2 * Math.floor(th / 2) + 2 * CROP_R + 1, cw = 2 * Math.floor(tw / 2) + 2 * CROP_R + 1;
  const crop: Img = { w: cw, h: ch, d: new Float64Array(cw * ch) }; // zero outside the patch
  for (let y = 0; y < ch; y++) {
    const sy = y0 + y;
    if (sy < 0 || sy >= wimg.h) continue;
    for (let x = 0; x < cw; x++) {
      const sx = x0 + x;
      if (sx >= 0 && sx < wimg.w) crop.d[y * cw + x] = wimg.d[sy * wimg.w + sx];
    }
  }
  const [gx, gy] = sobel(crop);
  const sq = (v: number) => v * v;
  const sums: CropSums = {
    w: cw + 1, s: integral(crop, (v) => v), ss: integral(crop, sq), gxx: integral(gx, sq), gyy: integral(gy, sq),
  };
  return DECKS.map((deck) => {
    let best = -Infinity;
    for (const v of reader.variants.get(deck)!) {
      if (v.t.h > crop.h || v.t.w > crop.w) continue;
      best = Math.max(best, bestMatch(crop, gx, gy, sums, v));
    }
    return best;
  });
}
