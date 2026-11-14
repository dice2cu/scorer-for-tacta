// A user's photo as a resizable source (canvas-based, with progressive halving so large
// downscales average pixels rather than skipping them).
import type { PhotoSource, Rgba } from "../pipeline/analyze.ts";

export interface Photo extends PhotoSource {
  bitmap: ImageBitmap;
}

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return [c, ctx];
}

// Larger photos are shrunk once on loading: the analysis never uses more than 2560 px and the
// overlay 3200, and a 50-200 MP photo kept at full size would exhaust a phone's memory.
const MAX_PHOTO_SIDE = 4096;

export async function loadPhoto(file: Blob): Promise<Photo> {
  let bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const long = Math.max(bitmap.width, bitmap.height);
  if (long > MAX_PHOTO_SIDE) {
    const k = MAX_PHOTO_SIDE / long;
    try {
      const small = await createImageBitmap(bitmap, {
        resizeWidth: Math.round(bitmap.width * k), resizeHeight: Math.round(bitmap.height * k), resizeQuality: "high",
      });
      bitmap.close();
      bitmap = small;
    } catch { /* keep the full-size photo */ }
  }
  return {
    bitmap,
    width: bitmap.width,
    height: bitmap.height,
    resize(w: number, h: number): Rgba {
      let src: CanvasImageSource = bitmap, sw = bitmap.width, sh = bitmap.height;
      while (sw / 2 >= w && sh / 2 >= h) {
        const [c, ctx] = canvas(Math.round(sw / 2), Math.round(sh / 2));
        ctx.drawImage(src, 0, 0, c.width, c.height);
        src = c; sw = c.width; sh = c.height;
      }
      const [, ctx] = canvas(w, h);
      ctx.drawImage(src, 0, 0, w, h);
      const d = ctx.getImageData(0, 0, w, h);
      return { width: w, height: h, data: d.data };
    },
    region(x: number, y: number, w: number, h: number): Rgba {
      const [, ctx] = canvas(w, h); // 1:1, so no resampling; outside the photo stays black
      ctx.drawImage(bitmap, -x, -y);
      return { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data };
    },
  };
}
