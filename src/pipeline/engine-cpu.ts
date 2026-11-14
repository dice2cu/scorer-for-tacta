// Reference CPU implementation of the detector (plain TypeScript, single thread).
// Used as the fallback when WebGL2 float rendering is unavailable, and to check the GPU engine.
import type { ConvOp, Model, Tensor } from "./model.ts";
import { convOutSize, lastUse } from "./model.ts";

const BLOCK = 512; // pixels per tile in 1x1 convs (keeps the input tile in cache)

function conv(op: ConvOp, x: Tensor, wts: Float32Array, res: Tensor | null): Tensor {
  const oh = convOutSize(x.h, op), ow = convOutSize(x.w, op);
  const on = oh * ow;
  const out = new Float32Array(op.cout * on);
  const W = wts.subarray(op.w), B = wts.subarray(op.b);
  if (op.k === 1 && op.stride === 1 && !op.depthwise) {
    const cin = op.cin, src = x.data;
    for (let p0 = 0; p0 < on; p0 += BLOCK) {
      const p1 = Math.min(on, p0 + BLOCK);
      for (let o = 0; o < op.cout; o++) {
        const ob = o * on, b = B[o];
        for (let p = p0; p < p1; p++) out[ob + p] = b;
        for (let i = 0; i < cin; i++) {
          const wv = W[o * cin + i];
          if (wv === 0) continue;
          const ib = i * on - ob;
          for (let p = ob + p0, e = ob + p1; p < e; p++) out[p] += wv * src[p + ib];
        }
      }
    }
  } else {
    const k = op.k, s = op.stride, pad = op.pad, d = op.dil;
    const groupsIn = op.depthwise ? 1 : op.cin;
    for (let o = 0; o < op.cout; o++) {
      const ob = o * on;
      out.fill(B[o], ob, ob + on);
      for (let gi = 0; gi < groupsIn; gi++) {
        const ic = op.depthwise ? o : gi;
        const ib = ic * x.h * x.w;
        for (let ky = 0; ky < k; ky++) {
          for (let kx = 0; kx < k; kx++) {
            const wv = W[((o * groupsIn + gi) * k + ky) * k + kx];
            const offX = kx * d - pad;
            // valid output x range so that 0 <= ox*s + offX < x.w
            const x0 = Math.max(0, Math.ceil(-offX / s));
            const x1 = Math.min(ow, Math.floor((x.w - 1 - offX) / s) + 1);
            for (let oy = 0; oy < oh; oy++) {
              const iy = oy * s + ky * d - pad;
              if (iy < 0 || iy >= x.h) continue;
              const orow = ob + oy * ow, irow = ib + iy * x.w + offX;
              const src = x.data;
              if (s === 1) for (let ox = x0; ox < x1; ox++) out[orow + ox] += wv * src[irow + ox];
              else for (let ox = x0; ox < x1; ox++) out[orow + ox] += wv * src[irow + ox * s];
            }
          }
        }
      }
    }
  }
  if (res) for (let i = 0; i < out.length; i++) out[i] += res.data[i];
  if (op.relu) for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
  return { c: op.cout, h: oh, w: ow, data: out };
}

function upadd(a: Tensor, b: Tensor): Tensor {
  const out = new Float32Array(a.data.length);
  const n = a.h * a.w, bn = b.h * b.w;
  for (let c = 0; c < a.c; c++) {
    for (let y = 0; y < a.h; y++) {
      const ar = c * n + y * a.w, br = c * bn + (y >> 1) * b.w;
      for (let x = 0; x < a.w; x++) out[ar + x] = a.data[ar + x] + b.data[br + (x >> 1)];
    }
  }
  return { c: a.c, h: a.h, w: a.w, data: out };
}

export function runCpu(model: Model, input: Tensor): Tensor {
  const { meta, weights } = model;
  const last = lastUse(meta);
  const env = new Map<string, Tensor>([[meta.input, input]]);
  meta.ops.forEach((op, i) => {
    let y: Tensor;
    if (op.op === "conv") y = conv(op, env.get(op.src)!, weights, op.res ? env.get(op.res)! : null);
    else y = upadd(env.get(op.a)!, env.get(op.b)!);
    env.set(op.dst, y);
    for (const [t, j] of last) if (j === i && t !== meta.output) env.delete(t);
  });
  return env.get(meta.output)!;
}
