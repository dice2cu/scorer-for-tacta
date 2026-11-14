// The detector network as exported by tooling/detector/export_web.py: a list of ops over named
// tensors, BatchNorm already folded into the convolutions, float32 weights in one binary blob.

export interface ConvOp {
  op: "conv";
  src: string;
  dst: string;
  res: string | null; // residual added before the activation
  relu: boolean;
  cin: number;
  cout: number;
  k: number;
  stride: number;
  pad: number;
  dil: number;
  depthwise: boolean; // otherwise dense (groups = 1)
  w: number; // offset into weights: [cout][cin or 1][k][k]
  b: number; // offset into weights: [cout]
}

export interface UpAddOp {
  op: "upadd"; // dst = a + nearest_upsample_2x(b)
  a: string;
  b: string;
  dst: string;
}

export type Op = ConvOp | UpAddOp;

export interface ModelMeta {
  format: "tactanet-v1";
  input: string;
  output: string;
  input_channels: number;
  input_multiple: number;
  mean: number[];
  std: number[];
  stride: number;
  n_out: number;
  heads: [string, number][];
  decks: string[];
  suits: string[];
  icon_scale: number;
  target_card_width: number;
  weights_count: number;
  ops: Op[];
}

export interface Model {
  meta: ModelMeta;
  weights: Float32Array;
}

/** A CHW float tensor. */
export interface Tensor {
  c: number;
  h: number;
  w: number;
  data: Float32Array;
}

export function makeModel(meta: ModelMeta, weightBytes: ArrayBuffer): Model {
  if (meta.format !== "tactanet-v1") throw new Error(`unknown model format ${meta.format}`);
  const weights = new Float32Array(weightBytes);
  if (weights.length !== meta.weights_count) throw new Error("model weights file doesn't match the graph");
  return { meta, weights };
}

export function convOutSize(n: number, op: ConvOp): number {
  return Math.floor((n + 2 * op.pad - op.dil * (op.k - 1) - 1) / op.stride) + 1;
}

/** Output shape of every tensor for a given input size. */
export function shapes(meta: ModelMeta, h: number, w: number): Map<string, [number, number, number]> {
  const s = new Map<string, [number, number, number]>([[meta.input, [meta.input_channels, h, w]]]);
  for (const op of meta.ops) {
    if (op.op === "conv") {
      const [, ih, iw] = s.get(op.src)!;
      s.set(op.dst, [op.cout, convOutSize(ih, op), convOutSize(iw, op)]);
    } else {
      s.set(op.dst, s.get(op.a)!);
    }
  }
  return s;
}

/** Index of the last op that reads each tensor (so engines can free it afterwards). */
export function lastUse(meta: ModelMeta): Map<string, number> {
  const last = new Map<string, number>();
  meta.ops.forEach((op, i) => {
    const reads = op.op === "conv" ? [op.src, op.res] : [op.a, op.b];
    for (const t of reads) if (t) last.set(t, i);
  });
  return last;
}

/** RGB(A) bytes -> normalized CHW input tensor. */
export function normalizeInput(meta: ModelMeta, rgba: Uint8ClampedArray | Uint8Array, w: number, h: number, channels = 4): Tensor {
  const n = w * h;
  const data = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    const m = meta.mean[c], s = meta.std[c];
    const out = c * n;
    for (let i = 0; i < n; i++) data[out + i] = (rgba[i * channels + c] / 255 - m) / s;
  }
  return { c: 3, h, w, data };
}
