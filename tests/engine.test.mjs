// Checks the TypeScript CPU engine against PyTorch on the parity case written by
// tooling/detector/export_web.py (out/test-cases/parity/), or the committed one in tests/fixtures/.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeModel, normalizeInput } from "../src/pipeline/model.ts";
import { runCpu } from "../src/pipeline/engine-cpu.ts";
import { parityCase, root } from "./data.mjs";

export function loadModel() {
  const meta = JSON.parse(readFileSync(new URL("src/data/model/tactanet.json", root), "utf8"));
  const bin = readFileSync(new URL("src/data/model/tactanet.bin", root));
  return makeModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
}

test("CPU engine matches PyTorch", () => {
  const model = loadModel();
  const p = parityCase();
  const c = p.meta;
  const rgb = new Uint8Array(p.rgb);
  const ref = new Float32Array(p.out.buffer.slice(p.out.byteOffset, p.out.byteOffset + p.out.byteLength));
  const t0 = performance.now();
  const out = runCpu(model, normalizeInput(model.meta, rgb, c.width, c.height, 3));
  const ms = performance.now() - t0;
  assert.deepEqual([out.c, out.h, out.w], c.out_shape);
  let maxErr = 0;
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(out.data[i] - ref[i]));
  console.log(`  ${p.source} ${c.width}x${c.height}:${ms.toFixed(0)} ms, max abs error ${maxErr.toExponential(2)}`);
  assert.ok(maxErr < 1e-3, `max abs error ${maxErr}`);
});
