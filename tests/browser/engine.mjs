// Browser check of both engines against PyTorch (parity case from export_web.py) + GPU timing.
// Run with: node tests/browser/engine-check.mjs (served by Vite's dev server, which compiles the
// TypeScript sources imported here). Uses out/test-cases/parity/ if present, else tests/fixtures/.
import { makeModel, normalizeInput } from "/src/pipeline/model.ts";
import { runCpu } from "/src/pipeline/engine-cpu.ts";
import { WebGLEngine } from "/src/pipeline/engine-webgl.ts";

const report = (r) => fetch("/__report", { method: "POST", body: JSON.stringify(r) });
const get = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r; };

try {
  const meta = await (await get("/src/data/model/tactanet.json")).json();
  const model = makeModel(meta, await (await get("/src/data/model/tactanet.bin")).arrayBuffer());
  const full = (await fetch("/out/test-cases/parity/case.json")).ok;
  const [dir, names] = full ? ["/out/test-cases/parity/", ["case.json", "input_rgb.u8", "output.f32"]]
    : ["/tests/fixtures/", ["parity.json", "parity_input.u8", "parity_output.f32"]];
  const c = await (await get(dir + names[0])).json();
  const rgb = new Uint8Array(await (await get(dir + names[1])).arrayBuffer());
  const ref = new Float32Array(await (await get(dir + names[2])).arrayBuffer());
  const input = normalizeInput(meta, rgb, c.width, c.height, 3);
  const maxErr = (a) => { let m = 0; for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(a[i] - ref[i])); return m; };

  let t = performance.now();
  const cpu = runCpu(model, input);
  const cpuMs = performance.now() - t;

  t = performance.now();
  const gpu = new WebGLEngine(model);
  const initMs = performance.now() - t;
  const dbg = gpu.gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = dbg ? gpu.gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gpu.gl.getParameter(gpu.gl.RENDERER);
  t = performance.now();
  const g1 = gpu.run(input);
  const gpuFirstMs = performance.now() - t;
  t = performance.now();
  gpu.run(input);
  const gpuMs = performance.now() - t;

  // realistic board size: ~100 px per card, whole board in frame
  const big = { c: 3, h: 768, w: 1024, data: new Float32Array(3 * 768 * 1024).map(() => Math.random() - 0.5) };
  t = performance.now();
  gpu.run(big);
  const bigFirst = performance.now() - t;
  t = performance.now();
  gpu.run(big);
  const bigMs = performance.now() - t;

  const cpuErr = maxErr(cpu.data), gpuErr = maxErr(g1.data);
  report({
    ok: cpuErr < 1e-3 && gpuErr < 1e-3, renderer, maxTex: gpu.maxTex,
    parity: { size: `${c.width}x${c.height}`, cpuErr, gpuErr, cpuMs: Math.round(cpuMs), gpuInitMs: Math.round(initMs), gpuFirstMs: Math.round(gpuFirstMs), gpuMs: Math.round(gpuMs) },
    board1024x768: { gpuFirstMs: Math.round(bigFirst), gpuMs: Math.round(bigMs) },
  });
} catch (e) {
  report({ ok: false, error: String(e && e.stack || e) });
}
