// Loads the model and card library, and picks the inference engine for this device.
import type { Engine } from "../pipeline/analyze.ts";
import { runCpu } from "../pipeline/engine-cpu.ts";
import { WebGLEngine } from "../pipeline/engine-webgl.ts";
import type { IconsJson } from "../pipeline/icons.ts";
import { makeIconReader } from "../pipeline/icons.ts";
import type { CardLibraryJson, Library } from "../pipeline/library.ts";
import { makeLibrary } from "../pipeline/library.ts";
import type { Model, ModelMeta, Tensor } from "../pipeline/model.ts";
import { makeModel } from "../pipeline/model.ts";

export interface Setup {
  model: Model;
  lib: Library;
  engine: Engine;
  gpuProblem: string | null; // why the GPU engine isn't used, if it isn't
}

// the small JSON files are bundled into the app; the 0.7 MB of weights stay a separate file (the
// build gives it a content-hashed name)
import modelJson from "../data/model/tactanet.json";
import libraryJson from "../data/cards/card_library.json";
import iconsJson from "../data/cards/icons.json";

const WEIGHTS = new URL("../data/model/tactanet.bin", import.meta.url);

export function cpuEngine(model: Model): Engine {
  return { name: "CPU", run: (t: Tensor) => runCpu(model, t) };
}

/** The GPU engine, falling back to the CPU for any run the GPU can't do (e.g. out of GPU memory on
 *  a large photo). `name` reports the engine that did the latest run. */
function withCpuFallback(gpu: WebGLEngine, cpu: Engine): Engine {
  let last: Engine = gpu;
  return {
    get name() { return last.name; },
    run(t: Tensor) {
      try {
        last = gpu;
        return gpu.run(t);
      } catch (e) {
        console.warn("GPU run failed, using the CPU for this one:", e);
        gpu.trim();
        last = cpu;
        return cpu.run(t);
      }
    },
  };
}

/** Run both engines on a small random input; a GPU whose float maths is broken must not be used. */
function gpuAgrees(gpu: WebGLEngine, model: Model): boolean {
  const n = 48 * 64;
  const input: Tensor = { c: 3, h: 48, w: 64, data: new Float32Array(3 * n) };
  let seed = 1;
  for (let i = 0; i < input.data.length; i++) input.data[i] = ((seed = (seed * 16807) % 2147483647) / 2147483647) * 4 - 2;
  const a = gpu.run(input).data, b = runCpu(model, input).data;
  let err = 0;
  for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]));
  return err < 1e-2;
}

export async function setup(preferCpu = false): Promise<Setup> {
  const r = await fetch(WEIGHTS);
  if (!r.ok) throw new Error(`couldn't load the model (${r.status})`);
  const model = makeModel(modelJson as unknown as ModelMeta, await r.arrayBuffer());
  const lib = makeLibrary(libraryJson as unknown as CardLibraryJson);
  lib.icons = makeIconReader(iconsJson as unknown as IconsJson);
  let engine: Engine = cpuEngine(model);
  let gpuProblem: string | null = preferCpu ? "turned off in settings" : null;
  if (!preferCpu) {
    try {
      const canvas = document.createElement("canvas");
      const gpu = new WebGLEngine(model, canvas);
      if (gpuAgrees(gpu, model)) engine = withCpuFallback(gpu, engine);
      else gpuProblem = "the GPU's results didn't match the reference";
    } catch (e) {
      gpuProblem = e instanceof Error ? e.message : String(e);
    }
  }
  return { model, lib, engine, gpuProblem };
}
