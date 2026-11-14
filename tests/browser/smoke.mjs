// Quick check of the built app in headless Edge/Chrome that needs no test photos (runs in CI):
// it starts, sets up an engine, runs the whole pipeline on a generated image without cards, and
// the GPU setting switches engines.
//   node tests/browser/smoke.mjs [--gpu swiftshader]
import { launch, serve } from "./browser.mjs";

const args = process.argv.slice(2);
const gpu = args.includes("--gpu") ? args[args.indexOf("--gpu") + 1] : "default";
const { server, origin } = await serve();
const b = await launch({ gpu });
let failed = false;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); failed ||= !ok; };
const settled = `document.getElementById("status").textContent.startsWith("Ready") || document.getElementById("status").textContent.startsWith("Couldn't")`;
const engine = `window.__tacta.engine()`;
try {
  await b.phone();
  await b.goto(`${origin}/`);
  await b.waitFor(settled);
  const status = await b.eval(`document.getElementById("status").textContent`);
  check(status.startsWith("Ready"), `start: ${status}`);
  const gpuBox = await b.eval(`(() => { const c = document.getElementById("use-gpu"); return c.checked ? "on" : c.disabled ? "off (no GPU here)" : "off"; })()`);
  check(true, `engine: ${await b.eval(engine)}, GPU setting ${gpuBox}`);

  // an empty table (smooth gradient): the whole pipeline runs and reports that no cards were found
  const scoreBlank = `(async () => {
    const c = new OffscreenCanvas(1200, 900), g = c.getContext("2d");
    const grad = g.createRadialGradient(500, 400, 50, 600, 450, 800);
    grad.addColorStop(0, "#6b4a32"); grad.addColorStop(1, "#2a1d14");
    g.fillStyle = grad; g.fillRect(0, 0, 1200, 900);
    const blob = await c.convertToBlob({ type: "image/jpeg" });
    await window.__tacta.handleFile(new File([blob], "blank.jpg", { type: "image/jpeg" }));
    return document.getElementById("error").hidden ? null : document.getElementById("error-text").textContent;
  })()`;
  let err = await b.eval(scoreBlank);
  check(err?.startsWith("No Tacta cards"), `no-card image: ${err}`);

  await b.eval(`(() => { const c = document.getElementById("use-gpu"); c.checked = false; c.dispatchEvent(new Event("change")); })()`);
  await b.waitFor(settled);
  const cpu = await b.eval(engine);
  check(cpu === "CPU", `GPU off: ${cpu}`);
  err = await b.eval(scoreBlank);
  check(err?.startsWith("No Tacta cards"), `no-card image on the CPU: ${err}`);

  const exceptions = b.logs.filter((l) => l.startsWith("EXCEPTION"));
  check(exceptions.length === 0, `no uncaught exceptions${exceptions.length ? ":\n  " + exceptions.join("\n  ") : ""}`);
  if (b.logs.length) console.log("browser console:\n  " + b.logs.join("\n  "));
} finally {
  await b.close();
  server.close();
}
process.exit(failed ? 1 : 0);
