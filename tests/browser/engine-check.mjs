// GPU and CPU engines vs PyTorch in a real browser, plus GPU timing (page: engine.html).
//   node tests/browser/engine-check.mjs [--gpu swiftshader]
import { launch, serveDev } from "./browser.mjs";

const args = process.argv.slice(2);
const gpu = args.includes("--gpu") ? args[args.indexOf("--gpu") + 1] : "default";
const { server, reports, origin } = await serveDev();
const b = await launch({ gpu });
try {
  await b.goto(`${origin}/tests/browser/engine.html`);
  const t0 = Date.now();
  while (!reports.length && Date.now() - t0 < 120000) await new Promise((r) => setTimeout(r, 200));
  const r = reports[0] ?? { ok: false, error: "no report" };
  console.log(JSON.stringify(r, null, 2));
  process.exitCode = r.ok ? 0 : 1;
} finally {
  await b.close();
  server.close();
}
