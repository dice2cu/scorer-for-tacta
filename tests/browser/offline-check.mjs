// Offline support: load the app once (its service worker caches everything), cut the network,
// reload, and score a photo. Also checks that the page offers what phones need to install it.
//   node tests/browser/offline-check.mjs [photo]
// The photo defaults to the first one in training-data/games/; without one (as in CI) the scoring
// step is skipped.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { launch, serve, ROOT } from "./browser.mjs";

const GAMES = "training-data/games";
const first = existsSync(join(ROOT, GAMES))
  ? readdirSync(join(ROOT, GAMES)).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)).sort()[0] : undefined;
const photo = process.argv[2] ?? (first ? `${GAMES}/${first}` : null);
const { server, origin } = await serve();
const b = await launch();
let failed = false;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); failed ||= !ok; };
try {
  await b.phone();
  await b.goto(`${origin}/?sw`);
  await b.eval(`navigator.serviceWorker.ready.then(() => true)`);
  await b.goto(`${origin}/?sw`); // the second load is served by the worker
  check(await b.eval(`!!navigator.serviceWorker.controller`), "service worker controls the page");
  const manifest = await b.eval(`fetch(document.querySelector('link[rel=manifest]').href).then((r) => r.json())`);
  for (const icon of manifest.icons) check(await b.eval(`fetch(${JSON.stringify(icon.src)}).then((r) => r.ok)`), `manifest icon ${icon.src}`);
  check(await b.eval(`fetch(document.querySelector('link[rel=apple-touch-icon]').href).then((r) => r.ok)`), "apple-touch-icon");

  // keep the photo in the page before going offline (the test photos aren't part of the app)
  const bytes = photo ? readFileSync(join(ROOT, photo)).toString("base64") : null;
  await b.send("Network.enable");
  await b.send("Network.emulateNetworkConditions", { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  await b.goto(`${origin}/?sw`);
  await b.waitFor(`document.getElementById("status").textContent.startsWith("Ready") || document.getElementById("status").textContent.startsWith("Couldn't")`);
  const status = await b.eval(`document.getElementById("status").textContent`);
  check(status.startsWith("Ready"), `offline start: ${status}`);
  if (!photo) console.log("skip offline scoring (no photo)");
  else {
    const counts = await b.eval(`(async () => {
      const blob = await (await fetch("data:image/jpeg;base64,${bytes}")).blob();
      await window.__tacta.handleFile(new File([blob], "photo.jpg", { type: "image/jpeg" }));
      const s = window.__tacta.state();
      if (!s) return null;
      const c = {};
      for (const d of s.board.dots) if (d.counted) c[d.deck] = (c[d.deck] || 0) + 1;
      return c;
    })()`);
    check(counts !== null, `offline scoring: ${JSON.stringify(counts)}`);
  }
  if (b.logs.length) console.log("browser console:\n  " + b.logs.join("\n  "));
} finally {
  await b.close();
  server.close();
}
process.exit(failed ? 1 : 0);
