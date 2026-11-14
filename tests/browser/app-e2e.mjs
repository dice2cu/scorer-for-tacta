// End-to-end: drive the real app in headless Edge/Chrome on the photos in training-data/games/,
// compare with the known counts (training-data/games/counts.json) and with the Python pipeline
// (out/test-cases/scoring/photos.json), and save phone-sized screenshots.
//   node tests/browser/app-e2e.mjs [--gpu swiftshader] [--cpu] [photo ...]
// Output: out/browser/<photo>.png (whole page), <photo>_overlay.jpg (the annotated photo) and
// <photo>_cards.json (what was found; tooling/detector/check_icons.py reads it).
import { mkdirSync, readFileSync, readdirSync, writeFileSync, existsSync } from "node:fs";
import { basename, join, relative } from "node:path";
import { launch, serve, ROOT } from "./browser.mjs";

const args = process.argv.slice(2);
const gpu = args.includes("--gpu") ? args[args.indexOf("--gpu") + 1] : "default";
const GAMES = join(ROOT, "training-data", "games");
const countsFile = join(GAMES, "counts.json");
const known = existsSync(countsFile) ? JSON.parse(readFileSync(countsFile, "utf8")) : {};
// photos can be given on the command line; by default, every photo in training-data/games/
const given = args.filter((a) => /\.(jpe?g|png|webp)$/i.test(a)).map((a) => relative(ROOT, join(ROOT, a)));
const PHOTOS = (given.length ? given : existsSync(GAMES)
  ? readdirSync(GAMES).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)).sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" }))
    .map((f) => relative(ROOT, join(GAMES, f)))
  : []).map((p) => p.replace(/\\/g, "/"));
if (!PHOTOS.length) { console.log("No photos in training-data/games/."); process.exit(0); }
const expected = Object.fromEntries(PHOTOS.filter((p) => known[basename(p)]).map((p) => [p, known[basename(p)]]));
let totalErr = 0, exact = 0, judged = 0;
const casesFile = join(ROOT, "out", "test-cases", "scoring", "photos.json");
const python = existsSync(casesFile) ? Object.fromEntries(JSON.parse(readFileSync(casesFile, "utf8")).map((c) => [c.name, c.python])) : {};
const OUT = join(ROOT, "out", "browser");
mkdirSync(OUT, { recursive: true });

const { server, origin } = await serve();
const b = await launch({ gpu });
try {
  await b.phone();
  if (args.includes("--cpu")) {
    await b.goto(`${origin}/`);
    await b.eval(`localStorage.setItem("scorer-for-tacta:use-cpu", "1")`);
  }
  await b.goto(`${origin}/`);
  await b.waitFor(`document.getElementById("status").textContent.startsWith("Ready") || document.getElementById("status").textContent.startsWith("Couldn't")`);
  console.log("status:", await b.eval(`document.getElementById("status").textContent`));
  writeFileSync(join(OUT, "start.png"), await b.screenshot());
  for (const p of PHOTOS) {
    const name = p.split("/").pop();
    const r = await b.eval(`(async () => {
      const blob = await (await fetch("/${p}")).blob();
      await window.__tacta.handleFile(new File([blob], "${name}", { type: blob.type }));
      const s = window.__tacta.state();
      if (!s) return { error: document.getElementById("error-text").textContent };
      const counts = {};
      for (const d of s.board.dots) if (d.counted) counts[d.deck] = (counts[d.deck] || 0) + 1;
      const cards = {};
      for (const c of s.board.cards) cards[c.deck] = (cards[c.deck] || 0) + 1;
      const a = s.analysis;
      const cardInfo = a.result.cards.map((c) => ({ deck: c.deck, design: c.design, visible: c.visible, width: c.width,
        corners: c.corners, score: a.badges[c.badge].score, deck_p: a.badges[c.badge].deck_p, value_p: a.badges[c.badge].value_p,
        badge_deck: a.badges[c.badge].deck, deck_probs: a.badges[c.badge].deck_probs, value_probs: a.badges[c.badge].value_probs,
        badge_xy: [a.badges[c.badge].x, a.badges[c.badge].y], icon_xy: a.badges[c.badge].icon, suit: a.badges[c.badge].suit, side: a.badges[c.badge].side }));
      return { counts, cards, cardInfo, warnings: a.result.warnings, size: a.inputSize, scale: a.scale, decoded: a.decoded, ms: Math.round(s.ms), engine: s.analysis.engine, cardWidth: Math.round(s.analysis.cardWidth),
               passes: s.analysis.passes, icons: s.analysis.icons, overlay: document.getElementById("view").toDataURL("image/jpeg", 0.85) };
    })()`);
    if (r.error) {
      console.log(`${name}: ERROR ${r.error}`);
      const want = expected[p];
      if (want) { // "no cards found" is the right answer for a photo without cards
        const ok = Object.keys(want).length === 0;
        judged++; exact += ok ? 1 : 0;
        if (!ok) totalErr += Object.values(want).reduce((s, v) => s + v, 0);
        console.log(`    expected ${JSON.stringify(want)}  ${ok ? "EXACT" : "MISSED"}`);
      }
      continue;
    }
    writeFileSync(join(OUT, name.replace(/\.\w+$/, "_overlay.jpg")), Buffer.from(r.overlay.split(",")[1], "base64"));
    writeFileSync(join(OUT, name.replace(/\.\w+$/, "_cards.json")), JSON.stringify({ size: r.size, scale: r.scale, decoded: r.decoded, cards: r.cardInfo, warnings: r.warnings }));
    writeFileSync(join(OUT, name.replace(/\.\w+$/, ".png")), await b.screenshot());
    const py = python[name];
    const same = py && JSON.stringify(Object.entries(r.counts).sort()) === JSON.stringify(Object.entries(py.counts).filter(([, v]) => v).sort());
    console.log(`${name}: ${JSON.stringify(r.counts)} cards ${JSON.stringify(r.cards)}  ${r.ms} ms (${r.engine}), card ${r.cardWidth}px` +
      `  passes ${r.passes.map((q) => `${q.size.join("x")}:${q.ms}ms`).join(", ")}` +
      (r.icons ? `, icons ${r.icons.read}:${r.icons.ms}ms` : "") + (py ? `\n    python ${JSON.stringify(py.counts)} ${same ? "SAME" : "DIFFERENT"}` : ""));
    const want = expected[p];
    if (want) {
      const decks = new Set([...Object.keys(want), ...Object.keys(r.counts)]);
      const err = [...decks].reduce((s, d) => s + Math.abs((r.counts[d] || 0) - (want[d] || 0)), 0);
      totalErr += err; judged++; if (err === 0) exact++;
      console.log(`    expected ${JSON.stringify(want)}  ${err === 0 ? "EXACT" : `OFF BY ${err}`}`);
    }
  }
  if (judged) console.log(`\nphotos with known counts: ${exact}/${judged} exact, total dots off ${totalErr}`);
  if (b.logs.length) console.log("browser console:\n  " + b.logs.join("\n  "));
} finally {
  await b.close();
  server.close();
}
