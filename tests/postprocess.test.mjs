// The TypeScript scoring must reproduce the Python pipeline on the same detector output
// (cases from tooling/detector/make_web_cases.py in out/test-cases/scoring/, or the committed subset in
// tests/fixtures/).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeLibrary } from "../src/pipeline/library.ts";
import { score } from "../src/pipeline/postprocess.ts";
import { root, scoringCases } from "./data.mjs";

const cases = scoringCases();
const lib = makeLibrary(JSON.parse(readFileSync(new URL("src/data/cards/card_library.json", root), "utf8")));

function summary(res) {
  return {
    counts: res.counts, cards_per_deck: res.cards_per_deck, scale_factor: res.scale_factor,
    perspective_fit: res.perspective_fit, n_review: res.review.length, n_partial: res.partial.length,
    designs: res.cards.map((c) => `${c.deck} ${c.design} ${c.side}`).sort(),
  };
}

function compare(name, file) {
  const all = cases.load(file);
  let same = 0, exactTs = 0, exactPy = 0, ms = 0;
  const diffs = [];
  for (const c of all) {
    const t0 = performance.now();
    const res = score(lib, structuredClone(c.badges), structuredClone(c.dots), c.size);
    ms += performance.now() - t0;
    const s = summary(res);
    const py = c.python;
    py.scale_factor = Math.round(py.scale_factor * 1e6) / 1e6;
    s.scale_factor = Math.round(s.scale_factor * 1e6) / 1e6;
    try { assert.deepEqual(s, py); same++; } catch { diffs.push({ name: c.name, ts: s.counts, py: py.counts, k: [s.scale_factor, py.scale_factor] }); }
    if (c.gt) {
      const keys = new Set([...Object.keys(c.gt), ...Object.keys(s.counts), ...Object.keys(py.counts)]);
      if ([...keys].every((k) => (s.counts[k] ?? 0) === (c.gt[k] ?? 0))) exactTs++;
      if ([...keys].every((k) => (py.counts[k] ?? 0) === (c.gt[k] ?? 0))) exactPy++;
    }
  }
  console.log(`  ${cases.source} ${name}: ${same}/${all.length} identical to Python, ${(ms / all.length).toFixed(0)} ms/board` +
    (all[0].gt ? `; exact counts TS ${exactTs}, Python ${exactPy}` : ""));
  for (const d of diffs.slice(0, 10)) console.log("   differs:", JSON.stringify(d));
  return { same, total: all.length };
}

test("scoring matches Python on synthetic boards", () => {
  const r = compare("fullboard", "fullboard.json");
  assert.ok(r.same >= r.total - 2, `${r.total - r.same} boards differ`);
});

test("scoring matches Python on real photos", () => {
  const r = compare("photos", "photos.json");
  assert.equal(r.same, r.total);
});
