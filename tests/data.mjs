// Test data: the full sets in out/test-cases/ (made by tooling.export, not in source control) when
// present, otherwise the small committed set in tests/fixtures/ (also made by tooling.export).
// TACTA_TEST_DATA=fixtures forces the committed set (to check what CI runs).
import { existsSync, readFileSync } from "node:fs";

export const root = new URL("../", import.meta.url);
const fixtures = new URL("tests/fixtures/", root);
const useFull = (dir) => process.env.TACTA_TEST_DATA !== "fixtures" && existsSync(dir);

const read = (dir, name) => readFileSync(new URL(name, dir));

/** Scoring cases (fullboard.json, photos.json): { source, load(name) }. */
export function scoringCases() {
  const full = new URL("out/test-cases/scoring/", root);
  const dir = useFull(full) ? full : fixtures;
  return { source: dir === full ? "out/test-cases/scoring" : "tests/fixtures", load: (name) => JSON.parse(read(dir, name).toString("utf8")) };
}

/** The engine parity case: input RGB bytes and PyTorch's output for the exported model. */
export function parityCase() {
  const full = new URL("out/test-cases/parity/", root);
  if (useFull(full)) {
    return { source: "out/test-cases/parity", meta: JSON.parse(read(full, "case.json")), rgb: read(full, "input_rgb.u8"), out: read(full, "output.f32") };
  }
  return { source: "tests/fixtures", meta: JSON.parse(read(fixtures, "parity.json")), rgb: read(fixtures, "parity_input.u8"), out: read(fixtures, "parity_output.f32") };
}
