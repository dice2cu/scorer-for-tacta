// UI controller: pick a photo -> analyze -> show scores and the annotated photo; taps correct it.
import type { Analysis } from "../pipeline/analyze.ts";
import { analyze } from "../pipeline/analyze.ts";
import type { Deck } from "../pipeline/decode.ts";
import { DECKS } from "../pipeline/postprocess.ts";
import type { Board } from "./board.ts";
import { cardsPerDeck, counts, makeBoard, resetEdits, tap } from "./board.ts";
import { DECK_INFO, deckSwatch } from "./decks.ts";
import { drawBoard } from "./overlay.ts";
import type { Photo } from "./photo.ts";
import { loadPhoto } from "./photo.ts";
import type { Setup } from "./setup.ts";
import { setup } from "./setup.ts";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const sections = ["start", "busy", "error", "results"] as const;

function show(id: (typeof sections)[number]) {
  for (const s of sections) $(s).hidden = s !== id;
  $("advanced").hidden = id === "busy";
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & { className?: string } = {}, ...kids: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
}

const CPU_KEY = "scorer-for-tacta:use-cpu";
const readPref = () => { try { return localStorage.getItem(CPU_KEY) === "1"; } catch { return false; } };
const writePref = (v: boolean) => { try { localStorage.setItem(CPU_KEY, v ? "1" : "0"); } catch { /* private mode */ } };

let ready: Promise<Setup>;
let current: { photo: Photo; analysis: Analysis; board: Board; ms: number } | null = null;
/** why the GPU can't be used on this device (null: it can, or it is turned off) */
let gpuProblem: string | null = null;
const gpuBox = $("use-gpu") as HTMLInputElement;

/** (Re)load the model and pick the engine; the GPU setting takes effect from here on. */
function start(useCpu: boolean) {
  const p = (ready = setup(useCpu));
  $("status").textContent = "Loading…";
  p.then(
    (s) => {
      if (p !== ready) return; // the setting was changed again meanwhile
      gpuProblem = useCpu ? null : s.gpuProblem;
      // no GPU here: show the setting as off (without saving that, so a later visit tries again)
      if (gpuProblem) { gpuBox.checked = false; gpuBox.disabled = true; }
      $("status").textContent = gpuProblem ? "Ready (this device will be a bit slower)." : "Ready.";
    },
    (e) => {
      if (p !== ready) return;
      $("status").textContent = `Couldn't load the app: ${e instanceof Error ? e.message : e}`;
    },
  );
}

async function handleFile(file: File | undefined) {
  if (!file) return;
  await score(() => loadPhoto(file));
}

async function score(getPhoto: () => Promise<Photo>) {
  current = null; // the old result is gone once a new photo is being read
  show("busy");
  $("progress").textContent = "Opening the photo…";
  try {
    const s = await ready;
    const photo = await getPhoto();
    const t0 = performance.now();
    const analysis = await analyze(s.engine, s.model, s.lib, photo, (msg) => ($("progress").textContent = msg));
    const ms = performance.now() - t0;
    if (!analysis.result.cards.length) throw new Error("No Tacta cards were found in this photo. Try a photo of the whole board from above, in good light.");
    current = { photo, analysis, board: makeBoard(analysis, s.lib.dotRadius), ms };
    show("results");
    renderResults();
    $("view").scrollIntoView({ block: "nearest" });
    window.scrollTo({ top: 0 });
  } catch (e) {
    console.error(e);
    $("error-text").textContent = e instanceof Error ? e.message : String(e);
    show("error");
  }
  $("rescore").hidden = !current;
}

function renderScores() {
  if (!current) return;
  const { board } = current;
  const c = counts(board), cards = cardsPerDeck(board);
  const decks = DECKS.filter((d) => c[d] || cards[d]).sort((a, b) => c[b] - c[a]);
  const top = decks.length ? c[decks[0]] : 0;
  const table = $("scores");
  table.replaceChildren(
    el("thead", {}, el("tr", {}, el("th", { textContent: "Player" }), el("th", { className: "num", textContent: "Dots" }), el("th", { className: "num", textContent: "Cards" }))),
    el("tbody", {}, ...decks.map((d: Deck) => {
      const name = el("span", { className: "name" }, deckSwatch(d), DECK_INFO[d].name);
      if (c[d] === top && decks.length > 1 && c[decks[1]] !== top) name.append(el("span", { className: "lead-tag", textContent: "MOST" }));
      return el("tr", {},
        el("td", {}, name),
        el("td", { className: "num dots", textContent: String(c[d]) }),
        el("td", { className: "num cards", textContent: String(cards[d]) }));
    })),
  );
  if (board.edits) table.append(el("caption", { className: "edited", textContent: "Includes your changes." }));
  ($("reset") as HTMLButtonElement).disabled = board.edits === 0;
}

function renderNotes() {
  if (!current) return;
  const { board, analysis } = current;
  const notes: string[] = [];
  const cards = cardsPerDeck(board);
  for (const d of DECKS) {
    const n = cards[d];
    if (n && n < 18) notes.push(`Found ${n} ${DECK_INFO[d].name.toLowerCase()} cards (a deck has 18). That's fine if a card was removed for an illegal placement; otherwise a card may be hidden or out of the photo.`);
    if (n > 18) notes.push(`Found ${n} ${DECK_INFO[d].name.toLowerCase()} cards, more than a deck has. Some cards' colours may have been misread; check the outlines.`);
  }
  const partial = board.dots.filter((d) => d.source === "partial" && !d.counted).length;
  if (partial) notes.push(`${partial} partly covered dot${partial > 1 ? "s" : ""} (a card edge crosses ${partial > 1 ? "them" : "it"}) ${partial > 1 ? "were" : "was"} not counted (dashed coloured rings). Tap to count ${partial > 1 ? "them" : "it"} if your group does.`);
  const review = board.dots.filter((d) => d.source === "review").length;
  if (review) notes.push(`${review} possible dot${review > 1 ? "s" : ""} not on any found card ${review > 1 ? "were" : "was"} not counted (dashed rings). Tap to count ${review > 1 ? "them" : "it"} if needed.`);
  $("notes").replaceChildren(...notes.map((t) => el("li", { textContent: t })));

  const det = [
    `Analysis took ${(current.ms / 1000).toFixed(1)} s using the ${analysis.engine}${gpuProblem ? ` (the GPU can't be used on this device: ${gpuProblem})` : ""}.`,
    `Photo ${current.photo.width}×${current.photo.height}, cards about ${Math.round(analysis.cardWidth)} px wide; analysed at ${analysis.inputSize.join("×")}.`,
    ...analysis.passes.map((p, i) => `Pass ${i + 1}: ${p.size.join("×")}, ${p.badges} badges, ${p.ms} ms`),
    analysis.icons ? `Deck symbols read on ${analysis.icons.read} cards in ${analysis.icons.ms} ms.` : "Deck symbols not read (colour only).",
    `Perspective correction: ${analysis.result.perspective_fit ? "yes" : "no (too few cards)"}.`,
    ...analysis.result.warnings,
  ];
  $("details").replaceChildren(...det.map((t) => el("li", { textContent: t })));
}

function redraw() {
  if (!current) return;
  drawBoard($("view") as HTMLCanvasElement, current.photo.bitmap, current.board, ($("marks") as HTMLInputElement).checked);
}

function renderResults() {
  renderScores();
  renderNotes();
  redraw();
}

let toastTimer = 0;
function toast(msg: string) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => (t.hidden = true), 1400);
}

$("view").addEventListener("click", (ev) => {
  if (!current || !($("marks") as HTMLInputElement).checked) return;
  const canvas = ev.currentTarget as HTMLCanvasElement;
  const r = canvas.getBoundingClientRect();
  const k = current.photo.width / r.width; // photo px per CSS px
  const p: [number, number] = [(ev.clientX - r.left) * k, (ev.clientY - r.top) * k];
  // ~14 px of finger slop on screen: when pinch-zoomed in, that is fewer CSS (and photo) px, so a
  // tap between close dots hits the one under the finger
  const zoom = window.visualViewport?.scale ?? 1;
  const what = tap(current.board, p, (14 / zoom) * k);
  if (what) {
    toast(what);
    renderScores();
    redraw();
  }
});

$("marks").addEventListener("change", redraw);
$("reset").addEventListener("click", () => {
  if (!current) return;
  resetEdits(current.board);
  renderScores();
  redraw();
});

gpuBox.checked = !readPref();
start(!gpuBox.checked);
gpuBox.addEventListener("change", () => {
  writePref(!gpuBox.checked);
  start(!gpuBox.checked);
});
// redo the current photo (e.g. after changing the GPU setting); this drops the user's taps
$("rescore").addEventListener("click", () => {
  if (current) { const { photo } = current; score(async () => photo); }
});

for (const input of [$("file"), ...document.querySelectorAll<HTMLInputElement>("[data-pick]")] as HTMLInputElement[]) {
  input.addEventListener("change", () => {
    const f = input.files?.[0];
    input.value = ""; // allow picking the same file again
    handleFile(f);
  });
}

// desktop: drop a photo anywhere
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
});

// Offline support and "add to home screen". Service workers need https; the dev server (http) is
// never cached. "?sw" turns it on for the local offline test.
if ("serviceWorker" in navigator && (location.protocol === "https:" || new URLSearchParams(location.search).has("sw"))) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("offline support unavailable:", e));
}

// test hook: tests/browser/app.mjs drives the real UI with this
(window as unknown as { __tacta: unknown }).__tacta = { handleFile, state: () => current, engine: async () => (await ready).engine.name };
