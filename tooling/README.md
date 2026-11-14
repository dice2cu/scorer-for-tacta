# Build-time tooling (Python)

Offline scripts that produce data the TypeScript app consumes.
Nothing here ships with the app. To train a new model, see [TRAINING.md](../TRAINING.md): it
comes down to `tooling.train` and `tooling.export`, which run the scripts below in order.

## Setup

```sh
python -m venv tooling/.venv
tooling/.venv/Scripts/python -m pip install -r tooling/requirements.txt
```

The commands in these docs use the Windows path `tooling/.venv/Scripts/python`; on macOS and
Linux it is `tooling/.venv/bin/python`. A CUDA GPU makes training practical (PyTorch's CPU
build works, but slowly).

Run scripts as modules from the repo root, for example
`tooling/.venv/Scripts/python -m tooling.card_library.build_card_library ...`.

Every script finds its files through `paths.py`: the photos in `training-data/`, everything made
from them in `out/` (neither is in source control).

## Commands

- `train.py`: card textures, synthetic boards, training, and a side-by-side report of the new and
  the shipped model. Each step is skipped when its inputs haven't changed (fingerprints).
- `report.py`: the full pipeline (as the app runs it, `detector/photo.py`) on the photos in
  `training-data/games/`, checked against `counts.json`; `--casts` adds simulated colour casts.
- `export.py`: a trained model into the app (`src/data/model/`), its weights into
  `detector/tactanet.pt` (the starting point for the next training), and the test data.

## card_library/

Builds `src/data/cards/` (the 18 Tacta card designs: dot positions, badge/icon geometry,
layout masks) from photos of a full deck laid out in a 3x6 grid (or two decks in 6x6), such as
those in `training-data/decks/`. Only needed if the card designs change; the library is in source
control. Debug output: `out/card-library/`.

- `build_card_library.py`: detect, rectify and measure every card; cross-check counts,
  positions and front/back orientation across all photos; write the library.
- `render_card_library.py`: draw the saved library (front and mirrored back) for review.

## synth/

Synthetic training boards with exact ground truth.

- `card_assets.py`: split each rectified card photo into soft layers (deck colour, white ink)
  so any card can be re-rendered in any deck colour; icons per deck from the app's templates
  (`src/data/cards/icons.json`). Sources: every photo in `training-data/decks/`. Output:
  `out/card-assets/`.
- `render.py`: recolour/compose a card image from the layers. `DECK_RGB` holds the printed deck
  colours as measured in photos of the real cards (white-ink balanced). Red and pink are close, so
  re-measure if a new print run looks different.
- `board.py`: rule-following card placement (45-degree relative rotations, badge zones never
  covered; about half the boards get a few partly covered dots from slightly misaligned cards),
  tilted pinhole camera (up to ~58 degrees), tables, clutter (box art, rug/floor/legs and shoes
  past the table edge, dark gadgets with light buttons), lighting incl. white balance (kept
  within what phones produce, or red and pink become indistinguishable) and mixed-colour light,
  glare (up to strong veils that lift black to ~0.8 of white, as under an overhead lamp), wood
  grain with fine streaks and pores that glint under glare (without them a sharp detector reads
  real wood grain as thousands of dots), blur, noise. Card textures are area-downsampled to their
  on-image size and most boards stay sharp, like a phone photo shrunk to the model's scale.
  `python -m tooling.synth.board` writes previews to `out/synth-preview/`.
  Icon-first colour: with `recolour` > 0 some cards are printed in (most of the way towards)
  another deck's colour but keep their own icon and label, so the network must read the icon
  under the badge rather than the colour (red and pink, for one, look alike in photos). Their dots
  are labelled with the colour shown (a dot can't see the icon). Off by default; it was the v8
  experiment (see TRAINING.md).
- `make_dataset.py`: multi-process generation into `out/boards/<split>/`. `tooling.train` makes
  the `train`, `val` and `test` splits.

## detector/

Tiny CenterNet-style network (~183k params, ~5 GMAC for a full board at ~100 px/card) that
finds badges (shape, value, side, deck, icon direction), visible dots (with owner colour) and
partly covered dots (shown in the app, never counted). Sized for in-browser inference on phones
with the app's own WebGL2/CPU engine (`src/pipeline/engine-*.ts`).

- `model.py`, `data.py`, `train.py`: model, targets, training (`--out` folder).
  `--init <ckpt>` fine-tunes from an earlier checkpoint (new output heads start fresh);
  splits combine with `+`. How the shipped model (v7) was trained: see TRAINING.md.
  `tactanet.pt`: the shipped model's weights.
- `photo.py`: photo -> scores with the same steps as the app (`src/pipeline/analyze.ts`).
- `decode.py`: output maps -> badges and dots (the app mirrors this).
- `icons.py`: reads each card's deck icon (the shape under the badge) from the full-resolution
  photo: a card-aligned patch around the detector's icon point, the icon blob found in it, then
  template matching of the 6 icons (intensity + gradient correlation). The primary deck evidence;
  the colour readings only back it up (white balance shifts colours, not shapes). Real photos:
  every card read right except one blurry orange icon (score margin 0.05). Skipped for cards under
  150 px (the synthetic test boards, 35-95 px, so they don't exercise it).
  The templates are in source control (`src/data/cards/icons.json`); `src/pipeline/icons.ts`
  mirrors every step, and
  `check_icons.py` compares the two on the real photos after an app-e2e run.
- `postprocess.py`: classical scoring: whole-board perspective (`perspective.py`, with a prior
  against shear when all cards are square to each other), size filter (card size taken from
  confident badges, since carpet/clothes/box art can yield more junk badges than real cards), per-card
  value/side/rotation fit from dot positions (explaining-away, one design per deck), card deck
  from its icon (when read) backed by the colour readings, which count as one piece of evidence
  (their mean); without an icon, by colour vote (badge + dots; dots don't choose between red and
  pink, the badge does); stacked dots go to the card whose colour the dot looks most like; then one
  design per deck again for colours (two "red square-4"s: the less likely one moves to the deck
  in play that lacks it, if its badge allows that colour), dot counting
  policy, clutter rules (weak cards without dots, weak cards standing alone away from the board,
  colours with only a few cards; a few "pink" cards in a game with red count as red and vice
  versa), warnings and review list.
- `evaluate.py`: metrics on held-out synthetic boards, including exact per-colour counts of the
  full pipeline (`--split test` for the boards `tooling.train` makes); `--save-cache` keeps
  decoded outputs (with icon readings unless `--no-icons`).
- `compare_runs.py`, `deck_confusion.py`: per-board comparison of two cached runs; which card
  colours end up wrong.
- `tune_policy.py`: re-score cached detector outputs under different counting thresholds.
- `run_photo.py`: run on a real photo and draw detections into `out/detections/`.
- `export_web.py`: fold BatchNorm and write the app's model (`src/data/model/tactanet.json` +
  `.bin`), plus a PyTorch parity case in `out/test-cases/parity/`.
- `make_web_cases.py`: decoded detector output + Python scoring for the synthetic test boards
  (`--cache`, an `evaluate.py --save-cache` file) and the photos in `training-data/games/`, in
  `out/test-cases/scoring/`; `npm test` checks the TypeScript port against them.
  `postprocess.py` and `src/pipeline/postprocess.ts` must stay in step.
- `make_ci_fixtures.py`: the small committed test set in `tests/fixtures/` that CI uses (parity
  case on a noise image, the first 15 synthetic boards, the photo cases, re-scored with the current
  `postprocess.py`). Re-run after `export_web.py` / `make_web_cases.py` or any scoring change.
- `scale_probe.py`: detection/fit quality of real photos across input scales.

`node tests/browser/app-e2e.mjs` runs the app itself on every photo in `training-data/games/`
and checks the known counts.

GPU memory: training uses `--batch 10` by default; use a smaller batch if the GPU runs out of
memory.
