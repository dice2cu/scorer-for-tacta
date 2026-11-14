# Training the model

You add photos to `training-data/` and run one command. Everything the scripts make goes to `out/`,
which you can ignore and delete at any time.

```
training-data/
  decks/          photos of decks laid out in a grid (used to make the training images)
  games/          photos of finished games (used to test each model)
    counts.json   optional: the known dot counts of some of those photos
```

## Photos to add

**`decks/`**: one deck of 18 cards in a grid, 3 rows (circle, square, triangle) × 6 columns
(values 1 to 6). Or two decks of 36 cards in 6 rows. The cards may show their fronts or their
backs. Take them straight from above in soft, even light, with the cards not touching. Any glare
in these photos ends up in the training images. More decks under more lights give more realistic
training images.

**`games/`**: finished games, photographed the way players will. Variety matters more than number:

- different tables, rooms and light (daylight, warm lamps, one overhead lamp with glare)
- every deck colour, especially red and pink together (they look alike in photos)
- crowded boards, partly covered dots, clutter around the edges, steep angles, blur

Images found online work too.

**`games/counts.json`**: for any photo whose dots you have counted by hand, the visible dots per
colour. Partly covered dots don't count; `{}` means a photo without any cards. Photos not listed
are still scored, just not checked.

```json
{
  "kitchen_table.jpg": { "red": 45, "pink": 48 },
  "empty_table.jpg": {}
}
```

## Training

Set up Python once (see [tooling/README.md](tooling/README.md#setup)), then from the repo root:

```sh
tooling/.venv/Scripts/python -m tooling.train
```

This can take several hours, mostly in two steps: rendering the 8,000 training
images (about an hour) and training (about 70 minutes on the GPU). It:

1. makes recolourable card textures from the photos in `decks/`;
2. renders synthetic games from them, each image with exact labels for every card and dot;
3. trains, starting from the shipped model;
4. scores the new model and the shipped model side by side, on synthetic test games and on the
   photos in `games/`.

If it's interrupted, run the same command again and it carries on from where it stopped. Steps whose
inputs haven't changed are skipped. `--help` lists the options. For example, `--batch 6` helps if
the GPU runs out of memory.

The real photos are never used for training. Labelling every dot in a photo by hand would take
hours, so the network learns only from the synthetic games. The photos in `games/` are the test
that counts: the synthetic test games have been misleading before.

To score the photos in `games/` without training, for example after adding new ones, run:

```sh
tooling/.venv/Scripts/python -m tooling.report
```

## Putting the new model into the app

If the new model does better on the photos in `games/`:

```sh
tooling/.venv/Scripts/python -m tooling.export
npm test
npm run test:browser
```

`tooling.export` writes the app's model (`src/data/model/`), the weights the next training starts
from (`tooling/detector/tactanet.pt`), and the test data (`tests/fixtures/`). Commit those. To go
back, check the old versions out of git.

After changing the scoring code (`postprocess.py`/`.ts`), run `tooling.export shipped` to bring the
test data up to date.

## Improving the model

When the model gets a kind of photo wrong, the fix is usually in the synthetic games:
`tooling/synth/board.py` decides what the training images contain (tables, light, glare,
clutter, camera angles). [tooling/README.md](tooling/README.md) describes every script.

## How the shipped model was trained

v7 is a chain of fine-tunes, each on 8,000 new synthetic games:

| Version | What changed |
|---|---|
| v3 | base model |
| v4 | deck colours measured from real cards; clutter |
| v5 | white balance kept within what phones produce |
| v6 | sharp card textures; reads red better, but fires on wood grain |
| v7 | glinting wood grain, strong glare |
| v8 (rejected) | taught to read the deck icon instead of colour. At ~100 px per card the icon is ~7 px, too small for this network; real photos went from 16/17 to 10/17 exact. The app reads the icons from the full-resolution photo instead. |
