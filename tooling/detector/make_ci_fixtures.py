"""Write the small test data set that is committed in tests/fixtures/ for CI.

The full checks (npm test) use out/test-cases/, which is not in source
control. When those are missing, as on GitHub, the tests fall back to this smaller set:

tests/fixtures/parity.json + parity_input.u8 + parity_output.f32
    the exported model (src/data/model) run by PyTorch on a random-noise image, so the
    TypeScript engine can be checked without shipping a photo.
tests/fixtures/fullboard.json     the first N synthetic boards of test-cases/scoring/fullboard.json
tests/fixtures/photos.json        test-cases/scoring/photos.json (detector output only, no pixels)
    re-scored here with the current postprocess.py, so the expected results are up to date.

Rerun after retraining/exporting the model or changing the scoring (postprocess.py/.ts):
  tooling/.venv/Scripts/python -m tooling.detector.make_ci_fixtures [--ckpt <checkpoint>]
(tooling.export runs this.)
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from tooling import paths
from tooling.paths import REPO_ROOT
from tooling.detector import postprocess as P
from tooling.detector.data import normalize
from tooling.detector.make_web_cases import plain, summary
from tooling.detector.model import load_model

OUT = os.path.join(REPO_ROOT, "tests", "fixtures")
CASES = paths.SCORING_CASES
N_BOARDS = 15
PARITY_W, PARITY_H = 160, 128  # multiples of 16, not square


def write(name, data):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)


def rescore(cases):
    out = []
    for c in cases:
        res = P.score(copy.deepcopy(c["badges"]), copy.deepcopy(c["dots"]), tuple(c["size"]))
        out.append({**c, "python": plain(summary(res))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=paths.MODEL)
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)

    model = load_model(args.ckpt)
    rgb = np.random.default_rng(1).integers(0, 256, (PARITY_H, PARITY_W, 3), dtype=np.uint8)
    with torch.no_grad():
        y = model(torch.from_numpy(normalize(rgb))[None])[0].numpy().astype(np.float32)
    write("parity_input.u8", rgb.tobytes())
    write("parity_output.f32", y.tobytes())
    with open(os.path.join(OUT, "parity.json"), "w") as f:
        json.dump(dict(width=PARITY_W, height=PARITY_H, out_shape=list(y.shape)), f)
    print(f"parity case {PARITY_W}x{PARITY_H}, output {list(y.shape)}")

    boards = rescore(json.load(open(os.path.join(CASES, "fullboard.json")))[:N_BOARDS])
    write("fullboard.json", json.dumps(boards).encode())
    photos = rescore(json.load(open(os.path.join(CASES, "photos.json"))))
    write("photos.json", json.dumps(photos).encode())
    print(f"{len(boards)} synthetic boards, {len(photos)} photos -> {OUT}")
    for name in sorted(os.listdir(OUT)):
        print(f"  {name}: {os.path.getsize(os.path.join(OUT, name)) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
