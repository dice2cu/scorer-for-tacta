"""Generate a synthetic dataset of boards + ground truth (multi-process).

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.synth.make_dataset --split train --n 6000
  tooling/.venv/Scripts/python -m tooling.synth.make_dataset --split val --n 300 --seed-offset 1000000
Output: out/boards/<split>/<index>.jpg + <index>.json
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from tooling import paths
from tooling.paths import REPO_ROOT

OUT = paths.BOARDS
_assets = None


def _init():
    global _assets
    cv2.setNumThreads(1)
    from tooling.synth.render import Assets
    _assets = Assets()


def _one(job):
    split, idx, seed, frame_all, size, recolour = job
    from tooling.synth.board import make_sample
    path = os.path.join(OUT, split, f"{idx:06d}")
    if os.path.exists(path + ".json"):
        return 0
    rng = np.random.default_rng(seed)
    img, gt = make_sample(rng, _assets, out_w=size[0], out_h=size[1], frame_all=frame_all, recolour=recolour)
    # JPEG quality is itself an augmentation. A downscaled phone photo shows few artifacts, so
    # most boards get mild compression; some get heavy compression (small web images).
    q = int(rng.integers(70, 98)) if rng.random() < 0.85 else int(rng.integers(45, 70))
    cv2.imwrite(path + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    with open(path + ".json", "w") as f:
        json.dump(gt, f)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--frame-all", action="store_true", help="whole board in frame (real-photo-like test set)")
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--recolour", type=float, default=0.0,
                    help="share of cards printed in another deck's colour (icon and label kept); training only")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(OUT, args.split), exist_ok=True)
    size = tuple(int(v) for v in args.size.split("x"))
    jobs = [(args.split, i, args.seed_offset + i, args.frame_all, size, args.recolour) for i in range(args.n)]
    t = time.time()
    done = 0
    with Pool(args.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=4)):
            done += r
            if (k + 1) % 250 == 0:
                print(f"{args.split}: {k + 1}/{args.n}  ({(time.time() - t) / (k + 1):.2f}s/board wall)", flush=True)
    print(f"{args.split}: generated {done} new boards in {time.time() - t:.0f}s -> {os.path.join(OUT, args.split)}")


if __name__ == "__main__":
    main()
