"""Score the photos in training-data/games/ and compare with the known counts in counts.json.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.report [--model <checkpoint>] [--casts]

Photos without known counts are scored but not judged. --casts also scores every counted photo
with warm, cool and magenta colour casts (phones get white balance wrong; decks must still be told
apart). tooling.train runs this for the new model next to the shipped one.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from tooling import paths
from tooling.detector.model import load_model
from tooling.detector.photo import analyze

# channel gains (B, G, R) in linear light
CASTS = {"warm": (0.72, 1.0, 1.25), "cool": (1.3, 1.0, 0.8), "magenta": (1.15, 0.85, 1.15)}


def known_counts():
    if not os.path.exists(paths.COUNTS):
        return {}
    return {k: v for k, v in json.load(open(paths.COUNTS)).items() if not k.startswith("_")}


def with_cast(img, gains):
    lin = (img.astype(np.float32) / 255) ** 2.2
    lin = np.clip(lin * np.array(gains, np.float32), 0, 1)
    return np.round(lin ** (1 / 2.2) * 255).astype(np.uint8)


def cases(casts=False):
    """(label, image loader, known counts or None) for every photo in games/."""
    counts = known_counts()
    out = []
    for path in paths.photos(paths.GAMES):
        name = os.path.basename(path)
        known = counts.get(name)
        out.append((name, lambda p=path: cv2.imread(p), known))
        if casts and known is not None:
            for cast, gains in CASTS.items():
                out.append((f"{name} ({cast})", lambda p=path, g=gains: with_cast(cv2.imread(p), g), known))
    return out


def judge(found, known):
    """Dots off (0 = exact) or None when the counts aren't known."""
    if known is None:
        return None
    return sum(abs(found.get(d, 0) - known.get(d, 0)) for d in set(found) | set(known))


def report(models, casts=False, out_file=None):
    """models: {label: checkpoint path}. Prints one row per photo and a summary per model."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {label: load_model(ckpt, dev) for label, ckpt in models.items()}
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    todo = cases(casts)
    if not todo:
        emit(f"no photos in {paths.GAMES}")
        return {}
    totals = {label: [0, 0, 0] for label in loaded}  # exact, judged, dots off
    width = max(len(name) for name, _, _ in todo)
    emit(f"{'photo':{width}}  " + "  ".join(f"{label:>24}" for label in loaded) + "  known")
    for name, load, known in todo:
        img = load()
        cells = []
        for label, model in loaded.items():
            found = analyze(model, img, dev)["result"]["counts"]
            off = judge(found, known)
            if off is not None:
                t = totals[label]
                t[0] += off == 0
                t[1] += 1
                t[2] += off
            mark = "" if off is None else ("ok" if off == 0 else f"off {off}")
            cells.append(f"{' '.join(f'{d[:2]}{n}' for d, n in sorted(found.items())) or 'none':>17} {mark:>6}")
        emit(f"{name:{width}}  " + "  ".join(cells) + "  " + ("" if known is None else
             (" ".join(f"{d[:2]}{n}" for d, n in sorted(known.items())) or "none")))
    emit()
    for label, (exact, judged, off) in totals.items():
        emit(f"{label}: {exact}/{judged} photos exact, {off} dots off in total")
    if out_file:
        with open(out_file, "w") as f:
            f.write("\n".join(lines) + "\n")
    return {label: dict(exact=e, judged=j, dots_off=o) for label, (e, j, o) in totals.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=paths.MODEL, help="checkpoint (default: the shipped model)")
    ap.add_argument("--casts", action="store_true", help="also score counted photos with colour casts")
    args = ap.parse_args(argv)
    report({"model" if args.model != paths.MODEL else "shipped": args.model}, casts=args.casts)


if __name__ == "__main__":
    main()
