"""Train a new model from the photos in training-data/. See TRAINING.md.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.train [--boards 8000] [--epochs 24] [--from-scratch]

  1. card textures from the photos in training-data/decks/          -> out/card-assets/
  2. synthetic boards rendered from them: train, val and test       -> out/boards/
  3. training, starting from the shipped model                      -> out/runs/<run>/
  4. the new and the shipped model on the synthetic test boards and on the photos in
     training-data/games/, side by side                             -> out/runs/<run>/report.txt

Every step is skipped when its output is already up to date, so after an interruption the same
command carries on. A run's name is a fingerprint of everything it was trained from; change the
photos, the board generator or the settings and a new run starts. To put a model into the app:
  tooling/.venv/Scripts/python -m tooling.export
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import torch

from tooling import paths

SYNTH_SOURCES = ["tooling/synth/*.py", "tooling/card_library/build_card_library.py",
                 "src/data/cards/card_library.json", "src/data/cards/icons.json"]
SPLITS = {  # split: (boards, seed offset, whole board in frame)
    "train": (None, 0, False),  # count from --boards
    "val": (300, 10_000_000, False),
    "test": (150, 20_000_000, True),
}


def fingerprint(*parts, files=()):
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode())
    for f in files:
        with open(f, "rb") as fh:
            h.update(hashlib.sha1(fh.read()).digest())
    return h.hexdigest()[:10]


def source_files(patterns):
    return sorted(f for p in patterns for f in glob.glob(os.path.join(paths.REPO_ROOT, p)))


def up_to_date(folder, fp):
    stamp = os.path.join(folder, "fingerprint.txt")
    return os.path.exists(stamp) and open(stamp).read().strip() == fp


def stamp(folder, fp):
    with open(os.path.join(folder, "fingerprint.txt"), "w") as f:
        f.write(fp + "\n")


def card_textures():
    """Step 1. Returns the textures' fingerprint."""
    decks = paths.photos(paths.DECKS)
    if not decks:
        raise SystemExit(f"Put photos of decks laid out in a grid in {paths.DECKS} (see TRAINING.md).")
    fp = fingerprint("textures", files=decks + source_files(SYNTH_SOURCES))
    if up_to_date(paths.CARD_ASSETS, fp):
        print(f"card textures: up to date ({len(decks)} deck photos)")
        return fp
    print(f"card textures: from {len(decks)} deck photos")
    shutil.rmtree(paths.CARD_ASSETS, ignore_errors=True)
    from tooling.synth import card_assets
    card_assets.main()
    stamp(paths.CARD_ASSETS, fp)
    return fp


def boards(textures_fp, n_train, splits=tuple(SPLITS)):
    """Step 2. Returns the boards' fingerprint."""
    from tooling.synth import make_dataset
    fps = []
    for split in splits:
        n, seed, frame_all = SPLITS[split]
        n = n or n_train
        folder = os.path.join(paths.BOARDS, split)
        fp = fingerprint("boards", split, seed, frame_all, textures_fp, files=source_files(SYNTH_SOURCES))
        have = len(glob.glob(os.path.join(folder, "*.json")))
        if up_to_date(folder, fp) and have >= n:
            print(f"boards/{split}: up to date ({have} boards)")
        else:
            if not up_to_date(folder, fp):
                shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder)
                stamp(folder, fp)
            print(f"boards/{split}: rendering {n - have} of {n}")
            args = ["--split", split, "--n", str(n), "--seed-offset", str(seed)]
            make_dataset.main(args + (["--frame-all"] if frame_all else []))
        fps.append((fp, n))
    return fingerprint(fps)


class Tee:
    """Copies stdout to a log file."""

    def __init__(self, path):
        self.file, self.stdout = open(path, "a", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def train(run, settings):
    """Step 3."""
    from tooling.detector import train as T
    best = os.path.join(run, "best.pt")
    if os.path.exists(os.path.join(run, "done")):
        print(f"training: done ({best})")
        return best
    args = ["--train-split", "train", "--val-split", "val", "--out", run, "--epochs", str(settings["epochs"]),
            "--lr", str(settings["lr"]), "--warmup", str(settings["warmup"]), "--batch", str(settings["batch"])]
    if settings["init"]:
        args += ["--init", settings["init"]]
    if os.path.exists(os.path.join(run, "last.pt")):
        args.append("--resume")
    print(f"training: {settings['epochs']} epochs -> {run}")
    sys.stdout = Tee(os.path.join(run, "train.log"))
    try:
        T.main(args)
    finally:
        sys.stdout.file.close()
        sys.stdout = sys.stdout.stdout
    open(os.path.join(run, "done"), "w").close()
    return best


def test_boards(run, ckpt, cache_name):
    """Step 4a: the full pipeline on the synthetic test boards (cached: tooling.export reuses it)."""
    from tooling.detector import evaluate
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return evaluate.main(["--ckpt", ckpt, "--split", "test", "--n", str(SPLITS["test"][0]), "--device", dev,
                          "--save-cache", os.path.join(run, cache_name)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=8000, help="training boards to render (default 8000)")
    ap.add_argument("--epochs", type=int, default=None, help="default 24 (40 with --from-scratch)")
    ap.add_argument("--batch", type=int, default=10, help="lower it if the GPU runs out of memory")
    ap.add_argument("--from-scratch", action="store_true", help="don't start from the shipped model")
    ap.add_argument("--name", default=None, help="run name (default: a fingerprint of the inputs)")
    args = ap.parse_args(argv)
    scratch = args.from_scratch
    settings = dict(epochs=args.epochs or (40 if scratch else 24), lr=3e-3 if scratch else 1e-3,
                    warmup=500 if scratch else 200, batch=args.batch, init=None if scratch else paths.MODEL)

    textures_fp = card_textures()
    boards_fp = boards(textures_fp, args.boards)
    init_fp = None if scratch else fingerprint(files=[paths.MODEL])
    name = args.name or fingerprint(boards_fp, init_fp, sorted((k, v) for k, v in settings.items() if k != "init"),
                                    files=source_files(["tooling/detector/model.py", "tooling/detector/data.py",
                                                        "tooling/detector/train.py"]))
    run = os.path.join(paths.RUNS, name)
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "settings.json"), "w") as f:
        json.dump(dict(settings, boards=args.boards), f, indent=1)

    best = train(run, settings)

    print("\nsynthetic test boards, new model:")
    new = test_boards(run, best, "test.pkl")
    print("\nsynthetic test boards, shipped model:")
    old = test_boards(run, paths.MODEL, "test-shipped.pkl")
    print(f"\nphotos in {paths.GAMES}:")
    from tooling.report import report
    report({"new": best, "shipped": paths.MODEL}, out_file=os.path.join(run, "report.txt"))
    with open(os.path.join(run, "report.txt"), "a") as f:
        for label, r in (("new", new), ("shipped", old)):
            f.write(f"{label}: synthetic test boards exact {r['exact']}/{r['boards']}, "
                    f"{r['mean_abs_err']:.2f} dots off per board\n")
    print(f"\nsynthetic test boards: new {new['exact']}/{new['boards']} exact, shipped {old['exact']}/{old['boards']}")
    print(f"\nRun {name} is in {run}. To put it into the app:\n  tooling/.venv/Scripts/python -m tooling.export {name}")


if __name__ == "__main__":
    main()
