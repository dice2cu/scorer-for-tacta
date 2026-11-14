"""Put a trained model into the app. See TRAINING.md.

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.export [run]

run: a folder name in out/runs/ (default: the latest finished run), or "shipped" to re-export the
current model (after changing the scoring code, to bring the test data up to date).

Writes the app's model (src/data/model/), the model's weights for the next training
(tooling/detector/tactanet.pt), and the test data: out/test-cases/ for `npm test` and
tests/fixtures/ for CI. Then check the app with `npm test` and `npm run test:browser`.
"""
import argparse
import os
import sys

import torch

from tooling import paths


def latest_run():
    runs = [os.path.join(paths.RUNS, d) for d in os.listdir(paths.RUNS)] if os.path.isdir(paths.RUNS) else []
    runs = [r for r in runs if os.path.exists(os.path.join(r, "done"))]
    if not runs:
        raise SystemExit("No finished training run in out/runs/. Run: tooling/.venv/Scripts/python -m tooling.train")
    return max(runs, key=lambda r: os.path.getmtime(os.path.join(r, "done")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None)
    args = ap.parse_args(argv)

    from tooling.detector import export_web, make_ci_fixtures, make_web_cases
    from tooling.train import boards, card_textures, test_boards

    if args.run == "shipped":
        run = os.path.join(paths.RUNS, "shipped")
        os.makedirs(run, exist_ok=True)
        ckpt = paths.MODEL
    else:
        run = os.path.join(paths.RUNS, args.run) if args.run else latest_run()
        ckpt = os.path.join(run, "best.pt")
        if not os.path.exists(os.path.join(run, "done")):
            sys.exit(f"{run} isn't a finished training run")
    print(f"exporting {ckpt}")

    export_web.main(["--ckpt", ckpt])
    if ckpt != paths.MODEL:
        torch.save({"model": torch.load(ckpt, map_location="cpu", weights_only=False)["model"]}, paths.MODEL)
        print(f"weights -> {paths.MODEL}")

    cache = os.path.join(run, "test.pkl")
    if args.run == "shipped" or not os.path.exists(cache):
        boards(card_textures(), 0, splits=["test"])
        test_boards(run, paths.MODEL, "test.pkl")
    make_web_cases.main(["--cache", cache, "--ckpt", paths.MODEL])
    make_ci_fixtures.main(["--ckpt", paths.MODEL])
    print("\nDone. Check the app with `npm test` and `npm run test:browser`, then commit src/data/model/,"
          "\ntooling/detector/tactanet.pt and tests/fixtures/.")


if __name__ == "__main__":
    main()
