"""Run a photo through the same steps as the app and draw what was found (out/detections/).

Usage, from the repo root:
  tooling/.venv/Scripts/python -m tooling.detector.run_photo training-data/games/sample_photo.jpg
"""
import argparse
import os

import cv2
import numpy as np

from tooling import paths
from tooling.detector.model import load_model
from tooling.detector.photo import analyze

DRAW = {"blue": (230, 160, 60), "green": (60, 200, 80), "orange": (30, 140, 255), "pink": (170, 60, 240),
        "purple": (200, 90, 140), "red": (40, 40, 230)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--ckpt", default=paths.MODEL)
    ap.add_argument("--device", default="cpu", help="cpu (default; safe while training) or cuda")
    args = ap.parse_args()
    model = load_model(args.ckpt, args.device)

    img = cv2.imread(args.photo)
    a = analyze(model, img, args.device)
    res, badges, dots, s = a["result"], a["badges"], a["dots"], a["scale"]
    W, H = a["size"]
    print(f"card width ~{a['card_w']:.0f}px -> scale {s:.3f}, model input {W}x{H}")
    by_suit = {}
    for b in badges:
        by_suit[b["suit"]] = by_suit.get(b["suit"], 0) + 1
    print(f"badges: {len(badges)} {by_suit}   perspective fit: {res['perspective_fit']}")
    print("cards per deck:", res["cards_per_deck"])
    print("visible dots per deck:", res["counts"], " total", len(dots))
    for wmsg in res["warnings"]:
        print("  warning:", wmsg)

    k = 2.0
    vis = (cv2.resize(img, None, fx=s * k, fy=s * k, interpolation=cv2.INTER_AREA) * 0.55).astype(np.uint8)
    for c in res["cards"]:
        col = DRAW[c["deck"]]
        cv2.polylines(vis, [(np.array(c["corners"]) * k).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        for x, y in c["dots"]:  # printed dot positions (small marks), covered or not
            cv2.drawMarker(vis, (int(x * k), int(y * k)), col, cv2.MARKER_CROSS, 6, 1)
    for b in badges:
        col = DRAW.get(b["deck"], (255, 255, 255)) if b["suit"] != "start" else (255, 255, 255)
        p = (int(b["x"] * k), int(b["y"] * k))
        q = (int(b["icon"][0] * k), int(b["icon"][1] * k))
        cv2.arrowedLine(vis, p, q, col, 2, tipLength=0.35)
        label = "X" if b["suit"] == "start" else f"{b['suit'][0]}{b['value']}{'b' if b['side'] == 'back' else ''}"
        cv2.putText(vis, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    for d in dots:  # counted dots: ring in owner colour; unexplained ones get a white square
        if d.get("partial"):  # partly covered (not counted): thin ring + yellow cross
            cv2.circle(vis, (int(d["x"] * k), int(d["y"] * k)), 9, DRAW[d["deck"]], 1, cv2.LINE_AA)
            cv2.drawMarker(vis, (int(d["x"] * k), int(d["y"] * k)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            continue
        cv2.circle(vis, (int(d["x"] * k), int(d["y"] * k)), 9, DRAW[d["deck"]], 2, cv2.LINE_AA)
        if not d.get("explained", True):
            cv2.rectangle(vis, (int(d["x"] * k) - 12, int(d["y"] * k) - 12), (int(d["x"] * k) + 12, int(d["y"] * k) + 12),
                          (255, 255, 255), 2)
    y = 30
    for dk, n in res["counts"].items():
        cv2.putText(vis, f"{dk}: {n} dots, {res['cards_per_deck'].get(dk, 0)} cards", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, DRAW[dk], 2, cv2.LINE_AA)
        y += 34
    name = os.path.splitext(os.path.basename(args.photo))[0]
    out_path = os.path.join(paths.OUT, "detections", f"{name}_detections.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
