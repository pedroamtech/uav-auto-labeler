"""Draw the generated YOLO labels over the images, for a quick visual review.

    python visualize.py --source <dataset>            # 60 random labelled frames
    python visualize.py --source <dataset> --sample 0 # every labelled frame

If a sibling ``labels_meta`` folder exists (written by ``--model union``),
boxes are coloured by provenance:

    green  = both models        cyan = yolov12 only        orange = rfdetr only

Otherwise every box is green. Output goes to ``--out`` (default
``review/<dataset name>/``).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

import config
from main import IMG_EXTENSIONS, resolve_scan_root, resolve_labels_root

COLORS = {"both": (60, 220, 60), "yolov12": (0, 200, 255),
          "rfdetr": (255, 150, 0), None: (60, 220, 60)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay generated YOLO labels on the images for review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=config.SOURCE,
                   help="Dataset root or images folder (same as main.py).")
    p.add_argument("--labels", type=Path, default=config.LABELS,
                   help="Label folder. Default: sibling 'labels' of 'images'.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output folder. Default: review/<dataset name>/.")
    p.add_argument("--sample", type=int, default=60,
                   help="How many labelled images to draw (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=3, help="Box line width (px).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scan_root = resolve_scan_root(args.source)
    labels_root = resolve_labels_root(scan_root, args.labels)
    meta_root = labels_root.with_name(labels_root.name + "_meta")
    out = args.out or Path("review") / scan_root.resolve().parent.name
    out.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in scan_root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    pairs = []
    for img in imgs:
        try:
            rel = img.resolve().relative_to(scan_root.resolve())
        except ValueError:
            rel = Path(img.name)
        lbl = (labels_root / rel).with_suffix(".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            pairs.append((img, rel, lbl))

    if not pairs:
        raise SystemExit(f"No non-empty labels found under {labels_root}")
    if args.sample and args.sample < len(pairs):
        random.seed(args.seed)
        pairs = random.sample(pairs, args.sample)

    print(f"{len(pairs)} images -> {out}")
    for img, rel, lbl in pairs:
        im = Image.open(img).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)

        meta = None
        mp = (meta_root / rel).with_suffix(".json")
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = None

        rows = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        for i, parts in enumerate(rows):
            _, x, y, w, h = (float(v) for v in parts[:5])
            src = meta[i]["src"] if meta and i < len(meta) else None
            d.rectangle([(x - w / 2) * W, (y - h / 2) * H,
                         (x + w / 2) * W, (y + h / 2) * H],
                        outline=COLORS.get(src, COLORS[None]), width=args.width)

        dst = out / rel.with_suffix(".jpg").name
        im.save(dst, quality=88)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
