"""Entry point of the UAV auto-labeling pipeline.

Runs a locally trained YOLOv12 model (single class ``person``, index 0) over
a folder of aerial images and writes YOLO-format pseudo-labels next to the
images, following the standard dataset layout:

    <dataset>/images/<...>/frame.jpg  ->  <dataset>/labels/<...>/frame.txt

Each label line is::

    0 x_center y_center width height confidence

with the bbox values normalized to 0-1 and the confidence appended as an
extra column (use it later to filter the candidates; YOLO training ignores
the extra column if the files are used as-is).

These are model-generated pseudo-labels, NOT verified ground truth. The
script does not compare against any existing annotations.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

import config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate YOLO-format person pseudo-labels for aerial images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=config.SOURCE,
                   help="Dataset root (containing an 'images' folder) or an "
                        "images folder itself. Searched recursively.")
    p.add_argument("--labels", type=Path, default=config.LABELS,
                   help="Explicit output folder for the .txt files. Default: "
                        "sibling 'labels' folder derived from 'images' in the "
                        "source path.")
    p.add_argument("--weights", type=Path, default=config.WEIGHTS,
                   help="Path to the trained .pt weights.")
    p.add_argument("--conf", type=float, default=config.CONF,
                   help="Minimum detection confidence (0-1).")
    p.add_argument("--imgsz", type=int, default=config.IMGSZ,
                   help="Inference resolution.")
    p.add_argument("--device", default=config.DEVICE,
                   help="Inference device: 'cpu', 'mps', a CUDA index like 0, "
                        "or leave unset for auto.")
    p.add_argument("--batch", type=int, default=config.BATCH,
                   help="Images per inference batch. Lower it if the GPU stalls "
                        "(Windows CUDA sysmem fallback) or runs out of memory.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N images (0 = all). Use it for a "
                        "quick end-to-end test before the full run.")
    p.add_argument("--save-empty", action="store_true",
                   help="Also write an empty .txt for images with no detection "
                        "(useful as negative samples for training).")
    p.add_argument("--verbose", action="store_true",
                   help="Let Ultralytics print per-image detection logs.")
    return p.parse_args()


def resolve_scan_root(source: Path) -> Path:
    """The folder actually scanned for images."""
    if source.is_dir() and (source / "images").is_dir():
        return source / "images"
    return source


def resolve_labels_root(scan_root: Path, labels_override: Path | None) -> Path:
    """Where the .txt files go: --labels if given, else 'images' -> 'labels'."""
    if labels_override is not None:
        return labels_override.resolve()

    parts = list(scan_root.resolve().parts)
    idx = next((i for i in range(len(parts) - 1, -1, -1)
                if parts[i].lower() == "images"), None)
    if idx is None:
        raise SystemExit(
            f"Source path has no 'images' folder: {scan_root}\n"
            f"Pass --labels <folder> to set the output location explicitly."
        )
    parts[idx] = "labels"
    return Path(*parts)


def collect_images(scan_root: Path) -> list[Path]:
    if scan_root.is_file():
        return [scan_root] if scan_root.suffix.lower() in IMG_EXTENSIONS else []
    return sorted(p for p in scan_root.rglob("*")
                  if p.suffix.lower() in IMG_EXTENSIONS)


def write_label(label_path: Path, boxes) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for b in boxes or []:
            x, y, w, h = b.xywhn[0].tolist()          # normalized YOLO format
            c = float(b.conf[0])
            f.write(f"{int(b.cls[0])} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {c:.6f}\n")


def run_labeling(args: argparse.Namespace) -> None:
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights.resolve()}")
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source.resolve()}")

    scan_root = resolve_scan_root(args.source)
    labels_root = resolve_labels_root(scan_root, args.labels)
    images = collect_images(scan_root)
    if not images:
        raise SystemExit(f"No images found under: {scan_root}")
    if args.limit:
        images = images[:args.limit]

    print(f"[1/3] Loading model: {args.weights}")
    model = YOLO(str(args.weights))  # YOLOv12, nc=1, names=['person']

    try:
        import torch
        auto_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        auto_dev = "cpu"

    print(f"[2/3] Images:  {len(images)} under {scan_root}")
    print(f"      Labels:  {labels_root}")
    print(f"      conf={args.conf}  imgsz={args.imgsz}  batch={args.batch}  "
          f"device={args.device or auto_dev}")
    print("[3/3] Labeling images...  (the first batch also does model/GPU "
          "warm-up, so the bar can sit at 0% for a bit)")

    scan_root_abs = scan_root.resolve()
    images_processed = 0
    images_with_detections = 0
    total_detections = 0
    t0 = time.time()

    # Chunked instead of one big stream: the progress bar advances every batch
    # (real feedback), memory stays flat, and Ctrl-C leaves a clean partial run.
    pbar = tqdm(total=len(images), desc="Labeling", unit="img", smoothing=0.05)
    try:
        for i in range(0, len(images), args.batch):
            chunk = [str(p) for p in images[i:i + args.batch]]
            for r in model.predict(
                source=chunk,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                save=False,
                stream=False,
                verbose=args.verbose,
            ):
                images_processed += 1
                boxes = r.boxes
                n = 0 if boxes is None else len(boxes)

                if n or args.save_empty:
                    img_path = Path(r.path).resolve()
                    try:
                        rel = img_path.relative_to(scan_root_abs)
                    except ValueError:
                        rel = Path(img_path.name)
                    write_label((labels_root / rel).with_suffix(".txt"), boxes)

                total_detections += n
                if n:
                    images_with_detections += 1
                pbar.update(1)
    except KeyboardInterrupt:
        print("\nInterrupted — reporting the partial run.")
    finally:
        pbar.close()

    elapsed = time.time() - t0
    rate = images_processed / elapsed if elapsed else 0.0
    print("\nLabeling complete.")
    print(f"  Images processed:          {images_processed}")
    print(f"  Images with detections:    {images_with_detections}")
    print(f"  Images without detections: {images_processed - images_with_detections}"
          + ("  (empty .txt written)" if args.save_empty else "  (no .txt written)"))
    print(f"  Total detections:          {total_detections}")
    print(f"  Elapsed:                   {elapsed:.1f}s  ({rate:.1f} img/s)")
    print(f"  Labels saved to:           {labels_root}")
    print("  Line format:               <class> x_center y_center width height "
          "confidence  (normalized 0-1)")


if __name__ == "__main__":
    run_labeling(parse_args())
