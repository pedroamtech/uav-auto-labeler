"""Entry point of the UAV auto-labeling pipeline.

Runs one or two detectors over a folder of aerial images and writes
YOLO-format ``person`` pseudo-labels next to the images, in the standard
dataset layout:

    <dataset>/images/<...>/frame.jpg  ->  <dataset>/labels/<...>/frame.txt

Each label line is standard normalized YOLO::

    0 x_center y_center width height

with all four bbox values in 0-1. The confidence value is not kept.

Models (``--model``):
  yolov12  the local VisDrone-tuned YOLOv12-small (weights/best.pt)
  rfdetr   RF-DETR (COCO), person class only
  union    run both, merge per image (dedupe by IoU), for maximum recall;
           also writes labels_meta/<...>.json with per-box provenance

These are model-generated pseudo-labels, NOT verified ground truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

import config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# provenance -> color is defined in visualize.py; keep the strings in sync.
SRC_YOLO = "yolov12"
SRC_RF = "rfdetr"
SRC_BOTH = "both"


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
                        "sibling 'labels' folder derived from 'images'.")
    p.add_argument("--model", choices=("yolov12", "rfdetr", "union"),
                   default=config.MODEL,
                   help="Which detector(s) to run.")
    p.add_argument("--weights", type=Path, default=config.WEIGHTS,
                   help="YOLOv12 .pt weights (yolov12 / union).")
    p.add_argument("--conf", type=float, default=config.CONF,
                   help="YOLOv12 confidence threshold (0-1).")
    p.add_argument("--rf-conf", type=float, default=config.RF_CONF,
                   help="RF-DETR confidence threshold (0-1).")
    p.add_argument("--rf-checkpoint", default=config.RF_CHECKPOINT,
                   choices=("nano", "small", "medium", "base", "large"),
                   help="RF-DETR checkpoint size (rfdetr / union).")
    p.add_argument("--iou", type=float, default=config.MERGE_IOU,
                   help="union: boxes from the two models with IoU >= this are "
                        "treated as the same person.")
    p.add_argument("--imgsz", type=int, default=config.IMGSZ,
                   help="YOLOv12 inference resolution.")
    p.add_argument("--device", default=config.DEVICE,
                   help="Inference device: 'cpu', 'mps', a CUDA index like 0, "
                        "or leave unset for auto.")
    p.add_argument("--batch", type=int, default=config.BATCH,
                   help="Images per inference batch. Lower it on GPU stalls "
                        "(Windows CUDA sysmem fallback) or OOM.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N images (0 = all).")
    p.add_argument("--meta", action=argparse.BooleanOptionalAction, default=None,
                   help="Write labels_meta/<...>.json with per-box provenance. "
                        "Default: on for --model union, off otherwise.")
    p.add_argument("--save-empty", action="store_true",
                   help="Also write an empty .txt for images with no detection.")
    p.add_argument("--verbose", action="store_true",
                   help="Let the detector print per-image logs.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def resolve_scan_root(source: Path) -> Path:
    if source.is_dir() and (source / "images").is_dir():
        return source / "images"
    return source


def _swap_images_segment(scan_root: Path, new_name: str) -> Path:
    parts = list(scan_root.resolve().parts)
    idx = next((i for i in range(len(parts) - 1, -1, -1)
                if parts[i].lower() == "images"), None)
    if idx is None:
        return scan_root.resolve().with_name(new_name)
    parts[idx] = new_name
    return Path(*parts)


def resolve_labels_root(scan_root: Path, labels_override: Path | None) -> Path:
    if labels_override is not None:
        return labels_override.resolve()
    out = _swap_images_segment(scan_root, "labels")
    if out == scan_root.resolve():
        raise SystemExit(
            f"Source path has no 'images' folder: {scan_root}\n"
            f"Pass --labels <folder> to set the output location explicitly."
        )
    return out


def collect_images(scan_root: Path) -> list[Path]:
    if scan_root.is_file():
        return [scan_root] if scan_root.suffix.lower() in IMG_EXTENSIONS else []
    return sorted(p for p in scan_root.rglob("*")
                  if p.suffix.lower() in IMG_EXTENSIONS)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _xyxy(box):
    x, y, w, h = box[:4]
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def merge_predictions(a_boxes, b_boxes, iou_thr):
    """a_boxes/b_boxes: lists of [xc, yc, w, h, conf]. Greedy IoU match.

    Returns (boxes, srcs): the merged box list and a parallel list of
    "yolov12" / "rfdetr" / "both" tags.
    """
    used_b = set()
    boxes, srcs = [], []
    for a in a_boxes:
        best_j, best_iou = -1, iou_thr
        for j, b in enumerate(b_boxes):
            if j in used_b:
                continue
            v = _iou(a, b)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used_b.add(best_j)
            b = b_boxes[best_j]
            boxes.append(b if b[4] >= a[4] else a)  # keep higher-confidence box
            srcs.append(SRC_BOTH)
        else:
            boxes.append(a)
            srcs.append(SRC_YOLO)
    for j, b in enumerate(b_boxes):
        if j not in used_b:
            boxes.append(b)
            srcs.append(SRC_RF)
    return boxes, srcs


# --------------------------------------------------------------------------- #
# Backends -> { "<image path str>": [[xc, yc, w, h, conf], ...] }
# --------------------------------------------------------------------------- #
def predict_yolov12(images, weights: Path, conf, imgsz, device, batch, verbose):
    from ultralytics import YOLO

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights.resolve()}")
    print(f"  [yolov12] loading {weights}")
    model = YOLO(str(weights))
    preds = {}
    pbar = tqdm(total=len(images), desc="yolov12", unit="img", smoothing=0.05)
    try:
        for i in range(0, len(images), batch):
            chunk = [str(p) for p in images[i:i + batch]]
            results = model.predict(source=chunk, conf=conf, imgsz=imgsz,
                                    device=device, save=False, stream=False,
                                    verbose=verbose)
            for path, r in zip(chunk, results):
                rows = []
                if r.boxes is not None:
                    for b in r.boxes:
                        x, y, w, h = b.xywhn[0].tolist()
                        rows.append([x, y, w, h, float(b.conf[0])])
                preds[path] = rows
                pbar.update(1)
    finally:
        pbar.close()
    return preds


_RF_CLASSES = {
    "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
    "base": "RFDETRBase", "large": "RFDETRLarge",
}


def predict_rfdetr(images, conf, checkpoint, batch):
    import rfdetr
    from PIL import Image

    cls = getattr(rfdetr, _RF_CLASSES[checkpoint])
    print(f"  [rfdetr] loading {cls.__name__} (downloads the checkpoint on first use)")
    model = cls()
    bs = min(batch, len(images)) or 1
    try:
        model.optimize_for_inference(batch_size=bs)
    except Exception as e:  # optional speed-up; safe to skip
        print(f"  [rfdetr] optimize_for_inference skipped: {e}")

    preds = {}
    pbar = tqdm(total=len(images), desc="rfdetr", unit="img", smoothing=0.05)
    try:
        for i in range(0, len(images), bs):
            chunk = [str(p) for p in images[i:i + bs]]
            fed = chunk + [chunk[-1]] * (bs - len(chunk))  # pad to fixed batch
            try:
                dets = model.predict(fed, threshold=conf)
            except ValueError as e:
                if "Batch size mismatch" not in str(e):
                    raise
                model.remove_optimized_model()
                dets = model.predict(chunk, threshold=conf)
                fed = chunk
            if not isinstance(dets, list):
                dets = [dets]
            for path, det in zip(chunk, dets[:len(chunk)]):
                names = det.data.get("class_name", None)
                rows = []
                if names is not None and len(det.xyxy):
                    with Image.open(path) as im:
                        W, H = im.size
                    for (x1, y1, x2, y2), nm, cf in zip(det.xyxy, names, det.confidence):
                        if str(nm) != "person":
                            continue
                        rows.append([float((x1 + x2) / 2 / W), float((y1 + y2) / 2 / H),
                                     float((x2 - x1) / W), float((y2 - y1) / H), float(cf)])
                preds[path] = rows
                pbar.update(1)
    finally:
        pbar.close()
    return preds


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_labeling(args: argparse.Namespace) -> None:
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source.resolve()}")

    scan_root = resolve_scan_root(args.source)
    scan_root_abs = scan_root.resolve()
    labels_root = resolve_labels_root(scan_root, args.labels)
    meta = args.meta if args.meta is not None else (args.model == "union")
    meta_root = labels_root.with_name(labels_root.name + "_meta")

    images = collect_images(scan_root)
    if not images:
        raise SystemExit(f"No images found under: {scan_root}")
    if args.limit:
        images = images[:args.limit]

    try:
        import torch
        auto_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        auto_dev = "cpu"

    print(f"[1/3] Model:   {args.model}")
    print(f"[2/3] Images:  {len(images)} under {scan_root}")
    print(f"      Labels:  {labels_root}" + (f"   Meta: {meta_root}" if meta else ""))
    print(f"      device={args.device or auto_dev}  batch={args.batch}  "
          f"conf(yolo)={args.conf}  conf(rf)={args.rf_conf}"
          + (f"  merge_iou={args.iou}" if args.model == "union" else ""))
    print("[3/3] Running inference...  (first batch also does model/GPU warm-up)")

    t0 = time.time()
    if args.model == "yolov12":
        a = predict_yolov12(images, args.weights, args.conf, args.imgsz,
                            args.device, args.batch, args.verbose)
        preds = {k: v for k, v in a.items()}
        srcs = {k: [SRC_YOLO] * len(v) for k, v in a.items()}
    elif args.model == "rfdetr":
        b = predict_rfdetr(images, args.rf_conf, args.rf_checkpoint, args.batch)
        preds = {k: v for k, v in b.items()}
        srcs = {k: [SRC_RF] * len(v) for k, v in b.items()}
    else:  # union
        a = predict_yolov12(images, args.weights, args.conf, args.imgsz,
                            args.device, args.batch, args.verbose)
        b = predict_rfdetr(images, args.rf_conf, args.rf_checkpoint, args.batch)
        preds, srcs = {}, {}
        for k in [str(p) for p in images]:
            preds[k], srcs[k] = merge_predictions(a.get(k, []), b.get(k, []), args.iou)

    # write
    n_img = n_with = n_det = n_from_yolo = n_from_rf = n_from_both = 0
    for path, boxes in preds.items():
        n_img += 1
        n_det += len(boxes)
        tags = srcs.get(path, [])
        n_from_yolo += tags.count(SRC_YOLO)
        n_from_rf += tags.count(SRC_RF)
        n_from_both += tags.count(SRC_BOTH)
        if not boxes and not args.save_empty:
            continue
        try:
            rel = Path(path).resolve().relative_to(scan_root_abs)
        except ValueError:
            rel = Path(Path(path).name)
        lbl = (labels_root / rel).with_suffix(".txt")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        with open(lbl, "w") as f:
            for x, y, w, h, _c in boxes:
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        if boxes:
            n_with += 1
        if meta:
            mp = (meta_root / rel).with_suffix(".json")
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps([
                {"box": [round(float(x), 6), round(float(y), 6),
                         round(float(w), 6), round(float(h), 6)],
                 "conf": round(float(c), 4), "src": s}
                for (x, y, w, h, c), s in zip(boxes, tags)
            ]), encoding="utf-8")

    elapsed = time.time() - t0
    print("\nLabeling complete.")
    print(f"  Images processed:          {n_img}")
    print(f"  Images with detections:    {n_with}")
    print(f"  Total boxes written:       {n_det}")
    if args.model == "union":
        print(f"    from both models:        {n_from_both}")
        print(f"    yolov12 only:            {n_from_yolo}")
        print(f"    rfdetr only:             {n_from_rf}")
    print(f"  Elapsed:                   {elapsed:.1f}s "
          f"({n_img / elapsed:.1f} img/s)" if elapsed else "")
    print(f"  Labels saved to:           {labels_root}")
    if meta:
        print(f"  Provenance saved to:       {meta_root}")
    print("  Line format:               0 x_center y_center width height (normalized 0-1)")


if __name__ == "__main__":
    run_labeling(parse_args())
