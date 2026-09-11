"""Default configuration for uav-auto-labeler.

Every value here is just a default: it can be overridden from the command
line (see `python main.py --help`). Edit this file if you want to change
the defaults permanently without passing flags each run.
"""
from pathlib import Path

# --- Model ---
# Locally trained YOLOv12 weights. Single class: nc=1, names=['person'] (index 0).
WEIGHTS = Path("weights/best.pt")

# --- Input ---
# Dataset root OR an images folder. If the path contains an "images" folder,
# labels are written to a sibling "labels" folder, mirroring subfolders:
#   D:\Dataset\Okutama-Action\images\...\frame.jpg
#   -> D:\Dataset\Okutama-Action\labels\...\frame.txt
SOURCE = Path(r"C:\Users\pedroam\Documents\Dataset\Cenidet-UAV")

# --- Output ---
# Leave as None to use the "images" -> "labels" rule above. Set an explicit
# path (or pass --labels) only if the source has no "images" folder.
LABELS = None

# --- Model ---
# yolov12  local VisDrone-tuned YOLOv12-small (weights/best.pt)
# rfdetr   RF-DETR (COCO), person class only
# union    run both and merge per image (dedupe by IoU) for maximum recall;
#          also writes labels_meta/<...>.json with per-box provenance
MODEL = "union"

# --- Inference parameters ---
CONF = 0.45         # YOLOv12 confidence threshold
RF_CONF = 0.30     # RF-DETR confidence threshold
RF_CHECKPOINT = "medium"   # nano | small | medium | base | large
MERGE_IOU = 0.55   # union: boxes from the two models with IoU >= this = same person
IMGSZ = 640        # YOLOv12 inference resolution (matches VisDrone training)
DEVICE = None       # None = auto (CUDA if available, else CPU). Also: "cpu", "mps", 0
BATCH = 16         # batch size — adjust based on available VRAM
