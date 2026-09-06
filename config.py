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
SOURCE = Path(r"C:\Users\pedroam\Documents\Dataset\Okutama-Action")

# --- Output ---
# Leave as None to use the "images" -> "labels" rule above. Set an explicit
# path (or pass --labels) only if the source has no "images" folder.
LABELS = None

# --- Inference parameters ---
CONF = 0.45         # minimum detection confidence for a box to be written
IMGSZ = 640         # inference resolution (matches VisDrone training)
DEVICE = None        # None = auto (CUDA if available, else CPU). Also: "cpu", "mps", 0
BATCH = 16          # batch size — adjust based on available VRAM
