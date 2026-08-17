"""Configuration for uav-auto-labeler.

Edit the paths and inference parameters in this file. main.py should not
need any changes to adapt the pipeline to a new dataset or machine.
"""
from pathlib import Path

# --- Model (Hugging Face Hub) ---
HF_REPO = "mshamrai/yolov8x-visdrone"
MODEL_FILE = Path("yolov8x-visdrone.pt")

# --- Dataset paths ---
IMG_DIR = Path("C:/Users/pedroam/Documents/Data-Augmentation/Datasets/AutoDA-UAV/train/images")
LABEL_DIR = Path("C:/Users/pedroam/Documents/Data-Augmentation/Datasets/AutoDA-UAV/train/labels")

# --- Inference parameters ---
# This project only detects "person": VisDrone classes 0 (pedestrian) and
# 1 (people) merged into a single output class 0. main.py writes every
# matched box as label 0 regardless of which of these two IDs it came
# from, so do not add other VisDrone classes (car, van, bus, etc.) here.
CLASSES = [0, 1]
CONF = 0.25         # minimum detection confidence
IMGSZ = 640         # input resolution — matches VisDrone training, do not change
DEVICE = 0          # GPU index (use "cpu" for CPU inference)
BATCH = 16          # batch size — adjust based on available VRAM
