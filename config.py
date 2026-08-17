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
CLASSES = [0, 1]   # VisDrone: 0=pedestrian, 1=people (merged into class 0 "person")
CONF = 0.25         # minimum detection confidence
IMGSZ = 640         # input resolution — matches VisDrone training, do not change
DEVICE = 0          # GPU index (use "cpu" for CPU inference)
BATCH = 16          # batch size — adjust based on available VRAM
