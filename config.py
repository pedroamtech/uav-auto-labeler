"""Configuracion de uav-auto-labeler.

Edita las rutas y parametros de inferencia en este archivo. main.py no
debe modificarse para adaptar el pipeline a un nuevo dataset o maquina.
"""
from pathlib import Path

# --- Modelo (Hugging Face Hub) ---
HF_REPO = "mshamrai/yolov8x-visdrone"
MODEL_FILE = Path("yolov8x-visdrone.pt")

# --- Rutas del dataset ---
IMG_DIR = Path("C:/Users/pedroam/Documents/Data-Augmentation/Datasets/AutoDA-UAV/train/images")
LABEL_DIR = Path("C:/Users/pedroam/Documents/Data-Augmentation/Datasets/AutoDA-UAV/train/labels")

# --- Parametros de inferencia ---
CLASSES = [0, 1]   # VisDrone: 0=pedestrian, 1=people (se fusionan en la clase 0 "person")
CONF = 0.25         # confianza minima de deteccion
IMGSZ = 640         # resolucion de entrada — coincide con el entrenamiento de VisDrone, no cambiar
DEVICE = 0          # indice de GPU (usar "cpu" para inferencia en CPU)
BATCH = 16          # tamano de batch — ajustar segun la VRAM disponible
