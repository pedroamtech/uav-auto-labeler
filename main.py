"""Punto de entrada del pipeline de auto-etiquetado UAV.

Descarga el modelo (si hace falta), corre inferencia por lotes sobre las
imagenes de config.IMG_DIR y escribe las etiquetas en formato YOLO en
config.LABEL_DIR.
"""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
from ultralytics import YOLO

import config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def ensure_model(hf_repo: str, model_file: Path) -> None:
    if model_file.exists():
        print(f"[1/3] Modelo local encontrado: {model_file}")
        return

    print(f"[1/3] Modelo no encontrado localmente, buscando en {hf_repo}...")
    pt_files = [f for f in list_repo_files(hf_repo) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {hf_repo}")

    print(f"      Descargando {pt_files[0]} desde HuggingFace...")
    tmp = hf_hub_download(hf_repo, pt_files[0])
    shutil.copy(tmp, model_file)
    print("      Descarga completa.")


def count_images(img_dir: Path) -> int:
    return sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def run_labeling() -> None:
    ensure_model(config.HF_REPO, config.MODEL_FILE)

    print("[2/3] Cargando modelo...")
    model = YOLO(str(config.MODEL_FILE))  # VisDrone fine-tuned; classes 0=pedestrian 1=people
    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)

    total_images = count_images(config.IMG_DIR)
    print(f"      {total_images} imagenes encontradas en {config.IMG_DIR}")

    results = model.predict(
        source=str(config.IMG_DIR),
        classes=config.CLASSES,
        conf=config.CONF,
        imgsz=config.IMGSZ,
        device=config.DEVICE,
        save=False,
        stream=True,
        batch=config.BATCH,
        verbose=False,
    )

    print("[3/3] Etiquetando imagenes...")
    valid_classes = set(config.CLASSES)
    total_detections = 0
    images_with_detections = 0

    for r in tqdm(results, total=total_images, desc="Etiquetando", unit="img"):
        img_path = Path(r.path)
        label_path = config.LABEL_DIR / (img_path.stem + ".txt")

        detections_in_image = 0
        with open(label_path, "w") as f:
            for box in r.boxes:
                if int(box.cls) in valid_classes:  # pedestrian + people -> label 0
                    x, y, w, h = box.xywhn[0].tolist()  # normalized YOLO format
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    detections_in_image += 1

        total_detections += detections_in_image
        if detections_in_image:
            images_with_detections += 1

    print("\nEtiquetado completo.")
    print(f"  Imagenes procesadas:      {total_images}")
    print(f"  Imagenes con detecciones: {images_with_detections}")
    print(f"  Detecciones totales:      {total_detections}")
    print(f"  Etiquetas guardadas en:   {config.LABEL_DIR}")


if __name__ == "__main__":
    run_labeling()
