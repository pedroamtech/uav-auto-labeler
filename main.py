"""Entry point of the UAV auto-labeling pipeline.

Downloads the model (if needed), runs batched inference over the images
in config.IMG_DIR, and writes YOLO-format labels to config.LABEL_DIR.
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
        print(f"[1/3] Local model found: {model_file}")
        return

    print(f"[1/3] Model not found locally, looking it up in {hf_repo}...")
    pt_files = [f for f in list_repo_files(hf_repo) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {hf_repo}")

    print(f"      Downloading {pt_files[0]} from HuggingFace...")
    tmp = hf_hub_download(hf_repo, pt_files[0])
    shutil.copy(tmp, model_file)
    print("      Download complete.")


def count_images(img_dir: Path) -> int:
    return sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def run_labeling() -> None:
    ensure_model(config.HF_REPO, config.MODEL_FILE)

    print("[2/3] Loading model...")
    model = YOLO(str(config.MODEL_FILE))  # VisDrone fine-tuned; classes 0=pedestrian 1=people
    config.LABEL_DIR.mkdir(parents=True, exist_ok=True)

    total_images = count_images(config.IMG_DIR)
    print(f"      {total_images} images found in {config.IMG_DIR}")

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

    print("[3/3] Labeling images...")
    valid_classes = set(config.CLASSES)
    total_detections = 0
    images_with_detections = 0

    for r in tqdm(results, total=total_images, desc="Labeling", unit="img"):
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

    print("\nLabeling complete.")
    print(f"  Images processed:        {total_images}")
    print(f"  Images with detections:  {images_with_detections}")
    print(f"  Total detections:        {total_detections}")
    print(f"  Labels saved to:         {config.LABEL_DIR}")


if __name__ == "__main__":
    run_labeling()
