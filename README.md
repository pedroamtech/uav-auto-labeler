# UAV-Auto-Labeler

Auto-labeling tool for aerial/UAV imagery using a YOLOv8 model (`yolov8x-visdrone`) fine-tuned on the VisDrone dataset. Generates YOLO-format labels by merging the `pedestrian` and `people` classes into a single `person` class (id `0`), via batched GPU inference. The model is automatically downloaded from Hugging Face Hub if not present locally.

## Structure

- [main.py](main.py) — entry point. Downloads the model if missing, runs inference, and writes the labels. No changes needed for normal use.
- [config.py](config.py) — all editable configuration: dataset paths, model, and inference parameters.
- [requirements.txt](requirements.txt) — Python dependencies.

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, see `DEVICE` in the config to use CPU)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All paths and parameters are set in [config.py](config.py), without touching `main.py`:

| Variable     | Description                                                          |
|--------------|-----------------------------------------------------------------------|
| `HF_REPO`    | Hugging Face repository the model is downloaded from                  |
| `MODEL_FILE` | Local path where the `.pt` model file is stored/looked up             |
| `IMG_DIR`    | Folder with the input images to label                                 |
| `LABEL_DIR`  | Output folder where `.txt` labels are written (YOLO format)           |
| `CLASSES`    | VisDrone class IDs to detect (`0`=pedestrian, `1`=people)             |
| `CONF`       | Minimum detection confidence                                          |
| `IMGSZ`      | Model input resolution (do not change: must match VisDrone training)  |
| `DEVICE`     | Inference device (`0` for GPU, `"cpu"` for CPU)                       |
| `BATCH`      | Batch size, adjust based on available VRAM                            |

To label a different dataset, just edit `IMG_DIR` and `LABEL_DIR` in `config.py`.

## Usage

```bash
python main.py
```

The script:

1. Checks whether the model (`MODEL_FILE`) exists locally; if not, downloads it from `HF_REPO`.
2. Creates `LABEL_DIR` if it doesn't exist.
3. Runs streaming/batched inference over all images in `IMG_DIR`, showing a progress bar (`tqdm`).
4. For each image, writes a `.txt` file in `LABEL_DIR` with one line per detection in normalized YOLO format:

   ```
   0 x_center y_center width height
   ```

   where `0` is the `person` class (merge of `pedestrian` + `people`) and the coordinates are normalized between 0 and 1.

The console shows step-by-step progress (model download/load, image count, progress bar) and a final summary with total images processed, images with detections, and total detections:

```
[1/3] Local model found: yolov8x-visdrone.pt
[2/3] Loading model...
      842 images found in C:/.../train/images
[3/3] Labeling images...
Labeling: 100%|██████████| 842/842 [02:14<00:00,  6.28img/s]

Labeling complete.
  Images processed:        842
  Images with detections:  731
  Total detections:        5210
  Labels saved to:         C:/.../train/labels
```

## Related projects

- [yolo-dataset-toolkit](https://github.com/pedroamtech/yolo-dataset-toolkit) — cleaning, validation, visualization, and analysis utilities for YOLO datasets. Useful as a follow-up step to review and QA the labels generated here.
