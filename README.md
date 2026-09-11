# UAV Auto Labeler: Automatic Person Pre-Labeling for Aerial Imagery Datasets

Auto-labeling tool for aerial/UAV imagery. It runs one or two person
detectors over an unannotated dataset and writes standard normalized
**YOLO-format pseudo-labels** (`class x y w h`, values in 0-1) next to the
images. Detectors (`--model`):

| `--model` | detector | notes |
|---|---|---|
| `yolov12` | local YOLOv12-small fine-tuned on VisDrone (`weights/best.pt`, `nc: 1`, `person`) | best on high-altitude nadir frames with tiny people |
| `rfdetr` | RF-DETR (COCO), `person` class only | best on lower/oblique frames, people ≥ ~40 px |
| `union` *(default)* | run both, merge per image (dedupe by IoU) | highest recall; also writes per-box provenance |

> These are model-generated candidate labels, **not** verified ground truth.
> Intended for a human review pass afterward (see [visualize.py](visualize.py)).

## Structure

- [main.py](main.py) — entry point. Runs the chosen detector(s), writes the labels.
- [visualize.py](visualize.py) — draw the labels over the images for review.
- [config.py](config.py) — default values for every CLI flag.
- [requirements.txt](requirements.txt) — Python dependencies.
- [vendor/](vendor/) — bundled YOLOv12 `ultralytics` fork wheel (see below).

## Requirements

- Anaconda / Miniconda
- Python 3.10–3.13 (3.11 recommended)
- NVIDIA GPU with recent drivers (CPU works but is slow). The reference setup
  is an RTX 5060 Ti (Blackwell, `sm_120`) → **CUDA 12.8 wheels** (`cu128`).
- Trained weights at `weights/best.pt` (or pass `--weights`)

## Environment setup (Anaconda)

From the repository root:

```powershell
conda create -n uav-auto-labeler python=3.11 -y
conda activate uav-auto-labeler

pip install -r requirements.txt
```

One `conda create` + one `pip install`. Every dependency in
`requirements.txt` is version-pinned (`==`), so installs are reproducible
and an unrelated upstream release can't silently change behavior — bump a
pin deliberately (and re-test) rather than leaving it open. `requirements.txt` installs:

- the CUDA 12.8 build of `torch` / `torchvision` (`--extra-index-url` +
  `+cu128` pins);
- `vendor/ultralytics-8.3.63-py3-none-any.whl` — the YOLOv12 fork of
  `ultralytics`, bundled in this repo (see next section); pip resolves its
  usual deps (numpy, opencv, pillow, pyyaml, scipy, pandas, tqdm, ...);
- `rfdetr` + `opencv-python` — for `--model rfdetr` / `union`. Both models
  run in this one env (tested together: torch 2.11+cu128, numpy 2.x,
  transformers 5.x). **`rfdetr` needs network to install** (no offline
  wheel) and downloads its checkpoint (~386 MB) to `~/.roboflow/models/`
  on first use. If you only ever use `--model yolov12`, you can delete the
  `rfdetr` / `opencv-python` lines.

Verify:

```powershell
python -c "import ultralytics, torch; print(ultralytics.__version__); print('cuda', torch.cuda.is_available())"
# -> 8.3.63
# -> cuda True
```

CPU-only or a different CUDA version: edit the `torch` lines in
`requirements.txt` — drop `--extra-index-url` and the `+cu128` suffixes for
CPU, or swap `cu128` for your toolkit (e.g. `cu124`).

> `FlashAttention is not available on this device. Using scaled_dot_product_attention instead.`
> is expected on Windows and harmless — PyTorch SDPA is used as the fallback.

### Why a bundled `ultralytics` wheel instead of `pip install ultralytics`

`weights/best.pt` was trained with **YOLOv12-small** using the
[pedroamtech/YOLOv12](https://github.com/pedroamtech/YOLOv12) fork (a fork of
[sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)) and its
**bundled `ultralytics` fork** (version `8.3.63`; its `AAttn` attention block
uses a fused `qkv` layer). Mainline `ultralytics` from PyPI (8.3.78+ / 8.4.x)
ships a different YOLOv12 attention block (`AAttn` with separate `qk` + `v`),
so it **cannot load these weights** and fails with:

```
AttributeError: 'AAttn' object has no attribute 'qkv'. Did you mean: 'qk'?
```

That fork is not on PyPI, so a pure-Python wheel built from it is checked in
at [`vendor/ultralytics-8.3.63-py3-none-any.whl`](vendor/) and referenced
directly from `requirements.txt`. **Never `pip install ultralytics` into this
environment** — it would shadow the fork and break weight loading.

To rebuild the vendored wheel from a fresh clone of the fork:

```powershell
pip wheel --no-deps -w vendor https://github.com/pedroamtech/YOLOv12
```

## Dataset layout

Point `--source` at the dataset root (the folder that contains `images`).
Run it once per dataset. Labels are written to a sibling `labels` folder
(and `labels_meta` for `union`), mirroring any subfolders:

```
D:\Dataset\<name>\
  images\       ... \frame.jpg
  labels\       ... \frame.txt    <- created by the script
  labels_meta\  ... \frame.json   <- created by --model union / --meta
```

You can also pass `--source` the `images` folder directly. If the source path
has no `images` folder, pass `--labels <folder>` to set the output explicitly.

## Usage

```powershell
conda activate uav-auto-labeler

# quick end-to-end test first
python main.py --limit 20

# full run, one dataset at a time
python main.py --source C:\Users\pedroam\Documents\Dataset\Okutama-Action --model union
python main.py --source C:\Users\pedroam\Documents\Dataset\Cenidet-UAV   --model union
```

Pick the detector with `--model`:

```powershell
python main.py --source D:\Dataset\X --model yolov12          # local VisDrone model only
python main.py --source D:\Dataset\X --model rfdetr           # RF-DETR only
python main.py --source D:\Dataset\X --model union            # both, merged (recommended)
```

Other common overrides:

```powershell
python main.py --source D:\Dataset\X --model union --conf 0.4 --rf-conf 0.35 --iou 0.6
python main.py --source D:\Dataset\X --batch 8                # lower if the GPU stalls / OOMs
python main.py --source D:\Dataset\X --save-empty             # empty .txt when no detection
python main.py --source D:\some\folder --labels D:\some\folder_labels
```

All flags (`python main.py --help`):

| Flag              | Default (`config.py`)                                | Description |
|-------------------|-----------------------------------------------------|-------------|
| `--source`        | `…\Dataset\Okutama-Action`                          | Dataset root (with an `images` folder) or an images folder; searched recursively |
| `--labels`        | derived (`images` -> `labels`)                      | Explicit output folder for the `.txt` files |
| `--model`         | `union`                                             | `yolov12` \| `rfdetr` \| `union` |
| `--weights`       | `weights/best.pt`                                   | YOLOv12 `.pt` weights (`yolov12` / `union`) |
| `--conf`          | `0.45`                                              | YOLOv12 confidence threshold |
| `--rf-conf`       | `0.30`                                              | RF-DETR confidence threshold (`rfdetr` / `union`) |
| `--rf-checkpoint` | `medium`                                            | RF-DETR size: `nano` \| `small` \| `medium` \| `base` \| `large` |
| `--iou`           | `0.55`                                              | `union`: boxes from the two models with IoU ≥ this are the same person |
| `--imgsz`         | `640`                                               | YOLOv12 inference resolution |
| `--device`        | auto (`cuda:0` if available)                        | `cpu`, `mps`, CUDA index (`0`), or unset for auto |
| `--batch`         | `16`                                               | Images per inference batch; lower it on GPU stalls (Windows CUDA sysmem fallback) or OOM |
| `--limit`         | `0`                                                | Process only the first N images (`0` = all) |
| `--meta` / `--no-meta` | on for `union`, off otherwise                 | Write `labels_meta/<…>.json` with per-box provenance |
| `--save-empty`    | off                                                | Also write an empty `.txt` for images with no detection |
| `--verbose`       | off                                                | Per-image detector logs |

## Output

`labels/` mirrors `images/`. One `.txt` per image with detections, standard
normalized YOLO (values 0-1), **no confidence column**:

```
0 x_center y_center width height
```

With `--model union` (or `--meta`), a parallel `labels_meta/` tree holds one
`.json` per image with the source of every box, to guide the review:

```json
[{"box": [0.51, 0.62, 0.03, 0.08], "conf": 0.82, "src": "both"},
 {"box": [0.71, 0.55, 0.02, 0.06], "conf": 0.44, "src": "rfdetr"}]
```

`src` is `yolov12`, `rfdetr`, or `both`. `visualize.py` colours boxes by it
(green = both, cyan = yolov12 only, orange = rfdetr only).

The console ends with a summary (images processed, boxes written, and for
`union` the per-source split, elapsed time, and the output paths).

## Review the labels

```powershell
python visualize.py --source C:\Users\pedroam\Documents\Dataset\Okutama-Action
```

Draws 60 random labelled frames (`--sample 0` for all) into
`review/<dataset>/`, boxes coloured by provenance when `labels_meta/` exists.
Use it to spot and delete non-person / false-positive boxes before training.

## Two datasets — which model

Measured on 300-image samples (`conf` 0.3, both models):

| dataset | YOLOv12 vs RF-DETR (person) | recommendation |
|---|---|---|
| **Okutama-Action** (oblique drone, people ≥ ~40 px) | RF-DETR **+36 %** recall, near-superset of YOLOv12, low noise | `union` (or `rfdetr`) |
| **Cenidet-UAV** (mixed, many nadir <30 px, vehicle-heavy) | roughly a tie; RF-DETR misses tiny nadir people but sees cars as noise | `union`, expect more junk to prune |

`union` recovers what each model misses (YOLOv12 → tiny nadir people;
RF-DETR → mid-altitude and occluded people); the review pass removes the
false positives RF-DETR adds on the vehicle-heavy frames.

## Related projects

- [YOLO Dataset Toolkit](https://github.com/pedroamtech/yolo-dataset-toolkit) — cleaning, validation, visualization, and analysis utilities for YOLO datasets. Useful as a follow-up step to review and QA the labels generated here.
