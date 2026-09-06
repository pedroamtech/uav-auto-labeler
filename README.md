# UAV Auto Labeler: Automatic Person Pre-Labeling for Aerial Imagery Datasets

Auto-labeling tool for aerial/UAV imagery using a locally trained **YOLOv12**
model (single class `person`, id `0`; trained on VisDrone reduced to `nc: 1`,
`names: ['person']`). It runs inference over an unannotated dataset and writes
standard normalized **YOLO-format pseudo-labels** (`class x y w h`, values in
0-1) next to the images.

> These are model-generated candidate labels, **not** verified ground truth.
> The script does not validate against any existing annotations.

## Structure

- [main.py](main.py) — entry point. Loads the model, runs inference, writes the labels.
- [config.py](config.py) — default values for every CLI flag.
- [requirements.txt](requirements.txt) — Python dependencies (self-contained).
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

That is the whole setup — **no other repositories to clone**, `git` is not
required, and it works offline. `requirements.txt` installs:

- the CUDA 12.8 build of `torch` / `torchvision` (`--extra-index-url` +
  `+cu128` pins);
- `vendor/ultralytics-8.3.63-py3-none-any.whl` — the YOLOv12 fork of
  `ultralytics`, bundled in this repo (see next section); pip resolves its
  usual deps (numpy, opencv, pillow, pyyaml, scipy, pandas, tqdm, ...);
- `tqdm`.

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
Labels are written to a sibling `labels` folder, mirroring any subfolders:

```
D:\Dataset\Okutama-Action\
  images\  Drone1\ Morning\ ... \frame.jpg
  labels\  Drone1\ Morning\ ... \frame.txt   <- created by the script
```

You can also pass `--source` the `images` folder directly. If the source path
has no `images` folder, pass `--labels <folder>` to set the output explicitly.

## Usage

```powershell
conda activate uav-auto-labeler          # or use the training env's python directly

# quick end-to-end test first
python main.py --limit 20

# full run
python main.py --source C:\Users\pedroam\Documents\Dataset\Okutama-Action
```

Common overrides:

```powershell
python main.py --source D:\Dataset\Okutama-Action --conf 0.35 --device 0
python main.py --source D:\Dataset\Okutama-Action --batch 8          # lower if the GPU stalls / OOMs
python main.py --source D:\Dataset\Okutama-Action --save-empty       # empty .txt when no detection
python main.py --source D:\some\folder --labels D:\some\folder_labels
```

All flags (`python main.py --help`):

| Flag           | Default (from `config.py`)                          | Description |
|----------------|----------------------------------------------------|-------------|
| `--source`     | `C:\Users\pedroam\Documents\Dataset\Okutama-Action` | Dataset root (with an `images` folder) or an images folder; searched recursively |
| `--labels`     | derived (`images` -> `labels`)                      | Explicit output folder for the `.txt` files |
| `--weights`    | `weights/best.pt`                                   | Trained `.pt` weights |
| `--conf`       | `0.25`                                              | Minimum detection confidence |
| `--imgsz`      | `640`                                               | Inference resolution |
| `--device`     | auto (`cuda:0` if available, else `cpu`)            | `cpu`, `mps`, CUDA index (`0`), or unset for auto |
| `--batch`      | `16`                                               | Images per inference batch; lower it on GPU stalls (Windows CUDA sysmem fallback) or OOM |
| `--limit`      | `0`                                                | Process only the first N images (`0` = all); for a quick test |
| `--save-empty` | off                                                | Also write an empty `.txt` for images with no detection |
| `--verbose`    | off                                                | Per-image Ultralytics logs |

## Output

One `.txt` per image (with detections), in the `labels` tree that mirrors
`images`. One line per detection, standard normalized YOLO (all values 0-1),
no confidence column:

```
0 x_center y_center width height
```

Only detections at or above `--conf` are written. By default no file is
written for images with zero detections; pass `--save-empty` to create
empty `.txt` files for those too.

The console prints where the labels went plus a summary:

```
[1/3] Loading model: weights\best.pt
[2/3] Images:  18103 under C:\Users\pedroam\Documents\Dataset\Okutama-Action\images
      Labels:  C:\Users\pedroam\Documents\Dataset\Okutama-Action\labels
      conf=0.25  imgsz=640  batch=16  device=cuda:0
[3/3] Labeling images...  (the first batch also does model/GPU warm-up, so the bar can sit at 0% for a bit)
Labeling: 100%|██████████| 18103/18103 [12:41<00:00, 23.8img/s]

Labeling complete.
  Images processed:          18103
  Images with detections:    15980
  Images without detections: 2123  (no .txt written)
  Total detections:          98123
  Elapsed:                   761.2s  (23.8 img/s)
  Labels saved to:           C:\Users\pedroam\Documents\Dataset\Okutama-Action\labels
  Line format:               <class> x_center y_center width height (normalized 0-1)
```

## Related projects

- [YOLO Dataset Toolkit](https://github.com/pedroamtech/yolo-dataset-toolkit) — cleaning, validation, visualization, and analysis utilities for YOLO datasets. Useful as a follow-up step to review and QA the labels generated here.
