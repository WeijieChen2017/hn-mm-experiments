# Head & Neck Multimodal (CT + Dose + Contour) — Overlay‑Free Stages 1/2/3

This repo trains and evaluates **binary** outcomes (`label` is `"True"`/`"False"`) for head–neck RT using:
- **Stage 1** — CT‑only: frozen 2D encoder over K slices + attention pooling → MLP.
- **Stage 2** — Late fusion: Stage 1 + structured **dose/contour numeric features** (+ optional tiny 3D adapters).
- **Stage 3** — Cross‑attention: CT slice tokens cross‑attend **dose/geometry tokens** → pooled → MLP.

> **No overlays** are used; CT is kept in‑distribution for the image encoder.


## Data format

Provide a JSON like:

```json
{
  "0": {
    "label": "True",
    "CT": "/path/to/ct_volume.nii.gz",
    "dose": "/path/to/dose_volume.nii.gz",
    "contour": "/path/to/gtv_mask.nii.gz",
    "split": "train",
    "radiology_report": "",
    "clinical_notes": ""
  },
  "1": {
    "label": "False",
    "CT": "/path/to/ct_volume.nii.gz",
    "dose": "/path/to/dose_volume.nii.gz",
    "contour": "/path/to/gtv_mask.nii.gz",
    "split": "test"
  }
}
```

Supported volumes: `.nii/.nii.gz`, `.mha/.mhd`, `.nrrd`.
**Assumptions**: `dose` is registered to `CT`; `contour` is a binary mask (GTV/primary).
(If you have DICOM RTDOSE/RTSTRUCT, convert to NIfTI first or extend `data/io.py`.)


## Quick start

```bash
# 1) Install deps (CUDA-compatible PyTorch recommended)
pip install -r requirements.txt

# 2) Stage 1: CT-only
python src/train.py --data /path/to/dataset.json --stage 1 --out runs/stage1 --epochs 20

# 3) Evaluate on test split
python src/evaluate.py --data /path/to/dataset.json --ckpt runs/stage1/best.ckpt

# 4) Stage 2: add dose/contour numeric features (+ optional 3D adapters)
python src/train.py --data /path/to/dataset.json --stage 2 --out runs/stage2 --use_dose_adapter --use_contour_adapter

# 5) Stage 3: cross-attention tokens
python src/train.py --data /path/to/dataset.json --stage 3 --out runs/stage3
```

Key flags:

- `--slices 16` (number of CT slices used per case)

- `--img_size 224` (2D encoder input size)

- `--batch_size 2` (per‑case batch; increase with GPU memory)

- `-lr 3e-4` (AdamW)

- `--use_dose_adapter`, `--use_contour_adapter` (enable tiny 3D CNN adapters in Stage 2/3)

- `--cfg src/configs/stageX.yaml` (override any setting via YAML)

Outputs:

`best.ckpt`, `last.ckpt`, `metrics.json`, and `test_predictions.csv` in the run folder.


## Switching to MedImageInsight (optional)

By default, the CT encoder is a frozen 2D timm backbone (ImageNet pretrained).
To use MedImageInsight, implement the stub in `models/encoders.py` (`MIIImageEncoder`) and set `--ct_encoder mii`. (Kept optional to avoid network/API keys.)

## Note

- CT HU is clipped to `[-1000, 1000]` and rescaled to `[-1, 1]`; resampled to `1 mm` iso by default.

- Dose is assumed in `Gy` and scaled by `dose_scale=80` before z‑score; numeric features include DVH stats and `concentric shell` doses around the GTV (no overlays).

- Class imbalance: `--pos_weight` sets BCE positive class weighting (default auto‑computed from train split).

- All stages produce `AUC`, `AUPRC`, `accuracy`, `F1`, plus `confusion matrix`.

## Extensions

- **Speed/Memory:** default `batch_size=2` with `slices=16` fits a typical 24 GB GPU for ViT‑Small; adjust as needed.

- **MedImageInsight**: replace `CT2DEncoder` with `MIIImageEncoder` once wired; keep CT native (no overlays).

- **Survival**: if you later add `time_to_event` and `event_observed` fields, you can drop in a Cox/discrete‑time head — happy to add that head if you want it now.

- **DICOM RT input**: easiest route is to convert RTDOSE/RTSTRUCT to NIfTI first; or extend `data/io.py` with pydicom + `dicompyler-core`.

## requirements.txt

```txt
torch>=2.2
torchvision>=0.17
timm>=0.9.12
numpy>=1.26.4
pandas>=2.2.2
scikit-learn>=1.4.2
scipy>=1.11.4
SimpleITK>=2.3.1
nibabel>=5.2.0
scikit-image>=0.24.0
torchmetrics>=1.4.0
einops>=0.7.0
tqdm>=4.66.5
pyyaml>=6.0.2
matplotlib>=3.8.4