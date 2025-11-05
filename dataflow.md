## Data Flow Overview

This document explains how data is loaded, preprocessed, batched, modeled, trained, and evaluated in this project.

### High-level pipeline
- Input metadata: a JSON file mapping `case_id` to filepaths and labels
  - Keys: `CT`, `dose`, `contour`, `label`, `split`
- Loading: 3D medical volumes are read and resampled to a common spacing
- Preprocessing: CT windowing, ROI crop around contour, K evenly spaced 2D slices
- Feature building: numeric dose/geometry features, optional 3D patches (dose, signed distance)
- Batching: collate to tensors for model consumption
- Modeling: CT slice encoder + optional adapters, pooled/fused to binary head
- Training/Eval: standard BCE-with-logits; metrics tracked (AUC, AUPRC, F1, Acc)
- Outputs: checkpoints (`best.ckpt`, `last.ckpt`), `metrics.json`, and test CSV predictions

## Data ingestion
The dataset consumes a JSON metadata file. For each `case_id`:
- `CT`: path to CT volume
- `dose`: path to dose volume
- `contour`: path to ROI mask
- `label`: boolean-like classification target
- `split`: `train` | `val` | `test`

Volumes are loaded via SimpleITK or nibabel, returning array `[z,y,x]` and `(z,y,x)` spacing.

```1:31:hn-mm-experiments/src/data/io.py
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

# load_volume -> (arr[z,y,x], spacing[z,y,x])
```

## Preprocessing pipeline
1) Resample all volumes to a target spacing (default 1 mm isotropic).

```5:21:hn-mm-experiments/src/data/preprocess.py
def resample_to_spacing(arr:np.ndarray, spacing:tuple, target=(1.0,1.0,1.0), is_label=False) -> np.ndarray:
    # ... SimpleITK resample to target spacing ...
    return out_arr
```

2) CT windowing to [-1,1].

```23:27:hn-mm-experiments/src/data/preprocess.py
def window_ct(ct:np.ndarray, w=(-1000, 1000)) -> np.ndarray:
    lo, hi = w
    ct = np.clip(ct, lo, hi)
    ct = (ct - lo) / (hi - lo) * 2 - 1.0
    return ct.astype(np.float32)
```

3) ROI extraction around the contour’s bounding box with margin, then select K evenly spaced axial slices and resize to `img_size`.

```102:126:hn-mm-experiments/src/data/dataset.py
# Compute bbox on mask; crop CT/dose/mask ROI; select K slices; resize to HxW; stack -> [K,1,H,W]
```

4) Numeric features from dose and geometry, standardized by train split stats if available. Includes percentiles, mean/max, volume, multi-shell dose stats, and coverage proxy.

```53:68:hn-mm-experiments/src/data/dvh.py
def compute_numeric_features(dose:np.ndarray, mask:np.ndarray, spacing:tuple[float,float,float],
                             dose_scale:float=80.0, num_shells:int=5, shell_w:float=3.0):
    # percentiles, mean/max, volume, shell stats, coverage_ge_p95
    return feats
```

5) Optional 3D patches for adapters: down-cropped dose and signed distance map.

```133:139:hn-mm-experiments/src/data/dataset.py
# dose_patch: [1,Dz,Hy,Wx]
# sdf_patch:  [1,Dz,Hy,Wx]
```

6) Stage 3 helper tokens: per-shell means are assembled for tokenization.

```149:156:hn-mm-experiments/src/data/dataset.py
# shell_tokens_raw: [1,S] (S=5)
```

## Sample structure emitted by the dataset
Each `__getitem__` returns a dictionary (per case):
- `case_id`: string
- `label`: tensor scalar 0/1
- `ct_slices`: `[K,1,H,W]` float32
- `num_feats`: `[F]` float32 (F = 19; fixed key order)
- `dose_patch`: `[1,Dz,Hy,Wx]` float32
- `sdf_patch`: `[1,Dz,Hy,Wx]` float32
- `shell_tokens_raw`: `[1,S]` float32 (S=5)

Batched via `collate_cases` into:

```159:173:hn-mm-experiments/src/data/dataset.py
def collate_cases(batch):
    out = {"case_id": [...],
           "label": [B,],
           "ct_slices": [B,K,1,H,W],
           "num_feats": [B,F],
           "dose_patch": [B,1,Dz,Hy,Wx],
           "sdf_patch": [B,1,Dz,Hy,Wx],
           "shell_tokens": [B,S] }
    return out
```

## Model data flow
There are three stages (controlled by config/CLI). All use a 2D slice encoder over CT, then fuse optional modalities.

### CT encoder and pooling
- `CT2DEncoder` (timm backbone) produces per-slice embeddings `[B,K,D]` from `[B,K,1,H,W]`.
- `AttnPool1D` pools across K slices to `[B,D]` when used directly.

```18:26:hn-mm-experiments/src/models/encoders.py
def encode_slices(self, x:torch.Tensor)->torch.Tensor:
    # x: [B,K,1,H,W] -> tokens [B,K,D]
    return z.view(B, K, -1)
```

```4:15:hn-mm-experiments/src/models/fusion.py
class AttnPool1D(nn.Module):
    # tokens [B,T,D] -> pooled [B,D]
```

### Adapters
Optional modality adapters map non-CT inputs to the model width `D`:
- `NumericAdapter`: `[B,F] -> [B,D]`
- `DoseAdapter3D`: 3D CNN over `[B,1,Dz,Hy,Wx] -> [B,D]`
- `ContourAdapter3D`: 3D CNN over signed distance maps `[B,1,Dz,Hy,Wx] -> [B,D]`

```5:17:hn-mm-experiments/src/models/adapters.py
class NumericAdapter(nn.Module):
    def forward(self, x):  # x: [B,F]
        return self.proj(x)
```

```19:34:hn-mm-experiments/src/models/adapters.py
class DoseAdapter3D(nn.Module):
    def forward(self, x):  # [B,1,Dz,Hy,Wx]
        z = self.net(x).flatten(1)
        return self.proj(z)
```

```36:51:hn-mm-experiments/src/models/adapters.py
class ContourAdapter3D(nn.Module):
    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.proj(z)
```

### Stage behavior
- Stage 1: CT only. Pool slice tokens; feed to binary head.
- Stage 2: CT pooled + concatenated adapters (numeric/dose/contour) -> binary head.
- Stage 3: Build a token sequence: CT slice tokens + optional numeric shell tokens + adapter vectors (as 1-token each); CT tokens query others via cross-attention blocks; pool CT queries; binary head.

```61:102:hn-mm-experiments/src/train.py
class StageModel(nn.Module):
    def forward(self, batch):
        ct_tokens = self.ct.encode_slices(batch["ct_slices"])  # [B,K,D]
        if self.stage==3:
            # concat tokens (CT + optional)
            for blk in self.ca_blocks:
                tokens = blk(q=tokens[:, :K, :], kv=tokens)
            pooled = self.pool(tokens[:, :K, :])
            logits = self.head(pooled)
            return logits
        else:
            pooled = self.pool(ct_tokens)
            feats = [pooled, optional_adapter_feats...]
            fused = torch.cat(feats, dim=1)
            logits = self.head(fused)
            return logits
```

The binary classification head maps the fused/pool vector to a single logit:

```3:13:hn-mm-experiments/src/models/heads.py
class BinaryHead(nn.Module):
    def forward(self, x): return self.net(x)
```

Cross-attention block used in Stage 3:

```5:33:hn-mm-experiments/src/models/attention.py
class CrossAttentionBlock(nn.Module):
    # CT queries attend to all tokens; residual + MLP
```

## Training loop and metrics
- Datasets: train/val splits are constructed from the JSON file; if no explicit `val`, 10% of train is used.
- Loss: BCEWithLogits with optional positive-class weighting auto-derived from training prevalence.
- Optimizer: AdamW
- Checkpoints: save `best.ckpt` by validation AUC and `last.ckpt` each epoch; metrics persisted to `metrics.json`.

```137:197:hn-mm-experiments/src/train.py
def main():
    train_set = HNJsonDataset(..., split="train", ...)
    val_set = HNJsonDataset(..., split="val", ...)
    loss_fn = make_bce_loss(pos_weight=pos_w)
    for epoch in range(...):
        tr = train_one_epoch(...)
        va = validate(...)
        if va["auroc"] > best_auc: save best.ckpt
        save_json(history, out/"metrics.json")
```

Metrics tracked during training/evaluation:

```4:27:hn-mm-experiments/src/utils/metrics.py
class BinaryMetrics:
    # auroc, auprc, f1, acc
```

## Evaluation flow
- Build the same model architecture as training, load `--ckpt`.
- Iterate over test set, gather probabilities, compute metrics, and write `test_predictions.csv` next to the checkpoint.

```20:59:hn-mm-experiments/src/evaluate.py
def main():
    test_set = HNJsonDataset(..., split="test", ...)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    # run inference, update metrics, save CSV
```

## Configuration
YAML configs under `src/configs/` set defaults per stage; CLI flags override them.

```1:9:hn-mm-experiments/src/configs/stage2.yaml
stage: 2
slices: 16
img_size: 224
batch_size: 2
epochs: 25
ct_encoder: vit_small_patch16_224
use_dose_adapter: true
use_contour_adapter: true
```

## Inputs and outputs summary
- Inputs: JSON metadata + referenced volumes (`CT`, `dose`, `contour`)
- Model inputs per batch: `ct_slices`, optional `num_feats`, `dose_patch`, `sdf_patch`, `shell_tokens`
- Outputs:
  - Training: `runs/stageX/best.ckpt`, `runs/stageX/last.ckpt`, `runs/stageX/metrics.json`
  - Evaluation: `test_predictions.csv` adjacent to the loaded checkpoint

## Stage-by-stage minimal data use
- Stage 1: `ct_slices` only
- Stage 2: `ct_slices` + `num_feats` (+ optionally `dose_patch` / `sdf_patch` if enabled)
- Stage 3: token fusion of CT + shells (+ optional adapters), with cross-attention

