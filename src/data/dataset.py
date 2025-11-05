import json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import load_volume
from .preprocess import resample_to_spacing, window_ct
from .dvh import signed_distance_map, compute_numeric_features

def _bbox_zyx(mask):
    idx = np.argwhere(mask>0)
    if idx.size==0: return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    z0,y0,x0 = idx.min(0); z1,y1,x1 = idx.max(0)+1
    return int(z0),int(z1),int(y0),int(y1),int(x0),int(x1)

def _crop_with_margin(arr, bbox, margin=16):
    z0,z1,y0,y1,x0,x1 = bbox
    z0 = max(0, z0 - margin); y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    z1 = min(arr.shape[0], z1 + margin); y1 = min(arr.shape[1], y1 + margin); x1 = min(arr.shape[2], x1 + margin)
    return arr[z0:z1, y0:y1, x0:x1]

def resize_2d(img, out_h, out_w):
    import cv2
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

def even_slices(z0,z1,K):
    L = max(1, z1 - z0)
    if K>=L:  # pad repeats if very thin
        idx = np.linspace(z0, z1-1, num=L, dtype=int).tolist()
        while len(idx)<K: idx += idx[:max(1, K-len(idx))]
        return idx[:K]
    return np.linspace(z0, z1-1, num=K, dtype=int).tolist()

class HNJsonDataset(Dataset):
    """
    Prepares a case-level sample with:
      - ct_slices: [K, 1, H, W]
      - num_feats: [F] (Stage 2/3)
      - dose_patch: [1, Dz, Hy, Wx] (optional adapters)
      - sdf_patch: [1, Dz, Hy, Wx] (optional adapters)
      - shell_tokens: [S, D] (Stage 3 will project these via NumericAdapter inside model)
      - label: scalar 0/1
    """
    def __init__(self, json_path:str, split:str, num_slices=16, img_size=224, stage:int=1,
                 fit_stats:bool=False, stats=None, target_spacing=(1.0,1.0,1.0)):
        self.meta = json.load(open(json_path))
        self.ids = [k for k,v in self.meta.items() if v.get("split","train")==split]
        if split=="val" and len(self.ids)==0:  # convenience: hold 10% of train as val if no explicit val
            all_train = [k for k,v in self.meta.items() if v.get("split","train")=="train"]
            n_val = max(1, int(0.1*len(all_train)))
            self.ids = all_train[:n_val]
        self.num_slices = num_slices
        self.img_size = img_size
        self.stage = stage
        self.target_spacing = target_spacing
        # Stats for numeric feature standardization
        self.stats = stats or {"mean": None, "std": None}
        if fit_stats:
            feats = []
            for k in [k for k in self.meta.keys() if self.meta[k].get("split","train")=="train"]:
                dose, sp = load_volume(self.meta[k]["dose"])
                mask, _ = load_volume(self.meta[k]["contour"])
                dose = resample_to_spacing(dose, sp, target_spacing)
                mask = resample_to_spacing(mask, sp, target_spacing, is_label=True)>0.5
                f = compute_numeric_features(dose, mask, target_spacing)
                arr = np.array(list(f.values()), dtype=np.float32)
                feats.append(arr)
            if len(feats)>0:
                M = np.stack(feats,0)
                self.stats["mean"] = M.mean(0)
                self.stats["std"] = M.std(0) + 1e-6
        # fixed numeric feature keys order (so dim is stable)
        # keep in sync with compute_numeric_features
        self.numeric_keys = ["p2","p5","p50","p95","p98","mean","max","vol_mm3",
                             "shell0_mean","shell0_max","shell1_mean","shell1_max",
                             "shell2_mean","shell2_max","shell3_mean","shell3_max",
                             "shell4_mean","shell4_max","coverage_ge_p95"]
        self.numeric_dim = len(self.numeric_keys)
        # Pos fraction for class weighting
        ys = [(1 if self.meta[k]["label"].lower()=="true" else 0) for k in self.meta if self.meta[k].get("split","train")=="train"]
        self.pos_fraction = (np.sum(ys)/max(1,len(ys))) if len(ys)>0 else 0.5

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        item = self.meta[cid]
        y = 1 if str(item["label"]).lower()=="true" else 0

        ct, ct_sp = load_volume(item["CT"])
        dose, d_sp = load_volume(item["dose"])
        mask, m_sp = load_volume(item["contour"])

        # Resample to common spacing
        ct = resample_to_spacing(ct, ct_sp, self.target_spacing)
        dose = resample_to_spacing(dose, d_sp, self.target_spacing)
        mask = resample_to_spacing(mask, m_sp, self.target_spacing, is_label=True)>0.5

        # Window & normalize CT
        ct = window_ct(ct)

        # Bounding box around mask (fallback to center crop if empty)
        bbox = _bbox_zyx(mask)
        ct_roi = _crop_with_margin(ct, bbox, margin=16)
        dose_roi = _crop_with_margin(dose, bbox, margin=16)
        mask_roi = _crop_with_margin(mask.astype(np.uint8), bbox, margin=16)

        # Choose K evenly spaced slices across the ROI (z-axis)
        z0,z1,_,_,_,_ = _bbox_zyx(mask_roi)
        if z1<=z0: z0,z1 = 0, ct_roi.shape[0]
        z_idx = even_slices(z0, z1, self.num_slices)
        H = W = self.img_size
        ct_slices = []
        for z in z_idx:
            sl = ct_roi[z]
            # pad to square then resize
            pad_h = max(0, ct_roi.shape[1]-ct_roi.shape[2])
            pad_w = max(0, ct_roi.shape[2]-ct_roi.shape[1])
            if pad_h>0 or pad_w>0:
                pad = ((0,0),(0,pad_h),(0,pad_w))
                sl = np.pad(ct_roi[z], ((0,0),(0,pad_w)), mode='edge') if ct_roi.shape[1]==ct_roi.shape[2] else \
                     np.pad(ct_roi[z], ((0,pad_h),(0,0)), mode='edge')
            sl = resize_2d(sl, H, W)
            ct_slices.append(sl[None,...])  # [1,H,W]
        ct_slices = np.stack(ct_slices, 0).astype(np.float32)  # [K,1,H,W]

        # Numeric features (dose/geometry) — Stage 2/3
        feats = compute_numeric_features(dose_roi, mask_roi, self.target_spacing)
        x_num = np.array([feats[k] for k in self.numeric_keys], dtype=np.float32)
        if self.stats["mean"] is not None:
            x_num = (x_num - self.stats["mean"]) / self.stats["std"]

        # Optional 3D patches for adapters
        # Resize/crop to manageable size (e.g., 96^3) if large
        Dz = min(dose_roi.shape[0], 96); Hy = min(dose_roi.shape[1], 96); Wx = min(dose_roi.shape[2], 96)
        dose_patch = dose_roi[:Dz, :Hy, :Wx][None,...].astype(np.float32)  # [1,Dz,Hy,Wx]
        sdf = signed_distance_map(mask_roi, self.target_spacing, clip_mm=50.0)
        sdf_patch = sdf[:Dz, :Hy, :Wx][None,...].astype(np.float32)

        sample = {
            "case_id": cid,
            "label": torch.tensor(y, dtype=torch.long),
            "ct_slices": torch.from_numpy(ct_slices),     # [K,1,H,W]
            "num_feats": torch.from_numpy(x_num),         # [F]
            "dose_patch": torch.from_numpy(dose_patch),   # [1,Dz,Hy,Wx]
            "sdf_patch": torch.from_numpy(sdf_patch)      # [1,Dz,Hy,Wx]
        }

        # For Stage 3, create simple token set from shells: here we reuse numeric features per shell
        # (The model's NumericAdapter turns these into D-dim tokens when used.)
        # We'll bundle per-shell means only (S=5) for tokenization prototype.
        shell_means = [feats[f"shell{i}_mean"] for i in range(5)]
        shell_means = (np.array(shell_means, dtype=np.float32) - (self.stats["mean"][8:18:2] if self.stats["mean"] is not None else 0.0)) \
                        / (self.stats["std"][8:18:2] if self.stats["std"] is not None else 1.0)
        sample["shell_tokens_raw"] = torch.from_numpy(shell_means[None,:])  # [1,S]

        return sample

def collate_cases(batch):
    out = {"case_id": [b["case_id"] for b in batch]}
    out["label"] = torch.stack([b["label"] for b in batch],0)
    out["ct_slices"] = torch.stack([b["ct_slices"] for b in batch],0)
    if "num_feats" in batch[0]:
        out["num_feats"] = torch.stack([b["num_feats"] for b in batch],0)
    if "dose_patch" in batch[0]:
        out["dose_patch"] = torch.stack([b["dose_patch"] for b in batch],0)
    if "sdf_patch" in batch[0]:
        out["sdf_patch"] = torch.stack([b["sdf_patch"] for b in batch],0)
    # Stage 3 helper: expand raw shell features into token vectors later
    if "shell_tokens_raw" in batch[0]:
        # Shape [B,1,S] -> we'll project to [B,S,D] in NumericAdapter within model
        out["shell_tokens"] = torch.stack([b["shell_tokens_raw"] for b in batch],0).squeeze(1)
    return out
