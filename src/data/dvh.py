import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

def percentiles_inside(dose:np.ndarray, mask:np.ndarray, qs=(2,5,50,95,98)):
    vals = dose[mask>0].astype(np.float32)
    if vals.size==0:
        return {f"p{q}": 0.0 for q in qs}
    prc = np.percentile(vals, qs)
    return {f"p{q}": float(v) for q,v in zip(qs, prc)}

def mean_max(dose:np.ndarray, mask:np.ndarray):
    vals = dose[mask>0].astype(np.float32)
    if vals.size==0: return {"mean":0.0,"max":0.0}
    return {"mean": float(vals.mean()), "max": float(vals.max())}

def volume_mm3(mask:np.ndarray, spacing:tuple[float,float,float]):
    vox = np.prod(np.array(spacing, dtype=np.float32))
    return float(mask.astype(np.bool_).sum() * vox)

def signed_distance_map(mask:np.ndarray, spacing:tuple[float,float,float], clip_mm:float=50.0):
    m = mask.astype(bool)
    if m.any():
        outside = distance_transform_edt(~m, sampling=spacing)
        inside = distance_transform_edt(m, sampling=spacing)
        sdf = outside
        sdf[m] = -inside[m]
    else:
        sdf = np.zeros_like(mask, dtype=np.float32)
    sdf = np.clip(sdf, -clip_mm, clip_mm) / clip_mm  # normalize to [-1,1]
    return sdf.astype(np.float32)

def shell_masks(mask:np.ndarray, spacing:tuple[float,float,float], shell_width_mm=3.0, num_shells=5):
    """
    Build concentric ring shells around the mask boundary: [0,3), [3,6), ...
    Returns a list of boolean masks, same shape as input.
    """
    sdf = distance_transform_edt(~mask.astype(bool), sampling=spacing)
    shells = []
    for i in range(num_shells):
        lo = i*shell_width_mm
        hi = (i+1)*shell_width_mm
        shells.append((sdf>=lo) & (sdf<hi))
    return shells

def shell_dose_stats(dose:np.ndarray, shells:list[np.ndarray]):
    out = {}
    for i,sh in enumerate(shells):
        vals = dose[sh].astype(np.float32)
        out[f"shell{i}_mean"] = float(vals.mean()) if vals.size>0 else 0.0
        out[f"shell{i}_max"]  = float(vals.max()) if vals.size>0 else 0.0
    return out

def compute_numeric_features(dose:np.ndarray, mask:np.ndarray, spacing:tuple[float,float,float],
                             dose_scale:float=80.0, num_shells:int=5, shell_w:float=3.0):
    # scale dose roughly to [0,1] range before stats
    d = np.clip(dose / dose_scale, 0.0, 2.0)
    feats = {}
    feats.update(percentiles_inside(d, mask))
    feats.update(mean_max(d, mask))
    feats["vol_mm3"] = volume_mm3(mask, spacing)
    shells = shell_masks(mask, spacing, shell_width_mm=shell_w, num_shells=num_shells)
    feats.update(shell_dose_stats(d, shells))
    # Coverage proxy: fraction of mask with dose >= p95(mask dose) of that case (robust)
    vals = d[mask>0]
    thr = np.percentile(vals, 95) if vals.size>0 else 0.0
    feats["coverage_ge_p95"] = float(((vals>=thr).sum()/max(1,vals.size)))
    return feats
