import numpy as np
import SimpleITK as sitk
from .io import load_volume

def resample_to_spacing(arr:np.ndarray, spacing:tuple, target=(1.0,1.0,1.0), is_label=False) -> np.ndarray:
    """Resample [z,y,x] volume to target spacing in mm using SimpleITK."""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((spacing[2], spacing[1], spacing[0]))  # sitk expects (x,y,z)
    orig_size = np.array(img.GetSize(), dtype=np.int64)
    orig_sp = np.array(img.GetSpacing(), dtype=np.float32)
    target_sp = np.array((target[2], target[1], target[0]), dtype=np.float32)
    new_size = np.round(orig_size * (orig_sp / target_sp)).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetOutputSpacing(tuple(target_sp.tolist()))
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    out = resampler.Execute(img)
    out_arr = sitk.GetArrayFromImage(out).astype(arr.dtype)
    return out_arr

def window_ct(ct:np.ndarray, w=(-1000, 1000)) -> np.ndarray:
    lo, hi = w
    ct = np.clip(ct, lo, hi)
    ct = (ct - lo) / (hi - lo) * 2 - 1.0
    return ct.astype(np.float32)
