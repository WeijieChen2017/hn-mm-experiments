from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def _sitk_read(path:str) -> tuple[np.ndarray, tuple[float,float,float]]:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
    sp = img.GetSpacing()  # (x,y,z)
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
    return arr, spacing

def load_volume(path:str) -> tuple[np.ndarray, tuple[float,float,float]]:
    """
    Load a 3D volume from NIfTI/MHA/NRRD using SimpleITK or nibabel.
    Returns array [z,y,x] and spacing (z,y,x) in mm.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in [".nii", ".gz", ".mha", ".mhd", ".nrrd"]:
        # favor SimpleITK for spacing/origin handling
        return _sitk_read(str(p))
    # fallback to nibabel if needed
    nii = nib.load(str(p))
    arr = nii.get_fdata().astype(np.float32)
    if arr.ndim==4: arr = arr[...,0]
    arr = np.transpose(arr, (2,1,0))  # [x,y,z] -> [z,y,x]
    hdr = nii.header
    zooms = hdr.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return arr, spacing
