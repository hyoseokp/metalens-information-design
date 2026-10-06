"""Raw-to-RGB reconstruction used for the manuscript imaging figures.

Bilinear RGGB demosaic, spatially varying Wiener deconvolution over the
direct signed PSF bank (sensor-side placement, no flip, ``f_um=None``), then
one fixed 3x3 colour matrix.  All arrays are ``[H,W,3]`` float64 in the
image domain.
"""
from __future__ import annotations

import numpy as np
import torch

from .fov_psf import SENSOR_COORDINATE_CONTRACT
from .isp import demosaic_bilinear
from .psf_cache import PSFCache, spatially_varying_wiener_recon

OBJECT_Z_UM = -47_697.0
PHOTOSITE_PITCH_UM = 1.0
WIENER_REG = 0.03


def rggb_mask(h: int, w: int) -> torch.Tensor:
    m = torch.zeros((3, h, w), dtype=torch.float32)
    m[0, 0::2, 0::2] = 1.0
    m[1, 0::2, 1::2] = 1.0
    m[1, 1::2, 0::2] = 1.0
    m[2, 1::2, 1::2] = 1.0
    return m


def demosaic(raw: np.ndarray, mask: torch.Tensor) -> np.ndarray:
    r = demosaic_bilinear(torch.from_numpy(raw.astype(np.float32)), mask)
    return r.permute(1, 2, 0).numpy().astype(np.float64)


def psf_cache_from_bank(bank: dict, nodes: int) -> PSFCache:
    """Select ``nodes`` x ``nodes`` fields of a stored bank by direct signed indexing."""
    psf_all = bank["psf_rgb_normalized"]
    full = psf_all.shape[0]
    if nodes < 2 or nodes > full:
        raise ValueError(f"nodes must be in [2,{full}]")
    sel = np.rint(np.linspace(0, full - 1, nodes)).astype(int)
    if np.unique(sel).size != nodes:
        raise ValueError("duplicate node selection")
    psfs = psf_all[sel[:, None], sel[None, :]].astype(np.float32, copy=True)
    if not np.isfinite(psfs).all() or np.any(psfs < 0.0):
        raise ValueError("invalid PSF values")
    drift = float(np.max(np.abs(psfs.sum(axis=(-2, -1)) - 1.0)))
    if drift > 2.0e-5:
        raise ValueError(f"PSF normalization drift {drift:.2e}")
    ox = bank["object_x_um"]
    oy = bank["object_y_um"]
    return PSFCache(
        field_x_um=torch.from_numpy(ox[sel[:, None], sel[None, :]].mean(axis=0).astype(np.float64)),
        field_y_um=torch.from_numpy(oy[sel[:, None], sel[None, :]].mean(axis=1).astype(np.float64)),
        psf_rgb_pixel=torch.from_numpy(psfs),
        pixel_pitch_um=PHOTOSITE_PITCH_UM,
        object_z_um=OBJECT_Z_UM,
        description=f"{nodes}x{nodes} direct signed PSF bank",
        sensor_x_um=torch.from_numpy(bank["sensor_x_um"][sel].astype(np.float64)),
        sensor_y_um=torch.from_numpy(bank["sensor_y_um"][sel].astype(np.float64)),
        chief_ray_xy_um=torch.from_numpy(bank["chief_ray_xy_um"][sel[:, None], sel[None, :]].astype(np.float64)),
        coordinate_mode=SENSOR_COORDINATE_CONTRACT,
    )


def reconstruct(demosaicked: np.ndarray, cache: PSFCache, reg: float = WIENER_REG) -> np.ndarray:
    # No flip, sensor-side placement, f_um=None.
    image = torch.from_numpy(np.ascontiguousarray(demosaicked).transpose(2, 0, 1)).to(torch.float32)
    rec = spatially_varying_wiener_recon(image, cache, f_um=None, reg=reg, clamp=(0.0, None))
    return rec.numpy().transpose(1, 2, 0).astype(np.float64)


def apply_colour_matrix(M: np.ndarray, linear_rgb: np.ndarray) -> np.ndarray:
    """``out_i = sum_j M_ij rgb_j`` per pixel, clipped at zero."""
    return np.clip(np.einsum("ij,hwj->hwi", M, linear_rgb), 0.0, None)


def to_srgb8(linear: np.ndarray) -> np.ndarray:
    """sRGB transfer of clipped linear RGB as uint8."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.rint(srgb * 255.0).astype(np.uint8)


__all__ = [
    "OBJECT_Z_UM",
    "PHOTOSITE_PITCH_UM",
    "WIENER_REG",
    "apply_colour_matrix",
    "demosaic",
    "psf_cache_from_bank",
    "reconstruct",
    "rggb_mask",
    "to_srgb8",
]
