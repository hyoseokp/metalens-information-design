from __future__ import annotations

import torch

from .types import ObjectPointBatch, SpatialGrid


def scene_to_spectral_radiance(
    scene: torch.Tensor,
    wavelengths_um: torch.Tensor,
    radiance_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if scene.ndim == 2:
        base = scene.unsqueeze(0)
        weights = torch.ones_like(wavelengths_um) if radiance_weights is None else radiance_weights
        return base * weights[:, None, None]
    if scene.ndim == 3 and scene.shape[0] == wavelengths_um.numel():
        return scene
    raise ValueError("scene must be [H,W] or [N_lambda,H,W].")


def flatten_object_radiance(
    spectral_radiance: torch.Tensor,
    object_grid: SpatialGrid,
    threshold: float = 0.0,
) -> ObjectPointBatch:
    if spectral_radiance.ndim != 3:
        raise ValueError("spectral_radiance must have shape [N_lambda,H,W].")
    support_mask = spectral_radiance.amax(dim=0) > threshold
    if not torch.any(support_mask):
        raise ValueError("No object points survived the threshold.")

    coords_um = torch.stack(
        [
            object_grid.xx_um[support_mask],
            object_grid.yy_um[support_mask],
        ],
        dim=-1,
    )
    spectra = spectral_radiance[:, support_mask].transpose(0, 1).contiguous()
    z_um = torch.full(
        (coords_um.shape[0],),
        fill_value=object_grid.z_um,
        device=coords_um.device,
        dtype=coords_um.dtype,
    )
    return ObjectPointBatch(coords_um=coords_um, z_um=z_um, spectral_radiance=spectra)


def slice_object_batch(batch: ObjectPointBatch, start: int, end: int) -> ObjectPointBatch:
    return ObjectPointBatch(
        coords_um=batch.coords_um[start:end],
        z_um=batch.z_um[start:end],
        spectral_radiance=batch.spectral_radiance[start:end],
    )
