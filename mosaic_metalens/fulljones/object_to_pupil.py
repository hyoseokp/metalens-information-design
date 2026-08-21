from __future__ import annotations

import torch

from .spectral import wavenumber_rad_per_um
from .types import IncidentBatch, ObjectPointBatch, SpatialGrid


def compute_incident_field(
    batch: ObjectPointBatch,
    pupil_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    *,
    refractive_index: float | torch.Tensor = 1.0,
) -> IncidentBatch:
    x_obj = batch.coords_um[:, 0][:, None, None]
    y_obj = batch.coords_um[:, 1][:, None, None]
    z_obj = batch.z_um[:, None, None]

    dx = pupil_grid.xx_um[None, :, :] - x_obj
    dy = pupil_grid.yy_um[None, :, :] - y_obj
    dz = torch.as_tensor(pupil_grid.z_um, device=dx.device, dtype=dx.dtype) - z_obj

    distance_um = torch.sqrt(dx.square() + dy.square() + dz.square()).clamp_min(1e-9)
    index = torch.as_tensor(
        refractive_index,
        device=wavelengths_um.device,
        dtype=wavelengths_um.dtype,
    )
    if index.ndim == 0:
        index = index.expand_as(wavelengths_um)
    if index.shape != wavelengths_um.shape:
        raise ValueError(
            "refractive_index must be scalar or have one value per wavelength"
        )
    if not bool(torch.isfinite(index).all()) or not bool((index > 0.0).all()):
        raise ValueError("refractive_index must be finite and positive")
    k0 = (
        wavenumber_rad_per_um(wavelengths_um) * index
    ).view(1, -1, 1, 1)

    # E14: evaluate the spherical phase k0·R in float64. At R ~ 48-115 mm the
    # phase reaches ~1.7e6 rad; float32 quantization (ULP 0.125 rad at d500)
    # imprints quasi-random pixel-scale phase noise of RMS ~0.03-0.05 rad on
    # the pupil, scattering ~0.1-0.3% of each PSF into a white halo. The
    # float64 transient is per-chunk only; the field stays complex64.
    dist64 = torch.sqrt(dx.double().square() + dy.double().square()
                        + dz.double().square()).clamp_min(1e-9)
    phase64 = k0.double() * dist64[:, None, :, :]
    phase32 = torch.remainder(phase64, 2.0 * torch.pi).to(torch.float32)
    del phase64, dist64
    field = torch.exp(1j * phase32) / distance_um[:, None, :, :]

    dz_for_angle = dz.abs().clamp_min(1e-9)
    # theta and distance are wavelength-independent: keep [N_chunk, 1, Hp, Wp]
    # and let downstream broadcast. Bit-identical with previous .expand(...) but
    # avoids 9x memory bloat in EMT's dtype-promotion step.
    theta_x = torch.atan2(dx, dz_for_angle)[:, None, :, :]
    theta_y = torch.atan2(dy, dz_for_angle)[:, None, :, :]

    return IncidentBatch(
        field=field.to(torch.complex64),
        theta_x_rad=theta_x,
        theta_y_rad=theta_y,
        distance_um=distance_um[:, None, :, :],
    )
