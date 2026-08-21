from __future__ import annotations

import math

import torch


PLANCK_J_S = 6.62607015e-34
LIGHT_M_PER_S = 2.99792458e8


def wavelengths_to_tensor(
    wavelengths_um: list[float] | tuple[float, ...] | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(wavelengths_um, device=device, dtype=dtype)


def wavelengths_um_to_m(wavelengths_um: torch.Tensor) -> torch.Tensor:
    return wavelengths_um * 1e-6


def photon_energy_j(wavelengths_um: torch.Tensor) -> torch.Tensor:
    wavelengths_m = wavelengths_um_to_m(wavelengths_um)
    return (PLANCK_J_S * LIGHT_M_PER_S) / wavelengths_m


def photon_factor_from_irradiance(wavelengths_um: torch.Tensor) -> torch.Tensor:
    wavelengths_m = wavelengths_um_to_m(wavelengths_um)
    return wavelengths_m / (PLANCK_J_S * LIGHT_M_PER_S)


def spectral_weights_or_ones(
    wavelengths_um: torch.Tensor,
    weights: list[float] | tuple[float, ...] | torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return torch.ones_like(wavelengths_um)
    return torch.as_tensor(weights, device=wavelengths_um.device, dtype=wavelengths_um.dtype)


def wavenumber_rad_per_um(wavelengths_um: torch.Tensor, refractive_index: float = 1.0) -> torch.Tensor:
    return (2.0 * math.pi * refractive_index) / wavelengths_um
