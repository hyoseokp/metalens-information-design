from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpatialGrid:
    x_um: torch.Tensor
    y_um: torch.Tensor
    xx_um: torch.Tensor
    yy_um: torch.Tensor
    pitch_um: float
    z_um: float
    height: int
    width: int


@dataclass
class FrequencyGrid:
    fx_cyc_per_um: torch.Tensor
    fy_cyc_per_um: torch.Tensor
    fxx_cyc_per_um: torch.Tensor
    fyy_cyc_per_um: torch.Tensor


@dataclass
class CommonGridMeta:
    """Metadata describing the rational-aligned common Fourier grid used to bridge
    pupil-plane and sensor-plane samplings with potentially different pitches.

    The common grid is chosen so that pupil_pad_extent == common_pad_extent (in µm),
    which makes pupil and common spectra share the EXACT same frequency step. This
    eliminates Fourier interpolation error when transferring spectra between grids.

    Pupil pad shape  : (np_pad_h, np_pad_w) at pupil pitch dp
    Common pad shape : (nc_pad_h, nc_pad_w) at sensor pitch ds (>= sensor native shape)
    Sensor native    : (sensor_h, sensor_w) at ds — final crop after common-grid IFFT
    Spectrum scale   : (nc_pad_h * nc_pad_w) / (np_pad_h * np_pad_w) — applied after
                       IFFT to compensate for PyTorch's ifft 1/N normalization
                       difference between pupil and common transform sizes.
    """
    np_pad_h: int
    np_pad_w: int
    nc_pad_h: int
    nc_pad_w: int
    sensor_h: int
    sensor_w: int
    pupil_pitch_um: float
    sensor_pitch_um: float
    common_pad_extent_um: float
    spectrum_amp_scale: float
    pupil_h: int
    pupil_w: int


@dataclass
class ObjectPointBatch:
    coords_um: torch.Tensor
    z_um: torch.Tensor
    spectral_radiance: torch.Tensor


@dataclass
class IncidentBatch:
    field: torch.Tensor
    theta_x_rad: torch.Tensor
    theta_y_rad: torch.Tensor
    distance_um: torch.Tensor


@dataclass
class VectorialFieldBatch:
    """Per-object-point vectorial E-field at the sensor plane."""
    Ex: torch.Tensor  # [N_obj, N_λ, H, W] complex
    Ey: torch.Tensor  # [N_obj, N_λ, H, W] complex
    Ez: torch.Tensor  # [N_obj, N_λ, H, W] complex


@dataclass
class RawDeterministicBatch:
    spectral_irradiance: torch.Tensor
    mu_rgb_e: torch.Tensor
    mu_raw_e: torch.Tensor


@dataclass
class RawNoisyBatch:
    raw_electrons: torch.Tensor
    raw_dn: torch.Tensor


@dataclass
class ForwardOutputs:
    spectral_irradiance: torch.Tensor
    mu_rgb_e: torch.Tensor
    mu_raw_e: torch.Tensor
    raw_electrons: torch.Tensor | None = None
    raw_dn: torch.Tensor | None = None
    rgb: torch.Tensor | None = None
