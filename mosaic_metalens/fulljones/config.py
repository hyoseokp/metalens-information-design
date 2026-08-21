from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch


class ResponseModelType(Enum):
    EMT = "emt"
    RCWA_LOOKUP = "rcwa_lookup"


@dataclass(frozen=True)
class ResponseModelConfig:
    model_type: ResponseModelType = ResponseModelType.EMT
    # EMT parameters (defaults match SiN-on-SiO2 standard config)
    h_um: float = 0.5
    P_um: float = 0.29
    n_cl: float = 1.0
    h_slab_um: float = 0.0
    kappa: float = 0.47
    include_fp: bool = True
    # Polarization: "te", "tm", or "unpolarized" (default: 45° → 0.5*TE + 0.5*TM)
    polarization: str = "unpolarized"
    # RCWA lookup path (required when model_type == RCWA_LOOKUP)
    rcwa_data_path: str | None = None
    rcwa_wl_min_nm: float = 400.0
    rcwa_wl_max_nm: float = 700.0


@dataclass(frozen=True)
class PlaneSpec:
    name: str
    height: int
    width: int
    pitch_um: float
    z_um: float = 0.0


@dataclass(frozen=True)
class SpectralSpec:
    wavelengths_um: Sequence[float]
    radiance_weights: Sequence[float] | None = None


@dataclass(frozen=True)
class SensorReadoutSpec:
    exposure_s: float = 1.0
    full_well_e: float = 12000.0
    read_noise_e: float = 1.5
    dark_current_e_per_s: float = 0.0
    prnu_std: float = 0.0
    dsnu_e_std: float = 0.0
    adc_bits: int = 12
    adc_gain_dn_per_e: float = 1.0
    black_level_dn: float = 0.0


@dataclass(frozen=True)
class EngineSpec:
    object_plane: PlaneSpec
    pupil_plane: PlaneSpec
    sensor_plane: PlaneSpec
    spectral: SpectralSpec
    sensor: SensorReadoutSpec
    pixel_grid_shape: tuple[int, int] | None = None
    aperture_radius_um: float | None = None
    chunk_size: int = 256
    propagation_index: float = 1.0
    propagation_pad_factor: int = 1
    # When True, pupil and sensor planes must have identical (height, width, pitch_um).
    # Restores legacy single-grid propagation. Default False enables the cross-grid
    # path which lets pupil pitch match the metasurface pillar pitch independent of
    # the sensor pitch (which must divide the pixel pitch evenly).
    propagation_strict_grid_match: bool = False
    use_cuda_if_available: bool = True
    default_dtype: torch.dtype = torch.float32

    def resolve_device(self) -> torch.device:
        if self.use_cuda_if_available and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def resolved_pixel_grid_shape(self) -> tuple[int, int]:
        if self.pixel_grid_shape is not None:
            return self.pixel_grid_shape
        return (self.sensor_plane.height, self.sensor_plane.width)
