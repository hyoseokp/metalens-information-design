"""Full-Jones target-information scorer.

Reconstructs the production Q64 target-information (``I_tar``) evaluation path
from the vendored forward engine. Given a stored width map it returns the
field-quadrature-weighted ``I_tar`` in bit/raw-pixel, reproducing the manuscript
Table 1 values. Forward scoring only; no optimizer, no audit gates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import EngineSpec, PlaneSpec, SensorReadoutSpec, SpectralSpec
from .cfa_library import build_cfa_transmission
from .metalens_response import HybridLUTResponseModel, FullJonesLUTResponseModel
from .pixel_stack import PixelStackConfig
from .pipeline_forward import MetalensImagingEngine
from .field_local import (
    reference_half_space_indices,
    field_local_sensor_shift_xy_um,
)
from .sensor_objective import natural_scene_power_spectrum, _field_object_batch
from .production_multirate import (
    build_calibrated_fine_grid_transfer,
    build_normalized_sliding_target_otf,
    build_q64_full_rgb_target_blocks,
    build_q64_rggb_measurement_blocks,
    gather_q64_prior_blocks,
    target_information_real_field_q64,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "fulljones"

# --- fixed forward-model geometry / calibration (production final-exact) ------
NA = 0.3
D_UM = 208.0
F_UM = D_UM / (2.0 * NA)
PUPIL_GRID = 720
PILLAR_PITCH_UM = 0.29
OBJ_GRID = 416
OBJ_SIZE_UM = 30_000.0
OBJ_DIST_UM = 47_697.0
SENSOR_GRID = 832
SENSOR_PITCH_UM = 0.25
PIXEL_GRID = 208
PAD_FACTOR = 2
CHUNK = 4
CAMERA_NAME = "samsung_galaxy_s20"
WL_NM = (420.0, 450.0, 470.0, 510.0, 540.0, 570.0, 600.0, 635.0, 670.0)
WIDTH_MIN_UM, WIDTH_MAX_UM = 0.10, 0.24

ELECTRON_CALIBRATION = 8.411904861119353e-13   # production frozen scalar
READ_NOISE_E = 1.5
POINT_RADIANCE = 1.0e10
SENSOR_FIELD_MARGIN_UM = 5.0
SCENE_GRID_SPACING_UM = 0.25
SENSOR_PIXEL_PITCH_UM = 1.0

LUT_PATH = DATA / "full_jones_lut.npz"
PRIOR_PATH = DATA / "scene_prior_cov9_production.npz"


@dataclass
class FieldPoint:
    name: str
    theta_deg: float
    azimuth_deg: float = 0.0


# ----------------------------------------------------------------------------
def build_engine(device, *, lut_path=LUT_PATH):
    """Full-Jones final-exact imaging engine on the fixed production geometry."""
    pitch_sen = SENSOR_PITCH_UM
    obj_pitch = OBJ_SIZE_UM / OBJ_GRID
    pixel_pitch = (SENSOR_GRID * pitch_sen) / PIXEL_GRID
    wls_um = [w / 1000.0 for w in WL_NM]
    spec = EngineSpec(
        object_plane=PlaneSpec("obj", OBJ_GRID, OBJ_GRID, obj_pitch, z_um=-OBJ_DIST_UM),
        pupil_plane=PlaneSpec("pup", PUPIL_GRID, PUPIL_GRID, PILLAR_PITCH_UM),
        sensor_plane=PlaneSpec("sen", SENSOR_GRID, SENSOR_GRID, pitch_sen, z_um=F_UM),
        spectral=SpectralSpec(wavelengths_um=wls_um),
        sensor=SensorReadoutSpec(exposure_s=0.001, full_well_e=12000.0,
                                 read_noise_e=1.5, adc_bits=12,
                                 adc_gain_dn_per_e=1.0, black_level_dn=0.0),
        pixel_grid_shape=(PIXEL_GRID, PIXEL_GRID),
        aperture_radius_um=D_UM * 0.5,
        chunk_size=CHUNK, use_cuda_if_available=(torch.device(device).type == "cuda"),
        propagation_pad_factor=PAD_FACTOR,
    )
    base = HybridLUTResponseModel(
        str(DATA.parent / "rcwa" / "torcwa_sq_P290_H1000_o7.npz"),
        oblique_lut_path=str(DATA.parent / "rcwa" / "oblique_correction_H1000.npz"),
    )
    cfa = build_cfa_transmission(
        CAMERA_NAME, torch.tensor(wls_um, dtype=torch.float32, device=device),
        device=device)
    ps = PixelStackConfig.planar_bsi_unit_fill()
    ps.pixel_pitch_um = pixel_pitch
    width_init = torch.full((PUPIL_GRID, PUPIL_GRID), 0.17, dtype=torch.float32)
    engine = MetalensImagingEngine(spec=spec, response_model=base,
                                   width_map_um=width_init,
                                   trainable_width_map=False,
                                   cfa_transmission=cfa, pixel_stack_config=ps)
    # Swap in the rigorous full-Jones response (fail-closed).
    exact = FullJonesLUTResponseModel.from_npz(Path(lut_path), fail_closed=True)
    exact.validate_final_exact_protocol()
    engine.response_model = exact.to(device)
    engine._require_full_jones_response = True
    engine._validate_final_exact_response()
    return engine.to(device)


def load_prior(device, dtype=torch.float64, *, path=PRIOR_PATH):
    """Load the empirical K=L=9 scene prior (basis, covariance, target, source)."""
    with np.load(path, allow_pickle=False) as a:
        arrays = {k: np.asarray(a[k], dtype=np.float64) for k in a.files}
    t = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
    return {
        "scene_spectral_basis": t(arrays["fractional_identity_basis"]),
        "scene_color_covariance": t(arrays["covariance_shape"]),
        "target_color_transform": t(arrays["linear_srgb_transform"]),
        "source_spectrum": t(arrays["source_spectrum"]),
    }


def make_field_points(engine, n=5):
    """25-point square-field quadrature restricted to physical sensor cover."""
    n_in, n_out = (float(x) for x in reference_half_space_indices(engine))
    z_object = abs(float(engine.spec.object_plane.z_um))
    sensor_distance = float(engine.sensor_grid.z_um - engine.pupil_grid.z_um)
    half_x = 0.5 * engine.sensor_grid.width * float(engine.sensor_grid.pitch_um)
    half_y = 0.5 * engine.sensor_grid.height * float(engine.sensor_grid.pitch_um)
    usable_x = half_x - SENSOR_FIELD_MARGIN_UM
    usable_y = half_y - SENSOR_FIELD_MARGIN_UM
    object_half = 0.5 * OBJ_SIZE_UM
    azimuths = (0.0, 22.5, 45.0, 67.5, 90.0)
    azimuth_w = (1.0 / 12.0, 4.0 / 12.0, 2.0 / 12.0, 4.0 / 12.0, 1.0 / 12.0)
    points, weights = [], []
    for radial_index in range(n):
        fraction = math.sqrt((radial_index + 0.5) / n)
        for azimuth, qw in zip(azimuths, azimuth_w):
            az = math.radians(azimuth)
            cos_az, sin_az = abs(math.cos(az)), abs(math.sin(az))
            sensor_radial_limit = min(usable_x / max(cos_az, 1e-15),
                                      usable_y / max(sin_az, 1e-15))
            theta_out = math.atan(sensor_radial_limit / sensor_distance)
            sin_in = (n_out / n_in) * math.sin(theta_out)
            sensor_obj_r = z_object * math.tan(math.asin(sin_in))
            obj_r = min(object_half / max(cos_az, 1e-15),
                        object_half / max(sin_az, 1e-15))
            boundary = min(sensor_obj_r, obj_r)
            radius = fraction * boundary
            theta_in = math.degrees(math.atan(radius / z_object))
            points.append(FieldPoint(f"field_r{radial_index}_az{azimuth:g}deg",
                                     theta_in, azimuth))
            weights.append(qw * boundary ** 2 / n)
    w = torch.tensor(weights, dtype=torch.float64)
    return points, w / w.sum()


class TargetInformationScorer:
    """Production Q64 ``I_tar`` scorer (forward evaluation only)."""

    def __init__(self, engine, protocol, device, dtype=torch.float64,
                 electron_calibration=ELECTRON_CALIBRATION):
        self.engine = engine
        self.device = device
        self.dtype = dtype
        self.basis = protocol["scene_spectral_basis"]
        self.color_cov = protocol["scene_color_covariance"]
        self.color_transform = protocol["target_color_transform"]
        self.source = protocol["source_spectrum"]
        self.electron_calibration = torch.as_tensor(
            electron_calibration, device=device, dtype=dtype).reshape(())
        self.fine_shape = (int(engine.sensor_grid.height), int(engine.sensor_grid.width))
        self.fine_pitch_um = float(engine.sensor_grid.pitch_um)
        scene_psd = natural_scene_power_spectrum(
            self.fine_shape, contrast_rms=0.18,
            k0_cyc_per_pixel=0.02 * self.fine_pitch_um / SENSOR_PIXEL_PITCH_UM,
            beta=2.0, device=device, dtype=dtype)
        production_cov = scene_psd[..., None, None] * self.color_cov
        self.q64_prior_blocks = gather_q64_prior_blocks(production_cov)
        target_otf = build_normalized_sliding_target_otf(
            self.fine_shape, self.color_transform, device=device, dtype=dtype)
        self.q64_target_blocks = build_q64_full_rgb_target_blocks(target_otf).blocks

    def _transfer(self, widths, field_point):
        batch = _field_object_batch(self.engine, field_point, POINT_RADIANCE)
        local_shift = field_local_sensor_shift_xy_um(self.engine, batch)
        spectral_psf = self.engine.forward_optics_from_object_batch(
            batch, width_map_override=widths, sensor_field_shift_xy_um=local_shift)
        return build_calibrated_fine_grid_transfer(
            spectral_psf, self.engine.wavelengths_um, self.source,
            self.engine.cfa_transmission, self.engine.qe, self.basis,
            psf_origin="centered_discrete_sample", psf_spatial_dims=(-2, -1),
            fine_sample_pitch_um=self.fine_pitch_um,
            photosite_pitch_um=SENSOR_PIXEL_PITCH_UM,
            electron_calibration=self.electron_calibration,
            read_noise_e=READ_NOISE_E)

    def target_information_bits(self, widths, field_point):
        transfer = self._transfer(widths, field_point)
        measurement = build_q64_rggb_measurement_blocks(transfer.fine_color_mixing_otf)
        result = target_information_real_field_q64(
            measurement.blocks, self.q64_target_blocks, self.q64_prior_blocks,
            transfer.noise_covariance_e2, fine_grid_shape=self.fine_shape)
        return result.bits_per_raw_pixel.reshape(())

    def score(self, widths, field_points, field_weight):
        widths = widths.to(self.device, self.dtype).clamp(WIDTH_MIN_UM, WIDTH_MAX_UM)
        per_field, total = [], 0.0
        for i, fp in enumerate(field_points):
            bits = float(self.target_information_bits(widths, fp))
            per_field.append(bits)
            total += float(field_weight[i]) * bits
        return total, per_field
