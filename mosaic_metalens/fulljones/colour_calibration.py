"""Analytic per-design colour calibration for the fixed bilinear/Wiener decoder.

Method ``exact_hyp600_analytic_target_dc_inverse_v1``.  One exact on-axis
spectral DC response of a design and the covariance-weighted right inverse of
the linear-sRGB target give the camera DC matrix ``M L``.  The bilinear-RGGB
plus field-dependent Wiener decoder is replayed on analytic constant channel
probes to get its DC matrix ``D``.  The colour calibration is ``(D M L)^-1``,
applied as ``calibrated_rgb_col = colour_matrix @ reconstructed_sensor_rgb_col``
(``np.einsum('ij,hwj->hwi', colour_matrix, reconstruction)``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .isp import demosaic_bilinear
from .psf_cache import PSFCache, spatially_varying_wiener_recon
from .scoring import POINT_RADIANCE
from .sensor_information import trapezoid_bin_widths_m
from .spectral import photon_factor_from_irradiance
from .types import ObjectPointBatch


def fit_shared_reference_colour_matrix(
    sensor_spectral_dc_e: np.ndarray,
    scene_colour_covariance: np.ndarray,
    target_colour_transform: np.ndarray,
    decoder_dc_matrix: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return the one reference-fitted matrix used for every design.

    Column-vector convention is used throughout.  The covariance-weighted
    right inverse ``L`` maps a target differential ``z`` to the minimum-prior-
    norm spectral differential ``x=Lz``.  The exact reference camera and the
    fixed decoder give ``r=D M L z``.  Its inverse is the colour calibration.
    """

    sensor = np.asarray(sensor_spectral_dc_e, dtype=np.float64)
    covariance = np.asarray(scene_colour_covariance, dtype=np.float64)
    target = np.asarray(target_colour_transform, dtype=np.float64)
    decoder = np.asarray(decoder_dc_matrix, dtype=np.float64)
    if sensor.ndim != 2 or sensor.shape[0] != 3:
        raise ValueError("sensor_spectral_dc_e must have shape [3,L]")
    wavelengths = sensor.shape[1]
    if covariance.shape != (wavelengths, wavelengths):
        raise ValueError("scene colour covariance shape mismatch")
    if target.shape != (3, wavelengths):
        raise ValueError("target colour transform shape mismatch")
    if decoder.shape != (3, 3):
        raise ValueError("decoder DC matrix must be 3x3")
    if not all(np.isfinite(value).all() for value in (sensor, covariance, target, decoder)):
        raise ValueError("calibration operands must be finite")
    if np.any(sensor < 0.0) or np.any(sensor.sum(axis=1) <= 0.0):
        raise ValueError(
            "exact sensor spectral DC response must be nonnegative with positive "
            "energy in every sensor channel"
        )
    covariance = 0.5 * (covariance + covariance.T)
    target_covariance = target @ covariance @ target.T
    lift = covariance @ target.T @ np.linalg.inv(target_covariance)
    target_closure = target @ lift
    target_closure_error = float(np.max(np.abs(target_closure - np.eye(3))))
    if target_closure_error > 2.0e-10:
        raise ValueError("target spectral lift closure failed")
    camera_dc = sensor @ lift
    precalibration = decoder @ camera_dc
    condition = float(np.linalg.cond(precalibration))
    if not math.isfinite(condition) or condition > 1.0e6:
        raise ValueError(f"reference DC calibration matrix is ill-conditioned: {condition}")
    colour_matrix = np.linalg.inv(precalibration)
    closure = colour_matrix @ precalibration
    closure_error = float(np.max(np.abs(closure - np.eye(3))))
    if closure_error > 2.0e-10:
        raise ValueError("reference colour-calibration closure failed")
    return {
        "colour_matrix": colour_matrix,
        "target_spectral_lift": lift,
        "reference_camera_dc_matrix": camera_dc,
        "reference_precalibration_dc_matrix": precalibration,
        "condition_number": condition,
        "target_lift_closure_max_abs": target_closure_error,
        "calibration_closure_max_abs": closure_error,
    }


def _rggb_mask(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    mask[0, 0::2, 0::2] = 1.0
    mask[1, 0::2, 1::2] = 1.0
    mask[1, 1::2, 0::2] = 1.0
    mask[2, 1::2, 1::2] = 1.0
    return mask


@torch.no_grad()
def replay_decoder_dc_matrix(
    cache: PSFCache,
    *,
    image_shape: tuple[int, int],
    wiener_reg: float,
    centre_roi_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    """Replay the actual bilinear/Wiener decoder on analytic constant probes."""

    height, width = (int(value) for value in image_shape)
    if height % 2 or width % 2:
        raise ValueError("production calibration requires an even RGGB image grid")
    if centre_roi_size < 2 or centre_roi_size > min(height, width) or centre_roi_size % 2:
        raise ValueError("centre ROI size must be positive, even, and inside the image")
    if not math.isfinite(wiener_reg) or wiener_reg <= 0.0:
        raise ValueError("Wiener regularization must be finite and positive")
    mask = _rggb_mask(height, width, device=device)
    decoder = np.empty((3, 3), dtype=np.float64)
    off_channel_max = 0.0
    half = centre_roi_size // 2
    cy, cx = height // 2, width // 2
    for input_channel in range(3):
        raw = mask[input_channel]
        demosaicked = demosaic_bilinear(raw, mask)
        reconstruction = spatially_varying_wiener_recon(
            demosaicked,
            cache,
            f_um=None,
            reg=float(wiener_reg),
            clamp=None,
        )
        roi = reconstruction[
            :, cy - half:cy + half, cx - half:cx + half
        ]
        decoder[:, input_channel] = (
            roi.to(torch.float64).mean(dim=(-2, -1)).detach().cpu().numpy()
        )
        off = decoder[:, input_channel].copy()
        off[input_channel] = 0.0
        off_channel_max = max(off_channel_max, float(np.max(np.abs(off))))
    if not np.isfinite(decoder).all() or np.any(np.diag(decoder) <= 0.0):
        raise RuntimeError("analytic decoder DC replay is invalid")
    if off_channel_max > 1.0e-8:
        raise RuntimeError("per-channel decoder unexpectedly mixed colour channels")
    return decoder, {
        "centre_roi_size_pixels": float(centre_roi_size),
        "off_channel_max_abs": off_channel_max,
        "minimum_diagonal_gain": float(np.min(np.diag(decoder))),
        "maximum_diagonal_gain": float(np.max(np.diag(decoder))),
    }


@torch.no_grad()
def exact_reference_sensor_spectral_dc(
    engine: Any,
    width: torch.Tensor,
    *,
    electron_calibration: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return exact on-axis sensor-channel electrons per spectral state unit."""

    wavelengths = engine.wavelengths_um
    wavelength_count = int(wavelengths.numel())
    batch = ObjectPointBatch(
        coords_um=torch.zeros((1, 2), device=engine.device_, dtype=torch.float32),
        z_um=torch.full(
            (1,), float(engine.spec.object_plane.z_um),
            device=engine.device_, dtype=torch.float32,
        ),
        spectral_radiance=torch.full(
            (1, wavelength_count), float(POINT_RADIANCE),
            device=engine.device_, dtype=torch.float32,
        ),
    )
    spectral = engine.forward_optics_from_object_batch(
        batch,
        width_map_override=width,
        progress=False,
        accelerated_4fold=False,
        accelerated_8fold=False,
    )
    fine_h, fine_w = (int(value) for value in spectral.shape[-2:])
    pixel_h, pixel_w = (int(value) for value in engine.spec.resolved_pixel_grid_shape())
    if fine_h % pixel_h or fine_w % pixel_w:
        raise RuntimeError("fine sensor grid does not divide the readout grid")
    sy, sx = fine_h // pixel_h, fine_w // pixel_w
    if (sy, sx) != (4, 4):
        raise RuntimeError("production colour calibration requires exact 4x4 integration")
    integrated = spectral.reshape(
        wavelength_count, pixel_h, sy, pixel_w, sx
    ).sum(dim=(2, 4)) / float(sy * sx)
    spectral_energy = integrated.sum(dim=(-2, -1)).to(torch.float64)

    # Both production CFA and QE tables are channel-resolved and stored as
    # [3,L,1,1].  Collapse only their trailing spatial singleton axes before
    # multiplying them.  Adding a new leading axis to QE here would broadcast
    # [3,L] against [1,3,L] and silently turn the reference response into
    # [1,3,L], which then violates the fixed colour-calibration [3,L]
    # contract.  A wavelength-only QE table remains supported explicitly.
    def collapse_response_table(value: torch.Tensor, name: str) -> torch.Tensor:
        table = torch.as_tensor(
            value, device=engine.device_, dtype=torch.float64,
        )
        while table.ndim > 2 and table.shape[-1] == 1:
            table = table.squeeze(-1)
        if table.ndim > 2:
            raise RuntimeError(
                f"{name} may only have trailing singleton spatial axes"
            )
        return table

    cfa = collapse_response_table(engine.cfa_transmission, "cfa_transmission")
    if tuple(cfa.shape) != (3, wavelength_count):
        raise RuntimeError(
            "production CFA response must have shape "
            f"[3,{wavelength_count}], got {tuple(cfa.shape)}"
        )
    qe = collapse_response_table(engine.qe, "qe")
    if tuple(qe.shape) == (wavelength_count,):
        qe = qe.unsqueeze(0).expand(3, -1)
    elif tuple(qe.shape) != (3, wavelength_count):
        raise RuntimeError(
            "production QE response must have shape "
            f"[{wavelength_count}] or [3,{wavelength_count}], got {tuple(qe.shape)}"
        )
    response = cfa * qe
    weights = (
        response
        * trapezoid_bin_widths_m(wavelengths.to(torch.float64))[None]
        * photon_factor_from_irradiance(wavelengths.to(torch.float64))[None]
        * float(electron_calibration)
    )
    sensor = weights * spectral_energy[None]
    if (
        not bool(torch.isfinite(sensor).all())
        or bool((sensor < 0.0).any())
        or bool((sensor.sum(dim=1) <= 0.0).any())
    ):
        raise RuntimeError("exact reference spectral DC response is invalid")
    return (
        sensor.detach().cpu().numpy().astype(np.float64),
        (wavelengths.detach().cpu().numpy() * 1000.0).astype(np.float64),
        {
            "spectral_energy_min": float(spectral_energy.min().detach().cpu()),
            "spectral_energy_max": float(spectral_energy.max().detach().cpu()),
            "sensor_dc_min_e": float(sensor.min().detach().cpu()),
            "sensor_dc_max_e": float(sensor.max().detach().cpu()),
        },
    )


def colour_matrix(
    sensor_dc: np.ndarray,
    cache: PSFCache,
    cov: np.ndarray,
    target: np.ndarray,
    *,
    wiener_reg: float = 0.03,
    centre_roi_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Per-design 3x3 colour matrix: decoder replay on ``cache`` then DC-inverse fit.

    ``sensor_dc`` [3,L] from :func:`exact_reference_sensor_spectral_dc`,
    ``cov``/``target`` the scene prior covariance [L,L] and linear-sRGB target
    [3,L].  Image grid is the 208x208 production readout.
    """
    decoder, _ = replay_decoder_dc_matrix(
        cache, image_shape=(208, 208), wiener_reg=wiener_reg,
        centre_roi_size=centre_roi_size, device=device,
    )
    cov = np.asarray(cov, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return fit_shared_reference_colour_matrix(sensor_dc, cov, target, decoder)["colour_matrix"]
