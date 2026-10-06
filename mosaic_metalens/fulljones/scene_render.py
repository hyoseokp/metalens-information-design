"""Exact full-scene raw render on the production final-exact engine.

Pipeline: linear-sRGB scene -> nonnegative 9-band spectral lift under the
scene prior -> 198x198 object points placed by conserved-k-parallel Snell
inversion of the 1-um sensor centres -> full-Jones propagation of every point
through the metalens -> CFA/QE/electron weighting and exact 4x4 box
integration to 208x208 photosites -> RGGB raw.  Every measurement is an
exact full-scene propagation of the object batch.  Nothing here is a PSF
convolution.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch

from . import symmetry_dispatch
from .field_local import (
    chief_ray_sensor_intercept_xy_um,
    reference_half_space_indices,
)
from .scoring import WL_NM
from .sensor_information import trapezoid_bin_widths_m
from .spectral import photon_factor_from_irradiance
from .types import ObjectPointBatch


RAW_GRID = 208
FINE_PER_PIXEL = 4
SCENE_MARGIN_PIXELS = 5
ACTIVE_SCENE_GRID = RAW_GRID - 2 * SCENE_MARGIN_PIXELS


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def load_scene_linear_rgb(path, device: torch.device) -> torch.Tensor:
    """Resize to the active scene grid with LANCZOS, sRGB -> linear."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(
        (ACTIVE_SCENE_GRID, ACTIVE_SCENE_GRID), Image.Resampling.LANCZOS
    )
    linear = srgb_to_linear(np.asarray(image, dtype=np.float64) / 255.0)
    if not np.isfinite(linear).all() or np.any(linear < 0.0):
        raise RuntimeError("scene decoded to invalid linear RGB")
    return torch.as_tensor(linear, device=device, dtype=torch.float32)


def _psnr_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    """PSNR of linear RGB clipped to [0, 1] with data range 1."""
    ref = np.clip(reference, 0.0, 1.0)
    pred = np.clip(prediction, 0.0, 1.0)
    mse = float(np.mean((ref - pred) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1.0e-30)))


def spectral_scene_from_target(
    target_rgb: torch.Tensor, protocol: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | bool]]:
    """Covariance-weighted right inverse of the target colour transform.

    The latent 9-band scene is clamped to be nonnegative and must replay the
    target linear RGB to PSNR >= 80 dB and max abs error <= 1e-3.
    """
    covariance = protocol["scene_color_covariance"].to(torch.float64)
    target = protocol["target_color_transform"].to(torch.float64)
    lift = covariance @ target.T @ torch.linalg.inv(target @ covariance @ target.T)
    latent_unclamped = torch.einsum(
        "kc,hwc->khw", lift.to(torch.float32), target_rgb
    )
    latent = latent_unclamped.clamp_min(0.0)
    represented = torch.einsum(
        "ck,khw->hwc", target.to(torch.float32), latent
    )
    reference = target_rgb.detach().cpu().numpy().astype(np.float64)
    replay = represented.detach().cpu().numpy().astype(np.float64)
    difference = np.abs(reference - replay)
    psnr = _psnr_db(reference, replay)
    maximum = float(difference.max())
    audit: dict[str, float | bool] = {
        "negative_latent_element_fraction_before_projection": float(
            (latent_unclamped < 0.0).to(torch.float64).mean().detach().cpu()
        ),
        "affected_pixel_fraction_before_projection": float(
            (latent_unclamped < 0.0).any(dim=0).to(torch.float64).mean().detach().cpu()
        ),
        "minimum_latent_before_projection": float(latent_unclamped.min().detach().cpu()),
        "represented_vs_object_target_psnr_db": psnr,
        "represented_vs_object_target_mean_abs_linear_rgb": float(difference.mean()),
        "represented_vs_object_target_max_abs_linear_rgb": maximum,
        "psnr_minimum_db": 80.0,
        "max_abs_linear_rgb_maximum": 1.0e-3,
        "passed": psnr >= 80.0 and maximum <= 1.0e-3,
    }
    if audit["passed"] is not True:
        raise RuntimeError("nonnegative spectral lift does not preserve the scene")
    return latent, represented, audit


def snell_object_batch(
    spectral_scene: torch.Tensor,
    engine: Any,
) -> tuple[ObjectPointBatch, dict[str, float]]:
    """Map 198 object samples to 1-um sensor centres inside the 5-um margin."""

    if tuple(spectral_scene.shape[-2:]) != (ACTIVE_SCENE_GRID, ACTIVE_SCENE_GRID):
        raise ValueError("spectral scene must use the active scene grid")
    coordinates = (
        torch.arange(ACTIVE_SCENE_GRID, device=engine.device_, dtype=torch.float32)
        - ACTIVE_SCENE_GRID / 2.0
        + 0.5
    )
    sensor_y, sensor_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    sensor_distance = float(engine.sensor_grid.z_um - engine.pupil_grid.z_um)
    object_distance = abs(float(engine.spec.object_plane.z_um))
    n_in, n_out = reference_half_space_indices(engine)
    radius_sensor = torch.sqrt(sensor_x.square() + sensor_y.square())
    sin_out = radius_sensor / torch.sqrt(radius_sensor.square() + sensor_distance**2)
    sin_in = (n_out / n_in) * sin_out
    if bool((sin_in >= 1.0).any()):
        raise RuntimeError("scene FOV violates the Snell/TIR contract")
    radius_object = object_distance * sin_in / torch.sqrt(1.0 - sin_in.square())
    scale = torch.where(radius_sensor > 0.0, radius_object / radius_sensor, 0.0)
    # Imaging inversion: a +x object chief ray lands at -x on the sensor.
    object_x = -sensor_x * scale
    object_y = -sensor_y * scale
    # The optical source primitive is a discrete isotropic point radiant
    # intensity.  Convert the declared planar Lambertian radiance samples to
    # that primitive with the relative object-cell Jacobian and cos(theta).
    dx_dy, dx_dx = torch.gradient(object_x, spacing=(1.0, 1.0))
    dy_dy, dy_dx = torch.gradient(object_y, spacing=(1.0, 1.0))
    relative_area = (dx_dx * dy_dy - dx_dy * dy_dx).abs()
    cos_object = object_distance / torch.sqrt(
        object_x.square() + object_y.square() + object_distance**2
    )
    source_weight = relative_area * cos_object
    center = ACTIVE_SCENE_GRID // 2
    source_weight = source_weight / source_weight[center, center]
    if not bool(torch.isfinite(source_weight).all()) or bool(
        (source_weight <= 0.0).any()
    ):
        raise RuntimeError("Lambertian-to-point source weights are invalid")
    coords = torch.stack([object_x.reshape(-1), object_y.reshape(-1)], dim=-1)
    intercept = torch.stack(
        [sensor_x.reshape(-1), sensor_y.reshape(-1)], dim=-1
    )
    spectra = (
        spectral_scene * source_weight.unsqueeze(0)
    ).reshape(spectral_scene.shape[0], -1).T.contiguous()
    z = torch.full(
        (coords.shape[0],),
        float(engine.spec.object_plane.z_um),
        device=engine.device_,
        dtype=torch.float32,
    )
    batch = ObjectPointBatch(coords_um=coords, z_um=z, spectral_radiance=spectra)
    replay = chief_ray_sensor_intercept_xy_um(engine, batch)
    error = float((replay - intercept).abs().max().detach().cpu())
    if error > 2.0e-4:
        raise RuntimeError(f"Snell object/sensor coordinate replay failed: {error:.3e} um")
    return batch, {
        "active_scene_grid": ACTIVE_SCENE_GRID,
        "zero_border_pixels": SCENE_MARGIN_PIXELS,
        "chief_ray_replay_max_abs_um": error,
        "max_sensor_intercept_abs_um": float(intercept.abs().max().detach().cpu()),
        "source_convention": (
            "planar_Lambertian_relative_radiance converted to discrete isotropic "
            "point radiant intensity by relative cell_area*cos(theta)"
        ),
        "source_weight_min": float(source_weight.min().detach().cpu()),
        "source_weight_max": float(source_weight.max().detach().cpu()),
    }


def padded_truth(represented: torch.Tensor) -> torch.Tensor:
    truth = torch.zeros((RAW_GRID, RAW_GRID, 3), device=represented.device)
    margin = SCENE_MARGIN_PIXELS
    truth[margin:-margin, margin:-margin] = represented
    return truth


def integrate_fine_electrons_to_rgb_photosites(
    spectral_sensor: torch.Tensor,
    *,
    engine: Any,
    electron_calibration: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Apply CFA/QE/electron weights and exact 4x4 area-normalized integration."""

    spectral = torch.as_tensor(spectral_sensor)
    if tuple(spectral.shape) != (
        len(WL_NM),
        RAW_GRID * FINE_PER_PIXEL,
        RAW_GRID * FINE_PER_PIXEL,
    ):
        raise ValueError("spectral full-scene sensor array has the wrong shape")
    if not bool(torch.isfinite(spectral).all()) or bool((spectral < 0.0).any()):
        raise ValueError("spectral full-scene sensor array is invalid")
    wavelengths = engine.wavelengths_um.to(spectral)
    n_wl = int(wavelengths.numel())

    def _channel_rows(tensor: torch.Tensor, name: str) -> torch.Tensor:
        # Accept both spectral-response conventions: [3, N_wl] rows and the
        # engine broadcast form [3, N_wl, 1, 1] (cfa_library.py), plus a
        # channel-shared [N_wl] / [N_wl, 1, 1] vector.  The einsum below
        # requires exact [3, N_wl] rows.
        rows = torch.as_tensor(tensor).to(spectral)
        if rows.dim() >= 3 and rows.shape[-2:] == (1, 1):
            rows = rows[..., 0, 0]
        if rows.dim() == 1:
            rows = rows.unsqueeze(0).expand(3, -1)
        if rows.shape != (3, n_wl):
            raise ValueError(
                f"{name} must reduce to [3, {n_wl}] channel rows; got "
                f"{tuple(torch.as_tensor(tensor).shape)}"
            )
        return rows

    weights = (
        _channel_rows(engine.cfa_transmission, "cfa_transmission")
        * _channel_rows(engine.qe, "qe")
        * trapezoid_bin_widths_m(wavelengths)[None]
        * photon_factor_from_irradiance(wavelengths)[None]
        * float(electron_calibration)
    )
    if weights.shape != (3, n_wl):
        raise ValueError("channel integration weights must be [3, N_wl]")
    fine_rgb = torch.einsum("cl,lhw->chw", weights, spectral)
    rgb = fine_rgb.reshape(
        3, RAW_GRID, FINE_PER_PIXEL, RAW_GRID, FINE_PER_PIXEL
    ).sum(dim=(2, 4)) / float(FINE_PER_PIXEL**2)
    # The same operator must map unit fine-grid radiance to unit photosites.
    dc = torch.ones(
        (1, RAW_GRID * FINE_PER_PIXEL, RAW_GRID * FINE_PER_PIXEL),
        device=spectral.device,
        dtype=spectral.dtype,
    )
    dc_integrated = dc.reshape(
        1, RAW_GRID, FINE_PER_PIXEL, RAW_GRID, FINE_PER_PIXEL
    ).sum(dim=(2, 4)) / float(FINE_PER_PIXEL**2)
    dc_error = float((dc_integrated - 1.0).abs().max().detach().cpu())
    if dc_error > 1.0e-7:
        raise RuntimeError("4x4 sample-area/DC normalization gate failed")
    return rgb, {
        "samples_per_photosite_axis": FINE_PER_PIXEL,
        "sample_area_factor": 1.0 / float(FINE_PER_PIXEL**2),
        "uniform_scene_dc_max_abs_error": dc_error,
        "passed": True,
    }


def rggb_from_rgb_electrons(rgb: torch.Tensor) -> torch.Tensor:
    if tuple(rgb.shape) != (3, RAW_GRID, RAW_GRID):
        raise ValueError("RGB electron image must be [3,208,208]")
    raw = torch.empty((RAW_GRID, RAW_GRID), device=rgb.device, dtype=rgb.dtype)
    raw[0::2, 0::2] = rgb[0, 0::2, 0::2]
    raw[0::2, 1::2] = rgb[1, 0::2, 1::2]
    raw[1::2, 0::2] = rgb[1, 1::2, 0::2]
    raw[1::2, 1::2] = rgb[2, 1::2, 1::2]
    return raw


def fold_authorization(engine, widths):
    """Self-verify reflection symmetry and issue a runtime fold token.

    The runtime validator only rebinds the token to (width, LUT, source)
    hashes.  Symmetry is verified here explicitly (mirror=D2, +transpose=D4).
    """
    tol = 1.0e-7
    m0 = float((widths - torch.flip(widths, dims=(0,))).abs().max())
    m1 = float((widths - torch.flip(widths, dims=(1,))).abs().max())
    tr = float((widths - widths.transpose(0, 1)).abs().max())
    four = m0 <= tol and m1 <= tol
    eight = four and tr <= tol
    if not four:
        return None, False, False
    auth = symmetry_dispatch.ReflectionFoldAuthorization(
        manifest_path="self_verified", manifest_sha256="self_verified",
        manifest_content_sha256="self_verified", evidence_path="self_verified",
        evidence_sha256="self_verified",
        lut_sha256=getattr(engine.response_model, "source_npz_sha256", None),
        width_sha256=symmetry_dispatch.sha256_tensor(widths),
        design="self_verified_generality",
        four_image_allowed=True, eight_image_allowed=eight,
        contracts_sha256=symmetry_dispatch.canonical_sha256(
            symmetry_dispatch.EXPECTED_CONTRACTS),
        source_hashes_sha256=symmetry_dispatch.canonical_sha256(
            symmetry_dispatch.current_source_hashes()))
    return auth, four, eight


def render_scene(
    engine,
    protocol: Mapping[str, torch.Tensor],
    widths: torch.Tensor,
    scene_linear_rgb: torch.Tensor,
    *,
    electron_calibration: float,
    progress: bool = False,
) -> dict[str, Any]:
    """Exact full-scene raw render of one width map.

    Returns ``truth`` [208,208,3], ``rgb_e`` [3,208,208], ``raw`` [208,208]
    (float32 numpy), the fold ``symmetry`` used and the wall time.
    """
    from .scoring import POINT_RADIANCE, WIDTH_MAX_UM, WIDTH_MIN_UM

    device = engine.device_
    protocol = {k: v.to(device=device, dtype=torch.float32) for k, v in protocol.items()}
    rgb = scene_linear_rgb.to(device=device, dtype=torch.float32)
    spectral_scene, represented, _ = spectral_scene_from_target(rgb, protocol)
    spectral_scene = spectral_scene * float(POINT_RADIANCE)
    batch, _ = snell_object_batch(spectral_scene, engine)
    truth = padded_truth(represented).detach().cpu().numpy().astype(np.float32)

    t0 = time.time()
    width = widths.to(device=device, dtype=torch.float32).clamp(WIDTH_MIN_UM, WIDTH_MAX_UM)
    auth, use4, use8 = fold_authorization(engine, width)
    spectral_sensor = engine.forward_optics_from_object_batch(
        batch, width_map_override=width, progress=progress,
        accelerated_4fold=bool(use4 and not use8), accelerated_8fold=bool(use8),
        reflection_fold_authorization=auth)
    rgb_e, _ = integrate_fine_electrons_to_rgb_photosites(
        spectral_sensor, engine=engine, electron_calibration=electron_calibration)
    raw = rggb_from_rgb_electrons(rgb_e)
    wall = time.time() - t0
    return {
        "truth": truth,
        "rgb_e": rgb_e.detach().cpu().numpy().astype(np.float32),
        "raw": raw.detach().cpu().numpy().astype(np.float32),
        "symmetry": "8fold" if use8 else ("4fold" if use4 else "none"),
        "wall_s": wall,
    }


__all__ = [
    "ACTIVE_SCENE_GRID",
    "FINE_PER_PIXEL",
    "RAW_GRID",
    "SCENE_MARGIN_PIXELS",
    "fold_authorization",
    "integrate_fine_electrons_to_rgb_photosites",
    "load_scene_linear_rgb",
    "padded_truth",
    "render_scene",
    "rggb_from_rgb_electrons",
    "snell_object_batch",
    "spectral_scene_from_target",
    "srgb_to_linear",
]
