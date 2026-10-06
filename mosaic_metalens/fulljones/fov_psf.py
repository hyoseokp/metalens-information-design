"""Full-field PSF bank on a direct signed Cartesian sensor grid.

Every node is one object point propagated independently by the full-Jones
forward (no quadrant fold, reflection, rotation or recentering).  The node
positions are chosen on the sensor: a signed grid of physical intercepts is
inverted through the 540-nm Snell map to object coordinates, and the chief
ray replayed from those coordinates must land back on the requested
intercepts.  The stored sensor axes and chief-ray centres later fix both PSF
registration and spatial blending in the Wiener reconstruction.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from .field_local import (
    FIELD_LOCAL_REFERENCE_WAVELENGTH_NM,
    chief_ray_sensor_intercept_xy_um,
    reference_half_space_indices,
)
from .scoring import ELECTRON_CALIBRATION, POINT_RADIANCE, READ_NOISE_E
from .sensor_information import build_calibrated_fixed_exposure_transfer
from .types import ObjectPointBatch


ARRAY_ORIENTATION_CONTRACT = "object_aligned_same_array_indices_no_flip_v1"
SENSOR_COORDINATE_CONTRACT = (
    "physical_sensor_intercept_axes_and_replayed_540nm_chief_ray_centres_v1"
)
DEFAULT_GRID_NODES = 8
DEFAULT_SENSOR_MARGIN_UM = 5.0


def physical_sensor_axes(
    engine: Any, *, nodes: int, margin_um: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return a direct full signed Cartesian grid in physical sensor units."""

    if nodes < 2:
        raise ValueError("production FOV grid needs at least two nodes per axis")
    if not math.isfinite(margin_um) or margin_um < 0.0:
        raise ValueError("sensor margin must be finite and non-negative")
    pixel_h, pixel_w = (int(v) for v in engine.spec.resolved_pixel_grid_shape())
    optical_h, optical_w = engine.sensor_grid.height, engine.sensor_grid.width
    pitch_x = float(engine.spec.sensor_plane.pitch_um) * optical_w / pixel_w
    pitch_y = float(engine.spec.sensor_plane.pitch_um) * optical_h / pixel_h
    if not math.isclose(pitch_x, pitch_y, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("production FOV producer requires square photosites")
    extent_x = (pixel_w - 1) * 0.5 * pitch_x - margin_um
    extent_y = (pixel_h - 1) * 0.5 * pitch_y - margin_um
    if extent_x <= 0.0 or extent_y <= 0.0:
        raise ValueError("sensor margin leaves no signed field extent")
    device = engine.device_
    x = torch.linspace(-extent_x, extent_x, nodes, device=device, dtype=torch.float32)
    y = torch.linspace(-extent_y, extent_y, nodes, device=device, dtype=torch.float32)
    return x, y, pitch_x


def invert_sensor_grid_to_object_batch(
    engine: Any,
    sensor_x_um: torch.Tensor,
    sensor_y_um: torch.Tensor,
    *,
    point_radiance: float,
) -> tuple[ObjectPointBatch, torch.Tensor, dict[str, Any]]:
    """Invert the sealed 540-nm Snell map and replay every signed intercept."""

    sy, sx = torch.meshgrid(sensor_y_um, sensor_x_um, indexing="ij")
    sensor_distance = float(engine.sensor_grid.z_um - engine.pupil_grid.z_um)
    object_z = float(engine.spec.object_plane.z_um)
    object_distance = abs(object_z)
    n_in, n_out = reference_half_space_indices(
        engine, wavelength_nm=FIELD_LOCAL_REFERENCE_WAVELENGTH_NM
    )
    radius_sensor = torch.sqrt(sx.square() + sy.square())
    sin_out = radius_sensor / torch.sqrt(radius_sensor.square() + sensor_distance**2)
    sin_in = (n_out / n_in) * sin_out
    if bool((sin_in >= 1.0).any()):
        raise RuntimeError("production signed FOV grid violates Snell/TIR contract")
    radius_object = object_distance * sin_in / torch.sqrt(1.0 - sin_in.square())
    scale = torch.where(radius_sensor > 0.0, radius_object / radius_sensor, 0.0)
    # A target sensor coordinate corresponds to the opposite object coordinate.
    object_x = -sx * scale
    object_y = -sy * scale
    coords = torch.stack((object_x.reshape(-1), object_y.reshape(-1)), dim=-1)
    z = torch.full(
        (coords.shape[0],), object_z, device=engine.device_, dtype=torch.float32
    )
    radiance = torch.full(
        (coords.shape[0], int(engine.wavelengths_um.numel())),
        float(point_radiance), device=engine.device_, dtype=torch.float32,
    )
    batch = ObjectPointBatch(coords_um=coords, z_um=z, spectral_radiance=radiance)
    replay = chief_ray_sensor_intercept_xy_um(
        engine, batch, wavelength_nm=FIELD_LOCAL_REFERENCE_WAVELENGTH_NM
    ).reshape(*sx.shape, 2)
    target = torch.stack((sx, sy), dim=-1)
    error = (replay - target).abs()
    maximum = float(error.max().detach().cpu())
    if maximum > 2.0e-4:
        raise RuntimeError(f"chief-ray sensor replay failed: {maximum:.3e} um")
    audit = {
        "reference_wavelength_nm": FIELD_LOCAL_REFERENCE_WAVELENGTH_NM,
        "chief_ray_replay_max_abs_um": maximum,
        "chief_ray_replay_rms_um": float(
            torch.sqrt(error.square().mean()).detach().cpu()
        ),
        "full_signed_x": bool((sensor_x_um < 0).any() and (sensor_x_um > 0).any()),
        "full_signed_y": bool((sensor_y_um < 0).any() and (sensor_y_um > 0).any()),
        "object_sensor_inversion_x_passed": bool(
            torch.all(object_x[sx != 0.0] * sx[sx != 0.0] < 0.0)
        ),
        "object_sensor_inversion_y_passed": bool(
            torch.all(object_y[sy != 0.0] * sy[sy != 0.0] < 0.0)
        ),
        "passed": maximum <= 2.0e-4,
    }
    if not all(
        audit[name] is True
        for name in (
            "full_signed_x", "full_signed_y", "object_sensor_inversion_x_passed",
            "object_sensor_inversion_y_passed", "passed",
        )
    ):
        raise RuntimeError("signed Cartesian field-coordinate audit failed")
    return batch, replay, audit


def _single_point(batch: ObjectPointBatch, index: int) -> ObjectPointBatch:
    return ObjectPointBatch(
        coords_um=batch.coords_um[index:index + 1],
        z_um=batch.z_um[index:index + 1],
        spectral_radiance=batch.spectral_radiance[index:index + 1],
    )


@torch.no_grad()
def render_direct_psf_grid(
    engine: Any,
    width: torch.Tensor,
    batch: ObjectPointBatch,
    *,
    nodes: int,
    source_spectrum: torch.Tensor,
    electron_calibration: float,
    progress: bool,
    border_ring_photosites: int = 4,
    border_gate_threshold: float = 1.0e-2,
    border_gate_policy: str = "report_only",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one independent, unfolded full-Jones forward per signed field."""

    if border_gate_policy not in {"report_only", "fail"}:
        raise ValueError(
            f"unknown border-energy gate policy {border_gate_policy!r}"
        )
    ring = int(border_ring_photosites)
    if ring <= 0:
        raise ValueError("border_ring_photosites must be positive")
    pixel_shape = tuple(int(v) for v in engine.spec.resolved_pixel_grid_shape())
    raw = np.empty((nodes, nodes, 3, *pixel_shape), dtype=np.float32)
    means = np.empty((nodes, nodes, 3), dtype=np.float64)
    border = np.empty((nodes, nodes, 3), dtype=np.float64)
    for flat in range(nodes * nodes):
        spectral_psf = engine.forward_optics_from_object_batch(
            _single_point(batch, flat), width_map_override=width,
            progress=False, accelerated_4fold=False, accelerated_8fold=False,
        )
        transfer = build_calibrated_fixed_exposure_transfer(
            spectral_psf,
            engine.wavelengths_um,
            source_spectrum,
            engine.cfa_transmission,
            engine.qe,
            pixel_grid_shape=pixel_shape,
            electron_calibration=electron_calibration,
            read_noise_e=READ_NOISE_E,
        )
        broadband = transfer.spectral_electron_psf.sum(dim=1)
        row, column = divmod(flat, nodes)
        raw[row, column] = broadband.detach().cpu().to(torch.float32).numpy()
        totals = broadband.sum(dim=(-2, -1))
        means[row, column] = totals.detach().cpu().numpy()
        # Outer-ring share of the un-normalized field energy, computed before
        # any normalization so window truncation is visible.
        inner = broadband[..., ring:-ring, ring:-ring].sum(dim=(-2, -1))
        border[row, column] = (
            ((totals - inner) / totals).detach().cpu().numpy()
        )
        del spectral_psf, transfer, broadband, totals, inner
        if progress:
            print(f"[fov-psf] direct field {flat + 1}/{nodes * nodes}", flush=True)
        torch.cuda.empty_cache()
    if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(means <= 0.0):
        raise RuntimeError("direct broadband CFA PSF grid is invalid")
    if not np.isfinite(border).all() or np.any(border < 0.0):
        raise RuntimeError("border-energy accounting is invalid")
    fields_exceeding = int(
        (border.max(axis=-1) > float(border_gate_threshold)).sum()
    )
    if border_gate_policy == "fail" and fields_exceeding:
        raise RuntimeError(
            f"{fields_exceeding} field(s) exceed the border-energy gate "
            f"threshold {border_gate_threshold:g}"
        )
    normalized = raw / means[..., None, None].astype(np.float32)
    normalization_error = float(
        np.max(np.abs(normalized.sum(axis=(-2, -1)) - 1.0))
    )
    if normalization_error > 2.0e-5:
        raise RuntimeError("production PSF channel normalization failed")
    return raw, normalized.astype(np.float32), {
        "independent_direct_forward_calls": nodes * nodes,
        "expected_direct_forward_calls": nodes * nodes,
        "symmetry_execution": "direct_no_fold",
        "first_quadrant_reuse": False,
        "reflection_or_rotation_reuse": False,
        "normalization_max_abs_error": normalization_error,
        "channel_energy_e_min": float(means.min()),
        "channel_energy_e_max": float(means.max()),
        "channel_energy_e_per_field": np.round(means, 4).tolist(),
        "border_ring_photosites": ring,
        "border_energy_fraction_max": float(border.max()),
        "border_energy_fraction_per_field": np.round(
            border.max(axis=-1), 6
        ).tolist(),
        "border_energy_gate": {
            "threshold": float(border_gate_threshold),
            "policy": border_gate_policy,
            "fields_exceeding": fields_exceeding,
        },
    }


@torch.no_grad()
def render_psf_bank(
    engine: Any,
    protocol_or_source: Mapping[str, torch.Tensor] | torch.Tensor,
    widths: torch.Tensor,
    *,
    nodes: int = DEFAULT_GRID_NODES,
    sensor_margin_um: float = DEFAULT_SENSOR_MARGIN_UM,
    electron_calibration: float = ELECTRON_CALIBRATION,
    progress: bool = False,
) -> dict[str, Any]:
    """Render the full-field PSF bank of one width map.

    ``protocol_or_source`` is either the prior dict from ``scoring.load_prior``
    or its ``source_spectrum`` tensor ``[L]``.  Returned arrays:

    - ``psf_rgb_raw_e`` ``[nodes,nodes,3,H,W]`` broadband CFA-class electrons
    - ``psf_rgb_normalized`` same layout, unit channel energy
    - ``sensor_x_um``, ``sensor_y_um`` ``[nodes]`` signed intercept axes
    - ``object_x_um``, ``object_y_um`` ``[nodes,nodes]`` inverted object points
    - ``chief_ray_xy_um`` ``[nodes,nodes,2]`` replayed 540-nm centres

    Row index follows ``sensor_y_um``, column index ``sensor_x_um``.
    """

    if isinstance(protocol_or_source, Mapping):
        source_spectrum = protocol_or_source["source_spectrum"]
    else:
        source_spectrum = protocol_or_source
    source_spectrum = torch.as_tensor(
        source_spectrum, device=engine.device_, dtype=torch.float32
    )
    widths = widths.to(engine.device_, torch.float32)
    sensor_x, sensor_y, pixel_pitch = physical_sensor_axes(
        engine, nodes=nodes, margin_um=sensor_margin_um
    )
    field_batch, replay, field_audit = invert_sensor_grid_to_object_batch(
        engine, sensor_x, sensor_y, point_radiance=POINT_RADIANCE
    )
    raw, normalized, audit = render_direct_psf_grid(
        engine, widths, field_batch, nodes=nodes,
        source_spectrum=source_spectrum,
        electron_calibration=float(electron_calibration),
        progress=progress,
    )
    return {
        "psf_rgb_raw_e": raw,
        "psf_rgb_normalized": normalized,
        "sensor_x_um": sensor_x.detach().cpu().numpy().astype(np.float64),
        "sensor_y_um": sensor_y.detach().cpu().numpy().astype(np.float64),
        "object_x_um": field_batch.coords_um[:, 0].reshape(nodes, nodes).detach().cpu().numpy().astype(np.float64),
        "object_y_um": field_batch.coords_um[:, 1].reshape(nodes, nodes).detach().cpu().numpy().astype(np.float64),
        "chief_ray_xy_um": replay.detach().cpu().numpy().astype(np.float64),
        "nodes": nodes,
        "sensor_margin_um": float(sensor_margin_um),
        "photosite_pitch_um": pixel_pitch,
        "object_z_um": float(engine.spec.object_plane.z_um),
        "array_orientation_contract": ARRAY_ORIENTATION_CONTRACT,
        "sensor_coordinate_contract": SENSOR_COORDINATE_CONTRACT,
        "field_coordinate_audit": field_audit,
        "render_audit": audit,
    }


__all__ = [
    "ARRAY_ORIENTATION_CONTRACT",
    "DEFAULT_GRID_NODES",
    "DEFAULT_SENSOR_MARGIN_UM",
    "SENSOR_COORDINATE_CONTRACT",
    "invert_sensor_grid_to_object_batch",
    "physical_sensor_axes",
    "render_direct_psf_grid",
    "render_psf_bank",
]
