"""Field-local coordinate contract shared by optimization and diagnostics."""

from __future__ import annotations

import torch


FIELD_LOCAL_REFERENCE_WAVELENGTH_NM = 540.0
GLOBAL_SENSOR_FRAME = "global_sensor"
FIELD_LOCAL_FRAME = "field_local_540nm"


def reference_half_space_indices(
    engine,
    *,
    wavelength_nm: float = FIELD_LOCAL_REFERENCE_WAVELENGTH_NM,
) -> tuple[torch.Tensor, torch.Tensor]:
    wavelength = torch.tensor(
        [float(wavelength_nm) / 1000.0],
        device=engine.device_,
        dtype=torch.float32,
    )
    n_in = engine.response_model.incident_refractive_index(wavelength)[0]
    n_out = engine.response_model.output_refractive_index(wavelength)[0]
    if (
        not bool(torch.isfinite(n_in).detach())
        or not bool(torch.isfinite(n_out).detach())
        or not bool((n_in > 0).detach())
        or not bool((n_out > 0).detach())
    ):
        raise RuntimeError("field-local reference medium indices are invalid")
    return n_in, n_out


def chief_ray_sensor_intercept_xy_um(
    engine,
    object_batch,
    *,
    wavelength_nm: float = FIELD_LOCAL_REFERENCE_WAVELENGTH_NM,
) -> torch.Tensor:
    """Trace the chief ray using conserved tangential wavevector."""

    coordinates = torch.as_tensor(
        object_batch.coords_um, device=engine.device_, dtype=torch.float32
    )
    z_object = torch.as_tensor(
        object_batch.z_um, device=engine.device_, dtype=torch.float32
    )
    if coordinates.ndim != 2 or int(coordinates.shape[-1]) != 2:
        raise ValueError("object_batch.coords_um must have shape [N,2]")
    if tuple(z_object.shape) != (int(coordinates.shape[0]),):
        raise ValueError("object_batch.z_um must have shape [N]")
    if bool((z_object.detach() == 0).any()):
        raise ValueError("field-local chief ray requires nonzero object distance")
    n_in, n_out = reference_half_space_indices(
        engine, wavelength_nm=wavelength_nm
    )
    distance = torch.sqrt(
        coordinates.square().sum(dim=-1) + z_object.square()
    )
    sin_in_xy = -coordinates / distance[:, None]
    sin_out_xy = (n_in / n_out) * sin_in_xy
    sin_out_squared = sin_out_xy.square().sum(dim=-1)
    if bool((sin_out_squared.detach() >= 1.0).any()):
        raise ValueError(
            "field chief ray is evanescent/TIR in the declared output medium"
        )
    cos_out = torch.sqrt((1.0 - sin_out_squared).clamp_min(0.0))
    sensor_distance = float(engine.sensor_grid.z_um - engine.pupil_grid.z_um)
    return sensor_distance * sin_out_xy / cos_out[:, None]


def field_local_sensor_shift_xy_um(engine, object_batch) -> torch.Tensor:
    """Common 540-nm shift placing the chief ray on the real centre sample."""

    coordinates = torch.as_tensor(
        object_batch.coords_um, device=engine.device_, dtype=torch.float32
    )
    chief_xy = chief_ray_sensor_intercept_xy_um(engine, object_batch)
    origin_x = (
        engine.sensor_grid.width // 2
        - engine.sensor_grid.width / 2.0
        + 0.5
    ) * float(engine.sensor_grid.pitch_um)
    origin_y = (
        engine.sensor_grid.height // 2
        - engine.sensor_grid.height / 2.0
        + 0.5
    ) * float(engine.sensor_grid.pitch_um)
    target = coordinates.new_tensor([origin_x, origin_y]).expand_as(coordinates)
    return target - chief_xy


__all__ = [
    "FIELD_LOCAL_FRAME",
    "FIELD_LOCAL_REFERENCE_WAVELENGTH_NM",
    "GLOBAL_SENSOR_FRAME",
    "chief_ray_sensor_intercept_xy_um",
    "field_local_sensor_shift_xy_um",
    "reference_half_space_indices",
]
