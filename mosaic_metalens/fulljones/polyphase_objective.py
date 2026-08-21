"""Optimizer-facing exact periodic-RGGB A-optimal objective.

This module is the deliberately small bridge between the calibrated
common-exposure sensor transfer and :mod:`engine2.polyphase_mmse`.  It keeps
the complete complex OTF, exact 2x2 RGGB phase/alias operator, absolute
electron gain, and shot/read-noise covariance.  No row-wise DC normalization
or design-wise exposure matching is performed.

The model is local periodic LSI.  It is therefore a differentiable design
objective and matched local reconstruction risk, not a claim that a finite,
spatially varying raw image is globally circulant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Sequence

import torch

from .bayer_polyphase import (
    build_aliased_scene_covariance,
    build_rggb_polyphase_operator,
    rggb_phase_noise_variances,
)
from .polyphase_mmse import (
    PolyphaseAOptimalResult,
    aggregate_real_gaussian_information_bpp,
    build_alias_block_target_transform,
    rggb_polyphase_a_optimal_risk,
)
from .sensor_information import (
    InformationMode,
    SensorInformationTransfer,
    build_calibrated_fixed_exposure_transfer,
    color_mixing_otf,
)
from .sensor_objective import natural_scene_power_spectrum
from .types import ObjectPointBatch


def _require_frozen_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.requires_grad:
        raise ValueError(f"{name} must be frozen (requires_grad=False)")
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _reject_boolean_numeric(value: object, name: str) -> None:
    """Reject booleans before tensor dtype conversion can turn them into 0/1."""

    seen: set[int] = set()

    def visit(candidate: object) -> None:
        if isinstance(candidate, bool):
            raise TypeError(f"{name} must not contain boolean values")
        if isinstance(candidate, torch.Tensor):
            if candidate.dtype == torch.bool:
                raise TypeError(f"{name} must not contain boolean values")
            return
        if isinstance(candidate, Mapping):
            marker = id(candidate)
            if marker in seen:
                return
            seen.add(marker)
            for child in candidate.values():
                visit(child)
            return
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            marker = id(candidate)
            if marker in seen:
                return
            seen.add(marker)
            for child in candidate:
                visit(child)

    visit(value)


def _frozen_real_tensor(
    value: object,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    _reject_boolean_numeric(value, name)
    raw = torch.as_tensor(value, device=device)
    if raw.is_complex():
        raise TypeError(f"{name} must be real")
    return _require_frozen_finite(raw.to(dtype=dtype), name)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float32, torch.complex64):
        return torch.float32
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    raise ValueError("expected float32/64 or complex64/128")


def _frozen_finite_scalar(value: object, name: str) -> float:
    """Convert one frozen real scalar without bool or autograd detachment."""

    _reject_boolean_numeric(value, name)
    raw = torch.as_tensor(value)
    if raw.is_complex():
        raise TypeError(f"{name} must be real")
    _require_frozen_finite(raw, name)
    if raw.numel() != 1:
        raise ValueError(f"{name} must be one scalar")
    result = float(raw.detach().cpu().reshape(()).item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resolved_engine_storage_device(
    declared_device: torch.device | str, storage_device: torch.device | str
) -> torch.device:
    """Resolve an index-free CUDA declaration against an actual engine tensor.

    ``torch.device("cuda")`` is a valid engine declaration but compares unequal
    to the materialized tensor device ``torch.device("cuda:0")``.  The storage
    tensor is authoritative only when the declared device type agrees and any
    explicitly declared index is identical; explicit cross-device mismatches
    still fail closed.
    """

    declared = torch.device(declared_device)
    storage = torch.device(storage_device)
    if declared.type != storage.type or (
        declared.index is not None and declared.index != storage.index
    ):
        raise ValueError("engine.device_ conflicts with engine.width_map_um.device")
    return storage


def _color_protocol(
    scene_spectral_basis: torch.Tensor,
    target_color_transform: torch.Tensor,
    target_color_weight: torch.Tensor | None,
    *,
    wavelength_count: int,
    device: torch.device,
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    basis = _frozen_real_tensor(
        scene_spectral_basis,
        "scene_spectral_basis",
        device=device,
        dtype=real_dtype,
    )
    if basis.ndim != 2 or basis.shape[1] != wavelength_count or basis.shape[0] <= 0:
        raise ValueError("scene_spectral_basis must have shape [K, wavelength_count]")
    latent_channels = int(basis.shape[0])

    color_transform = _frozen_real_tensor(
        target_color_transform,
        "target_color_transform",
        device=device,
        dtype=real_dtype,
    )
    if (
        color_transform.ndim != 2
        or color_transform.shape[1] != latent_channels
        or color_transform.shape[0] <= 0
    ):
        raise ValueError("target_color_transform must have shape [C_target, K]")

    color_weight: torch.Tensor | None = None
    if target_color_weight is not None:
        color_weight = _frozen_real_tensor(
            target_color_weight,
            "target_color_weight",
            device=device,
            dtype=real_dtype,
        )
        target_channels = int(color_transform.shape[0])
        if color_weight.shape != (target_channels, target_channels):
            raise ValueError("target_color_weight must have shape [C_target, C_target]")
    return basis, color_transform, color_weight


def rggb_polyphase_a_optimal_from_transfer(
    transfer: SensorInformationTransfer,
    scene_spectral_basis: torch.Tensor,
    full_grid_scene_covariance: torch.Tensor,
    target_color_transform: torch.Tensor,
    *,
    target_color_weight: torch.Tensor | None = None,
    aggregation_weights: torch.Tensor | None = None,
    jitter: float = 0.0,
    condition_number_limit: float | None = None,
    maximum_whitened_signal_power: float | None = None,
    return_channel_diagnostics: bool = False,
    precomputed_aliased_covariance: torch.Tensor | None = None,
    return_full_posterior: bool = True,
) -> PolyphaseAOptimalResult:
    """Evaluate full-fine-grid RGB/XYZ posterior risk from one transfer.

    ``full_grid_scene_covariance`` is the latent covariance on the unshifted
    fine FFT grid, with shape ``[..., H, W, K, K]``.  The color transform and
    optional color weight are repeated over *all four* aliases, so this is a
    full-pixel reconstruction objective rather than a baseband-only score.

    The returned primary objective is the mean target-channel A-risk.  When
    ``aggregation_weights`` is supplied it is additionally normalized and
    averaged over every leading field/frequency sample by the stable core.

    Only ``CALIBRATED_FIXED_EXPOSURE`` transfers are accepted.  In particular,
    row-wise or global fixed-signal transfers are shape-only/auto-exposure
    diagnostics and would reintroduce design-dependent exposure scaling.  All
    scientific quadrature weights are likewise required to be frozen.
    """

    if transfer.mode is not InformationMode.CALIBRATED_FIXED_EXPOSURE:
        raise ValueError(
            "exact optimizer objective requires a calibrated fixed common "
            "exposure transfer"
        )
    jitter_value = _frozen_finite_scalar(jitter, "jitter")
    condition_limit_value = (
        None
        if condition_number_limit is None
        else _frozen_finite_scalar(condition_number_limit, "condition_number_limit")
    )
    signal_power_limit_value = (
        None
        if maximum_whitened_signal_power is None
        else _frozen_finite_scalar(
            maximum_whitened_signal_power, "maximum_whitened_signal_power"
        )
    )

    spectral_otf = torch.as_tensor(transfer.spectral_otf_e)
    if not spectral_otf.is_complex():
        raise ValueError("transfer.spectral_otf_e must be complex")
    real_dtype = _real_dtype(spectral_otf.dtype)
    basis, color_transform, color_weight = _color_protocol(
        scene_spectral_basis,
        target_color_transform,
        target_color_weight,
        wavelength_count=int(spectral_otf.shape[-3]),
        device=spectral_otf.device,
        real_dtype=real_dtype,
    )
    mixing = color_mixing_otf(transfer, basis)
    operator = build_rggb_polyphase_operator(mixing)

    _reject_boolean_numeric(full_grid_scene_covariance, "full_grid_scene_covariance")
    covariance = torch.as_tensor(
        full_grid_scene_covariance, device=operator.matrix.device
    ).to(dtype=operator.matrix.dtype)
    if precomputed_aliased_covariance is None:
        covariance = _require_frozen_finite(
            covariance, "full_grid_scene_covariance"
        )
    elif covariance.requires_grad:
        raise ValueError(
            "full_grid_scene_covariance must be frozen when its prevalidated "
            "aliased covariance is supplied"
        )
    if covariance.shape[-4:-2] != operator.fine_grid_shape:
        raise ValueError("scene covariance frequency grid must match the transfer grid")
    if covariance.shape[-2:] != (
        operator.scene_channels,
        operator.scene_channels,
    ):
        raise ValueError("scene covariance latent dimension must match the basis")
    if precomputed_aliased_covariance is None:
        aliased_covariance = build_aliased_scene_covariance(covariance)
        prior_prevalidated = False
    else:
        aliased_covariance = torch.as_tensor(
            precomputed_aliased_covariance,
            device=operator.matrix.device,
            dtype=operator.matrix.dtype,
        )
        expected_shape = (
            *covariance.shape[:-4],
            operator.coarse_grid_shape[0],
            operator.coarse_grid_shape[1],
            4 * operator.scene_channels,
            4 * operator.scene_channels,
        )
        if tuple(aliased_covariance.shape) != expected_shape:
            raise ValueError(
                "precomputed_aliased_covariance has the wrong shape: "
                f"{tuple(aliased_covariance.shape)} != {expected_shape}"
            )
        if aliased_covariance.requires_grad:
            raise ValueError(
                "precomputed_aliased_covariance must be frozen"
            )
        prior_prevalidated = True

    phase_noise = rggb_phase_noise_variances(
        torch.as_tensor(
            transfer.noise_variance_e2,
            device=operator.matrix.device,
            dtype=real_dtype,
        )
    )
    optical_batch_shape = tuple(operator.matrix.shape[:-4])
    if tuple(phase_noise.shape[:-1]) != optical_batch_shape:
        raise ValueError(
            "transfer noise leading shape must exactly match the spectral OTF "
            "leading shape"
        )
    # The raw noise is constant over coarse frequency, but its leading field /
    # batch axes must remain to the *left* of both frequency axes.  Passing
    # [..., 4, 4] directly would right-align a batch axis with W/2 whenever
    # their sizes happen to agree and silently assign one field's noise to a
    # frequency column.
    phase_noise = phase_noise.reshape(optical_batch_shape + (1, 1, 4))
    noise_covariance = torch.diag_embed(phase_noise).to(operator.matrix.dtype)
    alias_transform = build_alias_block_target_transform(color_transform).to(
        operator.matrix.dtype
    )
    alias_weight = (
        None
        if color_weight is None
        else build_alias_block_target_transform(color_weight).to(operator.matrix.dtype)
    )
    frozen_aggregation_weights = (
        None
        if aggregation_weights is None
        else _frozen_real_tensor(
            aggregation_weights,
            "aggregation_weights",
            device=operator.matrix.device,
            dtype=real_dtype,
        )
    )
    if frozen_aggregation_weights is not None:
        coarse_shape = operator.coarse_grid_shape
        full_sample_shape = optical_batch_shape + coarse_shape
        weight_shape = tuple(frozen_aggregation_weights.shape)
        if weight_shape not in {coarse_shape, full_sample_shape}:
            raise ValueError(
                "aggregation_weights must have exactly the coarse frequency "
                "shape or the full leading-batch plus coarse-frequency shape"
            )
    return rggb_polyphase_a_optimal_risk(
        operator.matrix,
        aliased_covariance,
        noise_covariance,
        target_transform=alias_transform,
        target_weight=alias_weight,
        target_channel_reduction="mean",
        aggregation_weights=frozen_aggregation_weights,
        jitter=jitter_value,
        condition_number_limit=condition_limit_value,
        maximum_whitened_signal_power=signal_power_limit_value,
        return_channel_diagnostics=return_channel_diagnostics,
        prior_covariance_prevalidated=prior_prevalidated,
        noise_covariance_prevalidated=True,
        return_full_posterior=return_full_posterior,
    )


def _normalized_weights(
    count: int,
    value: torch.Tensor | Sequence[float] | None,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError(f"{name} cannot be formed for an empty collection")
    if value is None:
        result = torch.ones(count, device=device, dtype=dtype)
    else:
        result = _frozen_real_tensor(
            value,
            name,
            device=device,
            dtype=dtype,
        )
        if result.shape != (count,):
            raise ValueError(f"{name} must have shape [{count}]")
    if bool((result.detach() < 0).any()) or not bool((result.detach().sum() > 0)):
        raise ValueError(f"{name} must be non-negative with a positive sum")
    maximum = result.max()
    return (result / maximum) / (result / maximum).sum()


def _field_object_batch(
    engine: Any, field_point: object, point_radiance_value: float
) -> ObjectPointBatch:
    radiance_value = _frozen_finite_scalar(point_radiance_value, "point_radiance_value")
    if radiance_value <= 0.0:
        raise ValueError("point_radiance_value must be finite and positive")
    z_object_um = _frozen_finite_scalar(
        engine.spec.object_plane.z_um, "object_plane.z_um"
    )
    theta = math.radians(
        _frozen_finite_scalar(getattr(field_point, "theta_deg"), "field theta_deg")
    )
    azimuth = math.radians(
        _frozen_finite_scalar(
            getattr(field_point, "azimuth_deg", 0.0), "field azimuth_deg"
        )
    )
    radius = abs(z_object_um) * math.tan(theta)
    coordinates = torch.tensor(
        [[radius * math.cos(azimuth), radius * math.sin(azimuth)]],
        dtype=torch.float32,
        device=engine.device_,
    )
    z = torch.tensor([z_object_um], dtype=torch.float32, device=engine.device_)
    radiance = torch.full(
        (1, int(engine.wavelengths_um.numel())),
        radiance_value,
        dtype=torch.float32,
        device=engine.device_,
    )
    return ObjectPointBatch(coords_um=coordinates, z_um=z, spectral_radiance=radiance)


def _target_prior_risk(
    full_grid_covariance: torch.Tensor,
    color_transform: torch.Tensor,
    color_weight: torch.Tensor | None,
    frequency_weight: torch.Tensor,
) -> torch.Tensor:
    aliased = build_aliased_scene_covariance(full_grid_covariance)
    transform = build_alias_block_target_transform(color_transform).to(aliased.dtype)
    target = transform @ aliased @ transform.conj().transpose(-2, -1)
    if color_weight is not None:
        weight = build_alias_block_target_transform(color_weight).to(aliased.dtype)
        target = weight @ target
    trace = torch.diagonal(target, dim1=-2, dim2=-1).sum(-1).real
    trace = trace / float(transform.shape[-2])
    return (frequency_weight * trace).sum()


_DENSE_INFO_RESERVED_KEYS = frozenset(
    {
        "objective",
        "primary_loss",
        "frequency_layout",
        "local_model_scope",
        "exposure_policy",
        "auto_exposure_target_e",
        "auto_exposure_scale",
        "scene_psd_override",
        "scene_psd_source",
        "field_weight",
        "frequency_weight",
        "electron_calibration",
        "source_spectrum",
        "scene_spectral_basis",
        "scene_color_covariance",
        "target_color_transform",
        "target_color_weight",
        "target_prior_risk",
        "point_radiance_value",
        "jitter",
        "condition_number_limit",
        "maximum_whitened_signal_power",
        "compute_real_dtype",
        "compute_device",
        "field_averaged_a_optimal_risk",
        "field_averaged_normalized_risk",
    }
)


def _validated_field_names(fields: Sequence[object]) -> list[str]:
    names: list[str] = []
    for index, field_point in enumerate(fields):
        raw = getattr(field_point, "name", None)
        if raw is None:
            name = f"field_{index}"
        elif not isinstance(raw, str):
            raise TypeError("field names must be strings when supplied")
        else:
            name = raw
        if not name or name != name.strip():
            raise ValueError("field names must be non-empty without outer whitespace")
        if name in _DENSE_INFO_RESERVED_KEYS:
            raise ValueError(f"field name collides with objective metadata: {name}")
        if name in names:
            raise ValueError(f"duplicate field name: {name}")
        names.append(name)
    return names


def dense_exact_rggb_a_optimal_loss(
    engine: Any,
    width_map: torch.Tensor,
    field_points: Sequence[object],
    *,
    electron_calibration: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    scene_spectral_basis: torch.Tensor,
    scene_color_covariance: torch.Tensor,
    target_color_transform: torch.Tensor,
    target_color_weight: torch.Tensor | None = None,
    field_weight: torch.Tensor | Sequence[float] | None = None,
    frequency_weight: torch.Tensor | None = None,
    source_spectrum: torch.Tensor | None = None,
    scene_contrast_rms: float = 0.18,
    scene_k0_cyc_per_pixel: float = 0.02,
    scene_beta: float = 2.0,
    point_radiance_value: float = 1.0e10,
    jitter: float = 0.0,
    condition_number_limit: float | None = None,
    maximum_whitened_signal_power: float | None = None,
    return_channel_diagnostics: bool = False,
    auto_exposure_target_e: float | None = None,
    scene_psd_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return field-averaged exact common-exposure Bayer A-risk.

    The same frozen ``electron_calibration`` and source spectrum are reused for
    every field and design.  The returned loss is positive and is minimized.
    ``normalized_risk`` values in the diagnostics divide by a fixed zero-signal
    prior risk only for scale/readability; that normalization never changes the
    primary objective.

    ``auto_exposure_target_e=None`` (default) keeps the legacy frozen
    common-exposure policy byte-for-byte. When set, the exposure is metered per
    evaluation batch (auto-exposure camera convention): the frozen calibration
    scalar is rescaled by a detached factor so the field-weighted RGB-mean
    electron count of THIS design equals the target. Equal-brightness scoring:
    throughput differences are absorbed into exposure time, so the risk
    reflects only transfer shape (sharpness + chromatic allocation) and the
    per-channel noise structure at matched brightness. The scale is detached
    (exposure servo, not a gradient path).
    """

    widths = torch.as_tensor(width_map)
    if not widths.is_floating_point() or not bool(
        torch.isfinite(widths.detach()).all()
    ):
        raise ValueError("width_map must be a finite floating tensor")
    fields = list(field_points)
    device = widths.device
    real_dtype = widths.dtype
    engine_width = getattr(engine, "width_map_um", None)
    if not isinstance(engine_width, torch.Tensor):
        raise TypeError("engine.width_map_um must be a tensor")
    engine_device = _resolved_engine_storage_device(engine.device_, engine_width.device)
    if device != engine_device:
        raise ValueError("width_map must be on the engine physical-width device")
    if real_dtype != engine_width.dtype:
        raise ValueError(
            "width_map dtype must exactly match the engine physical-width dtype"
        )
    if tuple(widths.shape) != tuple(engine_width.shape):
        raise ValueError("width_map shape must exactly match engine.width_map_um")
    field_names = _validated_field_names(fields)
    contrast_rms_value = _frozen_finite_scalar(scene_contrast_rms, "scene_contrast_rms")
    scene_k0_value = _frozen_finite_scalar(
        scene_k0_cyc_per_pixel, "scene_k0_cyc_per_pixel"
    )
    scene_beta_value = _frozen_finite_scalar(scene_beta, "scene_beta")
    point_radiance_value = _frozen_finite_scalar(
        point_radiance_value, "point_radiance_value"
    )
    jitter_value = _frozen_finite_scalar(jitter, "jitter")
    condition_limit_value = (
        None
        if condition_number_limit is None
        else _frozen_finite_scalar(condition_number_limit, "condition_number_limit")
    )
    signal_power_limit_value = (
        None
        if maximum_whitened_signal_power is None
        else _frozen_finite_scalar(
            maximum_whitened_signal_power, "maximum_whitened_signal_power"
        )
    )
    weights = _normalized_weights(
        len(fields),
        field_weight,
        name="field_weight",
        device=device,
        dtype=real_dtype,
    )

    pixel_shape = tuple(int(value) for value in engine.spec.resolved_pixel_grid_shape())
    if len(pixel_shape) != 2 or any(value <= 0 or value % 2 for value in pixel_shape):
        raise ValueError("exact 2x2 RGGB objective requires a positive even pixel grid")
    height, width = pixel_shape
    if scene_psd_override is not None:
        scene_psd = torch.as_tensor(
            scene_psd_override, device=device, dtype=real_dtype
        ).detach()
        if tuple(scene_psd.shape) != pixel_shape:
            raise ValueError(
                "scene_psd_override must match the pixel grid "
                f"{pixel_shape}, got {tuple(scene_psd.shape)}"
            )
        if not bool(torch.isfinite(scene_psd).all()) or bool(
            (scene_psd < 0).any()
        ):
            raise ValueError("scene_psd_override must be finite and non-negative")
    else:
        scene_psd = natural_scene_power_spectrum(
            pixel_shape,
            contrast_rms=contrast_rms_value,
            k0_cyc_per_pixel=scene_k0_value,
            beta=scene_beta_value,
            device=device,
            dtype=real_dtype,
        )
    basis, color_transform, color_weight = _color_protocol(
        scene_spectral_basis,
        target_color_transform,
        target_color_weight,
        wavelength_count=int(engine.wavelengths_um.numel()),
        device=device,
        real_dtype=real_dtype,
    )
    latent_channels = int(basis.shape[0])
    color_covariance = _frozen_real_tensor(
        scene_color_covariance,
        "scene_color_covariance",
        device=device,
        dtype=real_dtype,
    )
    if color_covariance.shape != (latent_channels, latent_channels):
        raise ValueError("scene_color_covariance must have shape [K, K]")
    full_covariance = (scene_psd[..., None, None] * color_covariance).to(
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )

    source = (
        torch.ones_like(engine.wavelengths_um, device=device, dtype=real_dtype)
        if source_spectrum is None
        else _frozen_real_tensor(
            source_spectrum,
            "source_spectrum",
            device=device,
            dtype=real_dtype,
        )
    )
    if source.shape != engine.wavelengths_um.shape or bool((source.detach() < 0).any()):
        raise ValueError("source_spectrum must be non-negative and match wavelengths")
    calibration = _frozen_real_tensor(
        electron_calibration,
        "electron_calibration",
        device=device,
        dtype=real_dtype,
    )
    if calibration.numel() != 1 or not bool((calibration.detach() > 0)):
        raise ValueError("electron_calibration must be one frozen positive scalar")
    read_noise = _frozen_real_tensor(
        read_noise_e,
        "read_noise_e",
        device=device,
        dtype=real_dtype,
    )
    if read_noise.numel() != 1 or bool((read_noise.detach() < 0)):
        raise ValueError("read_noise_e must be one frozen non-negative scalar")
    auto_target = (
        None
        if auto_exposure_target_e is None
        else _frozen_finite_scalar(auto_exposure_target_e, "auto_exposure_target_e")
    )
    if auto_target is not None and auto_target <= 0.0:
        raise ValueError("auto_exposure_target_e must be positive when set")

    coarse_shape = (height // 2, width // 2)
    if frequency_weight is None:
        coarse_weight = torch.ones(coarse_shape, device=device, dtype=real_dtype)
    else:
        coarse_weight = _frozen_real_tensor(
            frequency_weight,
            "frequency_weight",
            device=device,
            dtype=real_dtype,
        )
        if coarse_weight.shape != coarse_shape:
            raise ValueError("frequency_weight must match the coarse Bayer grid")
        if bool((coarse_weight.detach() < 0).any()):
            raise ValueError("frequency_weight must be non-negative")
    if not bool((coarse_weight.detach().sum() > 0)):
        raise ValueError("frequency_weight must have a positive sum")
    scaled_frequency_weight = coarse_weight / coarse_weight.max()
    normalized_frequency_weight = (
        scaled_frequency_weight / scaled_frequency_weight.sum()
    )
    prior_risk = _target_prior_risk(
        full_covariance,
        color_transform,
        color_weight,
        normalized_frequency_weight,
    ).detach()
    if not bool(torch.isfinite(prior_risk)) or not bool(prior_risk > 0):
        raise ValueError(
            "the target scene protocol must have positive finite prior risk"
        )

    def _field_spectral_psf(field_point):
        batch = _field_object_batch(engine, field_point, point_radiance_value)
        spectral_psf = engine.forward_optics_from_object_batch(
            batch, width_map_override=widths
        )
        if spectral_psf.device != device or spectral_psf.dtype != real_dtype:
            raise RuntimeError(
                "engine optical forward dtype/device must exactly match the "
                "physical width optimization tensor"
            )
        return spectral_psf

    cached_psfs: list[torch.Tensor] | None = None
    effective_calibration = calibration
    auto_scale = None
    if auto_target is not None:
        # Pass 1 (exposure metering): field-weighted RGB-mean electrons of
        # THIS design under the frozen base calibration, then one detached
        # rescale so the batch hits the target brightness (auto-exposure).
        cached_psfs = []
        metered = widths.new_zeros(())
        for index, field_point in enumerate(fields):
            spectral_psf = _field_spectral_psf(field_point)
            cached_psfs.append(spectral_psf)
            probe = build_calibrated_fixed_exposure_transfer(
                spectral_psf,
                engine.wavelengths_um,
                source,
                engine.cfa_transmission,
                engine.qe,
                pixel_grid_shape=pixel_shape,
                electron_calibration=calibration,
                read_noise_e=read_noise,
            )
            metered = metered + weights[index] * probe.mean_electrons.mean()
        auto_scale = (
            auto_target / metered.detach().clamp_min(1e-12)
        ).reshape(())
        effective_calibration = calibration * auto_scale

    total = widths.new_zeros(())
    info: dict[str, object] = {
        "objective": "exact_periodic_2x2_RGGB_A_optimal_target_MMSE",
        "primary_loss": "field_weighted_mean_tr_QTPT_H_per_target_channel",
        "frequency_layout": "unshifted_fft_dc_at_0_0",
        "local_model_scope": "periodic_LSI_not_global_finite_boundary",
        "exposure_policy": (
            "one_frozen_calibrated_common_exposure_no_design_rescaling"
            if auto_target is None
            else "per_batch_auto_exposure_rgb_mean_"
                 f"target_{auto_target:g}e_detached_scale"
        ),
        "field_weight": weights.detach(),
        "frequency_weight": normalized_frequency_weight.detach(),
        "electron_calibration": calibration.detach(),
        "source_spectrum": source.detach(),
        "scene_spectral_basis": basis.detach(),
        "scene_color_covariance": color_covariance.detach(),
        "target_color_transform": color_transform.detach(),
        "target_color_weight": None if color_weight is None else color_weight.detach(),
        "target_prior_risk": prior_risk,
        "point_radiance_value": point_radiance_value,
        "jitter": jitter_value,
        "condition_number_limit": condition_limit_value,
        "maximum_whitened_signal_power": signal_power_limit_value,
        "compute_real_dtype": str(real_dtype),
        "compute_device": str(device),
    }
    if auto_target is not None:
        info["auto_exposure_target_e"] = auto_target
        info["auto_exposure_scale"] = auto_scale.detach()
    info["scene_psd_source"] = (
        "parametric_natural_scene_power_spectrum"
        if scene_psd_override is None
        else "empirical_override"
    )
    for index, field_point in enumerate(fields):
        spectral_psf = (
            cached_psfs[index]
            if cached_psfs is not None
            else _field_spectral_psf(field_point)
        )
        transfer = build_calibrated_fixed_exposure_transfer(
            spectral_psf,
            engine.wavelengths_um,
            source,
            engine.cfa_transmission,
            engine.qe,
            pixel_grid_shape=pixel_shape,
            electron_calibration=effective_calibration,
            read_noise_e=read_noise,
        )
        result = rggb_polyphase_a_optimal_from_transfer(
            transfer,
            basis,
            full_covariance,
            color_transform,
            target_color_weight=color_weight,
            aggregation_weights=coarse_weight,
            jitter=jitter_value,
            condition_number_limit=condition_limit_value,
            maximum_whitened_signal_power=signal_power_limit_value,
            return_channel_diagnostics=return_channel_diagnostics,
        )
        field_risk = result.objective
        if field_risk.numel() != 1:
            raise RuntimeError("field A-optimal aggregation did not produce one scalar")
        total = total + weights[index] * field_risk.reshape(())
        name = field_names[index]
        field_info: dict[str, object] = {
            "a_optimal_risk": field_risk.detach().reshape(()),
            "normalized_risk": (field_risk.detach().reshape(()) / prior_risk),
            "mean_electrons": transfer.mean_electrons.detach(),
            "noise_variance_e2": transfer.noise_variance_e2.detach(),
            "noise_condition_number_max": (
                result.solver_diagnostics.noise_condition_number.detach().max()
            ),
            "innovation_condition_number_max": (
                result.solver_diagnostics.innovation_condition_number.detach().max()
            ),
            "whitened_signal_power_upper_bound_max": (
                result.solver_diagnostics.whitened_signal_power_upper_bound.detach().max()
            ),
        }
        if result.channel_diagnostics is not None:
            field_info["information_bits_per_raw_pixel"] = (
                aggregate_real_gaussian_information_bpp(
                    result.channel_diagnostics.real_gaussian_information_bits_per_coarse_frequency,
                    aggregation_weights=coarse_weight,
                ).detach()
            )
            singular = result.channel_diagnostics.whitened_singular_values.detach()
            field_info["whitened_singular_value_min"] = singular[..., -1].min()
            field_info["whitened_singular_value_max"] = singular[..., 0].max()
        info[name] = field_info
    info["field_averaged_a_optimal_risk"] = total.detach()
    info["field_averaged_normalized_risk"] = total.detach() / prior_risk
    return total, info


__all__ = [
    "dense_exact_rggb_a_optimal_loss",
    "rggb_polyphase_a_optimal_from_transfer",
]
