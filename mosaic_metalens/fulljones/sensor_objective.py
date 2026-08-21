"""Throughput-aware dense sensor-information objective for width-map design.

This module is the optimizer-facing bridge to :mod:`engine2.sensor_information`.
It deliberately accepts only a *frozen* relative-response-to-electron scalar:
the current simulator PSF scale is not radiometrically calibrated, so values
from this objective are relative common-exposure model bits, never lux- or
absolute-irradiance claims.

Unlike the legacy capacity objectives, the spectral PSF is block-summed onto
the readout-pixel grid before its OTF is formed and its DC gain is retained.
Consequently throughput, vignetting, cropped energy, wavelength quadrature,
CFA/QE response, shot noise, and read noise all affect the loss.  The current
objective is an RGGB-area-weighted diagonal proxy; exact periodic Bayer alias
accounting belongs to the separate polyphase operator.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .sensor_information import (
    build_calibrated_fixed_exposure_transfer,
    diagonal_information_density,
)
from .types import ObjectPointBatch


RGGB_FRACTIONS = (0.25, 0.50, 0.25)


def natural_scene_power_spectrum(
    shape: tuple[int, int],
    *,
    contrast_rms: float = 0.18,
    k0_cyc_per_pixel: float = 0.02,
    beta: float = 2.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an unshifted dense natural-scene PSD on a pixel FFT grid.

    ``torch.fft.fftfreq`` ordering is intentional because
    :class:`~engine2.sensor_information.SensorInformationTransfer` stores an
    unshifted OTF whose DC is ``[0, 0]``.  The full-grid discrete mean is
    normalized to ``contrast_rms**2``; under the unitary-DFT covariance
    convention this is the zero-mean spatial contrast variance.
    """

    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError("shape must contain two positive dimensions")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    values = (float(contrast_rms), float(k0_cyc_per_pixel), float(beta))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scene PSD parameters must be finite")
    if contrast_rms < 0.0:
        raise ValueError("contrast_rms must be non-negative")
    if k0_cyc_per_pixel <= 0.0 or beta <= 0.0:
        raise ValueError("k0_cyc_per_pixel and beta must be positive")

    height, width = int(shape[0]), int(shape[1])
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    profile = (radius + float(k0_cyc_per_pixel)).pow(-float(beta))
    if contrast_rms == 0.0:
        return torch.zeros_like(profile)
    return profile * (float(contrast_rms) ** 2 / profile.mean())


def _field_weights(
    n_fields: int,
    field_weight: torch.Tensor | Sequence[float] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_fields <= 0:
        raise ValueError("field_points must not be empty")
    if field_weight is None:
        weight = torch.ones(n_fields, device=device, dtype=dtype)
    else:
        weight = torch.as_tensor(field_weight, device=device, dtype=dtype)
        if weight.ndim != 1 or weight.numel() != n_fields:
            raise ValueError(f"field_weight must have shape [{n_fields}]")
    if not bool(torch.isfinite(weight).all().detach()) or bool(
        (weight < 0).any().detach()
    ):
        raise ValueError("field_weight must be finite and non-negative")
    total = weight.sum()
    if not bool((total > 0).detach()):
        raise ValueError("field_weight must have a positive sum")
    return weight / total


def _field_object_batch(engine, field_point, radiance_value: float) -> ObjectPointBatch:
    if not math.isfinite(radiance_value) or radiance_value <= 0.0:
        raise ValueError("point_radiance_value must be finite and positive")
    z_object_um = float(engine.spec.object_plane.z_um)
    theta = math.radians(float(field_point.theta_deg))
    azimuth = math.radians(float(getattr(field_point, "azimuth_deg", 0.0)))
    radius = abs(z_object_um) * math.tan(theta)
    coordinates = torch.tensor(
        [[radius * math.cos(azimuth), radius * math.sin(azimuth)]],
        dtype=torch.float32,
        device=engine.device_,
    )
    z = torch.tensor([z_object_um], dtype=torch.float32, device=engine.device_)
    radiance = torch.full(
        (1, int(engine.wavelengths_um.numel())),
        float(radiance_value),
        dtype=torch.float32,
        device=engine.device_,
    )
    return ObjectPointBatch(
        coords_um=coordinates,
        z_um=z,
        spectral_radiance=radiance,
    )


def dense_sensor_information_loss(
    engine,
    width_map: torch.Tensor,
    field_points: Sequence[object],
    *,
    electron_calibration: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    field_weight: torch.Tensor | Sequence[float] | None = None,
    scene_contrast_rms: float = 0.18,
    scene_k0_cyc_per_pixel: float = 0.02,
    scene_beta: float = 2.0,
    point_radiance_value: float = 1.0e10,
    rggb_fractions: Sequence[float] = RGGB_FRACTIONS,
    information_factor: float = 0.5,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return negative dense diagonal sensor information and diagnostics.

    The calibration scalar must have been frozen from one declared reference
    PSF generated with the same ``point_radiance_value``, wavelength grid,
    CFA/QE convention, optical sampling, and pixel grid.  It is reused without
    per-design or per-field re-exposure.
    """

    device = engine.device_
    dtype = width_map.dtype
    fields = list(field_points)
    weights = _field_weights(
        len(fields), field_weight, device=device, dtype=dtype,
    )
    pixel_shape = tuple(int(x) for x in engine.spec.resolved_pixel_grid_shape())
    if len(pixel_shape) != 2:
        raise ValueError("engine pixel grid must be two-dimensional")
    scene_psd = natural_scene_power_spectrum(
        pixel_shape,
        contrast_rms=scene_contrast_rms,
        k0_cyc_per_pixel=scene_k0_cyc_per_pixel,
        beta=scene_beta,
        device=device,
        dtype=dtype,
    )
    fractions = torch.as_tensor(rggb_fractions, device=device, dtype=dtype)
    n_channels = int(engine.cfa_transmission.shape[0])
    if fractions.shape != (n_channels,) or not bool(
        torch.isfinite(fractions).all().detach()
    ) or bool((fractions < 0).any().detach()):
        raise ValueError(f"rggb_fractions must be finite with shape [{n_channels}]")
    if not bool((fractions.sum() > 0).detach()):
        raise ValueError("rggb_fractions must have a positive sum")
    fractions = fractions / fractions.sum()

    calibration = torch.as_tensor(
        electron_calibration, device=device, dtype=dtype,
    )
    if calibration.numel() != 1 or calibration.requires_grad or not bool(
        torch.isfinite(calibration).all().detach()
    ) or not bool((calibration > 0).detach()):
        raise ValueError("electron_calibration must be one frozen positive scalar")

    total_information = width_map.new_zeros(())
    info: dict[str, object] = {
        "frequency_layout": "unshifted_fft_dc_at_0_0",
        "scene_psd_mean": scene_psd.mean().detach(),
        "field_weight": weights.detach(),
        "electron_calibration": calibration.detach(),
        "point_radiance_value": float(point_radiance_value),
    }
    for index, field_point in enumerate(fields):
        batch = _field_object_batch(engine, field_point, point_radiance_value)
        spectral_psf = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )
        transfer = build_calibrated_fixed_exposure_transfer(
            spectral_psf,
            engine.wavelengths_um,
            torch.ones_like(engine.wavelengths_um),
            engine.cfa_transmission,
            engine.qe,
            pixel_grid_shape=pixel_shape,
            electron_calibration=calibration,
            read_noise_e=read_noise_e,
        )
        density = diagonal_information_density(
            transfer.diagonal_otf_e,
            scene_psd,
            transfer.noise_variance_e2,
            information_factor=information_factor,
        )
        channel_bpp = density.mean(dim=(-2, -1))
        field_bpp = (fractions * channel_bpp).sum()
        total_information = total_information + weights[index] * field_bpp
        name = str(getattr(field_point, "name", f"field_{index}"))
        info[name] = {
            "diagonal_bpp": field_bpp.detach(),
            "channel_bpp": channel_bpp.detach(),
            "mean_electrons": transfer.mean_electrons.detach(),
            "noise_variance_e2": transfer.noise_variance_e2.detach(),
        }

    info["field_averaged_diagonal_bpp"] = total_information.detach()
    return -total_information, info


def dense_diagonal_wiener_risk_loss(
    engine,
    width_map: torch.Tensor,
    field_points: Sequence[object],
    *,
    electron_calibration: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    field_weight: torch.Tensor | Sequence[float] | None = None,
    scene_contrast_rms: float = 0.18,
    scene_k0_cyc_per_pixel: float = 0.02,
    scene_beta: float = 2.0,
    point_radiance_value: float = 1.0e10,
    rggb_fractions: Sequence[float] = RGGB_FRACTIONS,
    relative_source_spectrum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Diagonal channel-Wiener reconstruction risk (the diagonal of exact ℰ).

    Identical electron-domain machinery to :func:`dense_sensor_information_loss`
    (absolute per-channel OTF ``H_c^(e)``, design-dependent noise
    ``sigma_c^2 = mu_c + sigma_r^2``, the natural-scene PSD ``S(nu)``), but the
    per-frequency functional is the MMSE of a per-channel Wiener reconstruction
    instead of the log-capacity::

        snr_c(nu)  = |H_c^(e)(nu)|^2 S(nu) / sigma_c^2
        risk_c(nu) = S(nu) / (1 + snr_c(nu))
        L = sum_field w_field * sum_c f_c * <risk_c(nu)>_nu        (minimize)

    This is exactly the exact-aopt A-optimal risk with the colour posterior
    covariance collapsed to its diagonal: per-frequency concavity, absolute
    gain, design-dependent noise, scene prior and CFA weighting are retained,
    and ONLY colour mixing and alias folding are omitted.  ``f_c`` are the RGGB
    pixel-area fractions (the same channel weighting the diagonal machinery
    uses), making it the faithful diagonal of the mosaic objective.
    """

    device = engine.device_
    dtype = width_map.dtype
    fields = list(field_points)
    weights = _field_weights(
        len(fields), field_weight, device=device, dtype=dtype,
    )
    pixel_shape = tuple(int(x) for x in engine.spec.resolved_pixel_grid_shape())
    if len(pixel_shape) != 2:
        raise ValueError("engine pixel grid must be two-dimensional")
    scene_psd = natural_scene_power_spectrum(
        pixel_shape,
        contrast_rms=scene_contrast_rms,
        k0_cyc_per_pixel=scene_k0_cyc_per_pixel,
        beta=scene_beta,
        device=device,
        dtype=dtype,
    )
    fractions = torch.as_tensor(rggb_fractions, device=device, dtype=dtype)
    n_channels = int(engine.cfa_transmission.shape[0])
    if fractions.shape != (n_channels,) or not bool(
        torch.isfinite(fractions).all().detach()
    ) or bool((fractions < 0).any().detach()):
        raise ValueError(f"rggb_fractions must be finite with shape [{n_channels}]")
    if not bool((fractions.sum() > 0).detach()):
        raise ValueError("rggb_fractions must have a positive sum")
    fractions = fractions / fractions.sum()

    calibration = torch.as_tensor(
        electron_calibration, device=device, dtype=dtype,
    )
    if calibration.numel() != 1 or calibration.requires_grad or not bool(
        torch.isfinite(calibration).all().detach()
    ) or not bool((calibration > 0).detach()):
        raise ValueError("electron_calibration must be one frozen positive scalar")

    total_risk = width_map.new_zeros(())
    info: dict[str, object] = {
        "frequency_layout": "unshifted_fft_dc_at_0_0",
        "scene_psd_mean": scene_psd.mean().detach(),
        "field_weight": weights.detach(),
        "electron_calibration": calibration.detach(),
        "point_radiance_value": float(point_radiance_value),
    }
    for index, field_point in enumerate(fields):
        batch = _field_object_batch(engine, field_point, point_radiance_value)
        spectral_psf = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )
        # Match the exposure/source convention of exact ℰ so the frozen
        # calibration lands the same electron scale: pass the exact-aopt source
        # spectrum when supplied, else a flat spectrum (legacy diagonal index).
        source = (
            relative_source_spectrum
            if relative_source_spectrum is not None
            else torch.ones_like(engine.wavelengths_um)
        )
        transfer = build_calibrated_fixed_exposure_transfer(
            spectral_psf,
            engine.wavelengths_um,
            source,
            engine.cfa_transmission,
            engine.qe,
            pixel_grid_shape=pixel_shape,
            electron_calibration=calibration,
            read_noise_e=read_noise_e,
        )
        # Per-(channel, frequency) Wiener posterior MMSE: S / (1 + snr).
        snr = (
            transfer.diagonal_otf_e.abs().square()
            * scene_psd.unsqueeze(-3)
            / transfer.noise_variance_e2[..., :, None, None]
        )
        density = scene_psd.unsqueeze(-3) / (1.0 + snr)          # [..., C, H, W]
        channel_risk = density.mean(dim=(-2, -1))                # [..., C]
        field_risk = (fractions * channel_risk).sum()
        total_risk = total_risk + weights[index] * field_risk
        name = str(getattr(field_point, "name", f"field_{index}"))
        info[name] = {
            "diagonal_wiener_risk": field_risk.detach(),
            "channel_wiener_risk": channel_risk.detach(),
            "mean_electrons": transfer.mean_electrons.detach(),
            "noise_variance_e2": transfer.noise_variance_e2.detach(),
        }

    info["field_averaged_diagonal_wiener_risk"] = total_risk.detach()
    return total_risk, info


__all__ = [
    "RGGB_FRACTIONS",
    "dense_diagonal_wiener_risk_loss",
    "dense_sensor_information_loss",
    "natural_scene_power_spectrum",
]
