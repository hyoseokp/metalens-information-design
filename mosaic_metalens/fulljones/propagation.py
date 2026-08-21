from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .grids import (
    crop_common_to_sensor,
    embed_pupil_field,
    embed_spectrum_into_common,
)
from .types import FrequencyGrid

if TYPE_CHECKING:
    from .types import CommonGridMeta


def angular_spectrum_transfer_function(
    frequency_grid: FrequencyGrid,
    wavelengths_um: torch.Tensor,
    distance_um: float,
    refractive_index: float = 1.0,
) -> torch.Tensor:
    wavelengths = wavelengths_um.to(torch.float32)
    fx = frequency_grid.fxx_cyc_per_um.to(torch.float32)
    fy = frequency_grid.fyy_cyc_per_um.to(torch.float32)

    k = (2.0 * math.pi * refractive_index / wavelengths).view(-1, 1, 1)
    kx = 2.0 * math.pi * fx.unsqueeze(0)
    ky = 2.0 * math.pi * fy.unsqueeze(0)
    kz_sq = (k**2 - kx**2 - ky**2).to(torch.complex64)
    kz = torch.sqrt(kz_sq)
    H = torch.exp(1j * kz * distance_um)
    # Circular bandlimit: zero out evanescent + aliased diagonal components
    # Only propagating waves inside the circular NA boundary are kept
    H[kz_sq.real <= 0] = 0.0
    return H


def propagate_angular_spectrum(
    field: torch.Tensor,
    transfer_function: torch.Tensor,
) -> torch.Tensor:
    spectrum = torch.fft.fft2(field, dim=(-2, -1))
    propagated = torch.fft.ifft2(spectrum * transfer_function.unsqueeze(0), dim=(-2, -1))
    return propagated


def build_kz_vectors(
    frequency_grid: FrequencyGrid,
    wavelengths_um: torch.Tensor,
    refractive_index: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute kx, ky, kz for vectorial operations. Returns (kx, ky, kz_safe)."""
    wavelengths = wavelengths_um.to(torch.float32)
    fx = frequency_grid.fxx_cyc_per_um.to(torch.float32)
    fy = frequency_grid.fyy_cyc_per_um.to(torch.float32)

    k = (2.0 * math.pi * refractive_index / wavelengths).view(-1, 1, 1)
    kx = (2.0 * math.pi * fx).unsqueeze(0).to(torch.complex64)
    ky = (2.0 * math.pi * fy).unsqueeze(0).to(torch.complex64)
    kz_sq = (k**2 - kx.real**2 - ky.real**2).to(torch.complex64)
    kz = torch.sqrt(kz_sq)
    # Zero out evanescent modes (circular bandlimit)
    evanescent = kz_sq.real <= 0
    kz[evanescent] = 0.0
    kz_safe = kz.clone()
    kz_safe[kz.abs() < 1e-12] = 1.0
    return kx, ky, kz_safe


def propagate_vectorial_fused(
    Ex_pupil: torch.Tensor,
    Ey_pupil: torch.Tensor,
    transfer_function: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    kz_safe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate Ex,Ey and compute Ez in a single fused FFT pass.

    1. Batch FFT2 of (Ex, Ey) stacked → 1 FFT call instead of 2
    2. Multiply by transfer function in k-space
    3. Compute Ez_hat = -(kx*Ex_hat + ky*Ey_hat)/kz in k-space (no extra FFT)
    4. Batch IFFT2 of (Ex, Ey, Ez) stacked → 1 IFFT call instead of 3
    """
    H_prop = transfer_function.unsqueeze(0)  # [1, N_λ, H, W]

    # Stack Ex, Ey → [2, N_obj, N_λ, H, W], then batch FFT
    Exy_pup = torch.stack([Ex_pupil, Ey_pupil], dim=0)
    Exy_hat = torch.fft.fft2(Exy_pup, dim=(-2, -1))

    # Propagate in k-space
    Exy_hat = Exy_hat * H_prop

    # Compute Ez_hat from div E = 0 (still in k-space, no extra FFT needed)
    Ex_hat, Ey_hat = Exy_hat[0], Exy_hat[1]
    kx_b = kx.expand_as(Ex_hat)
    ky_b = ky.expand_as(Ex_hat)
    kz_b = kz_safe.expand_as(Ex_hat)
    Ez_hat = -(kx_b * Ex_hat + ky_b * Ey_hat) / kz_b
    Ez_hat[kz_safe.abs().expand_as(Ez_hat) == 1.0] = 0.0  # zero where kz was clamped

    # Stack all 3 and batch IFFT → [3, N_obj, N_λ, H, W]
    Exyz_hat = torch.stack([Ex_hat, Ey_hat, Ez_hat], dim=0)
    Exyz = torch.fft.ifft2(Exyz_hat, dim=(-2, -1))

    return Exyz[0], Exyz[1], Exyz[2]


def propagate_vectorial_with_H_fused(
    Ex_pupil: torch.Tensor,
    Ey_pupil: torch.Tensor,
    transfer_function: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    kz_safe: torch.Tensor,
    k_lambda: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate Ex,Ey and compute Ez, Hx, Hy in a single fused FFT pass.

    Returns (Ex, Ey, Ez, Hx, Hy) at the sensor plane in real space.

    H is computed in k-space via Maxwell's curl equation:
        H = (k × E) / (ω μ₀)
    Here we drop the global 1/(c μ₀) constant (does not affect relative
    photometry) and keep the wavelength-dependent factor 1/k_lambda, so:
        H_x_hat = (ky·Ez_hat − kz·Ey_hat) / k_lambda
        H_y_hat = (kz·Ex_hat − kx·Ez_hat) / k_lambda
    The z-component of the time-averaged Poynting vector
        S_z = (1/2) Re(E_x · H_y* − E_y · H_x*)
    then carries the natural cos(θ_k) = kz/k weighting per Fourier mode,
    which is the missing Lambertian factor in the |E|² accumulator.

    1 FFT (Ex, Ey batched) + 1 IFFT (Ex, Ey, Ez, Hx, Hy batched as 5 channels).

    Parameters
    ----------
    k_lambda : Tensor [N_λ]
        Vacuum wavenumber per wavelength: k_lambda = 2π·n / λ in [rad/μm].
    """
    H_prop = transfer_function.unsqueeze(0)  # [1, N_λ, H, W]

    Exy_pup = torch.stack([Ex_pupil, Ey_pupil], dim=0)
    Exy_hat = torch.fft.fft2(Exy_pup, dim=(-2, -1))
    Exy_hat = Exy_hat * H_prop

    Ex_hat, Ey_hat = Exy_hat[0], Exy_hat[1]
    kx_b = kx.expand_as(Ex_hat)
    ky_b = ky.expand_as(Ex_hat)
    kz_b = kz_safe.expand_as(Ex_hat)
    evanescent_mask = kz_safe.abs().expand_as(Ex_hat) == 1.0

    # Ez from div E = 0
    Ez_hat = -(kx_b * Ex_hat + ky_b * Ey_hat) / kz_b
    Ez_hat[evanescent_mask] = 0.0

    # H from Maxwell curl in k-space, normalized by 1/k_lambda.
    # k_lambda shape: [N_λ] → [1, N_λ, 1, 1] for broadcasting against Ex_hat
    # which has shape [N_obj, N_λ, H, W].
    inv_k = (1.0 / k_lambda.to(Ex_hat.real.dtype)).view(1, -1, 1, 1).to(Ex_hat.dtype)
    Hx_hat = (ky_b * Ez_hat - kz_b * Ey_hat) * inv_k
    Hy_hat = (kz_b * Ex_hat - kx_b * Ez_hat) * inv_k
    # At evanescent positions Ex/Ey/Ez are already zero, but be defensive.
    Hx_hat[evanescent_mask] = 0.0
    Hy_hat[evanescent_mask] = 0.0

    # Batch IFFT of (Ex, Ey, Ez, Hx, Hy)
    stack5_hat = torch.stack([Ex_hat, Ey_hat, Ez_hat, Hx_hat, Hy_hat], dim=0)
    stack5 = torch.fft.ifft2(stack5_hat, dim=(-2, -1))
    return stack5[0], stack5[1], stack5[2], stack5[3], stack5[4]


def _apply_common_grid_output_shift(
    spectrum: torch.Tensor,
    output_shift_xy_um: torch.Tensor | None,
    *,
    sensor_pitch_um: float,
) -> torch.Tensor:
    """Apply one achromatic, batch-resolved sensor-plane translation.

    The shift is applied before the common-grid inverse FFT and native sensor
    crop, so an off-axis chief ray is not clipped and then circularly wrapped
    into a local convolution kernel.  ``output_shift_xy_um[b]`` is the
    physical translation ``(dx,dy)`` of batch item ``b``.  Positive ``dx``
    moves a feature toward increasing sensor x.
    """

    if output_shift_xy_um is None:
        return spectrum
    if spectrum.ndim < 4:
        raise ValueError("shifted common-grid spectrum must include B,L,H,W")
    shifts = torch.as_tensor(
        output_shift_xy_um,
        device=spectrum.device,
        dtype=spectrum.real.dtype,
    )
    batch_count = int(spectrum.shape[-4])
    if tuple(shifts.shape) != (batch_count, 2):
        raise ValueError(
            "output_shift_xy_um must have shape [batch,2] matching the "
            f"propagated spectrum; expected {(batch_count, 2)}"
        )
    if not bool(torch.isfinite(shifts.detach()).all()):
        raise ValueError("output_shift_xy_um must be finite")
    height, width = map(int, spectrum.shape[-2:])
    fy = torch.fft.fftfreq(
        height,
        d=float(sensor_pitch_um),
        device=spectrum.device,
        dtype=spectrum.real.dtype,
    )
    fx = torch.fft.fftfreq(
        width,
        d=float(sensor_pitch_um),
        device=spectrum.device,
        dtype=spectrum.real.dtype,
    )
    phase = torch.exp(
        -2j
        * math.pi
        * (
            shifts[:, 1, None, None] * fy[None, :, None]
            + shifts[:, 0, None, None] * fx[None, None, :]
        )
    )
    phase = phase.reshape(
        *((1,) * (spectrum.ndim - 4)), batch_count, 1, height, width
    )
    return spectrum * phase


def propagate_vectorial_with_H_cross_grid(
    Ex_pupil: torch.Tensor,
    Ey_pupil: torch.Tensor,
    transfer_common: torch.Tensor,
    kx_common: torch.Tensor,
    ky_common: torch.Tensor,
    kz_safe_common: torch.Tensor,
    k_lambda: torch.Tensor,
    *,
    common_meta: "CommonGridMeta",
    evanescent_mask: torch.Tensor | None = None,
    skip_alignment: bool = False,
    output_shift_xy_um: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-grid vectorial AS propagation: pupil grid -> sensor grid.

    Embeds the pupil field into a "common Fourier grid" whose physical extent
    matches the pupil's padded extent exactly (rational pitch alignment), so
    pupil and common spectra share the same frequency step. The AS transfer +
    Ez/Hx/Hy derivations are evaluated on the common grid; the result is
    IFFT'd at common-grid sample density (sensor pitch) and center-cropped to
    the sensor's native shape.

    Parameters
    ----------
    Ex_pupil, Ey_pupil : Tensor [..., N_λ, Hp, Wp]
        Field components on the native pupil grid.
    transfer_common : Tensor [N_λ, Nc_pad_h, Nc_pad_w]
        AS transfer function evaluated on common-grid frequencies (which are
        the same as sensor-pad frequencies under rational alignment).
    kx_common, ky_common, kz_safe_common : Tensor [1, N_λ, Nc_pad_h, Nc_pad_w]
        Wavevectors on the common grid (used for Ez/Hx/Hy derivation).
    k_lambda : Tensor [N_λ]
        Vacuum wavenumber per wavelength (used to normalize H amplitude).
    common_meta : CommonGridMeta
        Pupil-pad and common-pad shapes, sensor native shape, amplitude scale.

    Returns
    -------
    Tuple of (Ex, Ey, Ez, Hx, Hy) at sensor native grid (..., N_λ, Hs, Ws).
    """
    # 1+2. Stack on the pupil-NATIVE grid, pad once, batched FFT (P6: the
    # stack-after-pad ordering copied the full pupil-pad tensor twice)
    Exy_pup = torch.stack([Ex_pupil, Ey_pupil], dim=0)
    Exy_pup_pad = embed_pupil_field(
        Exy_pup, common_meta.np_pad_h, common_meta.np_pad_w)
    del Exy_pup
    Exy_hat_pup = torch.fft.fft2(Exy_pup_pad, dim=(-2, -1))
    del Exy_pup_pad

    # 3. Embed both spectra into the common grid in ONE call (leading batch
    # dim writes both components' corners into a single zeros buffer)
    Exy_hat_common = embed_spectrum_into_common(
        Exy_hat_pup, common_meta.nc_pad_h, common_meta.nc_pad_w,
        common_meta.spectrum_amp_scale,
        pup_pitch_um=common_meta.pupil_pitch_um,
        sen_pitch_um=common_meta.sensor_pitch_um,
        skip_alignment=skip_alignment,
    )
    del Exy_hat_pup

    # 4. AS transfer on common grid
    Exy_hat_common = Exy_hat_common * transfer_common.unsqueeze(0)
    Exy_hat_common = _apply_common_grid_output_shift(
        Exy_hat_common,
        output_shift_xy_um,
        sensor_pitch_um=common_meta.sensor_pitch_um,
    )

    # 5. Compute Ez_hat, Hx_hat, Hy_hat on common freq grid
    Ex_hat = Exy_hat_common[0]
    Ey_hat = Exy_hat_common[1]
    kx_b = kx_common.expand_as(Ex_hat)
    ky_b = ky_common.expand_as(Ex_hat)
    kz_b = kz_safe_common.expand_as(Ex_hat)
    if evanescent_mask is None:
        evanescent_mask = kz_safe_common.abs().expand_as(Ex_hat) == 1.0
    else:
        evanescent_mask = evanescent_mask.expand_as(Ex_hat)

    Ez_hat = -(kx_b * Ex_hat + ky_b * Ey_hat) / kz_b
    Ez_hat[evanescent_mask] = 0.0

    inv_k = (1.0 / k_lambda.to(Ex_hat.real.dtype)).view(1, -1, 1, 1).to(Ex_hat.dtype)
    Hx_hat = (ky_b * Ez_hat - kz_b * Ey_hat) * inv_k
    Hy_hat = (kz_b * Ex_hat - kx_b * Ez_hat) * inv_k
    Hx_hat[evanescent_mask] = 0.0
    Hy_hat[evanescent_mask] = 0.0

    # 6. Stack 5 channels and IFFT on common grid
    stack5_hat = torch.stack([Ex_hat, Ey_hat, Ez_hat, Hx_hat, Hy_hat], dim=0)
    del Exy_hat_common, Ex_hat, Ey_hat, Ez_hat, Hx_hat, Hy_hat
    stack5_common = torch.fft.ifft2(stack5_hat, dim=(-2, -1))
    del stack5_hat

    # 7. Center-crop to sensor native (Hs, Ws)
    Ex_sens = crop_common_to_sensor(stack5_common[0], common_meta.sensor_h, common_meta.sensor_w)
    Ey_sens = crop_common_to_sensor(stack5_common[1], common_meta.sensor_h, common_meta.sensor_w)
    Ez_sens = crop_common_to_sensor(stack5_common[2], common_meta.sensor_h, common_meta.sensor_w)
    Hx_sens = crop_common_to_sensor(stack5_common[3], common_meta.sensor_h, common_meta.sensor_w)
    Hy_sens = crop_common_to_sensor(stack5_common[4], common_meta.sensor_h, common_meta.sensor_w)
    return Ex_sens, Ey_sens, Ez_sens, Hx_sens, Hy_sens


def propagate_vectorial_E_cross_grid(
    Ex_pupil: torch.Tensor,
    Ey_pupil: torch.Tensor,
    transfer_common: torch.Tensor,
    *,
    common_meta: "CommonGridMeta",
    skip_alignment: bool = False,
    output_shift_xy_um: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-grid AS propagation of the TANGENTIAL E components only.

    For the realistic-pixel-stack path the air-side Ez/Hx/Hy returned by
    propagate_vectorial_with_H_cross_grid are DISCARDED (H is re-derived in
    Si from the post-stack E), so deriving and inverse-transforming them is
    pure waste: 3 of the 5 common-grid IFFT channels (the dominant cost on
    awkward common-grid sizes). This variant transforms only (Ex, Ey).

    Returns (Ex_sens, Ey_sens) at sensor native grid.
    """
    Exy_pup = torch.stack([Ex_pupil, Ey_pupil], dim=0)
    Exy_pup_pad = embed_pupil_field(
        Exy_pup, common_meta.np_pad_h, common_meta.np_pad_w)
    del Exy_pup
    Exy_hat_pup = torch.fft.fft2(Exy_pup_pad, dim=(-2, -1))
    del Exy_pup_pad
    Exy_hat_common = embed_spectrum_into_common(
        Exy_hat_pup, common_meta.nc_pad_h, common_meta.nc_pad_w,
        common_meta.spectrum_amp_scale,
        pup_pitch_um=common_meta.pupil_pitch_um,
        sen_pitch_um=common_meta.sensor_pitch_um,
        skip_alignment=skip_alignment,
    )
    del Exy_hat_pup
    Exy_hat_common = Exy_hat_common * transfer_common.unsqueeze(0)
    Exy_hat_common = _apply_common_grid_output_shift(
        Exy_hat_common,
        output_shift_xy_um,
        sensor_pitch_um=common_meta.sensor_pitch_um,
    )
    Exy_common = torch.fft.ifft2(Exy_hat_common, dim=(-2, -1))
    del Exy_hat_common
    Ex_sens = crop_common_to_sensor(Exy_common[0], common_meta.sensor_h, common_meta.sensor_w)
    Ey_sens = crop_common_to_sensor(Exy_common[1], common_meta.sensor_h, common_meta.sensor_w)
    return Ex_sens, Ey_sens


def propagate_angular_spectrum_cross_grid(
    field_pupil: torch.Tensor,
    transfer_common: torch.Tensor,
    *,
    common_meta: "CommonGridMeta",
    skip_alignment: bool = False,
    output_shift_xy_um: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar AS propagation, pupil grid -> sensor grid via common grid embedding."""
    field_pad = embed_pupil_field(field_pupil, common_meta.np_pad_h, common_meta.np_pad_w)
    spec_pup = torch.fft.fft2(field_pad, dim=(-2, -1))
    spec_common = embed_spectrum_into_common(
        spec_pup, common_meta.nc_pad_h, common_meta.nc_pad_w,
        common_meta.spectrum_amp_scale,
        pup_pitch_um=common_meta.pupil_pitch_um,
        sen_pitch_um=common_meta.sensor_pitch_um,
        skip_alignment=skip_alignment,
    )
    spec_common = spec_common * transfer_common.unsqueeze(0)
    spec_common = _apply_common_grid_output_shift(
        spec_common,
        output_shift_xy_um,
        sensor_pitch_um=common_meta.sensor_pitch_um,
    )
    field_common = torch.fft.ifft2(spec_common, dim=(-2, -1))
    return crop_common_to_sensor(field_common, common_meta.sensor_h, common_meta.sensor_w)


def compute_Ez_from_transverse(
    Ex: torch.Tensor,
    Ey: torch.Tensor,
    frequency_grid: FrequencyGrid,
    wavelengths_um: torch.Tensor,
    refractive_index: float = 1.0,
) -> torch.Tensor:
    """Recover Ez from div E = 0: Ez(kx,ky) = -(kx*Ex_hat + ky*Ey_hat) / kz."""
    kx, ky, kz_safe = build_kz_vectors(frequency_grid, wavelengths_um, refractive_index)

    Ex_hat = torch.fft.fft2(Ex, dim=(-2, -1))
    Ey_hat = torch.fft.fft2(Ey, dim=(-2, -1))

    kx_b = kx.expand_as(Ex_hat)
    ky_b = ky.expand_as(Ex_hat)
    kz_b = kz_safe.expand_as(Ex_hat)
    kz_mask = kz_safe.abs().expand_as(Ex_hat) == 1.0

    Ez_hat = -(kx_b * Ex_hat + ky_b * Ey_hat) / kz_b
    Ez_hat[kz_mask] = 0.0

    return torch.fft.ifft2(Ez_hat, dim=(-2, -1))
