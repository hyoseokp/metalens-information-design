from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SensorReadoutSpec
from .spectral import photon_factor_from_irradiance, wavelengths_um_to_m
from .types import RawNoisyBatch


def build_rggb_masks(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mask_r = ((yy % 2 == 0) & (xx % 2 == 0)).to(dtype)
    mask_g = (((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))).to(dtype)
    mask_b = ((yy % 2 == 1) & (xx % 2 == 1)).to(dtype)
    return torch.stack([mask_r, mask_g, mask_b], dim=0)


def channelwise_cfa_filter(
    spectral_irradiance: torch.Tensor,
    cfa_transmission: torch.Tensor,
) -> torch.Tensor:
    return cfa_transmission * spectral_irradiance.unsqueeze(0)


def pixel_integrate_channel_images(
    channel_spectral: torch.Tensor,
    out_shape: tuple[int, int],
) -> torch.Tensor:
    channels, n_lambda, height, width = channel_spectral.shape
    out_h, out_w = out_shape
    if out_h > height or out_w > width:
        raise ValueError("Requested pixel grid shape cannot exceed the optical grid shape.")

    flat = channel_spectral.reshape(1, channels * n_lambda, height, width)
    if height % out_h == 0 and width % out_w == 0:
        kernel_h = height // out_h
        kernel_w = width // out_w
        pooled = F.avg_pool2d(
            flat,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        )
    else:
        pooled = F.adaptive_avg_pool2d(flat, output_size=(out_h, out_w))
    return pooled.reshape(channels, n_lambda, out_h, out_w)


def expected_photoelectrons(
    channel_spectral_pixels: torch.Tensor,
    wavelengths_um: torch.Tensor,
    qe: torch.Tensor,
    exposure_s: float,
) -> torch.Tensor:
    wavelengths_m = wavelengths_um_to_m(wavelengths_um)
    photon_factor = photon_factor_from_irradiance(wavelengths_um).view(1, -1, 1, 1)
    integrand = channel_spectral_pixels * qe * photon_factor
    if wavelengths_um.numel() == 1:
        # trapz over a length-1 axis is identically zero — a monochromatic
        # spec would silently produce zero electrons. Treat the single sample
        # as line power (irradiance already integrated over the line).
        return exposure_s * integrand.squeeze(1)
    return exposure_s * torch.trapz(integrand, x=wavelengths_m, dim=1)


def apply_bayer_sampling(
    channel_electrons: torch.Tensor,
    bayer_masks: torch.Tensor,
) -> torch.Tensor:
    return (channel_electrons * bayer_masks).sum(dim=0)


def adc_encode(
    electrons: torch.Tensor,
    spec: SensorReadoutSpec,
) -> torch.Tensor:
    max_dn = float((1 << spec.adc_bits) - 1)
    dn = torch.round(electrons * spec.adc_gain_dn_per_e + spec.black_level_dn)
    return dn.clamp(0.0, max_dn)


def apply_sensor_noise(
    mu_raw_e: torch.Tensor,
    spec: SensorReadoutSpec,
) -> RawNoisyBatch:
    dark_e = spec.exposure_s * spec.dark_current_e_per_s
    total_mean = (mu_raw_e + dark_e).clamp_min(0.0)
    shot = torch.poisson(total_mean)

    if spec.prnu_std > 0.0:
        prnu_gain = 1.0 + spec.prnu_std * torch.randn_like(shot)
    else:
        prnu_gain = torch.ones_like(shot)

    if spec.dsnu_e_std > 0.0:
        dsnu_offset = spec.dsnu_e_std * torch.randn_like(shot)
    else:
        dsnu_offset = torch.zeros_like(shot)

    read_noise = spec.read_noise_e * torch.randn_like(shot)
    raw_electrons = prnu_gain * shot + dsnu_offset + read_noise
    raw_electrons = raw_electrons.clamp(0.0, spec.full_well_e)
    raw_dn = adc_encode(raw_electrons, spec)
    return RawNoisyBatch(raw_electrons=raw_electrons, raw_dn=raw_dn)
