"""Production multirate sampling and low-memory target information.

This module implements the sampling/loss contract used by the retargeted
manuscript.  It deliberately does not generate an optical response: callers
must first provide a fine-grid colour-mixing OTF that already includes the
physically consistent full-Jones propagation and the *sliding* 1-um photosite
aperture response.

Contract
--------
* scene/presampling grid: 0.25 um;
* sensor pitch: 1 um, with a 2x2 RGGB cell of period 2 um;
* unitary DFTs (``norm="ortho"``);
* scene-to-cell decimation: 8 per axis;
* aliases per coarse frequency: 8x8 = 64;
* exact unitary-DFT sampling coefficient: 1/8;
* measurement rows: R, G(0,1), G(1,0), B;
* target rows: four 1-um phases, each with exactly three RGB target channels;
* noise covariance: common, explicit batch-only, or fully frequency-resolved;
  implicit suffix broadcasting is forbidden.

The mutual-information path streams the 64 independent K-by-K prior blocks
into small measurement/target covariances.  It never constructs the dense
(64K)-by-(64K) latent covariance or posterior.

``fine_color_mixing_otf`` and ``fine_target_mixing_otf`` are convolution
multipliers in unitary scene coordinates.  For a spatial kernel ``h`` they are
``sqrt(H*W) * fft2(h, norm="ortho")`` (equivalently the default backward FFT),
not the unscaled unitary FFT of ``h``.  A centred PSF cache must be declared and
inverse-shifted to the FFT origin before this transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from .sensor_information import (
    clamp_tiny_negative_roundoff,
    trapezoid_bin_widths_m,
)
from .spectral import photon_factor_from_irradiance


SCENE_GRID_SPACING_UM = 0.25
SENSOR_PIXEL_PITCH_UM = 1.0
RGGB_CELL_PITCH_UM = 2.0
SCENE_TO_RGGB_CELL_DECIMATION = 8
PRODUCTION_ALIAS_COUNT = 64
UNITARY_SAMPLING_COEFFICIENT = 1.0 / 8.0
RAW_SAMPLES_PER_CELL = 4


@dataclass(frozen=True)
class ProductionOrdering:
    """Immutable row, phase, alias, and Fourier convention metadata."""

    phase_offsets_scene_yx: tuple[tuple[int, int], ...]
    phase_labels: tuple[str, ...]
    phase_sensor_channels: tuple[int, ...]
    alias_offsets_yx: tuple[tuple[int, int], ...]
    fft_convention: str
    measurement_row_order: str
    target_row_order: str
    block_order: str


PRODUCTION_ORDERING = ProductionOrdering(
    phase_offsets_scene_yx=((0, 0), (0, 4), (4, 0), (4, 4)),
    phase_labels=("R", "G_at_01", "G_at_10", "B"),
    phase_sensor_channels=(0, 1, 1, 2),
    alias_offsets_yx=tuple((q_y, q_x) for q_y in range(8) for q_x in range(8)),
    fft_convention='unshifted unitary DFT (torch FFT norm="ortho")',
    measurement_row_order="R(0,0), G(0,1), G(1,0), B(1,1)",
    target_row_order="phase-major, then target-channel-minor",
    block_order="alias-major, then latent-channel-minor",
)


@dataclass(frozen=True)
class AliasBlockOperator:
    """Frequency-batched alias blocks without a dense latent matrix.

    ``blocks`` has shape ``[..., H/8, W/8, 64, rows, K]``.  The optional
    ``matrix`` property is intended only for small-grid parity tests.
    """

    blocks: torch.Tensor
    ordering: ProductionOrdering
    fine_grid_shape: tuple[int, int]
    coarse_grid_shape: tuple[int, int]
    rows: int
    scene_channels: int

    @property
    def matrix(self) -> torch.Tensor:
        """Return ``[..., Hc, Wc, rows, 64*K]`` for small-grid tests only."""

        transposed = self.blocks.transpose(-3, -2)
        return transposed.reshape(
            *transposed.shape[:-3], self.rows, PRODUCTION_ALIAS_COUNT * self.scene_channels
        )


@dataclass(frozen=True)
class MeasurementSpaceCovariances:
    sigma_y: torch.Tensor
    sigma_z: torch.Tensor
    sigma_yz: torch.Tensor
    sigma_y_given_z: torch.Tensor


@dataclass(frozen=True)
class MeasurementTargetCovariances:
    """Raw measurement and target covariances without target inversion."""

    sigma_y: torch.Tensor
    sigma_z: torch.Tensor
    sigma_yz: torch.Tensor


@dataclass(frozen=True)
class TargetInformationResult:
    """Local and frequency-averaged target information."""

    local_bits_per_cell: torch.Tensor
    bits_per_raw_pixel: torch.Tensor
    covariances: MeasurementSpaceCovariances


@dataclass(frozen=True)
class ProductionFineGridTransfer:
    """Electron-domain fine-grid transfer feeding the Q64 operator.

    The spatial axes remain on the 0.25-um pre-sampling grid.  In particular,
    this object has *not* been pooled onto the 1-um readout lattice.  The
    photosite aperture is instead a sliding 4-by-4 integration kernel, so all
    optical-to-pixel aliases remain available to the explicit Q64 sampler.
    """

    fine_color_mixing_otf: torch.Tensor
    spectral_electron_kernel: torch.Tensor
    mean_electrons: torch.Tensor
    noise_covariance_e2: torch.Tensor
    fine_grid_shape: tuple[int, int]
    input_psf_origin: str
    input_psf_spatial_dims: tuple[int, int]


@dataclass(frozen=True)
class RealFieldAliasBlocks:
    """Independent real blocks for one self-conjugate coarse frequency."""

    measurement_blocks: torch.Tensor
    target_blocks: torch.Tensor
    prior_blocks: torch.Tensor
    source_orbits: tuple[tuple[int, ...], ...]


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    if dtype in (torch.float32, torch.complex64):
        return torch.float32
    raise ValueError("expected float32, float64, complex64, or complex128")


def _validate_fine_grid(height: int, width: int) -> None:
    decimation = SCENE_TO_RGGB_CELL_DECIMATION
    if height <= 0 or width <= 0 or height % decimation or width % decimation:
        raise ValueError(
            "production RGGB analysis requires positive H and W divisible by "
            f"{decimation}, got {(height, width)}"
        )


def _validate_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must be finite")


def _gather_frequency_aliases(value: torch.Tensor, trailing_dims: int) -> torch.Tensor:
    """Gather the fine frequency grid, preserving ``trailing_dims`` axes."""

    height_axis = value.ndim - trailing_dims - 2
    width_axis = height_axis + 1
    height, width = int(value.shape[height_axis]), int(value.shape[width_axis])
    _validate_fine_grid(height, width)
    coarse_h, coarse_w = height // 8, width // 8
    aliases = []
    for q_y, q_x in PRODUCTION_ORDERING.alias_offsets_yx:
        index = [slice(None)] * value.ndim
        index[height_axis] = slice(q_y * coarse_h, (q_y + 1) * coarse_h)
        index[width_axis] = slice(q_x * coarse_w, (q_x + 1) * coarse_w)
        aliases.append(value[tuple(index)])
    return torch.stack(aliases, dim=height_axis + 2)


def gather_q64_spectrum(full_grid_spectrum: torch.Tensor) -> torch.Tensor:
    """Gather ``[..., H, W, K]`` into ``[..., H/8, W/8, 64, K]``."""

    spectrum = torch.as_tensor(full_grid_spectrum)
    if spectrum.ndim < 3:
        raise ValueError("full_grid_spectrum must have shape [..., H, W, K]")
    if not (spectrum.is_floating_point() or spectrum.is_complex()):
        raise ValueError("full_grid_spectrum must be floating or complex")
    _real_dtype(spectrum.dtype)
    if int(spectrum.shape[-1]) <= 0:
        raise ValueError("K must be positive")
    _validate_finite(spectrum, "full_grid_spectrum")
    return _gather_frequency_aliases(spectrum, trailing_dims=1)


def gather_q64_prior_blocks(full_grid_scene_covariance: torch.Tensor) -> torch.Tensor:
    """Gather ``[..., H, W, K, K]`` into independent Q64 prior blocks."""

    covariance = torch.as_tensor(full_grid_scene_covariance)
    if covariance.ndim < 4:
        raise ValueError(
            "full_grid_scene_covariance must have shape [..., H, W, K, K]"
        )
    if not (covariance.is_floating_point() or covariance.is_complex()):
        raise ValueError("full_grid_scene_covariance must be floating or complex")
    _real_dtype(covariance.dtype)
    if covariance.shape[-1] <= 0 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError("prior blocks must be non-empty square matrices")
    _validate_finite(covariance, "full_grid_scene_covariance")
    return _gather_frequency_aliases(covariance, trailing_dims=2)


def _phase_coefficients(
    height: int,
    width: int,
    *,
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[H/8, W/8, 64, 4]`` unitary sampling coefficients."""

    coarse_h, coarse_w = height // 8, width // 8
    base_y = torch.arange(coarse_h, device=device, dtype=real_dtype)
    base_x = torch.arange(coarse_w, device=device, dtype=real_dtype)
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    aliases = []
    for q_y, q_x in PRODUCTION_ORDERING.alias_offsets_yx:
        omega_y = 2.0 * math.pi * (grid_y + q_y * coarse_h) / height
        omega_x = 2.0 * math.pi * (grid_x + q_x * coarse_w) / width
        phases = [
            UNITARY_SAMPLING_COEFFICIENT
            * torch.exp(1j * (omega_y * p_y + omega_x * p_x))
            for p_y, p_x in PRODUCTION_ORDERING.phase_offsets_scene_yx
        ]
        aliases.append(torch.stack(phases, dim=-1))
    return torch.stack(aliases, dim=-2)


def spatial_kernel_to_convolution_otf(
    spatial_kernel: torch.Tensor,
    *,
    kernel_origin: str,
    spatial_dims: tuple[int, int],
) -> torch.Tensor:
    """Convert a spatial kernel to the multiplier for unitary scene DFTs.

    With ``norm="ortho"`` scene coefficients, the circular-convolution theorem
    contains ``sqrt(H*W)``.  Centralizing that factor prevents a caller from
    shrinking every optical response by ``1/sqrt(H*W)`` when connecting a PSF
    cache to the Q64 builders. Both keyword declarations are mandatory because
    raw PSF caches ``[...,L,H,W]`` and mixing kernels ``[...,H,W,C,K]`` place
    their spatial axes differently. ``kernel_origin="fft"`` means
    the zero displacement is already at index ``[0,0]`` on the spatial axes;
    ``"centered"`` applies ``ifftshift`` first.  The explicit declaration keeps
    a centred diagnostic PSF from silently adding a phase ramp.
    """

    kernel = torch.as_tensor(spatial_kernel)
    if not (kernel.is_floating_point() or kernel.is_complex()):
        raise ValueError("spatial_kernel must be floating or complex")
    ndim = kernel.ndim
    normalized_dims = tuple(dim if dim >= 0 else ndim + dim for dim in spatial_dims)
    if (
        len(normalized_dims) != 2
        or normalized_dims[0] == normalized_dims[1]
        or any(dim < 0 or dim >= ndim for dim in normalized_dims)
    ):
        raise ValueError("spatial_dims must name two distinct kernel axes")
    height = int(kernel.shape[normalized_dims[0]])
    width = int(kernel.shape[normalized_dims[1]])
    if height <= 0 or width <= 0:
        raise ValueError("spatial kernel axes must be non-empty")
    _validate_finite(kernel, "spatial_kernel")
    if kernel_origin == "fft":
        prepared = kernel
    elif kernel_origin == "centered":
        prepared = torch.fft.ifftshift(kernel, dim=normalized_dims)
    else:
        raise ValueError('kernel_origin must be either "fft" or "centered"')
    return math.sqrt(height * width) * torch.fft.fft2(
        prepared, dim=normalized_dims, norm="ortho"
    )


def _normalize_spatial_dims(
    ndim: int,
    spatial_dims: tuple[int, int],
) -> tuple[int, int]:
    normalized = tuple(dim if dim >= 0 else ndim + dim for dim in spatial_dims)
    if (
        len(normalized) != 2
        or normalized[0] == normalized[1]
        or any(dim < 0 or dim >= ndim for dim in normalized)
    ):
        raise ValueError("spatial_dims must name two distinct tensor axes")
    return int(normalized[0]), int(normalized[1])


def _sliding_window_sum_fft_origin(
    value: torch.Tensor,
    *,
    spatial_dims: tuple[int, int],
    samples_per_pixel: int,
) -> torch.Tensor:
    """Return start-aligned circular sliding-window sums.

    Output sample ``(n_y,n_x)`` integrates input samples
    ``(n_y+d_y,n_x+d_x)`` for ``0 <= d_y,d_x < samples_per_pixel``.  The
    output index is therefore the photosite-window start, which is exactly the
    phase convention used by :data:`PRODUCTION_ORDERING`.  This explicit
    convention avoids the half-sample ambiguity of an even 4-by-4 aperture.
    """

    if samples_per_pixel <= 0:
        raise ValueError("samples_per_pixel must be positive")
    y_dim, x_dim = _normalize_spatial_dims(value.ndim, spatial_dims)
    integrated = torch.zeros_like(value)
    for d_y in range(samples_per_pixel):
        shifted_y = torch.roll(value, shifts=-d_y, dims=y_dim)
        for d_x in range(samples_per_pixel):
            integrated = integrated + torch.roll(
                shifted_y, shifts=-d_x, dims=x_dim
            )
    return integrated


def _prepare_psf_at_fft_origin(
    spectral_psf: torch.Tensor,
    *,
    psf_origin: str,
    spatial_dims: tuple[int, int],
) -> tuple[torch.Tensor, tuple[int, int]]:
    normalized_dims = _normalize_spatial_dims(spectral_psf.ndim, spatial_dims)
    if psf_origin == "fft":
        prepared = spectral_psf
    elif psf_origin in {"centered", "centered_discrete_sample"}:
        prepared = torch.fft.ifftshift(spectral_psf, dim=normalized_dims)
    else:
        raise ValueError(
            'psf_origin must be "fft", "centered", or '
            '"centered_discrete_sample"'
        )
    return prepared, normalized_dims


def _channel_response_matrix(
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    wavelengths: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a fail-closed ``[3,L]`` CFA-times-QE response."""

    def collapse(value: torch.Tensor, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, device=device, dtype=dtype)
        while result.ndim > 2 and result.shape[-1] == 1:
            result = result.squeeze(-1)
        if result.ndim > 2:
            raise ValueError(
                f"{name} may only have trailing singleton spatial axes"
            )
        return result

    cfa = collapse(cfa_transmission, "cfa_transmission")
    if tuple(cfa.shape) != (3, wavelengths):
        raise ValueError(
            f"cfa_transmission must have shape [3,{wavelengths}]"
        )
    quantum_efficiency = collapse(qe, "qe")
    if quantum_efficiency.ndim == 1:
        if int(quantum_efficiency.numel()) != wavelengths:
            raise ValueError(f"qe must have {wavelengths} wavelength samples")
        quantum_efficiency = quantum_efficiency.unsqueeze(0).expand(3, -1)
    elif tuple(quantum_efficiency.shape) != (3, wavelengths):
        raise ValueError(f"qe must have shape [{wavelengths}] or [3,{wavelengths}]")
    _validate_finite(cfa, "cfa_transmission")
    _validate_finite(quantum_efficiency, "qe")
    if bool((cfa.detach() < 0).any()) or bool(
        (quantum_efficiency.detach() < 0).any()
    ):
        raise ValueError("cfa_transmission and qe must be non-negative")
    return cfa * quantum_efficiency


def _canonical_read_noise(
    read_noise_e: float | torch.Tensor,
    *,
    mean_electrons: torch.Tensor,
) -> torch.Tensor:
    """Accept only scalar, channel-only, or exact-batch read noise."""

    read = torch.as_tensor(
        read_noise_e,
        device=mean_electrons.device,
        dtype=mean_electrons.dtype,
    )
    channels = int(mean_electrons.shape[-1])
    if read.ndim == 0:
        read = read.expand_as(mean_electrons)
    elif tuple(read.shape) == (channels,):
        read = read.reshape(
            *((1,) * (mean_electrons.ndim - 1)), channels
        ).expand_as(mean_electrons)
    elif tuple(read.shape) != tuple(mean_electrons.shape):
        raise ValueError(
            "read_noise_e must be scalar, [3], or exactly match the "
            f"mean-electron batch shape {tuple(mean_electrons.shape)}"
        )
    _validate_finite(read, "read_noise_e")
    if bool((read.detach() < 0).any()):
        raise ValueError("read_noise_e must be non-negative")
    return read


def build_calibrated_fine_grid_transfer(
    spectral_psf_relative: torch.Tensor,
    wavelengths_um: torch.Tensor,
    relative_source_spectrum: torch.Tensor,
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    scene_spectral_basis: torch.Tensor,
    *,
    psf_origin: str,
    psf_spatial_dims: tuple[int, int],
    fine_sample_pitch_um: float,
    photosite_pitch_um: float,
    electron_calibration: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    single_wavelength_bin_width_um: float | torch.Tensor | None = None,
) -> ProductionFineGridTransfer:
    """Bridge a fine full-physics PSF to electron-domain Q64 inputs.

    ``spectral_psf_relative`` must have axes ``[...,L,H,W]`` and callers must
    explicitly identify its two spatial axes and whether the zero-displacement
    sample is centred or already at the FFT origin.  The production contract
    requires 0.25-um samples and a 1-um photosite, hence a sliding 4-by-4 sum.

    The sliding sum is multiplied by ``(0.25/1.0)^2 = 1/16``.  This is the
    fine-scene sample-area factor: it preserves constant-radiance DC and the
    frozen 1-um electron calibration instead of spuriously increasing charge
    by sixteen when the latent scene lattice is refined.
    """

    psf = torch.as_tensor(spectral_psf_relative)
    if not psf.is_floating_point() or psf.ndim < 3:
        raise ValueError(
            "spectral_psf_relative must be floating with shape [...,L,H,W]"
        )
    psf = clamp_tiny_negative_roundoff(psf, "spectral_psf_relative")
    spatial_dims = _normalize_spatial_dims(psf.ndim, psf_spatial_dims)
    if spatial_dims != (psf.ndim - 2, psf.ndim - 1):
        raise ValueError(
            "production spectral_psf_relative axes must be explicitly declared "
            "as its final two axes; expected psf_spatial_dims=(-2,-1)"
        )
    wavelength_axis = psf.ndim - 3
    wavelengths = torch.as_tensor(
        wavelengths_um, device=psf.device, dtype=psf.dtype
    )
    if wavelengths.ndim != 1 or int(psf.shape[wavelength_axis]) != int(
        wavelengths.numel()
    ):
        raise ValueError(
            "spectral_psf_relative must have axes [...,L,H,W] matching "
            "wavelengths_um"
        )
    source = torch.as_tensor(
        relative_source_spectrum, device=psf.device, dtype=psf.dtype
    )
    if tuple(source.shape) != tuple(wavelengths.shape):
        raise ValueError("relative_source_spectrum must have shape [L]")
    _validate_finite(source, "relative_source_spectrum")
    if bool((source.detach() < 0).any()):
        raise ValueError("relative_source_spectrum must be non-negative")

    fine_pitch = float(fine_sample_pitch_um)
    pixel_pitch = float(photosite_pitch_um)
    if not math.isclose(
        fine_pitch, SCENE_GRID_SPACING_UM, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"production fine_sample_pitch_um must be {SCENE_GRID_SPACING_UM}"
        )
    if not math.isclose(
        pixel_pitch, SENSOR_PIXEL_PITCH_UM, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"production photosite_pitch_um must be {SENSOR_PIXEL_PITCH_UM}"
        )
    samples_per_pixel_float = pixel_pitch / fine_pitch
    samples_per_pixel = int(round(samples_per_pixel_float))
    if not math.isclose(
        samples_per_pixel_float,
        samples_per_pixel,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("photosite pitch must be an integer fine-grid multiple")
    if samples_per_pixel != 4:
        raise ValueError("production photosite must span exactly 4-by-4 samples")

    height, width = map(int, psf.shape[-2:])
    _validate_fine_grid(height, width)
    psf_fft_origin, normalized_dims = _prepare_psf_at_fft_origin(
        psf,
        psf_origin=psf_origin,
        spatial_dims=psf_spatial_dims,
    )
    integrated = _sliding_window_sum_fft_origin(
        psf_fft_origin,
        spatial_dims=normalized_dims,
        samples_per_pixel=samples_per_pixel,
    )
    scene_sample_area = (fine_pitch / pixel_pitch) ** 2
    integrated = integrated * scene_sample_area

    response = _channel_response_matrix(
        cfa_transmission,
        qe,
        wavelengths=int(wavelengths.numel()),
        device=psf.device,
        dtype=psf.dtype,
    )
    calibration = torch.as_tensor(
        electron_calibration, device=psf.device, dtype=psf.dtype
    )
    if (
        calibration.numel() != 1
        or calibration.requires_grad
        or not bool(torch.isfinite(calibration.detach()))
        or not bool((calibration.detach() > 0))
    ):
        raise ValueError("electron_calibration must be one frozen positive scalar")
    bin_width_m = trapezoid_bin_widths_m(
        wavelengths,
        single_wavelength_bin_width_um=single_wavelength_bin_width_um,
    )
    electron_weight = (
        response
        * source.unsqueeze(0)
        * bin_width_m.unsqueeze(0)
        * photon_factor_from_irradiance(wavelengths).unsqueeze(0)
        * calibration.reshape(())
    )
    prefix_ndim = psf.ndim - 3
    spectral_electron_kernel = integrated.unsqueeze(-4) * electron_weight.reshape(
        (1,) * prefix_ndim + (3, int(wavelengths.numel()), 1, 1)
    )
    mean_electrons = spectral_electron_kernel.sum(dim=(-3, -2, -1))
    _validate_finite(mean_electrons, "mean_electrons")
    if bool((mean_electrons.detach() <= 0).any()):
        raise ValueError("every sensor channel must have positive DC throughput")

    basis = torch.as_tensor(
        scene_spectral_basis,
        device=psf.device,
        dtype=psf.dtype,
    )
    if basis.ndim != 2 or int(basis.shape[1]) != int(wavelengths.numel()):
        raise ValueError("scene_spectral_basis must have shape [K,L]")
    _validate_finite(basis, "scene_spectral_basis")
    color_kernel = torch.einsum(
        "...clhw,kl->...hwck", spectral_electron_kernel, basis
    )
    color_otf = spatial_kernel_to_convolution_otf(
        color_kernel,
        kernel_origin="fft",
        spatial_dims=(-4, -3),
    )

    read = _canonical_read_noise(
        read_noise_e, mean_electrons=mean_electrons
    )
    channel_variance = mean_electrons + read.square()
    row_indices = torch.tensor(
        PRODUCTION_ORDERING.phase_sensor_channels,
        device=psf.device,
        dtype=torch.long,
    )
    raw_variance = channel_variance.index_select(-1, row_indices)
    noise_covariance = torch.diag_embed(raw_variance)
    return ProductionFineGridTransfer(
        fine_color_mixing_otf=color_otf,
        spectral_electron_kernel=spectral_electron_kernel,
        mean_electrons=mean_electrons,
        noise_covariance_e2=noise_covariance,
        fine_grid_shape=(height, width),
        input_psf_origin=psf_origin,
        input_psf_spatial_dims=tuple(psf_spatial_dims),
    )


def build_normalized_sliding_target_otf(
    fine_grid_shape: tuple[int, int],
    target_color_transform: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a declared 1-um linear target on the 0.25-um grid.

    The target aperture is a normalized start-aligned 4-by-4 sliding sum, so
    its DC gain is one.  The result has axes ``[H,W,C,K]`` and can be passed
    directly to :func:`build_q64_full_rgb_target_blocks`.
    """

    height, width = map(int, fine_grid_shape)
    _validate_fine_grid(height, width)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("target OTF dtype must be float32 or float64")
    transform = torch.as_tensor(
        target_color_transform, device=device, dtype=dtype
    )
    if (
        transform.ndim != 2
        or int(transform.shape[0]) <= 0
        or int(transform.shape[1]) <= 0
    ):
        raise ValueError("target_color_transform must have shape [C,K] with C,K > 0")
    _validate_finite(transform, "target_color_transform")

    impulse = torch.zeros((height, width), device=device, dtype=dtype)
    impulse[0, 0] = 1.0
    aperture = _sliding_window_sum_fft_origin(
        impulse,
        spatial_dims=(-2, -1),
        samples_per_pixel=4,
    ) / 16.0
    kernel = aperture[..., None, None] * transform[None, None, :, :]
    return spatial_kernel_to_convolution_otf(
        kernel,
        kernel_origin="fft",
        spatial_dims=(0, 1),
    )


def self_conjugate_coarse_indices(
    coarse_grid_shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return the coarse DFT cells satisfying ``u == -u`` modulo the grid."""

    height, width = map(int, coarse_grid_shape)
    if height <= 0 or width <= 0:
        raise ValueError("coarse grid dimensions must be positive")
    y_indices = (0, height // 2) if height % 2 == 0 else (0,)
    x_indices = (0, width // 2) if width % 2 == 0 else (0,)
    return tuple((y, x) for y in y_indices for x in x_indices)


def _conjugacy_violation(
    value: torch.Tensor,
    partner_indices: torch.Tensor,
) -> torch.Tensor:
    """Return one device-side Boolean for all 64 conjugate relations.

    The tolerance combines an alias-relative scale with a global FFT scale.
    The latter is required because an FFT of a real kernel can leave
    round-off-sized absolute residue in an alias whose true coefficient is
    nearly zero.  The caller can combine several such Booleans before the
    single host synchronization used by the fail-closed validation gate.
    """

    detached = value.detach()
    partner = detached.index_select(-3, partner_indices).conj()
    residual = (detached - partner).abs()
    magnitude = torch.maximum(detached.abs(), partner.abs())
    alias_axis = residual.ndim - 3
    reduction_dims = tuple(index for index in range(residual.ndim) if index != alias_axis)
    residual_by_alias = residual.amax(dim=reduction_dims)
    scale_by_alias = magnitude.amax(dim=reduction_dims)
    finfo = torch.finfo(_real_dtype(value.dtype))
    global_scale = scale_by_alias.amax().clamp_min(finfo.tiny)
    tolerance_scale = torch.maximum(scale_by_alias, global_scale)
    return (residual_by_alias > (512.0 * finfo.eps * tolerance_scale)).any()


def build_real_field_self_conjugate_blocks(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
    *,
    coarse_index_yx: tuple[int, int],
    fine_grid_shape: tuple[int, int],
) -> RealFieldAliasBlocks:
    """Convert one self-conjugate Q64 cell to independent real coordinates.

    Inputs have shapes ``[..., 64, M, K]``, ``[..., 64, R, K]`` and
    ``[..., 64, K, K]``.  A conjugate pair ``(X_f, X_-f)`` is represented by
    its independent real and imaginary parts, each with covariance ``C/2``.
    A fine-grid DC/Nyquist singleton is represented by one real block with
    covariance ``C``.  The resulting real representation still contains 64
    K-dimensional blocks, so the streamed production contraction is unchanged.
    """

    a = torch.as_tensor(measurement_blocks)
    if not a.is_complex():
        raise ValueError("measurement_blocks must be complex")
    t = torch.as_tensor(target_blocks, device=a.device, dtype=a.dtype)
    prior = torch.as_tensor(prior_blocks, device=a.device, dtype=a.dtype)
    aliases, _, _ = _validate_block_shapes(a, t, prior)
    fine_h, fine_w = map(int, fine_grid_shape)
    _validate_fine_grid(fine_h, fine_w)
    coarse_h, coarse_w = fine_h // 8, fine_w // 8
    u_y, u_x = map(int, coarse_index_yx)
    if (u_y, u_x) not in self_conjugate_coarse_indices((coarse_h, coarse_w)):
        raise ValueError(
            f"coarse cell {(u_y, u_x)} is not self-conjugate on {(coarse_h, coarse_w)}"
        )

    alias_lookup = {
        offset: index for index, offset in enumerate(PRODUCTION_ORDERING.alias_offsets_yx)
    }
    conjugate_index: list[int] = []
    for q_y, q_x in PRODUCTION_ORDERING.alias_offsets_yx:
        fine_y = (u_y + q_y * coarse_h) % fine_h
        fine_x = (u_x + q_x * coarse_w) % fine_w
        conjugate_y = (-fine_y) % fine_h
        conjugate_x = (-fine_x) % fine_w
        delta_y = (conjugate_y - u_y) % fine_h
        delta_x = (conjugate_x - u_x) % fine_w
        if delta_y % coarse_h or delta_x % coarse_w:
            raise RuntimeError("conjugate frequency left its self-conjugate coarse cell")
        conjugate_index.append(
            alias_lookup[(delta_y // coarse_h, delta_x // coarse_w)]
        )

    # Check every pair, including the self-conjugate real singletons, in one
    # device reduction.  The old per-orbit `.item()` checks caused hundreds of
    # serial GPU synchronizations per objective evaluation.
    partner_indices = torch.tensor(
        conjugate_index, device=a.device, dtype=torch.long
    )
    violations = torch.stack(
        [
            _conjugacy_violation(a, partner_indices),
            _conjugacy_violation(t, partner_indices),
            _conjugacy_violation(prior, partner_indices),
        ]
    )
    if bool(violations.any().item()):
        raise ValueError(
            "measurement, target, or prior blocks violate real-field "
            "conjugate symmetry"
        )

    output_a: list[torch.Tensor] = []
    output_t: list[torch.Tensor] = []
    output_prior: list[torch.Tensor] = []
    source_orbits: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for q in range(aliases):
        if q in visited:
            continue
        partner = conjugate_index[q]
        visited.add(q)
        visited.add(partner)
        aq = a[..., q, :, :]
        tq = t[..., q, :, :]
        cq = prior[..., q, :, :]
        if partner == q:
            output_a.append(aq.real)
            output_t.append(tq.real)
            output_prior.append(cq.real)
            source_orbits.append((q,))
            continue

        ap = a[..., partner, :, :]
        tp = t[..., partner, :, :]
        cp = prior[..., partner, :, :]
        real_covariance = (0.5 * (cq + cp.conj())).real
        output_a.extend(
            [
                (aq + ap).real,
                (1j * (aq - ap)).real,
            ]
        )
        output_t.extend(
            [
                (tq + tp).real,
                (1j * (tq - tp)).real,
            ]
        )
        output_prior.extend([0.5 * real_covariance, 0.5 * real_covariance])
        source_orbits.extend([(q, partner), (q, partner)])

    if len(output_a) != PRODUCTION_ALIAS_COUNT:
        raise RuntimeError(
            "real-field Q64 conversion must preserve 64 K-dimensional blocks, "
            f"got {len(output_a)}"
        )
    return RealFieldAliasBlocks(
        measurement_blocks=torch.stack(output_a, dim=-3),
        target_blocks=torch.stack(output_t, dim=-3),
        prior_blocks=torch.stack(output_prior, dim=-3),
        source_orbits=tuple(source_orbits),
    )


def build_q64_rggb_measurement_blocks(
    fine_color_mixing_otf: torch.Tensor,
) -> AliasBlockOperator:
    """Build the exact four-row Q64 RGGB measurement blocks.

    Input has shape ``[..., H, W, 3, K]`` and must already include the optical
    response and sliding photosite aperture.  Output blocks have shape
    ``[..., H/8, W/8, 64, 4, K]``.
    """

    mixing = torch.as_tensor(fine_color_mixing_otf)
    if mixing.ndim < 4:
        raise ValueError("fine_color_mixing_otf must have shape [..., H, W, 3, K]")
    if not (mixing.is_floating_point() or mixing.is_complex()):
        raise ValueError("fine_color_mixing_otf must be floating or complex")
    real_dtype = _real_dtype(mixing.dtype)
    height, width, sensor_channels, scene_channels = map(int, mixing.shape[-4:])
    _validate_fine_grid(height, width)
    if sensor_channels != 3:
        raise ValueError("RGGB measurement requires three sensor response rows")
    if scene_channels <= 0:
        raise ValueError("K must be positive")
    _validate_finite(mixing, "fine_color_mixing_otf")

    aliases = _gather_frequency_aliases(mixing, trailing_dims=2)
    selected = torch.stack(
        [aliases[..., :, channel, :] for channel in PRODUCTION_ORDERING.phase_sensor_channels],
        dim=-2,
    )
    coefficients = _phase_coefficients(
        height, width, device=mixing.device, real_dtype=real_dtype
    )
    batch_ndim = mixing.ndim - 4
    coefficients = coefficients.reshape(
        (1,) * batch_ndim + (*coefficients.shape, 1)
    )
    blocks = selected * coefficients
    return AliasBlockOperator(
        blocks=blocks,
        ordering=PRODUCTION_ORDERING,
        fine_grid_shape=(height, width),
        coarse_grid_shape=(height // 8, width // 8),
        rows=4,
        scene_channels=scene_channels,
    )


def build_q64_full_rgb_target_blocks(
    fine_target_mixing_otf: torch.Tensor,
) -> AliasBlockOperator:
    """Build the 1-um four-phase full-colour target sampler.

    ``fine_target_mixing_otf`` has shape ``[..., H, W, C, K]``.  It includes
    the declared target-pixel aperture and colour transform.  Output blocks
    have shape ``[..., H/8, W/8, 64, 4*C, K]`` in phase-major row order.
    For linear RGB, ``C=3`` and there are 12 target rows.
    """

    target = torch.as_tensor(fine_target_mixing_otf)
    if target.ndim < 4:
        raise ValueError("fine_target_mixing_otf must have shape [..., H, W, C, K]")
    if not (target.is_floating_point() or target.is_complex()):
        raise ValueError("fine_target_mixing_otf must be floating or complex")
    real_dtype = _real_dtype(target.dtype)
    height, width, target_channels, scene_channels = map(int, target.shape[-4:])
    _validate_fine_grid(height, width)
    if target_channels <= 0:
        raise ValueError("production target requires a positive target-channel count")
    if scene_channels <= 0:
        raise ValueError("scene channel dimension must be positive")
    _validate_finite(target, "fine_target_mixing_otf")

    aliases = _gather_frequency_aliases(target, trailing_dims=2)
    expanded = aliases.unsqueeze(-3).expand(
        *aliases.shape[:-2], 4, target_channels, scene_channels
    )
    coefficients = _phase_coefficients(
        height, width, device=target.device, real_dtype=real_dtype
    )
    batch_ndim = target.ndim - 4
    coefficients = coefficients.reshape(
        (1,) * batch_ndim + (*coefficients.shape, 1, 1)
    )
    blocks = (expanded * coefficients).reshape(
        *aliases.shape[:-2], 4 * target_channels, scene_channels
    )
    return AliasBlockOperator(
        blocks=blocks,
        ordering=PRODUCTION_ORDERING,
        fine_grid_shape=(height, width),
        coarse_grid_shape=(height // 8, width // 8),
        rows=4 * target_channels,
        scene_channels=scene_channels,
    )


def _adjoint(value: torch.Tensor) -> torch.Tensor:
    return value.conj().transpose(-2, -1)


def _hermitian(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + _adjoint(value))


def _validate_block_shapes(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
) -> tuple[int, int, int]:
    if measurement_blocks.ndim < 3 or target_blocks.ndim < 3 or prior_blocks.ndim < 3:
        raise ValueError("alias blocks require shapes [..., 64, rows, K] and [..., 64, K, K]")
    aliases = int(measurement_blocks.shape[-3])
    if aliases != PRODUCTION_ALIAS_COUNT:
        raise ValueError(f"measurement blocks must contain {PRODUCTION_ALIAS_COUNT} aliases")
    if int(target_blocks.shape[-3]) != aliases or int(prior_blocks.shape[-3]) != aliases:
        raise ValueError("measurement, target, and prior alias counts must match")
    k = int(measurement_blocks.shape[-1])
    if k <= 0 or int(target_blocks.shape[-1]) != k:
        raise ValueError("measurement and target blocks must share a positive K")
    if tuple(prior_blocks.shape[-2:]) != (k, k):
        raise ValueError("each prior block must be K by K")
    return aliases, int(measurement_blocks.shape[-2]), int(target_blocks.shape[-2])


def _broadcast_block_batch_shape(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
) -> tuple[int, ...]:
    """Return the explicit batch shape produced by the three block tensors."""

    try:
        return tuple(
            torch.broadcast_shapes(
                tuple(measurement_blocks.shape[:-3]),
                tuple(target_blocks.shape[:-3]),
                tuple(prior_blocks.shape[:-3]),
            )
        )
    except RuntimeError as error:
        raise ValueError(
            "measurement, target, and prior block batch shapes are not broadcastable"
        ) from error


def _canonical_noise_for_block_batch(
    noise_covariance: torch.Tensor,
    *,
    batch_shape: tuple[int, ...],
    measurement_rows: int,
) -> torch.Tensor:
    """Expand common or fully resolved noise without implicit axis guessing.

    Generic alias-block contractions accept only ``[M,M]`` or the exact
    ``batch_shape+[M,M]`` form.  In particular, a leading field axis is never
    allowed to right-align silently with a coarse-frequency axis.
    """

    noise = noise_covariance
    if tuple(noise.shape[-2:]) != (measurement_rows, measurement_rows):
        raise ValueError("noise_covariance must match the measurement row count")
    leading_shape = tuple(map(int, noise.shape[:-2]))
    if not leading_shape:
        reshaped = noise.reshape(
            *((1,) * len(batch_shape)), measurement_rows, measurement_rows
        )
        return reshaped.expand(*batch_shape, measurement_rows, measurement_rows)
    if leading_shape != batch_shape:
        raise ValueError(
            "noise_covariance must be common [M,M] or have the exact block "
            f"batch shape {batch_shape}+[M,M]; got {leading_shape}+[M,M]"
        )
    return noise


def _canonical_q64_noise(
    noise_covariance: torch.Tensor,
    *,
    batch_prefix: tuple[int, ...],
    coarse_shape: tuple[int, int],
    measurement_rows: int,
) -> torch.Tensor:
    """Canonicalize the three production Q64 noise forms.

    Accepted forms are common ``[M,M]``, batch-only ``B+[M,M]`` and fully
    frequency-resolved ``B+[Hc,Wc,M,M]``.  Singleton placeholders and suffix
    broadcasting are rejected so coincident dimension sizes cannot change the
    physical meaning of an input.
    """

    noise = noise_covariance
    if tuple(noise.shape[-2:]) != (measurement_rows, measurement_rows):
        raise ValueError("noise_covariance must match the measurement row count")
    leading_shape = tuple(map(int, noise.shape[:-2]))
    full_shape = batch_prefix + coarse_shape
    if not leading_shape:
        reshaped = noise.reshape(
            *((1,) * len(full_shape)), measurement_rows, measurement_rows
        )
        return reshaped.expand(*full_shape, measurement_rows, measurement_rows)
    if leading_shape == batch_prefix:
        reshaped = noise.reshape(
            *batch_prefix, 1, 1, measurement_rows, measurement_rows
        )
        return reshaped.expand(*full_shape, measurement_rows, measurement_rows)
    if leading_shape == full_shape:
        return noise
    raise ValueError(
        "Q64 noise_covariance must be [M,M], B+[M,M], or "
        f"B+{coarse_shape}+[M,M] for B={batch_prefix}; got "
        f"{leading_shape}+[M,M]"
    )


def accumulate_measurement_target_covariances(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> MeasurementTargetCovariances:
    """Stream Q64 prior blocks without inverting the target covariance."""

    a = torch.as_tensor(measurement_blocks)
    if not (a.is_floating_point() or a.is_complex()):
        raise ValueError("measurement_blocks must be floating or complex")
    _real_dtype(a.dtype)
    t = torch.as_tensor(target_blocks, device=a.device, dtype=a.dtype)
    prior = torch.as_tensor(prior_blocks, device=a.device, dtype=a.dtype)
    noise = torch.as_tensor(noise_covariance, device=a.device, dtype=a.dtype)
    aliases, measurement_rows, target_rows = _validate_block_shapes(a, t, prior)
    batch_shape = _broadcast_block_batch_shape(a, t, prior)
    noise = _canonical_noise_for_block_batch(
        noise,
        batch_shape=batch_shape,
        measurement_rows=measurement_rows,
    )
    _validate_finite(a, "measurement_blocks")
    _validate_finite(t, "target_blocks")
    _validate_finite(prior, "prior_blocks")
    _validate_finite(noise, "noise_covariance")

    sigma_y_signal = None
    sigma_z = None
    sigma_yz = None
    for q in range(aliases):
        aq = a[..., q, :, :]
        tq = t[..., q, :, :]
        cq = prior[..., q, :, :]
        aq_cq = aq @ cq
        tq_cq = tq @ cq
        term_y = aq_cq @ _adjoint(aq)
        term_z = tq_cq @ _adjoint(tq)
        term_yz = aq_cq @ _adjoint(tq)
        sigma_y_signal = term_y if sigma_y_signal is None else sigma_y_signal + term_y
        sigma_z = term_z if sigma_z is None else sigma_z + term_z
        sigma_yz = term_yz if sigma_yz is None else sigma_yz + term_yz

    sigma_y = _hermitian(sigma_y_signal + noise)
    sigma_z = _hermitian(sigma_z)
    return MeasurementTargetCovariances(
        sigma_y=sigma_y,
        sigma_z=sigma_z,
        sigma_yz=sigma_yz,
    )


def accumulate_measurement_space_covariances(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
    noise_covariance: torch.Tensor,
    *,
    allow_singular_target: bool = False,
) -> MeasurementSpaceCovariances:
    """Stream Q64 prior blocks into small measurement-space covariances."""

    raw = accumulate_measurement_target_covariances(
        measurement_blocks,
        target_blocks,
        prior_blocks,
        noise_covariance,
    )
    sigma_y = raw.sigma_y
    sigma_z = raw.sigma_z
    sigma_yz = raw.sigma_yz
    if allow_singular_target:
        detached = sigma_z.detach()
        eigenvalues, eigenvectors = torch.linalg.eigh(detached)
        scale = eigenvalues.amax(dim=-1, keepdim=True).clamp_min(0.0)
        tolerance = (
            128.0
            * torch.finfo(eigenvalues.dtype).eps
            * max(1, int(sigma_z.shape[-1]))
            * scale
        )
        if bool((eigenvalues < -tolerance).any()):
            raise RuntimeError("target covariance is not positive semidefinite")
        inverse = torch.where(
            eigenvalues > tolerance,
            eigenvalues.reciprocal(),
            torch.zeros_like(eigenvalues),
        )
        target_pseudoinverse = (
            eigenvectors * inverse.unsqueeze(-2)
        ) @ _adjoint(eigenvectors)
        solved = target_pseudoinverse @ _adjoint(sigma_yz)
    else:
        chol_z, info_z = torch.linalg.cholesky_ex(sigma_z)
        if bool((info_z.detach() != 0).any()):
            raise RuntimeError("target covariance is not positive definite")
        solved = torch.cholesky_solve(_adjoint(sigma_yz), chol_z)
    sigma_y_given_z = _hermitian(sigma_y - sigma_yz @ solved)
    return MeasurementSpaceCovariances(
        sigma_y=sigma_y,
        sigma_z=sigma_z,
        sigma_yz=sigma_yz,
        sigma_y_given_z=sigma_y_given_z,
    )


def _checked_logdet_spd(value: torch.Tensor, name: str) -> torch.Tensor:
    chol, info = torch.linalg.cholesky_ex(_hermitian(value))
    if bool((info.detach() != 0).any()):
        raise RuntimeError(f"{name} is not positive definite")
    diagonal = torch.diagonal(chol, dim1=-2, dim2=-1).real
    return 2.0 * torch.log(diagonal).sum(dim=-1)


def target_information_from_alias_blocks(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
    noise_covariance: torch.Tensor,
    *,
    frequency_dims: tuple[int, ...] = (-2, -1),
    allow_singular_target: bool = False,
) -> TargetInformationResult:
    """Evaluate exact Gaussian ``I(Y; Z)`` without a dense latent posterior.

    ``frequency_dims`` selects the coarse-frequency axes in the local result.
    The default expects blocks batched as ``[..., Hc, Wc, 64, rows, K]``.
    """

    covariances = accumulate_measurement_space_covariances(
        measurement_blocks,
        target_blocks,
        prior_blocks,
        noise_covariance,
        allow_singular_target=allow_singular_target,
    )
    logdet_y = _checked_logdet_spd(covariances.sigma_y, "Sigma_Y")
    logdet_conditional = _checked_logdet_spd(
        covariances.sigma_y_given_z, "Sigma_Y_given_Z"
    )
    local_bits = 0.5 * (logdet_y - logdet_conditional) / math.log(2.0)
    if frequency_dims:
        averaged = local_bits.mean(dim=frequency_dims)
    else:
        averaged = local_bits
    return TargetInformationResult(
        local_bits_per_cell=local_bits,
        bits_per_raw_pixel=averaged / RAW_SAMPLES_PER_CELL,
        covariances=covariances,
    )


def target_information_real_field_q64(
    measurement_blocks: torch.Tensor,
    target_blocks: torch.Tensor,
    prior_blocks: torch.Tensor,
    noise_covariance: torch.Tensor,
    *,
    fine_grid_shape: tuple[int, int],
    aggregation_weights: torch.Tensor | None = None,
    allow_singular_target: bool = False,
) -> TargetInformationResult:
    """Evaluate Q64 information with exact real-field self-conjugate cells.

    The ordinary alias-block contraction is valid away from coarse DC/Nyquist
    cells. At each self-conjugate cell this wrapper replaces the circular-complex
    construction by :func:`build_real_field_self_conjugate_blocks`, then applies
    the same small-matrix covariance contraction. The returned frequency mean
    retains the manuscript's real-field factor ``1/2`` and raw-cell factor
    ``1/4``.
    """

    a = torch.as_tensor(measurement_blocks)
    t = torch.as_tensor(target_blocks, device=a.device, dtype=a.dtype)
    prior = torch.as_tensor(prior_blocks, device=a.device, dtype=a.dtype)
    if a.ndim < 5:
        raise ValueError(
            "production blocks must have shape [..., Hc, Wc, 64, rows, K]"
        )
    coarse_shape = tuple(map(int, a.shape[-5:-3]))
    expected_coarse = (
        int(fine_grid_shape[0]) // SCENE_TO_RGGB_CELL_DECIMATION,
        int(fine_grid_shape[1]) // SCENE_TO_RGGB_CELL_DECIMATION,
    )
    if coarse_shape != expected_coarse:
        raise ValueError(
            f"block coarse grid {coarse_shape} does not match fine grid "
            f"{fine_grid_shape} with decimation eight"
        )
    if tuple(t.shape[-5:-3]) != coarse_shape or tuple(prior.shape[-5:-3]) != coarse_shape:
        raise ValueError("measurement, target, and prior coarse grids must match")
    if int(a.shape[-2]) != RAW_SAMPLES_PER_CELL:
        raise ValueError("production Q64 measurement must have exactly four RGGB rows")
    target_rows = int(t.shape[-2])
    if target_rows <= 0 or target_rows % 4:
        raise ValueError(
            "production Q64 target rows must contain a positive multiple of "
            "four spatial phases"
        )
    try:
        batch_prefix = tuple(
            torch.broadcast_shapes(
                tuple(a.shape[:-5]), tuple(t.shape[:-5]), tuple(prior.shape[:-5])
            )
        )
    except RuntimeError as error:
        raise ValueError(
            "measurement, target, and prior Q64 batch prefixes are not broadcastable"
        ) from error

    noise = torch.as_tensor(noise_covariance, device=a.device, dtype=a.dtype)
    noise = _canonical_q64_noise(
        noise,
        batch_prefix=batch_prefix,
        coarse_shape=coarse_shape,
        measurement_rows=RAW_SAMPLES_PER_CELL,
    )

    result = target_information_from_alias_blocks(
        a,
        t,
        prior,
        noise,
        allow_singular_target=allow_singular_target,
    )
    local_bits = result.local_bits_per_cell.clone()
    sigma_y = result.covariances.sigma_y.clone()
    sigma_z = result.covariances.sigma_z.clone()
    sigma_yz = result.covariances.sigma_yz.clone()
    sigma_conditional = result.covariances.sigma_y_given_z.clone()
    for u_y, u_x in self_conjugate_coarse_indices(coarse_shape):
        real_blocks = build_real_field_self_conjugate_blocks(
            a[..., u_y, u_x, :, :, :],
            t[..., u_y, u_x, :, :, :],
            prior[..., u_y, u_x, :, :, :],
            coarse_index_yx=(u_y, u_x),
            fine_grid_shape=fine_grid_shape,
        )
        cell_noise = noise[..., u_y, u_x, :, :]
        cell = target_information_from_alias_blocks(
            real_blocks.measurement_blocks,
            real_blocks.target_blocks,
            real_blocks.prior_blocks,
            cell_noise.real,
            frequency_dims=(),
            allow_singular_target=allow_singular_target,
        )
        local_bits[..., u_y, u_x] = cell.local_bits_per_cell
        sigma_y[..., u_y, u_x, :, :] = cell.covariances.sigma_y
        sigma_z[..., u_y, u_x, :, :] = cell.covariances.sigma_z
        sigma_yz[..., u_y, u_x, :, :] = cell.covariances.sigma_yz
        sigma_conditional[..., u_y, u_x, :, :] = cell.covariances.sigma_y_given_z

    if aggregation_weights is None:
        aggregate = local_bits.mean(dim=(-2, -1))
    else:
        weights = torch.as_tensor(aggregation_weights, device=a.device)
        if weights.is_complex() or not weights.is_floating_point():
            raise ValueError("aggregation_weights must be real floating")
        if weights.requires_grad:
            raise ValueError("aggregation_weights must be frozen")
        if tuple(weights.shape) != coarse_shape:
            raise ValueError(
                "aggregation_weights must exactly match the Q64 coarse grid "
                f"{coarse_shape}"
            )
        weights = weights.to(dtype=local_bits.dtype)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("aggregation_weights must be finite")
        if bool((weights < 0).any()):
            raise ValueError("aggregation_weights must be non-negative")
        total_weight = weights.sum()
        if not bool(total_weight > 0):
            raise ValueError("aggregation_weights must have a positive sum")
        normalized_weights = weights / total_weight
        aggregate = (local_bits * normalized_weights).sum(dim=(-2, -1))

    return TargetInformationResult(
        local_bits_per_cell=local_bits,
        bits_per_raw_pixel=aggregate / RAW_SAMPLES_PER_CELL,
        covariances=MeasurementSpaceCovariances(
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            sigma_yz=sigma_yz,
            sigma_y_given_z=sigma_conditional,
        ),
    )


__all__ = [
    "AliasBlockOperator",
    "MeasurementSpaceCovariances",
    "MeasurementTargetCovariances",
    "PRODUCTION_ALIAS_COUNT",
    "PRODUCTION_ORDERING",
    "ProductionFineGridTransfer",
    "RAW_SAMPLES_PER_CELL",
    "RealFieldAliasBlocks",
    "RGGB_CELL_PITCH_UM",
    "SCENE_GRID_SPACING_UM",
    "SCENE_TO_RGGB_CELL_DECIMATION",
    "SENSOR_PIXEL_PITCH_UM",
    "TargetInformationResult",
    "UNITARY_SAMPLING_COEFFICIENT",
    "accumulate_measurement_space_covariances",
    "accumulate_measurement_target_covariances",
    "build_calibrated_fine_grid_transfer",
    "build_normalized_sliding_target_otf",
    "build_real_field_self_conjugate_blocks",
    "build_q64_full_rgb_target_blocks",
    "build_q64_rggb_measurement_blocks",
    "gather_q64_prior_blocks",
    "gather_q64_spectrum",
    "self_conjugate_coarse_indices",
    "spatial_kernel_to_convolution_otf",
    "target_information_from_alias_blocks",
    "target_information_real_field_q64",
]
