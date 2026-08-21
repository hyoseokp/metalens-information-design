"""Differentiable sensor-domain information primitives.

This module keeps two operating points deliberately separate:

``fixed_exposure``
    The caller supplies an *absolute* source spectral irradiance density,
    optical-sample area, and exposure.  Optical throughput therefore changes
    both the electron-domain transfer and the Poisson variance.

``fixed_signal``
    The caller supplies only a relative source spectrum and an explicit target
    mean electron count.  Each sensor row is independently gain-matched to that
    target, independently for every channel and every leading batch/field item.
    This is a shape-only diagnostic: it is not a shared physical exposure and
    must not be used when batch members belong to one exposure group.

``calibrated_fixed_exposure``
    A frozen scalar converts the relative simulator response to electrons.  It
    does not claim an absolute lux calibration, but because the same scalar is
    reused for every design it preserves throughput and shot-noise differences.

``global_fixed_signal``
    One global gain matches a reference statistic (brightest channel by
    default) to a target electron count.  Channel ratios are preserved.  This
    is the transfer-level analogue of global ETTR auto-exposure.  The complete
    supplied batch is one exposure group; call it separately for independent
    exposures or supply their frozen external reference statistics.

All paths use the same irregular-grid trapezoid quadrature, photon-energy
factor, CFA/QE response, and pixel integration.  The returned tensors can feed
either a diagonal RGB information objective or a full colour-MIMO log-det.

Unit convention for ``fixed_exposure``
--------------------------------------
``spectral_psf_relative[..., l, y, x]`` is a dimensionless spatial response
per unit source irradiance. ``source_spectral_irradiance_w_m2_per_m[l]`` is in
W m^-2 m^-1 and is sampled on ``wavelengths_um``.  Summing each optical sample
over its physical area produces W m^-1 at a sensor pixel.  Spectral integration,
``lambda/(h c)``, QE, and exposure then produce electrons.  If the simulator's
PSF does not obey this convention, use ``fixed_signal`` or first provide an
explicit radiometric calibration; arbitrary simulator units must not be
reported as absolute electrons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from .spectral import photon_factor_from_irradiance


class InformationMode(str, Enum):
    """Sensor operating point used to construct an electron transfer."""

    FIXED_EXPOSURE = "fixed_exposure"
    FIXED_SIGNAL = "fixed_signal"
    CALIBRATED_FIXED_EXPOSURE = "calibrated_fixed_exposure"
    GLOBAL_FIXED_SIGNAL = "global_fixed_signal"


@dataclass
class SensorInformationTransfer:
    """Electron-domain transfer shared by diagonal and colour-MIMO metrics.

    Shapes use ``...`` for optional field/batch dimensions, ``C`` for sensor
    channels, ``L`` for wavelengths, and ``(H, W)`` for the pixel grid.

    Attributes
    ----------
    spectral_electron_psf:
        Per-wavelength, per-sensor-channel impulse response
        ``[..., C, L, H, W]`` after the selected operating-point row scaling.
    spectral_otf_e:
        Complex, unshifted ``torch.fft.fft2`` of ``spectral_electron_psf`` with
        identical shape.  DC is at ``[..., 0, 0]``; any PSD/mask paired with it
        must use this same unshifted bin layout.
    diagonal_otf_e:
        Broadband diagonal transfer ``[..., C, H, W]`` obtained by summing L.
    normalized_diagonal_otf:
        ``diagonal_otf_e / H_c(0)``.  Its DC is one, but unlike
        ``diagonal_otf_e`` it contains no absolute electron gain.
    mean_electrons:
        Mean electrons ``[..., C]`` after the selected row/global gain.
    uncalibrated_mean_electrons:
        Historical field name for the DC response before any fixed-signal row
        or global gain.  Prefer :attr:`pre_gain_mean_electrons` in new code.  It
        is physical only in ``FIXED_EXPOSURE`` when the input unit convention is
        satisfied.  In ``CALIBRATED_FIXED_EXPOSURE`` it is already a calibrated
        electron count tied to the frozen scalar, but not to an absolute lux.
    noise_variance_e2:
        Gaussian approximation to shot + read variance, ``mu + sigma_read^2``.
    """

    mode: InformationMode
    wavelengths_um: torch.Tensor
    spectral_electron_psf: torch.Tensor
    spectral_otf_e: torch.Tensor
    diagonal_otf_e: torch.Tensor
    normalized_diagonal_otf: torch.Tensor
    mean_electrons: torch.Tensor
    uncalibrated_mean_electrons: torch.Tensor
    noise_variance_e2: torch.Tensor
    row_scale: torch.Tensor

    @property
    def pre_gain_mean_electrons(self) -> torch.Tensor:
        """Unambiguous alias for the historical uncalibrated-mean field."""

        return self.uncalibrated_mean_electrons


def _validate_finite(tensor: torch.Tensor, name: str) -> None:
    """Value-only validation that leaves the differentiable tensor untouched."""

    if not bool(torch.isfinite(tensor.detach()).all()):
        raise ValueError(f"{name} must be finite")


def _validate_finite_nonnegative(tensor: torch.Tensor, name: str) -> None:
    _validate_finite(tensor, name)
    if not bool((tensor.detach() >= 0).all()):
        raise ValueError(f"{name} must be non-negative")


def clamp_tiny_negative_roundoff(
    tensor: torch.Tensor,
    name: str,
    *,
    relative_tolerance: float = 1e-9,
) -> torch.Tensor:
    """Clamp propagation-scale negative round-off without hiding bad physics.

    Vectorial FFT/Poynting accumulation can leave isolated negative samples at
    roughly machine precision. The stored-PSF rescoring path already accepts
    values no smaller than ``-tol * max(peak, 1)``; the differentiable live
    path must use the same policy or identical off-axis data can be accepted
    offline and rejected during optimization. Material negatives still fail.
    ``clamp_min`` preserves the autograd graph for all positive samples.
    """

    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    if not bool(torch.isfinite(tensor).all().detach()):
        raise ValueError(f"{name} must be finite")
    detached = tensor.detach()
    peak = detached.max()
    scale = torch.maximum(peak, peak.new_tensor(1.0))
    if bool((detached.min() < -tolerance * scale).detach()):
        raise ValueError(f"{name} must be non-negative beyond round-off tolerance")
    return tensor.clamp_min(0.0)


def _validate_broadcasts_to(
    source_shape: torch.Size | tuple[int, ...],
    target_shape: torch.Size | tuple[int, ...],
    name: str,
) -> None:
    """Require normal PyTorch broadcasting to produce exactly target_shape."""

    source = tuple(source_shape)
    target = tuple(target_shape)
    try:
        result = tuple(torch.broadcast_shapes(source, target))
    except RuntimeError as exc:
        raise ValueError(f"{name} shape {source} cannot broadcast to {target}") from exc
    if result != target:
        raise ValueError(f"{name} shape {source} cannot broadcast to {target}")


def _validate_hermitian(
    covariance: torch.Tensor, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached Hermitian value/tolerance without altering autograd."""

    _validate_finite(covariance, name)
    value = covariance.detach()
    adjoint = value.conj().transpose(-2, -1)
    real_dtype = value.real.dtype
    tolerance = 64.0 * torch.finfo(real_dtype).eps * torch.maximum(
        value.abs().amax(), torch.ones((), device=value.device, dtype=real_dtype)
    )
    if not bool(((value - adjoint).abs() <= tolerance).all()):
        raise ValueError(f"{name} must be Hermitian")
    return 0.5 * (value + adjoint), tolerance


def _validate_hermitian_psd(covariance: torch.Tensor, name: str) -> None:
    hermitian, tolerance = _validate_hermitian(covariance, name)
    eigenvalues = torch.linalg.eigvalsh(hermitian)
    if not bool((eigenvalues >= -tolerance).all()):
        raise ValueError(f"{name} must be positive semidefinite")


def trapezoid_bin_widths_m(
    wavelengths_um: torch.Tensor,
    *,
    single_wavelength_bin_width_um: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return trapezoid quadrature weights in metres for an irregular grid.

    For one wavelength there is no inferable integration interval, so an
    explicit ``single_wavelength_bin_width_um`` is mandatory.  For two or more
    samples, wavelengths must be finite and strictly increasing.
    """

    wl = torch.as_tensor(wavelengths_um)
    if not wl.is_floating_point() or wl.ndim != 1 or wl.numel() == 0:
        raise ValueError("wavelengths_um must be a non-empty 1D floating tensor")
    if not bool(torch.isfinite(wl).all().detach()):
        raise ValueError("wavelengths_um must be finite")
    if not bool((wl.detach() > 0).all()):
        raise ValueError("wavelengths_um must be positive")

    if wl.numel() == 1:
        if single_wavelength_bin_width_um is None:
            raise ValueError(
                "a one-sample wavelength grid requires "
                "single_wavelength_bin_width_um"
            )
        width_um = torch.as_tensor(
            single_wavelength_bin_width_um, device=wl.device, dtype=wl.dtype
        )
        if (
            width_um.numel() != 1
            or not bool(torch.isfinite(width_um).detach())
            or not bool((width_um > 0).detach())
        ):
            raise ValueError(
                "single_wavelength_bin_width_um must be finite and positive"
            )
        return width_um.reshape(1) * 1e-6

    delta = wl[1:] - wl[:-1]
    if not bool((delta > 0).all().detach()):
        raise ValueError("wavelengths_um must be strictly increasing")
    weights_um = torch.empty_like(wl)
    weights_um[0] = 0.5 * delta[0]
    weights_um[-1] = 0.5 * delta[-1]
    weights_um[1:-1] = 0.5 * (wl[2:] - wl[:-2])
    return weights_um * 1e-6


def pixel_integrate_samples(
    samples: torch.Tensor,
    out_shape: tuple[int, int],
    *,
    optical_sample_area_m2: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Exactly sum a uniform optical grid into aligned rectangular pixels.

    ``samples`` has shape ``[..., H_opt, W_opt]``.  The optical grid must divide
    the output grid exactly; silent adaptive pooling would not conserve a known
    physical area.  Set ``optical_sample_area_m2=1`` for relative fixed-signal
    calculations, where the common spatial scale is removed by row matching.
    """

    if samples.ndim < 2 or not samples.is_floating_point():
        raise ValueError("samples must be a floating tensor with at least 2 dimensions")
    _validate_finite(samples, "samples")
    out_h, out_w = (int(out_shape[0]), int(out_shape[1]))
    height, width = samples.shape[-2:]
    if out_h <= 0 or out_w <= 0 or height % out_h or width % out_w:
        raise ValueError(
            f"optical grid {(height, width)} must be exactly divisible by "
            f"pixel grid {(out_h, out_w)}"
        )
    kernel_h, kernel_w = height // out_h, width // out_w
    reshaped = samples.reshape(
        *samples.shape[:-2], out_h, kernel_h, out_w, kernel_w
    )
    integrated = reshaped.sum(dim=(-3, -1))
    area = torch.as_tensor(
        optical_sample_area_m2, device=samples.device, dtype=samples.dtype
    )
    if (
        area.numel() != 1
        or not bool(torch.isfinite(area).detach())
        or not bool((area > 0).detach())
    ):
        raise ValueError("optical_sample_area_m2 must be finite and positive")
    return integrated * area


def _channel_response(
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    n_wavelengths: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    def collapse_spatial_singletons(value: torch.Tensor, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        while value.ndim > 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim > 2:
            raise ValueError(f"{name} may only have trailing singleton spatial axes")
        return value

    cfa = collapse_spatial_singletons(cfa_transmission, "cfa_transmission")
    if cfa.ndim != 2 or cfa.shape[1] != n_wavelengths:
        raise ValueError(
            f"cfa_transmission must have shape [C, {n_wavelengths}], "
            f"got {tuple(cfa.shape)}"
        )
    qe_t = collapse_spatial_singletons(qe, "qe")
    if qe_t.ndim == 1:
        if qe_t.numel() != n_wavelengths:
            raise ValueError(f"qe must have {n_wavelengths} wavelength samples")
        qe_t = qe_t.unsqueeze(0).expand(cfa.shape[0], -1)
    elif qe_t.shape != cfa.shape:
        raise ValueError(
            f"qe must have shape [{n_wavelengths}] or {tuple(cfa.shape)}, "
            f"got {tuple(qe_t.shape)}"
        )
    _validate_finite_nonnegative(cfa, "cfa_transmission")
    _validate_finite_nonnegative(qe_t, "qe")
    return cfa * qe_t


def _spectral_electron_psf(
    spectral_psf_relative: torch.Tensor,
    wavelengths_um: torch.Tensor,
    source_spectrum: torch.Tensor,
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    pixel_grid_shape: tuple[int, int],
    optical_sample_area_m2: float | torch.Tensor,
    exposure_s: float | torch.Tensor,
    single_wavelength_bin_width_um: float | torch.Tensor | None,
) -> torch.Tensor:
    if spectral_psf_relative.ndim < 3 or not spectral_psf_relative.is_floating_point():
        raise ValueError(
            "spectral_psf_relative must be floating with shape [..., L, H, W]"
        )
    spectral_psf_relative = clamp_tiny_negative_roundoff(
        spectral_psf_relative, "spectral_psf_relative",
    )
    device, dtype = spectral_psf_relative.device, spectral_psf_relative.dtype
    wl = torch.as_tensor(wavelengths_um, device=device, dtype=dtype)
    n_wl = int(wl.numel())
    if wl.ndim != 1 or spectral_psf_relative.shape[-3] != n_wl:
        raise ValueError(
            "the spectral PSF wavelength axis must match wavelengths_um"
        )
    spectrum = torch.as_tensor(source_spectrum, device=device, dtype=dtype)
    if spectrum.shape != (n_wl,):
        raise ValueError(f"source spectrum must have shape [{n_wl}]")
    _validate_finite_nonnegative(spectrum, "source spectrum")
    response = _channel_response(
        cfa_transmission, qe, n_wavelengths=n_wl, device=device, dtype=dtype
    )
    bin_width_m = trapezoid_bin_widths_m(
        wl, single_wavelength_bin_width_um=single_wavelength_bin_width_um
    )
    exposure = torch.as_tensor(exposure_s, device=device, dtype=dtype)
    if (
        exposure.numel() != 1
        or not bool(torch.isfinite(exposure).detach())
        or not bool((exposure > 0).detach())
    ):
        raise ValueError("exposure_s must be finite and positive")

    pixel_spectral = pixel_integrate_samples(
        spectral_psf_relative,
        pixel_grid_shape,
        optical_sample_area_m2=optical_sample_area_m2,
    )  # [..., L, H, W]
    photon_factor = photon_factor_from_irradiance(wl)
    spectral_factor = spectrum * bin_width_m * photon_factor * exposure
    electron_weight = response * spectral_factor.unsqueeze(0)  # [C, L]
    prefix_ndim = pixel_spectral.ndim - 3
    weight_shape = (1,) * prefix_ndim + (*electron_weight.shape, 1, 1)
    return pixel_spectral.unsqueeze(-4) * electron_weight.reshape(weight_shape)


def _target_mean_tensor(
    target_mean_electrons: float | torch.Tensor,
    raw_mean: torch.Tensor,
) -> torch.Tensor:
    target = torch.as_tensor(
        target_mean_electrons, device=raw_mean.device, dtype=raw_mean.dtype
    )
    _validate_broadcasts_to(target.shape, raw_mean.shape, "target_mean_electrons")
    if not bool(torch.isfinite(target.detach()).all()) or not bool(
        (target.detach() > 0).all()
    ):
        raise ValueError("target_mean_electrons must be finite and positive")
    return torch.broadcast_to(target, raw_mean.shape)


def _prepare_transfer(
    spectral_electron_psf: torch.Tensor,
    wavelengths_um: torch.Tensor,
    *,
    mode: InformationMode,
    read_noise_e: float | torch.Tensor,
    target_mean_electrons: float | torch.Tensor | None = None,
    global_reference_statistic: str | float | torch.Tensor | None = None,
    eps: float = 1e-30,
) -> SensorInformationTransfer:
    _validate_finite_nonnegative(spectral_electron_psf, "spectral_electron_psf")
    raw_spectral_otf = torch.fft.fft2(spectral_electron_psf, dim=(-2, -1))
    raw_diagonal_otf = raw_spectral_otf.sum(dim=-3)
    raw_mean = raw_diagonal_otf[..., 0, 0].real
    if not bool(torch.isfinite(raw_mean).all().detach()) or not bool(
        (raw_mean > eps).all().detach()
    ):
        raise ValueError("every sensor channel must have finite positive DC throughput")

    if mode in (
        InformationMode.FIXED_EXPOSURE,
        InformationMode.CALIBRATED_FIXED_EXPOSURE,
    ):
        if target_mean_electrons is not None:
            raise ValueError(f"{mode.value} does not accept target_mean_electrons")
        if global_reference_statistic is not None:
            raise ValueError(f"{mode.value} does not accept a global reference")
        mean_e = raw_mean
        row_scale = torch.ones_like(raw_mean)
    elif mode is InformationMode.FIXED_SIGNAL:
        if target_mean_electrons is None:
            raise ValueError("fixed_signal requires target_mean_electrons")
        mean_e = _target_mean_tensor(target_mean_electrons, raw_mean)
        row_scale = mean_e / raw_mean
    elif mode is InformationMode.GLOBAL_FIXED_SIGNAL:
        if target_mean_electrons is None:
            raise ValueError("global_fixed_signal requires target_mean_electrons")
        target = torch.as_tensor(
            target_mean_electrons, device=raw_mean.device, dtype=raw_mean.dtype
        )
        if (
            target.numel() != 1
            or not bool(torch.isfinite(target).detach())
            or not bool((target > 0).detach())
        ):
            raise ValueError(
                "global fixed-signal target_mean_electrons must be a finite "
                "positive scalar"
            )
        if global_reference_statistic is None:
            global_reference_statistic = "brightest_channel"
        if isinstance(global_reference_statistic, str):
            if global_reference_statistic == "brightest_channel":
                reference = raw_mean.max()
            elif global_reference_statistic == "mean_channel":
                reference = raw_mean.mean()
            else:
                raise ValueError(
                    "global_reference_statistic must be 'brightest_channel', "
                    "'mean_channel', or an explicit positive scalar"
                )
        else:
            reference = torch.as_tensor(
                global_reference_statistic,
                device=raw_mean.device,
                dtype=raw_mean.dtype,
            )
            if reference.numel() != 1:
                raise ValueError("an explicit global reference must be scalar")
        if not bool(torch.isfinite(reference).detach()) or not bool(
            (reference > eps).detach()
        ):
            raise ValueError("global reference statistic must be finite and positive")
        global_scale = target.reshape(()) / reference.reshape(())
        row_scale = torch.ones_like(raw_mean) * global_scale
        mean_e = raw_mean * row_scale
    else:  # defensive if constructed from a non-enum value internally
        raise ValueError(f"unsupported information mode {mode!r}")

    scale = row_scale[..., :, None, None, None]
    spectral_psf = spectral_electron_psf * scale
    spectral_otf = raw_spectral_otf * scale
    diagonal_otf = spectral_otf.sum(dim=-3)
    normalized = diagonal_otf / mean_e[..., :, None, None]
    read = torch.as_tensor(read_noise_e, device=mean_e.device, dtype=mean_e.dtype)
    _validate_broadcasts_to(read.shape, mean_e.shape, "read_noise_e")
    if not bool(torch.isfinite(read.detach()).all()) or not bool(
        (read.detach() >= 0).all()
    ):
        raise ValueError("read_noise_e must be finite and non-negative")
    read = torch.broadcast_to(read, mean_e.shape)
    noise_variance = mean_e + read.square()
    return SensorInformationTransfer(
        mode=mode,
        wavelengths_um=torch.as_tensor(
            wavelengths_um, device=mean_e.device, dtype=mean_e.dtype
        ),
        spectral_electron_psf=spectral_psf,
        spectral_otf_e=spectral_otf,
        diagonal_otf_e=diagonal_otf,
        normalized_diagonal_otf=normalized,
        mean_electrons=mean_e,
        uncalibrated_mean_electrons=raw_mean,
        noise_variance_e2=noise_variance,
        row_scale=row_scale,
    )


def build_fixed_exposure_transfer(
    spectral_psf_relative: torch.Tensor,
    wavelengths_um: torch.Tensor,
    source_spectral_irradiance_w_m2_per_m: torch.Tensor,
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    pixel_grid_shape: tuple[int, int],
    optical_sample_pitch_um: float,
    exposure_s: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    single_wavelength_bin_width_um: float | torch.Tensor | None = None,
) -> SensorInformationTransfer:
    """Build a throughput-sensitive transfer at one fixed physical exposure.

    This function intentionally requires the source spectral irradiance density
    and optical-sample pitch.  It must not be called with arbitrary normalized
    simulator intensity when absolute electron counts are claimed.
    """

    pitch_um = float(optical_sample_pitch_um)
    if not math.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError("optical_sample_pitch_um must be finite and positive")
    area_m2 = (pitch_um * 1e-6) ** 2
    spectral_e = _spectral_electron_psf(
        spectral_psf_relative,
        wavelengths_um,
        source_spectral_irradiance_w_m2_per_m,
        cfa_transmission,
        qe,
        pixel_grid_shape=pixel_grid_shape,
        optical_sample_area_m2=area_m2,
        exposure_s=exposure_s,
        single_wavelength_bin_width_um=single_wavelength_bin_width_um,
    )
    return _prepare_transfer(
        spectral_e,
        wavelengths_um,
        mode=InformationMode.FIXED_EXPOSURE,
        read_noise_e=read_noise_e,
    )


def build_calibrated_fixed_exposure_transfer(
    spectral_psf_relative: torch.Tensor,
    wavelengths_um: torch.Tensor,
    relative_source_spectrum: torch.Tensor,
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    pixel_grid_shape: tuple[int, int],
    electron_calibration: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    single_wavelength_bin_width_um: float | torch.Tensor | None = None,
) -> SensorInformationTransfer:
    """Apply one frozen relative-response-to-electron calibration.

    This is the optimizer-facing fixed-exposure mode for a simulator whose
    irradiance scale is relative.  ``electron_calibration`` multiplies the
    relative electron proxy produced by spectral quadrature, photon weighting,
    CFA/QE, and pixel summation.  Calibrate it once at a declared reference
    design/exposure and reuse the exact same scalar for every candidate.  Never
    recalibrate per design: doing so would silently turn this into fixed-signal
    comparison and erase throughput differences.

    The result is in calibrated electrons for shot/read-noise calculations,
    but the scalar alone is not an absolute radiometric or lux calibration.
    """

    calibration = torch.as_tensor(
        electron_calibration,
        device=spectral_psf_relative.device,
        dtype=spectral_psf_relative.dtype,
    )
    if (
        calibration.numel() != 1
        or not bool(torch.isfinite(calibration).detach())
        or not bool((calibration > 0).detach())
    ):
        raise ValueError("electron_calibration must be a finite positive scalar")
    if calibration.requires_grad:
        raise ValueError(
            "electron_calibration must be frozen (requires_grad=False)"
        )
    spectral_e_relative = _spectral_electron_psf(
        spectral_psf_relative,
        wavelengths_um,
        relative_source_spectrum,
        cfa_transmission,
        qe,
        pixel_grid_shape=pixel_grid_shape,
        optical_sample_area_m2=1.0,
        exposure_s=1.0,
        single_wavelength_bin_width_um=single_wavelength_bin_width_um,
    )
    return _prepare_transfer(
        spectral_e_relative * calibration.reshape(()),
        wavelengths_um,
        mode=InformationMode.CALIBRATED_FIXED_EXPOSURE,
        read_noise_e=read_noise_e,
    )


def build_fixed_signal_transfer(
    spectral_psf_relative: torch.Tensor,
    wavelengths_um: torch.Tensor,
    relative_source_spectrum: torch.Tensor,
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    pixel_grid_shape: tuple[int, int],
    target_mean_electrons: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    single_wavelength_bin_width_um: float | torch.Tensor | None = None,
) -> SensorInformationTransfer:
    """Build a gain-matched transfer for shape-only/fixed-signal comparison.

    ``relative_source_spectrum`` fixes the spectral *shape* and
    ``target_mean_electrons`` fixes the operating SNR.  Overall simulator and
    source scales cancel independently for each channel and each leading
    batch/field item.  Those rows do *not* share an exposure group.  The
    :attr:`SensorInformationTransfer.pre_gain_mean_electrons` value is therefore
    only a relative diagnostic and must not be labelled as a physical count.
    """

    spectral_e_relative = _spectral_electron_psf(
        spectral_psf_relative,
        wavelengths_um,
        relative_source_spectrum,
        cfa_transmission,
        qe,
        pixel_grid_shape=pixel_grid_shape,
        optical_sample_area_m2=1.0,
        exposure_s=1.0,
        single_wavelength_bin_width_um=single_wavelength_bin_width_um,
    )
    return _prepare_transfer(
        spectral_e_relative,
        wavelengths_um,
        mode=InformationMode.FIXED_SIGNAL,
        read_noise_e=read_noise_e,
        target_mean_electrons=target_mean_electrons,
    )


def build_global_fixed_signal_transfer(
    spectral_psf_relative: torch.Tensor,
    wavelengths_um: torch.Tensor,
    relative_source_spectrum: torch.Tensor,
    cfa_transmission: torch.Tensor,
    qe: torch.Tensor,
    *,
    pixel_grid_shape: tuple[int, int],
    target_reference_electrons: float | torch.Tensor,
    read_noise_e: float | torch.Tensor,
    reference_statistic: str | float | torch.Tensor = "brightest_channel",
    single_wavelength_bin_width_um: float | torch.Tensor | None = None,
) -> SensorInformationTransfer:
    """Apply one global auto-exposure gain while preserving channel ratios.

    The default reference is the largest unscaled channel mean across all
    supplied batch/field points: the entire call is one exposure group.
    ``"mean_channel"`` uses that group's global mean.
    A positive scalar may instead be supplied in the same relative-response
    units, e.g. an unscaled brightest-scene-pixel statistic computed by a scene
    forward.  That explicit statistic is required to reproduce stage5's pixel
    ETTR exactly; the default is a transfer-level channel-mean proxy.
    Independent images/exposures must be passed in separate calls (or use a
    separately frozen explicit reference statistic for each exposure group).

    Unlike :func:`build_fixed_signal_transfer`, every channel and batch item
    receives the same row scale.  Thus R/G/B ratios are retained even though a
    uniform throughput scale is removed by global auto-exposure.
    """

    spectral_e_relative = _spectral_electron_psf(
        spectral_psf_relative,
        wavelengths_um,
        relative_source_spectrum,
        cfa_transmission,
        qe,
        pixel_grid_shape=pixel_grid_shape,
        optical_sample_area_m2=1.0,
        exposure_s=1.0,
        single_wavelength_bin_width_um=single_wavelength_bin_width_um,
    )
    return _prepare_transfer(
        spectral_e_relative,
        wavelengths_um,
        mode=InformationMode.GLOBAL_FIXED_SIGNAL,
        read_noise_e=read_noise_e,
        target_mean_electrons=target_reference_electrons,
        global_reference_statistic=reference_statistic,
    )


def color_mixing_otf(
    transfer: SensorInformationTransfer,
    scene_spectral_basis: torch.Tensor,
) -> torch.Tensor:
    """Construct ``M_ak(nu)`` from per-wavelength complex electron OTFs.

    ``scene_spectral_basis`` has shape ``[K, L]`` and maps K latent scene
    colours to the sampled spectrum.  The result is
    ``[..., H, W, C_sensor, K_scene]``.  Keeping the complex OTF preserves
    field-dependent phase and colour mixing needed by a MIMO objective.
    """

    basis = torch.as_tensor(
        scene_spectral_basis,
        device=transfer.spectral_otf_e.device,
        dtype=transfer.spectral_otf_e.dtype,
    )
    n_wl = transfer.spectral_otf_e.shape[-3]
    if basis.ndim != 2 or basis.shape[1] != n_wl:
        raise ValueError(f"scene_spectral_basis must have shape [K, {n_wl}]")
    _validate_finite(transfer.spectral_otf_e, "transfer.spectral_otf_e")
    _validate_finite(basis, "scene_spectral_basis")
    return torch.einsum("...alhw,kl->...hwak", transfer.spectral_otf_e, basis)


def diagonal_information_density(
    transfer_otf_e: torch.Tensor,
    scene_power_spectrum: torch.Tensor,
    noise_variance_e2: torch.Tensor,
    *,
    information_factor: float = 0.5,
) -> torch.Tensor:
    """Per-channel, per-frequency Gaussian information in bits.

    Inputs have shapes ``[..., C, H, W]``, ``[..., H, W]``, and ``[..., C]``.
    The returned density is ``[..., C, H, W]``.  ``information_factor=0.5``
    is the real-Gaussian convention.  Set it to 1.0 only when reproducing a
    one-sided/complex-spectrum convention that already accounts for conjugate
    frequency pairs elsewhere.

    Frequency layout is unshifted, matching :func:`torch.fft.fft2`: DC is bin
    ``[..., 0, 0]``.  ``scene_power_spectrum`` must use that same layout; do
    not pass an ``fftshift``-centred PSD without undoing the shift.
    """

    if transfer_otf_e.ndim < 3:
        raise ValueError("transfer_otf_e must have shape [..., C, H, W]")
    _validate_finite(transfer_otf_e, "transfer_otf_e")
    factor = float(information_factor)
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("information_factor must be finite and non-negative")
    channels = transfer_otf_e.shape[-3]
    noise = torch.as_tensor(
        noise_variance_e2,
        device=transfer_otf_e.device,
        dtype=transfer_otf_e.real.dtype,
    )
    if noise.ndim < 1 or noise.shape[-1:] != (channels,):
        raise ValueError(f"noise_variance_e2 must end in {channels} channels")
    _validate_broadcasts_to(
        noise.shape[:-1], transfer_otf_e.shape[:-3], "noise_variance_e2"
    )
    if not bool(torch.isfinite(noise.detach()).all()) or not bool(
        (noise.detach() > 0).all()
    ):
        raise ValueError("noise_variance_e2 must be finite and positive")
    scene = torch.as_tensor(
        scene_power_spectrum,
        device=transfer_otf_e.device,
        dtype=transfer_otf_e.real.dtype,
    )
    if scene.shape[-2:] != transfer_otf_e.shape[-2:]:
        raise ValueError("scene_power_spectrum must match the frequency grid")
    _validate_broadcasts_to(
        scene.shape[:-2], transfer_otf_e.shape[:-3], "scene_power_spectrum"
    )
    _validate_finite_nonnegative(scene, "scene_power_spectrum")
    snr = (
        transfer_otf_e.abs().square()
        * scene.unsqueeze(-3)
        / noise[..., :, None, None]
    )
    # log1p retains both value and gradient when SNR is below float32 epsilon.
    return factor * torch.log1p(snr) / math.log(2.0)


def color_mimo_information_density(
    mixing_otf_e: torch.Tensor,
    scene_covariance: torch.Tensor,
    *,
    noise_variance_e2: torch.Tensor | None = None,
    noise_covariance_e2: torch.Tensor | None = None,
    information_factor: float = 0.5,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Full colour-MIMO Gaussian log-det in bits per frequency bin.

    ``mixing_otf_e`` is ``[..., H, W, M_sensor, K_scene]``.  The scene and
    noise covariance matrices may be constant or carry the same leading batch
    dimensions; singleton frequency axes are inserted automatically.  Specify
    exactly one of diagonal ``noise_variance_e2`` or full
    ``noise_covariance_e2``.  The result has shape ``[..., H, W]``.
    Its ``(H, W)`` axes use the unshifted FFT layout with DC at ``[0, 0]``;
    frequency-resolved scene covariance must use the same layout.

    ``jitter`` is retained for API compatibility as a dimensionless extra
    noise ridge: the signal covariance is divided by ``1 + jitter``.  It never
    adds a signal-independent diagonal term, so a zero transfer returns exactly
    zero information for every valid jitter value.
    """

    if mixing_otf_e.ndim < 4:
        raise ValueError(
            "mixing_otf_e must have shape [..., H, W, M_sensor, K_scene]"
        )
    if not (mixing_otf_e.is_floating_point() or mixing_otf_e.is_complex()):
        raise ValueError("mixing_otf_e must be floating or complex")
    _validate_finite(mixing_otf_e, "mixing_otf_e")
    if (noise_variance_e2 is None) == (noise_covariance_e2 is None):
        raise ValueError(
            "specify exactly one of noise_variance_e2 or noise_covariance_e2"
        )
    factor = float(information_factor)
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("information_factor must be finite and non-negative")
    ridge = float(jitter)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("jitter must be finite and non-negative")
    m_sensor, k_scene = mixing_otf_e.shape[-2:]
    matrix_dtype = mixing_otf_e.dtype
    real_dtype = mixing_otf_e.real.dtype

    scene = torch.as_tensor(
        scene_covariance, device=mixing_otf_e.device, dtype=matrix_dtype
    )
    if scene.shape[-2:] != (k_scene, k_scene):
        raise ValueError(
            f"scene_covariance must end in shape {(k_scene, k_scene)}"
        )
    batch_shape = mixing_otf_e.shape[:-4]
    frequency_shape = mixing_otf_e.shape[-4:-2]

    def expand_covariance(covariance: torch.Tensor, name: str) -> torch.Tensor:
        """Place omitted batch/frequency singleton axes without ambiguity."""
        leading = covariance.shape[:-2]
        if not leading:
            return covariance.reshape(
                (1,) * (len(batch_shape) + 2) + covariance.shape[-2:]
            )
        if leading == batch_shape:
            return covariance.reshape(*batch_shape, 1, 1, *covariance.shape[-2:])
        if len(leading) >= 2 and leading[-2:] == frequency_shape:
            covariance_batch = leading[:-2]
            try:
                broadcast_batch = torch.broadcast_shapes(covariance_batch, batch_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"{name} batch axes {covariance_batch} do not match {batch_shape}"
                ) from exc
            if broadcast_batch != batch_shape:
                raise ValueError(
                    f"{name} batch axes {covariance_batch} do not match {batch_shape}"
                )
            pad = (1,) * (len(batch_shape) - len(covariance_batch))
            return covariance.reshape(
                *pad, *covariance_batch, *frequency_shape, *covariance.shape[-2:]
            )
        try:
            _validate_broadcasts_to(
                leading, (*batch_shape, *frequency_shape), name
            )
        except ValueError as exc:
            raise ValueError(
                f"{name} leading axes {leading} are not compatible with "
                f"{(*batch_shape, *frequency_shape)}"
            ) from exc
        return covariance

    scene = expand_covariance(scene, "scene_covariance")
    _validate_hermitian_psd(scene, "scene_covariance")

    if noise_variance_e2 is not None:
        noise = torch.as_tensor(
            noise_variance_e2, device=mixing_otf_e.device, dtype=real_dtype
        )
        if noise.ndim < 1 or noise.shape[-1:] != (m_sensor,):
            raise ValueError(
                f"noise_variance_e2 must end in {m_sensor} channels"
            )
        prefix_ndim = mixing_otf_e.ndim - 4
        _validate_broadcasts_to(
            noise.shape[:-1], batch_shape, "noise_variance_e2"
        )
        if not bool(torch.isfinite(noise.detach()).all()) or not bool(
            (noise.detach() > 0).all()
        ):
            raise ValueError("noise_variance_e2 must be finite and positive")
        noise_prefix = noise.shape[:-1]
        pad = (1,) * (prefix_ndim - len(noise_prefix))
        scale_shape = (*pad, *noise_prefix, 1, 1, m_sensor, 1)
        whitened = mixing_otf_e / noise.sqrt().reshape(scale_shape)
    else:
        noise_cov = torch.as_tensor(
            noise_covariance_e2, device=mixing_otf_e.device, dtype=matrix_dtype
        )
        if noise_cov.shape[-2:] != (m_sensor, m_sensor):
            raise ValueError(
                f"noise_covariance_e2 must end in shape {(m_sensor, m_sensor)}"
            )
        noise_cov = expand_covariance(noise_cov, "noise_covariance_e2")
        _validate_hermitian(noise_cov, "noise_covariance_e2")
        noise_chol, cholesky_info = torch.linalg.cholesky_ex(
            noise_cov, check_errors=False
        )
        if not bool((cholesky_info.detach() == 0).all()):
            raise ValueError(
                "noise_covariance_e2 must be Hermitian positive definite"
            )
        whitened = torch.linalg.solve_triangular(
            noise_chol, mixing_otf_e, upper=False
        )

    signal_cov = whitened @ scene @ whitened.conj().transpose(-2, -1)
    # Suppress round-off anti-Hermitian components before Cholesky.
    signal_cov = 0.5 * (signal_cov + signal_cov.conj().transpose(-2, -1))
    eye = torch.eye(
        m_sensor, device=mixing_otf_e.device, dtype=matrix_dtype
    )
    information_matrix = eye + signal_cov / (1.0 + ridge)
    chol = torch.linalg.cholesky(information_matrix)
    log2det = 2.0 * torch.log(chol.diagonal(dim1=-2, dim2=-1).real).sum(-1)
    log2det = log2det / math.log(2.0)
    return factor * log2det


__all__ = [
    "InformationMode",
    "SensorInformationTransfer",
    "build_calibrated_fixed_exposure_transfer",
    "build_fixed_exposure_transfer",
    "build_fixed_signal_transfer",
    "build_global_fixed_signal_transfer",
    "clamp_tiny_negative_roundoff",
    "color_mixing_otf",
    "color_mimo_information_density",
    "diagonal_information_density",
    "pixel_integrate_samples",
    "trapezoid_bin_widths_m",
]
