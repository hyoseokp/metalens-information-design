"""Stable A-optimal risk for an exact 2x2 RGGB polyphase operator.

This module evaluates the linear-Gaussian model

``y = A x + n``

at each coarse Bayer frequency (and optional field/batch item).  ``A`` must
have shape ``[..., 4, 4*K]`` in the alias-major ordering documented by
``engine2.bayer_polyphase``.  The four rows are the R/G/G/B raw phases.  The
scene prior is Hermitian positive semidefinite and the raw-noise covariance is
Hermitian positive definite.

The primary quantity is the A-optimal posterior risk

``tr(Q T P T^H)``

where ``T`` maps the aliased scene state to a requested target (for example,
linear RGB/XYZ) and ``Q`` is a positive-semidefinite error weight in that
target space.  The result exposes both the trace and an explicitly selected
``sum`` or ``mean`` target-channel reduction; it never normalizes by
``trace(Q)`` implicitly.

Numerical stability
-------------------
The implementation whitens the raw noise and solves only the 4x4 innovation
covariance.  It constructs the posterior with the Joseph covariance update,
not by subtracting a nearly equal information reduction from the prior.  This
is important at high SNR, where ``Sigma - Sigma A^H S^-1 A Sigma`` can round
to zero and erase both risk and gradient.  Jitter is disabled by default.  A
requested jitter is interpreted as additional isotropic covariance in the
*noise-whitened* measurement, so the solve and Joseph noise term remain
consistent.

``noise_covariance`` is a Gaussian covariance in electron-squared units.  A
shot-noise approximation such as ``diag(mu_e + read_noise_e**2)`` may be
supplied by the caller, but this function does not evaluate an exact Poisson
likelihood and must not be described as doing so.

For full-fine-grid RGB/XYZ risk, ``T`` must include all four aliases.  Use
:func:`build_alias_block_target_transform` for a frequency-independent color
map (``I_4 kron T_color``), or
:func:`build_alias_dependent_target_transform` when the target color map is
different at the four aliased fine-grid frequencies.  A transform containing
only the baseband alias is not a full-pixel reconstruction objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


TargetChannelReduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class PolyphaseSolverDiagnostics:
    """Value-only diagnostics for the two Cholesky solves.

    Condition numbers are computed from detached Hermitian eigenvalues.  They
    are therefore diagnostics, not part of the optimization graph.
    """

    noise_condition_number: torch.Tensor
    innovation_condition_number: torch.Tensor
    condition_number_limit: float
    whitened_jitter: float
    whitened_signal_power_upper_bound: torch.Tensor
    whitened_signal_power_limit: float


@dataclass(frozen=True)
class PolyphaseChannelDiagnostics:
    """Optional channel diagnostics kept separate from the A-optimal loss.

    ``whitened_singular_values`` are the singular values of
    ``N^{-1/2} A Sigma^{1/2}``, in descending order.  The D-optimal value is
    the unjittered ``log det(I + B B^H)``.
    ``real_gaussian_information_bits_per_coarse_frequency`` applies the
    repository's real-Gaussian factor of one half.  A 2x2 Bayer cell contains
    four raw pixels, so this value must be averaged over the coarse frequency
    quadrature and divided by four before it is reported as bits/raw-pixel.
    It is a diagnostic and never changes the A-optimal objective.
    """

    whitened_singular_values: torch.Tensor
    d_optimal_logdet_nats: torch.Tensor
    real_gaussian_information_bits_per_coarse_frequency: torch.Tensor


@dataclass(frozen=True)
class PolyphaseAOptimalResult:
    """Posterior risk and optional weighted aggregation.

    ``weighted_trace_per_sample`` is exactly ``tr(Q T P T^H)``.
    ``risk_per_sample`` applies ``target_channel_reduction``: ``sum`` leaves
    that trace unchanged, while ``mean`` divides by the number of rows of
    ``T``.  ``objective`` equals ``risk_per_sample`` when no aggregation
    weights are supplied; otherwise it is their normalized weighted mean.
    """

    objective: torch.Tensor
    risk_per_sample: torch.Tensor
    weighted_trace_per_sample: torch.Tensor
    posterior_covariance: torch.Tensor | None
    target_posterior_covariance: torch.Tensor
    target_channels: int
    target_channel_reduction: TargetChannelReduction
    normalized_aggregation_weights: torch.Tensor | None
    solver_diagnostics: PolyphaseSolverDiagnostics
    channel_diagnostics: PolyphaseChannelDiagnostics | None


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise ValueError("forward_operator must use complex64 or complex128")


def _validate_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must be finite")


def _check_runtime_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise RuntimeError(f"{name} became non-finite")


def build_alias_dependent_target_transform(
    alias_target_transforms: torch.Tensor,
) -> torch.Tensor:
    """Build an alias-major block-diagonal full-resolution target map.

    ``alias_target_transforms`` has shape ``[..., 4, C_target, K]``.  Alias
    index follows :data:`engine2.bayer_polyphase.RGGB_ORDERING`, and the
    return has shape ``[..., 4*C_target, 4*K]``.  This explicitly represents
    ``blockdiag(T(omega_0), ..., T(omega_3))``.
    """

    transform = torch.as_tensor(alias_target_transforms)
    if transform.ndim < 3 or transform.shape[-3] != 4:
        raise ValueError(
            "alias_target_transforms must have shape [..., 4, C_target, K]"
        )
    target_channels = int(transform.shape[-2])
    latent_channels = int(transform.shape[-1])
    if target_channels <= 0 or latent_channels <= 0:
        raise ValueError("alias target dimensions must be positive")
    if not (transform.is_floating_point() or transform.is_complex()):
        raise ValueError("alias_target_transforms must be floating or complex")
    _validate_finite(transform, "alias_target_transforms")
    alias_eye = torch.eye(4, device=transform.device, dtype=transform.dtype)
    blocks = torch.einsum("qr,...rck->...qcrk", alias_eye, transform)
    return blocks.reshape(
        *transform.shape[:-3], 4 * target_channels, 4 * latent_channels
    )


def build_alias_block_target_transform(
    color_transform: torch.Tensor,
) -> torch.Tensor:
    """Return ``I_4 kron color_transform`` in the repository alias ordering.

    ``color_transform`` has shape ``[..., C_target, K]`` and the result has
    shape ``[..., 4*C_target, 4*K]``.  The target and state rows are both
    alias-major, then channel-minor.
    """

    transform = torch.as_tensor(color_transform)
    if transform.ndim < 2:
        raise ValueError("color_transform must have shape [..., C_target, K]")
    target_channels = int(transform.shape[-2])
    latent_channels = int(transform.shape[-1])
    if target_channels <= 0 or latent_channels <= 0:
        raise ValueError("color_transform dimensions must be positive")
    if not (transform.is_floating_point() or transform.is_complex()):
        raise ValueError("color_transform must be floating or complex")
    _validate_finite(transform, "color_transform")
    aliases = transform.unsqueeze(-3).expand(
        *transform.shape[:-2], 4, target_channels, latent_channels
    )
    return build_alias_dependent_target_transform(aliases)


def aggregate_real_gaussian_information_bpp(
    bits_per_coarse_frequency: torch.Tensor,
    *,
    aggregation_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate a 2x2-polyphase D diagnostic into bits per raw pixel.

    With no weights all leading samples are averaged.  Otherwise the supplied
    non-negative weights must broadcast exactly to the diagnostic shape and
    are normalized to unit sum.  Division by four converts one coarse Bayer
    cell/frequency vector to one raw-pixel unit.
    """

    bits = torch.as_tensor(bits_per_coarse_frequency)
    if bits.is_complex() or not bits.is_floating_point():
        raise ValueError("bits_per_coarse_frequency must be real floating")
    _validate_finite(bits, "bits_per_coarse_frequency")
    if bits.numel() == 0:
        raise ValueError("bits_per_coarse_frequency must not be empty")
    if aggregation_weights is None:
        return bits.mean() / 4.0
    normalized = _prepare_aggregation_weights(
        aggregation_weights,
        batch_shape=tuple(bits.shape),
        device=bits.device,
        dtype=bits.dtype,
    )
    return (normalized * bits).sum() / 4.0


def _broadcast_matrix(
    value: torch.Tensor,
    *,
    name: str,
    batch_shape: tuple[int, ...],
    matrix_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != matrix_shape:
        raise ValueError(
            f"{name} must have shape [..., {matrix_shape[0]}, "
            f"{matrix_shape[1]}], got {tuple(tensor.shape)}"
        )
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(f"{name} must be floating or complex")
    _validate_finite(tensor, name)
    leading_shape = tuple(tensor.shape[:-2])
    try:
        result_shape = tuple(torch.broadcast_shapes(leading_shape, batch_shape))
    except RuntimeError as exc:
        raise ValueError(
            f"{name} batch shape {leading_shape} cannot broadcast to " f"{batch_shape}"
        ) from exc
    if result_shape != batch_shape:
        raise ValueError(
            f"{name} batch shape {leading_shape} would expand the operator "
            f"batch shape {batch_shape} to {result_shape}"
        )
    tensor = tensor.to(dtype=dtype)
    _validate_finite(tensor, name)
    padding = (1,) * (len(batch_shape) - len(leading_shape))
    return tensor.reshape(*padding, *tensor.shape).expand(*batch_shape, *matrix_shape)


def _hermitian_tolerance(value: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(value.real.dtype)
    detached = value.detach()
    local_scale = torch.maximum(
        detached.abs().amax(dim=(-2, -1), keepdim=True),
        torch.full(
            (*detached.shape[:-2], 1, 1),
            finfo.tiny,
            device=value.device,
            dtype=value.real.dtype,
        ),
    )
    return 128.0 * finfo.eps * local_scale


def _detached_eigvalsh(value: torch.Tensor) -> torch.Tensor:
    """Evaluate validation-only Hermitian eigenvalues without CUDA batching.

    The Windows RTX 5880 stack used by this project has rejected some batched
    ``cusolverDnXsyevBatched`` calls.  Eigenvalues are never part of the loss
    graph, so CUDA inputs are audited on the host and copied back only for
    shape/device-consistent diagnostics.  Cholesky remains the hot-path solve.
    """

    detached = value.detach()
    if detached.device.type == "cuda":
        return torch.linalg.eigvalsh(detached.cpu()).to(device=value.device)
    return torch.linalg.eigvalsh(detached)


def _validate_hermitian_psd(value: torch.Tensor, name: str) -> torch.Tensor:
    tolerance = _hermitian_tolerance(value)
    detached = value.detach()
    adjoint = detached.conj().transpose(-2, -1)
    if not bool(((detached - adjoint).abs() <= tolerance).all()):
        raise ValueError(f"{name} must be Hermitian")
    hermitian = 0.5 * (value + value.conj().transpose(-2, -1))
    eigenvalues = _detached_eigvalsh(hermitian).real
    if not bool((eigenvalues >= -tolerance.squeeze(-1)).all()):
        raise ValueError(f"{name} must be positive semidefinite")
    return hermitian


def _validate_hermitian_pd(value: torch.Tensor, name: str) -> torch.Tensor:
    tolerance = _hermitian_tolerance(value)
    detached = value.detach()
    adjoint = detached.conj().transpose(-2, -1)
    if not bool(((detached - adjoint).abs() <= tolerance).all()):
        raise ValueError(f"{name} must be Hermitian")
    hermitian = 0.5 * (value + value.conj().transpose(-2, -1))
    eigenvalues = _detached_eigvalsh(hermitian).real
    if not bool((eigenvalues > 0).all()):
        raise ValueError(f"{name} must be positive definite")
    return hermitian


def _condition_number_from_cholesky(
    value: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    """Return a detached induced-infinity-norm condition number.

    This avoids the batched CUDA Hermitian-eigensolver that is unsupported on
    some Windows RTX 5880/CUDA combinations.  It is an exact condition number
    in the induced infinity norm, not an estimate of the spectral condition.
    """

    detached_factor = factor.detach()
    inverse = torch.cholesky_inverse(detached_factor)
    matrix_norm = value.detach().abs().sum(dim=-1).amax(dim=-1)
    inverse_norm = inverse.abs().sum(dim=-1).amax(dim=-1)
    return matrix_norm * inverse_norm


def _check_condition_number(condition: torch.Tensor, name: str, limit: float) -> None:
    if not bool(torch.isfinite(condition).all()):
        raise RuntimeError(f"{name} condition number is non-finite")
    maximum = float(condition.amax().cpu())
    if maximum > limit:
        raise RuntimeError(
            f"{name} condition number {maximum:.6g} exceeds fail-closed "
            f"limit {limit:.6g}"
        )


def _checked_cholesky(value: torch.Tensor, name: str) -> torch.Tensor:
    factor, info = torch.linalg.cholesky_ex(value, check_errors=False)
    if bool((info.detach() != 0).any()):
        failing = int((info.detach() != 0).sum().cpu())
        raise RuntimeError(
            f"{name} Cholesky failed for {failing} batch item(s); no fallback "
            "inverse was used"
        )
    if not bool(torch.isfinite(factor.detach()).all()):
        raise RuntimeError(f"{name} Cholesky returned non-finite values")
    return factor


def _prepare_aggregation_weights(
    aggregation_weights: torch.Tensor,
    *,
    batch_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.as_tensor(aggregation_weights, device=device)
    if weight.is_complex() or not weight.is_floating_point():
        raise ValueError("aggregation_weights must be a real floating tensor")
    _validate_finite(weight, "aggregation_weights")
    try:
        result_shape = tuple(torch.broadcast_shapes(tuple(weight.shape), batch_shape))
    except RuntimeError as exc:
        raise ValueError(
            f"aggregation_weights shape {tuple(weight.shape)} cannot broadcast "
            f"to operator batch shape {batch_shape}"
        ) from exc
    if result_shape != batch_shape:
        raise ValueError(
            f"aggregation_weights shape {tuple(weight.shape)} would expand "
            f"operator batch shape {batch_shape} to {result_shape}"
        )
    weight = torch.broadcast_to(weight.to(dtype=dtype), batch_shape)
    _validate_finite(weight, "aggregation_weights")
    if bool((weight.detach() < 0).any()):
        raise ValueError("aggregation_weights must be non-negative")
    maximum = weight.amax()
    if not bool((maximum.detach() > 0)):
        raise ValueError("aggregation_weights must not be all zero")
    scaled = weight / maximum
    total = scaled.sum()
    if not bool(torch.isfinite(total.detach())) or not bool((total.detach() > 0)):
        raise ValueError("aggregation_weights normalization must remain finite")
    normalized = scaled / total
    _validate_finite(normalized, "normalized aggregation_weights")
    return normalized


def rggb_polyphase_a_optimal_risk(
    forward_operator: torch.Tensor,
    prior_covariance: torch.Tensor,
    noise_covariance: torch.Tensor,
    *,
    target_transform: torch.Tensor | None = None,
    target_weight: torch.Tensor | None = None,
    target_channel_reduction: TargetChannelReduction = "mean",
    aggregation_weights: torch.Tensor | None = None,
    jitter: float = 0.0,
    condition_number_limit: float | None = None,
    maximum_whitened_signal_power: float | None = None,
    return_channel_diagnostics: bool = False,
    prior_covariance_prevalidated: bool = False,
    noise_covariance_prevalidated: bool = False,
    return_full_posterior: bool = True,
) -> PolyphaseAOptimalResult:
    """Evaluate exact periodic-RGGB A-optimal posterior risk.

    Parameters
    ----------
    forward_operator:
        Complex tensor ``[..., 4, 4*K]``.  Rows are R/G/G/B Bayer phases and
        columns are alias-major then latent-component-minor.
    prior_covariance:
        Hermitian PSD Gaussian scene covariance ``[..., 4*K, 4*K]``.
    noise_covariance:
        Hermitian PD Gaussian raw covariance ``[..., 4, 4]``.  This can be a
        Gaussian shot/read covariance in electron-squared units; it is not an
        exact Poisson noise model.
    target_transform:
        Optional complex/real ``[..., C_target, 4*K]`` map from the aliased
        scene state to the target.  The identity is used when omitted.
    target_weight:
        Optional Hermitian PSD ``[..., C_target, C_target]`` error weight.
        It acts *after* ``target_transform``.  The identity is used when
        omitted.  Leading axes may carry frequency-wise weights.
    target_channel_reduction:
        ``"sum"`` returns the A-optimal trace.  ``"mean"`` divides that trace
        by exactly ``C_target``.  Neither mode normalizes by ``trace(Q)``.
    aggregation_weights:
        Optional non-negative weights broadcastable exactly to the operator's
        leading shape.  They may combine field and frequency quadrature (for
        example ``[field, 1, 1] * [1, fy, fx]``).  They are normalized to unit
        sum.  All-zero weights fail rather than silently producing zero.
    jitter:
        Non-negative isotropic covariance added in whitened raw space.  The
        default is exactly zero.  Nonzero jitter changes the stated Gaussian
        model and is reported in solver diagnostics.
    condition_number_limit:
        Fail-closed induced-infinity-norm condition limit for the noise and
        innovation solves.  ``None`` uses ``1 / (16*eps)`` for the operator
        precision.
    maximum_whitened_signal_power:
        Fail-closed limit on the maximum absolute row sum of
        ``N^-1/2 A Sigma A^H N^-H/2``.  This guards absolute dynamic range,
        which a condition number cannot detect.  ``None`` uses
        ``sqrt(finfo.max)`` for the real precision.  Primary scientific runs
        should record the selected limit and observed bound.
    return_channel_diagnostics:
        If true, separately return D-optimal log-determinant/information and
        whitened singular values.  These never enter the primary loss.
    prior_covariance_prevalidated:
        Skip the detached batched PSD eigendecomposition for a frozen prior
        that was already validated by this module in the same process.  This
        exists for optimization hot loops; callers remain responsible for a
        fail-closed validation before setting it.
    noise_covariance_prevalidated:
        Skip repeated PD validation when the caller has just constructed the
        covariance from already validated positive phase variances.  This is a
        hot-loop optimization contract, not a relaxation of the public default.
    return_full_posterior:
        If false, form the exact target-space Joseph posterior directly and
        omit the much larger state posterior.  The primary objective and its
        gradient are unchanged; ``posterior_covariance`` is then ``None``.

    Returns
    -------
    PolyphaseAOptimalResult
        The differentiable risk, Joseph posterior covariance, strict solver
        diagnostics, and optional channel diagnostics.
    """

    a = torch.as_tensor(forward_operator)
    if a.ndim < 2 or a.shape[-2] != 4:
        raise ValueError(
            "forward_operator must have shape [..., 4, 4*K] with four raw rows"
        )
    if not a.is_complex():
        raise ValueError("forward_operator must be complex")
    real_dtype = _real_dtype(a.dtype)
    _validate_finite(a, "forward_operator")
    n_state = int(a.shape[-1])
    if n_state <= 0 or n_state % 4:
        raise ValueError(
            "forward_operator state dimension must be positive and divisible "
            "by four (four aliases times K)"
        )
    batch_shape = tuple(a.shape[:-2])

    sigma = _broadcast_matrix(
        prior_covariance,
        name="prior_covariance",
        batch_shape=batch_shape,
        matrix_shape=(n_state, n_state),
        device=a.device,
        dtype=a.dtype,
    )
    if prior_covariance_prevalidated:
        if sigma.requires_grad:
            raise ValueError("a prevalidated prior covariance must be frozen")
    else:
        sigma = _validate_hermitian_psd(sigma, "prior_covariance")
    noise = _broadcast_matrix(
        noise_covariance,
        name="noise_covariance",
        batch_shape=batch_shape,
        matrix_shape=(4, 4),
        device=a.device,
        dtype=a.dtype,
    )
    if noise_covariance_prevalidated:
        # Design-dependent shot noise may retain a gradient.  The caller has
        # already checked the underlying real phase variances for finiteness
        # and strict positivity before forming this diagonal covariance.
        pass
    else:
        noise = _validate_hermitian_pd(noise, "noise_covariance")

    if target_transform is None:
        target_channels = n_state
        transform = torch.eye(n_state, device=a.device, dtype=a.dtype)
        transform = transform.reshape((1,) * len(batch_shape) + transform.shape)
        transform = transform.expand(*batch_shape, n_state, n_state)
    else:
        raw_transform = torch.as_tensor(target_transform, device=a.device)
        if raw_transform.ndim < 2 or raw_transform.shape[-1] != n_state:
            raise ValueError("target_transform must have shape [..., C_target, 4*K]")
        target_channels = int(raw_transform.shape[-2])
        if target_channels <= 0:
            raise ValueError("target_transform must have at least one target row")
        transform = _broadcast_matrix(
            raw_transform,
            name="target_transform",
            batch_shape=batch_shape,
            matrix_shape=(target_channels, n_state),
            device=a.device,
            dtype=a.dtype,
        )

    if target_weight is None:
        q = torch.eye(target_channels, device=a.device, dtype=a.dtype)
        q = q.reshape((1,) * len(batch_shape) + q.shape)
        q = q.expand(*batch_shape, target_channels, target_channels)
    else:
        q = _broadcast_matrix(
            target_weight,
            name="target_weight",
            batch_shape=batch_shape,
            matrix_shape=(target_channels, target_channels),
            device=a.device,
            dtype=a.dtype,
        )
        q = _validate_hermitian_psd(q, "target_weight")

    if target_channel_reduction not in ("sum", "mean"):
        raise ValueError("target_channel_reduction must be 'sum' or 'mean'")

    normalized_weights: torch.Tensor | None = None
    if aggregation_weights is not None:
        normalized_weights = _prepare_aggregation_weights(
            aggregation_weights,
            batch_shape=batch_shape,
            device=a.device,
            dtype=real_dtype,
        )

    # Reject the trivial objective T^H Q T == 0 while retaining legitimate
    # rank-deficient, nonzero target weights and zero-weighted aperture bins.
    effective_target = (transform.conj().transpose(-2, -1) @ q @ transform).detach()
    _check_runtime_finite(effective_target, "effective target weight")
    effective_energy = effective_target.abs().amax(dim=(-2, -1))
    if normalized_weights is None:
        effective_energy_total = effective_energy.sum()
    else:
        effective_energy_total = (normalized_weights * effective_energy).sum()
    if not bool(effective_energy_total > 0):
        raise ValueError(
            "target_transform and target_weight define an all-zero effective target"
        )

    whitened_jitter = float(jitter)
    if not math.isfinite(whitened_jitter) or whitened_jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    if condition_number_limit is None:
        condition_limit = 1.0 / (16.0 * torch.finfo(real_dtype).eps)
    else:
        condition_limit = float(condition_number_limit)
        if not math.isfinite(condition_limit) or condition_limit < 1.0:
            raise ValueError("condition_number_limit must be finite and at least one")

    hard_signal_power_limit = math.sqrt(torch.finfo(real_dtype).max)
    if maximum_whitened_signal_power is None:
        signal_power_limit = hard_signal_power_limit
    else:
        signal_power_limit = float(maximum_whitened_signal_power)
        if not math.isfinite(signal_power_limit) or signal_power_limit <= 0.0:
            raise ValueError(
                "maximum_whitened_signal_power must be finite and positive"
            )
        if signal_power_limit > hard_signal_power_limit:
            raise ValueError(
                "maximum_whitened_signal_power must not exceed the dtype "
                f"hard limit {hard_signal_power_limit:.6g}"
            )

    noise_cholesky = _checked_cholesky(noise, "noise_covariance")
    noise_condition = _condition_number_from_cholesky(noise, noise_cholesky)
    _check_condition_number(noise_condition, "noise_covariance", condition_limit)

    # L_n^{-1} A makes the measurement noise exactly identity.  Every solve
    # below is therefore a four-row operation independent of K.
    whitened_a = torch.linalg.solve_triangular(noise_cholesky, a, upper=False)
    _check_runtime_finite(whitened_a, "noise-whitened forward operator")
    whitened_signal = whitened_a @ sigma @ whitened_a.conj().transpose(-2, -1)
    whitened_signal = 0.5 * (whitened_signal + whitened_signal.conj().transpose(-2, -1))
    _validate_finite(whitened_signal, "whitened signal covariance")
    signal_power_upper_bound = whitened_signal.detach().abs().sum(dim=-1).amax(dim=-1)
    maximum_signal_power = float(signal_power_upper_bound.amax().cpu())
    if maximum_signal_power > signal_power_limit:
        raise RuntimeError(
            "whitened signal power upper bound "
            f"{maximum_signal_power:.6g} exceeds fail-closed limit "
            f"{signal_power_limit:.6g}"
        )
    eye4 = torch.eye(4, device=a.device, dtype=a.dtype)
    eye4 = eye4.reshape((1,) * len(batch_shape) + (4, 4))
    innovation = eye4 + whitened_signal
    if whitened_jitter:
        innovation = innovation + whitened_jitter * eye4
    innovation = 0.5 * (innovation + innovation.conj().transpose(-2, -1))

    _validate_finite(innovation, "whitened innovation")
    innovation_cholesky = _checked_cholesky(innovation, "whitened innovation")
    innovation_condition = _condition_number_from_cholesky(
        innovation, innovation_cholesky
    )
    _check_condition_number(
        innovation_condition, "whitened innovation", condition_limit
    )

    # K = Sigma A_w^H S^-1, evaluated through the 4x4 Cholesky solve.
    cross_covariance = sigma @ whitened_a.conj().transpose(-2, -1)
    _check_runtime_finite(cross_covariance, "state-measurement cross covariance")
    kalman_gain = (
        torch.cholesky_solve(
            cross_covariance.conj().transpose(-2, -1), innovation_cholesky
        )
        .conj()
        .transpose(-2, -1)
    )
    _check_runtime_finite(kalman_gain, "Kalman gain")

    # Joseph form is algebraically equal to the Bayesian posterior at zero
    # jitter, but remains a sum of PSD terms at high SNR.  The target-only path
    # applies T before constructing the covariance, avoiding an O(n_state^2)
    # state-posterior allocation while remaining exactly the same algebra:
    #   T_R = T - (T K) A_w
    #   P_T = T_R Sigma T_R^H + (1+j) (T K)(T K)^H.
    posterior: torch.Tensor | None
    if return_full_posterior:
        state_eye = torch.eye(n_state, device=a.device, dtype=a.dtype)
        state_eye = state_eye.reshape((1,) * len(batch_shape) + (n_state, n_state))
        residual = state_eye - kalman_gain @ whitened_a
        posterior = residual @ sigma @ residual.conj().transpose(-2, -1)
        posterior = posterior + (1.0 + whitened_jitter) * (
            kalman_gain @ kalman_gain.conj().transpose(-2, -1)
        )
        posterior = 0.5 * (posterior + posterior.conj().transpose(-2, -1))
        _check_runtime_finite(posterior, "Joseph posterior covariance")
        target_posterior = (
            transform @ posterior @ transform.conj().transpose(-2, -1)
        )
    else:
        posterior = None
        target_gain = transform @ kalman_gain
        target_residual = transform - target_gain @ whitened_a
        target_posterior = (
            target_residual @ sigma @ target_residual.conj().transpose(-2, -1)
        )
        target_posterior = target_posterior + (1.0 + whitened_jitter) * (
            target_gain @ target_gain.conj().transpose(-2, -1)
        )
    target_posterior = 0.5 * (
        target_posterior + target_posterior.conj().transpose(-2, -1)
    )
    _check_runtime_finite(target_posterior, "target posterior covariance")
    weighted_diagonal = torch.diagonal(q @ target_posterior, dim1=-2, dim2=-1)
    weighted_trace = weighted_diagonal.sum(dim=-1).real
    _check_runtime_finite(weighted_trace, "weighted posterior trace")
    risk_scale = weighted_diagonal.detach().abs().sum(dim=-1)
    risk_tolerance = (
        512.0
        * torch.finfo(real_dtype).eps
        * torch.maximum(
            risk_scale,
            torch.full_like(risk_scale, torch.finfo(real_dtype).tiny),
        )
    )
    if bool((weighted_trace.detach() < -risk_tolerance).any()):
        raise RuntimeError("Joseph posterior produced a negative weighted risk")
    # Clamp only accepted round-off below zero.  The Joseph evaluation keeps
    # genuine high-SNR risks positive, unlike a prior-minus-reduction formula.
    weighted_trace = weighted_trace.clamp_min(0.0)
    if target_channel_reduction == "mean":
        risk_per_sample = weighted_trace / float(target_channels)
    else:
        risk_per_sample = weighted_trace

    if aggregation_weights is None:
        objective = risk_per_sample
    else:
        assert normalized_weights is not None
        objective = (normalized_weights * risk_per_sample).sum()
    _check_runtime_finite(objective, "A-optimal objective")

    channel_diagnostics: PolyphaseChannelDiagnostics | None = None
    if return_channel_diagnostics:
        signal_eigenvalues = _detached_eigvalsh(whitened_signal).real
        signal_tolerance = (
            256.0
            * torch.finfo(real_dtype).eps
            * torch.maximum(
                signal_eigenvalues.detach().abs().amax(dim=-1, keepdim=True),
                torch.full_like(
                    signal_eigenvalues[..., :1].real,
                    torch.finfo(real_dtype).tiny,
                ),
            )
        )
        if bool((signal_eigenvalues.detach() < -signal_tolerance).any()):
            raise RuntimeError("whitened signal covariance is materially indefinite")
        signal_eigenvalues = signal_eigenvalues.clamp_min(0.0)
        singular_values = signal_eigenvalues.sqrt().flip(-1)
        logdet_nats = torch.log1p(signal_eigenvalues).sum(dim=-1)
        channel_diagnostics = PolyphaseChannelDiagnostics(
            whitened_singular_values=singular_values,
            d_optimal_logdet_nats=logdet_nats,
            real_gaussian_information_bits_per_coarse_frequency=(
                0.5 * logdet_nats / math.log(2.0)
            ),
        )

    return PolyphaseAOptimalResult(
        objective=objective,
        risk_per_sample=risk_per_sample,
        weighted_trace_per_sample=weighted_trace,
        posterior_covariance=posterior,
        target_posterior_covariance=target_posterior,
        target_channels=target_channels,
        target_channel_reduction=target_channel_reduction,
        normalized_aggregation_weights=normalized_weights,
        solver_diagnostics=PolyphaseSolverDiagnostics(
            noise_condition_number=noise_condition,
            innovation_condition_number=innovation_condition,
            condition_number_limit=condition_limit,
            whitened_jitter=whitened_jitter,
            whitened_signal_power_upper_bound=signal_power_upper_bound,
            whitened_signal_power_limit=signal_power_limit,
        ),
        channel_diagnostics=channel_diagnostics,
    )


__all__ = [
    "aggregate_real_gaussian_information_bpp",
    "build_alias_block_target_transform",
    "build_alias_dependent_target_transform",
    "PolyphaseAOptimalResult",
    "PolyphaseChannelDiagnostics",
    "PolyphaseSolverDiagnostics",
    "TargetChannelReduction",
    "rggb_polyphase_a_optimal_risk",
]
