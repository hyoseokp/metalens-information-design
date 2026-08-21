"""Projected optimization primitives for physical metalens widths.

The optimization variable in this module is the physical width tensor itself.
There is deliberately no sigmoid/tanh latent variable: box feasibility is
enforced by an exact projection after every optimizer update.

The optional response-manifold metric implemented here is a safeguarded,
diagonal *preconditioner*.  It is not an exact natural gradient (nor a claim
that the full optical objective has been locally quadraticized exactly).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


__all__ = [
    "KKTDiagnostics",
    "ProjectedAdam",
    "diagonal_response_preconditioner",
    "project_box_",
    "projected_gradient_mapping",
    "projected_kkt_residual",
    "response_neighbor_smoothness",
]


def _is_finite(value: Tensor) -> bool:
    if value.is_complex():
        return bool(
            torch.isfinite(value.real).all() and torch.isfinite(value.imag).all()
        )
    return bool(torch.isfinite(value).all())


def _require_finite(name: str, value: Tensor) -> None:
    if not _is_finite(value):
        raise FloatingPointError(f"{name} contains NaN or infinite values")


def _is_finite_real_number(value: object) -> bool:
    """Return true for finite Python real scalars, explicitly excluding bool."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _reject_boolean_numeric_input(name: str, value: object) -> None:
    """Reject booleans before a numeric input can be cast to floating point.

    ``torch.as_tensor(..., dtype=float)`` silently maps both Python booleans
    and ``torch.bool`` tensors to zero/one.  Numeric public APIs in this module
    must not make that semantic conversion.  Containers are inspected
    recursively so a mixed input such as ``[0.5, True]`` cannot evade the
    check by inferring a floating dtype before broadcast.
    """

    seen: set[int] = set()

    def visit(candidate: object) -> None:
        if isinstance(candidate, bool):
            raise TypeError(f"{name} must not contain boolean values")
        if isinstance(candidate, Tensor):
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
            return

        # This catches scalar/array boolean types from array libraries without
        # importing an optional dependency.  Conversion is inference-only: the
        # real dtype/device conversion still happens in the caller after this
        # boolean gate.
        try:
            inferred = torch.as_tensor(candidate)
        except (RuntimeError, TypeError, ValueError):
            return
        if inferred.dtype == torch.bool:
            raise TypeError(f"{name} must not contain boolean values")

    visit(value)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def _stable_rms(value: Tensor) -> Tensor:
    """Compute RMS without squaring values at their original scale."""

    if value.numel() == 0:
        raise ValueError("RMS input must be non-empty")
    magnitude = value.abs()
    scale = magnitude.amax()
    if bool(scale == 0):
        return torch.zeros((), dtype=_real_dtype(value.dtype), device=value.device)
    return scale * torch.sqrt(torch.mean((magnitude / scale).square()))


def _materialize_bound(name: str, value: float | Tensor, reference: Tensor) -> Tensor:
    _reject_boolean_numeric_input(name, value)
    if isinstance(value, Tensor) and value.is_complex():
        raise TypeError(f"{name} must be real")
    try:
        bound = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        bound = torch.broadcast_to(bound, reference.shape).clone().detach()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} is not broadcastable to width shape {tuple(reference.shape)}"
        ) from exc
    _require_finite(name, bound)
    return bound


def _validated_bounds(
    widths: Tensor,
    lower: float | Tensor,
    upper: float | Tensor,
) -> tuple[Tensor, Tensor]:
    if widths.is_complex() or not widths.is_floating_point():
        raise TypeError("physical widths must have a real floating-point dtype")
    if widths.numel() == 0:
        raise ValueError("physical width tensor must be non-empty")
    _require_finite("widths", widths)
    lo = _materialize_bound("lower bound", lower, widths)
    hi = _materialize_bound("upper bound", upper, widths)
    if not bool(torch.all(lo < hi)):
        raise ValueError(
            "every lower bound must be strictly smaller than its upper bound"
        )
    return lo, hi


@torch.no_grad()
def project_box_(
    widths: Tensor, lower: float | Tensor, upper: float | Tensor
) -> Tensor:
    """Project ``widths`` exactly onto elementwise closed box bounds in-place."""

    lo, hi = _validated_bounds(widths, lower, upper)
    widths.copy_(torch.maximum(torch.minimum(widths, hi), lo))
    _require_finite("projected widths", widths)
    return widths


@dataclass(frozen=True)
class KKTDiagnostics:
    """First-order box-constrained minimization diagnostics.

    ``residual`` is signed: it is ``g`` in the interior, ``min(g, 0)`` at a
    lower bound, and ``max(g, 0)`` at an upper bound.  Thus a lower-bound
    gradient is KKT-compatible when ``g >= 0`` and an upper-bound gradient is
    compatible when ``g <= 0``.
    """

    residual: Tensor
    infinity_norm: Tensor
    rms: Tensor
    sign_violation_fraction: Tensor
    active_sign_violation_fraction: Tensor
    active_fraction: Tensor
    lower_active_fraction: Tensor
    upper_active_fraction: Tensor
    interior_fraction: Tensor

    def scalar_dict(self) -> dict[str, float]:
        """Return detached JSON-friendly scalar diagnostics."""

        return {
            "kkt_infinity_norm": float(self.infinity_norm.detach().cpu()),
            "kkt_rms": float(self.rms.detach().cpu()),
            "sign_violation_fraction": float(
                self.sign_violation_fraction.detach().cpu()
            ),
            "active_sign_violation_fraction": float(
                self.active_sign_violation_fraction.detach().cpu()
            ),
            "active_fraction": float(self.active_fraction.detach().cpu()),
            "lower_active_fraction": float(self.lower_active_fraction.detach().cpu()),
            "upper_active_fraction": float(self.upper_active_fraction.detach().cpu()),
            "interior_fraction": float(self.interior_fraction.detach().cpu()),
        }


def projected_kkt_residual(
    widths: Tensor,
    gradient: Tensor,
    lower: float | Tensor,
    upper: float | Tensor,
    *,
    active_tolerance: float = 0.0,
) -> KKTDiagnostics:
    """Compute the normal-cone KKT residual for box minimization.

    Widths must be feasible up to ``active_tolerance``.  The default zero
    tolerance is appropriate after :func:`project_box_` or
    :class:`ProjectedAdam`, both of which put active coordinates exactly on a
    bound.  This active-set residual is independent of a step size and must not
    be confused with :func:`projected_gradient_mapping`, whose value can depend
    on the positive projection step ``eta`` away from a stationary point.
    """

    _reject_boolean_numeric_input("active_tolerance", active_tolerance)
    if not _is_finite_real_number(active_tolerance):
        raise ValueError("active_tolerance must be a finite non-negative scalar")
    if active_tolerance < 0:
        raise ValueError("active_tolerance must be non-negative")
    lo, hi = _validated_bounds(widths, lower, upper)
    if gradient.shape != widths.shape:
        raise ValueError("gradient shape must exactly match the physical width shape")
    if gradient.device != widths.device or gradient.dtype != widths.dtype:
        raise TypeError("gradient dtype and device must match physical widths")
    _require_finite("gradient", gradient)

    tol = torch.as_tensor(active_tolerance, dtype=widths.dtype, device=widths.device)
    if bool(torch.any(2 * tol >= (hi - lo))):
        raise ValueError(
            "active_tolerance must be less than half of every bound interval"
        )
    if bool(torch.any(widths < lo - tol) or torch.any(widths > hi + tol)):
        raise ValueError("KKT diagnostics require feasible physical widths")

    lower_active = widths <= lo + tol
    upper_active = widths >= hi - tol
    interior = ~(lower_active | upper_active)
    residual = torch.where(
        lower_active,
        torch.minimum(gradient, torch.zeros_like(gradient)),
        torch.where(
            upper_active, torch.maximum(gradient, torch.zeros_like(gradient)), gradient
        ),
    )
    lower_bad = lower_active & (gradient < 0)
    upper_bad = upper_active & (gradient > 0)
    sign_bad = lower_bad | upper_bad
    active = lower_active | upper_active

    real_dtype = _real_dtype(widths.dtype)
    total = widths.numel()
    total_t = torch.as_tensor(total, dtype=real_dtype, device=widths.device)
    active_count = active.sum().to(real_dtype)

    def fraction(mask: Tensor) -> Tensor:
        return mask.sum().to(real_dtype) / total_t

    active_bad_fraction = torch.where(
        active_count > 0,
        sign_bad.sum().to(real_dtype) / active_count.clamp_min(1),
        torch.zeros((), dtype=real_dtype, device=widths.device),
    )
    infinity_norm = residual.abs().amax()
    rms = _stable_rms(residual)
    for name, value in (
        ("KKT residual", residual),
        ("KKT infinity norm", infinity_norm),
        ("KKT RMS", rms),
        ("active sign-violation fraction", active_bad_fraction),
    ):
        _require_finite(name, value)
    return KKTDiagnostics(
        residual=residual,
        infinity_norm=infinity_norm,
        rms=rms,
        sign_violation_fraction=fraction(sign_bad),
        active_sign_violation_fraction=active_bad_fraction,
        active_fraction=fraction(active),
        lower_active_fraction=fraction(lower_active),
        upper_active_fraction=fraction(upper_active),
        interior_fraction=fraction(interior),
    )


def projected_gradient_mapping(
    widths: Tensor,
    gradient: Tensor,
    lower: float | Tensor,
    upper: float | Tensor,
    eta: float | Tensor,
) -> Tensor:
    """Return the exact box projected-gradient mapping for positive ``eta``.

    The returned tensor is

    ``(widths - project_box(widths - eta * gradient)) / eta``.

    Unlike :func:`projected_kkt_residual`, this mapping retains the requested
    finite positive projection step and therefore remains correct when a trial
    step crosses the opposite bound.  ``eta`` may be scalar or elementwise
    broadcastable to the width shape.  Widths must already be feasible.
    """

    lo, hi = _validated_bounds(widths, lower, upper)
    if gradient.shape != widths.shape:
        raise ValueError("gradient shape must exactly match the physical width shape")
    if gradient.device != widths.device or gradient.dtype != widths.dtype:
        raise TypeError("gradient dtype and device must match physical widths")
    _require_finite("gradient", gradient)
    if bool(torch.any(widths < lo) or torch.any(widths > hi)):
        raise ValueError("projected-gradient mapping requires feasible physical widths")
    _reject_boolean_numeric_input("eta", eta)
    if isinstance(eta, Tensor) and eta.is_complex():
        raise TypeError("eta must be real")
    try:
        step = torch.as_tensor(eta, dtype=widths.dtype, device=widths.device)
        step = torch.broadcast_to(step, widths.shape)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "eta is not broadcastable to the physical width shape"
        ) from exc
    _require_finite("eta", step)
    if bool(torch.any(step <= 0)):
        raise ValueError("eta must be strictly positive")

    scaled_gradient = step * gradient
    _require_finite("eta-scaled gradient", scaled_gradient)
    trial = widths - scaled_gradient
    _require_finite("projected-gradient trial widths", trial)
    projected = torch.maximum(torch.minimum(trial, hi), lo)
    mapping = (widths - projected) / step
    _require_finite("projected-gradient mapping", mapping)
    return mapping


def diagonal_response_preconditioner(
    transmission: Tensor,
    dtransmission_dwidth: Tensor,
    beta: float | Tensor | None = None,
    *,
    response_ndim: int = 1,
    mode: str = "stable_log",
    amplitude_floor: float = 1.0e-6,
    epsilon: float = 1.0e-12,
    minimum: float = 1.0e-6,
    maximum: float = 1.0e6,
) -> Tensor:
    """Build a safeguarded diagonal response-manifold preconditioner.

    In ``stable_log`` mode the metric is

    ``sum(beta * |dt/dW|^2 / (|t|^2 + amplitude_floor^2))``.

    This is a stable analogue of ``sum(beta * |d log(t)/dW|^2)``.  In
    ``complex_derivative`` mode the denominator is omitted.  The final metric
    is epsilon-shifted and clamped.  It is only a diagonal preconditioner, not
    an exact natural gradient.

    The leading dimensions index independent widths; the final
    ``response_ndim`` dimensions (for example wavelength and field) are
    reduced.
    """

    if beta is not None:
        _reject_boolean_numeric_input("beta", beta)
    if transmission.shape != dtransmission_dwidth.shape:
        raise ValueError("transmission and derivative shapes must match exactly")
    if transmission.device != dtransmission_dwidth.device:
        raise TypeError("transmission and derivative devices must match")
    if transmission.dtype != dtransmission_dwidth.dtype:
        raise TypeError("transmission and derivative dtypes must match")
    if not transmission.is_floating_point() and not transmission.is_complex():
        raise TypeError("transmission must have floating or complex dtype")
    if (
        type(response_ndim) is not int
        or response_ndim < 0
        or response_ndim > transmission.ndim
    ):
        raise ValueError(
            "response_ndim must be an integer between zero and transmission.ndim"
        )
    if mode not in {"stable_log", "complex_derivative"}:
        raise ValueError("mode must be 'stable_log' or 'complex_derivative'")
    for name, value, allow_zero in (
        ("amplitude_floor", amplitude_floor, False),
        ("epsilon", epsilon, True),
        ("minimum", minimum, False),
        ("maximum", maximum, False),
    ):
        _reject_boolean_numeric_input(name, value)
        if not _is_finite_real_number(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(
                f"{name} must be {'non-negative' if allow_zero else 'positive'}"
            )
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    _require_finite("transmission", transmission)
    _require_finite("dtransmission_dwidth", dtransmission_dwidth)

    real_dtype = _real_dtype(transmission.dtype)
    derivative_energy = dtransmission_dwidth.abs().square()
    _require_finite("response derivative energy", derivative_energy)
    if mode == "stable_log":
        amplitude_floor_tensor = torch.as_tensor(
            amplitude_floor, dtype=real_dtype, device=transmission.device
        )
        amplitude_floor_energy = amplitude_floor_tensor.square()
        _require_finite("amplitude-floor energy", amplitude_floor_energy)
        denominator = transmission.abs().square() + amplitude_floor_energy
        _require_finite("stable-log denominator", denominator)
        contributions = derivative_energy / denominator
    else:
        contributions = derivative_energy
    _require_finite("response metric contributions", contributions)

    if beta is not None:
        if isinstance(beta, Tensor) and beta.is_complex():
            raise TypeError("beta weights must be real")
        weight = torch.as_tensor(beta, dtype=real_dtype, device=transmission.device)
        try:
            weight = torch.broadcast_to(weight, transmission.shape)
        except RuntimeError as exc:
            raise ValueError("beta is not broadcastable to the response shape") from exc
        _require_finite("beta", weight)
        if bool(torch.any(weight < 0)):
            raise ValueError("beta weights must be non-negative")
        contributions = contributions * weight
        _require_finite("weighted response metric contributions", contributions)

    if response_ndim:
        dims = tuple(range(transmission.ndim - response_ndim, transmission.ndim))
        metric = contributions.sum(dim=dims)
    else:
        metric = contributions
    _require_finite("reduced response metric", metric)
    metric = metric + torch.as_tensor(epsilon, dtype=real_dtype, device=metric.device)
    _require_finite("epsilon-shifted response metric", metric)
    metric = metric.clamp(min=minimum, max=maximum)
    _require_finite("response preconditioner", metric)
    return metric


def response_neighbor_smoothness(
    transmission: Tensor,
    edges: Tensor | Sequence[Sequence[int]],
    gamma: float | Tensor | None = None,
) -> Tensor:
    """Return ``sum_edges,response gamma * |t_i - t_j|^2``.

    ``transmission`` has shape ``(num_sites, ...)``.  Regularizing complex
    response, rather than raw width, naturally respects phase wraps: two very
    different widths with the same complex transmission incur zero cost.
    ``gamma`` may be scalar or broadcastable to ``(num_edges, ...)``.
    """

    if gamma is not None:
        _reject_boolean_numeric_input("gamma", gamma)
    _reject_boolean_numeric_input("edges", edges)
    if transmission.ndim < 1:
        raise ValueError("transmission must have a leading site dimension")
    if not transmission.is_floating_point() and not transmission.is_complex():
        raise TypeError("transmission must have floating or complex dtype")
    _require_finite("transmission", transmission)
    edge_tensor = torch.as_tensor(edges, device=transmission.device)
    if edge_tensor.numel() == 0 and edge_tensor.ndim == 1 and edge_tensor.shape[0] == 0:
        edge_tensor = torch.empty((0, 2), dtype=torch.int64, device=transmission.device)
    else:
        if edge_tensor.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("edges must contain integer indices")
        edge_tensor = edge_tensor.to(torch.int64)
    if edge_tensor.ndim != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("edges must have shape (num_edges, 2)")
    if edge_tensor.numel() and (
        bool(torch.any(edge_tensor < 0))
        or bool(torch.any(edge_tensor >= transmission.shape[0]))
    ):
        raise ValueError("edge index is outside the transmission site dimension")
    if edge_tensor.shape[0] == 0:
        return transmission.real.sum() * 0.0

    delta = transmission.index_select(0, edge_tensor[:, 0]) - transmission.index_select(
        0, edge_tensor[:, 1]
    )
    energy = delta.abs().square()
    if gamma is not None:
        real_dtype = _real_dtype(transmission.dtype)
        if isinstance(gamma, Tensor) and gamma.is_complex():
            raise TypeError("gamma weights must be real")
        weight = torch.as_tensor(gamma, dtype=real_dtype, device=transmission.device)
        try:
            weight = torch.broadcast_to(weight, energy.shape)
        except RuntimeError as exc:
            raise ValueError(
                "gamma is not broadcastable to edge-response shape"
            ) from exc
        _require_finite("gamma", weight)
        if bool(torch.any(weight < 0)):
            raise ValueError("gamma weights must be non-negative")
        energy = energy * weight
    result = energy.sum()
    _require_finite("response smoothness", result)
    return result


class ProjectedAdam(torch.optim.Optimizer):
    """Adam for one physical-width parameter with exact box projection.

    A supplied diagonal ``preconditioner`` scales the bias-corrected Adam
    direction before projection.  It must be finite and non-negative and is
    clamped to configured safeguards.  This should be interpreted as a useful
    diagonal response-space preconditioner, not as an exact natural gradient.

    By default, moments are reset only where an update tries to leave the box.
    This prevents stale outward Adam momentum from pinning a coordinate after
    its gradient turns inward.  A per-coordinate ``moment_age`` supplies the
    corresponding bias correction, so a reset coordinate's first inward step
    is exactly a fresh Adam step even when the global checkpoint step is old.
    """

    FORMAT_VERSION = 2
    _GROUP_KEYS = frozenset(
        {
            "lr",
            "betas",
            "eps",
            "lower_bound",
            "upper_bound",
            "preconditioner_min",
            "preconditioner_max",
            "reset_outward_moments",
            "format_version",
            "params",
        }
    )
    _STATE_KEYS = frozenset(
        {
            "step",
            "moment_age",
            "exp_avg",
            "exp_avg_sq",
            "last_projection_max",
            "last_preconditioner_min",
            "last_preconditioner_max",
            "last_preconditioner_supplied",
            "last_outward_fraction",
        }
    )
    _SCALAR_STATE_KEYS = (
        "last_projection_max",
        "last_preconditioner_min",
        "last_preconditioner_max",
        "last_outward_fraction",
    )

    def __init__(
        self,
        widths: nn.Parameter,
        lower: float | Tensor,
        upper: float | Tensor,
        *,
        lr: float = 1.0e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        preconditioner_min: float = 1.0e-6,
        preconditioner_max: float = 1.0e6,
        reset_outward_moments: bool = True,
    ) -> None:
        if not isinstance(widths, nn.Parameter):
            raise TypeError(
                "widths must be a torch.nn.Parameter holding physical widths"
            )
        if not widths.is_leaf or not widths.requires_grad:
            raise ValueError("widths must be a leaf Parameter with requires_grad=True")
        lo, hi = _validated_bounds(widths.detach(), lower, upper)
        if bool(torch.any(widths.detach() < lo) or torch.any(widths.detach() > hi)):
            raise ValueError(
                "initial physical widths must lie inside the requested bounds"
            )
        if not _is_finite_real_number(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or any(not _is_finite_real_number(b) for b in betas)
            or not (0 <= betas[0] < 1)
            or not (0 <= betas[1] < 1)
        ):
            raise ValueError("betas must be a two-tuple with each value in [0, 1)")
        for name, value in (
            ("eps", eps),
            ("preconditioner_min", preconditioner_min),
            ("preconditioner_max", preconditioner_max),
        ):
            if not _is_finite_real_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if preconditioner_min > preconditioner_max:
            raise ValueError("preconditioner_min cannot exceed preconditioner_max")
        if not preconditioner_min <= 1.0 <= preconditioner_max:
            raise ValueError(
                "preconditioner safeguards must contain the identity metric 1"
            )
        if not isinstance(reset_outward_moments, bool):
            raise TypeError("reset_outward_moments must be boolean")

        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "lower_bound": lo,
            "upper_bound": hi,
            "preconditioner_min": float(preconditioner_min),
            "preconditioner_max": float(preconditioner_max),
            "reset_outward_moments": reset_outward_moments,
            "format_version": self.FORMAT_VERSION,
        }
        super().__init__([widths], defaults)

    @property
    def width_parameter(self) -> nn.Parameter:
        if len(self.param_groups) != 1:
            raise RuntimeError(
                "ProjectedAdam state must contain exactly one parameter group"
            )
        params = self.param_groups[0]["params"]
        if len(params) != 1:
            raise RuntimeError(
                "ProjectedAdam state must contain exactly one width parameter"
            )
        return params[0]

    def _validate_raw_checkpoint_group(self, group: Mapping[str, Any]) -> None:
        """Validate the exact, uncast parameter-group representation."""

        if frozenset(group) != self._GROUP_KEYS:
            raise ValueError(
                "ProjectedAdam checkpoint parameter-group schema is not exact"
            )
        if (
            type(group["format_version"]) is not int
            or group["format_version"] != self.FORMAT_VERSION
        ):
            raise ValueError("unsupported ProjectedAdam checkpoint format_version")
        for key in ("lr", "eps", "preconditioner_min", "preconditioner_max"):
            value = group[key]
            if not _is_finite_real_number(value) or value <= 0:
                raise ValueError(f"invalid raw checkpoint value for {key}")
        if group["preconditioner_min"] > group["preconditioner_max"]:
            raise ValueError("invalid raw checkpoint preconditioner safeguards")
        if not group["preconditioner_min"] <= 1.0 <= group["preconditioner_max"]:
            raise ValueError("raw checkpoint preconditioner safeguards must contain 1")
        betas = group["betas"]
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or not all(_is_finite_real_number(beta) and 0 <= beta < 1 for beta in betas)
        ):
            raise ValueError("invalid raw checkpoint betas")
        if type(group["reset_outward_moments"]) is not bool:
            raise TypeError("raw checkpoint reset_outward_moments must be boolean")

        p = self.width_parameter
        for key in ("lower_bound", "upper_bound"):
            bound = group[key]
            if (
                not isinstance(bound, Tensor)
                or bound.is_complex()
                or not bound.is_floating_point()
            ):
                raise TypeError(f"raw checkpoint {key} must be a real floating tensor")
            if bound.shape != p.shape:
                raise ValueError(
                    f"raw checkpoint {key} must have the full parameter shape"
                )
            if bound.dtype != p.dtype or bound.device != p.device:
                raise TypeError(
                    f"raw checkpoint {key} dtype and device must exactly match physical widths"
                )
            _require_finite(f"raw checkpoint {key}", bound)
        lo = group["lower_bound"]
        hi = group["upper_bound"]
        if not bool(torch.all(lo < hi)):
            raise ValueError(
                "every raw lower bound must be smaller than its upper bound"
            )
        _require_finite("target physical widths", p.detach())
        if bool(torch.any(p.detach() < lo) or torch.any(p.detach() > hi)):
            raise ValueError(
                "target physical widths are outside the raw checkpoint bounds"
            )

    @staticmethod
    def _validate_diagnostic_semantics(
        state: Mapping[str, Any], group: Mapping[str, Any], *, prefix: str
    ) -> None:
        projection = state["last_projection_max"].item()
        preconditioner_min = state["last_preconditioner_min"].item()
        preconditioner_max = state["last_preconditioner_max"].item()
        outward_fraction = state["last_outward_fraction"].item()
        diagnostic_reference = state["last_preconditioner_min"]
        configured_min = torch.as_tensor(
            group["preconditioner_min"],
            dtype=diagnostic_reference.dtype,
            device=diagnostic_reference.device,
        ).item()
        configured_max = torch.as_tensor(
            group["preconditioner_max"],
            dtype=diagnostic_reference.dtype,
            device=diagnostic_reference.device,
        ).item()
        if projection < 0:
            raise ValueError(f"{prefix} last_projection_max must be non-negative")
        if preconditioner_min <= 0 or preconditioner_max < preconditioner_min:
            raise ValueError(f"{prefix} preconditioner diagnostic range is invalid")
        if preconditioner_min < configured_min or preconditioner_max > configured_max:
            raise ValueError(f"{prefix} preconditioner diagnostics violate safeguards")
        if not 0 <= outward_fraction <= 1:
            raise ValueError(f"{prefix} last_outward_fraction must lie in [0, 1]")
        if type(state["last_preconditioner_supplied"]) is not bool:
            raise TypeError(f"{prefix} last_preconditioner_supplied must be boolean")

    def _validate_loaded_group(self) -> None:
        group = self.param_groups[0]
        p = self.width_parameter
        if frozenset(group) != self._GROUP_KEYS:
            raise ValueError(
                "ProjectedAdam checkpoint parameter-group schema is not exact"
            )
        if group.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported ProjectedAdam checkpoint format_version")
        lo, hi = _validated_bounds(
            p.detach(), group["lower_bound"], group["upper_bound"]
        )
        group["lower_bound"] = lo
        group["upper_bound"] = hi
        for key in ("lr", "eps", "preconditioner_min", "preconditioner_max"):
            value = group[key]
            if not _is_finite_real_number(value) or value <= 0:
                raise ValueError(f"invalid checkpoint value for {key}")
        if group["preconditioner_min"] > group["preconditioner_max"]:
            raise ValueError("invalid checkpoint preconditioner safeguards")
        if not group["preconditioner_min"] <= 1.0 <= group["preconditioner_max"]:
            raise ValueError("checkpoint preconditioner safeguards must contain 1")
        betas = group["betas"]
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or not all(_is_finite_real_number(b) and 0 <= b < 1 for b in betas)
        ):
            raise ValueError("invalid checkpoint betas")
        if type(group["reset_outward_moments"]) is not bool:
            raise ValueError("invalid checkpoint reset_outward_moments")
        _require_finite("checkpoint widths", p.detach())
        if bool(torch.any(p.detach() < lo) or torch.any(p.detach() > hi)):
            raise ValueError("checkpoint physical widths are outside the saved bounds")

    def _validate_loaded_state(self) -> None:
        p = self.width_parameter
        state = self.state.get(p, {})
        if not state:
            return
        if frozenset(state) != self._STATE_KEYS:
            raise ValueError(
                "ProjectedAdam checkpoint optimizer-state schema is not exact"
            )
        step = state["step"]
        if type(step) is not int or step < 1:
            raise ValueError("invalid ProjectedAdam checkpoint step")

        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if not isinstance(value, Tensor) or value.shape != p.shape:
                raise ValueError(f"invalid ProjectedAdam checkpoint {key} shape")
            if value.dtype != p.dtype or value.device != p.device:
                raise TypeError(
                    f"checkpoint {key} dtype and device must match physical widths"
                )
            _require_finite(f"checkpoint {key}", value)
        if bool(torch.any(state["exp_avg_sq"] < 0)):
            raise ValueError("checkpoint exp_avg_sq must be non-negative")

        age = state["moment_age"]
        if not isinstance(age, Tensor) or age.shape != p.shape:
            raise ValueError("invalid ProjectedAdam checkpoint moment_age shape")
        if age.dtype != torch.int64 or age.device != p.device:
            raise TypeError("checkpoint moment_age must be int64 on the width device")
        if bool(torch.any(age < 0)) or bool(torch.any(age > step)):
            raise ValueError("checkpoint moment_age must lie in [0, step]")

        for key in self._SCALAR_STATE_KEYS:
            value = state[key]
            if not isinstance(value, Tensor) or value.shape != torch.Size([]):
                raise ValueError(f"invalid ProjectedAdam checkpoint {key} shape")
            if value.dtype != p.dtype or value.device != p.device:
                raise TypeError(
                    f"checkpoint {key} dtype and device must match physical widths"
                )
            _require_finite(f"checkpoint {key}", value)
        self._validate_diagnostic_semantics(
            state, self.param_groups[0], prefix="checkpoint"
        )

    def _validate_serialized_layout(self, state_dict: Mapping[str, Any]) -> None:
        """Validate checkpoint structure before allowing Optimizer to mutate state."""

        if not isinstance(state_dict, Mapping) or frozenset(state_dict) != {
            "state",
            "param_groups",
        }:
            raise ValueError("ProjectedAdam checkpoint top-level schema is not exact")
        groups = state_dict["param_groups"]
        states = state_dict["state"]
        if type(groups) is not list or len(groups) != 1:
            raise ValueError(
                "ProjectedAdam checkpoint must contain exactly one parameter group"
            )
        group = groups[0]
        if not isinstance(group, Mapping) or frozenset(group) != self._GROUP_KEYS:
            raise ValueError(
                "ProjectedAdam checkpoint parameter-group schema is not exact"
            )
        self._validate_raw_checkpoint_group(group)
        identifiers = group.get("params")
        if type(identifiers) is not list or len(identifiers) != 1:
            raise ValueError(
                "ProjectedAdam checkpoint must identify exactly one parameter"
            )
        if not isinstance(states, Mapping):
            raise TypeError("ProjectedAdam checkpoint state must be a mapping")
        identifier = identifiers[0]
        if type(identifier) is not int or identifier < 0:
            raise TypeError(
                "ProjectedAdam checkpoint parameter ID must be a non-negative integer"
            )
        if len(states) == 0:
            if self.state.get(self.width_parameter, {}):
                raise ValueError(
                    "ProjectedAdam checkpoint is missing initialized parameter state"
                )
            return
        if len(states) != 1 or identifier not in states:
            raise ValueError(
                "ProjectedAdam checkpoint contains orphan or missing parameter state"
            )
        state = states[identifier]
        if not isinstance(state, Mapping):
            raise TypeError(
                "ProjectedAdam checkpoint parameter state must be a mapping"
            )
        if not state:
            if self.state.get(self.width_parameter, {}):
                raise ValueError(
                    "ProjectedAdam checkpoint contains empty state for an initialized target"
                )
            return
        if frozenset(state) != self._STATE_KEYS:
            raise ValueError(
                "ProjectedAdam checkpoint optimizer-state schema is not exact"
            )

        p = self.width_parameter
        step = state["step"]
        if type(step) is not int or step < 1:
            raise ValueError("invalid ProjectedAdam checkpoint step")
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        for key, value in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if (
                not isinstance(value, Tensor)
                or value.shape != p.shape
                or not value.is_floating_point()
            ):
                raise ValueError(f"invalid serialized ProjectedAdam {key}")
            if value.dtype != p.dtype or value.device != p.device:
                raise TypeError(
                    f"serialized {key} dtype and device must exactly match physical widths"
                )
            _require_finite(f"serialized checkpoint {key}", value)
        if exp_avg.dtype != exp_avg_sq.dtype or exp_avg.device != exp_avg_sq.device:
            raise TypeError("serialized optimizer moments must share dtype and device")
        if bool(torch.any(exp_avg_sq < 0)):
            raise ValueError("serialized checkpoint exp_avg_sq must be non-negative")

        age = state["moment_age"]
        if (
            not isinstance(age, Tensor)
            or age.shape != p.shape
            or age.dtype != torch.int64
            or age.device != p.device
        ):
            raise TypeError(
                "serialized moment_age must be int64 beside the optimizer moments"
            )
        if bool(torch.any(age < 0)) or bool(torch.any(age > step)):
            raise ValueError("serialized checkpoint moment_age must lie in [0, step]")

        for key in self._SCALAR_STATE_KEYS:
            value = state[key]
            if not isinstance(value, Tensor) or value.shape != torch.Size([]):
                raise ValueError(f"serialized {key} must be a scalar tensor")
            if value.dtype != p.dtype or value.device != p.device:
                raise TypeError(
                    f"serialized {key} dtype and device must exactly match physical widths"
                )
            _require_finite(f"serialized checkpoint {key}", value)
        self._validate_diagnostic_semantics(state, group, prefix="raw checkpoint")

    def _restore_serialized_moment_age(self, state_dict: Mapping[str, Any]) -> None:
        """Undo PyTorch's generic parameter-dtype cast for integer age state."""

        group = state_dict["param_groups"][0]
        serialized_state = state_dict["state"]
        if not serialized_state:
            return
        raw_state = serialized_state[group["params"][0]]
        if not raw_state:
            return
        raw_age = raw_state["moment_age"]
        self.state[self.width_parameter]["moment_age"] = (
            raw_age.detach()
            .clone()
            .to(dtype=torch.int64, device=self.width_parameter.device)
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load an exact-resume checkpoint without implicit dtype/device casts.

        Callers must deserialize tensors onto the target parameter device and
        use the same parameter dtype.  Dtype migration requires a separate,
        explicit migration tool; PyTorch optimizer auto-casting is rejected.
        """

        self._validate_serialized_layout(state_dict)
        previous = copy.deepcopy(super().state_dict())
        try:
            super().load_state_dict(state_dict)
            self._restore_serialized_moment_age(state_dict)
            self._validate_loaded_group()
            self._validate_loaded_state()
        except Exception:
            try:
                super().load_state_dict(previous)
                self._restore_serialized_moment_age(previous)
                self._validate_loaded_group()
                self._validate_loaded_state()
            except Exception as rollback_error:
                raise RuntimeError(
                    "ProjectedAdam checkpoint rollback failed"
                ) from rollback_error
            raise

    @torch.no_grad()
    def project_(self) -> nn.Parameter:
        """Project the managed physical-width tensor onto its saved box."""

        group = self.param_groups[0]
        project_box_(self.width_parameter, group["lower_bound"], group["upper_bound"])
        return self.width_parameter

    @torch.no_grad()
    def step(
        self,
        closure: Any | None = None,
        *,
        preconditioner: Tensor | float | None = None,
    ) -> Tensor | None:
        if preconditioner is not None:
            # Validate before invoking a closure or taking the no-gradient
            # projection path, so boolean rejection is side-effect free.
            _reject_boolean_numeric_input("preconditioner", preconditioner)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise TypeError("closure must return a scalar Tensor")
            _require_finite("closure loss", loss.detach())

        p = self.width_parameter
        group = self.param_groups[0]
        lo = group["lower_bound"]
        hi = group["upper_bound"]
        _require_finite("physical widths before step", p)
        if p.grad is None:
            self.project_()
            return loss
        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("ProjectedAdam does not support sparse gradients")
        if grad.shape != p.shape or grad.dtype != p.dtype or grad.device != p.device:
            raise TypeError(
                "width gradient must match parameter shape, dtype, and device"
            )
        _require_finite("width gradient", grad)

        if preconditioner is None:
            metric = torch.ones_like(p)
            metric_was_supplied = False
        else:
            if isinstance(preconditioner, Tensor) and preconditioner.is_complex():
                raise TypeError("preconditioner must be real")
            try:
                metric = torch.as_tensor(preconditioner, dtype=p.dtype, device=p.device)
                metric = torch.broadcast_to(metric, p.shape).clone().detach()
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "preconditioner is not broadcastable to width shape"
                ) from exc
            _require_finite("preconditioner", metric)
            if bool(torch.any(metric < 0)):
                raise ValueError("preconditioner must be non-negative")
            metric = metric.clamp(
                min=group["preconditioner_min"], max=group["preconditioner_max"]
            )
            metric_was_supplied = True

        state = self.state.get(p, {})
        old_step = state.get("step", 0)
        if type(old_step) is not int or old_step < 0:
            raise ValueError("optimizer step state must be a non-negative integer")
        old_avg = state.get("exp_avg", torch.zeros_like(p))
        old_sq = state.get("exp_avg_sq", torch.zeros_like(p))
        old_age = state.get(
            "moment_age", torch.zeros_like(p, dtype=torch.int64, device=p.device)
        )
        if old_avg.shape != p.shape or old_sq.shape != p.shape:
            raise ValueError("optimizer moment state shape does not match widths")
        if old_avg.dtype != p.dtype or old_sq.dtype != p.dtype:
            raise TypeError("optimizer moment dtype must match widths")
        if old_avg.device != p.device or old_sq.device != p.device:
            raise TypeError("optimizer moment device must match widths")
        if (
            old_age.shape != p.shape
            or old_age.dtype != torch.int64
            or old_age.device != p.device
        ):
            raise TypeError("optimizer moment_age must be int64 on the width device")
        if bool(torch.any(old_age < 0)) or bool(torch.any(old_age > old_step)):
            raise ValueError("optimizer moment_age must lie in [0, step]")
        if bool(torch.any(old_age == torch.iinfo(torch.int64).max)):
            raise OverflowError("optimizer moment_age exhausted int64 range")
        _require_finite("first moment", old_avg)
        _require_finite("second moment", old_sq)
        if bool(torch.any(old_sq < 0)):
            raise ValueError("optimizer second moment must be non-negative")

        beta1, beta2 = group["betas"]
        new_step = old_step + 1
        new_age = old_age + 1
        new_avg = old_avg * beta1 + grad * (1.0 - beta1)
        new_sq = old_sq * beta2 + grad.square() * (1.0 - beta2)
        beta1_tensor = torch.as_tensor(beta1, dtype=p.dtype, device=p.device)
        beta2_tensor = torch.as_tensor(beta2, dtype=p.dtype, device=p.device)
        bias_correction1 = 1.0 - torch.pow(beta1_tensor, new_age)
        bias_correction2 = 1.0 - torch.pow(beta2_tensor, new_age)
        if bool(torch.any(bias_correction1 <= 0)) or bool(
            torch.any(bias_correction2 <= 0)
        ):
            raise FloatingPointError(
                "Adam bias correction underflowed at the parameter dtype"
            )
        avg_hat = new_avg / bias_correction1
        sq_hat = new_sq / bias_correction2
        direction = avg_hat / (sq_hat.sqrt() + group["eps"])
        direction = direction / metric
        unprojected = p - group["lr"] * direction
        projected = torch.maximum(torch.minimum(unprojected, hi), lo)
        for name, value in (
            ("updated first moment", new_avg),
            ("updated second moment", new_sq),
            ("first-moment bias correction", bias_correction1),
            ("second-moment bias correction", bias_correction2),
            ("Adam direction", direction),
            ("unprojected widths", unprojected),
            ("projected widths", projected),
        ):
            _require_finite(name, value)

        outward = (unprojected < lo) | (unprojected > hi)
        if group["reset_outward_moments"]:
            new_avg = torch.where(outward, torch.zeros_like(new_avg), new_avg)
            new_sq = torch.where(outward, torch.zeros_like(new_sq), new_sq)
            new_age = torch.where(outward, torch.zeros_like(new_age), new_age)

        p.copy_(projected)
        if p not in self.state:
            self.state[p] = state
        state["step"] = new_step
        state["moment_age"] = new_age
        state["exp_avg"] = new_avg
        state["exp_avg_sq"] = new_sq
        state["last_projection_max"] = (
            (unprojected - projected).abs().amax().detach().clone()
        )
        state["last_preconditioner_min"] = metric.amin().detach().clone()
        state["last_preconditioner_max"] = metric.amax().detach().clone()
        state["last_preconditioner_supplied"] = metric_was_supplied
        state["last_outward_fraction"] = outward.to(p.dtype).mean().detach().clone()
        return loss

    @torch.no_grad()
    def checkpoint_diagnostics(
        self,
        gradient: Tensor | None = None,
        *,
        active_tolerance: float = 0.0,
        projected_eta: float | Tensor | None = None,
    ) -> dict[str, Any]:
        """Return deterministic, JSON-friendly KKT and optional PG diagnostics.

        ``projected_eta`` requests the exact eta-dependent mapping from
        :func:`projected_gradient_mapping`; the normal-cone KKT fields remain
        distinct and eta-independent.
        """

        _reject_boolean_numeric_input("active_tolerance", active_tolerance)
        if projected_eta is not None:
            _reject_boolean_numeric_input("projected_eta", projected_eta)

        p = self.width_parameter
        self._validate_loaded_group()
        self._validate_loaded_state()
        group = self.param_groups[0]
        state = self.state.get(p, {})
        moment_age = state.get(
            "moment_age", torch.zeros_like(p, dtype=torch.int64, device=p.device)
        )
        result: dict[str, Any] = {
            "schema": "projected_width_optimizer_diagnostics_v2",
            "format_version": self.FORMAT_VERSION,
            "step": int(state.get("step", 0)),
            "moment_age_min": int(moment_age.amin().detach().cpu()),
            "moment_age_max": int(moment_age.amax().detach().cpu()),
            "numel": p.numel(),
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "device": str(p.device),
            "width_min": float(p.detach().amin().cpu()),
            "width_max": float(p.detach().amax().cpu()),
            "last_projection_max": float(
                state.get("last_projection_max", torch.zeros((), device=p.device))
                .detach()
                .cpu()
            ),
            "last_preconditioner_min": float(
                state.get("last_preconditioner_min", torch.ones((), device=p.device))
                .detach()
                .cpu()
            ),
            "last_preconditioner_max": float(
                state.get("last_preconditioner_max", torch.ones((), device=p.device))
                .detach()
                .cpu()
            ),
            "last_preconditioner_supplied": bool(
                state.get("last_preconditioner_supplied", False)
            ),
            "last_outward_fraction": float(
                state.get("last_outward_fraction", torch.zeros((), device=p.device))
                .detach()
                .cpu()
            ),
        }
        chosen_gradient = gradient if gradient is not None else p.grad
        if chosen_gradient is not None:
            detached_gradient = chosen_gradient.detach()
            result.update(
                projected_kkt_residual(
                    p.detach(),
                    detached_gradient,
                    group["lower_bound"],
                    group["upper_bound"],
                    active_tolerance=active_tolerance,
                ).scalar_dict()
            )
            if projected_eta is not None:
                mapping = projected_gradient_mapping(
                    p.detach(),
                    detached_gradient,
                    group["lower_bound"],
                    group["upper_bound"],
                    projected_eta,
                )
                mapping_rms = _stable_rms(mapping)
                _require_finite("projected-gradient RMS", mapping_rms)
                step = torch.as_tensor(projected_eta, dtype=p.dtype, device=p.device)
                step = torch.broadcast_to(step, p.shape)
                result.update(
                    {
                        "projected_gradient_eta_min": float(step.amin().detach().cpu()),
                        "projected_gradient_eta_max": float(step.amax().detach().cpu()),
                        "projected_gradient_infinity_norm": float(
                            mapping.abs().amax().detach().cpu()
                        ),
                        "projected_gradient_rms": float(mapping_rms.detach().cpu()),
                    }
                )
        elif projected_eta is not None:
            raise ValueError("projected_eta requires an explicit or stored gradient")
        return result
