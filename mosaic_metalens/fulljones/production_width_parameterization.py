"""Feasible width parameterization for final-exact width optimization.

The paper objective remains exactly ``-I_tar``.  The feasible set is box
projection to the declared width bounds plus the declared symmetry (radial or
mirror-quadrant) and nothing else.

2026-08-10 revision: the binomial lowpass was REMOVED.  It was applied to the
optimized designs but not to the reference, which is scored from its stored
map, so it was a self-imposed handicap on one side of the paper's central
comparison.  Filtering in phase space is not an available alternative: the
response is broadband, so a width does not have "a phase" and there is no
well-defined phase function to smooth for a design that targets none.  The
coherent options were to apply the constraint symmetrically or to drop it,
and dropping it makes the design space match how the reference is evaluated.
No replacement smoothness penalty is introduced; lattice strain is measured
and recorded (see :func:`neighbour_width_difference_stats`), never penalized.
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F


PRODUCTION_WIDTH_PARAMETERIZATION_SCHEMA = (
    "fable_projected_box_symmetry_width_v2"
)
SUPERSEDED_WIDTH_PARAMETERIZATION_SCHEMA = (
    "fable_projected_binomial_lowpass_width_v1"
)
# Retained for provenance only: the superseded lowpass stencil.  Nothing in
# the active path convolves with it.
ARCHIVED_BINOMIAL_KERNEL_1D = (1, 8, 28, 56, 70, 56, 28, 8, 1)
ARCHIVED_BINOMIAL_NORMALIZATION = 256
BINOMIAL_KERNEL_1D = ARCHIVED_BINOMIAL_KERNEL_1D
BINOMIAL_NORMALIZATION = ARCHIVED_BINOMIAL_NORMALIZATION
HIGH_FREQUENCY_START_NYQUIST_FRACTION = 0.75
# Diagnostic-only neighbour-strain thresholds (micrometres).  These are
# REPORTED, never gated: the LPA-strain proxy is evidence about model
# exploitation, not a constraint on the feasible set.
NEIGHBOUR_STRAIN_THRESHOLDS_UM = (0.005, 0.010, 0.020, 0.040)


def contract_payload(*, pupil_pitch_um: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": PRODUCTION_WIDTH_PARAMETERIZATION_SCHEMA,
        "supersedes": SUPERSEDED_WIDTH_PARAMETERIZATION_SCHEMA,
        "feasible_set": "box_projection_plus_declared_symmetry_only",
        "spatial_filter": "none",
        "smoothness_penalty": "none",
        "revision_reason": (
            "the lowpass was applied to optimized designs but not to the "
            "reference, which is scored from its stored map; dropping it "
            "makes the design space match how the reference is evaluated"
        ),
        "pupil_pitch_um": float(pupil_pitch_um),
        "neighbour_strain_thresholds_um": list(NEIGHBOUR_STRAIN_THRESHOLDS_UM),
        "neighbour_strain_role": "recorded_diagnostic_never_gated",
        "role": "hard_feasible_parameterization_not_objective_penalty",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def production_width_map(width_map_um: torch.Tensor) -> torch.Tensor:
    """Return the fabricated width map for an expanded parameterization.

    The active feasible set is box projection plus the declared symmetry, both
    enforced by the parameterization itself, so this is an identity map that
    keeps the shape/finiteness guards.  It is deliberately NOT a filter: see
    the module docstring for why the lowpass was removed rather than replaced.
    """

    width = torch.as_tensor(width_map_um)
    if width.ndim != 2 or not width.is_floating_point():
        raise ValueError("production width map must be a floating 2-D tensor")
    if not bool(torch.isfinite(width.detach()).all()):
        raise ValueError("production width map must be finite")
    return width


# Compatibility spelling used across the production drivers.  Since the
# 2026-08-10 revision it performs no filtering.
filtered_production_width_map = production_width_map


def neighbour_width_difference_stats(
    width_map_um: torch.Tensor,
    *,
    aperture_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Local-lattice strain proxy: nearest-neighbour width differences.

    Reported for every scored design, including the reference and the archived
    champion, so an optimized map's strain can be compared against maps that
    are known to be physically reasonable.  Never gated.
    """

    width = torch.as_tensor(width_map_um).detach()
    if width.ndim != 2:
        raise ValueError("width_map_um must be two-dimensional")
    width = width.to(torch.float64)
    pairs = []
    for axis in (-2, -1):
        difference = torch.diff(width, dim=axis).abs()
        if aperture_mask is not None:
            mask = torch.as_tensor(aperture_mask, device=width.device)
            if mask.shape != width.shape:
                raise ValueError("aperture_mask must match the width map")
            both = mask.narrow(axis % 2, 0, mask.shape[axis % 2] - 1) & (
                mask.narrow(axis % 2, 1, mask.shape[axis % 2] - 1)
            )
            difference = difference[both]
        pairs.append(difference.reshape(-1))
    values = torch.cat(pairs)
    if values.numel() == 0:
        raise ValueError("no neighbour pairs inside the aperture")
    stats: dict[str, object] = {
        "pair_count": int(values.numel()),
        "max_um": float(values.max()),
        "rms_um": float(values.pow(2).mean().sqrt()),
        "mean_um": float(values.mean()),
    }
    stats["fraction_above"] = {
        f"{threshold:.3f}": float((values > threshold).to(torch.float64).mean())
        for threshold in NEIGHBOUR_STRAIN_THRESHOLDS_UM
    }
    return stats


def high_frequency_width_rms_um(
    width_map_um: torch.Tensor,
    *,
    start_nyquist_fraction: float = HIGH_FREQUENCY_START_NYQUIST_FRACTION,
) -> torch.Tensor:
    """RMS amplitude carried near the discrete lattice Nyquist boundary."""

    width = torch.as_tensor(width_map_um)
    if width.ndim != 2:
        raise ValueError("width_map_um must be two-dimensional")
    if not 0.0 < float(start_nyquist_fraction) < 1.0:
        raise ValueError("start_nyquist_fraction must lie strictly inside (0,1)")
    centred = width - width.mean()
    spectrum = torch.fft.fft2(centred, norm="ortho")
    fy = torch.fft.fftfreq(width.shape[-2], device=width.device)
    fx = torch.fft.fftfreq(width.shape[-1], device=width.device)
    threshold = 0.5 * float(start_nyquist_fraction)
    mask = (fy[:, None].abs() >= threshold) | (fx[None, :].abs() >= threshold)
    high_energy = spectrum.abs().square()[mask].sum()
    # Orthonormal FFT: total spatial mean square is spectral sum / N.
    return torch.sqrt(high_energy / float(width.numel()))


def validate_filtered_width_map(
    width_map_um: torch.Tensor,
    *,
    width_min_um: float,
    width_max_um: float,
) -> dict[str, object]:
    width = torch.as_tensor(width_map_um)
    minimum = float(width.detach().min())
    maximum = float(width.detach().max())
    bounds_passed = (
        minimum >= float(width_min_um) - 1.0e-7
        and maximum <= float(width_max_um) + 1.0e-7
    )
    # The high-frequency RMS is retained as a RECORDED diagnostic.  It was the
    # lowpass's own gate; keeping it as a gate would re-impose the removed
    # constraint by another name, so it no longer contributes to "passed".
    high_rms = float(high_frequency_width_rms_um(width).detach())
    return {
        "schema": PRODUCTION_WIDTH_PARAMETERIZATION_SCHEMA,
        "minimum_width_um": minimum,
        "maximum_width_um": maximum,
        "bounds_passed": bounds_passed,
        "high_frequency_rms_um": high_rms,
        "high_frequency_rms_is_recorded_diagnostic": True,
        "neighbour_strain": neighbour_width_difference_stats(width),
        "passed": bounds_passed,
    }


__all__ = [
    "ARCHIVED_BINOMIAL_KERNEL_1D",
    "NEIGHBOUR_STRAIN_THRESHOLDS_UM",
    "PRODUCTION_WIDTH_PARAMETERIZATION_SCHEMA",
    "SUPERSEDED_WIDTH_PARAMETERIZATION_SCHEMA",
    "contract_payload",
    "filtered_production_width_map",
    "high_frequency_width_rms_um",
    "neighbour_width_difference_stats",
    "production_width_map",
    "validate_filtered_width_map",
]
