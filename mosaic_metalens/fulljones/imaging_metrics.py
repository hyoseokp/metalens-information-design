"""Pure, policy-locked array metrics for completed imaging simulations.

RETARGET-2026-08-04: This module accepts only saved linear-RGB arrays.  A
reference/prediction pair is validated and clipped once to [0, 1] before the
same prepared arrays are passed to PSNR, CIE Lab, and S-CIELAB.  It contains no
forward, reconstruction, calibration-fitting, or random-noise code.
"""
from __future__ import annotations

from typing import Any

import numpy as np


METRIC_POLICY_VERSION = "fable_linear_rgb_metrics_v3"
LINEAR_RGB_CLIP_RANGE = (0.0, 1.0)
PSNR_DATA_RANGE = 1.0
RGB_TO_XYZ_D65 = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float64)
D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_CIE_EPSILON = 0.008856
_CIE_LINEAR_SLOPE = 7.787
_OPP = np.array([
    [0.279, 0.720, -0.107],
    [-0.449, 0.290, -0.077],
    [0.086, -0.590, 0.501],
], dtype=np.float64)
_OPP_INV = np.linalg.inv(_OPP)
_SCIELAB_PPD = 10.0
_SCIELAB_XYZ_FLOOR = 1.0e-9
_SCIELAB_GAUSSIAN_MODE = "nearest"
_SCIELAB_GAUSSIAN_TRUNCATE = 4.0
_SCIELAB_IMPLEMENTATION = "fable_opponent_gaussian_scielab_v1"
_CIEDE2000_IMPLEMENTATION = "sharma_wu_dalal_2005_vectorized_v1"
_SCIELAB_KERNELS = {
    0: ((0.921, 0.0283), (0.105, 0.133), (-0.108, 4.336)),
    1: ((0.531, 0.0392), (0.330, 0.494)),
    2: ((0.488, 0.0536), (0.371, 0.386)),
}


def _validate_rgb(image: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("image metrics require finite arrays")
    return value


def _clip_diagnostics(value: np.ndarray) -> dict[str, float]:
    low, high = LINEAR_RGB_CLIP_RANGE
    below = value < low
    above = value > high
    return {
        "below_range_value_fraction": float(np.mean(below)),
        "above_range_value_fraction": float(np.mean(above)),
        "changed_value_fraction": float(np.mean(below | above)),
    }


def prepare_linear_rgb_pair(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Validate and identically clip a metric pair exactly once.

    All metric implementations consume the two returned arrays without another
    RGB-domain clip.  XYZ-domain floors used by the Lab transforms are separate
    numerical conventions and are recorded by :func:`metric_policy`.
    """
    ref = _validate_rgb(reference, name="reference")
    pred = _validate_rgb(prediction, name="prediction")
    if ref.shape != pred.shape:
        raise ValueError(f"expected matching [H,W,3] arrays, got {ref.shape} and {pred.shape}")
    diagnostics = {
        "reference": _clip_diagnostics(ref),
        "prediction": _clip_diagnostics(pred),
    }
    low, high = LINEAR_RGB_CLIP_RANGE
    return np.clip(ref, low, high), np.clip(pred, low, high), diagnostics


def _prepare_linear_rgb(image: np.ndarray, *, name: str) -> np.ndarray:
    value = _validate_rgb(image, name=name)
    low, high = LINEAR_RGB_CLIP_RANGE
    return np.clip(value, low, high)


def _lab_nonlinearity(ratio: np.ndarray) -> np.ndarray:
    return np.where(
        ratio > _CIE_EPSILON,
        np.cbrt(ratio),
        _CIE_LINEAR_SLOPE * ratio + 16.0 / 116.0,
    )


def _xyz_to_lab(xyz: np.ndarray, *, xyz_floor: float) -> np.ndarray:
    ratio = np.maximum(xyz, xyz_floor) / D65_WHITE
    transformed = _lab_nonlinearity(ratio)
    lightness = 116.0 * transformed[..., 1] - 16.0
    a = 500.0 * (transformed[..., 0] - transformed[..., 1])
    b = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return np.stack((lightness, a, b), axis=-1)


def _linear_rgb_to_lab_prepared(linear_rgb: np.ndarray) -> np.ndarray:
    return _xyz_to_lab(linear_rgb @ RGB_TO_XYZ_D65.T, xyz_floor=0.0)


def linear_rgb_to_lab(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert one finite linear-sRGB image after the locked [0, 1] clip."""
    return _linear_rgb_to_lab_prepared(
        _prepare_linear_rgb(linear_rgb, name="linear_rgb")
    )


def _delta_e76_prepared(reference: np.ndarray, prediction: np.ndarray) -> float:
    difference = (
        _linear_rgb_to_lab_prepared(prediction)
        - _linear_rgb_to_lab_prepared(reference)
    )
    return float(np.sqrt(np.sum(difference * difference, axis=-1)).mean())


def delta_e76_mean(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref, pred, _ = prepare_linear_rgb_pair(reference, prediction)
    return _delta_e76_prepared(ref, pred)


def ciede2000_delta_e(lab_reference: np.ndarray, lab_prediction: np.ndarray) -> np.ndarray:
    """Return elementwise CIEDE2000 colour difference for CIELAB arrays.

    This is the Sharma--Wu--Dalal implementation of the CIEDE2000 formula
    with the standard parametric factors ``k_L = k_C = k_H = 1``.  The last
    axis is ``(L*, a*, b*)``; all leading dimensions are preserved.  Keeping
    this Lab-domain function public also lets the published reference pairs
    test the formula independently of the RGB-to-Lab conversion.
    """

    first = np.asarray(lab_reference, dtype=np.float64)
    second = np.asarray(lab_prediction, dtype=np.float64)
    if first.shape != second.shape or first.ndim < 1 or first.shape[-1] != 3:
        raise ValueError("CIEDE2000 inputs must have matching [...,3] shapes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("CIEDE2000 inputs must be finite")

    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar_7 = c_bar ** 7
    twenty_five_7 = 25.0 ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar_7 / (c_bar_7 + twenty_five_7)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)
    h1_prime = np.mod(np.degrees(np.arctan2(b1, a1_prime)), 360.0)
    h2_prime = np.mod(np.degrees(np.arctan2(b2, a2_prime)), 360.0)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    hue_difference = h2_prime - h1_prime
    chroma_product_nonzero = (c1_prime * c2_prime) != 0.0
    delta_h_prime = np.where(
        ~chroma_product_nonzero,
        0.0,
        np.where(
            np.abs(hue_difference) <= 180.0,
            hue_difference,
            np.where(hue_difference > 180.0, hue_difference - 360.0,
                     hue_difference + 360.0),
        ),
    )
    delta_big_h_prime = (
        2.0 * np.sqrt(c1_prime * c2_prime)
        * np.sin(np.radians(0.5 * delta_h_prime))
    )

    l_bar_prime = 0.5 * (l1 + l2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)
    hue_sum = h1_prime + h2_prime
    hue_absolute_difference = np.abs(h1_prime - h2_prime)
    h_bar_prime = np.where(
        ~chroma_product_nonzero,
        hue_sum,
        np.where(
            hue_absolute_difference <= 180.0,
            0.5 * hue_sum,
            np.where(hue_sum < 360.0, 0.5 * (hue_sum + 360.0),
                     0.5 * (hue_sum - 360.0)),
        ),
    )

    h_bar_radians = np.radians(h_bar_prime)
    t = (
        1.0
        - 0.17 * np.cos(h_bar_radians - np.radians(30.0))
        + 0.24 * np.cos(2.0 * h_bar_radians)
        + 0.32 * np.cos(3.0 * h_bar_radians + np.radians(6.0))
        - 0.20 * np.cos(4.0 * h_bar_radians - np.radians(63.0))
    )
    delta_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0) ** 2)
    c_bar_prime_7 = c_bar_prime ** 7
    r_c = 2.0 * np.sqrt(
        c_bar_prime_7 / (c_bar_prime_7 + twenty_five_7)
    )
    l_offset_squared = (l_bar_prime - 50.0) ** 2
    s_l = 1.0 + 0.015 * l_offset_squared / np.sqrt(20.0 + l_offset_squared)
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_big_h_prime / s_h
    squared = l_term * l_term + c_term * c_term + h_term * h_term
    squared += r_t * c_term * h_term
    return np.sqrt(np.maximum(squared, 0.0))


def _delta_e00_prepared(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(ciede2000_delta_e(
        _linear_rgb_to_lab_prepared(reference),
        _linear_rgb_to_lab_prepared(prediction),
    ).mean())


def delta_e00_mean(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean CIEDE2000 after the shared locked linear-RGB preparation."""
    ref, pred, _ = prepare_linear_rgb_pair(reference, prediction)
    return _delta_e00_prepared(ref, pred)


def _psnr_prepared(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference - prediction) ** 2))
    return float(
        10.0 * np.log10(
            (PSNR_DATA_RANGE * PSNR_DATA_RANGE) / max(mse, 1.0e-30)
        )
    )


def psnr_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    """PSNR of clipped linear RGB with the fixed data range 1."""
    ref, pred, _ = prepare_linear_rgb_pair(reference, prediction)
    return _psnr_prepared(ref, pred)


def _gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px < 0.2:
        return image
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(
        image,
        sigma=sigma_px,
        mode=_SCIELAB_GAUSSIAN_MODE,
        truncate=_SCIELAB_GAUSSIAN_TRUNCATE,
    )


def _scielab_filtered_lab_prepared(linear_rgb: np.ndarray) -> np.ndarray:
    xyz = linear_rgb @ RGB_TO_XYZ_D65.T
    opponent = xyz @ _OPP.T
    for channel, kernels in _SCIELAB_KERNELS.items():
        accumulator = np.zeros_like(opponent[..., channel])
        weight_sum = 0.0
        for weight, sigma_degrees in kernels:
            accumulator += weight * _gaussian_blur(
                opponent[..., channel], sigma_degrees * _SCIELAB_PPD,
            )
            weight_sum += weight
        opponent[..., channel] = accumulator / weight_sum
    filtered_xyz = opponent @ _OPP_INV.T
    return _xyz_to_lab(filtered_xyz, xyz_floor=_SCIELAB_XYZ_FLOOR)


def _scielab_delta_e_prepared(reference: np.ndarray, prediction: np.ndarray) -> float:
    difference = (
        _scielab_filtered_lab_prepared(reference)
        - _scielab_filtered_lab_prepared(prediction)
    )
    return float(np.sqrt(np.sum(difference * difference, axis=-1)).mean())


def scielab_delta_e_mean(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref, pred, _ = prepare_linear_rgb_pair(reference, prediction)
    return _scielab_delta_e_prepared(ref, pred)


def score_image_with_diagnostics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score one pair after one shared preparation pass."""
    ref, pred, diagnostics = prepare_linear_rgb_pair(reference, prediction)
    return {
        "psnr_db": _psnr_prepared(ref, pred),
        "delta_e00": _delta_e00_prepared(ref, pred),
        "delta_e76": _delta_e76_prepared(ref, pred),
        "scielab_delta_e": _scielab_delta_e_prepared(ref, pred),
    }, diagnostics


def score_image(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return score_image_with_diagnostics(reference, prediction)[0]


def summarize(samples: np.ndarray | list[float]) -> dict[str, float | int]:
    """Return descriptive statistics only; never manufacture a confidence interval."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summary samples must be a finite non-empty vector")
    standard_deviation = 0.0 if values.size == 1 else float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "std": standard_deviation,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "n": int(values.size),
    }


def hierarchical_bootstrap_pending(
    *,
    scene_count: int,
    paired_noise_realizations: int,
) -> dict[str, Any]:
    """Describe the intentionally unavailable scene/noise confidence interval."""
    if scene_count < 0 or paired_noise_realizations < 0:
        raise ValueError("bootstrap axis counts must be nonnegative")
    return {
        "method": "paired_hierarchical_scene_then_noise_bootstrap",
        "status": "pending_insufficient_saved_scene_axis",
        "confidence_level": 0.95,
        "interval": None,
        "available_scene_count": int(scene_count),
        "available_paired_noise_realizations": int(paired_noise_realizations),
        "required_axes": ["scene", "paired_noise_realization"],
        "note": (
            "No confidence interval is reported until one immutable evaluation "
            "collection contains multiple test scenes and paired noise realizations."
        ),
    }


def metric_policy() -> dict[str, Any]:
    """Return the complete JSON-serializable primary metric convention."""
    import scipy

    return {
        "version": METRIC_POLICY_VERSION,
        "input": {
            "encoding": "linear_sRGB",
            "clip_range": list(LINEAR_RGB_CLIP_RANGE),
            "clip_application": (
                "reference_and_reconstruction_once_before_all_metrics"
            ),
        },
        "psnr": {
            "domain": "linear_RGB_all_values",
            "data_range": PSNR_DATA_RANGE,
        },
        "cie_lab": {
            "rgb_to_xyz_matrix": RGB_TO_XYZ_D65.tolist(),
            "reference_white": "D65_Y_equals_1",
            "reference_white_xyz": D65_WHITE.tolist(),
            "input_rgb_clip": list(LINEAR_RGB_CLIP_RANGE),
            "xyz_floor_before_white_normalization": 0.0,
            "epsilon": _CIE_EPSILON,
            "linear_slope": _CIE_LINEAR_SLOPE,
        },
        "ciede2000": {
            "implementation": _CIEDE2000_IMPLEMENTATION,
            "parametric_factors": {"k_L": 1.0, "k_C": 1.0, "k_H": 1.0},
            "aggregation": "arithmetic_mean_of_per_pixel_delta_E_00",
            "reference": "Sharma_Wu_Dalal_2005",
        },
        "scielab": {
            "implementation": _SCIELAB_IMPLEMENTATION,
            "pixels_per_degree": _SCIELAB_PPD,
            "opponent_matrix": _OPP.tolist(),
            "kernels_by_opponent_channel": {
                str(channel): [
                    {"weight": weight, "sigma_degrees": sigma_degrees}
                    for weight, sigma_degrees in kernels
                ]
                for channel, kernels in _SCIELAB_KERNELS.items()
            },
            "filtered_xyz_floor_before_D65_normalization": _SCIELAB_XYZ_FLOOR,
            "gaussian_filter": {
                "implementation": "scipy.ndimage.gaussian_filter",
                "scipy_version": scipy.__version__,
                "mode": _SCIELAB_GAUSSIAN_MODE,
                "truncate_sigma": _SCIELAB_GAUSSIAN_TRUNCATE,
            },
        },
    }
