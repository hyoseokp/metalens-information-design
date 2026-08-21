"""Exact 2x2 RGGB polyphase operator for a periodic LSI sensor model.

The routines in this module keep all Fourier arrays in *unshifted* FFT
ordering.  For a fine grid ``(H, W)``, a coarse frequency ``(u, v)`` sees the
four aliases

``(u + q_y H/2, v + q_x W/2)``, ``q = (0,0), (0,1), (1,0), (1,1)``.

With unitary DFTs on the fine and phase-decimated grids, the RGGB sample at
phase ``p`` has coefficient

``1/2 * exp(+i * omega_q dot p)``.

This is exact for a locally shift-invariant Gaussian scene/noise model with an
infinite periodic 2x2 Bayer mosaic (or its circular finite-grid analogue).  It
is not a global raw operator for a spatially varying or finite-boundary optical
system; field integration and optimization are intentionally outside this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .sensor_information import color_mimo_information_density


@dataclass(frozen=True)
class BayerPolyphaseOrdering:
    """Explicit row/column ordering metadata for the polyphase matrix.

    Matrix rows are phase-major in ``phase_offsets_yx`` order.  Matrix columns
    are alias-major, scene-channel-minor: ``column = alias * K + scene``.
    Sensor channel indices assume an input row order of ``(R, G, B)``.
    """

    phase_offsets_yx: tuple[tuple[int, int], ...]
    phase_labels: tuple[str, ...]
    phase_sensor_channels: tuple[int, ...]
    alias_offsets_yx: tuple[tuple[int, int], ...]
    sensor_channel_labels: tuple[str, ...]
    frequency_order: str
    matrix_row_order: str
    matrix_column_order: str


RGGB_ORDERING = BayerPolyphaseOrdering(
    phase_offsets_yx=((0, 0), (0, 1), (1, 0), (1, 1)),
    phase_labels=("R", "G_at_01", "G_at_10", "B"),
    phase_sensor_channels=(0, 1, 1, 2),
    alias_offsets_yx=((0, 0), (0, 1), (1, 0), (1, 1)),
    sensor_channel_labels=("R", "G", "B"),
    frequency_order="unshifted FFT",
    matrix_row_order="phase-major: R(0,0), G(0,1), G(1,0), B(1,1)",
    matrix_column_order="alias-major then scene-channel-minor",
)


@dataclass(frozen=True)
class BayerPolyphaseOperator:
    """An RGGB polyphase matrix and the metadata needed to interpret it.

    ``matrix`` has shape ``[..., H/2, W/2, 4, 4*K]``.  The final two axes are
    documented by :attr:`ordering`; ``fine_grid_shape`` and
    ``coarse_grid_shape`` make the frequency-grid convention explicit.
    """

    matrix: torch.Tensor
    ordering: BayerPolyphaseOrdering
    fine_grid_shape: tuple[int, int]
    coarse_grid_shape: tuple[int, int]
    scene_channels: int


def _validate_fine_grid(height: int, width: int) -> None:
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(
            f"RGGB polyphase analysis requires positive even H and W, got "
            f"{(height, width)}"
        )


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    if dtype in (torch.float32, torch.complex64):
        return torch.float32
    raise ValueError(
        "mixing/covariance tensors must use float32, float64, complex64, or "
        "complex128"
    )


def build_rggb_polyphase_operator(
    color_mixing_otf: torch.Tensor,
) -> BayerPolyphaseOperator:
    """Build the exact periodic-RGGB alias operator.

    Parameters
    ----------
    color_mixing_otf:
        Unshifted fine-grid OTF with shape
        ``[..., H, W, M_sensor, K_scene]``.  Sensor rows 0, 1, and 2 are
        interpreted as R, G, and B.  ``H`` and ``W`` must both be even.

    Returns
    -------
    BayerPolyphaseOperator
        A matrix of shape ``[..., H/2, W/2, 4, 4*K_scene]``.  At each coarse
        frequency its row ``p`` and alias/scene column ``(q, k)`` equals
        ``0.5 * exp(+i omega_q dot p) * M[omega_q, c(p), k]``.
    """

    mixing = torch.as_tensor(color_mixing_otf)
    if mixing.ndim < 4:
        raise ValueError(
            "color_mixing_otf must have shape [..., H, W, M_sensor, K_scene]"
        )
    if not (mixing.is_floating_point() or mixing.is_complex()):
        raise ValueError("color_mixing_otf must be floating or complex")
    real_dtype = _real_dtype(mixing.dtype)
    height, width, n_sensor, n_scene = mixing.shape[-4:]
    _validate_fine_grid(height, width)
    if n_sensor < 3:
        raise ValueError("RGGB requires at least three input sensor rows (R, G, B)")
    if n_scene <= 0:
        raise ValueError("K_scene must be positive")
    if not bool(torch.isfinite(mixing).all().detach()):
        raise ValueError("color_mixing_otf must be finite")

    half_h, half_w = height // 2, width // 2
    aliases = torch.stack(
        [
            mixing[
                ...,
                q_y * half_h : (q_y + 1) * half_h,
                q_x * half_w : (q_x + 1) * half_w,
                :,
                :,
            ]
            for q_y, q_x in RGGB_ORDERING.alias_offsets_yx
        ],
        dim=-3,
    )  # [..., H/2, W/2, alias, sensor, scene]

    # Select the R/G/G/B sensor row for each phase.  Stacking before the alias
    # axis gives [..., H/2, W/2, phase, alias, scene].
    selected = torch.stack(
        [
            aliases[..., :, sensor_channel, :]
            for sensor_channel in RGGB_ORDERING.phase_sensor_channels
        ],
        dim=-3,
    )

    base_y = torch.arange(half_h, device=mixing.device, dtype=real_dtype)
    base_x = torch.arange(half_w, device=mixing.device, dtype=real_dtype)
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    phase_coefficients = []
    for p_y, p_x in RGGB_ORDERING.phase_offsets_yx:
        alias_coefficients = []
        for q_y, q_x in RGGB_ORDERING.alias_offsets_yx:
            omega_y = 2.0 * math.pi * (grid_y + q_y * half_h) / height
            omega_x = 2.0 * math.pi * (grid_x + q_x * half_w) / width
            angle = omega_y * p_y + omega_x * p_x
            alias_coefficients.append(0.5 * torch.exp(1j * angle))
        phase_coefficients.append(torch.stack(alias_coefficients, dim=-1))
    coefficient = torch.stack(phase_coefficients, dim=-2)
    # coefficient is [H/2, W/2, phase, alias].  Explicit singleton batch axes
    # avoid relying on accidental alignment when field/batch dimensions exist.
    batch_ndim = mixing.ndim - 4
    coefficient = coefficient.reshape(
        (1,) * batch_ndim + (half_h, half_w, 4, 4, 1)
    )
    matrix = (selected * coefficient).reshape(
        *mixing.shape[:-4], half_h, half_w, 4, 4 * n_scene
    )
    return BayerPolyphaseOperator(
        matrix=matrix,
        ordering=RGGB_ORDERING,
        fine_grid_shape=(height, width),
        coarse_grid_shape=(half_h, half_w),
        scene_channels=n_scene,
    )


def rggb_phase_noise_variances(
    channel_noise_variance_e2: torch.Tensor,
) -> torch.Tensor:
    """Map ``[..., R, G, B, ...]`` channel variances to R/G/G/B phases.

    The last input axis must contain at least the three sensor channels in
    ``(R, G, B)`` order.  Output shape is ``[..., 4]`` and the green variance
    is repeated for phases ``(0,1)`` and ``(1,0)``.
    """

    noise = torch.as_tensor(channel_noise_variance_e2)
    if noise.ndim < 1 or not noise.is_floating_point():
        raise ValueError("channel noise variances must be a floating tensor")
    if noise.shape[-1] < 3:
        raise ValueError("channel noise variances must contain R, G, and B")
    if not bool(torch.isfinite(noise).all().detach()) or not bool(
        (noise > 0).all().detach()
    ):
        raise ValueError("channel noise variances must be finite and positive")
    indices = torch.tensor(
        RGGB_ORDERING.phase_sensor_channels, device=noise.device, dtype=torch.long
    )
    return noise.index_select(-1, indices)


def _validate_hermitian_psd(covariance: torch.Tensor) -> None:
    if not bool(torch.isfinite(covariance).all().detach()):
        raise ValueError("scene covariance must be finite")
    detached = covariance.detach()
    adjoint = detached.conj().transpose(-2, -1)
    real_dtype = _real_dtype(detached.dtype)
    finfo = torch.finfo(real_dtype)
    local_scale = torch.maximum(
        detached.abs().amax(dim=(-2, -1), keepdim=True),
        torch.full(
            (*detached.shape[:-2], 1, 1),
            finfo.tiny,
            device=detached.device,
            dtype=real_dtype,
        ),
    )
    tolerance = 128.0 * finfo.eps * local_scale
    if not bool(((detached - adjoint).abs() <= tolerance).all()):
        raise ValueError("scene covariance must be Hermitian")

    # Validation is detached from the objective.  Audit CUDA inputs on the
    # host because the Windows RTX 5880 stack used by this project can reject
    # batched cusolverDnXsyevBatched calls.  Per-matrix tolerances prevent a
    # bright frequency/field sample from hiding a small invalid covariance.
    hermitian = 0.5 * (detached + adjoint)
    if hermitian.device.type == "cuda":
        eigenvalues = torch.linalg.eigvalsh(hermitian.cpu()).real
        eigen_tolerance = tolerance.cpu().squeeze(-1)
    else:
        eigenvalues = torch.linalg.eigvalsh(hermitian).real
        eigen_tolerance = tolerance.squeeze(-1)
    if not bool((eigenvalues >= -eigen_tolerance).all()):
        raise ValueError("scene covariance must be positive semidefinite")


def build_aliased_scene_covariance(
    full_grid_scene_covariance: torch.Tensor,
) -> torch.Tensor:
    """Collect four independent aliases into a block-diagonal covariance.

    ``full_grid_scene_covariance`` must have shape ``[..., H, W, K, K]`` in
    unshifted FFT order.  It must be Hermitian positive semidefinite at every
    fine-grid frequency.  The return shape is
    ``[..., H/2, W/2, 4*K, 4*K]`` with alias-major block order matching
    :data:`RGGB_ORDERING`.

    Cross-frequency covariance is absent by construction.  That is the exact
    Fourier representation of a periodic stationary Gaussian scene, but not a
    model of globally nonstationary scene statistics.
    """

    covariance = torch.as_tensor(full_grid_scene_covariance)
    if covariance.ndim < 4:
        raise ValueError(
            "full_grid_scene_covariance must have shape [..., H, W, K, K]"
        )
    if not (covariance.is_floating_point() or covariance.is_complex()):
        raise ValueError("scene covariance must be floating or complex")
    _real_dtype(covariance.dtype)
    height, width, rows, columns = covariance.shape[-4:]
    _validate_fine_grid(height, width)
    if rows <= 0 or columns != rows:
        raise ValueError("scene covariance matrices must be non-empty and square")
    _validate_hermitian_psd(covariance)

    half_h, half_w = height // 2, width // 2
    aliases = torch.stack(
        [
            covariance[
                ...,
                q_y * half_h : (q_y + 1) * half_h,
                q_x * half_w : (q_x + 1) * half_w,
                :,
                :,
            ]
            for q_y, q_x in RGGB_ORDERING.alias_offsets_yx
        ],
        dim=-3,
    )  # [..., H/2, W/2, alias, K, K]
    alias_eye = torch.eye(4, device=covariance.device, dtype=covariance.dtype)
    # [alias, i, other_alias, j], then flatten the two alias/scene pairs.
    blocks = torch.einsum("...qij,qr->...qirj", aliases, alias_eye)
    return blocks.reshape(
        *covariance.shape[:-4], half_h, half_w, 4 * rows, 4 * rows
    )


def rggb_polyphase_information_bpp(
    color_mixing_otf: torch.Tensor,
    full_grid_scene_covariance: torch.Tensor,
    channel_noise_variance_e2: torch.Tensor,
) -> torch.Tensor:
    """Return Gaussian information in bits per raw Bayer pixel.

    The real-Gaussian factor is fixed at ``1/2``.  Log-determinants are summed
    over the coarse frequency grid and divided by ``H*W``; equivalently, the
    mean information per coarse frequency is divided by the four raw pixels in
    each 2x2 Bayer cell.  Optional leading batch/field dimensions are retained.
    """

    operator = build_rggb_polyphase_operator(color_mixing_otf)
    covariance = torch.as_tensor(
        full_grid_scene_covariance,
        device=operator.matrix.device,
    )
    if covariance.shape[-4:-2] != operator.fine_grid_shape:
        raise ValueError(
            "scene covariance frequency grid must match color_mixing_otf: "
            f"expected {operator.fine_grid_shape}, got {covariance.shape[-4:-2]}"
        )
    if covariance.shape[-1] != operator.scene_channels:
        raise ValueError(
            "scene covariance channel count must match K_scene: "
            f"expected {operator.scene_channels}, got {covariance.shape[-1]}"
        )
    aliased_covariance = build_aliased_scene_covariance(covariance)
    phase_noise = rggb_phase_noise_variances(
        torch.as_tensor(
            channel_noise_variance_e2,
            device=operator.matrix.device,
            dtype=operator.matrix.real.dtype,
        )
    )
    density = color_mimo_information_density(
        operator.matrix,
        aliased_covariance,
        noise_variance_e2=phase_noise,
        information_factor=0.5,
    )
    return density.mean(dim=(-2, -1)) / 4.0


__all__ = [
    "BayerPolyphaseOperator",
    "BayerPolyphaseOrdering",
    "RGGB_ORDERING",
    "build_aliased_scene_covariance",
    "build_rggb_polyphase_operator",
    "rggb_phase_noise_variances",
    "rggb_polyphase_information_bpp",
]
