"""Width-map optimization with spot-MTF loss + radial / mirror-quadrant parameterization.

Loss
----
For each (λ, field angle θ) and each spatial frequency f ∈ {f1, f2} we sample
the incoherent MTF of the simulator's per-λ PSF and normalize by the ideal
diffraction-limited circular-aperture incoherent OTF at the same (f, λ).
The loss is the squared shortfall, summed over (λ, θ, f):

    L = Σ_{λ, θ, f} w(λ, θ, f) · (1 − MTF_λθ(f) / MTF_ideal_λ(f))²
        + γ · L_smooth(θ_param)

The PSF for each field angle θ is computed by placing a single point source on
the object plane at (x_obj = z_obj·tan θ, 0). Engine forward produces
``spectral_irradiance`` of shape ``(N_λ, Hs, Ws)`` — the incoherent PSF per λ.
We FFT this PSF, take ``|·|``, normalize by DC, and bilinearly sample at the
two spot frequencies (f1, f2). Everything is differentiable.

Why spot-MTF (not integral)
---------------------------
Two well-chosen frequencies are enough to characterize the interesting
imaging band (low-freq contrast + near-Nyquist sharpness). Direct sampling is
cheaper than integration, has no quadrature artifacts, and gives the optimizer
a sparse but well-conditioned gradient. The ideal-OTF normalization makes the
loss bounded in [0, 1] per term and λ-comparable.

Parameterization
----------------
* ``RadialWidthParam(N_r)``: 1D radial profile w(r), broadcast to a (Hp, Wp)
  width map by radial-bin lookup. Reduces 720·720 = 5.2e5 DOFs to N_r ≈ 360.
  Natural for on-axis, bandwidth-limited problems with C∞ symmetry. The
  legacy half-diagonal grid is retained; an aperture-radius annular grid is
  available explicitly to avoid radial parameters outside the clear aperture.

* ``MirrorQuadrantWidthParam(Hq, Wq)``: top-left (Hq, Wq) quadrant tile,
  expanded to full (Hp, Wp) by x/y reflection. ~4× DOF reduction. This is
  mirror-quadrant (D2 reflection) symmetry, not 90-degree rotational C4
  symmetry. ``C4QuadrantWidthParam`` remains as a compatibility alias.

Both parameterizations are nn.Modules; a single trainable Tensor lives in
.theta and ``self.expand()`` returns the (Hp, Wp) width map. The width map
is fed to the engine via the ``width_map_override`` keyword on
``MetalensImagingEngine.forward_optics_from_object_batch``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .object_source import slice_object_batch  # noqa: F401  (re-export for callers)
from .pipeline_forward import MetalensImagingEngine
from .polyphase_objective import dense_exact_rggb_a_optimal_loss
from .sensor_objective import (
    dense_diagonal_wiener_risk_loss,
    dense_sensor_information_loss,
)
from .types import ObjectPointBatch


# ─────────────────────────────────────────────────────────────────────────────
# Parameterizations
# ────────────────────────────────────────────────────────────────────────────

RADIAL_GRID_LEGACY = "legacy_half_diagonal"
RADIAL_GRID_APERTURE = "aperture_radius"


def _canonical_radial_grid(radial_grid: str) -> str:
    aliases = {
        "legacy_half_diagonal": RADIAL_GRID_LEGACY,
        "legacy-half-diagonal": RADIAL_GRID_LEGACY,
        "half_diagonal": RADIAL_GRID_LEGACY,
        "half-diagonal": RADIAL_GRID_LEGACY,
        "aperture_radius": RADIAL_GRID_APERTURE,
        "aperture-radius": RADIAL_GRID_APERTURE,
        "aperture": RADIAL_GRID_APERTURE,
    }
    try:
        return aliases[str(radial_grid)]
    except KeyError as exc:
        valid = "legacy_half_diagonal or aperture_radius"
        raise ValueError(f"unknown radial_grid={radial_grid!r}; expected {valid}") from exc


def _radial_grid_geometry(
    pupil_h: int,
    pupil_w: int,
    pupil_pitch_um: float,
    n_radial: int,
    *,
    radial_grid: str,
    aperture_radius_um: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str, float]:
    """Build radial indices/radii in micrometres.

    The legacy mode deliberately reproduces the original endpoint-centred,
    nearest-radius lookup through the square's half diagonal. The aperture
    mode instead uses ``n_radial`` equal-width annuli on ``[0, R_ap)`` and a
    floor lookup. The annular convention makes the central four samples of an
    even pupil use bin zero and prevents the otherwise-dead r=0 endpoint DOF.
    """
    if n_radial < 1:
        raise ValueError("n_radial must be >= 1")
    if pupil_pitch_um <= 0.0:
        raise ValueError("pupil_pitch_um must be positive")

    mode = _canonical_radial_grid(radial_grid)
    ys = torch.arange(pupil_h, dtype=torch.float32) - (pupil_h - 1) / 2.0
    xs = torch.arange(pupil_w, dtype=torch.float32) - (pupil_w - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rr = torch.sqrt(xx.square() + yy.square()) * float(pupil_pitch_um)
    legacy_step_um = float(rr.max().item()) / max(n_radial - 1, 1)

    if mode == RADIAL_GRID_LEGACY:
        r_step_um = legacy_step_um
        if n_radial == 1:
            idx = torch.zeros_like(rr, dtype=torch.long)
        else:
            idx = torch.round(rr / r_step_um).long().clamp_(0, n_radial - 1)
        radial_r_um = torch.arange(n_radial, dtype=torch.float32) * r_step_um
        active_mask = torch.ones_like(rr, dtype=torch.bool)
        return idx, radial_r_um, active_mask, r_step_um, mode, legacy_step_um

    if aperture_radius_um is None or float(aperture_radius_um) <= 0.0:
        raise ValueError(
            "aperture_radius_um must be positive for radial_grid='aperture_radius'"
        )
    radius = float(aperture_radius_um)
    r_step_um = radius / n_radial
    idx = torch.floor(rr / r_step_um).long().clamp_(0, n_radial - 1)
    radial_r_um = (torch.arange(n_radial, dtype=torch.float32) + 0.5) * r_step_um
    active_mask = rr <= radius

    # This opt-in grid promises that every latent variable affects the clear
    # aperture. Refuse an over-resolved grid with empty native-grid annuli.
    counts = torch.bincount(idx[active_mask], minlength=n_radial)
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty.numel():
        raise ValueError(
            f"aperture-radius grid has {empty.numel()} dead radial bins; "
            "reduce n_radial or increase pupil sampling"
        )
    return idx, radial_r_um, active_mask, r_step_um, mode, legacy_step_um


def radial_profile_from_width_map(
    width_full_um: torch.Tensor,
    n_radial: int,
    pupil_pitch_um: float,
    *,
    radial_grid: str = RADIAL_GRID_LEGACY,
    aperture_radius_um: float | None = None,
) -> torch.Tensor:
    """Area-average a full width map into the selected radial grid.

    In aperture-radius mode, samples outside the clear aperture are excluded
    rather than folded into the last trainable annulus. Thus an existing
    ``width_full`` checkpoint safely warm-starts either grid mode.
    """
    if width_full_um.ndim != 2:
        raise ValueError("width_full_um must be a 2D tensor")
    H, W = width_full_um.shape
    idx, _, active, _, _, _ = _radial_grid_geometry(
        H, W, pupil_pitch_um, n_radial,
        radial_grid=radial_grid, aperture_radius_um=aperture_radius_um,
    )
    idx = idx.to(width_full_um.device)
    active = active.to(width_full_um.device)
    values = width_full_um.to(torch.float32)
    idx_active = idx[active]
    values_active = values[active]
    sums = torch.zeros(n_radial, dtype=torch.float32, device=values.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, idx_active, values_active)
    counts.scatter_add_(0, idx_active, torch.ones_like(values_active))
    if torch.any(counts == 0):
        raise ValueError("selected radial grid contains an uninitialized (dead) bin")
    return sums / counts


class RadialWidthParam(nn.Module):
    """1D radial profile w(r) → 2D width map by radial-bin lookup.

    Parameters
    ----------
    pupil_h, pupil_w : int
        Native pupil grid shape.
    pupil_pitch_um : float
        Pupil sampling pitch.
    n_radial : int
        Number of radial bins (recommended: max(H,W)/2 + 1).
    init_w_um : Tensor [n_radial], optional
        Initial 1D radial widths. If None, fills with mid-range value.
    width_range_um : (float, float)
        Hard clamp range applied during ``expand()`` (fab limits).
    radial_grid : str
        ``legacy_half_diagonal`` (default, reproduction compatible) or the
        opt-in ``aperture_radius`` equal-annulus grid.
    aperture_radius_um : float, optional
        Required by ``aperture_radius``. Units are micrometres.
    """

    def __init__(
        self,
        pupil_h: int,
        pupil_w: int,
        pupil_pitch_um: float,
        n_radial: int,
        init_w_um: torch.Tensor | None = None,
        width_range_um: tuple[float, float] = (0.08, 0.26),
        radial_grid: str = RADIAL_GRID_LEGACY,
        aperture_radius_um: float | None = None,
    ) -> None:
        super().__init__()
        self.pupil_h = int(pupil_h)
        self.pupil_w = int(pupil_w)
        self.pupil_pitch_um = float(pupil_pitch_um)
        self.n_radial = int(n_radial)
        self.w_min, self.w_max = float(width_range_um[0]), float(width_range_um[1])

        (idx, radial_r_um, active_mask, self.r_step_um, self.radial_grid,
         self.legacy_r_step_um) = (
            _radial_grid_geometry(
                pupil_h, pupil_w, pupil_pitch_um, n_radial,
                radial_grid=radial_grid, aperture_radius_um=aperture_radius_um,
            )
        )
        self.aperture_radius_um = (
            None if aperture_radius_um is None else float(aperture_radius_um)
        )
        self.smoothness_scale = (
            1.0 if self.radial_grid == RADIAL_GRID_LEGACY else
            (self.legacy_r_step_um / self.r_step_um) ** 2
        )
        self.register_buffer("radial_index", idx)  # [Hp, Wp] int64
        self.register_buffer("radial_r_um", radial_r_um)
        # Geometry aid only: keep it out of state_dict so legacy param_state
        # keys remain exactly load-compatible (theta/radial_index/radial_r_um).
        self.register_buffer("radial_active_mask", active_mask, persistent=False)

        # Invert tanh squashing so theta starts at the physical width target.
        # expand: w = mid + half · tanh((θ - mid) / half)
        # init  : θ = mid + half · atanh((w - mid) / half)
        mid = 0.5 * (self.w_min + self.w_max)
        half = 0.5 * (self.w_max - self.w_min)
        if init_w_um is None:
            init_w = torch.full((n_radial,), mid, dtype=torch.float32)
        else:
            init_w = init_w_um.to(torch.float32).clone()
            if init_w.numel() != n_radial:
                raise ValueError(f"init_w_um length {init_w.numel()} != n_radial {n_radial}")
        # avoid atanh singularity at ±1 by clamping just inside
        eps = 1e-3
        norm = ((init_w - mid) / half).clamp(-1.0 + eps, 1.0 - eps)
        theta_init = mid + half * torch.atanh(norm)
        self.theta = nn.Parameter(theta_init)

    def expand(self) -> torch.Tensor:
        """Return (Hp, Wp) width map mapped to fab limits via tanh.

        Uses tanh-squashing instead of hard clamp so the gradient flows back
        to the 1D ``theta`` parameter even when the value is near the fab
        boundary. theta is unconstrained internally; the visible width is
        ``mid + half · tanh(theta_norm)`` where theta_norm = (theta - mid) / half.
        """
        mid = 0.5 * (self.w_min + self.w_max)
        half = 0.5 * (self.w_max - self.w_min)
        w_soft = mid + half * torch.tanh((self.theta - mid) / half)
        return w_soft[self.radial_index]  # [Hp, Wp]

    def smoothness_loss(self) -> torch.Tensor:
        """1D radial total variation (L2): ‖d w / d r‖²."""
        d = self.theta[1:] - self.theta[:-1]
        loss = d.square().mean()
        if self.radial_grid == RADIAL_GRID_LEGACY:
            # Keep the legacy path bit-exact (no multiply-by-one operation).
            return loss
        # A sampled first difference scales as Δr² for the same continuous
        # profile. Match the historical half-diagonal regularizer strength.
        return loss * self.smoothness_scale

    def fab_barrier_loss(self) -> torch.Tensor:
        """Soft barrier penalizing theta drifting beyond fab limits.

        Using tanh-mapping in expand() keeps physical width in [w_min, w_max]
        but theta itself can drift far outside, slowing convergence and
        making smoothness regularization meaningless. This barrier pulls
        theta back toward the fab range with a quadratic loss outside.
        """
        over = torch.relu(self.theta - self.w_max)
        under = torch.relu(self.w_min - self.theta)
        return over.square().mean() + under.square().mean()


class MirrorQuadrantWidthParam(nn.Module):
    """Top-left quadrant tile, x/y-mirror-expanded to full (Hp, Wp).

    Use this for fine-tuning after a radial parameterization fixes the bulk
    of the chromatic aberration. This enforces reflection across the x and y
    axes only; it does *not* enforce 90-degree rotation or x/y transposition.
    """

    def __init__(
        self,
        pupil_h: int,
        pupil_w: int,
        init_w_full_um: torch.Tensor,
        width_range_um: tuple[float, float] = (0.08, 0.26),
    ) -> None:
        super().__init__()
        if pupil_h % 2 != 0 or pupil_w % 2 != 0:
            raise ValueError("mirror-quadrant parameterization requires even pupil dimensions.")
        self.pupil_h = int(pupil_h)
        self.pupil_w = int(pupil_w)
        self.w_min, self.w_max = float(width_range_um[0]), float(width_range_um[1])

        # Quadrant of size (Hp/2, Wp/2), top-left corner of init.
        if init_w_full_um.shape != (pupil_h, pupil_w):
            raise ValueError(
                f"init_w_full_um shape {tuple(init_w_full_um.shape)} != "
                f"({pupil_h}, {pupil_w})"
            )
        Hq, Wq = pupil_h // 2, pupil_w // 2
        init_w = init_w_full_um[:Hq, :Wq].to(torch.float32).clone()
        # Invert tanh so theta init recovers the physical width after expand.
        mid = 0.5 * (self.w_min + self.w_max)
        half = 0.5 * (self.w_max - self.w_min)
        eps = 1e-3
        norm = ((init_w - mid) / half).clamp(-1.0 + eps, 1.0 - eps)
        theta_init = mid + half * torch.atanh(norm)
        self.theta = nn.Parameter(theta_init)

    def expand(self) -> torch.Tensor:
        mid = 0.5 * (self.w_min + self.w_max)
        half = 0.5 * (self.w_max - self.w_min)
        q = mid + half * torch.tanh((self.theta - mid) / half)
        top = torch.cat([q, torch.flip(q, dims=[1])], dim=1)
        bot = torch.flip(top, dims=[0])
        return torch.cat([top, bot], dim=0)

    def smoothness_loss(self) -> torch.Tensor:
        dx = self.theta[:, 1:] - self.theta[:, :-1]
        dy = self.theta[1:, :] - self.theta[:-1, :]
        return dx.square().mean() + dy.square().mean()

    def fab_barrier_loss(self) -> torch.Tensor:
        over = torch.relu(self.theta - self.w_max)
        under = torch.relu(self.w_min - self.theta)
        return over.square().mean() + under.square().mean()


# Backward-compatible import used by v27-v31 launch scripts/checkpoints. State
# dict structure is unchanged; new code and CLI should say mirror-quadrant.
C4QuadrantWidthParam = MirrorQuadrantWidthParam


class ProjectedRadialWidthParam(nn.Module):
    """Radial physical-width parameter for an exact projected optimizer.

    Unlike :class:`RadialWidthParam`, the trainable tensor is the physical
    pillar width itself.  Bounds are enforced by the optimizer's exact box
    projection, so a true boundary optimum remains reachable and smoothness is
    evaluated in physical-width space.
    """

    def __init__(
        self,
        pupil_h: int,
        pupil_w: int,
        pupil_pitch_um: float,
        n_radial: int,
        init_w_um: torch.Tensor | None = None,
        width_range_um: tuple[float, float] = (0.08, 0.26),
        radial_grid: str = RADIAL_GRID_LEGACY,
        aperture_radius_um: float | None = None,
    ) -> None:
        super().__init__()
        self.pupil_h = int(pupil_h)
        self.pupil_w = int(pupil_w)
        self.pupil_pitch_um = float(pupil_pitch_um)
        self.n_radial = int(n_radial)
        self.w_min, self.w_max = map(float, width_range_um)
        if not self.w_min < self.w_max:
            raise ValueError("width_range_um must be strictly increasing")

        (idx, radial_r_um, active_mask, self.r_step_um, self.radial_grid,
         self.legacy_r_step_um) = _radial_grid_geometry(
            pupil_h, pupil_w, pupil_pitch_um, n_radial,
            radial_grid=radial_grid, aperture_radius_um=aperture_radius_um,
        )
        self.aperture_radius_um = (
            None if aperture_radius_um is None else float(aperture_radius_um)
        )
        self.smoothness_scale = (
            1.0 if self.radial_grid == RADIAL_GRID_LEGACY else
            (self.legacy_r_step_um / self.r_step_um) ** 2
        )
        self.register_buffer("radial_index", idx)
        self.register_buffer("radial_r_um", radial_r_um)
        self.register_buffer("radial_active_mask", active_mask, persistent=False)

        mid = 0.5 * (self.w_min + self.w_max)
        if init_w_um is None:
            initial = torch.full((n_radial,), mid, dtype=torch.float32)
        else:
            initial = init_w_um.to(torch.float32).reshape(-1).clone()
            if initial.numel() != n_radial:
                raise ValueError(
                    f"init_w_um length {initial.numel()} != n_radial {n_radial}"
                )
        self.width = nn.Parameter(initial.clamp(self.w_min, self.w_max))

    @property
    def projected_parameter(self) -> nn.Parameter:
        return self.width

    def expand(self) -> torch.Tensor:
        return self.width[self.radial_index]

    def smoothness_loss(self) -> torch.Tensor:
        delta = self.width[1:] - self.width[:-1]
        loss = delta.square().mean()
        return loss if self.radial_grid == RADIAL_GRID_LEGACY else (
            loss * self.smoothness_scale
        )

    def fab_barrier_loss(self) -> torch.Tensor:
        # Exact projection owns feasibility; retain the common regularizer API.
        return self.width.sum() * 0.0


class ProjectedMirrorQuadrantWidthParam(nn.Module):
    """Active-aperture mirror-quadrant physical-width parameterization.

    Only quadrant sites that expand into the circular pupil are trainable.
    Exterior sites are fixed to the mid-range fill value and are excluded from
    both optimizer state and the physical smoothness graph.
    """

    def __init__(
        self,
        pupil_h: int,
        pupil_w: int,
        init_w_full_um: torch.Tensor,
        width_range_um: tuple[float, float] = (0.08, 0.26),
        *,
        pupil_pitch_um: float,
        aperture_radius_um: float,
    ) -> None:
        super().__init__()
        if pupil_h % 2 or pupil_w % 2:
            raise ValueError(
                "projected mirror-quadrant parameterization requires even dimensions"
            )
        if init_w_full_um.shape != (pupil_h, pupil_w):
            raise ValueError(
                f"init_w_full_um shape {tuple(init_w_full_um.shape)} != "
                f"({pupil_h}, {pupil_w})"
            )
        self.pupil_h = int(pupil_h)
        self.pupil_w = int(pupil_w)
        self.pupil_pitch_um = float(pupil_pitch_um)
        self.aperture_radius_um = float(aperture_radius_um)
        self.w_min, self.w_max = map(float, width_range_um)
        if not self.w_min < self.w_max:
            raise ValueError("width_range_um must be strictly increasing")

        hq, wq = pupil_h // 2, pupil_w // 2
        y = (
            torch.arange(hq, dtype=torch.float64) - pupil_h / 2 + 0.5
        ) * self.pupil_pitch_um
        x = (
            torch.arange(wq, dtype=torch.float64) - pupil_w / 2 + 0.5
        ) * self.pupil_pitch_um
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        active = xx.square() + yy.square() <= self.aperture_radius_um ** 2
        active_flat_index = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
        if active_flat_index.numel() == 0:
            raise ValueError("aperture contains no mirror-quadrant sites")
        self.register_buffer("quadrant_active_mask", active, persistent=False)
        self.register_buffer("active_flat_index", active_flat_index)
        mid = 0.5 * (self.w_min + self.w_max)
        self.register_buffer(
            "exterior_fill",
            torch.full((hq * wq,), mid, dtype=torch.float32),
            persistent=False,
        )
        # A constructor must not assume its registered buffers and its inputs
        # share a device: callers may pass a CUDA map and apply ``.to(device)``
        # to the RESULT, which leaves the freshly registered CPU buffers on the
        # wrong device for the gather below (and later for ``_quadrant``'s
        # scatter).  Align the module with its input up front so the instance
        # is internally consistent whether or not ``.to`` is called afterwards.
        self.to(init_w_full_um.device)
        initial_q = init_w_full_um[:hq, :wq].to(torch.float32).reshape(-1)
        initial = initial_q.index_select(0, self.active_flat_index).clamp(
            self.w_min, self.w_max
        )
        self.width = nn.Parameter(initial.clone())

    @property
    def projected_parameter(self) -> nn.Parameter:
        return self.width

    @property
    def active_independent_count(self) -> int:
        return int(self.active_flat_index.numel())

    def _quadrant(self) -> torch.Tensor:
        q = self.exterior_fill.to(dtype=self.width.dtype, device=self.width.device)
        q = q.scatter(0, self.active_flat_index, self.width)
        return q.reshape(self.pupil_h // 2, self.pupil_w // 2)

    def expand(self) -> torch.Tensor:
        q = self._quadrant()
        top = torch.cat([q, torch.flip(q, dims=[1])], dim=1)
        return torch.cat([top, torch.flip(top, dims=[0])], dim=0)

    def smoothness_loss(self) -> torch.Tensor:
        q = self._quadrant()
        mask = self.quadrant_active_mask
        dx_mask = mask[:, 1:] & mask[:, :-1]
        dy_mask = mask[1:, :] & mask[:-1, :]
        dx = (q[:, 1:] - q[:, :-1]).square()[dx_mask]
        dy = (q[1:, :] - q[:-1, :]).square()[dy_mask]
        terms = []
        if dx.numel():
            terms.append(dx.mean())
        if dy.numel():
            terms.append(dy.mean())
        if not terms:
            return self.width.sum() * 0.0
        return sum(terms)

    def fab_barrier_loss(self) -> torch.Tensor:
        return self.width.sum() * 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Spot-MTF loss
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldPoint:
    """A single object-plane field location for PSF evaluation."""
    name: str
    theta_deg: float        # field angle from optical axis
    azimuth_deg: float = 0.0  # 0 = +x, 90 = +y


def _prepare_field_weights(
    field_points: list[FieldPoint],
    field_weight: torch.Tensor | None,
    device: torch.device | str,
    *,
    normalize: bool,
) -> torch.Tensor:
    n_field = len(field_points)
    if n_field == 0:
        raise ValueError("field_points must not be empty")
    if field_weight is None:
        weights = torch.ones(n_field, dtype=torch.float32, device=device)
    else:
        weights = torch.as_tensor(field_weight, dtype=torch.float32, device=device)
        if weights.ndim != 1 or weights.numel() != n_field:
            raise ValueError(
                f"field_weight expected [{n_field}], got {tuple(weights.shape)}"
            )
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("field_weight must contain finite, non-negative values")
    weight_sum = weights.sum()
    if float(weight_sum) <= 0.0:
        raise ValueError("field_weight must have a positive sum")
    return weights / weight_sum if normalize else weights


def normalize_field_weights(
    field_points: list[FieldPoint],
    field_weight: torch.Tensor | None,
    device: torch.device | str,
) -> torch.Tensor:
    """Return non-negative field quadrature weights with unit sum.

    Information/MMSE objectives historically average over fields. Spot and
    band MTF are exceptions: their legacy contract is a raw weighted sum and
    therefore uses ``_prepare_field_weights(..., normalize=False)``.
    """
    return _prepare_field_weights(
        field_points, field_weight, device, normalize=True,
    )


def make_field_points(
    on_axis: bool = True,
    off_axis_deg: tuple[float, ...] = (5.0, 10.0),
    azimuth_deg: float = 0.0,
) -> list[FieldPoint]:
    """Convenience: build a list of field points (on-axis + off-axis along one azimuth)."""
    pts: list[FieldPoint] = []
    if on_axis:
        pts.append(FieldPoint(name="on_axis", theta_deg=0.0, azimuth_deg=0.0))
    for t in off_axis_deg:
        pts.append(FieldPoint(name=f"off_{t:g}deg", theta_deg=float(t),
                              azimuth_deg=azimuth_deg))
    return pts


def build_field_object_batch(
    engine: MetalensImagingEngine,
    field_pt: FieldPoint,
    radiance_value: float = 1.0e10,
) -> ObjectPointBatch:
    """Single-point ObjectPointBatch at given field angle, on object plane z_obj."""
    z_obj_um = float(engine.spec.object_plane.z_um)  # negative
    z_abs = abs(z_obj_um)
    th = math.radians(field_pt.theta_deg)
    az = math.radians(field_pt.azimuth_deg)
    x_obj = z_abs * math.tan(th) * math.cos(az)
    y_obj = z_abs * math.tan(th) * math.sin(az)
    n_wl = int(engine.wavelengths_um.numel())
    coords = torch.tensor([[x_obj, y_obj]], dtype=torch.float32, device=engine.device_)
    z = torch.tensor([z_obj_um], dtype=torch.float32, device=engine.device_)
    radiance = torch.full((1, n_wl), float(radiance_value),
                          dtype=torch.float32, device=engine.device_)
    return ObjectPointBatch(coords_um=coords, z_um=z, spectral_radiance=radiance)


def ideal_circular_otf(
    freq_cyc_per_um: torch.Tensor,
    wavelengths_um: torch.Tensor,
    f_um: float,
    aperture_diameter_um: float,
) -> torch.Tensor:
    """Analytic incoherent OTF of a uniform circular aperture per λ.

    f_c(λ) = D / (λ · F),   for f ≤ f_c:
        H(f) = (2/π) [ arccos(f/f_c) − (f/f_c) √(1 − (f/f_c)²) ]
    H(f > f_c) = 0.

    Returns
    -------
    Tensor of shape ``[N_λ, N_f]`` in [0, 1].
    """
    f = freq_cyc_per_um.to(torch.float32).abs().view(1, -1)
    wl = wavelengths_um.to(torch.float32).view(-1, 1)
    fc = aperture_diameter_um / (wl * float(f_um))                    # [N_λ, 1]
    rho = (f / fc).clamp(0.0, 1.0)                                    # [N_λ, N_f]
    H = (2.0 / math.pi) * (torch.arccos(rho) - rho * torch.sqrt((1.0 - rho * rho).clamp_min(0.0)))
    H = torch.where(f <= fc, H, torch.zeros_like(H))
    return H


def sample_mtf_at_freqs(
    psf: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    sensor_pitch_um: float,
) -> torch.Tensor:
    """Sample the magnitude MTF of a real PSF at K spot frequencies.

    Parameters
    ----------
    psf : Tensor [N_λ, H, W]
        Real, non-negative PSF (engine.spectral_irradiance per λ).
    freqs_cyc_per_um : Tensor [K]
        Spatial frequencies (cycles/µm) at which to sample MTF.
    sensor_pitch_um : float
        Sensor optical-grid pitch (FFT freq step = 1 / (N · pitch)).

    Returns
    -------
    Tensor [N_λ, K] of MTF values normalized so MTF(0) = 1.
    Direction: radial average across the FFT angle (so the value depends only
    on |f|), to avoid binding the loss to a single (kx, ky) line.
    """
    if psf.dim() != 3:
        raise ValueError(f"psf must be [N_λ, H, W]; got {tuple(psf.shape)}")
    Nl, H, W = psf.shape

    # Real FFT magnitude → MTF(kx, ky). Center DC.
    otf = torch.fft.fftshift(torch.fft.fft2(psf, dim=(-2, -1)), dim=(-2, -1))
    mtf = otf.abs()
    # Normalize per-λ by MTF(0) (=DC = sum of psf).
    dc = mtf[:, H // 2, W // 2].clamp_min(1e-30).view(Nl, 1, 1)
    mtf = mtf / dc                                                     # [N_λ, H, W]

    # Build a grid of (fx, fy) cycles/µm matching the centered MTF.
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=sensor_pitch_um, device=psf.device)).to(psf.dtype)
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=sensor_pitch_um, device=psf.device)).to(psf.dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(fxx * fxx + fyy * fyy)                             # [H, W] cycles/µm

    # For each requested frequency f_k, compute a soft-binned radial average
    # (Gaussian weight on |rr - f_k|) — differentiable, smooth, radial.
    sigma = float((fy[1] - fy[0]).item())   # one freq-bin width
    K = int(freqs_cyc_per_um.numel())
    f_targets = freqs_cyc_per_um.to(psf.dtype).to(psf.device).view(1, K, 1, 1)
    diff = rr.unsqueeze(0).unsqueeze(0) - f_targets                    # [1, K, H, W]
    weights = torch.exp(-0.5 * (diff / sigma).square())                # [1, K, H, W]
    norm = weights.sum(dim=(-2, -1)).clamp_min(1e-30)                  # [1, K]
    # weighted sum over (H,W) per λ
    weighted = (mtf.unsqueeze(1) * weights).sum(dim=(-2, -1))          # [N_λ, K]
    return weighted / norm                                             # [N_λ, K]


def spot_mtf_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    field_weight: torch.Tensor | None = None,
    freq_weight: torch.Tensor | None = None,
    aperture_diameter_um: float,
    f_um: float,
    log_per_field: bool = False,
) -> tuple[torch.Tensor, dict]:
    """End-to-end MTF-shortfall loss over (λ, field, freq).

    For each field point θ:
      forward 1 point source → spectral_irradiance [N_λ, Hs, Ws]
      sample MTF at ``freqs_cyc_per_um`` → [N_λ, K]
      compute target = ``ideal_circular_otf`` at same freqs → [N_λ, K]
      contribution = ((target - mtf) / target).clamp_min(0)² summed over (λ, K)
        (negative shortfalls — i.e. metalens > ideal? — are clipped to 0)

    Returns
    -------
    (loss_scalar, info_dict). info_dict keyed by field-point name with the
    per-(λ, K) MTF values for logging.
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    field_weight = _prepare_field_weights(
        field_points, field_weight, engine.device_, normalize=False,
    )
    if freq_weight is None:
        freq_weight = torch.ones(K, device=engine.device_)

    target = ideal_circular_otf(
        freqs_cyc_per_um, engine.wavelengths_um,
        f_um=f_um, aperture_diameter_um=aperture_diameter_um,
    )                                                                  # [N_λ, K]
    target_safe = target.clamp_min(1e-6)

    info: dict[str, torch.Tensor] = {"target": target.detach()}
    sensor_pitch_um = float(engine.spec.sensor_plane.pitch_um)
    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )                                                              # [N_λ, Hs, Ws]
        mtf = sample_mtf_at_freqs(spec_irr, freqs_cyc_per_um, sensor_pitch_um)  # [N_λ, K]
        info[fp.name] = mtf.detach()

        shortfall = ((target - mtf) / target_safe).clamp_min(0.0)      # [N_λ, K]
        per_term = (shortfall.square() * freq_weight.view(1, K)).sum(dim=1)  # [N_λ]
        total = total + field_weight[fi] * per_term.mean()             # scalar

    if log_per_field:
        info["loss_scalar"] = total.detach()

    return total, info


# ─────────────────────────────────────────────────────────────────────────────
# Wavelength-adaptive dense band MTF/OTF loss
# ─────────────────────────────────────────────────────────────────────────────

def sample_mtf_at_freqs_per_lambda(
    psf: torch.Tensor,
    freqs_per_lambda: torch.Tensor,
    sensor_pitch_um: float,
) -> torch.Tensor:
    """Sample MTF at K frequencies that DEPEND on λ.

    Parameters
    ----------
    psf : Tensor [N_λ, H, W]
        Real PSF per λ.
    freqs_per_lambda : Tensor [N_λ, K]
        Per-λ frequency targets (cyc/µm).
    sensor_pitch_um : float

    Returns
    -------
    Tensor [N_λ, K], MTF values normalized per-λ by DC.
    """
    Nl, H, W = psf.shape
    otf = torch.fft.fftshift(torch.fft.fft2(psf, dim=(-2, -1)), dim=(-2, -1))
    mtf = otf.abs()
    dc = mtf[:, H // 2, W // 2].clamp_min(1e-30).view(Nl, 1, 1)
    mtf = mtf / dc                                                         # [N_λ, H, W]

    fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=sensor_pitch_um, device=psf.device)).to(psf.dtype)
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=sensor_pitch_um, device=psf.device)).to(psf.dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(fxx * fxx + fyy * fyy)                                 # [H, W]

    sigma = float((fy[1] - fy[0]).item())
    K = freqs_per_lambda.shape[1]
    f_targets = freqs_per_lambda.to(psf.dtype).to(psf.device).view(Nl, K, 1, 1)
    diff = rr.unsqueeze(0).unsqueeze(0) - f_targets                        # [N_λ, K, H, W]
    weights = torch.exp(-0.5 * (diff / sigma).square())
    norm = weights.sum(dim=(-2, -1)).clamp_min(1e-30)                      # [N_λ, K]
    weighted = (mtf.unsqueeze(1) * weights).sum(dim=(-2, -1))              # [N_λ, K]
    return weighted / norm


def band_mtf_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_normalized: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    aperture_diameter_um: float,
    f_um: float,
    field_weight: torch.Tensor | None = None,
    lambda_weight: torch.Tensor | None = None,
    freq_weight: torch.Tensor | None = None,
    loss_power: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Dense λ-adaptive band MTF (OTF magnitude) loss.

    For each (λ, field, k):
      f_λk = freqs_normalized[k] · f_c(λ)         where f_c(λ) = D / (λ · F)
      target_λk = analytic incoherent OTF(f_λk; λ)
      pred_λk   = sample_mtf_at_freqs_per_lambda(...)
      shortfall = ((target − pred) / target).clamp_min(0)²
    L = Σ_{field, λ, k} w_field · w_λ · w_k · shortfall

    Sampling each λ on its own f_c-normalized grid means EVERY wavelength
    contributes a finite gradient — even far-cutoff λ are not zeroed out
    (which is what made spot-MTF favor 520 nm exclusively). Default
    ``lambda_weight`` is uniform; can be set to CFA-weighted to bias toward
    R/B channel relevance.

    Parameters
    ----------
    freqs_normalized : Tensor [K]
        K samples in (0, 1] — fractions of f_c(λ). Recommended: linspace
        (0.05, 0.95, K=16) covers the band excluding singular DC and cutoff.
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_normalized.numel())
    device = engine.device_

    fc_lambda = aperture_diameter_um / (engine.wavelengths_um * f_um)      # [N_λ]
    freqs_per_lambda = (freqs_normalized.to(device).view(1, K)
                        * fc_lambda.view(n_wl, 1))                         # [N_λ, K]

    rho = freqs_normalized.to(device).clamp(0.0, 1.0)                      # [K]
    target = (2.0 / math.pi) * (
        torch.arccos(rho) - rho * torch.sqrt((1.0 - rho * rho).clamp_min(0.0))
    )                                                                      # [K] (λ-independent)
    target = target.view(1, K).expand(n_wl, K)                             # [N_λ, K]
    target_safe = target.clamp_min(1e-6)

    field_weight = _prepare_field_weights(
        field_points, field_weight, device, normalize=False,
    )
    if lambda_weight is None:
        lambda_weight = torch.ones(n_wl, device=device)
    if freq_weight is None:
        freq_weight = torch.ones(K, device=device)

    info: dict[str, torch.Tensor] = {
        "target": target.detach(),
        "freqs_per_lambda": freqs_per_lambda.detach(),
    }
    sensor_pitch_um = float(engine.spec.sensor_plane.pitch_um)
    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )
        mtf = sample_mtf_at_freqs_per_lambda(
            spec_irr, freqs_per_lambda, sensor_pitch_um,
        )                                                                  # [N_λ, K]
        info[fp.name] = mtf.detach()

        shortfall = ((target - mtf) / target_safe).clamp_min(0.0)          # [N_λ, K]
        # Use p-norm to control water-filling behavior:
        #   p=2 (default): mean of squared shortfall (uniform penalty, current).
        #   p=4 or 6: outliers (worst λ,k) get amplified → optimizer naturally
        #             pushes them up first, leveling MTFs ("water-filling").
        #   p→∞: equivalent to max-shortfall (worst-case minimization).
        weighted = (
            shortfall.abs() ** loss_power
            * lambda_weight.view(n_wl, 1)
            * freq_weight.view(1, K)
        )
        per_field_term = (weighted.sum() / (n_wl * K)) ** (1.0 / loss_power)
        total = total + field_weight[fi] * per_field_term

    return total, info


def mtf_volume_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    field_weight: torch.Tensor | None = None,
    lambda_weight: torch.Tensor | None = None,
    freq_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Broadband MTF-volume objective (Fröch-style, sensor-agnostic).

    Maximizes the wavelength-uniform broadband MTF volume: the sum over the
    engine wavelengths and radial spatial frequencies of the DC-normalized
    magnitude MTF of the OPTICAL (pre-CFA) per-wavelength intensity PSF.  It is
    returned as a shortfall-from-unity loss so the same minimizing Adam loop the
    other objectives use applies unchanged::

        volume = mean_{λ, k} MTF(λ, f_k)            in [0, 1]
        L      = Σ_field w_field · (1 − volume)

    ``MTF(λ, f_k)`` is the per-λ DC-normalized radial-average MTF of the optical
    intensity PSF — the identical primitive ``spot_mtf_loss`` samples
    (``sample_mtf_at_freqs``).  No CFA/QE weighting enters, so every wavelength
    is weighted uniformly by default (``lambda_weight`` overrides).  ``freqs``
    are ABSOLUTE (cyc/µm), the same grid the wiener/capacity objectives use.
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    device = engine.device_
    field_weight = _prepare_field_weights(
        field_points, field_weight, device, normalize=False,
    )
    if lambda_weight is None:
        lambda_weight = torch.ones(n_wl, device=device)
    if freq_weight is None:
        freq_weight = torch.ones(K, device=device)
    lam_w = (lambda_weight.to(device).view(n_wl, 1)
             / lambda_weight.sum().clamp_min(1e-30))
    frq_w = (freq_weight.to(device).view(1, K)
             / freq_weight.sum().clamp_min(1e-30))

    sensor_pitch_um = float(engine.spec.sensor_plane.pitch_um)
    info: dict[str, torch.Tensor] = {}
    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )                                                          # [N_λ, Hs, Ws]
        mtf = sample_mtf_at_freqs(
            spec_irr, freqs_cyc_per_um, sensor_pitch_um,
        )                                                          # [N_λ, K], DC-normalized
        info[fp.name] = mtf.detach()
        volume = (mtf * lam_w * frq_w).sum()                       # weighted mean in [0,1]
        total = total + field_weight[fi] * (1.0 - volume)
    return total, info


def cfa_log_mtf_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    field_weight: torch.Tensor | None = None,
    freq_weight: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """CFA-weighted, channel-balanced log-MTF-volume objective.

    The "smart MTF engineering" intermediate between the wavelength-uniform
    ``mtf_volume_loss`` and the full reconstruction-error objective. For each
    Bayer channel c ∈ {R, G, B}::

        V_c = Σ_λ Σ_ν  CFA_c(λ) · MTF(λ, ν) · w(ν)

    where ``CFA_c(λ)`` is the engine's S20 colour-filter transmission
    normalized to unit sum over λ (so V_c is a per-channel CFA-weighted mean of
    the DC-normalized optical MTF, ∈ [0, 1]; same MTF/ν machinery as
    ``mtf_volume_loss``, ``w(ν)`` unit-sum). The loss is the channel-wise log
    volume::

        L = − Σ_c log(V_c + ε) .

    The concave per-channel log gives diminishing returns, so a starved channel
    (typically blue) dominates the gradient — a second spectral ridge may EMERGE
    near the G/B overlap (~500 nm) WITHOUT being prescribed. Minimizing L
    maximizes the product Π_c V_c (a channel-balanced geometric-mean objective).
    No QE weighting enters (CFA curves only, per definition).
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    device = engine.device_
    field_weight = _prepare_field_weights(
        field_points, field_weight, device, normalize=False,
    )
    if freq_weight is None:
        freq_weight = torch.ones(K, device=device)
    frq_w = (freq_weight.to(device).view(1, K)
             / freq_weight.sum().clamp_min(1e-30))
    cfa = engine.cfa_transmission.squeeze(-1).squeeze(-1).to(device)   # [3, N_λ]
    cfa = cfa.clamp_min(0.0)
    cfa_norm = cfa / cfa.sum(dim=1, keepdim=True).clamp_min(1e-30)     # unit sum/chan

    sensor_pitch_um = float(engine.spec.sensor_plane.pitch_um)
    info: dict[str, torch.Tensor] = {}
    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )                                                             # [N_λ, Hs, Ws]
        mtf = sample_mtf_at_freqs(
            spec_irr, freqs_cyc_per_um, sensor_pitch_um,
        )                                                             # [N_λ, K]
        mtf_lam = (mtf * frq_w).sum(dim=1)                            # [N_λ] freq-mean MTF
        v_channel = cfa_norm @ mtf_lam                                # [3] per-channel volume
        info[fp.name] = mtf.detach()
        info[fp.name + "_Vc"] = v_channel.detach()
        total = total + field_weight[fi] * (
            -torch.log(v_channel + epsilon).sum()
        )
    return total, info


# ─────────────────────────────────────────────────────────────────────────────
# Channel-effective MTF loss — uses CFA × QE × photon factor weights so each
# RGB channel sees its REAL spectrum (per-channel CFA curve summed over λ).
# More physically meaningful than per-λ MTF since this is what the sensor
# actually reads. Lateral chromatic aberration is implicitly penalized via
# FFT phase cancellation when per-λ PSFs are spatially misaligned.
# ─────────────────────────────────────────────────────────────────────────────

def channel_effective_mtf_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    channel_weights: torch.Tensor,    # [3, N_λ] CFA·QE·photon factor (positive)
    aperture_diameter_um: float,
    f_um: float,
    field_weight: torch.Tensor | None = None,
    channel_weight_loss: torch.Tensor | None = None,   # [3] R/G/B weight in loss
    loss_power: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Per-RGB-channel MTF loss using actual sensor spectrum response.

    PSF_eff_c(x,y) = Σ_λ PSF_λ(x,y) · channel_weights[c, λ]
    MTF_eff_c(f) = |FFT(PSF_eff_c)|, DC-normalized.
    Compare to MTF_ideal_c = CFA-weighted Airy MTF (analytic).

    The freq grid is ABSOLUTE (cyc/µm) — pass freqs in [0, sensor_Nyquist]
    typically.

    Returns
    -------
    (loss_scalar, info_dict).
    info has 'target_channel' [3, K] (ideal MTF per RGB channel) and
    per-field-point [3, K] (achieved MTF per RGB channel).
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    device = engine.device_
    sensor_pitch = float(engine.spec.sensor_plane.pitch_um)

    field_weight = normalize_field_weights(field_points, field_weight, device)
    if channel_weight_loss is None:
        channel_weight_loss = torch.ones(3, device=device)

    cw = channel_weights.to(device).to(torch.float32)
    if cw.shape != (3, n_wl):
        raise ValueError(f"channel_weights expected [3, {n_wl}], got {tuple(cw.shape)}")

    # ----- Ideal channel MTF (analytic, CFA-weighted Airy MTF) -----
    fc_lambda = aperture_diameter_um / (engine.wavelengths_um * f_um)        # [N_λ]
    rho = freqs_cyc_per_um.to(device).view(1, K) / fc_lambda.view(n_wl, 1)
    rho = rho.clamp(0.0, 1.0)
    mtf_lambda_ideal = (2.0 / math.pi) * (
        torch.arccos(rho) - rho * torch.sqrt((1.0 - rho * rho).clamp_min(0.0))
    )                                                                          # [N_λ, K]
    # Normalize CW so each channel sums to 1 in λ → ideal MTF is convex average
    cw_norm = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-30)               # [3, N_λ]
    mtf_ideal_c = cw_norm @ mtf_lambda_ideal                                   # [3, K]
    target_safe = mtf_ideal_c.clamp_min(1e-6)

    info: dict[str, torch.Tensor] = {
        "target_channel": mtf_ideal_c.detach(),
        "freqs_cyc_per_um": freqs_cyc_per_um.detach(),
    }

    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )                                                                       # [N_λ, Hs, Ws]

        # Channel-effective PSF: weighted sum over λ
        # spec_irr [N_λ, Hs, Ws]; cw [3, N_λ] → einsum to [3, Hs, Ws]
        psf_eff = torch.einsum("cl,lhw->chw", cw, spec_irr)                   # [3, Hs, Ws]

        # MTF magnitude
        otf = torch.fft.fftshift(torch.fft.fft2(psf_eff, dim=(-2, -1)), dim=(-2, -1))
        mtf_2d = otf.abs()
        Hs, Ws = psf_eff.shape[-2], psf_eff.shape[-1]
        dc = mtf_2d[:, Hs // 2, Ws // 2].clamp_min(1e-30).view(3, 1, 1)
        mtf_2d = mtf_2d / dc                                                    # [3, Hs, Ws]

        # Radial sampling at freqs_cyc_per_um (Gaussian soft bin)
        fy = torch.fft.fftshift(torch.fft.fftfreq(Hs, d=sensor_pitch, device=device)).to(mtf_2d.dtype)
        fx = torch.fft.fftshift(torch.fft.fftfreq(Ws, d=sensor_pitch, device=device)).to(mtf_2d.dtype)
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
        rr = torch.sqrt(fxx * fxx + fyy * fyy)
        sigma = float((fy[1] - fy[0]).item())
        f_targets = freqs_cyc_per_um.to(device).to(mtf_2d.dtype).view(1, K, 1, 1)
        diff = rr.unsqueeze(0).unsqueeze(0) - f_targets                         # [1, K, H, W]
        weights = torch.exp(-0.5 * (diff / sigma).square())
        wnorm = weights.sum(dim=(-2, -1)).clamp_min(1e-30)                      # [1, K]
        # mtf_2d [3, H, W] × weights [1, K, H, W] → [3, K, H, W] → sum H,W
        mtf_eff_c = (mtf_2d.unsqueeze(1) * weights).sum(dim=(-2, -1)) / wnorm   # [3, K]
        info[fp.name] = mtf_eff_c.detach()

        shortfall = ((mtf_ideal_c - mtf_eff_c) / target_safe).clamp_min(0.0)    # [3, K]
        weighted_term = (
            shortfall.abs() ** loss_power
            * channel_weight_loss.view(3, 1)
        )
        per_field = (weighted_term.sum() / (3 * K)) ** (1.0 / loss_power)
        total = total + field_weight[fi] * per_field

    return total, info


# ─────────────────────────────────────────────────────────────────────────────
# Wiener Reconstruction MSE loss (closed-form, target-free)  [V18]
# ─────────────────────────────────────────────────────────────────────────────

def wiener_recon_mse_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    channel_weights: torch.Tensor,        # [3, N_λ] CFA·QE·photon factor (positive)
    photon_per_channel: torch.Tensor,      # [3] mean per-pixel photon count for R/G/B
    read_noise_e: float,
    scene_power_beta: float = 2.0,
    scene_power_f0_cyc_per_um: float = 0.05,
    scene_power_amplitude: float = 1.0,
    aperture_diameter_um: float,
    f_um: float,
    field_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
    dense2d: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Wiener-optimal reconstruction MSE — closed-form, target-free.

    Derivation (single channel, single field):
        y(x) = h(x)*s(x) + n(x)
        Wiener filter:  G(f) = H*(f) / (|H(f)|² + R(f)),   R = S_n / S_s
        E[|s - ŝ|²] = ∫ S_s(f) · R(f) / (|H(f)|² + R(f)) df  (closed-form)

    Per-channel:
        |H_c(f)| = effective channel MTF = | Σ_λ CW_c(λ) · OTF(f, λ, w) |
                                            / Σ_λ CW_c(λ)              (DC-normalized)
        R_c(f)   = (σ²_read + N_photon_c) / S_s(f)        (white noise floor)
        S_s(f)   = 1 / (f0 + f)^β                          (natural-image prior)

    Total loss (3 channels × fields):
        L = (1/|F|) Σ_field Σ_c ∫ S_s(f) · R_c(f) / (|H_c(f)|² + R_c(f)) df

    No "target" MTF needed — directly minimizes Wiener-deconvolution MSE.

    Parameters
    ----------
    channel_weights : [3, N_λ]
        Same as ``channel_effective_mtf_loss`` — CFA × QE × photon-factor weight
        used to aggregate per-λ PSF into per-channel PSF.
    photon_per_channel : [3]
        Mean per-pixel photon count for R/G/B (used for shot-noise variance).
        Compute externally as Σ_λ CFA(λ)·QE(λ)·I_scene(λ)·t·λ.
    read_noise_e : float
        Read noise (e- RMS) per pixel.
    scene_power_beta : float
        Power-law exponent for natural-image prior S_s(f) = 1/(f0+f)^β.
    scene_power_f0_cyc_per_um : float
        Low-freq offset (avoid 1/0 at DC).

    Returns
    -------
    (loss_scalar, info_dict).
    info has 'mtf_eff_c'[3, K] per field point and 'R_c'[3, K], 'S_s'[K].
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    device = engine.device_
    sensor_pitch = float(engine.spec.sensor_plane.pitch_um)

    field_weight = normalize_field_weights(field_points, field_weight, device)

    cw = channel_weights.to(device).to(torch.float32)
    if cw.shape != (3, n_wl):
        raise ValueError(f"channel_weights expected [3, {n_wl}], got {tuple(cw.shape)}")
    cw_norm = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-30)              # [3, N_λ]

    f_t = freqs_cyc_per_um.to(device).to(torch.float32)                       # [K]

    # Scene power S_s(f) = scene_power_amplitude / (f0 + f)^β  — shape [K]
    # V20: scene_power_amplitude (default 1.0) lets us scale signal PSD to match
    # physical units, e.g. 1e6 to enter signal-dominated regime where |H|² >> R_c
    # for relevant freqs. Default 1.0 reproduces V18 behavior.
    S_s = scene_power_amplitude / (scene_power_f0_cyc_per_um + f_t).clamp_min(eps).pow(scene_power_beta)
    # Per-channel noise variance (white): σ²_read + N_photon_c
    ppc = photon_per_channel.to(device).to(torch.float32)                     # [3]
    sigma2_n_c = (read_noise_e ** 2) + ppc                                    # [3]
    # R_c(f) = sigma2_n_c / S_s(f) — broadcast to [3, K]
    R_c = sigma2_n_c.view(3, 1) / S_s.view(1, K).clamp_min(eps)

    info: dict[str, torch.Tensor] = {
        "freqs_cyc_per_um": f_t.detach(),
        "S_s": S_s.detach(),
        "R_c": R_c.detach(),
        "photon_per_channel": ppc.detach(),
    }

    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map,
        )                                                                      # [N_λ, Hs, Ws]

        # Channel-effective PSF: weighted sum over λ (DC-normalized via cw_norm)
        psf_eff = torch.einsum("cl,lhw->chw", cw_norm, spec_irr)              # [3, Hs, Ws]

        # |H_c(f)| via FFT magnitude, DC-normalized
        otf = torch.fft.fftshift(torch.fft.fft2(psf_eff, dim=(-2, -1)), dim=(-2, -1))
        mtf_2d = otf.abs()
        Hs, Ws = psf_eff.shape[-2], psf_eff.shape[-1]
        dc = mtf_2d[:, Hs // 2, Ws // 2].clamp_min(1e-30).view(3, 1, 1)
        mtf_2d = mtf_2d / dc                                                   # [3, Hs, Ws]

        # Radial sampling at requested freqs (Gaussian soft bin) — same as A1
        fy = torch.fft.fftshift(torch.fft.fftfreq(Hs, d=sensor_pitch, device=device)).to(mtf_2d.dtype)
        fx = torch.fft.fftshift(torch.fft.fftfreq(Ws, d=sensor_pitch, device=device)).to(mtf_2d.dtype)
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
        rr = torch.sqrt(fxx * fxx + fyy * fyy)
        sigma = float((fy[1] - fy[0]).item())
        f_targets = f_t.view(1, K, 1, 1)
        diff = rr.unsqueeze(0).unsqueeze(0) - f_targets
        weights = torch.exp(-0.5 * (diff / sigma).square())
        wnorm = weights.sum(dim=(-2, -1)).clamp_min(1e-30)                     # [1, K]
        mtf_eff_c = (mtf_2d.unsqueeze(1) * weights).sum(dim=(-2, -1)) / wnorm  # [3, K]
        info[fp.name] = mtf_eff_c.detach()

        if dense2d:
            # v27 fix (2026-06-13): integrate over the FULL 2D band instead of
            # K radial point samples. Point sampling leaves ~85% of the band
            # unobserved, and a high-DOF width map exploits the gaps by parking
            # deconvolution-hostile ghost energy between samples (seen as
            # lambda-dependent periodic image replicas at d500). Dense
            # integration leaves no blind frequencies; mtf_2d already exists,
            # so the cost is unchanged.
            band = rr <= float(f_t.max())
            S2 = scene_power_amplitude / (scene_power_f0_cyc_per_um + rr).clamp_min(eps).pow(scene_power_beta)
            R2 = sigma2_n_c.view(3, 1, 1) / S2.unsqueeze(0).clamp_min(eps)
            integrand2 = S2.unsqueeze(0) * R2 / (mtf_2d.square() + R2).clamp_min(eps)
            per_field = integrand2[:, band].mean()
        else:
            # Wiener MSE integrand:  S_s(f) · R_c(f) / (|H_c(f)|² + R_c(f))
            denom = mtf_eff_c.pow(2) + R_c                                     # [3, K]
            integrand = S_s.view(1, K) * R_c / denom.clamp_min(eps)            # [3, K]
            per_field = integrand.sum() / (3 * K)                              # scalar (mean)
        total = total + field_weight[fi] * per_field

    return total, info


def _channel_effective_mtf_at_freqs(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_pt: FieldPoint,
    cw_norm: torch.Tensor,
    extras: bool = False,
):
    """[3, K] DC-normalized channel-effective MTF at radial freqs (Gaussian bin).

    Same sampling scheme as ``wiener_recon_mse_loss`` (kept separate so the
    v-series loss stays byte-identical). With ``extras=True`` also returns
    the full 2D MTF map and the radial-frequency map (for dense2d losses).
    """
    device = engine.device_
    K = int(freqs_cyc_per_um.numel())
    sensor_pitch = float(engine.spec.sensor_plane.pitch_um)
    batch = build_field_object_batch(engine, field_pt)
    spec_irr = engine.forward_optics_from_object_batch(
        batch, width_map_override=width_map,
    )
    psf_eff = torch.einsum("cl,lhw->chw", cw_norm, spec_irr)                  # [3, Hs, Ws]
    otf = torch.fft.fftshift(torch.fft.fft2(psf_eff, dim=(-2, -1)), dim=(-2, -1))
    mtf_2d = otf.abs()
    Hs, Ws = psf_eff.shape[-2], psf_eff.shape[-1]
    dc = mtf_2d[:, Hs // 2, Ws // 2].clamp_min(1e-30).view(3, 1, 1)
    mtf_2d = mtf_2d / dc
    fy = torch.fft.fftshift(torch.fft.fftfreq(Hs, d=sensor_pitch, device=device)).to(mtf_2d.dtype)
    fx = torch.fft.fftshift(torch.fft.fftfreq(Ws, d=sensor_pitch, device=device)).to(mtf_2d.dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(fxx * fxx + fyy * fyy)
    sigma = float((fy[1] - fy[0]).item())
    f_targets = freqs_cyc_per_um.to(device).to(mtf_2d.dtype).view(1, K, 1, 1)
    diff = rr.unsqueeze(0).unsqueeze(0) - f_targets
    weights = torch.exp(-0.5 * (diff / sigma).square())
    wnorm = weights.sum(dim=(-2, -1)).clamp_min(1e-30)
    mtf_eff_c = (mtf_2d.unsqueeze(1) * weights).sum(dim=(-2, -1)) / wnorm     # [3, K]
    if extras:
        return mtf_eff_c, mtf_2d, rr
    return mtf_eff_c


def multichannel_wiener_mse_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    channel_weights: torch.Tensor,
    photon_per_channel: torch.Tensor,
    read_noise_e: float,
    scene_power_beta: float = 2.0,
    scene_power_f0_cyc_per_um: float = 0.05,
    scene_power_amplitude: float = 1.0,
    channel_corr_adj: float = 0.9,
    channel_corr_far: float = 0.8,
    aperture_diameter_um: float,
    f_um: float,
    field_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
    dense2d: bool = False,
) -> tuple[torch.Tensor, dict]:
    """M1 — joint 3-channel MMSE (multichannel Wiener) closed form.

    Natural scenes share spatial structure across color channels, so the
    scene covariance is S(f) = S_s(f) * C with
        C = [[1, a, b], [a, 1, a], [b, a, 1]]   (a=RG/GB, b=RB correlation).
    Observation y_c = H_c s_c + n_c with diagonal H (channel-effective MTF).
    The linear-MMSE (multichannel Wiener) residual is

        E(f) = tr[ S - S H (H S H + N)^{-1} H S ]

    which lets a strong G-channel MTF substitute for an R/B null. With
    C = I this reduces exactly to ``wiener_recon_mse_loss``.
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    device = engine.device_
    field_weight = normalize_field_weights(field_points, field_weight, device)
    cw = channel_weights.to(device).to(torch.float32)
    if cw.shape != (3, n_wl):
        raise ValueError(f"channel_weights expected [3, {n_wl}], got {tuple(cw.shape)}")
    cw_norm = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-30)

    f_t = freqs_cyc_per_um.to(device).to(torch.float32)
    S_s = scene_power_amplitude / (scene_power_f0_cyc_per_um + f_t).clamp_min(eps).pow(scene_power_beta)
    ppc = photon_per_channel.to(device).to(torch.float32)
    sigma2_n_c = (read_noise_e ** 2) + ppc                                    # [3]
    a, b = float(channel_corr_adj), float(channel_corr_far)
    C = torch.tensor([[1.0, a, b], [a, 1.0, a], [b, a, 1.0]],
                     dtype=torch.float32, device=device)                      # [3, 3]
    S = S_s.view(K, 1, 1) * C.view(1, 3, 3)                                   # [K, 3, 3]
    N = torch.diag(sigma2_n_c).view(1, 3, 3)                                  # [1, 3, 3]
    tr_S = S.diagonal(dim1=-2, dim2=-1).sum(-1)                               # [K]

    info: dict[str, torch.Tensor] = {
        "freqs_cyc_per_um": f_t.detach(), "S_s": S_s.detach(),
        "channel_corr": C.detach(),
    }
    total = 0.0
    for fi, fp in enumerate(field_points):
        mtf_eff_c, mtf_2d, rr = _channel_effective_mtf_at_freqs(
            engine, width_map, f_t, fp, cw_norm, extras=True)
        info[fp.name] = mtf_eff_c.detach()
        if dense2d:
            # full-band 2D integration — same no-blind-frequency rationale as
            # wiener_recon_mse_loss (the K-point sampling is exploitable by
            # high-DOF maps). ~34k band bins -> batched 3x3 solves, cheap.
            band = rr <= float(f_t.max())
            S_b = (scene_power_amplitude
                   / (scene_power_f0_cyc_per_um + rr[band]).clamp_min(eps)
                   .pow(scene_power_beta))                                    # [Nb]
            Hb = torch.diag_embed(mtf_2d[:, band].transpose(0, 1))            # [Nb,3,3]
            Sm = S_b.view(-1, 1, 1) * C.view(1, 3, 3)
            Bm = Hb @ Sm
            Am = Bm @ Hb + N
            Xm = torch.linalg.solve(Am, Bm)
            E_b = (Sm.diagonal(dim1=-2, dim2=-1).sum(-1)
                   - (Xm * Bm).sum(dim=(-2, -1)))                             # [Nb]
            per_field = E_b.mean() / 3.0
        else:
            H = torch.diag_embed(mtf_eff_c.transpose(0, 1))                   # [K, 3, 3]
            B = H @ S                                                         # M S, [K, 3, 3]
            A = B @ H + N                                                     # H S H + N
            X = torch.linalg.solve(A, B)                                      # A^{-1} H S
            # tr(S H A^{-1} H S) = tr(B^T A^{-1} B) = sum(X ⊙ B)
            E_f = tr_S - (X * B).sum(dim=(-2, -1))                            # [K]
            per_field = E_f.sum() / (3 * K)
        total = total + field_weight[fi] * per_field
    return total, info


CAPACITY_WEIGHT_LEGACY = "legacy_uniform"
CAPACITY_WEIGHT_RADIAL_2PI_F = "radial_2pi_f"


def _canonical_capacity_weighting(weighting: str) -> str:
    aliases = {
        "legacy_uniform": CAPACITY_WEIGHT_LEGACY,
        "legacy-uniform": CAPACITY_WEIGHT_LEGACY,
        "uniform": CAPACITY_WEIGHT_LEGACY,
        "radial_2pi_f": CAPACITY_WEIGHT_RADIAL_2PI_F,
        "radial-2pi-f": CAPACITY_WEIGHT_RADIAL_2PI_F,
        "2pi_f": CAPACITY_WEIGHT_RADIAL_2PI_F,
    }
    try:
        return aliases[str(weighting)]
    except KeyError as exc:
        raise ValueError(
            f"unknown sparse capacity weighting {weighting!r}; expected "
            "legacy_uniform or radial_2pi_f"
        ) from exc


def capacity_sparse_frequency_weights(
    freqs_cyc_per_um: torch.Tensor,
    weighting: str = CAPACITY_WEIGHT_LEGACY,
) -> torch.Tensor:
    """Unit-sum weights for sparse radial capacity samples.

    ``legacy_uniform`` reproduces the old mean over sampled radii.
    ``radial_2pi_f`` uses trapezoidal ``2π f Δf`` weights, including the
    polar frequency-plane Jacobian and nonuniform radial-bin widths. Unit-sum
    normalization keeps the objective a mean in bits/bin while changing only
    relative sample importance. The unnormalized weights have units µm⁻².
    """
    mode = _canonical_capacity_weighting(weighting)
    f = freqs_cyc_per_um.to(torch.float32)
    if f.ndim != 1 or f.numel() == 0:
        raise ValueError("freqs_cyc_per_um must be a non-empty 1D tensor")
    if not bool(torch.isfinite(f).all()) or bool((f < 0).any()):
        raise ValueError("capacity frequencies must be finite and non-negative")
    if mode == CAPACITY_WEIGHT_LEGACY:
        weights = torch.ones_like(f)
    else:
        if f.numel() < 2:
            raise ValueError("radial_2pi_f weighting requires at least two frequencies")
        df = f[1:] - f[:-1]
        if bool((df <= 0).any()):
            raise ValueError(
                "radial_2pi_f frequencies must be strictly increasing"
            )
        # Trapezoid coefficients for integral g(f)df: endpoints receive half
        # their adjacent interval; interiors receive half of both intervals.
        delta_f = torch.empty_like(f)
        delta_f[0] = 0.5 * df[0]
        delta_f[-1] = 0.5 * df[-1]
        if f.numel() > 2:
            delta_f[1:-1] = 0.5 * (df[:-1] + df[1:])
        weights = 2.0 * math.pi * f * delta_f
    denom = weights.sum()
    if float(denom) <= 0.0:
        raise ValueError("radial_2pi_f weighting requires at least one positive frequency")
    return weights / denom


def capacity_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    channel_weights: torch.Tensor,
    photon_per_channel: torch.Tensor,
    read_noise_e: float,
    scene_power_beta: float = 2.0,
    scene_power_f0_cyc_per_um: float = 0.05,
    scene_power_amplitude: float = 1.0,
    aperture_diameter_um: float,
    f_um: float,
    field_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
    dense2d: bool = False,
    sparse_frequency_weighting: str = CAPACITY_WEIGHT_LEGACY,
) -> tuple[torch.Tensor, dict]:
    """M2 — negative Shannon capacity of the linear-Gaussian imaging channel.

        L = - mean_{c, f} log2(1 + S_s(f) |H_c(f)|^2 / sigma^2_c)

    Water-filling-style objective: log's diminishing returns spread MTF
    across the band instead of concentrating it where MSE gains are largest.
    Dense 2D integration is recommended for optimization because it has no
    blind frequencies. For legacy sparse runs, ``sparse_frequency_weighting``
    explicitly selects the historical uniform-radius mean or a ``2πf`` polar
    area weighting.
    """
    n_wl = int(engine.wavelengths_um.numel())
    K = int(freqs_cyc_per_um.numel())
    device = engine.device_
    field_weight = normalize_field_weights(field_points, field_weight, device)
    cw = channel_weights.to(device).to(torch.float32)
    if cw.shape != (3, n_wl):
        raise ValueError(f"channel_weights expected [3, {n_wl}], got {tuple(cw.shape)}")
    cw_norm = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-30)

    f_t = freqs_cyc_per_um.to(device).to(torch.float32)
    sparse_weighting_mode = _canonical_capacity_weighting(
        sparse_frequency_weighting,
    )
    sparse_freq_weight = capacity_sparse_frequency_weights(
        f_t, sparse_weighting_mode,
    )
    S_s = scene_power_amplitude / (scene_power_f0_cyc_per_um + f_t).clamp_min(eps).pow(scene_power_beta)
    ppc = photon_per_channel.to(device).to(torch.float32)
    sigma2_n_c = (read_noise_e ** 2) + ppc                                    # [3]

    info: dict[str, torch.Tensor] = {
        "freqs_cyc_per_um": f_t.detach(), "S_s": S_s.detach(),
        "sparse_frequency_weight": sparse_freq_weight.detach(),
    }
    total = 0.0
    for fi, fp in enumerate(field_points):
        mtf_eff_c, mtf_2d, rr = _channel_effective_mtf_at_freqs(
            engine, width_map, f_t, fp, cw_norm, extras=True)
        info[fp.name] = mtf_eff_c.detach()
        if dense2d:
            band = rr <= float(f_t.max())
            S_b = (scene_power_amplitude
                   / (scene_power_f0_cyc_per_um + rr[band]).clamp_min(eps)
                   .pow(scene_power_beta))                                    # [Nb]
            snr = (S_b.view(1, -1) * mtf_2d[:, band].square()
                   / sigma2_n_c.view(3, 1))
        else:
            snr = S_s.view(1, K) * mtf_eff_c.pow(2) / sigma2_n_c.view(3, 1)
        cap_bins = torch.log2(1.0 + snr)
        if dense2d or sparse_weighting_mode == CAPACITY_WEIGHT_LEGACY:
            # Keep the default sparse path byte-for-byte equivalent to the
            # historical ``torch.log2(...).mean()`` reduction.
            cap = cap_bins.mean()
        else:
            cap = (cap_bins * sparse_freq_weight.view(1, K)).sum() / 3.0
        total = total + field_weight[fi] * (-cap)
    return total, info


def _mimo_scene_basis(wavelengths_um: torch.Tensor) -> torch.Tensor:
    """RGB->spectral BOX binning b_k(l) as in stage5_e2e_pipeline_6wl.py.

    Each wavelength maps to exactly one scene color (sum_k b_k(l) = 1):
      b_R(l)=1 for l>590 nm, b_G(l)=1 for 490<l<=590 nm, b_B(l)=1 for l<=490 nm.
    Returns [3, N_l] in (R, G, B) channel order matching cfa_transmission.
    """
    wl_nm = wavelengths_um.to(torch.float32) * 1000.0
    b = torch.zeros(3, wl_nm.numel(), dtype=torch.float32, device=wavelengths_um.device)
    b[0] = (wl_nm > 590.0).to(torch.float32)                     # R
    b[1] = ((wl_nm > 490.0) & (wl_nm <= 590.0)).to(torch.float32)  # G
    b[2] = (wl_nm <= 490.0).to(torch.float32)                    # B
    return b


def _mimo_logdet(M, s_s, C, sigma2_n_c, jitter):
    """log2 det( I3 + Sigma_n^-1 M (S_s C) M^H + jitter I ) per freq bin.

    M : [Nb, 3, 3] complex mixing matrix (a=sensor channel, k=scene color)
    s_s : [Nb] real scene power; C : [3, 3] real color correlation;
    sigma2_n_c : [3] real per-channel noise variance.
    Sigma_n^-1 M (S_s C) M^H has real positive eigenvalues (D^-1 x Hermitian-PD),
    so the log|det| returned by slogdet is the real capacity density.
    """
    cdt = M.dtype
    SC = (s_s.view(-1, 1, 1) * C.view(1, 3, 3)).to(cdt)          # [Nb,3,3]
    Mh = M.conj().transpose(-2, -1)
    sig = M @ SC @ Mh                                            # sensor signal cov
    Dinv = torch.diag(1.0 / sigma2_n_c).to(cdt)                 # Sigma_n^-1
    I3 = torch.eye(3, dtype=cdt, device=M.device).unsqueeze(0)
    A = I3 + Dinv.unsqueeze(0) @ sig + jitter * I3
    _sign, logabsdet = torch.linalg.slogdet(A)
    return logabsdet / math.log(2.0)                            # [Nb] real


def mimo_capacity_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    *,
    channel_weights: torch.Tensor,        # [3, N_l] sensor CFA*QE weights (positive)
    photon_per_channel: torch.Tensor,      # [3]
    read_noise_e: float,
    scene_power_beta: float = 2.0,
    scene_power_f0_cyc_per_um: float = 0.05,
    scene_power_amplitude: float = 1.0,
    channel_corr_adj: float = 0.9,
    channel_corr_far: float = 0.8,
    aperture_diameter_um: float,
    f_um: float,
    field_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
    dense2d: bool = False,
    jitter: float = 1e-9,
) -> tuple[torch.Tensor, dict]:
    """v31 -- negative field-integrated FULL 3x3 MIMO Shannon information.

    Replaces the per-channel DIAGONAL capacity (``capacity_loss``) with the
    full color-to-channel MIMO log-det, which captures measurement-side
    spectral crosstalk through the Bayer-CFA passband overlap:

        M_ak(nu) = sum_l w_a(l) b_k(l) OTF_l(nu)         (3x3 complex)
        I(nu)    = log2 det( I3 + Sigma_n^-1 M (S_s C) M^H )
        L        = - mean_field mean_nu I(nu)

    ``w_a`` = per-channel-normalized ``channel_weights`` (CFA*QE, sensor side,
    same cw/cw.sum(dim=1) convention as capacity_loss); ``b_k`` = the stage5
    RGB->spectral box binning (scene side); per-row DC normalization mirrors
    capacity_loss (each sensor row divided by its own DC). Per-lambda COMPLEX
    OTFs are kept (never collapsed to per-channel magnitude before mixing).

    The frequency measure is the 2D band |nu| <= max(freqs) (uniform bin mean,
    the physically correct frequency-plane integral, consistent with mimo_eval
    and the d500 --dense runs); ``dense2d`` is accepted for signature parity
    with the other losses but the 2D band is always used (radial complex
    averaging cancels phase for off-axis fields). Field weights use the same
    normalized area/polar quadrature as the other field-integrated objectives.
    """
    n_wl = int(engine.wavelengths_um.numel())
    device = engine.device_
    sensor_pitch = float(engine.spec.sensor_plane.pitch_um)
    field_weight = normalize_field_weights(field_points, field_weight, device)
    cw = channel_weights.to(device).to(torch.float32)
    if cw.shape != (3, n_wl):
        raise ValueError(f"channel_weights expected [3, {n_wl}], got {tuple(cw.shape)}")
    cw_norm = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-30)             # w_a(l) [3,N_l]
    basis = _mimo_scene_basis(engine.wavelengths_um)                        # b_k(l) [3,N_l]
    ppc = photon_per_channel.to(device).to(torch.float32)
    sigma2_n_c = (read_noise_e ** 2) + ppc                                  # [3]
    a, b = float(channel_corr_adj), float(channel_corr_far)
    C = torch.tensor([[1.0, a, b], [a, 1.0, a], [b, a, 1.0]],
                     dtype=torch.float32, device=device)                    # [3,3]
    f_max = float(freqs_cyc_per_um.to(device).max())

    info: dict[str, torch.Tensor] = {
        "freqs_cyc_per_um": freqs_cyc_per_um.to(device).detach(),
        "scene_basis": basis.detach(), "channel_corr": C.detach(),
    }
    total = 0.0
    for fi, fp in enumerate(field_points):
        batch = build_field_object_batch(engine, fp)
        spec_irr = engine.forward_optics_from_object_batch(
            batch, width_map_override=width_map)                            # [N_l, Hs, Ws] real
        otf = torch.fft.fftshift(torch.fft.fft2(spec_irr, dim=(-2, -1)),
                                 dim=(-2, -1))                              # [N_l,Hs,Ws] complex
        Hs, Ws = spec_irr.shape[-2], spec_irr.shape[-1]
        fy = torch.fft.fftshift(torch.fft.fftfreq(Hs, d=sensor_pitch, device=device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(Ws, d=sensor_pitch, device=device))
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
        rr = torch.sqrt(fxx * fxx + fyy * fyy)                              # [Hs,Ws]
        band = rr <= f_max
        otf_band = otf[:, band]                                            # [N_l, Nb] complex
        otf_dc = otf[:, Hs // 2, Ws // 2].real                            # [N_l] PSF power
        s_s = (scene_power_amplitude
               / (scene_power_f0_cyc_per_um + rr[band]).clamp_min(eps)
               .pow(scene_power_beta))                                     # [Nb]
        # M[nb,a,k] = sum_l w_a(l) b_k(l) OTF_l(nu_nb)
        M = torch.einsum("al,kl,lb->bak", cw_norm.to(otf_band.dtype),
                         basis.to(otf_band.dtype), otf_band)               # [Nb,3,3]
        dc_row = (cw_norm @ otf_dc).clamp_min(1e-30)                       # [3] = sum_k M_ak(0)
        M = M / dc_row.view(1, 3, 1)
        i_nu = _mimo_logdet(M, s_s.to(torch.float32), C, sigma2_n_c, jitter)
        per_field = i_nu.mean()
        info[fp.name] = per_field.detach()
        total = total + field_weight[fi] * (-per_field)
    return total, info


def deployment_wiener_mse_tile_loss(
    engine: MetalensImagingEngine,
    width_map: torch.Tensor,
    tile_ij: tuple[int, int],
    *,
    n_tiles: int = 8,
    channel_weights: torch.Tensor,
    photon_per_channel: torch.Tensor,
    read_noise_e: float,
    scene_power_beta: float = 2.0,
    scene_power_f0_cyc_per_um: float = 0.05,
    scene_power_amplitude: float = 1.0e6,
    deploy_reg: float = 0.1,
    n_freq: int = 16,
    obj_dist_um: float,
    magnification: float,
    sensor_size_um: float,
    eps: float = 1e-12,
    dense2d: bool = True,
) -> torch.Tensor:
    """M4 — expected post-recon MSE of ONE deployment Wiener tile.

    Unlike ``wiener_recon_mse_loss`` (ideal matched Wiener with prior S0),
    this models the *deployed* filter G = H*/(|H|^2 + reg) with the stage5
    regularization, and evaluates the mismatched-filter error

        E(f) = S_s(f) |1 - G(f) H(f)|^2 + sigma^2 |G(f)|^2

    under the signal-dominated prior (the v25 lesson: the prior weighting
    stays S0=1e6-scale; only the *filter* uses the deployment reg).
    The tile PSF is the true spatially-varying PSF at the tile center
    (full off-axis forward), channel-aggregated with the real CFA weights
    and box-integrated to the pixel grid — i.e. the quantity the deployed
    8x8 spatially-varying Wiener actually sees.

    Returns a scalar; caller backpropagates per tile (graph freed each call)
    so 64 tiles never coexist in memory.
    """
    device = engine.device_
    n_wl = int(engine.wavelengths_um.numel())
    cw = channel_weights.to(device).to(torch.float32)
    cw_norm = cw / cw.sum(dim=1, keepdim=True).clamp_min(1e-30)

    # tile center -> object-plane field angle (paraxial mapping, image
    # inversion irrelevant for PSF statistics)
    i, j = tile_ij
    x_img = ((j + 0.5) / n_tiles - 0.5) * sensor_size_um
    y_img = ((i + 0.5) / n_tiles - 0.5) * sensor_size_um
    x_obj = x_img / magnification
    y_obj = y_img / magnification
    r_obj = math.sqrt(x_obj * x_obj + y_obj * y_obj)
    theta = math.degrees(math.atan(r_obj / obj_dist_um))
    az = math.degrees(math.atan2(y_obj, x_obj))
    fp = FieldPoint(name=f"tile_{i}_{j}", theta_deg=theta, azimuth_deg=az)

    batch = build_field_object_batch(engine, fp)
    spec_irr = engine.forward_optics_from_object_batch(
        batch, width_map_override=width_map)                                  # [N_λ, Hs, Ws]
    psf_eff = torch.einsum("cl,lhw->chw", cw_norm, spec_irr)                  # [3, Hs, Ws]
    # box-integrate sensor grid -> pixel grid (deployment sees pixel-domain PSF)
    Hs = psf_eff.shape[-1]
    px = engine.spec.pixel_grid_shape[0]
    k = max(Hs // px, 1)
    psf_pix = F.avg_pool2d(psf_eff.unsqueeze(0), kernel_size=k, stride=k)[0]  # [3, px, px]
    psf_pix = psf_pix / psf_pix.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)

    otf = torch.fft.fftshift(torch.fft.fft2(psf_pix, dim=(-2, -1)), dim=(-2, -1))
    mtf_2d = otf.abs()                                                        # [3, px, px]

    pixel_pitch = sensor_size_um / px
    fy = torch.fft.fftshift(torch.fft.fftfreq(px, d=pixel_pitch, device=device)).to(mtf_2d.dtype)
    fyy, fxx = torch.meshgrid(fy, fy, indexing="ij")
    rr = torch.sqrt(fxx * fxx + fyy * fyy)
    sigma = float((fy[1] - fy[0]).item())
    f_nyq = 0.5 / pixel_pitch
    f_t = torch.linspace(0.05 * f_nyq, f_nyq, n_freq, device=device,
                         dtype=mtf_2d.dtype)
    ppc = photon_per_channel.to(device).to(torch.float32)
    sigma2 = (read_noise_e ** 2) + ppc                                        # [3]

    if dense2d:
        # full-band 2D integration — no loss-blind frequencies (see
        # wiener_recon_mse_loss dense2d note)
        band = rr <= f_nyq
        S2 = scene_power_amplitude / (scene_power_f0_cyc_per_um + rr).clamp_min(eps).pow(scene_power_beta)
        G2 = mtf_2d / (mtf_2d.square() + deploy_reg)
        resid2 = (1.0 - G2 * mtf_2d).square() * S2.unsqueeze(0)
        noise2 = sigma2.view(3, 1, 1) * G2.square()
        return (resid2 + noise2)[:, band].mean()

    diff = rr.unsqueeze(0).unsqueeze(0) - f_t.view(1, n_freq, 1, 1)
    w = torch.exp(-0.5 * (diff / sigma).square())
    wn = w.sum(dim=(-2, -1)).clamp_min(1e-30)
    H = (mtf_2d.unsqueeze(1) * w).sum(dim=(-2, -1)) / wn                      # [3, K]

    S_s = scene_power_amplitude / (scene_power_f0_cyc_per_um + f_t).clamp_min(eps).pow(scene_power_beta)

    H2 = H.square()
    G = H / (H2 + deploy_reg)                                                 # deployed filter (real, matched-phase)
    resid = (1.0 - G * H).square() * S_s.view(1, -1)
    noise = sigma2.view(3, 1) * G.square()
    return (resid + noise).mean()


def optimize_widthmap_deploy(
    engine: MetalensImagingEngine,
    param: nn.Module,
    *,
    n_steps: int,
    lr: float,
    lr_min: float = 1e-5,
    tiles_per_step: int = 16,
    n_tiles: int = 8,
    weight_smooth: float = 1e-3,
    weight_barrier: float = 10.0,
    grad_clip: float = 1.0,
    log_every: int = 10,
    seed: int = 0,
    loss_kwargs: dict,
) -> dict:
    """M4 driver loop — stochastic-tile gradient accumulation.

    Each step samples ``tiles_per_step`` of the 8x8 deployment tiles
    (unbiased estimator of the tile-average MSE) and backpropagates each
    tile separately so only one forward graph is alive at a time.
    """
    g = torch.Generator().manual_seed(seed)
    all_tiles = [(i, j) for i in range(n_tiles) for j in range(n_tiles)]
    optimizer = torch.optim.Adam(param.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr_min)
    has_barrier = hasattr(param, "fab_barrier_loss")
    history: list[dict] = []
    best_loss = float("inf")
    best_state = None

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        idx = torch.randperm(len(all_tiles), generator=g)[:tiles_per_step]
        step_loss = 0.0
        for t in idx.tolist():
            width_map = param.expand()
            lt = deployment_wiener_mse_tile_loss(
                engine, width_map, all_tiles[t], n_tiles=n_tiles, **loss_kwargs)
            (lt / tiles_per_step).backward()
            step_loss += float(lt.item()) / tiles_per_step
        reg = weight_smooth * param.smoothness_loss()
        if has_barrier and weight_barrier > 0.0:
            reg = reg + weight_barrier * param.fab_barrier_loss()
        reg.backward()
        step_loss += float(reg.item())
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(param.parameters(), grad_clip)

        # snapshot before the update so loss and state correspond; step 0 covers the warm start
        if step_loss < best_loss:
            best_loss = step_loss
            best_state = {k: v.detach().clone() for k, v in param.state_dict().items()}

        optimizer.step()
        scheduler.step()
        if (step + 1) % log_every == 0 or step == 0:
            print(f"[deploy] step {step + 1:04d}/{n_steps}  loss={step_loss:.5e}")
        history.append({"step": step + 1, "loss": step_loss})

    if best_state is not None:
        param.load_state_dict(best_state)
    return {"best_loss": best_loss, "history": history, "param": param}


def build_channel_weights_from_engine(engine: MetalensImagingEngine) -> torch.Tensor:
    """Helper: build channel_weights tensor from engine.cfa_transmission × qe × λ."""
    cfa = engine.cfa_transmission.squeeze(-1).squeeze(-1)       # [3, N_λ]
    qe = engine.qe.squeeze(-1).squeeze(-1)                       # [3, N_λ]
    wls_um = engine.wavelengths_um                                # [N_λ]
    photon_factor = wls_um / wls_um.max()                        # [N_λ] normalized
    return cfa * qe * photon_factor.view(1, -1)                  # [3, N_λ]


def estimate_photon_per_channel(
    engine: MetalensImagingEngine,
    *,
    scene_illuminance_lux: float = 1000.0,
    exposure_s: float = 0.001,
    scene_illuminant_photons_per_lux_s_per_m2_per_nm: float = 4.0e15,
) -> torch.Tensor:
    """Rough estimate of mean per-pixel photon count for R/G/B under D65-ish.

    Per-pixel photon count for channel c:
        N_c = pixel_area · t_exp · Σ_λ CFA_c(λ) · QE(λ) · I_λ_in_photons(λ) · Δλ

    where I_λ ≈ scene_illuminance_lux · K(λ) (D65 SPD, approximated as flat-ish
    in visible).

    For simplicity, treat I_λ as flat over the test wavelengths and just scale
    by total illuminance. Approximate `K_total = 4e15 photons / (lux · s · m² · nm)`
    averaged across visible — this is a coarse single-number model; calibration
    can be refined later.

    Returns [3] (R, G, B) photon counts.
    """
    pixel_pitch_um = float(engine.spec.pixel_grid_shape[0]) and float(
        engine.spec.aperture_radius_um) and 1.0  # placeholder
    # Use sensor pitch as pixel pitch proxy (per pipeline convention)
    pixel_pitch_um = float(engine.spec.sensor_plane.pitch_um)
    pixel_area_m2 = (pixel_pitch_um * 1e-6) ** 2
    cfa = engine.cfa_transmission.squeeze(-1).squeeze(-1)       # [3, N_λ]
    qe = engine.qe.squeeze(-1).squeeze(-1)                       # [3, N_λ]
    wls_um = engine.wavelengths_um                                # [N_λ]
    # Δλ assumed uniform 30 nm bin (test_wl_nm spacing)
    if wls_um.numel() > 1:
        d_um = float((wls_um[1:] - wls_um[:-1]).mean().item())
    else:
        d_um = 0.030
    d_nm = d_um * 1000.0
    flux_const = (scene_illuminance_lux * exposure_s * pixel_area_m2
                  * scene_illuminant_photons_per_lux_s_per_m2_per_nm * d_nm)
    # photons per channel per pixel
    n_per_channel = (cfa * qe).sum(dim=1) * flux_const           # [3]
    return n_per_channel


# ─────────────────────────────────────────────────────────────────────────────
# Driver helper (Adam loop with periodic logging)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpotMTFOptConfig:
    n_steps: int = 500
    lr: float = 5e-3
    lr_min: float = 1e-5
    cosine_schedule: bool = True
    weight_smooth: float = 1e-3
    weight_barrier: float = 10.0
    log_every: int = 25
    grad_clip: float | None = 1.0


def optimize_widthmap_spot_mtf(
    engine: MetalensImagingEngine,
    param: nn.Module,
    *,
    freqs_cyc_per_um: torch.Tensor,
    field_points: list[FieldPoint],
    field_weight: torch.Tensor | None = None,
    aperture_diameter_um: float,
    f_um: float,
    config: SpotMTFOptConfig | None = None,
    on_step_callback=None,
    use_band_mtf: bool = False,
    use_mtf_volume: bool = False,
    use_cfa_log_mtf: bool = False,
    lambda_weight: torch.Tensor | None = None,
    loss_power: float = 2.0,
    use_channel_effective: bool = False,
    channel_weights: torch.Tensor | None = None,
    # V18: Wiener-MSE mode
    use_wiener_mse: bool = False,
    photon_per_channel: torch.Tensor | None = None,
    read_noise_e: float = 1.5,
    scene_power_beta: float = 2.0,
    scene_power_f0_cyc_per_um: float = 0.05,
    scene_power_amplitude: float = 1.0,    # V20: scale S_s for SNR regime control
    # M1/M2 (v27): multichannel Wiener / capacity objectives
    use_mc_wiener: bool = False,
    use_capacity: bool = False,
    use_mimo: bool = False,
    # P0: throughput-aware, pixel-integrated diagonal sensor information.
    use_sensor_information: bool = False,
    # Diagonal channel-Wiener risk: a MODIFIER of the exact-aopt split path that
    # swaps the per-field functional (diagonal of exact ℰ). Requires the same
    # exact_aopt_protocol calibration setup; not a separate data-loss mode.
    use_diagonal_wiener_risk: bool = False,
    # Exact periodic 2x2 RGGB posterior-MMSE objective.  The protocol mapping
    # contains only the frozen spectral/color tensors; exposure/noise and scene
    # PSD controls intentionally reuse the calibrated sensor arguments below.
    exact_aopt_protocol: dict[str, object] | None = None,
    electron_calibration: float | torch.Tensor | None = None,
    sensor_scene_contrast_rms: float = 0.18,
    sensor_scene_k0_cyc_per_pixel: float = 0.02,
    sensor_scene_beta: float = 2.0,
    sensor_point_radiance_value: float = 1.0e10,
    sensor_fields_per_step: int | None = None,
    sensor_field_seed: int = 20260715,
    sensor_full_validation_every: int = 10,
    channel_corr_adj: float = 0.9,
    channel_corr_far: float = 0.8,
    dense2d: bool = False,
    capacity_sparse_frequency_weighting: str = CAPACITY_WEIGHT_LEGACY,
) -> dict:
    """Adam loop optimizing ``param`` to minimize spot-MTF (or band-MTF) shortfall.

    ``param`` must expose ``.expand() -> Tensor[Hp,Wp]``,
    ``.smoothness_loss() -> Tensor``, and (optionally) ``.fab_barrier_loss()``.
    Engine's width_map_um stays as a static buffer; we feed the param's
    expanded width map via ``width_map_override``.

    If ``use_band_mtf=True``, ``freqs_cyc_per_um`` is interpreted as the
    NORMALIZED freqs (in (0, 1] of f_c(λ)) and ``band_mtf_loss`` is used.
    """
    cfg = config or SpotMTFOptConfig()
    optimizer = torch.optim.Adam(param.parameters(), lr=cfg.lr)
    if cfg.cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_steps, eta_min=cfg.lr_min,
        )
    else:
        scheduler = None
    has_barrier = hasattr(param, "fab_barrier_loss")
    history: list[dict] = []
    loss_trace: list[float] = []
    data_loss_trace: list[float] = []
    gradient_norm_trace: list[float] = []
    data_gradient_norm_trace: list[float] = []
    post_clip_gradient_norm_trace: list[float] = []
    nonzero_gradient_fraction_trace: list[float] = []
    parameter_update_norm_trace: list[float] = []
    sensor_field_indices_trace: list[list[int]] = []
    sensor_full_validation_trace: list[dict[str, float | int | str]] = []
    sensor_terminal_validation: dict[str, float | int | str] | None = None
    initial_exact_total_loss: float | None = None
    best_loss = float("inf")
    best_state = None

    mode_count = sum(bool(flag) for flag in (
        use_band_mtf,
        use_mtf_volume,
        use_cfa_log_mtf,
        use_channel_effective,
        use_wiener_mse,
        use_mc_wiener,
        use_capacity,
        use_mimo,
        use_sensor_information,
        exact_aopt_protocol is not None,
    ))
    if mode_count > 1:
        raise ValueError("select at most one width-map data-loss mode")
    use_split_sensor_objective = (
        use_sensor_information or exact_aopt_protocol is not None
    )
    if use_split_sensor_objective and electron_calibration is None:
        raise ValueError(
            "electron_calibration is required for calibrated sensor objectives"
        )
    if use_diagonal_wiener_risk and exact_aopt_protocol is None:
        raise ValueError(
            "use_diagonal_wiener_risk requires the exact-aopt calibration setup "
            "(exact_aopt_protocol); it swaps only the per-field functional"
        )
    exact_protocol: dict[str, object] | None = None
    if exact_aopt_protocol is not None:
        if not isinstance(exact_aopt_protocol, dict):
            raise TypeError("exact_aopt_protocol must be a dict or None")
        exact_protocol = dict(exact_aopt_protocol)
        required_exact_keys = {
            "scene_spectral_basis",
            "scene_color_covariance",
            "target_color_transform",
        }
        missing_exact_keys = required_exact_keys.difference(exact_protocol)
        if missing_exact_keys:
            missing = ", ".join(sorted(missing_exact_keys))
            raise ValueError(f"exact_aopt_protocol is missing required keys: {missing}")
        reserved_exact_keys = {
            "electron_calibration",
            "read_noise_e",
            "field_weight",
            "scene_contrast_rms",
            "scene_k0_cyc_per_pixel",
            "scene_beta",
            "point_radiance_value",
        }
        collisions = reserved_exact_keys.intersection(exact_protocol)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(
                "exact_aopt_protocol must not override driver-controlled keys: "
                + names
            )
    if sensor_fields_per_step is not None and sensor_fields_per_step < 1:
        raise ValueError("sensor_fields_per_step must be positive or None")
    if sensor_full_validation_every < 1:
        raise ValueError("sensor_full_validation_every must be positive")
    if cfg.n_steps < 1:
        raise ValueError("n_steps must be positive")
    sensor_generator = torch.Generator(device="cpu").manual_seed(
        int(sensor_field_seed)
    )

    for step in range(cfg.n_steps):
        optimizer.zero_grad(set_to_none=True)
        width_map = param.expand()
        if use_split_sensor_objective:
            # Rebuild and backpropagate one field graph at a time. A 5x5
            # square-field rule otherwise retains 25 full optical graphs until
            # one final backward and exceeds the memory budget of a 12 GB GPU.
            # Linearity of differentiation makes this exactly equivalent to a
            # backward through their normalized weighted sum.
            del width_map
            sensor_field_weight = normalize_field_weights(
                field_points, field_weight, next(param.parameters()).device,
            )
            if (
                sensor_fields_per_step is not None
                and sensor_fields_per_step < len(field_points)
            ):
                selected_field_indices = torch.multinomial(
                    sensor_field_weight.detach().cpu(),
                    int(sensor_fields_per_step),
                    replacement=True,
                    generator=sensor_generator,
                ).tolist()
                backward_weights = sensor_field_weight.new_full(
                    (len(selected_field_indices),),
                    1.0 / len(selected_field_indices),
                )
            else:
                selected_field_indices = list(range(len(field_points)))
                backward_weights = sensor_field_weight
            sensor_field_indices_trace.append(selected_field_indices)
            loss_data = sensor_field_weight.new_zeros(())
            info: dict[str, object] = {}
            for sample_index, field_index in enumerate(selected_field_indices):
                field_point = field_points[field_index]
                objective_kwargs = {
                    "electron_calibration": electron_calibration,
                    "read_noise_e": read_noise_e,
                    "field_weight": None,
                    "scene_contrast_rms": sensor_scene_contrast_rms,
                    "scene_k0_cyc_per_pixel": sensor_scene_k0_cyc_per_pixel,
                    "scene_beta": sensor_scene_beta,
                    "point_radiance_value": sensor_point_radiance_value,
                }
                if use_diagonal_wiener_risk:
                    field_loss, field_info = dense_diagonal_wiener_risk_loss(
                        engine,
                        param.expand(),
                        [field_point],
                        **objective_kwargs,
                        relative_source_spectrum=exact_protocol["source_spectrum"],
                    )
                elif exact_protocol is None:
                    field_loss, field_info = dense_sensor_information_loss(
                        engine,
                        param.expand(),
                        [field_point],
                        **objective_kwargs,
                    )
                else:
                    field_loss, field_info = dense_exact_rggb_a_optimal_loss(
                        engine,
                        param.expand(),
                        [field_point],
                        **objective_kwargs,
                        **exact_protocol,
                    )
                weighted_field_loss = backward_weights[sample_index] * field_loss
                if not bool(torch.isfinite(weighted_field_loss).detach()):
                    raise FloatingPointError(
                        "non-finite sensor-information field loss at step "
                        f"{step + 1}, field {field_point.name}"
                    )
                weighted_field_loss.backward()
                loss_data = loss_data + weighted_field_loss.detach()
                info[field_point.name] = field_info[field_point.name]
                if sample_index == 0:
                    for key in (
                        "frequency_layout",
                        "scene_psd_mean",
                        "electron_calibration",
                        "point_radiance_value",
                    ):
                        if key in field_info:
                            info[key] = field_info[key]
            info["field_weight"] = sensor_field_weight.detach()
            if use_diagonal_wiener_risk:
                # A positive per-field risk (minimize), like exact-aopt; reported
                # through the a_optimal_risk channel so finish()/summary/audit
                # handle it identically, with a distinct objective label.
                info["objective"] = "diagonal_channel_wiener_risk"
                info["field_averaged_a_optimal_risk"] = loss_data.detach()
                info["field_averaged_diagonal_wiener_risk"] = loss_data.detach()
            elif exact_protocol is None:
                info["field_averaged_diagonal_bpp"] = (-loss_data).detach()
            else:
                info["objective"] = (
                    "exact_periodic_2x2_RGGB_A_optimal_target_MMSE"
                )
                info["field_averaged_a_optimal_risk"] = loss_data.detach()
            info["sampled_field_indices"] = list(selected_field_indices)
            info["field_gradient_estimator"] = (
                "full_quadrature"
                if len(selected_field_indices) == len(field_points)
                else "unbiased_sampling_with_replacement_p_equals_quadrature_weight"
            )
        elif use_mc_wiener or use_capacity or use_mimo:
            if channel_weights is None or photon_per_channel is None:
                raise ValueError("channel_weights AND photon_per_channel required")
            loss_fn = (mimo_capacity_loss if use_mimo else
                       multichannel_wiener_mse_loss if use_mc_wiener else capacity_loss)
            extra = {}
            if use_mc_wiener or use_mimo:
                extra.update(channel_corr_adj=channel_corr_adj,
                             channel_corr_far=channel_corr_far)
            if use_capacity:
                extra["sparse_frequency_weighting"] = (
                    capacity_sparse_frequency_weighting
                )
            loss_data, info = loss_fn(
                engine, width_map, freqs_cyc_per_um, field_points,
                channel_weights=channel_weights,
                photon_per_channel=photon_per_channel,
                read_noise_e=read_noise_e,
                scene_power_beta=scene_power_beta,
                scene_power_f0_cyc_per_um=scene_power_f0_cyc_per_um,
                scene_power_amplitude=scene_power_amplitude,
                aperture_diameter_um=aperture_diameter_um, f_um=f_um,
                field_weight=field_weight,
                dense2d=dense2d,
                **extra,
            )
        elif use_wiener_mse:
            if channel_weights is None or photon_per_channel is None:
                raise ValueError("channel_weights AND photon_per_channel required for wiener_mse mode")
            loss_data, info = wiener_recon_mse_loss(
                engine, width_map, freqs_cyc_per_um, field_points,
                channel_weights=channel_weights,
                photon_per_channel=photon_per_channel,
                read_noise_e=read_noise_e,
                scene_power_beta=scene_power_beta,
                scene_power_f0_cyc_per_um=scene_power_f0_cyc_per_um,
                scene_power_amplitude=scene_power_amplitude,
                aperture_diameter_um=aperture_diameter_um, f_um=f_um,
                field_weight=field_weight,
                dense2d=dense2d,
            )
        elif use_channel_effective:
            if channel_weights is None:
                raise ValueError("channel_weights required for channel-effective mode")
            loss_data, info = channel_effective_mtf_loss(
                engine, width_map, freqs_cyc_per_um, field_points,
                channel_weights=channel_weights,
                aperture_diameter_um=aperture_diameter_um, f_um=f_um,
                field_weight=field_weight,
                loss_power=loss_power,
            )
        elif use_band_mtf:
            loss_data, info = band_mtf_loss(
                engine, width_map, freqs_cyc_per_um, field_points,
                aperture_diameter_um=aperture_diameter_um, f_um=f_um,
                lambda_weight=lambda_weight,
                field_weight=field_weight,
                loss_power=loss_power,
            )
        elif use_mtf_volume:
            loss_data, info = mtf_volume_loss(
                engine, width_map, freqs_cyc_per_um, field_points,
                field_weight=field_weight,
                lambda_weight=lambda_weight,
            )
        elif use_cfa_log_mtf:
            loss_data, info = cfa_log_mtf_loss(
                engine, width_map, freqs_cyc_per_um, field_points,
                field_weight=field_weight,
            )
        else:
            loss_data, info = spot_mtf_loss(
                engine, width_map, freqs_cyc_per_um, field_points,
                aperture_diameter_um=aperture_diameter_um, f_um=f_um,
                field_weight=field_weight,
            )
        regularization = cfg.weight_smooth * param.smoothness_loss()
        if has_barrier and cfg.weight_barrier > 0.0:
            regularization = regularization + (
                cfg.weight_barrier * param.fab_barrier_loss()
            )
        loss = loss_data + regularization
        if not bool(torch.isfinite(loss).detach()):
            raise FloatingPointError(
                f"non-finite width-map loss at step {step + 1}: {loss.detach()}"
            )
        if use_split_sensor_objective:
            # The field contributions were already accumulated above; only
            # the once-per-step regularizer remains.
            data_grad_square = loss.detach().new_zeros(())
            data_grad_tensors = 0
            for parameter in param.parameters():
                if parameter.grad is None:
                    continue
                data_grad_tensors += 1
                data_grad_square = (
                    data_grad_square + parameter.grad.detach().square().sum()
                )
            if data_grad_tensors == 0:
                raise RuntimeError(
                    "calibrated sensor data loss produced no parameter gradients"
                )
            data_gradient_norm_trace.append(
                float(torch.sqrt(data_grad_square).item())
            )
            if regularization.requires_grad:
                regularization.backward()
        else:
            loss.backward()
        grad_square = loss.detach().new_zeros(())
        gradient_tensors = 0
        gradient_elements = 0
        nonzero_gradient_elements = 0
        for parameter in param.parameters():
            if parameter.grad is None:
                continue
            gradient_tensors += 1
            gradient_elements += parameter.grad.numel()
            nonzero_gradient_elements += int((parameter.grad.detach() != 0).sum())
            if not bool(torch.isfinite(parameter.grad).all().detach()):
                raise FloatingPointError(
                    f"non-finite width-map gradient at step {step + 1}"
                )
            grad_square = grad_square + parameter.grad.detach().square().sum()
        if gradient_tensors == 0:
            raise RuntimeError(
                f"width-map loss produced no parameter gradients at step {step + 1}"
            )
        gradient_norm = float(torch.sqrt(grad_square).item())
        nonzero_gradient_fraction_trace.append(
            nonzero_gradient_elements / gradient_elements
        )
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(param.parameters(), cfg.grad_clip)
        post_clip_square = loss.detach().new_zeros(())
        for parameter in param.parameters():
            if parameter.grad is not None:
                post_clip_square = (
                    post_clip_square + parameter.grad.detach().square().sum()
                )
        post_clip_gradient_norm_trace.append(
            float(torch.sqrt(post_clip_square).item())
        )

        loss_val = float(loss.item())
        data_loss_val = float(loss_data.detach().item())
        loss_trace.append(loss_val)
        data_loss_trace.append(data_loss_val)
        gradient_norm_trace.append(gradient_norm)
        # Snapshot before the update so loss and state correspond. Stochastic
        # field batches are unbiased gradients but biased checkpoint selectors:
        # choosing the lowest sampled loss creates a winner's curse. Select a
        # stochastic sensor run only on periodic exact full-quadrature replay.
        stochastic_sensor_fields = (
            use_split_sensor_objective
            and sensor_fields_per_step is not None
            and sensor_fields_per_step < len(field_points)
        )
        selection_loss = loss_val
        should_select = not stochastic_sensor_fields
        if stochastic_sensor_fields and (
            step == 0
            or (step + 1) % sensor_full_validation_every == 0
            or step + 1 == cfg.n_steps
        ):
            with torch.no_grad():
                full_objective_kwargs = {
                    "electron_calibration": electron_calibration,
                    "read_noise_e": read_noise_e,
                    "field_weight": field_weight,
                    "scene_contrast_rms": sensor_scene_contrast_rms,
                    "scene_k0_cyc_per_pixel": sensor_scene_k0_cyc_per_pixel,
                    "scene_beta": sensor_scene_beta,
                    "point_radiance_value": sensor_point_radiance_value,
                }
                if use_diagonal_wiener_risk:
                    full_data_loss, _ = dense_diagonal_wiener_risk_loss(
                        engine,
                        param.expand(),
                        field_points,
                        **full_objective_kwargs,
                        relative_source_spectrum=exact_protocol["source_spectrum"],
                    )
                elif exact_protocol is None:
                    full_data_loss, _ = dense_sensor_information_loss(
                        engine,
                        param.expand(),
                        field_points,
                        **full_objective_kwargs,
                    )
                else:
                    full_data_loss, _ = dense_exact_rggb_a_optimal_loss(
                        engine,
                        param.expand(),
                        field_points,
                        **full_objective_kwargs,
                        **exact_protocol,
                    )
                full_regularization = cfg.weight_smooth * param.smoothness_loss()
                if has_barrier and cfg.weight_barrier > 0.0:
                    full_regularization = full_regularization + (
                        cfg.weight_barrier * param.fab_barrier_loss()
                    )
                full_total_loss = full_data_loss + full_regularization
            selection_loss = float(full_total_loss)
            sensor_full_validation_trace.append({
                "step": step + 1,
                "updates_completed": step,
                "state": "pre_update",
                "data_loss": float(full_data_loss),
                "total_loss": selection_loss,
            })
            should_select = True
        if use_split_sensor_objective and step == 0:
            initial_exact_total_loss = selection_loss
        if should_select and selection_loss < best_loss:
            best_loss = selection_loss
            best_state = {
                key: value.detach().clone()
                for key, value in param.state_dict().items()
            }

        parameters_before_step = [
            parameter.detach().clone() for parameter in param.parameters()
        ]
        optimizer.step()
        update_square = loss.detach().new_zeros(())
        for parameter, before in zip(param.parameters(), parameters_before_step):
            if not bool(torch.isfinite(parameter).all().detach()):
                raise FloatingPointError(
                    f"non-finite width-map parameter after step {step + 1}"
                )
            update_square = update_square + (parameter.detach() - before).square().sum()
        update_norm = float(torch.sqrt(update_square).item())
        if not math.isfinite(update_norm):
            raise FloatingPointError(
                f"non-finite parameter update norm at step {step + 1}"
            )
        parameter_update_norm_trace.append(update_norm)
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            print(f"[opt] step {step + 1:04d}/{cfg.n_steps}  loss={loss_val:.5e}")
            if use_split_sensor_objective:
                for fp in field_points:
                    if fp.name not in info:
                        continue
                    row = info[fp.name]
                    mean_e = row["mean_electrons"].cpu()
                    if use_diagonal_wiener_risk:
                        metric = (
                            "diag_wiener_risk="
                            f"{float(row['diagonal_wiener_risk']):.6e}"
                        )
                    elif exact_protocol is None:
                        metric = (
                            f"sensor_info={float(row['diagonal_bpp']):.6f} "
                            "bit/raw-px"
                        )
                    else:
                        metric = (
                            "aopt_risk="
                            f"{float(row['a_optimal_risk']):.6e}"
                        )
                    print(
                        f"  {fp.name} | {metric} mean_e(R/G/B)="
                        f"{float(mean_e[0]):.1f}/{float(mean_e[1]):.1f}/"
                        f"{float(mean_e[2]):.1f}"
                    )
                history.append({
                    "step": step + 1,
                    "loss": loss_val,
                    "info": info,
                })
                if on_step_callback is not None:
                    on_step_callback(step, param, info, loss_val)
                continue
            if use_mimo:
                # info[fp.name] is the per-field MIMO information (scalar)
                for fp in field_points:
                    print(f"  {fp.name} | mimo_info={float(info[fp.name]):.4f} bits")
                history.append({"step": step + 1, "loss": loss_val,
                                "info": {k: v.cpu() for k, v in info.items()}})
                if on_step_callback is not None:
                    on_step_callback(step, param, info, loss_val)
                continue
            if use_wiener_mse or use_mc_wiener or use_capacity:
                # Log mean per-channel MTF over freq grid (no target)
                for fp in field_points:
                    mtf = info[fp.name].cpu()  # [3, K]
                    avg = mtf.mean(dim=1)
                    line = "  " + fp.name + " | " + " ".join(
                        f"{['R','G','B'][c]}_meanMTF={avg[c].item():.3f}"
                        for c in range(3))
                    print(line)
                history.append({"step": step + 1, "loss": loss_val,
                                "info": {k: v.cpu() for k, v in info.items()}})
                if on_step_callback is not None:
                    on_step_callback(step, param, info, loss_val)
                continue
            if use_mtf_volume:
                # Log the mean DC-normalized optical MTF (the volume maximized).
                for fp in field_points:
                    mtf = info[fp.name].cpu()  # [N_λ, K]
                    print(f"  {fp.name} | meanMTF(vol)={float(mtf.mean()):.4f}")
                history.append({"step": step + 1, "loss": loss_val,
                                "info": {k: v.cpu() for k, v in info.items()}})
                if on_step_callback is not None:
                    on_step_callback(step, param, info, loss_val)
                continue
            if use_cfa_log_mtf:
                # Log per-channel CFA-weighted MTF volume V_R/V_G/V_B.
                for fp in field_points:
                    vc = info[fp.name + "_Vc"].cpu()  # [3]
                    print(f"  {fp.name} | V_R={vc[0]:.4f} V_G={vc[1]:.4f} "
                          f"V_B={vc[2]:.4f}")
                history.append({"step": step + 1, "loss": loss_val,
                                "info": {k: v.cpu() for k, v in info.items()}})
                if on_step_callback is not None:
                    on_step_callback(step, param, info, loss_val)
                continue
            target_key = "target_channel" if use_channel_effective else "target"
            for fp in field_points:
                mtf = info[fp.name].cpu()
                tgt = info[target_key].cpu()
                ratio = (mtf / tgt.clamp_min(1e-6)).clamp(0, 1)
                if use_channel_effective:
                    avg = ratio.mean(dim=1)  # [3] avg over K freqs
                    line = "  " + fp.name + " | " + " ".join(
                        f"{['R','G','B'][c]}={avg[c].item():.3f}" for c in range(3))
                elif use_band_mtf:
                    avg_per_lambda = ratio.mean(dim=1)
                    line = "  " + fp.name + " | " + " ".join(
                        f"λ{li}={avg_per_lambda[li].item():.3f}"
                        for li in range(mtf.shape[0]))
                else:
                    line = "  " + fp.name + " | "
                    for li in range(mtf.shape[0]):
                        rs = ", ".join(f"{ratio[li, k].item():.3f}" for k in range(mtf.shape[1]))
                        line += f"λ{li}: [{rs}]  "
                print(line)
            history.append({"step": step + 1, "loss": loss_val,
                            "info": {k: v.cpu() for k, v in info.items()}})
            if on_step_callback is not None:
                on_step_callback(step, param, info, loss_val)

    # Every in-loop objective is evaluated before optimizer.step(), so the
    # state after update N would otherwise never be eligible for selection.
    # Sensor-information runs close that off-by-one explicitly with one exact
    # full-field, post-update evaluation. This is also the final unbiased
    # checkpoint-selection evaluation for stochastic field sampling.
    if use_split_sensor_objective:
        with torch.no_grad():
            terminal_objective_kwargs = {
                "electron_calibration": electron_calibration,
                "read_noise_e": read_noise_e,
                "field_weight": field_weight,
                "scene_contrast_rms": sensor_scene_contrast_rms,
                "scene_k0_cyc_per_pixel": sensor_scene_k0_cyc_per_pixel,
                "scene_beta": sensor_scene_beta,
                "point_radiance_value": sensor_point_radiance_value,
            }
            if use_diagonal_wiener_risk:
                terminal_data_loss, _ = dense_diagonal_wiener_risk_loss(
                    engine,
                    param.expand(),
                    field_points,
                    **terminal_objective_kwargs,
                    relative_source_spectrum=exact_protocol["source_spectrum"],
                )
            elif exact_protocol is None:
                terminal_data_loss, _ = dense_sensor_information_loss(
                    engine,
                    param.expand(),
                    field_points,
                    **terminal_objective_kwargs,
                )
            else:
                terminal_data_loss, _ = dense_exact_rggb_a_optimal_loss(
                    engine,
                    param.expand(),
                    field_points,
                    **terminal_objective_kwargs,
                    **exact_protocol,
                )
            terminal_regularization = (
                cfg.weight_smooth * param.smoothness_loss()
            )
            if has_barrier and cfg.weight_barrier > 0.0:
                terminal_regularization = terminal_regularization + (
                    cfg.weight_barrier * param.fab_barrier_loss()
                )
            terminal_total_loss = terminal_data_loss + terminal_regularization
        terminal_data_value = float(terminal_data_loss)
        terminal_total_value = float(terminal_total_loss)
        if not (
            math.isfinite(terminal_data_value)
            and math.isfinite(terminal_total_value)
        ):
            raise FloatingPointError(
                "non-finite terminal sensor-information full-field replay"
            )
        sensor_terminal_validation = {
            "step": cfg.n_steps,
            "updates_completed": cfg.n_steps,
            "state": "post_update_terminal",
            "data_loss": terminal_data_value,
            "total_loss": terminal_total_value,
        }
        if stochastic_sensor_fields:
            sensor_full_validation_trace.append(sensor_terminal_validation)
        if terminal_total_value < best_loss:
            best_loss = terminal_total_value
            best_state = {
                key: value.detach().clone()
                for key, value in param.state_dict().items()
            }

    if best_state is not None:
        param.load_state_dict(best_state)
    return {
        "best_loss": best_loss,
        "history": history,
        "loss_trace": loss_trace,
        "data_loss_trace": data_loss_trace,
        "gradient_norm_trace": gradient_norm_trace,
        "data_gradient_norm_trace": data_gradient_norm_trace,
        "post_clip_gradient_norm_trace": post_clip_gradient_norm_trace,
        "nonzero_gradient_fraction_trace": nonzero_gradient_fraction_trace,
        "parameter_update_norm_trace": parameter_update_norm_trace,
        "sensor_field_indices_trace": sensor_field_indices_trace,
        "sensor_full_validation_trace": sensor_full_validation_trace,
        "sensor_terminal_validation": sensor_terminal_validation,
        "initial_exact_total_loss": initial_exact_total_loss,
        "stochastic_sensor_fields": bool(stochastic_sensor_fields),
        "best_loss_basis": (
            "exact_full_field_quadrature"
            if use_split_sensor_objective else "per_step_training_objective"
        ),
        "steps_completed": len(loss_trace),
        "param": param,
    }
