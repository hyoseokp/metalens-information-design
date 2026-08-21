"""Realistic CMOS pixel stack: multi-layer thick stack + microlens + active area.

Models the optical chain inside each sensor pixel cell:

    air (n_air = 1.0)
    │
    ▼  ──────  Microlens curvature (real-space focusing phase φ_ML(x))
    │
    ▼  ──────  Stack layers [(d_i, n_i)] in order from top to bottom:
                 e.g. planarization (~0.5 μm, n~1.5)
                      color filter   (~0.8 μm, n~1.65)
                      SiO₂           (~0.2 μm, n~1.46)
    │
    ▼  ──────  Si photodiode surface (n_Si ≈ 4.0 + small k)
    │
    ▼  ──────  PD active area (square aperture, fill factor < 1.0)
                 + iso-cell wall (light outside pixel cell is absorbed by
                                  inter-pixel metal/oxide stack)
    │
    ▼  ──────  Photodiode integration → S_z in Si

Each layer interface and the propagation through each layer are handled
in Fourier space with the angular-spectrum + Fresnel transmission model
(single-pass forward, no multi-reflection). All operations are pure
tensor algebra and run on whichever device the inputs live on (CUDA-ready).

Key entry points
----------------
- :func:`build_pixel_stack_transfer_tmm_polarized`
  and :func:`build_pixel_stack_transfer_single_pass_polarized`
    Return separate complex TE and tangential-TM transfer functions. The
    vector detector rotates every Fourier mode from global ``(Ex, Ey)`` into
    its local ``(s, r)`` basis before applying these coefficients.
- :func:`build_pixel_stack_transfer_unpolarized`
    Legacy scalar compatibility only. Its coherent TE/TM amplitude average
    is not a physical unpolarized-light operator and is not used by the
    vector detector path.
    Returns a complex transfer function ``T(kx, ky, λ)`` of shape
    ``[N_λ, H, W]`` that maps the field amplitude just above the microlens
    (in air) to the field amplitude just inside the photodiode (in Si),
    averaged over TE/TM polarization.
- :func:`build_active_area_mask`
    Returns a real ``[H, W]`` amplitude-equivalent collection mask. The
    physical detector squares it and weights the already-computed Poynting
    flux; it is never multiplied into a propagating field.
- :func:`build_microlens_phase`
    Returns ``[N_λ, H, W]`` per-pixel quadratic focusing phase (already
    used by :mod:`engine.mla`; reproduced here for self-contained pixel
    stack pipelines).
- :class:`PixelStackConfig`
    Convenience container bundling all of the above for one pixel design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .types import SpatialGrid


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer Fresnel + Fourier transfer
# ─────────────────────────────────────────────────────────────────────────────

def _kz_in_medium(
    n: complex,
    k0: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
) -> torch.Tensor:
    """Compute kz_complex = sqrt((n·k0)² − kx² − ky²), branch-correct for
    propagating waves (Im(kz) ≥ 0 → decaying into +z)."""
    n_t = torch.as_tensor(n, dtype=k0.dtype, device=k0.device)
    kz_sq = (n_t * k0) ** 2 - kx * kx - ky * ky
    kz = torch.sqrt(kz_sq)
    # Enforce Im(kz) ≥ 0 (causal: evanescent decays in +z direction).
    # torch.sqrt on a complex64 picks the principal branch, which already
    # satisfies this for typical cases; explicit fix-up:
    needs_flip = kz.imag < 0
    kz = torch.where(needs_flip, -kz, kz)
    return kz


def _fresnel_transmission(
    n_a: complex,
    n_b: complex,
    kz_a: torch.Tensor,
    kz_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresnel amplitude transmission coefficients t_TE, t_TM for a wave
    going from medium a (above) into medium b (below) at the interface,
    with kx, ky preserved across the interface (Snell's law).

    Using the kz-form to avoid trig:
        cos(θ_a) = kz_a / (n_a · k0),   cos(θ_b) = kz_b / (n_b · k0)

        t_TE = 2·n_a·cos(θ_a) / (n_a·cos(θ_a) + n_b·cos(θ_b))
             = 2·kz_a / (kz_a + kz_b)

    TM is returned as the TANGENTIAL-E ratio (the transfer is applied to the
    tangential components Ex, Ey in k-space — same semantics as the TMM
    admittance path):
        t_TM_tang = t_TM_full · cos(θ_b)/cos(θ_a)
                  = 2·n_a²·kz_b / (n_b²·kz_a + n_a²·kz_b)
    (reduces to t_TE at normal incidence, as it must).
    """
    n_a_t = torch.as_tensor(n_a, dtype=kz_a.dtype, device=kz_a.device)
    n_b_t = torch.as_tensor(n_b, dtype=kz_a.dtype, device=kz_a.device)
    t_TE = 2.0 * kz_a / (kz_a + kz_b)
    t_TM = 2.0 * n_a_t * n_a_t * kz_b / (n_b_t * n_b_t * kz_a + n_a_t * n_a_t * kz_b)
    return t_TE, t_TM


def derive_H_in_medium(
    Ex_hat: torch.Tensor,
    Ey_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    kz_safe: torch.Tensor,
    evanescent: torch.Tensor,
    k_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tangential H (Fourier domain) from tangential E in a homogeneous medium.

    Ez from div(E)=0, then H = (k × E) normalized by ``k_norm`` per wavelength.
    PHYSICS: the correct normalization is the VACUUM wavenumber 2π/λ (∝ ω·μ0);
    normalizing by \\|k_medium\\| instead drops a factor n_medium in the Poynting
    flux (detected power 1/n of physical).

    Shapes: fields [chunk, N_λ, H, W]; kx/ky/kz_safe/evanescent broadcastable;
    k_norm [N_λ]. Returns (Hx_hat, Hy_hat) with evanescent modes zeroed.
    """
    Ez_hat = -(kx * Ex_hat + ky * Ey_hat) / kz_safe
    Ez_hat = torch.where(evanescent, torch.zeros((), dtype=Ez_hat.dtype, device=Ez_hat.device), Ez_hat)
    inv_k = (1.0 / k_norm.to(Ex_hat.real.dtype)).view(1, -1, 1, 1).to(Ex_hat.dtype)
    Hx_hat = (ky * Ez_hat - kz_safe * Ey_hat) * inv_k
    Hy_hat = (kz_safe * Ex_hat - kx * Ez_hat) * inv_k
    zero = torch.zeros((), dtype=Hx_hat.dtype, device=Hx_hat.device)
    Hx_hat = torch.where(evanescent, zero, Hx_hat)
    Hy_hat = torch.where(evanescent, zero, Hy_hat)
    return Hx_hat, Hy_hat


def _tmm_layer_matrix(
    n_layer: complex,
    d_um: float,
    k0: torch.Tensor,
    kz_layer: torch.Tensor,
    pol: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Abeles characteristic matrix elements (M11, M12, M21, M22) for one
    layer at all (kx, ky, λ) modes simultaneously.

    For polarization σ (TE = "s", TM = "p"):
        M(δ, η_σ) = [[ cos δ,    i sin δ / η_σ ],
                    [ i η_σ sin δ,  cos δ      ]]

    where δ = k_z d (the per-mode phase thickness in this layer) and the
    layer admittance η_σ in our normalized units (drop factors of vacuum
    impedance η₀ that cancel in the final transmission ratio):
        η_TE = k_z         (k_z carries the cos θ dependence)
        η_TM = (n^2 k_0^2) / k_z

    All inputs are tensors broadcastable to [N_λ, H, W] (complex64).
    """
    n_t = torch.as_tensor(n_layer, dtype=k0.dtype, device=k0.device)
    # e^{+ikz} field convention (matches engine propagation): the standard
    # Abeles matrix is written for e^{-ikz}, so flip the phase-thickness sign.
    # For absorbing layers this is NOT a mere conjugation — the legacy sign
    # produced gain (T>1) in lossy media. Verified against an independent
    # complex128 TMM reference to 4.7e-16.
    delta = -kz_layer * d_um
    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)

    if pol == "s" or pol == "TE":
        eta = kz_layer
    elif pol == "p" or pol == "TM":
        eta = (n_t * n_t * k0 * k0) / kz_layer
    else:
        raise ValueError(f"pol must be 's'/'TE' or 'p'/'TM', got {pol!r}")

    M11 = cos_d
    M12 = 1j * sin_d / eta
    M21 = 1j * eta * sin_d
    M22 = cos_d
    return M11, M12, M21, M22


def _tmm_compose(
    matrices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multiply a sequence of 2x2 element-wise matrices: M_total = M_1 · M_2 · ... · M_N.
    Each matrix is given as (M11, M12, M21, M22) tensors of identical shape."""
    M11, M12, M21, M22 = matrices[0]
    for N11, N12, N21, N22 in matrices[1:]:
        new11 = M11 * N11 + M12 * N21
        new12 = M11 * N12 + M12 * N22
        new21 = M21 * N11 + M22 * N21
        new22 = M21 * N12 + M22 * N22
        M11, M12, M21, M22 = new11, new12, new21, new22
    return M11, M12, M21, M22


def _tmm_transmission(
    M11: torch.Tensor,
    M12: torch.Tensor,
    M21: torch.Tensor,
    M22: torch.Tensor,
    eta_initial: torch.Tensor,
    eta_final: torch.Tensor,
) -> torch.Tensor:
    """Total amplitude transmission coefficient from initial → final medium
    given the composite Abeles matrix elements:

        t = (2 η_0) / ((M11 + M12 η_s) η_0 + (M21 + M22 η_s))

    where η_0 is the initial-medium admittance and η_s is the substrate
    (final-medium) admittance, both in the chosen polarization basis.
    """
    num = 2.0 * eta_initial
    den = (M11 + M12 * eta_final) * eta_initial + (M21 + M22 * eta_final)
    return num / den


def build_pixel_stack_transfer_tmm_polarized(
    sensor_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    layers: Sequence[tuple[float, complex]],
    *,
    n_initial: complex = 1.0 + 0.0j,
    n_final: complex = 4.0 + 0.05j,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate complex TE and tangential-TM transfers from Abeles TMM.

    The returned coefficients act on the local tangential ``s`` and ``r``
    components, respectively. Keeping them separate is essential because an
    unpolarized field is an incoherent mixture, not a coherent average of
    ``t_TE`` and ``t_TM``.

    It uses the full Fabry--Perot/multiple-reflection chain rather than the
    single-pass Fresnel approximation.
    Cost vs single-pass: ~3–4× more elementwise operations per layer,
    plus the matrix product chain (4 multiplies + 4 adds per matrix step).
    Captures multi-reflection / thin-film interference within the stack;
    typical |Δ|T|| < 5% for AR-coated visible-band stacks but can be 10-20%
    for un-AR-coated stacks at oblique incidence.
    """
    if device is None:
        device = sensor_grid.xx_um.device

    H, W = sensor_grid.xx_um.shape
    pitch = sensor_grid.pitch_um

    fy = torch.fft.fftfreq(H, d=pitch, device=device)
    fx = torch.fft.fftfreq(W, d=pitch, device=device)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    kx = (2.0 * torch.pi * fxx).to(torch.complex64)
    ky = (2.0 * torch.pi * fyy).to(torch.complex64)

    wls = wavelengths_um.to(device=device, dtype=torch.float32)
    k0 = (2.0 * torch.pi / wls).view(-1, 1, 1).to(torch.complex64)

    # k_z for initial and final media
    kz_initial = _kz_in_medium(n_initial, k0, kx, ky)
    kz_final = _kz_in_medium(n_final, k0, kx, ky)

    n_init_t = torch.as_tensor(n_initial, dtype=k0.dtype, device=device)
    n_fin_t  = torch.as_tensor(n_final,   dtype=k0.dtype, device=device)

    # TE and TM admittances in the boundary media
    eta0_TE = kz_initial
    etaS_TE = kz_final
    eta0_TM = (n_init_t * n_init_t * k0 * k0) / kz_initial
    etaS_TM = (n_fin_t  * n_fin_t  * k0 * k0) / kz_final

    # Compose layer matrices (in propagation order, top → bottom)
    mats_TE: list[tuple[torch.Tensor, ...]] = []
    mats_TM: list[tuple[torch.Tensor, ...]] = []
    for d_um, n_layer in layers:
        kz_l = _kz_in_medium(n_layer, k0, kx, ky)
        mats_TE.append(_tmm_layer_matrix(n_layer, d_um, k0, kz_l, "s"))
        mats_TM.append(_tmm_layer_matrix(n_layer, d_um, k0, kz_l, "p"))

    if not mats_TE:
        # No internal layers: direct interface initial→final
        # Equivalent to a single Fresnel transmission.
        t_TE = 2.0 * eta0_TE / (eta0_TE + etaS_TE)
        t_TM = 2.0 * eta0_TM / (eta0_TM + etaS_TM)
    else:
        M_TE = _tmm_compose(mats_TE)
        M_TM = _tmm_compose(mats_TM)
        t_TE = _tmm_transmission(*M_TE, eta0_TE, etaS_TE)
        t_TM = _tmm_transmission(*M_TM, eta0_TM, etaS_TM)

    # Zero out evanescent modes in the initial medium
    initial_kz_sq_real = ((n_init_t * k0) ** 2 - kx * kx - ky * ky).real
    evanescent = (initial_kz_sq_real <= 0).expand_as(t_TE)
    t_TE[evanescent] = 0.0
    t_TM[evanescent] = 0.0

    return t_TE, t_TM


def build_pixel_stack_transfer_tmm(
    sensor_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    layers: Sequence[tuple[float, complex]],
    *,
    n_initial: complex = 1.0 + 0.0j,
    n_final: complex = 4.0 + 0.05j,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the legacy coherent TE/TM amplitude average.

    This compatibility wrapper is retained for old diagnostics. It is not a
    physical unpolarized-light operator and the vector imaging path must use
    :func:`build_pixel_stack_transfer_tmm_polarized` instead.
    """
    t_te, t_tm = build_pixel_stack_transfer_tmm_polarized(
        sensor_grid,
        wavelengths_um,
        layers,
        n_initial=n_initial,
        n_final=n_final,
        device=device,
    )
    return 0.5 * (t_te + t_tm)


def build_pixel_stack_transfer_single_pass_polarized(
    sensor_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    layers: Sequence[tuple[float, complex]],
    *,
    n_initial: complex = 1.0 + 0.0j,
    n_final: complex = 4.0 + 0.05j,
    include_final_interface: bool = True,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build separate single-pass TE and tangential-TM stack transfers.

    Parameters
    ----------
    sensor_grid : SpatialGrid
        Sensor plane spatial grid; the Fourier grid is derived from
        ``pitch_um`` and the grid shape via ``torch.fft.fftfreq``.
    wavelengths_um : Tensor [N_λ]
        Free-space wavelengths in μm.
    layers : sequence of (thickness_um, complex_index)
        Layer stack in order from incident (top) to detector (bottom),
        EXCLUDING the initial medium (air above ML) and the final medium
        (Si). Each entry: ``(d_um, n + 1j·k)``. Pure-real index can be
        passed as ``(d_um, n)``.
    n_initial : complex
        Refractive index of the medium ABOVE the stack (just below the
        microlens curvature; usually planarization resin or air).
        Default: air ``1.0``.
    n_final : complex
        Refractive index of the detector medium (Silicon).
        Default ``4.0 + 0.05j`` (visible-band typical).
    include_final_interface : bool
        If True, include the Fresnel transmission into the final medium.
        Set False if the caller will handle the n_final interface
        externally.
    device : torch.device, optional
        Device for output tensor. Defaults to ``sensor_grid.xx_um.device``.

    Returns
    -------
    (T_TE, T_TM) : tuple of Tensor [N_λ, H, W], complex64
        Local-s and local-r (tangential-p) field-amplitude transfers. They
        must be applied after rotating the global Cartesian field into that
        per-mode basis.
    """
    if device is None:
        device = sensor_grid.xx_um.device

    H, W = sensor_grid.xx_um.shape
    pitch = sensor_grid.pitch_um

    # Build frequency grid (cycles/μm) → angular wavenumber components
    fy = torch.fft.fftfreq(H, d=pitch, device=device)
    fx = torch.fft.fftfreq(W, d=pitch, device=device)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    kx = (2.0 * math.pi * fxx).to(torch.complex64)
    ky = (2.0 * math.pi * fyy).to(torch.complex64)

    # k0 per wavelength: shape [N_λ, 1, 1]
    wls = wavelengths_um.to(device=device, dtype=torch.float32)
    k0 = (2.0 * math.pi / wls).view(-1, 1, 1).to(torch.complex64)

    # Initial medium kz
    kz_curr = _kz_in_medium(n_initial, k0, kx, ky)

    T_TE = torch.ones((wls.numel(), H, W), dtype=torch.complex64, device=device)
    T_TM = torch.ones_like(T_TE)
    n_curr: complex = n_initial

    # Each layer: cross interface (current → layer i) then propagate through layer i
    for d_um, n_i in layers:
        kz_i = _kz_in_medium(n_i, k0, kx, ky)
        t_TE, t_TM = _fresnel_transmission(n_curr, n_i, kz_curr, kz_i)
        T_TE = T_TE * t_TE
        T_TM = T_TM * t_TM
        if d_um > 0:
            phase = torch.exp(1j * kz_i * d_um)
            T_TE = T_TE * phase
            T_TM = T_TM * phase
        n_curr = n_i
        kz_curr = kz_i

    # Final interface to detector medium (Si)
    if include_final_interface:
        kz_final = _kz_in_medium(n_final, k0, kx, ky)
        t_TE, t_TM = _fresnel_transmission(n_curr, n_final, kz_curr, kz_final)
        T_TE = T_TE * t_TE
        T_TM = T_TM * t_TM

    # Zero-out evanescent modes in the initial medium (these never propagate
    # into the stack; they would have decayed inside the air gap above ML).
    initial_kz_sq_real = ((torch.as_tensor(n_initial, dtype=k0.dtype, device=device) * k0) ** 2 - kx * kx - ky * ky).real
    evanescent = (initial_kz_sq_real <= 0).expand_as(T_TE)
    T_TE[evanescent] = 0.0
    T_TM[evanescent] = 0.0

    return T_TE, T_TM


def build_pixel_stack_transfer_unpolarized(
    sensor_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    layers: Sequence[tuple[float, complex]],
    *,
    n_initial: complex = 1.0 + 0.0j,
    n_final: complex = 4.0 + 0.05j,
    include_final_interface: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the legacy coherent single-pass TE/TM amplitude average.

    Kept only for compatibility with older scalar diagnostics. The vector
    detector uses the two transfers returned by
    :func:`build_pixel_stack_transfer_single_pass_polarized`.
    """
    t_te, t_tm = build_pixel_stack_transfer_single_pass_polarized(
        sensor_grid,
        wavelengths_um,
        layers,
        n_initial=n_initial,
        n_final=n_final,
        include_final_interface=include_final_interface,
        device=device,
    )
    return 0.5 * (t_te + t_tm)


def build_local_sr_basis(
    sensor_grid: SpatialGrid,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local radial unit vector for every detector Fourier mode.

    ``(r_x, r_y)`` is the tangential part of the local p-polarization basis;
    the corresponding s basis is ``(-r_y, r_x)``. At DC the azimuth is
    undefined, so ``r=(1, 0)`` is chosen. TE and TM are degenerate there.
    """
    if device is None:
        device = sensor_grid.xx_um.device
    fy = torch.fft.fftfreq(sensor_grid.height, d=sensor_grid.pitch_um, device=device)
    fx = torch.fft.fftfreq(sensor_grid.width, d=sensor_grid.pitch_um, device=device)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    transverse = torch.hypot(fxx, fyy)
    safe = torch.where(transverse > 0.0, transverse, torch.ones_like(transverse))
    radial_x = fxx / safe
    radial_y = fyy / safe
    radial_x = torch.where(transverse > 0.0, radial_x, torch.ones_like(radial_x))
    radial_y = torch.where(transverse > 0.0, radial_y, torch.zeros_like(radial_y))
    return radial_x.to(torch.float32), radial_y.to(torch.float32)


def apply_pixel_stack_transfer_vectorial(
    Ex_hat: torch.Tensor,
    Ey_hat: torch.Tensor,
    transfer_te: torch.Tensor,
    transfer_tm: torch.Tensor,
    radial_x: torch.Tensor,
    radial_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a stratified stack to global tangential E in Fourier space.

    TM acts on ``E_r = r_x Ex + r_y Ey`` and TE acts on
    ``E_s = -r_y Ex + r_x Ey``. The result is rotated back to global
    Cartesian coordinates. Leading batch dimensions are supported.
    """
    if Ex_hat.shape != Ey_hat.shape:
        raise ValueError("Ex_hat and Ey_hat must have identical shapes")
    if Ex_hat.ndim < 3:
        raise ValueError("vector spectra must end in [wavelength, y, x]")
    expected_hw = Ex_hat.shape[-2:]
    if radial_x.shape != expected_hw or radial_y.shape != expected_hw:
        raise ValueError("local-basis tensors must match the spectrum spatial shape")
    if transfer_te.shape[-3:] != Ex_hat.shape[-3:]:
        raise ValueError("TE transfer must match [wavelength, y, x]")
    if transfer_tm.shape[-3:] != Ex_hat.shape[-3:]:
        raise ValueError("TM transfer must match [wavelength, y, x]")

    real_dtype = Ex_hat.real.dtype
    r_x = radial_x.to(device=Ex_hat.device, dtype=real_dtype)
    r_y = radial_y.to(device=Ex_hat.device, dtype=real_dtype)
    t_te = transfer_te.to(device=Ex_hat.device, dtype=Ex_hat.dtype)
    t_tm = transfer_tm.to(device=Ex_hat.device, dtype=Ex_hat.dtype)

    E_r = r_x * Ex_hat + r_y * Ey_hat
    E_s = -r_y * Ex_hat + r_x * Ey_hat
    E_r = t_tm * E_r
    E_s = t_te * E_s
    Ex_out = r_x * E_r - r_y * E_s
    Ey_out = r_y * E_r + r_x * E_s
    return Ex_out, Ey_out


# ─────────────────────────────────────────────────────────────────────────────
# Active area + iso-cell mask
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_local_coord(
    coord_um: torch.Tensor,
    pixel_pitch_um: float,
    n_samples: int,
    sample_pitch_um: float,
) -> torch.Tensor:
    """Local coordinate within the pixel cell containing each sample (E13).

    The physical pixel lattice is defined by the avg_pool binning of the
    centered sample grid: pixel centers at (m - (P-1)/2)·p with
    P = n_samples·sample_pitch/p pixels across the window. For even P the
    centers are at half-integer multiples of p; for odd P at integer
    multiples. Falls back to the integer-center (round) convention when the
    window is not an integer number of pixels.
    """
    p = float(pixel_pitch_um)
    n_pix = n_samples * float(sample_pitch_um) / p
    n_pix_i = round(n_pix)
    if abs(n_pix - n_pix_i) < 1e-6 and n_pix_i % 2 == 0:
        return torch.remainder(coord_um, p) - 0.5 * p
    return coord_um - p * torch.round(coord_um / p)


def build_active_area_mask(
    sensor_grid: SpatialGrid,
    *,
    pixel_pitch_um: float | None = None,
    aperture_ratio: float = 0.75,
    shape: str = "square",
    iso_cell: bool = True,
) -> torch.Tensor:
    """Per-pixel active-area amplitude mask.

    Each pixel cell of width ``pixel_pitch_um`` has a centred active
    aperture of side ``aperture_ratio · pixel_pitch_um`` (square) or
    diameter ``aperture_ratio · pixel_pitch_um`` (round). Light outside
    the active area is absorbed by the iso-cell metal/oxide wall.

    Two regimes, decided automatically from the optical-grid sample
    density relative to the pixel pitch:

    - **Sub-pixel resolution** (``sensor_grid.pitch_um < pixel_pitch_um``):
      Returns a hard binary mask in ``{0, 1}``. The Sz integral over each
      pixel cell then naturally captures the active aperture geometry.

    - **Pixel-aligned grid** (``sensor_grid.pitch_um ≈ pixel_pitch_um``,
      i.e. one optical sample per pixel — the common case for our
      simulator's metalens-aperture-matched grid): there is no sub-pixel
      structure to mask, so the active-area effect is applied as a uniform
      *amplitude* multiplier ``√(active_area / pixel_area)``, which gives
      a power reduction ``active_area / pixel_area`` after Sz = |amplitude|².
      For ``shape="square"`` this is simply ``aperture_ratio`` in
      amplitude (``aperture_ratio²`` in power). Iso-cell + sub-pixel
      angular sensitivity is then NOT captured (would require finer grid).

    Parameters
    ----------
    sensor_grid : SpatialGrid
        Sensor plane grid (xx_um, yy_um in μm; pitch_um).
    pixel_pitch_um : float, optional
        Pixel cell pitch. Defaults to ``sensor_grid.pitch_um``.
    aperture_ratio : float
        Active-area linear fraction (0 < ratio ≤ 1). Typical BSI: 0.7–0.85.
    shape : {"square", "round"}
        Active area geometry.
    iso_cell : bool
        Reserved; iso-cell separation is implicit in the local-coordinate
        construction (sub-pixel regime) or in the duty-cycle reduction
        (pixel-aligned regime).

    Returns
    -------
    mask : Tensor [H, W], float32
        Amplitude-domain mask. Multiply the field by this; the resulting
        Sz = |amplitude|² gets the correct active-area power fraction.
    """
    del iso_cell
    if pixel_pitch_um is None:
        pixel_pitch_um = sensor_grid.pitch_um

    p = float(pixel_pitch_um)
    sample_pitch = float(sensor_grid.pitch_um)

    # If the optical grid does not sub-sample the pixel cell, use a uniform
    # duty-cycle amplitude multiplier (aperture_area / pixel_area)^(1/2).
    if sample_pitch >= 0.95 * p:
        if shape == "square":
            area_fraction = aperture_ratio ** 2
        elif shape == "round":
            area_fraction = math.pi * (aperture_ratio / 2.0) ** 2
        else:
            raise ValueError(f"shape must be 'square' or 'round', got {shape!r}")
        amp_factor = math.sqrt(area_fraction)
        return torch.full_like(sensor_grid.xx_um, amp_factor, dtype=torch.float32)

    # Sub-pixel regime: area-weighted (anti-aliased) AMPLITUDE mask.
    # A hard binary mask realizes the wrong fill factor at coarse sampling
    # (2-4 samples/pixel all land inside -> fill factor 1.0); weighting each
    # sample cell by its covered AREA keeps the integrated power fraction
    # exact at every sampling density. Amplitude = sqrt(coverage) so that
    # S_z = |amp|^2 integrates to the geometric active-area fraction.
    xx = sensor_grid.xx_um
    yy = sensor_grid.yy_um
    # E13: align pixel cells with the avg_pool binned pixels. On the centered
    # grid the binned pixel centers sit at (m - (P-1)/2)·p (P = pixel count):
    # HALF-integer multiples of p when P is even (stage5: P=208), INTEGER
    # multiples when P is odd. The previous unconditional round() tiling was
    # displaced by half a pixel in the standard even-P configuration.
    x_local = _pixel_local_coord(xx, p, sensor_grid.width, sensor_grid.pitch_um)
    y_local = _pixel_local_coord(yy, p, sensor_grid.height, sensor_grid.pitch_um)
    half_active = 0.5 * aperture_ratio * p
    s = sample_pitch
    if shape == "square":
        covx = ((half_active - x_local.abs() + 0.5 * s) / s).clamp(0.0, 1.0)
        covy = ((half_active - y_local.abs() + 0.5 * s) / s).clamp(0.0, 1.0)
        coverage = covx * covy
    elif shape == "round":
        # radial signed distance to the circle edge, smoothed over one sample
        r = torch.sqrt(x_local * x_local + y_local * y_local)
        coverage = ((half_active - r + 0.5 * s) / s).clamp(0.0, 1.0)
    else:
        raise ValueError(f"shape must be 'square' or 'round', got {shape!r}")
    return torch.sqrt(coverage).to(torch.float32)


def apply_active_area_to_poynting(
    poynting_z: torch.Tensor,
    active_area_amplitude: torch.Tensor,
) -> torch.Tensor:
    """Apply detector fill factor as a local collected-power weight.

    ``build_active_area_mask`` returns ``sqrt(coverage)`` for historical API
    compatibility. Squaring it here recovers the geometric coverage. The
    weight is deliberately applied only after E and H have formed ``S_z``;
    multiplying the field first would turn the detector fill factor into an
    artificial diffracting aperture and make the reconstructed H nonlocal.
    """
    if active_area_amplitude.shape != poynting_z.shape[-2:]:
        raise ValueError("active-area mask must match the Poynting spatial shape")
    weight = active_area_amplitude.to(
        device=poynting_z.device, dtype=poynting_z.dtype
    ).square()
    return poynting_z * weight


# ─────────────────────────────────────────────────────────────────────────────
# Microlens phase
# ─────────────────────────────────────────────────────────────────────────────

def build_microlens_phase(
    sensor_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    *,
    pixel_pitch_um: float | None = None,
    ml_focal_length_um: float,
) -> torch.Tensor:
    """Per-pixel quadratic focusing phase from microlens curvature.

    φ(x, y) = -π / (λ · f_ML) · (x_local² + y_local²)

    Parameters
    ----------
    sensor_grid : SpatialGrid
    wavelengths_um : Tensor [N_λ]
    pixel_pitch_um : float, optional
        Defaults to ``sensor_grid.pitch_um``.
    ml_focal_length_um : float
        Microlens focal length (typical: ~ pixel pitch for unit f-number,
        or matched to ML→PD distance for in-focus design).

    Returns
    -------
    phase : Tensor [N_λ, H, W], float32
    """
    if pixel_pitch_um is None:
        pixel_pitch_um = sensor_grid.pitch_um

    xx = sensor_grid.xx_um
    yy = sensor_grid.yy_um
    p = float(pixel_pitch_um)
    # E13: parity-aware cell alignment (see build_active_area_mask) — the
    # ML apex must sit at the binned-pixel centers.
    x_local = _pixel_local_coord(xx, p, sensor_grid.width, sensor_grid.pitch_um)
    y_local = _pixel_local_coord(yy, p, sensor_grid.height, sensor_grid.pitch_um)
    r_local_sq = x_local.square() + y_local.square()  # [H, W]

    lam = wavelengths_um.to(xx.device, torch.float32).view(-1, 1, 1)  # [N_λ, 1, 1]
    phase = -(math.pi / (lam * float(ml_focal_length_um))) * r_local_sq.unsqueeze(0)
    return phase


# ─────────────────────────────────────────────────────────────────────────────
# Convenience config + full pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PixelStackConfig:
    """Bundles all parameters for one realistic CMOS pixel design."""

    PERIODIC_SPATIAL_CONTRACT = "periodic_microlens_active_area_v1"
    PLANAR_UNIT_FILL_CONTRACT = "planar_bsi_unit_fill_v1"
    LOSSLESS_INTERFACE_FLUX_CONTRACT = (
        "lossless_terminal_interface_flux_proxy_v1"
    )
    LEGACY_COMPLEX_TERMINAL_CONTRACT = (
        "legacy_complex_terminal_flux_approximation_v1"
    )

    # Microlens
    ml_focal_length_um: float = 3.0   # match ML→PD distance for in-focus design

    microlens_enabled: bool = True

    # Multi-layer stack between ML and PD: list of (thickness_um, n+ik)
    layers: list[tuple[float, complex]] = field(default_factory=lambda: [
        (0.5, 1.50 + 0.0j),    # planarization (acrylic resin)
        (0.8, 1.65 + 0.0j),    # color filter (organic dye in resin)
        (0.2, 1.46 + 0.0j),    # SiO₂ passivation
    ])
    # Initial medium (above ML); usually air above the device
    n_initial: complex = 1.0 + 0.0j
    # Final medium (Si photodiode bulk); 4.0 + 0.05j is typical for visible
    n_final: complex = 4.0 + 0.05j
    # Quantity represented at the terminal detector plane. Production uses a
    # real-admittance interface-flux proxy, not finite-depth Si absorption.
    terminal_detector_contract: str = LEGACY_COMPLEX_TERMINAL_CONTRACT

    # Active area
    aperture_ratio: float = 0.75       # linear fraction, BSI typical 0.70–0.85
    aperture_shape: str = "square"      # "square" or "round"

    spatial_contract: str = PERIODIC_SPATIAL_CONTRACT

    # Pixel pitch (defaults to sensor_grid pitch)
    pixel_pitch_um: float | None = None

    # Sensor type tag (BSI vs FSI). Currently informational only;
    # the physical model is the same in both, parameters differ.
    sensor_type: str = "BSI"

    # Multi-reflection model: "single_pass" (forward Fresnel chain,
    # cheaper, ignores intra-stack Fabry-Perot) or "tmm" (full Abeles
    # transfer-matrix method, accurate but ~3-4x more elementwise ops).
    transfer_method: str = "tmm"

    @classmethod
    def smartphone_bsi_default(cls) -> "PixelStackConfig":
        """Smartphone-class BSI defaults (~0.5–1 μm pitch sensors)."""
        return cls(
            ml_focal_length_um=2.5,
            layers=[
                (0.5, 1.50 + 0.0j),
                (0.8, 1.65 + 0.0j),
                (0.2, 1.46 + 0.0j),
            ],
            n_initial=1.0 + 0.0j,
            n_final=4.0 + 0.05j,
            aperture_ratio=0.78,
            aperture_shape="square",
            sensor_type="BSI",
        )

    @classmethod
    def planar_bsi_unit_fill(cls) -> "PixelStackConfig":
        """Planar BSI stack with unit fill and no periodic microlens.

        The multilayer TE/TM transfer into silicon remains active. Removing
        the periodic microlens and iso-cell aperture makes the detector-side
        response globally shift invariant, as required by a single fine-grid
        convolution OTF. Periodic pixel optics require a separate
        phase-dependent polyphase stack and are not represented by this mode.
        """

        return cls(
            ml_focal_length_um=2.5,
            microlens_enabled=False,
            layers=[
                (0.5, 1.50 + 0.0j),
                (0.8, 1.65 + 0.0j),
                (0.2, 1.46 + 0.0j),
            ],
            n_initial=1.0 + 0.0j,
            n_final=4.0 + 0.0j,
            terminal_detector_contract=cls.LOSSLESS_INTERFACE_FLUX_CONTRACT,
            aperture_ratio=1.0,
            aperture_shape="square",
            sensor_type="BSI",
            spatial_contract=cls.PLANAR_UNIT_FILL_CONTRACT,
        )

    def stack_total_thickness_um(self) -> float:
        return sum(d for d, _ in self.layers)


@dataclass
class PixelStackBuffers:
    """Precomputed CUDA tensors ready for per-frame application.

    All tensors live on the same device; build once, reuse per chunk."""

    ml_phase: torch.Tensor              # [N_λ, H, W] float32 (rad)
    stack_T_te: torch.Tensor             # [N_λ, H, W] complex64, local s
    stack_T_tm: torch.Tensor             # [N_λ, H, W] complex64, local r
    radial_x: torch.Tensor               # [H, W] float32
    radial_y: torch.Tensor               # [H, W] float32
    active_mask: torch.Tensor            # [H, W] sqrt(collection coverage)
    n_final: complex                     # for Sz computation in Si
    terminal_detector_contract: str

    @property
    def stack_T_unpol(self) -> torch.Tensor:
        """Legacy coherent amplitude average for old diagnostics only."""
        return 0.5 * (self.stack_T_te + self.stack_T_tm)


def build_pixel_stack_buffers(
    sensor_grid: SpatialGrid,
    wavelengths_um: torch.Tensor,
    config: PixelStackConfig,
    device: torch.device | None = None,
) -> PixelStackBuffers:
    """Build all precomputed tensors for the realistic pixel stack pipeline.
    Call once at engine init; reuse per chunk."""
    if device is None:
        device = sensor_grid.xx_um.device

    pitch = config.pixel_pitch_um
    if pitch is None:
        pitch = sensor_grid.pitch_um

    if config.spatial_contract not in {
        PixelStackConfig.PERIODIC_SPATIAL_CONTRACT,
        PixelStackConfig.PLANAR_UNIT_FILL_CONTRACT,
    }:
        raise ValueError(
            f"unknown pixel-stack spatial contract {config.spatial_contract!r}"
        )
    if config.spatial_contract == PixelStackConfig.PLANAR_UNIT_FILL_CONTRACT:
        if config.microlens_enabled or not math.isclose(
            float(config.aperture_ratio), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "planar_bsi_unit_fill_v1 requires microlens_enabled=False "
                "and aperture_ratio=1"
            )
        if config.terminal_detector_contract != (
            PixelStackConfig.LOSSLESS_INTERFACE_FLUX_CONTRACT
        ):
            raise ValueError(
                "planar_bsi_unit_fill_v1 requires the lossless terminal "
                "interface-flux proxy contract"
            )
        terminal_index = complex(config.n_final)
        if (
            not math.isfinite(float(terminal_index.real))
            or float(terminal_index.real) <= 0.0
            or not math.isclose(
                float(terminal_index.imag),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "lossless_terminal_interface_flux_proxy_v1 requires a "
                "positive real n_final; complex-Si absorption is not modeled"
            )
    if config.microlens_enabled:
        ml_phase = build_microlens_phase(
            sensor_grid, wavelengths_um,
            pixel_pitch_um=pitch,
            ml_focal_length_um=config.ml_focal_length_um,
        )
    else:
        ml_phase = torch.zeros(
            (int(wavelengths_um.numel()), sensor_grid.height, sensor_grid.width),
            device=device,
            dtype=torch.float32,
        )
    if config.transfer_method == "tmm":
        stack_T_te, stack_T_tm = build_pixel_stack_transfer_tmm_polarized(
            sensor_grid, wavelengths_um,
            layers=config.layers,
            n_initial=config.n_initial,
            n_final=config.n_final,
            device=device,
        )
    elif config.transfer_method == "single_pass":
        stack_T_te, stack_T_tm = build_pixel_stack_transfer_single_pass_polarized(
            sensor_grid, wavelengths_um,
            layers=config.layers,
            n_initial=config.n_initial,
            n_final=config.n_final,
            include_final_interface=True,
            device=device,
        )
    else:
        raise ValueError(
            f"transfer_method must be 'tmm' or 'single_pass', got {config.transfer_method!r}"
        )
    active = build_active_area_mask(
        sensor_grid,
        pixel_pitch_um=pitch,
        aperture_ratio=config.aperture_ratio,
        shape=config.aperture_shape,
    )
    radial_x, radial_y = build_local_sr_basis(sensor_grid, device=device)
    return PixelStackBuffers(
        ml_phase=ml_phase.to(device),
        stack_T_te=stack_T_te.to(device),
        stack_T_tm=stack_T_tm.to(device),
        radial_x=radial_x.to(device),
        radial_y=radial_y.to(device),
        active_mask=active.to(device),
        n_final=config.n_final,
        terminal_detector_contract=config.terminal_detector_contract,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk application
# ─────────────────────────────────────────────────────────────────────────────

def apply_pixel_stack_to_field(
    field_at_sensor: torch.Tensor,
    buffers: PixelStackBuffers,
) -> torch.Tensor:
    """Apply the legacy scalar stack approximation to one field component.

    This compatibility function coherently averages TE and TM and therefore
    must not be used for the vector imaging path. It also deliberately does
    not apply detector fill factor to the field; collection weighting belongs
    after Poynting-flux formation.

    Steps (per polarization component, e.g. Ex or Ey):
      1. Multiply by ML phase exp(i·φ_ML)
      2. FFT, multiply by Fourier-domain stack transfer T_unpol, IFFT

    Parameters
    ----------
    field_at_sensor : Tensor [..., N_λ, H, W], complex
        Field component at the sensor plane in air, just above the
        microlens. Leading dims may be empty or batch dims (object points).
    buffers : PixelStackBuffers
        Output of :func:`build_pixel_stack_buffers`.

    Returns
    -------
    field_at_pd : Tensor [..., N_λ, H, W], complex
        Field component just inside the photodiode surface in Si. Detector
        collection area has not yet been applied.
    """
    # ML phase (broadcast over leading dims)
    ml_factor = torch.exp(1j * buffers.ml_phase.to(field_at_sensor.real.dtype))
    field = field_at_sensor * ml_factor

    # Fourier propagation through stack
    field_hat = torch.fft.fft2(field, dim=(-2, -1))
    field_hat = field_hat * buffers.stack_T_unpol
    return torch.fft.ifft2(field_hat, dim=(-2, -1))
