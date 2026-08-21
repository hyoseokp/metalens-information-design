from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .config import EngineSpec
from .grids import build_common_propagation_grid, build_frequency_grid, build_spatial_grid
from .image_formation import incoherent_accumulate, vectorial_incoherent_accumulate, vectorial_poynting_accumulate
from .isp import demosaic_bilinear, demosaic_malvar
from .metalens_response import BaseResponseModel
from .object_source import flatten_object_radiance, scene_to_spectral_radiance, slice_object_batch
from .object_to_pupil import compute_incident_field
from .pixel_stack import (
    apply_active_area_to_poynting,
    apply_pixel_stack_transfer_vectorial,
    derive_H_in_medium,
)
from .propagation import (
    angular_spectrum_transfer_function,
    propagate_vectorial_E_cross_grid,
    build_kz_vectors,
    compute_Ez_from_transverse,
    propagate_angular_spectrum,
    propagate_angular_spectrum_cross_grid,
    propagate_vectorial_fused,
    propagate_vectorial_with_H_cross_grid,
    propagate_vectorial_with_H_fused,
)
from .sensor_readout import (
    adc_encode,
    apply_bayer_sampling,
    apply_sensor_noise,
    build_rggb_masks,
    channelwise_cfa_filter,
    expected_photoelectrons,
    pixel_integrate_channel_images,
)
from .sensor_stack import apply_sensor_stack
from .spectral import spectral_weights_or_ones, wavelengths_to_tensor
from .symmetry_dispatch import (
    ReflectionFoldAuthorization,
    load_reflection_fold_authorization as _load_reflection_fold_authorization,
    validate_runtime_authorization as _validate_runtime_fold_authorization,
)
from .types import ForwardOutputs, ObjectPointBatch, RawDeterministicBatch

# E28: default boundary handling for the detector-stage FFT chain (pixel-stack
# transfer -> in-Si H completion -> Poynting). 2.0 zero-pads the sensor window
# so that stage is a linear (non-circular) operator; 1.0 is the legacy
# circular-boundary path, preserved bit-identically as an explicit opt-out.
DEFAULT_DETECTOR_FFT_PAD_FACTOR = 2.0


def _vectorial_subbatch() -> int:
    """Sub-batch size (entries along the dipole-tripled chunk axis) for the
    common-grid vectorial propagation. Default 3 keeps the peak under ~8 GiB
    on the stage5 grids (12 GiB cards); override via ENGINE2_VEC_SUBBATCH."""
    import os
    try:
        return max(1, int(os.environ.get("ENGINE2_VEC_SUBBATCH", "3")))
    except ValueError:
        return 3


def _local_sp_dipole_basis(theta_x_rad, theta_y_rad):
    """Local output s/p basis and the three Cartesian-dipole projections."""
    ax = torch.tan(theta_x_rad)
    ay = torch.tan(theta_y_rad)
    kzn = torch.rsqrt(1.0 + ax * ax + ay * ay)            # cos(theta) = kz/k
    kxn = ax * kzn
    kyn = ay * kzn
    kt = torch.sqrt(kxn * kxn + kyn * kyn)
    safe = kt > 1e-9
    inv_kt = torch.where(safe, 1.0 / kt.clamp_min(1e-9), torch.zeros_like(kt))
    # local polarization unit vectors (3D):  s = z×k/|z×k|,  p = k×s
    sx = torch.where(safe, -kyn * inv_kt, torch.zeros_like(kt))
    sy = torch.where(safe, kxn * inv_kt, torch.ones_like(kt))
    ptx = torch.where(safe, -kzn * kxn * inv_kt, -torch.ones_like(kt))
    pty = torch.where(safe, -kzn * kyn * inv_kt, torch.zeros_like(kt))
    pz = kt
    # dipole projections a_s = d·s, a_p = d·p  (s has no z component)
    a_s = (sx, sy, torch.zeros_like(kt))          # d = x, y, z
    a_p = (ptx, pty, pz)
    return sx, sy, ptx, pty, a_s, a_p


def _full_jones_input_output_basis(
    theta_x_rad,
    theta_y_rad,
    incident_refractive_index,
    output_refractive_index,
):
    """Build substrate-input and air-output bases plus modal flux scales.

    The ray angles are geometric angles in the declared incident fused-silica
    half-space.  Zeroth-order transmission conserves tangential wavevector,
    not geometric angle.  Consequently input and output ``s`` axes coincide,
    while their ``p=k x s`` axes generally do not.  torcwa's
    ``power_norm=True`` Jones entries act on power-normalized modal
    amplitudes.  In its nonmagnetic convention, a unit-E s or p plane wave
    carries a common modal flux proportional to ``kz/k0=n*cos(theta)``.
    """

    ax = torch.tan(theta_x_rad)
    ay = torch.tan(theta_y_rad)
    cos_in = torch.rsqrt(1.0 + ax * ax + ay * ay)
    kx_hat_in = ax * cos_in
    ky_hat_in = ay * cos_in
    sin_in = torch.sqrt(kx_hat_in.square() + ky_hat_in.square())
    safe = sin_in > 1e-9
    inverse_sin = torch.where(
        safe, 1.0 / sin_in.clamp_min(1e-9), torch.zeros_like(sin_in)
    )
    sx = torch.where(safe, -ky_hat_in * inverse_sin, torch.zeros_like(sin_in))
    sy = torch.where(safe, kx_hat_in * inverse_sin, torch.ones_like(sin_in))

    # Input p = k_in x s and the three Cartesian-dipole projections.
    p_in_x = torch.where(
        safe, -cos_in * kx_hat_in * inverse_sin, -torch.ones_like(sin_in)
    )
    p_in_y = torch.where(
        safe, -cos_in * ky_hat_in * inverse_sin, torch.zeros_like(sin_in)
    )
    p_in_z = sin_in
    a_s = (sx, sy, torch.zeros_like(sin_in))
    a_p = (p_in_x, p_in_y, p_in_z)

    def _spectral_index(value, name):
        index = torch.as_tensor(value, device=ax.device, dtype=ax.dtype)
        if index.ndim == 0:
            index = index.reshape(1)
        if index.ndim != 1:
            raise ValueError(f"{name} must be scalar or one-dimensional")
        if not bool(torch.isfinite(index).all()) or not bool((index > 0).all()):
            raise ValueError(f"{name} must be finite and positive")
        return index.view(1, -1, 1, 1)

    n_in = _spectral_index(
        incident_refractive_index, "incident_refractive_index"
    )
    n_out = _spectral_index(output_refractive_index, "output_refractive_index")
    if n_in.shape[1] != n_out.shape[1] and min(n_in.shape[1], n_out.shape[1]) != 1:
        raise ValueError("incident/output index wavelength axes do not broadcast")

    qx = n_in * kx_hat_in
    qy = n_in * ky_hat_in
    q_squared = qx.square() + qy.square()
    output_k_squared = n_out.square()
    tolerance = 64.0 * torch.finfo(q_squared.dtype).eps * torch.maximum(
        output_k_squared, torch.ones_like(output_k_squared)
    )
    if bool((q_squared >= output_k_squared - tolerance).any()):
        maximum_ratio = float(torch.sqrt(q_squared / output_k_squared).max())
        raise RuntimeError(
            "full-Jones zeroth-order output is non-propagating/TIR in air; "
            f"max(n_in*sin(theta_in)/n_out)={maximum_ratio:.7g}"
        )
    kz_out = torch.sqrt((output_k_squared - q_squared).clamp_min(0.0))
    sin_out = torch.sqrt(q_squared) / n_out
    cos_out = kz_out / n_out

    # The azimuth is unchanged by tangential-k conservation.  At normal
    # incidence the deterministic gauge remains s=+y, p=-x.
    p_out_x = torch.where(
        safe, -cos_out * kx_hat_in * inverse_sin, -torch.ones_like(cos_out)
    )
    p_out_y = torch.where(
        safe, -cos_out * ky_hat_in * inverse_sin, torch.zeros_like(cos_out)
    )

    kz_in = n_in * cos_in
    input_e_to_power = torch.sqrt(kz_in)
    output_power_to_e = torch.rsqrt(kz_out)
    return (
        sx,
        sy,
        p_out_x,
        p_out_y,
        a_s,
        a_p,
        input_e_to_power,
        output_power_to_e,
        sin_out,
    )


def _assemble_unpolarized_pupil_fields(s_te_u, s_tm_u, theta_x_rad, theta_y_rad):
    """Tangential pupil fields for an UNPOLARIZED point source via the
    3-orthogonal-dipole incoherent decomposition.

    A per-ray independent s/p basis carries an arbitrary azimuthal gauge that
    imprints a vortex (azimuthal/radial polarization) on the pupil and puts a
    null at the focus — wrong for a point source, whose field across the pupil
    is mutually coherent. The physical model: an isotropic unpolarized emitter
    is the incoherent average of three orthogonal dipoles d ∈ {x,y,z}; each
    dipole radiates the smooth transverse field  E_d(k_hat) ∝ d − (d·k_hat)k_hat,
    which we decompose into the local s/p response channels (S_TE on s, S_TM
    on p) coherently WITHIN a dipole and sum incoherently ACROSS dipoles
    (stacked along the chunk axis, 3×chunk).

    Normalization: Σ_d (d·s)² = Σ_d (d·p)² = 1, so the total transverse power
    equals |S_TE·u|² + |S_TM·u|² (same scale as the scalar path at θ→0).
    Photometry: the projection magnitudes give the physical cos³ falloff for
    an isotropic source; at small angles the x/y dipoles reduce to the scalar
    Airy field (TE≡TM at normal incidence) and the z dipole vanishes.

    Returns (Ex, Ey) with leading dim = 3 × chunk.
    """
    sx, sy, ptx, pty, a_s, a_p = _local_sp_dipole_basis(
        theta_x_rad, theta_y_rad
    )
    Ex_parts, Ey_parts = [], []
    for d in range(3):
        su = a_s[d] * s_te_u
        pu = a_p[d] * s_tm_u
        Ex_parts.append(su * sx + pu * ptx)
        Ey_parts.append(su * sy + pu * pty)
    return torch.cat(Ex_parts, dim=0), torch.cat(Ey_parts, dim=0)


def _assemble_unpolarized_pupil_fields_jones(
    jones_local_sp,
    incident_u,
    theta_x_rad,
    theta_y_rad,
    *,
    incident_refractive_index=1.0,
    output_refractive_index=1.0,
):
    """Apply a full local-s/p Jones matrix to three incoherent dipoles.

    ``jones_local_sp`` is ``[2,2,chunk,Nlambda,H,W]`` with output
    polarization on the first matrix axis and input polarization on the
    second. Cross-polarized terms are combined coherently within each dipole;
    the returned three dipole fields remain stacked for incoherent summation.
    """
    if jones_local_sp.ndim != 6 or tuple(jones_local_sp.shape[:2]) != (2, 2):
        raise ValueError(
            "full Jones response must be [2,2,chunk,wavelength,y,x]"
        )
    if tuple(jones_local_sp.shape[2:]) != tuple(incident_u.shape):
        raise ValueError("full Jones response and incident pupil field shapes differ")
    (
        sx,
        sy,
        ptx,
        pty,
        a_s,
        a_p,
        input_e_to_power,
        output_power_to_e,
        _,
    ) = _full_jones_input_output_basis(
        theta_x_rad,
        theta_y_rad,
        incident_refractive_index,
        output_refractive_index,
    )
    j_ss, j_sp = jones_local_sp[0, 0], jones_local_sp[0, 1]
    j_ps, j_pp = jones_local_sp[1, 0], jones_local_sp[1, 1]
    Ex_parts, Ey_parts = [], []
    for d in range(3):
        # Convert physical incident E components to torcwa's common
        # power-normalized s/p modal amplitude, apply J, then convert the
        # output modal amplitude back to physical Cartesian E in air.
        in_s = input_e_to_power * a_s[d] * incident_u
        in_p = input_e_to_power * a_p[d] * incident_u
        out_s = output_power_to_e * (j_ss * in_s + j_sp * in_p)
        out_p = output_power_to_e * (j_ps * in_s + j_pp * in_p)
        Ex_parts.append(out_s * sx + out_p * ptx)
        Ey_parts.append(out_s * sy + out_p * pty)
    return torch.cat(Ex_parts, dim=0), torch.cat(Ey_parts, dim=0)


def _is_vectorial_response(response: torch.Tensor) -> bool:
    if response.ndim == 4:
        return False
    if response.ndim == 5 and response.shape[0] == 2:
        return True
    if response.ndim == 6 and tuple(response.shape[:2]) == (2, 2):
        return True
    raise ValueError(
        "response tensor contract must be scalar [B,L,H,W], diagonal "
        "[2,B,L,H,W], or full Jones [2,2,B,L,H,W]"
    )


def _assemble_vectorial_response_pupil_fields(
    response: torch.Tensor,
    incident_u: torch.Tensor,
    theta_x_rad: torch.Tensor,
    theta_y_rad: torch.Tensor,
    *,
    incident_refractive_index: float | torch.Tensor = 1.0,
    output_refractive_index: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch an explicit diagonal or full-Jones response contract."""
    if response.ndim == 5 and response.shape[0] == 2:
        return _assemble_unpolarized_pupil_fields(
            response[0] * incident_u,
            response[1] * incident_u,
            theta_x_rad,
            theta_y_rad,
        )
    if response.ndim == 6 and tuple(response.shape[:2]) == (2, 2):
        return _assemble_unpolarized_pupil_fields_jones(
            response,
            incident_u,
            theta_x_rad,
            theta_y_rad,
            incident_refractive_index=incident_refractive_index,
            output_refractive_index=output_refractive_index,
        )
    raise ValueError(
        "vector response must be diagonal [2,B,L,H,W] or full Jones "
        "[2,2,B,L,H,W]"
    )


class MetalensImagingEngine(nn.Module):
    @property
    def _ps_stack_T(self) -> torch.Tensor:
        """Legacy read-only coherent average for diagnostic compatibility.

        The detector never reads this value. A scalar replacement cannot
        represent the vector stack, so mutation fails loudly rather than
        silently leaving the TE/TM headline path unchanged.
        """
        if not getattr(self, "_has_pixel_stack", False):
            raise AttributeError("this engine has no configured pixel stack")
        return 0.5 * (self._ps_stack_T_te + self._ps_stack_T_tm)

    @_ps_stack_T.setter
    def _ps_stack_T(self, value: torch.Tensor) -> None:
        del value
        raise AttributeError(
            "_ps_stack_T is a read-only legacy diagnostic; update both "
            "_ps_stack_T_te and _ps_stack_T_tm for a vector stack"
        )

    def __init__(
        self,
        spec: EngineSpec,
        response_model: BaseResponseModel,
        width_map_um: torch.Tensor,
        aperture_mask: torch.Tensor | None = None,
        microlens_transfer: torch.Tensor | None = None,
        stack_transfer: torch.Tensor | None = None,
        cfa_transmission: torch.Tensor | None = None,
        qe: torch.Tensor | None = None,
        trainable_width_map: bool = True,
        pixel_stack_config: "PixelStackConfig | None" = None,
        require_full_jones_response: bool = False,
        detector_fft_pad_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.device_ = spec.resolve_device()
        self.dtype_ = spec.default_dtype
        self.response_model = response_model.to(self.device_)
        self._require_full_jones_response = bool(require_full_jones_response)
        if self._require_full_jones_response:
            self._validate_final_exact_response()

        self.object_grid = build_spatial_grid(spec.object_plane, self.device_, self.dtype_)
        self.pupil_grid = build_spatial_grid(spec.pupil_plane, self.device_, self.dtype_)
        self.sensor_grid = build_spatial_grid(spec.sensor_plane, self.device_, self.dtype_)
        self._validate_sampling()

        # Pupil-side frequency grid (used for object-to-pupil incidence math
        # and for legacy strict-grid-match mode). Kept on this attribute name
        # for back-compat with downstream callers that reference it.
        self.frequency_grid = build_frequency_grid(
            spec.pupil_plane.height,
            spec.pupil_plane.width,
            spec.pupil_plane.pitch_um,
            self.device_,
            self.dtype_,
        )
        # Sensor-side frequency grid (used by the realistic pixel stack
        # buffers and the in-Si k-vector recovery). Distinct from pupil's
        # under the cross-grid (Option D) decoupling.
        self.sensor_frequency_grid = build_frequency_grid(
            spec.sensor_plane.height,
            spec.sensor_plane.width,
            spec.sensor_plane.pitch_um,
            self.device_,
            self.dtype_,
        )

        wavelengths_um = wavelengths_to_tensor(spec.spectral.wavelengths_um, self.device_, self.dtype_)
        weights = spectral_weights_or_ones(wavelengths_um, spec.spectral.radiance_weights).to(self.device_)
        self.register_buffer("wavelengths_um", wavelengths_um)
        self.register_buffer("radiance_weights", weights)

        width_map = width_map_um.to(self.device_, torch.float32)
        if trainable_width_map:
            self.width_map_um = nn.Parameter(width_map)
        else:
            self.register_buffer("width_map_um", width_map)

        if aperture_mask is None:
            if spec.aperture_radius_um is None:
                aperture_mask = torch.ones_like(self.pupil_grid.xx_um, dtype=torch.float32)
            else:
                rho = torch.sqrt(self.pupil_grid.xx_um.square() + self.pupil_grid.yy_um.square())
                aperture_mask = (rho <= spec.aperture_radius_um).to(torch.float32)
        self.register_buffer("aperture_mask", aperture_mask.to(self.device_, torch.float32))

        if microlens_transfer is None:
            microlens_transfer = torch.ones(
                (self.wavelengths_um.numel(), self.sensor_grid.height, self.sensor_grid.width),
                device=self.device_,
                dtype=torch.complex64,
            )
        if stack_transfer is None:
            stack_transfer = torch.ones_like(microlens_transfer)
        self.register_buffer("microlens_transfer", microlens_transfer.to(self.device_, torch.complex64))
        self.register_buffer("stack_transfer", stack_transfer.to(self.device_, torch.complex64))

        pixel_h, pixel_w = spec.resolved_pixel_grid_shape()
        bayer_masks = build_rggb_masks(pixel_h, pixel_w, self.device_, torch.float32)
        self.register_buffer("bayer_masks", bayer_masks)

        if cfa_transmission is None:
            cfa_transmission = torch.ones((3, self.wavelengths_um.numel(), 1, 1), device=self.device_, dtype=torch.float32)
        if qe is None:
            qe = torch.ones((3, self.wavelengths_um.numel(), 1, 1), device=self.device_, dtype=torch.float32)
        self.register_buffer("cfa_transmission", cfa_transmission.to(self.device_, torch.float32))
        self.register_buffer("qe", qe.to(self.device_, torch.float32))

        distance_um = self.sensor_grid.z_um - self.pupil_grid.z_um
        # P7 (2026-07-02 review): the pupil-NATIVE transfer/k-vector buffers
        # (propagation_transfer, _kx, _ky, _kz_safe) had zero readers anywhere
        # in engine2 — all propagation runs on the common grid. Removed:
        # ~0.5 GB dead VRAM at d208, ~0.5 GB at d500 pupil-native shapes,
        # widening the subbatch headroom before the WDDM spill cliff.
        # Per-wavelength vacuum wavenumber k_lambda = 2π·n / λ in [rad/μm].
        # Used by vectorial propagation to normalize H = (k × E)/k.
        k_lambda = (2.0 * math.pi * self.spec.propagation_index / self.wavelengths_um).to(torch.float32)
        self.register_buffer("_k_lambda", k_lambda)

        pad = int(self.spec.propagation_pad_factor)
        if pad < 1:
            raise ValueError(f"propagation_pad_factor must be >= 1, got {pad}")
        self._propagation_pad = pad

        # ────────────────────────────────────────────────────────────────────
        # Cross-grid (Option D) common Fourier grid setup.
        # Build a "common pad grid" whose physical extent matches the pupil's
        # padded extent EXACTLY via rational pitch alignment. This lets the
        # pupil spectrum be embedded into the common spectrum without any
        # interpolation. AS propagation, Ez/Hx/Hy derivation, and the final
        # IFFT all happen on the common grid; the result is then center-cropped
        # to the sensor's native shape.
        # When pupil and sensor have identical (height, width, pitch), the
        # common grid degenerates to the sensor pad grid and the path
        # reproduces the legacy single-grid propagation bit-identically
        # (verified by parity unit test).
        # ────────────────────────────────────────────────────────────────────
        common_meta = build_common_propagation_grid(
            pupil_h=self.pupil_grid.height,
            pupil_w=self.pupil_grid.width,
            pupil_pitch_um=self.pupil_grid.pitch_um,
            sensor_h=self.sensor_grid.height,
            sensor_w=self.sensor_grid.width,
            sensor_pitch_um=self.sensor_grid.pitch_um,
            pad_factor=pad,
        )
        self._common_meta = common_meta
        # Common-grid frequency grid (sensor pitch, common-pad shape)
        from .types import FrequencyGrid as _FG
        fxc = torch.fft.fftfreq(common_meta.nc_pad_w, d=self.sensor_grid.pitch_um,
                                device=self.device_).to(torch.float32)
        fyc = torch.fft.fftfreq(common_meta.nc_pad_h, d=self.sensor_grid.pitch_um,
                                device=self.device_).to(torch.float32)
        fyyc, fxxc = torch.meshgrid(fyc, fxc, indexing="ij")
        freq_grid_common = _FG(
            fx_cyc_per_um=fxc, fy_cyc_per_um=fyc,
            fxx_cyc_per_um=fxxc, fyy_cyc_per_um=fyyc,
        )
        transfer_common = angular_spectrum_transfer_function(
            freq_grid_common, self.wavelengths_um,
            distance_um=distance_um,
            refractive_index=self.spec.propagation_index,
        )
        # P6: pre-fuse (alignment phase × amp_scale) into the static transfer
        # so the embed step skips two full-common-grid multiplies per
        # component per propagation (~28 GB/subbatch at d500). All internal
        # call sites pass skip_alignment=True; equivalence is fp-reassociation
        # only (~2e-7 rel). External callers must check
        # engine._transfer_alignment_fused before reusing this buffer.
        from .grids import common_alignment_factor
        _align = common_alignment_factor(common_meta, self.device_,
                                         torch.complex64)
        self.register_buffer(
            "_propagation_transfer_common",
            (transfer_common.to(torch.complex64) * _align).to(torch.complex64))
        self._transfer_alignment_fused = True
        # P7: the common-grid k-vectors and evanescent mask are consumed ONLY
        # by the flat-pixel 5-component path. With a pixel stack configured
        # (the standard IMX555 setup) they are dead VRAM (~1.7 GB at d500:
        # [9,4060,4060] c64 kz + masks). Register only when actually needed.
        if pixel_stack_config is None:
            kx_c, ky_c, kz_safe_c = build_kz_vectors(
                freq_grid_common, self.wavelengths_um, self.spec.propagation_index,
            )
            self.register_buffer("_kx_common", kx_c)
            self.register_buffer("_ky_common", ky_c)
            self.register_buffer("_kz_safe_common", kz_safe_c)
            # Evanescent mask precompute (bit-identical, kz_safe is static).
            # Shape: [N_λ, Hc, Wc]. Caller broadcasts/expands per-chunk.
            self.register_buffer("_evanescent_common",
                (kz_safe_c.abs() == 1.0))
        else:
            self._kx_common = None
            self._ky_common = None
            self._kz_safe_common = None
            self._evanescent_common = None

        # Optional realistic CMOS pixel stack (multi-layer + microlens
        # curvature + active area + iso-cell). When provided, takes
        # precedence over `microlens_transfer` for the vectorial path:
        # the field at the sensor plane is funneled through the stack with
        # separate local-s/local-r transfers. The active area is a detector
        # collection weight applied only after computing S_z in the final
        # medium; it is not a diffracting field mask.
        if pixel_stack_config is not None:
            from .pixel_stack import build_pixel_stack_buffers
            ps_buffers = build_pixel_stack_buffers(
                self.sensor_grid, self.wavelengths_um,
                pixel_stack_config, device=self.device_,
            )
            self.pixel_stack_spatial_contract = str(
                pixel_stack_config.spatial_contract
            )
            self.pixel_stack_microlens_enabled = bool(
                pixel_stack_config.microlens_enabled
            )
            self.pixel_stack_aperture_ratio = float(
                pixel_stack_config.aperture_ratio
            )
            self.pixel_stack_initial_refractive_index = complex(
                pixel_stack_config.n_initial
            )
            self.terminal_detector_contract = str(
                ps_buffers.terminal_detector_contract
            )
            self.register_buffer("_ps_ml_phase", ps_buffers.ml_phase)
            self.register_buffer("_ps_stack_T_te", ps_buffers.stack_T_te)
            self.register_buffer("_ps_stack_T_tm", ps_buffers.stack_T_tm)
            self.register_buffer("_ps_radial_x", ps_buffers.radial_x)
            self.register_buffer("_ps_radial_y", ps_buffers.radial_y)
            self.register_buffer("_ps_active_mask", ps_buffers.active_mask)
            # ml_factor = exp(i * ml_phase) is static; precompute once (bit-identical).
            _ps_ml_factor = torch.exp(1j * ps_buffers.ml_phase.to(torch.float32)).to(torch.complex64)
            self.register_buffer("_ps_ml_factor", _ps_ml_factor)
            # Production uses a positive-real terminal admittance by contract.
            # This is interface-plane acceptance, not propagation or finite-
            # depth absorption in dispersive complex silicon.
            n_si = complex(pixel_stack_config.n_final).real
            self._ps_n_si = n_si
            # vacuum wavenumber per λ — the correct H normalization (∝ ω·μ0)
            self.register_buffer("_ps_k_vac",
                                 (2.0 * math.pi / self.wavelengths_um).to(torch.float32))
            # k-vectors in Si (kx, ky preserved by Snell; kz_Si differs).
            # IMPORTANT: build on the SENSOR frequency grid (not pupil) since
            # the pixel-stack post-Si field lives on sensor pitch and shape.
            # Under Option D, pupil != sensor in general — using pupil's grid
            # would introduce a pitch mismatch in the FFT-based Si Hx/Hy
            # recovery inside forward_optics_from_object_batch.
            kx_si, ky_si, kz_safe_si = build_kz_vectors(
                self.sensor_frequency_grid, self.wavelengths_um, n_si,
            )
            self.register_buffer("_ps_kx_si", kx_si)
            self.register_buffer("_ps_ky_si", ky_si)
            self.register_buffer("_ps_kz_safe_si", kz_safe_si)
            # Pixel-stack evanescent mask precompute (bit-identical, static).
            self.register_buffer("_ps_evanescent_si",
                (kz_safe_si.abs() == 1.0))
            self._has_pixel_stack = True
            # E28: detector-stage FFT boundary. Default is the padded linear
            # boundary; detector_fft_pad_factor=1.0 is the explicit legacy
            # circular-boundary opt-out (bit-identical to pre-E28).
            self._pixel_stack_config = pixel_stack_config
            self._detector_fft_pad_factor = 1.0
            self._detector_pad_hw = None
            resolved_pad = (
                DEFAULT_DETECTOR_FFT_PAD_FACTOR
                if detector_fft_pad_factor is None
                else float(detector_fft_pad_factor)
            )
            self.configure_detector_fft_padding(resolved_pad)
        else:
            self._has_pixel_stack = False
            self._pixel_stack_config = None
            if detector_fft_pad_factor is not None and float(
                detector_fft_pad_factor
            ) != 1.0:
                raise ValueError(
                    "detector_fft_pad_factor requires a configured pixel stack"
                )
            self._detector_fft_pad_factor = 1.0
            self._detector_pad_hw = None
            self.pixel_stack_spatial_contract = "none"
            self.pixel_stack_microlens_enabled = False
            self.pixel_stack_aperture_ratio = 1.0
            self.pixel_stack_initial_refractive_index = None
            self.terminal_detector_contract = "none"

        if self._require_full_jones_response:
            # The early call validates only the response artifact.  At this
            # point wavelength, propagation and pixel-stack buffers also
            # exist, so enforce the complete cross-component medium contract.
            self._validate_final_exact_response()

    def _validate_final_exact_response(self) -> None:
        validator = getattr(
            self.response_model, "validate_final_exact_protocol", None
        )
        if not callable(validator):
            raise RuntimeError(
                "final-exact propagation requires a response model with an "
                "explicit full-Jones validation contract"
            )
        validator()
        # During the first constructor guard the wavelength and pixel-stack
        # buffers do not yet exist.  The constructor calls this method again
        # after those buffers are built; engines that install a Jones model
        # after construction also reach the complete branch immediately.
        if not hasattr(self, "wavelengths_um"):
            return
        output_index = torch.as_tensor(
            self._output_medium_index(),
            device=self.wavelengths_um.device,
            dtype=self.wavelengths_um.dtype,
        )
        propagation_index = torch.full_like(
            self.wavelengths_um, float(self.spec.propagation_index)
        )
        if not torch.allclose(
            output_index, propagation_index, rtol=0.0, atol=1.0e-6
        ):
            raise RuntimeError(
                "final-exact AS propagation_index does not match the Jones "
                "LUT output-medium refractive index"
            )
        if getattr(self, "_has_pixel_stack", False):
            stack_initial = complex(self.pixel_stack_initial_refractive_index)
            if abs(stack_initial.imag) > 1.0e-8 or not torch.allclose(
                output_index,
                torch.full_like(output_index, float(stack_initial.real)),
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    "final-exact pixel-stack n_initial does not match the "
                    "Jones LUT output medium"
                )
            if self.pixel_stack_spatial_contract == "planar_bsi_unit_fill_v1":
                if self.terminal_detector_contract != (
                    "lossless_terminal_interface_flux_proxy_v1"
                ):
                    raise RuntimeError(
                        "final-exact planar detector must use the lossless "
                        "terminal interface-flux proxy contract"
                    )

    def load_reflection_fold_authorization(
        self,
        manifest: str | Path,
        *,
        design: str,
        width_map_override: torch.Tensor | None = None,
    ) -> ReflectionFoldAuthorization:
        """Bind a passing immutable symmetry manifest to this live engine.

        Loading the token does not run optics and does not alter dispatch.
        The token must still be supplied explicitly to a public forward call
        that requests four- or eight-image reuse.
        """

        width_map = (
            self.width_map_um if width_map_override is None else width_map_override
        )
        return _load_reflection_fold_authorization(
            manifest,
            engine=self,
            width_map=width_map,
            design=design,
        )

    def _incident_medium_index(self) -> float | torch.Tensor:
        """Return the object/pupil half-space index for the active response.

        Legacy models retain the historical vacuum spherical wave.  A final-
        exact substrate-input Jones table must expose its dispersive input
        index explicitly so a vacuum ``k0 R`` phase cannot be combined with a
        fused-silica RCWA response.
        """

        if not self._require_full_jones_response:
            return 1.0
        provider = getattr(self.response_model, "incident_refractive_index", None)
        if not callable(provider):
            raise RuntimeError(
                "final-exact substrate-input response must provide the "
                "incident-medium refractive index"
            )
        index = provider(self.wavelengths_um)
        if torch.as_tensor(index).shape != self.wavelengths_um.shape:
            raise RuntimeError(
                "final-exact incident index must have one value per wavelength"
            )
        return index

    def _output_medium_index(self) -> float | torch.Tensor:
        """Return the output half-space index for full-Jones E assembly."""

        if not self._require_full_jones_response:
            return 1.0
        provider = getattr(self.response_model, "output_refractive_index", None)
        if not callable(provider):
            raise RuntimeError(
                "final-exact full-Jones response must provide the output-"
                "medium refractive index"
            )
        index = provider(self.wavelengths_um)
        if torch.as_tensor(index).shape != self.wavelengths_um.shape:
            raise RuntimeError(
                "final-exact output index must have one value per wavelength"
            )
        return index

    def _validate_sampling(self) -> None:
        # Strict mode (legacy): pupil and sensor must have identical (h, w, pitch).
        if self.spec.propagation_strict_grid_match:
            if (self.pupil_grid.height != self.sensor_grid.height
                    or self.pupil_grid.width != self.sensor_grid.width):
                raise ValueError(
                    "propagation_strict_grid_match=True requires identical "
                    "pupil/sensor optical grid shapes."
                )
            if abs(self.pupil_grid.pitch_um - self.sensor_grid.pitch_um) > 1e-9:
                raise ValueError(
                    "propagation_strict_grid_match=True requires identical "
                    "pupil/sensor optical grid pitch."
                )
        # Cross-grid (Option D) requirement: sensor optical grid must divide
        # pixel grid evenly so pixel binning uses uniform avg_pool2d (no Moire).
        pixel_h, pixel_w = self.spec.resolved_pixel_grid_shape()
        if self.sensor_grid.height % pixel_h != 0 or self.sensor_grid.width % pixel_w != 0:
            raise ValueError(
                f"sensor optical grid ({self.sensor_grid.height}, "
                f"{self.sensor_grid.width}) must be an integer multiple of "
                f"pixel grid ({pixel_h}, {pixel_w}) for uniform pixel binning"
            )
        if self.sensor_grid.height <= 0 or self.sensor_grid.width <= 0:
            raise ValueError("sensor grid dimensions must be positive")
        if self.pupil_grid.pitch_um <= 0 or self.sensor_grid.pitch_um <= 0:
            raise ValueError("pupil and sensor pitches must be positive")

    def scene_to_radiance(self, scene: torch.Tensor) -> torch.Tensor:
        scene = scene.to(self.device_, torch.float32)
        return scene_to_spectral_radiance(scene, self.wavelengths_um, self.radiance_weights)

    def build_object_batch(self, scene: torch.Tensor, threshold: float = 0.0) -> ObjectPointBatch:
        spectral_radiance = self.scene_to_radiance(scene)
        return flatten_object_radiance(spectral_radiance, self.object_grid, threshold=threshold)

    _DETECTOR_PAD_BUFFER_NAMES = (
        "_ps_pad_stack_T_te", "_ps_pad_stack_T_tm",
        "_ps_pad_radial_x", "_ps_pad_radial_y",
        "_ps_pad_kx_si", "_ps_pad_ky_si",
        "_ps_pad_kz_safe_si", "_ps_pad_evanescent_si",
    )

    def _register_or_replace_nonpersistent_buffer(self, name, value) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=False)

    def configure_detector_fft_padding(self, pad_factor: float) -> None:
        """E28: choose the detector-stage FFT boundary handling.

        ``pad_factor=1.0`` restores the legacy circular-boundary detector
        chain bit-identically. Any larger factor zero-pads the sensor-plane
        field before the pixel-stack transfer, in-Si H completion and
        Poynting product, making that stage a linear (non-circular)
        operator; the active-area weight is applied after cropping back.
        Padded auxiliary buffers are non-persistent, so the state dict is
        unchanged in either mode.
        """
        if not getattr(self, "_has_pixel_stack", False):
            raise RuntimeError(
                "detector FFT padding requires a configured pixel stack"
            )
        factor = float(pad_factor)
        if not math.isfinite(factor) or factor < 1.0:
            raise ValueError(
                "detector_fft_pad_factor must be a finite factor >= 1.0"
            )
        height = self.sensor_grid.height
        width = self.sensor_grid.width
        pad_h = int(round(height * factor))
        pad_w = int(round(width * factor))
        if abs(height * factor - pad_h) > 1e-9 or abs(width * factor - pad_w) > 1e-9:
            raise ValueError(
                "detector_fft_pad_factor must yield integer padded sizes"
            )
        if factor == 1.0:
            self._detector_fft_pad_factor = 1.0
            self._detector_pad_hw = None
            for name in self._DETECTOR_PAD_BUFFER_NAMES:
                if name in self._buffers:
                    del self._buffers[name]
            return
        from .config import PlaneSpec
        from .pixel_stack import build_pixel_stack_buffers
        pad_grid = build_spatial_grid(
            PlaneSpec(
                name="sensor_detector_pad",
                height=pad_h,
                width=pad_w,
                pitch_um=self.sensor_grid.pitch_um,
                z_um=self.sensor_grid.z_um,
            ),
            self.device_,
            self.dtype_,
        )
        pad_buffers = build_pixel_stack_buffers(
            pad_grid, self.wavelengths_um, self._pixel_stack_config,
            device=self.device_,
        )
        pad_freq = build_frequency_grid(
            pad_h, pad_w, self.sensor_grid.pitch_um, self.device_, self.dtype_,
        )
        kx_si, ky_si, kz_safe_si = build_kz_vectors(
            pad_freq, self.wavelengths_um, self._ps_n_si,
        )
        replace = self._register_or_replace_nonpersistent_buffer
        replace("_ps_pad_stack_T_te", pad_buffers.stack_T_te)
        replace("_ps_pad_stack_T_tm", pad_buffers.stack_T_tm)
        replace("_ps_pad_radial_x", pad_buffers.radial_x)
        replace("_ps_pad_radial_y", pad_buffers.radial_y)
        replace("_ps_pad_kx_si", kx_si)
        replace("_ps_pad_ky_si", ky_si)
        replace("_ps_pad_kz_safe_si", kz_safe_si)
        replace("_ps_pad_evanescent_si", (kz_safe_si.abs() == 1.0))
        self._detector_fft_pad_factor = factor
        self._detector_pad_hw = (pad_h, pad_w)

    def _pixel_stack_detect_poynting(
        self,
        Ex_pup,
        Ey_pup,
        prop_T,
        common_meta,
        *,
        output_shift_xy_um=None,
    ):
        """Shared vector pixel-stack detector used by direct and D4 paths.

        The stack operates on each Fourier mode in its local ``s/r`` basis.
        Tangential H is completed from the post-stack E spectrum in Si, then
        E and H are inverse transformed and form Poynting flux. Only at that
        final detector stage is the active-area collection weight applied.
        """
        Ex_sens, Ey_sens = propagate_vectorial_E_cross_grid(
            Ex_pup, Ey_pup, prop_T, common_meta=common_meta,
            skip_alignment=self._transfer_alignment_fused,
            output_shift_xy_um=output_shift_xy_um)
        ml_factor = self._ps_ml_factor
        pad_hw = self._detector_pad_hw
        if pad_hw is None:
            Ex_hat = torch.fft.fft2(Ex_sens * ml_factor, dim=(-2, -1))
            Ey_hat = torch.fft.fft2(Ey_sens * ml_factor, dim=(-2, -1))
            del Ex_sens, Ey_sens

            Ex_hat, Ey_hat = apply_pixel_stack_transfer_vectorial(
                Ex_hat,
                Ey_hat,
                self._ps_stack_T_te,
                self._ps_stack_T_tm,
                self._ps_radial_x,
                self._ps_radial_y,
            )
            Hx_hat_si, Hy_hat_si = derive_H_in_medium(
                Ex_hat, Ey_hat,
                self._ps_kx_si.expand_as(Ex_hat),
                self._ps_ky_si.expand_as(Ex_hat),
                self._ps_kz_safe_si.expand_as(Ex_hat),
                self._ps_evanescent_si.expand_as(Ex_hat),
                self._ps_k_vac)

            E_stack = torch.stack([Ex_hat, Ey_hat], dim=0)
            del Ex_hat, Ey_hat
            E_real = torch.fft.ifft2(E_stack, dim=(-2, -1))
            del E_stack
            H_stack = torch.stack([Hx_hat_si, Hy_hat_si], dim=0)
            del Hx_hat_si, Hy_hat_si
            H_real = torch.fft.ifft2(H_stack, dim=(-2, -1))
            del H_stack

            Sz = (E_real[0] * H_real[1].conj()
                  - E_real[1] * H_real[0].conj()).real
            del E_real, H_real
            return apply_active_area_to_poynting(Sz, self._ps_active_mask)

        # E28 padded linear-boundary detector stage. The microlens factor is
        # applied on the native window before padding; the active-area weight
        # is applied after cropping back, in the unchanged order.
        height = self.sensor_grid.height
        width = self.sensor_grid.width
        pad_h, pad_w = pad_hw
        Ex_ml = Ex_sens * ml_factor
        Ey_ml = Ey_sens * ml_factor
        del Ex_sens, Ey_sens
        Ex_pad = torch.nn.functional.pad(Ex_ml, (0, pad_w - width, 0, pad_h - height))
        Ey_pad = torch.nn.functional.pad(Ey_ml, (0, pad_w - width, 0, pad_h - height))
        del Ex_ml, Ey_ml
        Ex_hat = torch.fft.fft2(Ex_pad, dim=(-2, -1))
        del Ex_pad
        Ey_hat = torch.fft.fft2(Ey_pad, dim=(-2, -1))
        del Ey_pad

        Ex_hat, Ey_hat = apply_pixel_stack_transfer_vectorial(
            Ex_hat,
            Ey_hat,
            self._ps_pad_stack_T_te,
            self._ps_pad_stack_T_tm,
            self._ps_pad_radial_x,
            self._ps_pad_radial_y,
        )
        Hx_hat_si, Hy_hat_si = derive_H_in_medium(
            Ex_hat, Ey_hat,
            self._ps_pad_kx_si.expand_as(Ex_hat),
            self._ps_pad_ky_si.expand_as(Ex_hat),
            self._ps_pad_kz_safe_si.expand_as(Ex_hat),
            self._ps_pad_evanescent_si.expand_as(Ex_hat),
            self._ps_k_vac)

        E_stack = torch.stack([Ex_hat, Ey_hat], dim=0)
        del Ex_hat, Ey_hat
        E_real = torch.fft.ifft2(E_stack, dim=(-2, -1))
        del E_stack
        H_stack = torch.stack([Hx_hat_si, Hy_hat_si], dim=0)
        del Hx_hat_si, Hy_hat_si
        H_real = torch.fft.ifft2(H_stack, dim=(-2, -1))
        del H_stack

        Sz = (E_real[0] * H_real[1].conj()
              - E_real[1] * H_real[0].conj()).real
        del E_real, H_real
        Sz = Sz[..., :height, :width]
        return apply_active_area_to_poynting(Sz, self._ps_active_mask)

    def forward_optics_from_object_batch(
        self,
        object_batch: ObjectPointBatch,
        *,
        width_map_override: torch.Tensor | None = None,
        progress: bool = False,
        accelerated_4fold: bool = False,
        accelerated_8fold: bool = False,
        sensor_field_shift_xy_um: torch.Tensor | None = None,
        reflection_fold_authorization: ReflectionFoldAuthorization | None = None,
    ) -> torch.Tensor:
        width_map = self.width_map_um if width_map_override is None else width_map_override
        if self._require_full_jones_response:
            # Re-check because some archived scripts replace response_model on
            # an existing engine instance between reference/design renders.
            self._validate_final_exact_response()
            if accelerated_4fold or accelerated_8fold:
                _validate_runtime_fold_authorization(
                    reflection_fold_authorization,
                    engine=self,
                    width_map=width_map,
                    eightfold=bool(accelerated_8fold),
                )
        # Optional 4× speedup: exploit lens 4-fold mirror symmetry to compute
        # PSF for Q1 obj points only and reuse via flips for Q2/Q3/Q4. Requires
        # the input object_batch to tile a centered, even-N-on-each-axis grid
        # with NO points on the x=0 or y=0 axes (true for the standard scene
        # forward flatten of an even-N centered object grid).
        if accelerated_4fold or accelerated_8fold:
            if sensor_field_shift_xy_um is not None:
                raise ValueError(
                    "sensor_field_shift_xy_um is not implemented for the "
                    "D4/D8 accelerated scene path"
                )
            return self._forward_optics_q1_4fold(
                object_batch,
                width_map_override=width_map_override,
                progress=progress,
                eightfold=accelerated_8fold,
            )
        accumulated = torch.zeros(
            (self.wavelengths_um.numel(), self.sensor_grid.height, self.sensor_grid.width),
            device=self.device_,
            dtype=torch.float32,
        )
        ap_mask = self.aperture_mask.view(1, 1, *self.aperture_mask.shape).to(torch.complex64)
        # Cross-grid (Option D) common-Fourier-grid propagation: pupil-pad spectrum is
        # embedded into the common pad spectrum (which lives at sensor pitch), AS
        # transferred, Ez/Hx/Hy derived, IFFT'd, then center-cropped to sensor native.
        # When pupil and sensor are identical this collapses to legacy single-grid
        # propagation bit-identically.
        prop_T = self._propagation_transfer_common
        prop_kx = self._kx_common
        prop_ky = self._ky_common
        prop_kz = self._kz_safe_common
        common_meta = self._common_meta
        incident_medium_index = self._incident_medium_index()
        output_medium_index = self._output_medium_index()

        n_pts = object_batch.coords_um.shape[0]
        field_shifts = None
        if sensor_field_shift_xy_um is not None:
            field_shifts = torch.as_tensor(
                sensor_field_shift_xy_um,
                device=self.device_,
                dtype=torch.float32,
            )
            if tuple(field_shifts.shape) != (int(n_pts), 2):
                raise ValueError(
                    "sensor_field_shift_xy_um must have shape [object_points,2]"
                )
            if not bool(torch.isfinite(field_shifts.detach()).all()):
                raise ValueError("sensor_field_shift_xy_um must be finite")
        cs = self.spec.chunk_size
        starts = list(range(0, n_pts, cs))
        if progress:
            try:
                from tqdm import tqdm  # local import to avoid hard dep
                starts = tqdm(starts, desc=f"forward_optics ({n_pts} pts, chunk={cs})",
                              total=len(starts))
            except ImportError:
                pass
        for start in starts:
            end = min(start + self.spec.chunk_size, object_batch.coords_um.shape[0])
            chunk = slice_object_batch(object_batch, start, end)
            shift_chunk = (
                None if field_shifts is None else field_shifts[start:end]
            )
            incident = compute_incident_field(
                chunk,
                self.pupil_grid,
                self.wavelengths_um,
                refractive_index=incident_medium_index,
            )
            s_local = self.response_model(
                width_map,
                self.wavelengths_um,
                incident.theta_x_rad,
                incident.theta_y_rad,
            )

            if _is_vectorial_response(s_local):
                # Vectorial, unpolarized: diagonal TE/TM archive responses and
                # full 2x2 Jones responses share the same three-dipole source
                # model. Full-Jones cross terms mix coherently per dipole.
                Ex_pup, Ey_pup = _assemble_vectorial_response_pupil_fields(
                    s_local,
                    ap_mask * incident.field,
                    incident.theta_x_rad,
                    incident.theta_y_rad,
                    incident_refractive_index=incident_medium_index,
                    output_refractive_index=output_medium_index,
                )
                chunk = ObjectPointBatch(
                    coords_um=chunk.coords_um.repeat(3, *([1] * (chunk.coords_um.dim() - 1))),
                    z_um=chunk.z_um.repeat(3, *([1] * (chunk.z_um.dim() - 1))),
                    spectral_radiance=chunk.spectral_radiance.repeat(
                        3, *([1] * (chunk.spectral_radiance.dim() - 1))),
                )
                vector_shift_chunk = (
                    None if shift_chunk is None else shift_chunk.repeat(3, 1)
                )

                # Cross-grid vectorial propagation: 1 FFT (pupil-pad) + 1 IFFT
                # (common-pad), with spectrum embedded into common grid before AS.
                # Returns (Ex, Ey, Ez, Hx, Hy) on sensor native grid.
                if self._has_pixel_stack:
                    # Sub-batched detection (see _pixel_stack_detect_poynting):
                    # the common-grid IFFT tensors are the VRAM peak — spilling
                    # past the card costs ~100x. Slice the (3-dipole x chunk)
                    # batch; the python loop overhead amortizes over chunk_size.
                    B_tot = Ex_pup.shape[0]
                    sub = _vectorial_subbatch()
                    for s0 in range(0, B_tot, sub):
                        s1 = min(s0 + sub, B_tot)
                        Sz = self._pixel_stack_detect_poynting(
                            Ex_pup[s0:s1], Ey_pup[s0:s1], prop_T, common_meta,
                            output_shift_xy_um=(
                                None
                                if vector_shift_chunk is None
                                else vector_shift_chunk[s0:s1]
                            ))
                        weights = chunk.spectral_radiance[s0:s1].to(
                            Sz.dtype)[..., None, None]
                        accumulated = accumulated + 0.5 * (Sz * weights).sum(dim=0)
                        del Sz, weights
                    del Ex_pup, Ey_pup
                    continue

                # Flat-pixel path: full 5-channel propagation (Ez/H needed at
                # the sensor plane), then the SAME transfer on E and H so that
                # S_z scales by |T|^2 (matches the scalar branch) and a pure
                # phase mask leaves power invariant.
                Ex_sens, Ey_sens, Ez_sens, Hx_sens, Hy_sens = propagate_vectorial_with_H_cross_grid(
                    Ex_pup, Ey_pup, prop_T,
                    prop_kx, prop_ky, prop_kz, self._k_lambda,
                    common_meta=common_meta,
                    evanescent_mask=self._evanescent_common,
                    skip_alignment=self._transfer_alignment_fused,
                    output_shift_xy_um=vector_shift_chunk,
                )
                Ex_use = apply_sensor_stack(Ex_sens, microlens_transfer=self.microlens_transfer, stack_transfer=self.stack_transfer)
                Ey_use = apply_sensor_stack(Ey_sens, microlens_transfer=self.microlens_transfer, stack_transfer=self.stack_transfer)
                Hx_pd = apply_sensor_stack(Hx_sens, microlens_transfer=self.microlens_transfer, stack_transfer=self.stack_transfer)
                Hy_pd = apply_sensor_stack(Hy_sens, microlens_transfer=self.microlens_transfer, stack_transfer=self.stack_transfer)

                # Photodiode signal: z-component of the time-averaged Poynting
                # vector S_z = (1/2) Re(E × H*) · ẑ. With the s/p transverse
                # projection the photometric falloff is the physical cos³ law
                # for an isotropic point source (1/r² solid angle + obliquity),
                # verified at 2.95 exponent (check_vectorial_cos4_law).
                accumulated = accumulated + 0.5 * vectorial_poynting_accumulate(
                    Ex_use, Ey_use, Hx_pd, Hy_pd, chunk.spectral_radiance,
                )
            else:
                # Single polarization (TE or TM): scalar path via cross-grid propagation
                if getattr(self, "_has_pixel_stack", False):
                    raise ValueError(
                        "pixel_stack_config is set but the response model "
                        "returned a 4D scalar response — the scalar path "
                        "would silently skip the pixel stack (ML phase, "
                        "stack TMM, active-area collection, Si detection). "
                        "Use a diagonal [TE,TM] or full-Jones response, or "
                        "drop the pixel stack.")
                u_pup_out = ap_mask * s_local * incident.field
                u_sensor = propagate_angular_spectrum_cross_grid(
                    u_pup_out, prop_T, common_meta=common_meta,
                    skip_alignment=self._transfer_alignment_fused,
                    output_shift_xy_um=shift_chunk,
                )
                u_pd = apply_sensor_stack(
                    u_sensor,
                    microlens_transfer=self.microlens_transfer,
                    stack_transfer=self.stack_transfer,
                )
                accumulated = accumulated + incoherent_accumulate(u_pd, chunk.spectral_radiance)

        # NOTE: previously called torch.cuda.empty_cache() here. Removed for speed:
        # it forces allocator re-init on next call (1.05-1.15x penalty per call).
        return accumulated

    def _forward_optics_q1_4fold(
        self,
        object_batch: ObjectPointBatch,
        *,
        width_map_override: torch.Tensor | None = None,
        progress: bool = False,
        eightfold: bool = False,
    ) -> torch.Tensor:
        """Four-image reflection fold for a D2-symmetric optical system.

        Only Q1 object points are propagated; the other three orbit members
        are accumulated by detector-plane reflections. This is valid for a
        width map independently symmetric about the pupil x and y axes. The
        optional eight-image path additionally requires diagonal (x/y-
        transpose) symmetry. These map checks are necessary but not sufficient
        for a full-Jones response: production full-Jones callers remain blocked
        until LUT- and checkpoint-bound end-to-end parity evidence exists.
        """
        # A detector-plane flip is not a valid substitute for propagating a
        # mirrored object point unless the physical pupil is mirrored too.
        # The historical path checked transposition only for the eight-image
        # branch and could silently fold an arbitrary non-symmetric map.
        width_map = (
            self.width_map_um if width_map_override is None else width_map_override
        )
        if width_map.ndim != 2:
            raise ValueError("reflection-fold width map must be two-dimensional")
        reflection_deviations = {
            "x": float((width_map - torch.flip(width_map, dims=(1,))).abs().max()),
            "y": float((width_map - torch.flip(width_map, dims=(0,))).abs().max()),
        }
        reflection_tolerance_um = 1.0e-7
        if max(reflection_deviations.values()) > reflection_tolerance_um:
            raise ValueError(
                "accelerated_4fold requires an x/y-mirror-symmetric width map "
                "(max deviations: "
                f"x={reflection_deviations['x']:.2e} um, "
                f"y={reflection_deviations['y']:.2e} um); use the direct path."
            )
        aperture_deviation = max(
            float(
                (
                    self.aperture_mask
                    - torch.flip(self.aperture_mask, dims=(1,))
                ).abs().max()
            ),
            float(
                (
                    self.aperture_mask
                    - torch.flip(self.aperture_mask, dims=(0,))
                ).abs().max()
            ),
        )
        if aperture_deviation != 0.0:
            raise ValueError(
                "accelerated_4fold requires an exactly x/y-mirror-symmetric "
                f"aperture mask (max deviation={aperture_deviation:.2e})"
            )

        # Split full obj batch into Q1 + 4-quadrant radiances
        coords = object_batch.coords_um
        rad = object_batch.spectral_radiance
        z = object_batch.z_um
        x, y = coords[:, 0], coords[:, 1]
        is_q1 = (x > 0) & (y > 0)
        is_q2 = (x < 0) & (y > 0)
        is_q3 = (x > 0) & (y < 0)
        is_q4 = (x < 0) & (y < 0)
        nq = is_q1.sum().item()
        if not (nq == is_q2.sum().item() == is_q3.sum().item() == is_q4.sum().item()):
            raise ValueError("accelerated_4fold requires a 4-fold-symmetric object grid "
                             "(centered, even N on each axis, no points on x=0 or y=0).")

        # Sort each quadrant by (|y|, |x|) so position match is deterministic.
        # E12: exact float64 lexsort — the previous float32 composite key
        # (cy*1e6 + cx) lost all |x| information for |y| beyond ~650 um
        # (float32 ULP ~1024 um at 1.5e10), and the unstable argsort then
        # paired ~75% of Q2/Q3/Q4 points with the WRONG mirror radiance at
        # stage5-scale extents. lexsort is stable and exact in float64.
        import numpy as _np
        def _sort_key_idx(mask):
            idx = torch.where(mask)[0]
            cx = coords[idx, 0].abs().cpu().numpy().astype(_np.float64)
            cy = coords[idx, 1].abs().cpu().numpy().astype(_np.float64)
            order = _np.lexsort((cx, cy))
            return idx[order.tolist()]
        idx_q1 = _sort_key_idx(is_q1)
        idx_q2 = _sort_key_idx(is_q2)
        idx_q3 = _sort_key_idx(is_q3)
        idx_q4 = _sort_key_idx(is_q4)

        # E12 guard: the mirror pairing must be exact (centered grid ->
        # exact float equality of |coords|). Any violation fails loudly.
        _a1 = coords[idx_q1].abs()
        for _idx_m in (idx_q2, idx_q3, idx_q4):
            if not torch.equal(coords[_idx_m].abs(), _a1):
                raise ValueError(
                    "accelerated_4fold quadrant pairing mismatch: |coords| of a "
                    "mirror quadrant do not exactly match Q1 after lexsort — "
                    "object grid is not 4-fold mirror symmetric.")

        q1_coords = coords[idx_q1]
        q1_z = z[idx_q1]
        rad_q1 = rad[idx_q1]
        rad_q2 = rad[idx_q2]
        rad_q3 = rad[idx_q3]
        rad_q4 = rad[idx_q4]

        # ---- 8-fold (D4): forward only the lower octant |y| <= |x| of Q1 and
        # reuse the upper octant via Sz transpose. Exact when the width map is
        # also transpose-symmetric (radial designs on the square lattice are).
        if eightfold:
            tdev = (width_map - width_map.transpose(0, 1)).abs().max().item()
            if tdev > 1e-7:
                raise ValueError(
                    f"accelerated_8fold requires a transpose-symmetric width map "
                    f"(max |W - W^T| = {tdev:.2e}); use accelerated_4fold instead.")
            ax = q1_coords[:, 0].abs()
            ay = q1_coords[:, 1].abs()
            # partner[i] = index of the transposed point (|x|,|y|) -> (|y|,|x|)
            keys = {}
            kx_r = torch.round(ax * 1e6).long().tolist()
            ky_r = torch.round(ay * 1e6).long().tolist()
            for i, (kx_i, ky_i) in enumerate(zip(kx_r, ky_r)):
                keys[(ky_i, kx_i)] = i          # keyed by (|y|,|x|)
            partner = torch.tensor(
                [keys[(kx_i, ky_i)] for kx_i, ky_i in zip(kx_r, ky_r)],
                dtype=torch.long)
            lower = ay <= ax                     # forwarded octant (incl. diagonal)
            fwd_idx = torch.where(lower)[0]
            has_partner = (ay[fwd_idx] < ax[fwd_idx])   # strict: diagonal has none
            partner_f = partner[fwd_idx.cpu()]
            q1_coords = q1_coords[fwd_idx]
            q1_z = q1_z[fwd_idx]
            fwd_idx_cpu = fwd_idx.cpu()
        else:
            fwd_idx_cpu = None

        # Reuse engine internals via the chunked path on Q1-only batch, but
        # capture per-point Sz before accumulation by inlining the chunk loop.
        from .object_to_pupil import compute_incident_field
        from .propagation import (
            propagate_vectorial_with_H_cross_grid,
            propagate_angular_spectrum_cross_grid,
        )
        from .sensor_stack import apply_sensor_stack
        from .types import ObjectPointBatch

        accumulated = torch.zeros(
            (self.wavelengths_um.numel(), self.sensor_grid.height, self.sensor_grid.width),
            device=self.device_, dtype=torch.float32,
        )
        # transpose-pending accumulator (8-fold): collects the upper-octant
        # orbit contributions computed from lower-octant Sz; transposed ONCE at
        # the end (flip_x∘T = T∘flip_y ⇒ swap the q2/q3 flip-radiance pairing).
        acc_T = torch.zeros_like(accumulated) if eightfold else None
        ap_mask = self.aperture_mask.view(1, 1, *self.aperture_mask.shape).to(torch.complex64)

        prop_T = self._propagation_transfer_common
        prop_kx = self._kx_common
        prop_ky = self._ky_common
        prop_kz = self._kz_safe_common
        common_meta = self._common_meta
        incident_medium_index = self._incident_medium_index()
        output_medium_index = self._output_medium_index()

        cs = self.spec.chunk_size
        starts = list(range(0, q1_coords.shape[0], cs))
        if progress:
            try:
                from tqdm import tqdm
                starts = tqdm(starts, desc=f"forward_4fold (Q1 {q1_coords.shape[0]} pts, chunk={cs})",
                              total=len(starts))
            except ImportError:
                pass

        for start in starts:
            end = min(start + cs, q1_coords.shape[0])
            chunk = ObjectPointBatch(
                coords_um=q1_coords[start:end],
                z_um=q1_z[start:end],
                spectral_radiance=rad_q1[start:end],   # placeholder, ignored
            )
            incident = compute_incident_field(
                chunk,
                self.pupil_grid,
                self.wavelengths_um,
                refractive_index=incident_medium_index,
            )
            s_local = self.response_model(
                width_map, self.wavelengths_um,
                incident.theta_x_rad, incident.theta_y_rad,
            )
            if _is_vectorial_response(s_local):
                # Same explicit diagonal/full-Jones dispatch as the direct
                # path; keeping this assembly shared is required for D4 parity.
                Ex_pup, Ey_pup = _assemble_vectorial_response_pupil_fields(
                    s_local,
                    ap_mask * incident.field,
                    incident.theta_x_rad,
                    incident.theta_y_rad,
                    incident_refractive_index=incident_medium_index,
                    output_refractive_index=output_medium_index,
                )
                if self._has_pixel_stack:
                    # shared detection sequence, sub-batched for VRAM (per-point
                    # Sz is needed for the flip reuse, so collect slices)
                    B_tot = Ex_pup.shape[0]
                    sub = _vectorial_subbatch()
                    sz_parts = []
                    for s0 in range(0, B_tot, sub):
                        s1 = min(s0 + sub, B_tot)
                        Sz_part = self._pixel_stack_detect_poynting(
                            Ex_pup[s0:s1], Ey_pup[s0:s1], prop_T, common_meta)
                        sz_parts.append(Sz_part)
                    Sz = torch.cat(sz_parts, dim=0)
                    del sz_parts
                else:
                    Ex_sens, Ey_sens, Ez_sens, Hx_sens, Hy_sens = propagate_vectorial_with_H_cross_grid(
                        Ex_pup, Ey_pup, prop_T, prop_kx, prop_ky, prop_kz, self._k_lambda,
                        common_meta=common_meta,
                        evanescent_mask=self._evanescent_common,
                        skip_alignment=self._transfer_alignment_fused,
                    )
                    Ex_use = apply_sensor_stack(Ex_sens, microlens_transfer=self.microlens_transfer,
                                                stack_transfer=self.stack_transfer)
                    Ey_use = apply_sensor_stack(Ey_sens, microlens_transfer=self.microlens_transfer,
                                                stack_transfer=self.stack_transfer)
                    # same transfer on H: S_z scales |T|^2, phase masks power-neutral
                    Hx_pd = apply_sensor_stack(Hx_sens, microlens_transfer=self.microlens_transfer,
                                               stack_transfer=self.stack_transfer)
                    Hy_pd = apply_sensor_stack(Hy_sens, microlens_transfer=self.microlens_transfer,
                                               stack_transfer=self.stack_transfer)
                    Sz = (Ex_use * Hy_pd.conj() - Ey_use * Hx_pd.conj()).real
                    del Ex_use, Ey_use, Hx_pd, Hy_pd, Ex_sens, Ey_sens
                    del Ez_sens, Hx_sens, Hy_sens

                # Accumulate 4 quadrant contributions one at a time so we
                # never hold 4 flipped Sz copies simultaneously.
                if eightfold:
                    sel = fwd_idx_cpu[start:end]
                    pidx = partner_f[start:end]
                    hp = has_partner[start:end].to(Sz.dtype).view(-1, 1)

                    def _r3x_at(r, idx, w=None):
                        rr = r[idx].to(Sz.dtype)
                        if w is not None:
                            rr = rr * w
                        rr = rr[..., None, None]
                        return torch.cat([rr, rr, rr], dim=0)
                    r1, r2 = _r3x_at(rad_q1, sel), _r3x_at(rad_q2, sel)
                    r3, r4 = _r3x_at(rad_q3, sel), _r3x_at(rad_q4, sel)
                    # transposed-orbit radiances (zero for diagonal points)
                    t1, t2 = _r3x_at(rad_q1, pidx, hp), _r3x_at(rad_q2, pidx, hp)
                    t3, t4 = _r3x_at(rad_q3, pidx, hp), _r3x_at(rad_q4, pidx, hp)
                    Sfx = torch.flip(Sz, dims=(-1,))
                    Sfy = torch.flip(Sz, dims=(-2,))
                    Sfxy = torch.flip(Sz, dims=(-2, -1))
                    accumulated = accumulated + 0.5 * (
                        (Sz * r1).sum(dim=0) + (Sfx * r2).sum(dim=0)
                        + (Sfy * r3).sum(dim=0) + (Sfxy * r4).sum(dim=0))
                    # for u = T(l): flip_x(T S) = T(flip_y S) etc. → q2↔q3 swap
                    acc_T = acc_T + 0.5 * (
                        (Sz * t1).sum(dim=0) + (Sfy * t2).sum(dim=0)
                        + (Sfx * t3).sum(dim=0) + (Sfxy * t4).sum(dim=0))
                    del Sfx, Sfy, Sfxy
                else:
                    def _r3x(r):
                        r = r[start:end].to(Sz.dtype)[..., None, None]
                        return torch.cat([r, r, r], dim=0)   # 3 dipoles share the point's radiance
                    r1, r2, r3, r4 = _r3x(rad_q1), _r3x(rad_q2), _r3x(rad_q3), _r3x(rad_q4)
                    accumulated = accumulated + 0.5 * (Sz * r1).sum(dim=0)
                    accumulated = accumulated + 0.5 * (torch.flip(Sz, dims=(-1,)) * r2).sum(dim=0)
                    accumulated = accumulated + 0.5 * (torch.flip(Sz, dims=(-2,)) * r3).sum(dim=0)
                    accumulated = accumulated + 0.5 * (torch.flip(Sz, dims=(-2, -1)) * r4).sum(dim=0)
                del Sz
            else:
                if getattr(self, "_has_pixel_stack", False):
                    raise ValueError(
                        "pixel_stack_config is set but the response model "
                        "returned a 4D scalar response — the scalar 4fold "
                        "path would silently skip the pixel stack. Use a "
                        "diagonal [TE,TM] or full-Jones response, or drop "
                        "the pixel stack.")
                u_pup_out = ap_mask * s_local * incident.field
                u_sensor = propagate_angular_spectrum_cross_grid(
                    u_pup_out, prop_T, common_meta=common_meta,
                    skip_alignment=self._transfer_alignment_fused,
                )
                u_pd = apply_sensor_stack(
                    u_sensor, microlens_transfer=self.microlens_transfer,
                    stack_transfer=self.stack_transfer,
                )
                irrad = (u_pd.real**2 + u_pd.imag**2)
                irrad_q1 = irrad
                irrad_q2 = torch.flip(irrad, dims=(-1,))
                irrad_q3 = torch.flip(irrad, dims=(-2,))
                irrad_q4 = torch.flip(irrad, dims=(-2, -1))
                r1 = rad_q1[start:end].to(irrad.dtype)[..., None, None]
                r2 = rad_q2[start:end].to(irrad.dtype)[..., None, None]
                r3 = rad_q3[start:end].to(irrad.dtype)[..., None, None]
                r4 = rad_q4[start:end].to(irrad.dtype)[..., None, None]
                accumulated = accumulated + (
                    (irrad_q1 * r1).sum(dim=0)
                    + (irrad_q2 * r2).sum(dim=0)
                    + (irrad_q3 * r3).sum(dim=0)
                    + (irrad_q4 * r4).sum(dim=0)
                )
        # NOTE: previously called torch.cuda.empty_cache() here. Removed for speed.
        if eightfold:
            accumulated = accumulated + acc_T.transpose(-2, -1)
        return accumulated

    def forward_optics(
        self,
        scene: torch.Tensor,
        threshold: float = 0.0,
        *,
        width_map_override: torch.Tensor | None = None,
        progress: bool = False,
        accelerated_4fold: bool = False,
        accelerated_8fold: bool = False,
        reflection_fold_authorization: ReflectionFoldAuthorization | None = None,
    ) -> torch.Tensor:
        object_batch = self.build_object_batch(scene, threshold=threshold)
        return self.forward_optics_from_object_batch(
            object_batch,
            width_map_override=width_map_override,
            progress=progress,
            accelerated_4fold=accelerated_4fold,
            accelerated_8fold=accelerated_8fold,
            reflection_fold_authorization=reflection_fold_authorization,
        )

    def forward_raw_deterministic_from_object_batch(
        self,
        object_batch: ObjectPointBatch,
        *,
        width_map_override: torch.Tensor | None = None,
        progress: bool = False,
        accelerated_4fold: bool = False,
        accelerated_8fold: bool = False,
        reflection_fold_authorization: ReflectionFoldAuthorization | None = None,
    ) -> RawDeterministicBatch:
        spectral_irradiance = self.forward_optics_from_object_batch(
            object_batch,
            width_map_override=width_map_override,
            progress=progress,
            accelerated_4fold=accelerated_4fold,
            accelerated_8fold=accelerated_8fold,
            reflection_fold_authorization=reflection_fold_authorization,
        )
        channel_spectral = channelwise_cfa_filter(spectral_irradiance, self.cfa_transmission)
        channel_pixels = pixel_integrate_channel_images(channel_spectral, self.spec.resolved_pixel_grid_shape())
        mu_rgb_e = expected_photoelectrons(
            channel_pixels,
            self.wavelengths_um,
            self.qe,
            self.spec.sensor.exposure_s,
        )
        mu_raw_e = apply_bayer_sampling(mu_rgb_e, self.bayer_masks)
        return RawDeterministicBatch(
            spectral_irradiance=spectral_irradiance,
            mu_rgb_e=mu_rgb_e,
            mu_raw_e=mu_raw_e,
        )

    def forward_raw_deterministic(
        self,
        scene: torch.Tensor,
        threshold: float = 0.0,
        *,
        width_map_override: torch.Tensor | None = None,
        progress: bool = False,
        accelerated_4fold: bool = False,
        accelerated_8fold: bool = False,
        reflection_fold_authorization: ReflectionFoldAuthorization | None = None,
    ) -> RawDeterministicBatch:
        object_batch = self.build_object_batch(scene, threshold=threshold)
        return self.forward_raw_deterministic_from_object_batch(
            object_batch,
            width_map_override=width_map_override,
            progress=progress,
            accelerated_4fold=accelerated_4fold,
            accelerated_8fold=accelerated_8fold,
            reflection_fold_authorization=reflection_fold_authorization,
        )

    def forward(
        self,
        scene: torch.Tensor,
        *,
        threshold: float = 0.0,
        add_noise: bool = True,
        demosaic: str = "none",
        width_map_override: torch.Tensor | None = None,
        accelerated_4fold: bool = False,
        accelerated_8fold: bool = False,
        reflection_fold_authorization: ReflectionFoldAuthorization | None = None,
    ) -> ForwardOutputs:
        det = self.forward_raw_deterministic(
            scene,
            threshold=threshold,
            width_map_override=width_map_override,
            accelerated_4fold=accelerated_4fold,
            accelerated_8fold=accelerated_8fold,
            reflection_fold_authorization=reflection_fold_authorization,
        )
        if add_noise:
            noisy = apply_sensor_noise(det.mu_raw_e, self.spec.sensor)
            raw_dn = noisy.raw_dn
            raw_electrons = noisy.raw_electrons
        else:
            raw_electrons = det.mu_raw_e
            raw_dn = adc_encode(det.mu_raw_e, self.spec.sensor)

        rgb = None
        if demosaic == "bilinear":
            rgb = demosaic_bilinear(raw_dn, self.bayer_masks)
        elif demosaic == "malvar":
            rgb = demosaic_malvar(raw_dn, self.bayer_masks)
        elif demosaic != "none":
            raise ValueError(f"Unsupported demosaic mode: {demosaic}")

        return ForwardOutputs(
            spectral_irradiance=det.spectral_irradiance,
            mu_rgb_e=det.mu_rgb_e,
            mu_raw_e=det.mu_raw_e,
            raw_electrons=raw_electrons,
            raw_dn=raw_dn,
            rgb=rgb,
        )

    def describe(self) -> dict[str, object]:
        return {
            "spec": asdict(self.spec),
            "device": str(self.device_),
            "wavelengths_um": self.wavelengths_um.detach().cpu().tolist(),
            "pixel_grid_shape": self.spec.resolved_pixel_grid_shape(),
            "pixel_stack_spatial_contract": self.pixel_stack_spatial_contract,
            "pixel_stack_microlens_enabled": self.pixel_stack_microlens_enabled,
            "pixel_stack_aperture_ratio": self.pixel_stack_aperture_ratio,
            "terminal_detector_contract": self.terminal_detector_contract,
            "terminal_detector_scope": (
                "interface-plane flux only; no finite-depth Si absorption"
                if self.terminal_detector_contract
                == "lossless_terminal_interface_flux_proxy_v1"
                else "not production-defined"
            ),
        }
