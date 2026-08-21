from __future__ import annotations

import math
from math import gcd

import torch

from .config import PlaneSpec
from .types import CommonGridMeta, FrequencyGrid, SpatialGrid


def build_spatial_grid(
    spec: PlaneSpec,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SpatialGrid:
    x_um = (torch.arange(spec.width, device=device, dtype=dtype) - (spec.width - 1) / 2.0) * spec.pitch_um
    y_um = (torch.arange(spec.height, device=device, dtype=dtype) - (spec.height - 1) / 2.0) * spec.pitch_um
    yy_um, xx_um = torch.meshgrid(y_um, x_um, indexing="ij")
    return SpatialGrid(
        x_um=x_um,
        y_um=y_um,
        xx_um=xx_um,
        yy_um=yy_um,
        pitch_um=spec.pitch_um,
        z_um=spec.z_um,
        height=spec.height,
        width=spec.width,
    )


def build_frequency_grid(
    height: int,
    width: int,
    pitch_um: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> FrequencyGrid:
    fx = torch.fft.fftfreq(width, d=pitch_um, device=device).to(dtype)
    fy = torch.fft.fftfreq(height, d=pitch_um, device=device).to(dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    return FrequencyGrid(
        fx_cyc_per_um=fx,
        fy_cyc_per_um=fy,
        fxx_cyc_per_um=fxx,
        fyy_cyc_per_um=fyy,
    )


def _rational_ratio(dp_um: float, ds_um: float, precision_um: float = 1e-4) -> tuple[int, int]:
    """Express dp:ds as smallest integer ratio (a, b) at given precision.
    Returns (a, b) such that a*ds == b*dp (within precision)."""
    scale = 1.0 / precision_um
    ai = int(round(dp_um * scale))
    bi = int(round(ds_um * scale))
    if ai <= 0 or bi <= 0:
        raise ValueError(f"pitches must be positive, got dp={dp_um}, ds={ds_um}")
    g = gcd(ai, bi)
    return ai // g, bi // g


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def build_common_propagation_grid(
    pupil_h: int,
    pupil_w: int,
    pupil_pitch_um: float,
    sensor_h: int,
    sensor_w: int,
    sensor_pitch_um: float,
    pad_factor: int = 1,
    precision_um: float = 1e-4,
    max_ratio_denominator: int = 2048,
) -> CommonGridMeta:
    """Find pupil-pad and common-pad shapes (Np_pad, Nc_pad) such that
    Np_pad * dp == Nc_pad * ds exactly AND both are even (FFT-friendly).
    This guarantees pupil and common spectra share the same frequency step
    (Δf = 1/L where L is the shared physical extent), so the pupil spectrum
    embeds bit-exactly into the common spectrum without any interpolation.

    Algorithm
    ---------
    Express dp:ds = b:a in lowest terms via _rational_ratio. Then for any
    integer K >= 1, np_pad = b*K and nc_pad = a*K satisfy np_pad*dp == nc_pad*ds.
    Choose K_step = lcm(2 if b%2 else 1, 2 if a%2 else 1) so that any K which is
    a multiple of K_step makes both b*K and a*K even. Then pick the smallest
    such K such that np_pad >= pupil_h * pad_factor and nc_pad >= sensor_h * pad_factor.
    Same procedure independently per axis (h, w) — anisotropic pitches in future
    are supported.

    Parameters
    ----------
    pupil_h, pupil_w : int
        Native pupil grid shape (before padding).
    pupil_pitch_um : float
        Pupil sampling pitch.
    sensor_h, sensor_w : int
        Native sensor grid shape (also the IFFT crop target).
    sensor_pitch_um : float
        Sensor sampling pitch (must divide pixel pitch evenly upstream).
    pad_factor : int
        Minimum padding factor — pupil-pad must cover at least pupil_h * pad_factor
        samples and common-pad must cover at least sensor_h * pad_factor samples.
    precision_um : float
        Pitch precision used to express dp:ds as integer ratio (default 1e-4 µm = 0.1 nm).
    max_ratio_denominator : int
        Refuse pitches whose lowest-term ratio exceeds this denominator. Adversarial
        irrational pitches would otherwise blow up grid memory.

    Returns
    -------
    CommonGridMeta with all derived dimensions.
    """
    if pad_factor < 1:
        raise ValueError(f"pad_factor must be >= 1, got {pad_factor}")
    a, b = _rational_ratio(pupil_pitch_um, sensor_pitch_um, precision_um)
    if max(a, b) > max_ratio_denominator:
        raise ValueError(
            f"pupil:sensor pitch ratio {b}:{a} (in lowest terms) exceeds "
            f"max_ratio_denominator={max_ratio_denominator}. Pitches "
            f"({pupil_pitch_um}, {sensor_pitch_um}) are too far from a "
            f"low-order rational ratio; pick commensurable pitches or "
            f"increase max_ratio_denominator (will increase FFT cost)."
        )
    # K must be a multiple of K_step so that both b*K and a*K are even
    k_step = _lcm(2 if b % 2 else 1, 2 if a % 2 else 1)

    def pick_K(target_np: int, target_nc: int) -> int:
        K_min = max(math.ceil(target_np / b), math.ceil(target_nc / a))
        # Round up K_min to nearest multiple of k_step
        return ((K_min + k_step - 1) // k_step) * k_step

    K_h = pick_K(pupil_h * pad_factor, sensor_h * pad_factor)
    K_w = pick_K(pupil_w * pad_factor, sensor_w * pad_factor)
    np_pad_h = b * K_h
    np_pad_w = b * K_w
    nc_pad_h = a * K_h
    nc_pad_w = a * K_w
    # Sanity: both even, extent equality, large enough
    assert np_pad_h % 2 == 0 and np_pad_w % 2 == 0
    assert nc_pad_h % 2 == 0 and nc_pad_w % 2 == 0
    if nc_pad_h < sensor_h or nc_pad_w < sensor_w:
        raise ValueError(
            f"common pad shape ({nc_pad_h},{nc_pad_w}) cannot be smaller than "
            f"sensor native ({sensor_h},{sensor_w})"
        )
    if np_pad_h < pupil_h or np_pad_w < pupil_w:
        raise ValueError(
            f"pupil pad shape ({np_pad_h},{np_pad_w}) cannot be smaller than "
            f"pupil native ({pupil_h},{pupil_w})"
        )
    extent_h = np_pad_h * pupil_pitch_um
    extent_h_check = nc_pad_h * sensor_pitch_um
    if abs(extent_h - extent_h_check) > 1e-9:
        raise AssertionError(
            f"extent mismatch (axis h): pupil_pad {extent_h:.9f} != "
            f"common_pad {extent_h_check:.9f}"
        )
    spectrum_amp_scale = float(nc_pad_h * nc_pad_w) / float(np_pad_h * np_pad_w)
    return CommonGridMeta(
        np_pad_h=np_pad_h, np_pad_w=np_pad_w,
        nc_pad_h=nc_pad_h, nc_pad_w=nc_pad_w,
        sensor_h=sensor_h, sensor_w=sensor_w,
        pupil_pitch_um=pupil_pitch_um, sensor_pitch_um=sensor_pitch_um,
        common_pad_extent_um=extent_h,
        spectrum_amp_scale=spectrum_amp_scale,
        pupil_h=pupil_h, pupil_w=pupil_w,
    )


def embed_pupil_field(field: torch.Tensor, np_pad_h: int, np_pad_w: int) -> torch.Tensor:
    """Zero-pad pupil field [..., Hp, Wp] -> [..., Np_pad_h, Np_pad_w] centered."""
    Hp, Wp = field.shape[-2], field.shape[-1]
    if Hp == np_pad_h and Wp == np_pad_w:
        return field
    py0 = (np_pad_h - Hp) // 2
    py1 = np_pad_h - Hp - py0
    px0 = (np_pad_w - Wp) // 2
    px1 = np_pad_w - Wp - px0
    import torch.nn.functional as F
    return F.pad(field, (px0, px1, py0, py1), mode="constant", value=0.0)


_EMBED_PHASE_CACHE: dict = {}


def _alignment_phase(nc_pad_h, nc_pad_w, Np_h, Np_w, pup_pitch_um,
                     sen_pitch_um, device, dtype) -> torch.Tensor:
    """Cached half-pixel origin-alignment phase [nc_pad_h, nc_pad_w]."""
    key = (nc_pad_h, nc_pad_w, Np_h, Np_w,
           float(pup_pitch_um), float(sen_pitch_um), str(device), str(dtype))
    phase = _EMBED_PHASE_CACHE.get(key)
    if phase is None:
        import math
        dx = ((Np_w - 1) * pup_pitch_um - (nc_pad_w - 1) * sen_pitch_um) / 2.0
        dy = ((Np_h - 1) * pup_pitch_um - (nc_pad_h - 1) * sen_pitch_um) / 2.0
        fxc = torch.fft.fftfreq(nc_pad_w, d=sen_pitch_um, device=device)
        fyc = torch.fft.fftfreq(nc_pad_h, d=sen_pitch_um, device=device)
        phase = torch.exp(1j * 2 * math.pi * (fxc[None, :].to(torch.float64) * dx
                                              + fyc[:, None].to(torch.float64) * dy)).to(dtype)
        _EMBED_PHASE_CACHE[key] = phase
    return phase


def common_alignment_factor(common_meta, device,
                            dtype=torch.complex64) -> torch.Tensor:
    """P6: (alignment phase × amp_scale) [nc_pad_h, nc_pad_w] for pre-fusing
    into a static transfer function. Callers that fuse this into
    transfer_common must call the propagate_* functions with
    ``skip_alignment=True`` so the factor is applied exactly once.
    """
    phase = _alignment_phase(
        common_meta.nc_pad_h, common_meta.nc_pad_w,
        common_meta.np_pad_h, common_meta.np_pad_w,
        common_meta.pupil_pitch_um, common_meta.sensor_pitch_um,
        device, dtype)
    return phase * common_meta.spectrum_amp_scale


def embed_spectrum_into_common(
    spec_pup: torch.Tensor,
    nc_pad_h: int,
    nc_pad_w: int,
    amp_scale: float,
    *,
    pup_pitch_um: float | None = None,
    sen_pitch_um: float | None = None,
    skip_alignment: bool = False,
) -> torch.Tensor:
    """Embed pupil spectrum [..., Np_pad_h, Np_pad_w] into common spectrum
    [..., nc_pad_h, nc_pad_w] DC-centered (same frequency step assumed).

    Multiplies by amp_scale = (nc_pad_h * nc_pad_w) / (np_pad_h * np_pad_w)
    so that subsequent IFFT on the common grid recovers the same continuous
    physical field amplitude as IFFT on the pupil-pad grid.

    Both np_pad and nc_pad shapes must be EVEN (enforced upstream by
    build_common_propagation_grid) so that fftshift centering is unambiguous.

    Half-pixel origin alignment correction (when both pitches are provided):
        Pupil-pad centered grid origin (in physical x) sits at index
        (Np_pad-1)/2 * pup_pitch from index 0.  Common-pad origin sits at
        (Nc_pad-1)/2 * sen_pitch.  When pitches differ, these origins are
        offset by Δx = ((Np_pad-1)*pup_pitch - (Nc_pad-1)*sen_pitch) / 2.
        We compensate by multiplying the embedded common spectrum by the
        frequency-domain phase factor exp(+i 2π f_k Δx) so that the IFFT
        produces a real-space output aligned to common's centered grid
        (rather than pupil's), preserving array-flip mirror symmetry.
    """
    Np_h, Np_w = spec_pup.shape[-2], spec_pup.shape[-1]
    if (Np_h % 2) or (Np_w % 2) or (nc_pad_h % 2) or (nc_pad_w % 2):
        raise ValueError(
            f"embed_spectrum_into_common requires even pad shapes; got "
            f"input=({Np_h},{Np_w}) target=({nc_pad_h},{nc_pad_w})"
        )
    if Np_h == nc_pad_h and Np_w == nc_pad_w:
        # degenerate grids: amp_scale == 1 and alignment phase == 1, so the
        # skip_alignment contract is trivially satisfied either way
        return spec_pup if skip_alignment else spec_pup * amp_scale
    if nc_pad_h < Np_h or nc_pad_w < Np_w:
        raise ValueError(
            f"embed_spectrum_into_common: target common grid ({nc_pad_h},{nc_pad_w}) "
            f"is SMALLER than the source spectrum ({Np_h},{Np_w}) — spectrum "
            f"truncation (sensor pitch coarser than pupil pitch) is not "
            f"implemented; choose sensor pitch <= pupil pitch.")
    # Direct 4-corner embedding in the NATURAL (unshifted) fft layout —
    # algebraically identical to fftshift -> centered copy -> ifftshift but
    # without the two full-tensor rolls (which dominated the shift cost).
    # For even Np the fftshift convention assigns the Nyquist bin (index Np/2)
    # to the NEGATIVE-frequency block, so the split is [0:Np/2) and [Np/2:Np).
    h2 = Np_h // 2
    w2 = Np_w // 2
    out_natural = torch.zeros(
        (*spec_pup.shape[:-2], nc_pad_h, nc_pad_w),
        dtype=spec_pup.dtype,
        device=spec_pup.device,
    )
    out_natural[..., :h2, :w2] = spec_pup[..., :h2, :w2]
    out_natural[..., :h2, nc_pad_w - (Np_w - w2):] = spec_pup[..., :h2, w2:]
    out_natural[..., nc_pad_h - (Np_h - h2):, :w2] = spec_pup[..., h2:, :w2]
    out_natural[..., nc_pad_h - (Np_h - h2):, nc_pad_w - (Np_w - w2):] = spec_pup[..., h2:, w2:]
    if skip_alignment:
        # P6: caller pre-fused (amp_scale × alignment phase) into its static
        # transfer function — skip the two full-common-grid multiplies here
        # (each ~3.5 GB of traffic per component at d500).
        return out_natural
    out_natural = out_natural * amp_scale

    # Half-pixel alignment correction (only when both pitches given).
    if pup_pitch_um is not None and sen_pitch_um is not None:
        phase = _alignment_phase(
            nc_pad_h, nc_pad_w, Np_h, Np_w, pup_pitch_um, sen_pitch_um,
            out_natural.device, out_natural.dtype)
        out_natural = out_natural * phase
    return out_natural


def crop_common_to_sensor(field_common: torch.Tensor, sensor_h: int, sensor_w: int) -> torch.Tensor:
    """Center-crop common-grid real-space field [..., Nc_pad_h, Nc_pad_w]
    -> [..., sensor_h, sensor_w]."""
    Nc_h, Nc_w = field_common.shape[-2], field_common.shape[-1]
    if Nc_h == sensor_h and Nc_w == sensor_w:
        return field_common
    py0 = (Nc_h - sensor_h) // 2
    px0 = (Nc_w - sensor_w) // 2
    return field_common[..., py0:py0 + sensor_h, px0:px0 + sensor_w]
