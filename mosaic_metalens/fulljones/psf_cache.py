"""Spatially-varying PSF cache + multi-PSF Wiener reconstruction.

Workflow
--------
1. ``compute_psf_cache(engine, x_grid_um, y_grid_um, ...)`` evaluates the
   engine forward at a sparse grid of object-plane field points (Ky × Kx)
   using single unit-radiance point sources. Returns the per-field, per-λ
   spectral irradiance ``[Ky, Kx, N_λ, Hs, Ws]`` at the SENSOR-OPTICAL grid
   (so it can be averaged into pixel-grid PSFs by the caller). This is the
   PSF the user asked about — same physics as forward, but kept per-point.

2. ``cache_to_pixel_psfs(psf_cache, engine, ...)`` integrates the cache to
   per-channel pixel-level PSFs ``[Ky, Kx, 3, Hp, Wp]`` (R/G/B raw-domain
   signal expected from a unit on-axis source at each cached field point):
   apply CFA × pixel-binning × QE × exposure to each cached PSF and Bayer-
   sample → demosaic-bilinear so each cache entry yields the per-channel
   IRF a real image patch would experience at that field.

3. ``spatially_varying_wiener_recon(rgb_image, psf_pixel_cache, ...)``
   performs reconstruction:
   * For each cached PSF index (i, j), build a Wiener kernel ``W_ij`` and
     deconvolve the WHOLE image as if shift-invariant, giving R_ij(x, y).
   * Compose final image with bilinear blending: at position (x, y), find
     the surrounding cell (i, j .. i+1, j+1) of cached PSFs and blend
     R_{ij}(x, y) with bilinear weights matching (x, y)'s position in the
     field grid. This is overlap-blended multi-PSF Wiener — fast (4×Hp×Wp
     extra mem, 4 FFT-shaped image multiplies per output pixel) and seamless.

The cache is built once per width-map; reconstruction is cheap. For
on-axis-symmetric metalenses (radial param), 4 quadrants of the field grid
collapse, so K=4 covers a quadrant adequately. For full-DOF or C4 width
maps, K=8 covers ±half-field with 8×8 = 64 unique PSFs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .types import ObjectPointBatch

if TYPE_CHECKING:
    from .pipeline_forward import MetalensImagingEngine


@dataclass
class PSFCache:
    """Per-field, per-channel pixel-domain PSF cache."""
    field_x_um: torch.Tensor          # [Kx]   x-coordinates on object plane
    field_y_um: torch.Tensor          # [Ky]   y-coordinates on object plane
    psf_rgb_pixel: torch.Tensor       # [Ky, Kx, 3, Hp, Wp] float32, normalized
    psf_irrad_optical: torch.Tensor | None = None  # [Ky, Kx, N_λ, Hs, Ws] (optional, large)
    pixel_pitch_um: float = 1.0
    object_z_um: float = 0.0
    description: str = ""
    # Production caches are indexed in the physical sensor frame.  The two
    # separable axes determine spatial blending and the replayed chief-ray
    # centres determine one common registration shift for all colour channels.
    # ``None`` preserves the explicitly archived object/paraxial convention.
    sensor_x_um: torch.Tensor | None = None       # [Kx], strictly increasing
    sensor_y_um: torch.Tensor | None = None       # [Ky], strictly increasing
    chief_ray_xy_um: torch.Tensor | None = None   # [Ky,Kx,2], physical (x,y)
    coordinate_mode: str = "legacy_object_paraxial"


@torch.no_grad()
def _single_point_object_batch(
    engine: "MetalensImagingEngine",
    x_obj_um: float,
    y_obj_um: float,
    radiance_value: float = 1.0e10,
) -> ObjectPointBatch:
    z_obj_um = float(engine.spec.object_plane.z_um)
    n_wl = int(engine.wavelengths_um.numel())
    coords = torch.tensor([[x_obj_um, y_obj_um]], dtype=torch.float32, device=engine.device_)
    z = torch.tensor([z_obj_um], dtype=torch.float32, device=engine.device_)
    radiance = torch.full((1, n_wl), float(radiance_value),
                          dtype=torch.float32, device=engine.device_)
    return ObjectPointBatch(coords_um=coords, z_um=z, spectral_radiance=radiance)


@torch.no_grad()
def compute_psf_cache(
    engine: "MetalensImagingEngine",
    *,
    Kx: int = 8,
    Ky: int = 8,
    extent_um: tuple[float, float] | None = None,
    width_map_override: torch.Tensor | None = None,
    keep_optical: bool = False,
    progress: bool = True,
) -> PSFCache:
    """Build per-field PSF cache by sweeping single-point sources.

    Parameters
    ----------
    engine : MetalensImagingEngine
    Kx, Ky : int
        Number of cached field points along (x, y). Total Ky·Kx forwards.
    extent_um : (x_extent, y_extent), optional
        Half-extent on object plane. Defaults to ``OBJ_SIZE_UM/2`` from
        engine.spec.object_plane (i.e. ``(width-1) * pitch / 2``).
    width_map_override : Tensor [Hp, Wp], optional
        Use this width map instead of engine.width_map_um.
    keep_optical : bool
        If True, also keep the raw [Ky, Kx, N_λ, Hs, Ws] sensor-optical PSF
        (memory-heavy: K²·N_λ·Hs² · 4 bytes; for K=8, N_λ=6, Hs=832 ≈ 1 GB).
    progress : bool
        Print progress per row.

    Returns
    -------
    PSFCache. Per-channel pixel PSF normalized so sum_xy(psf_rgb_pixel[k, c, :, :]) = 1.
    """
    obj_plane = engine.spec.object_plane
    if extent_um is None:
        ext_x = (obj_plane.width - 1) * 0.5 * obj_plane.pitch_um
        ext_y = (obj_plane.height - 1) * 0.5 * obj_plane.pitch_um
    else:
        ext_x, ext_y = extent_um
    fx = torch.linspace(-ext_x, ext_x, Kx, device=engine.device_)
    fy = torch.linspace(-ext_y, ext_y, Ky, device=engine.device_)

    pix_h, pix_w = engine.spec.resolved_pixel_grid_shape()
    psf_pixel = torch.zeros((Ky, Kx, 3, pix_h, pix_w), dtype=torch.float32,
                             device=engine.device_)
    if keep_optical:
        n_wl = int(engine.wavelengths_um.numel())
        Hs, Ws = engine.sensor_grid.height, engine.sensor_grid.width
        psf_optical = torch.zeros((Ky, Kx, n_wl, Hs, Ws), dtype=torch.float32,
                                   device=engine.device_)
    else:
        psf_optical = None

    for i in range(Ky):
        for j in range(Kx):
            batch = _single_point_object_batch(engine, float(fx[j]), float(fy[i]))
            det = engine.forward_raw_deterministic_from_object_batch(
                batch, width_map_override=width_map_override,
            )
            # mu_rgb_e is [3, Hp, Wp]: per-channel pixel-domain PSF (electrons).
            psf_e = det.mu_rgb_e.detach()
            # Normalize each channel so sum = 1 (unit-energy PSF).
            psf_e_sum = psf_e.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            psf_pixel[i, j] = psf_e / psf_e_sum
            if keep_optical:
                psf_optical[i, j] = det.spectral_irradiance.detach()
            del det
        if progress:
            print(f"[psf-cache] row {i+1}/{Ky} done")
        torch.cuda.empty_cache()

    z_obj_um = float(engine.spec.object_plane.z_um)
    pix_pitch = float(engine.spec.sensor_plane.pitch_um) * (
        engine.sensor_grid.width / pix_w  # sensor optical → pixel pitch ratio
    )
    return PSFCache(
        field_x_um=fx.detach().cpu(),
        field_y_um=fy.detach().cpu(),
        psf_rgb_pixel=psf_pixel.detach().cpu(),
        psf_irrad_optical=(psf_optical.detach().cpu() if keep_optical else None),
        pixel_pitch_um=pix_pitch,
        object_z_um=z_obj_um,
        description=f"K={Ky}x{Kx} cached PSFs at object plane",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spatially-varying multi-PSF Wiener reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _translate_zero_fill_2d(
    image: torch.Tensor, shift_y: int, shift_x: int,
) -> torch.Tensor:
    """Translate the last two dimensions without circular wraparound."""
    H, W = image.shape[-2:]
    out = torch.zeros_like(image)
    src_y0 = max(0, -shift_y)
    src_y1 = min(H, H - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(W, W - shift_x)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0 = src_y0 + shift_y
    dst_y1 = src_y1 + shift_y
    dst_x0 = src_x0 + shift_x
    dst_x1 = src_x1 + shift_x
    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = image[
        ..., src_y0:src_y1, src_x0:src_x1
    ]
    return out


def _center_psf_to_origin(
    psf: torch.Tensor,
    *,
    common_center_yx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center each field's channel group with one common spatial shift.

    ``psf`` is interpreted as ``[..., C, H, W]``.  By default the common
    location is the intensity centroid of the sum over all channels.  The same
    integer, zero-filled translation is applied to every channel, preserving
    lateral chromatic offsets.  A caller that has calibrated chief-ray
    locations may supply ``common_center_yx`` with shape ``[..., 2]``.

    Off-axis PSFs include the field-dependent image displacement.  Removing
    its common component decouples placement (handled by the field-grid blend)
    from blur, while deliberately retaining channel-to-channel displacement.

    Parameters
    ----------
    psf : Tensor [..., C, H, W]
    common_center_yx : Tensor [..., 2], optional
        Common ``(y, x)`` location for each field.  If omitted, use the joint
        channel centroid.

    Returns
    -------
    Centered PSF, same shape.
    """
    if psf.ndim < 2:
        raise ValueError("psf must have at least two spatial dimensions")
    original_shape = psf.shape
    if psf.ndim == 2:
        grouped = psf.reshape(1, 1, *psf.shape)
        batch_shape: tuple[int, ...] = ()
    else:
        *batch_shape_list, channels, H, W = psf.shape
        batch_shape = tuple(batch_shape_list)
        grouped = psf.reshape(-1, channels, H, W)
    _, channels, H, W = grouped.shape
    centered = torch.empty_like(grouped)
    cy, cx = H // 2, W // 2
    supplied = None
    if common_center_yx is not None:
        supplied = common_center_yx.to(device=psf.device, dtype=psf.dtype)
        supplied = supplied.reshape(-1, 2)
        if supplied.shape[0] != grouped.shape[0]:
            raise ValueError("common_center_yx must provide one center per PSF field")
    yy = torch.arange(H, device=psf.device, dtype=psf.dtype).view(H, 1)
    xx = torch.arange(W, device=psf.device, dtype=psf.dtype).view(1, W)
    for k in range(grouped.shape[0]):
        if supplied is None:
            weight = grouped[k].clamp_min(0).sum(dim=0)
            total = weight.sum()
            if float(total) <= 1e-30:
                py, px = float(cy), float(cx)
            else:
                py = float((weight * yy).sum() / total)
                px = float((weight * xx).sum() / total)
        else:
            py, px = (float(v) for v in supplied[k])
        shift_y = int(round(cy - py))
        shift_x = int(round(cx - px))
        centered[k] = _translate_zero_fill_2d(
            grouped[k], shift_y=shift_y, shift_x=shift_x,
        )
    return centered.reshape(original_shape)


def _center_pad_2d(image: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Zero-pad the last two dimensions while preserving the array centre."""
    H, W = image.shape[-2:]
    target_h, target_w = (int(v) for v in target_hw)
    if target_h < H or target_w < W:
        raise ValueError("target padding shape must not be smaller than the input")
    top = target_h // 2 - H // 2
    bottom = target_h - H - top
    left = target_w // 2 - W // 2
    right = target_w - W - left
    return F.pad(image, (left, right, top, bottom))


def _center_crop_2d(image: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    """Crop the last two dimensions around their shared FFT centre."""
    H, W = image.shape[-2:]
    out_h, out_w = (int(v) for v in output_hw)
    if out_h > H or out_w > W:
        raise ValueError("output crop shape must not exceed the input")
    top = H // 2 - out_h // 2
    left = W // 2 - out_w // 2
    return image[..., top:top + out_h, left:left + out_w]


def _wiener_kernel(psf: torch.Tensor, reg: float) -> torch.Tensor:
    """Wiener inverse filter in Fourier domain.

    Parameters
    ----------
    psf : Tensor [..., H, W]
        Real, non-negative, normalized PSF (centered: peak at (H//2, W//2)).
    reg : float
        Wiener regularization parameter (NSR estimate).

    Returns
    -------
    Tensor of same shape as psf, complex.
        H_inv = conj(H) / (|H|² + reg)
    """
    psf_shift = torch.fft.ifftshift(psf, dim=(-2, -1))
    H = torch.fft.fft2(psf_shift, dim=(-2, -1))
    H_inv = torch.conj(H) / (H.abs().square() + reg)
    return H_inv


def _wiener_apply(image: torch.Tensor, H_inv: torch.Tensor) -> torch.Tensor:
    """Apply precomputed Wiener kernel to an image.

    Parameters
    ----------
    image : Tensor [..., H, W]
    H_inv : Tensor [..., H, W] complex (Wiener kernel from ``_wiener_kernel``)

    Returns
    -------
    Real-valued deconvolved image of same shape.
    """
    output_hw = tuple(int(v) for v in image.shape[-2:])
    fft_hw = tuple(int(v) for v in H_inv.shape[-2:])
    if output_hw != fft_hw:
        image = _center_pad_2d(image, fft_hw)
    G = torch.fft.fft2(image, dim=(-2, -1))
    F_out = G * H_inv
    result = torch.fft.ifft2(F_out, dim=(-2, -1)).real
    if output_hw != fft_hw:
        result = _center_crop_2d(result, output_hw)
    return result


@torch.no_grad()
def spatially_varying_wiener_recon(
    rgb_image: torch.Tensor,
    psf_cache: PSFCache,
    *,
    f_um: float | None = None,
    z_obj_um: float | None = None,
    reg: float = 1e-3,
    clamp: tuple[float, float] | None = (0.0, None),
    fft_padding_factor: float = 2.0,
) -> torch.Tensor:
    """Bilinear-blended multi-PSF Wiener over the field grid.

    Algorithm
    ---------
    Let cached PSFs be ``psf_cache.psf_rgb_pixel[i, j]`` at object-plane
    field positions (x_j, y_i). Image-side projection of these field
    points: x_img_j = m · x_j, y_img_i = m · y_i, where
    m = -F / (z_obj - F) is the paraxial magnification (image-side, sign-
    inverted because the engine emits an inverted image which the upstream
    pipeline un-flips). Convert to pixel-grid coordinates.

    For every (i, j) with i ∈ [0, Ky-1], j ∈ [0, Kx-1], compute a Wiener
    kernel from psf_rgb_pixel[i, j] and apply it to the FULL image to get
    R_{ij}(x, y). Then blend:

        recon(x, y) = Σ_{i, j} w_ij(x, y) · R_{ij}(x, y)

    where w_ij is a separable bilinear weight that's nonzero only at the
    cell whose 4 corners are (i, j), (i+1, j), (i, j+1), (i+1, j+1) and
    sums to 1 inside it. So each output pixel uses exactly 4 cached PSFs.
    Total cost: Ky·Kx Wiener apples (each Hp²·log Hp), plus O(Hp·Wp)
    blending — sub-second for K=8, Hp=208.

    Parameters
    ----------
    rgb_image : Tensor [3, Hp, Wp]
        Bilinear-demosaic'd or per-channel image to deconvolve. Channels
        match psf_cache.psf_rgb_pixel channels (R, G, B).
    psf_cache : PSFCache
    f_um : float, optional
        Metalens focal length used only by the archived paraxial fallback.
        Production caches carry physical sensor-side axes and do not use it.
    z_obj_um : float, optional
        Object distance (negative). If None, use psf_cache.object_z_um.
    reg : float
        Wiener regularization; tune per-image SNR. Smaller → sharper but
        noisier. 1e-3 is a reasonable visual default.
    clamp : (lo, hi), optional
        Post-recon clamp; default (0, None) clips negatives.
    fft_padding_factor : float
        Linear-boundary FFT extent relative to the image.  The default 2.0 is
        sufficient for a full-frame PSF and prevents opposite-edge circular
        ghosts.  Set to 1.0 only to reproduce the legacy circular convention.

    Returns
    -------
    Tensor [3, Hp, Wp] reconstructed RGB image.
    """
    device = rgb_image.device
    psf_pixel = psf_cache.psf_rgb_pixel.to(device=device, dtype=rgb_image.dtype)
    fy = psf_cache.field_y_um.to(device=device, dtype=rgb_image.dtype)  # [Ky]
    fx = psf_cache.field_x_um.to(device=device, dtype=rgb_image.dtype)  # [Kx]

    Ky, Kx = fy.numel(), fx.numel()
    C, Hp, Wp = rgb_image.shape

    # Field-grid → image-side pixel coords.
    pix_pitch = float(psf_cache.pixel_pitch_um)
    # Convert field pos in object µm → image-side pixel offsets.
    cx, cy = (Wp - 1) / 2.0, (Hp - 1) / 2.0
    has_sensor_axes = (
        psf_cache.sensor_x_um is not None
        or psf_cache.sensor_y_um is not None
        or psf_cache.chief_ray_xy_um is not None
    )
    if has_sensor_axes:
        if (
            psf_cache.sensor_x_um is None
            or psf_cache.sensor_y_um is None
            or psf_cache.chief_ray_xy_um is None
        ):
            raise ValueError(
                "physical PSF cache requires sensor_x_um, sensor_y_um, and "
                "chief_ray_xy_um together"
            )
        sensor_x = psf_cache.sensor_x_um.to(device=device, dtype=rgb_image.dtype)
        sensor_y = psf_cache.sensor_y_um.to(device=device, dtype=rgb_image.dtype)
        chief_xy_um = psf_cache.chief_ray_xy_um.to(
            device=device, dtype=rgb_image.dtype
        )
        if tuple(sensor_x.shape) != (Kx,) or tuple(sensor_y.shape) != (Ky,):
            raise ValueError("physical sensor axes do not match the PSF grid")
        if tuple(chief_xy_um.shape) != (Ky, Kx, 2):
            raise ValueError("chief_ray_xy_um must have shape [Ky,Kx,2]")
        if (
            not bool(torch.isfinite(sensor_x).all())
            or not bool(torch.isfinite(sensor_y).all())
            or not bool(torch.isfinite(chief_xy_um).all())
        ):
            raise ValueError("physical PSF-cache coordinates must be finite")
        if Kx < 2 or Ky < 2:
            raise ValueError("physical PSF cache needs at least 2x2 fields")
        if not bool((torch.diff(sensor_x) > 0).all()) or not bool(
            (torch.diff(sensor_y) > 0).all()
        ):
            raise ValueError("physical sensor axes must be strictly increasing")
        px_x = cx + sensor_x / pix_pitch
        px_y = cy + sensor_y / pix_pitch
        if (
            float(px_x.min()) < 0.0
            or float(px_x.max()) > Wp - 1.0
            or float(px_y.min()) < 0.0
            or float(px_y.max()) > Hp - 1.0
        ):
            raise ValueError("physical sensor PSF grid lies outside the image frame")
        chief_centers = torch.empty_like(chief_xy_um)
        chief_centers[..., 0] = cy + chief_xy_um[..., 1] / pix_pitch
        chief_centers[..., 1] = cx + chief_xy_um[..., 0] / pix_pitch
        if (
            float(chief_centers[..., 0].min()) < 0.0
            or float(chief_centers[..., 0].max()) > Hp - 1.0
            or float(chief_centers[..., 1].min()) < 0.0
            or float(chief_centers[..., 1].max()) > Wp - 1.0
        ):
            raise ValueError("replayed chief-ray centre lies outside the image frame")
    else:
        if f_um is None:
            raise ValueError("archived paraxial PSF cache requires f_um")
        if z_obj_um is None:
            z_obj_um = psf_cache.object_z_um
        z_abs = abs(float(z_obj_um))
        m = float(f_um) / max(z_abs - float(f_um), 1e-9)
        img_x_pix = m * fx / pix_pitch
        img_y_pix = m * fy / pix_pitch
        px_x = (cx + img_x_pix).clamp(0.0, Wp - 1.0)
        px_y = (cy + img_y_pix).clamp(0.0, Hp - 1.0)
        chief_y, chief_x = torch.meshgrid(px_y, px_x, indexing="ij")
        chief_centers = torch.stack((chief_y, chief_x), dim=-1)

    # Register every channel with the same geometric chief-ray shift.
    # Independent channel centering would erase lateral chromatic aberration;
    # an unconstrained full-frame centroid is also biased when an off-axis PSF
    # is clipped by the finite sensor window.
    psf_centered = _center_psf_to_origin(
        psf_pixel, common_center_yx=chief_centers,
    )
    if fft_padding_factor < 1.0:
        raise ValueError("fft_padding_factor must be >= 1")
    fft_h = max(Hp, int(math.ceil(fft_padding_factor * Hp)))
    fft_w = max(Wp, int(math.ceil(fft_padding_factor * Wp)))
    psf_fft = _center_pad_2d(psf_centered, (fft_h, fft_w))
    H_inv = _wiener_kernel(psf_fft, reg=reg)

    # Apply each Wiener to the full image. R[i, j, c] = recon as if PSF[i, j] uniform.
    # FFT the zero-padded RGB image once. Broadcasting against all field
    # kernels avoids Ky*Kx duplicate image FFTs; _wiener_apply crops back to
    # the observed frame after the linear-boundary inverse.
    R = _wiener_apply(rgb_image, H_inv)

    # Per-pixel bilinear weights into the (i, j) grid.
    # For each output pixel (px, py), find which cell (i*, j*) it falls into:
    #   j* = max{j : px_x[j] <= px}, j*+1 = j*+1
    #   i* = max{i : px_y[i] <= py}, i*+1 = i*+1
    # weights:
    #   wx = (px - px_x[j*]) / (px_x[j*+1] - px_x[j*])
    #   wy = same for y
    # output(px, py) = (1-wy)*((1-wx)*R[i*,j*] + wx*R[i*,j*+1])
    #                + wy   *((1-wx)*R[i*+1,j*] + wx*R[i*+1,j*+1])
    yy = torch.arange(Hp, device=device, dtype=rgb_image.dtype)
    xx = torch.arange(Wp, device=device, dtype=rgb_image.dtype)

    # bucketize gives the index k such that px_x[k-1] < x <= px_x[k]; we want
    # j* = clamp(k-1, 0, Kx-2). Same for y.
    j_lo = torch.bucketize(xx, px_x).clamp(1, Kx - 1) - 1           # [Wp]
    i_lo = torch.bucketize(yy, px_y).clamp(1, Ky - 1) - 1           # [Hp]
    j_hi = (j_lo + 1).clamp(max=Kx - 1)
    i_hi = (i_lo + 1).clamp(max=Ky - 1)

    wx = ((xx - px_x[j_lo]) /
          (px_x[j_hi] - px_x[j_lo]).clamp_min(1e-9)).clamp(0.0, 1.0)  # [Wp]
    wy = ((yy - px_y[i_lo]) /
          (px_y[i_hi] - px_y[i_lo]).clamp_min(1e-9)).clamp(0.0, 1.0)  # [Hp]

    # Gather 4 R panels per pixel via fancy indexing, then blend.
    # R has shape [Ky, Kx, C, Hp, Wp]. We pick element (i_lo[y], j_lo[x], c, y, x).
    # Build full 2D index grids:
    yy_2d = torch.arange(Hp, device=device).view(Hp, 1).expand(Hp, Wp)
    xx_2d = torch.arange(Wp, device=device).view(1, Wp).expand(Hp, Wp)
    i_lo_2d = i_lo.view(Hp, 1).expand(Hp, Wp)
    i_hi_2d = i_hi.view(Hp, 1).expand(Hp, Wp)
    j_lo_2d = j_lo.view(1, Wp).expand(Hp, Wp)
    j_hi_2d = j_hi.view(1, Wp).expand(Hp, Wp)

    def gather4(R_, ii, jj):
        # R_: [Ky, Kx, C, Hp, Wp] → pick [ii[y,x], jj[y,x], :, y, x] → [C, Hp, Wp]
        out = R_[ii, jj, :, yy_2d, xx_2d]  # [Hp, Wp, C]
        return out.permute(2, 0, 1)         # [C, Hp, Wp]

    R00 = gather4(R, i_lo_2d, j_lo_2d)
    R01 = gather4(R, i_lo_2d, j_hi_2d)
    R10 = gather4(R, i_hi_2d, j_lo_2d)
    R11 = gather4(R, i_hi_2d, j_hi_2d)

    wx_2d = wx.view(1, 1, Wp)
    wy_2d = wy.view(1, Hp, 1)
    out = ((1 - wy_2d) * ((1 - wx_2d) * R00 + wx_2d * R01)
           + wy_2d * ((1 - wx_2d) * R10 + wx_2d * R11))

    if clamp is not None:
        lo, hi = clamp
        if lo is not None:
            out = out.clamp_min(lo)
        if hi is not None:
            out = out.clamp_max(hi)
    return out
