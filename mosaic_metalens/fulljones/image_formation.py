from __future__ import annotations

import torch

from .sensor_stack import photodiode_irradiance


def incoherent_accumulate(
    photodiode_field: torch.Tensor,
    spectral_radiance: torch.Tensor,
) -> torch.Tensor:
    irradiance = photodiode_irradiance(photodiode_field)
    weights = spectral_radiance[..., None, None].to(irradiance.dtype)
    return (irradiance * weights).sum(dim=0)


def vectorial_incoherent_accumulate(
    Ex: torch.Tensor,
    Ey: torch.Tensor,
    Ez: torch.Tensor,
    spectral_radiance: torch.Tensor,
) -> torch.Tensor:
    """Incoherent accumulation of vectorial irradiance |Ex|²+|Ey|²+|Ez|².

    Note: this measures field energy density, not Poynting flux. For
    radiometrically correct sensor signal (with the natural cos⁴
    illumination law emerging from oblique-incidence physics), use
    :func:`vectorial_poynting_accumulate` which computes
    S_z = (1/2) Re(E_x · H_y* − E_y · H_x*).
    """
    irradiance = Ex.abs().square() + Ey.abs().square() + Ez.abs().square()
    weights = spectral_radiance[..., None, None].to(irradiance.dtype)
    return (irradiance * weights).sum(dim=0)


def vectorial_poynting_accumulate(
    Ex: torch.Tensor,
    Ey: torch.Tensor,
    Hx: torch.Tensor,
    Hy: torch.Tensor,
    spectral_radiance: torch.Tensor,
) -> torch.Tensor:
    """Incoherent accumulation of the z-component of the time-averaged
    Poynting vector at the sensor plane:

        S_z(x) = (1/2) Re( E(x) × H*(x) ) · ẑ
               = (1/2) Re( E_x · H_y* − E_y · H_x* )

    With H computed from E in k-space via H = (k × E) / (ω μ₀), each plane-
    wave component contributes to S_z with a kz/k = cos(θ_k) weighting.
    Combined with the 1/r spherical-wave amplitude in the object→pupil step
    and the pupil aperture clipping, the overall illumination falls off as
    the natural cos⁴(θ_CRA) law without any explicit cosine factor inserted
    by hand.

    The 1/2 prefactor is dropped (relative photometry only); the
    wavelength-dependent factor 1/k is already included in H_x, H_y by the
    propagator.
    """
    Sz = (Ex * Hy.conj() - Ey * Hx.conj()).real
    weights = spectral_radiance[..., None, None].to(Sz.dtype)
    return (Sz * weights).sum(dim=0)
