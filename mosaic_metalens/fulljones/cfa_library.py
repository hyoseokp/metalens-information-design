"""Real-camera color filter array (CFA) spectral sensitivity LUT.

Data source
-----------
Jiang, Liu, Gu, Süsstrunk (2013), "What is the Space of Spectral Sensitivity
Functions for Digital Color Cameras?" — Camera Spectral Sensitivity Database.
- Wavelengths: 400 nm to 720 nm in 10 nm increments (33 samples)
- Per-channel normalization: each (R/G/B) is normalized to its peak = 1
- Measured with PR655 spectrometer; covers ~28 cameras

Reference
- Database: https://www.gujinwei.org/research/camspec/db.html
- Mirror:   http://www.gujinwei.org/research/camspec/camspec_database.txt
- Zenodo:   https://zenodo.org/records/3245883

Below we hardcode 6 representative cameras (3 DSLRs from Canon/Nikon, one
mirrorless, one medium-format, one smartphone) plus the engine's prior
"gaussian_default" model for backward compatibility.

Public API
----------
- :data:`AVAILABLE_CAMERAS`: list of preset names.
- :func:`describe_camera(name)`: dict with camera vendor / sensor type / etc.
- :func:`build_cfa_transmission(name, wavelengths_um, *, normalize=True, device, dtype)`
   → Tensor [3, N_λ, 1, 1] suitable for ``MetalensImagingEngine(cfa_transmission=...)``.
   Linear interpolation from the 400-720 nm grid. If a requested wavelength
   falls outside [400, 720] nm, the value is clamped to the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# Jiang grid: 400-720 nm in 10 nm steps (33 samples)
_LAMBDA_NM = list(range(400, 721, 10))
assert len(_LAMBDA_NM) == 33


# ─────────────────────────────────────────────────────────────────────────────
# Camera spectral sensitivity LUT (R, G, B per row, 33 values each)
# Values are dimensionless and normalized to peak = 1 per channel.
# ─────────────────────────────────────────────────────────────────────────────

_CFA_DB: dict[str, dict[str, list[float]]] = {
    "canon_5d_mark_ii": {
        "R": [0.0019, 0.0045, 0.0103, 0.0055, 0.0034, 0.0021, 0.0023, 0.0039, 0.0073,
              0.0118, 0.0179, 0.0612, 0.0874, 0.1534, 0.1686, 0.1724, 0.2003, 0.3158,
              0.4514, 0.5258, 0.5989, 0.4728, 0.4084, 0.3562, 0.292, 0.226, 0.1704,
              0.1372, 0.0428, 0.0087, 0.0017, 0.0007, 0.0005],
        "G": [0.0036, 0.0123, 0.0377, 0.0422, 0.0565, 0.0704, 0.097, 0.209, 0.43,
              0.6381, 0.692, 1.0, 0.8735, 0.9058, 0.8326, 0.8057, 0.712, 0.6467,
              0.5426, 0.3935, 0.2958, 0.1287, 0.06, 0.0402, 0.0276, 0.0182, 0.0138,
              0.0143, 0.0061, 0.0017, 0.0008, 0.0006, 0.0005],
        "B": [0.0127, 0.0971, 0.3516, 0.4765, 0.56, 0.6476, 0.7745, 0.6759, 0.6858,
              0.5932, 0.3971, 0.3559, 0.1617, 0.0883, 0.0551, 0.0424, 0.0269, 0.0205,
              0.0159, 0.0122, 0.01, 0.0054, 0.0036, 0.0032, 0.0029, 0.0032, 0.0032,
              0.0034, 0.0013, 0.0005, 0.0004, 0.0004, 0.0004],
    },
    "nikon_d700": {
        "R": [0.0017775, 0.0027445, 0.048766, 0.05565, 0.039817, 0.031348, 0.026632,
              0.02811, 0.032437, 0.029863, 0.031673, 0.038347, 0.06215, 0.075373,
              0.039432, 0.021926, 0.02526, 0.095391, 0.49461, 0.74372, 0.71172,
              0.61752, 0.51236, 0.41128, 0.3198, 0.24236, 0.16547, 0.082116, 0.02628,
              0.0073938, 0.0028568, 0.0011169, 0.00047128],
        "G": [0.0014335, 0.001874, 0.02308, 0.043356, 0.059914, 0.082364, 0.13356,
              0.24498, 0.35961, 0.40937, 0.57699, 0.76619, 0.90061, 1.0, 0.95495,
              0.88728, 0.74714, 0.6249, 0.45115, 0.2869, 0.14489, 0.065037, 0.031947,
              0.019449, 0.012242, 0.0081427, 0.0059358, 0.0042266, 0.0020565,
              0.0008812, 0.00048014, 0.0002376, 0.00011756],
        "B": [0.0059157, 0.014395, 0.37455, 0.65271, 0.75309, 0.90218, 0.91167,
              0.86787, 0.81577, 0.64403, 0.46041, 0.27786, 0.13072, 0.064085,
              0.031748, 0.015075, 0.0074314, 0.0053662, 0.0041164, 0.0026565,
              0.001269, 0.00075685, 0.00050643, 0.0003322, 0.0003053, 0.00039176,
              0.00023539, 0.00018848, 0.00008047, 0.000045734, 0.000034504,
              0.000027962, 0.000024659],
    },
    "nikon_d90": {
        "R": [0.0028468, 0.010062, 0.061123, 0.064116, 0.047407, 0.038726, 0.034205,
              0.036572, 0.047892, 0.042438, 0.043595, 0.052616, 0.077728, 0.10753,
              0.06105, 0.032554, 0.031711, 0.082284, 0.38034, 0.69515, 0.64866,
              0.58339, 0.43881, 0.38277, 0.27678, 0.2214, 0.14436, 0.065292, 0.012061,
              0.0023253, 0.0006623, 0.00033038, 0.00023816],
        "G": [0.0021672, 0.0067144, 0.05263, 0.080844, 0.095359, 0.11855, 0.17451,
              0.27748, 0.3775, 0.38856, 0.53093, 0.77103, 0.87177, 1.0, 0.92027,
              0.862, 0.69384, 0.57252, 0.38985, 0.26256, 0.1333, 0.068442, 0.032339,
              0.02286, 0.014809, 0.011587, 0.0083058, 0.0048175, 0.0013214,
              0.00037885, 0.00016044, 0.000093286, 0.000065383],
        "B": [0.0091573, 0.051434, 0.44646, 0.66315, 0.71144, 0.82987, 0.90129,
              0.87253, 0.83253, 0.67033, 0.51074, 0.37096, 0.20735, 0.13943, 0.085438,
              0.050472, 0.022981, 0.012293, 0.0076535, 0.0049382, 0.0027535,
              0.0020545, 0.0016008, 0.002078, 0.0028889, 0.0040484, 0.0039968,
              0.0022416, 0.00049355, 0.00010513, 0.000033758, 0.000013442,
              0.0000074428],
    },
    "sony_nex_5n": {
        "R": [0.0073643, 0.055237, 0.051835, 0.043321, 0.033681, 0.028963, 0.025,
              0.031318, 0.034458, 0.033059, 0.036819, 0.04791, 0.075209, 0.083833,
              0.040534, 0.025882, 0.029435, 0.096315, 0.40981, 0.59492, 0.48306,
              0.4419, 0.32839, 0.26576, 0.18549, 0.14261, 0.10679, 0.073901, 0.045969,
              0.011987, 0.0019017, 0.00074491, 0.00024202],
        "G": [0.0087465, 0.075143, 0.10312, 0.12927, 0.14942, 0.19179, 0.24858,
              0.37672, 0.45492, 0.50906, 0.67036, 0.85874, 0.93855, 1.0, 0.87676,
              0.85895, 0.66097, 0.54573, 0.3987, 0.29141, 0.14386, 0.086686, 0.046569,
              0.032661, 0.021849, 0.016681, 0.013952, 0.012859, 0.011204, 0.0038035,
              0.00078707, 0.00032949, 0.00013348],
        "B": [0.033016, 0.35254, 0.49797, 0.60149, 0.65736, 0.78336, 0.73926, 0.776,
              0.72276, 0.61365, 0.4557, 0.30535, 0.17946, 0.12249, 0.073361, 0.04487,
              0.020265, 0.012304, 0.0085751, 0.0064746, 0.0032885, 0.002511,
              0.0020362, 0.0023356, 0.0027482, 0.0036074, 0.0036988, 0.0031634,
              0.0022407, 0.00069044, 0.00012112, 0.000063737, 0.000042452],
    },
    "phase_one": {
        "R": [0.010464, 0.015837, 0.022809, 0.029193, 0.032943, 0.042109, 0.048271,
              0.054768, 0.055121, 0.048086, 0.037059, 0.037067, 0.052124, 0.064936,
              0.076516, 0.13492, 0.29332, 0.63001, 0.7694, 0.81813, 0.65358, 0.52142,
              0.37956, 0.2838, 0.18967, 0.1305, 0.078487, 0.042293, 0.023358,
              0.012703, 0.0063621, 0.0033125, 0.0017011],
        "G": [0.0064171, 0.013114, 0.025186, 0.033325, 0.045721, 0.073911, 0.11369,
              0.17692, 0.26413, 0.41186, 0.5986, 0.75879, 0.91359, 1.0, 0.90399,
              0.77751, 0.55129, 0.41299, 0.30832, 0.20757, 0.10815, 0.059445,
              0.030895, 0.018277, 0.01124, 0.0068513, 0.0034711, 0.00181, 0.001035,
              0.00060223, 0.00034704, 0.00023128, 0.00015694],
        "B": [0.086173, 0.15791, 0.26574, 0.37978, 0.42988, 0.50637, 0.52789, 0.55929,
              0.53494, 0.48996, 0.40612, 0.30199, 0.25274, 0.20297, 0.14868, 0.11294,
              0.077274, 0.057152, 0.044852, 0.038781, 0.027546, 0.021149, 0.015703,
              0.012548, 0.0094433, 0.007636, 0.0054527, 0.0033028, 0.0020177,
              0.0012756, 0.00078306, 0.00052376, 0.00031011],
    },
    "nokia_n900": {
        "R": [0.0023404, 0.001859, 0.0021612, 0.0018785, 0.0020483, 0.0020453,
              0.0030901, 0.0079493, 0.015542, 0.024933, 0.045539, 0.074702, 0.11864,
              0.16129, 0.16291, 0.16386, 0.15411, 0.14902, 0.42572, 0.95723, 1.0,
              0.70057, 0.90976, 0.93954, 0.77878, 0.4029, 0.12257, 0.061958, 0.023989,
              0.0099138, 0.0076878, 0.0040222, 0.0012967],
        "G": [0.014258, 0.01572, 0.021887, 0.023455, 0.033772, 0.041642, 0.070455,
              0.16539, 0.28849, 0.36037, 0.51551, 0.6732, 0.86191, 0.92522, 0.87936,
              0.85065, 0.76661, 0.67318, 0.54526, 0.44673, 0.29153, 0.30535, 0.1601,
              0.13353, 0.11201, 0.059331, 0.019223, 0.010816, 0.0053431, 0.0028583,
              0.0026006, 0.0014277, 0.0005178],
        "B": [0.25552, 0.31293, 0.45069, 0.49407, 0.53804, 0.56515, 0.60609, 0.58896,
              0.51464, 0.45436, 0.35509, 0.26297, 0.21911, 0.21638, 0.19842, 0.17893,
              0.15676, 0.14759, 0.13721, 0.13343, 0.10382, 0.097638, 0.075487,
              0.077247, 0.072765, 0.042964, 0.01443, 0.0082901, 0.0037298, 0.0015722,
              0.0014485, 0.00087214, 0.0003729],
    },
    # Samsung Galaxy S20 main camera (Sony IMX555). Source: Tominaga et al. 2021,
    # Sensors 21(15):4985 (PMC8347217). Measured 400-700 nm @ 10 nm; values at
    # 710/720 nm extrapolated by 50% dropoff per step (negligible IR leakage).
    # Representative of 2020-present Samsung mobile CFA family (peaks R600/G540/B470).
    "samsung_galaxy_s20": {
        "R": [0.0189, 0.0234, 0.0278, 0.0253, 0.0222, 0.0177, 0.0163, 0.0188, 0.024,
              0.0264, 0.0307, 0.0397, 0.0602, 0.0738, 0.0609, 0.0487, 0.0493, 0.0661,
              0.335, 0.78, 0.83, 0.768, 0.687, 0.606, 0.488, 0.378, 0.287, 0.201,
              0.114, 0.0396, 0.0133, 0.00665, 0.003325],
        "G": [0.0265, 0.0254, 0.029, 0.0291, 0.0373, 0.0415, 0.0514, 0.158, 0.454,
              0.681, 0.847, 0.938, 0.983, 0.993, 1.0, 0.967, 0.903, 0.814, 0.732,
              0.587, 0.433, 0.282, 0.188, 0.135, 0.0972, 0.072, 0.0581, 0.0504, 0.0388,
              0.0203, 0.0106, 0.0053, 0.00265],
        "B": [0.0393, 0.165, 0.314, 0.435, 0.511, 0.583, 0.62, 0.652, 0.65, 0.607,
              0.52, 0.406, 0.285, 0.204, 0.165, 0.12, 0.0859, 0.065, 0.0587, 0.0517,
              0.0439, 0.0361, 0.0319, 0.0314, 0.0321, 0.0322, 0.0316, 0.0282, 0.0198,
              0.00896, 0.00434, 0.00217, 0.001085],
    },
}


@dataclass(frozen=True)
class CameraInfo:
    name: str
    vendor: str
    sensor_class: str  # DSLR / mirrorless / medium-format / smartphone
    description: str


_INFO: dict[str, CameraInfo] = {
    "canon_5d_mark_ii": CameraInfo(
        "canon_5d_mark_ii", "Canon", "DSLR full-frame",
        "Canon EOS 5D Mark II (CMOS, ~21 MP, 2008-era full-frame DSLR).",
    ),
    "nikon_d700":       CameraInfo(
        "nikon_d700", "Nikon", "DSLR full-frame",
        "Nikon D700 (Sony IMX021 CMOS, ~12 MP, 2008 full-frame DSLR).",
    ),
    "nikon_d90":        CameraInfo(
        "nikon_d90", "Nikon", "DSLR APS-C",
        "Nikon D90 (Sony CMOS, ~12 MP, 2008 APS-C DSLR).",
    ),
    "sony_nex_5n":      CameraInfo(
        "sony_nex_5n", "Sony", "Mirrorless APS-C",
        "Sony NEX-5N (Sony CMOS APS-C, ~16 MP, 2011 mirrorless).",
    ),
    "phase_one":        CameraInfo(
        "phase_one", "Phase One", "Medium-format",
        "Phase One medium-format back (Dalsa CCD).",
    ),
    "nokia_n900":       CameraInfo(
        "nokia_n900", "Nokia", "Smartphone (early)",
        "Nokia N900 (Toshiba CMOS, ~5 MP, 2009 smartphone reference).",
    ),
    "samsung_galaxy_s20": CameraInfo(
        "samsung_galaxy_s20", "Samsung", "Smartphone (modern flagship)",
        "Samsung Galaxy S20 main camera (Sony IMX555, 12MP, 2020). "
        "Measured by Tominaga+2021; representative of 2020-present Samsung "
        "mobile CFA family (peaks R 600 / G 540 / B 470 nm).",
    ),
    "boxcar": CameraInfo(
        "boxcar", "synthetic", "ablation (zero-crosstalk)",
        "Perfect step-function bandpass with band edges matching the scene "
        "box-binning (B<=490, 490<G<=590, R>590 nm); photon-equalized to S20.",
    ),
    "gaussian": CameraInfo(
        "gaussian", "synthetic", "ablation (smooth)",
        "Single Gaussian per channel least-squares-fit to the S20 curve; "
        "photon-equalized to S20.",
    ),
}

AVAILABLE_CAMERAS: list[str] = list(_CFA_DB.keys()) + ["boxcar", "gaussian"]


def describe_camera(name: str) -> CameraInfo:
    if name not in _INFO:
        raise KeyError(f"Unknown camera '{name}'. Available: {AVAILABLE_CAMERAS}")
    return _INFO[name]


def _interp_to_wavelengths(
    values_at_jiang_grid: list[float],
    wavelengths_um: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation from the Jiang 400-720 nm @ 10 nm grid to the
    target wavelengths_um. Out-of-range values are clamped to boundary."""
    src_x = torch.tensor(_LAMBDA_NM, dtype=torch.float32, device=wavelengths_um.device)
    src_y = torch.tensor(values_at_jiang_grid, dtype=torch.float32, device=wavelengths_um.device)
    target_nm = (wavelengths_um.to(torch.float32) * 1000.0).clamp(src_x[0], src_x[-1])
    # bucketize then linearly interpolate
    hi = torch.bucketize(target_nm, src_x).clamp(1, src_x.numel() - 1)
    lo = hi - 1
    x0 = src_x[lo]
    x1 = src_x[hi]
    t = (target_nm - x0) / (x1 - x0).clamp_min(1e-12)
    return src_y[lo] * (1.0 - t) + src_y[hi] * t


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic CFA models for the passband-shape ablation (boxcar / gaussian).
# Both are PHOTON-EQUALIZED to the real Samsung S20 CFA: each channel is scaled
# by one scalar so estimate_photon_per_channel returns the same per-channel
# counts as S20 (which depends only on the per-channel sum over the evaluation
# grid, since the flux constant is camera-independent). This isolates passband
# SHAPE from photon budget. The scene RGB->spectral box binning is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHETIC_CAMERAS = ("boxcar", "gaussian")
_REF_CAMERA = "samsung_galaxy_s20"       # photon-equalization + gaussian-fit target
# scene box-binning edges (stage5_e2e_pipeline_6wl.py): B lambda<=490, 490<G<=590, R>590
_BOX_EDGE_BG_NM = 490.0
_BOX_EDGE_GR_NM = 590.0
_GAUSSIAN_PARAMS_CACHE: dict[str, tuple[float, float]] | None = None


def _fit_gaussian_params() -> dict[str, tuple[float, float]]:
    """Least-squares single-Gaussian (center, sigma) per S20 channel.

    Fit A*exp(-0.5((lambda-mu)/sigma)^2) on the measured Jiang grid (400-700 nm;
    the 710/720 nm points are extrapolated, so excluded). Amplitude A is
    irrelevant here (absorbed by photon equalization); only mu/sigma set the
    passband shape. Falls back to intensity moments if the fit fails."""
    global _GAUSSIAN_PARAMS_CACHE
    if _GAUSSIAN_PARAMS_CACHE is not None:
        return _GAUSSIAN_PARAMS_CACHE
    import numpy as np
    lam = np.array(_LAMBDA_NM, dtype=float)
    m = lam <= 700.0
    x = lam[m]
    out: dict[str, tuple[float, float]] = {}
    for ch in ("R", "G", "B"):
        y = np.array(_CFA_DB[_REF_CAMERA][ch], dtype=float)[m]
        w = y / y.sum()
        mu0 = float((w * x).sum())
        sig0 = float(np.sqrt((w * (x - mu0) ** 2).sum()))
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(
                lambda xx, a, mu, sig: a * np.exp(-0.5 * ((xx - mu) / sig) ** 2),
                x, y, p0=[float(y.max()), mu0, sig0], maxfev=20000)
            mu, sig = float(popt[1]), abs(float(popt[2]))
        except Exception:
            mu, sig = mu0, sig0
        out[ch] = (mu, sig)
    _GAUSSIAN_PARAMS_CACHE = out
    return out


def _synthetic_cfa_raw(camera: str, wls_nm: torch.Tensor) -> torch.Tensor:
    """Unequalized [3, N] synthetic CFA (R, G, B) at wavelengths wls_nm (nm)."""
    raw = torch.zeros(3, wls_nm.numel(), dtype=torch.float32, device=wls_nm.device)
    if camera == "boxcar":
        raw[0] = (wls_nm > _BOX_EDGE_GR_NM).to(torch.float32)                       # R
        raw[1] = ((wls_nm > _BOX_EDGE_BG_NM) & (wls_nm <= _BOX_EDGE_GR_NM)).to(torch.float32)  # G
        raw[2] = (wls_nm <= _BOX_EDGE_BG_NM).to(torch.float32)                      # B
    elif camera == "gaussian":
        params = _fit_gaussian_params()
        for i, ch in enumerate(("R", "G", "B")):
            mu, sig = params[ch]
            raw[i] = torch.exp(-0.5 * ((wls_nm - mu) / sig) ** 2)
    else:
        raise ValueError(f"not a synthetic camera: {camera}")
    return raw


def _equalization_scale(raw: torch.Tensor, wavelengths_um: torch.Tensor) -> torch.Tensor:
    """Per-channel scalar making raw's per-channel sum equal the real S20's, so
    estimate_photon_per_channel matches S20 on this grid."""
    ref = torch.stack([_interp_to_wavelengths(_CFA_DB[_REF_CAMERA][ch], wavelengths_um)
                       for ch in ("R", "G", "B")])              # [3, N]
    return ref.sum(dim=1) / raw.sum(dim=1).clamp_min(1e-12)     # [3]


def _build_synthetic_cfa(camera: str, wavelengths_um: torch.Tensor) -> torch.Tensor:
    """Photon-equalized [3, N] synthetic CFA transmission."""
    wls_nm = wavelengths_um.to(torch.float32) * 1000.0
    raw = _synthetic_cfa_raw(camera, wls_nm)
    scale = _equalization_scale(raw, wavelengths_um)
    return raw * scale.view(3, 1)


def describe_synthetic_cfa(camera: str, wavelengths_um: torch.Tensor) -> dict:
    """Report per-channel shape params, equalization scalars, and per-channel
    transmission sums (raw and equalized) for the synthetic CFA on this grid."""
    wls_nm = wavelengths_um.to(torch.float32) * 1000.0
    raw = _synthetic_cfa_raw(camera, wls_nm)
    scale = _equalization_scale(raw, wavelengths_um)
    eq = raw * scale.view(3, 1)
    info: dict = {"camera": camera,
                  "equalization_scale": {c: float(scale[i]) for i, c in enumerate("RGB")},
                  "raw_sum": {c: float(raw[i].sum()) for i, c in enumerate("RGB")},
                  "eq_sum": {c: float(eq[i].sum()) for i, c in enumerate("RGB")},
                  "eq_inband_max": {c: float(eq[i].max()) for i, c in enumerate("RGB")}}
    if camera == "boxcar":
        info["band_edges_nm"] = {"B": ("<=", _BOX_EDGE_BG_NM),
                                 "G": (_BOX_EDGE_BG_NM, _BOX_EDGE_GR_NM),
                                 "R": (">", _BOX_EDGE_GR_NM)}
    elif camera == "gaussian":
        p = _fit_gaussian_params()
        info["fit_center_nm"] = {c: p[c][0] for c in "RGB"}
        info["fit_sigma_nm"] = {c: p[c][1] for c in "RGB"}
    return info


def build_cfa_transmission(
    camera: str,
    wavelengths_um: torch.Tensor,
    *,
    normalize: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a [3, N_λ, 1, 1] CFA transmission tensor for the named camera.

    Parameters
    ----------
    camera : str
        Preset name (see :data:`AVAILABLE_CAMERAS`).
    wavelengths_um : Tensor [N_λ]
        Target wavelengths (μm). Should lie within [0.4, 0.72] μm; values
        outside are clamped to the database boundary.
    normalize : bool
        If True, rescales each channel so its max over the requested
        wavelengths equals 1. Default False (preserves the database's
        per-channel peak normalization).
    device, dtype : standard tensor placement.

    Returns
    -------
    cfa : Tensor [3 channels (R,G,B), N_λ, 1, 1]
        Suitable for ``MetalensImagingEngine(cfa_transmission=cfa)``.

    The synthetic ablation cameras ``"boxcar"`` and ``"gaussian"`` are
    photon-equalized to ``samsung_galaxy_s20`` (``normalize`` is ignored for
    them; the equalization scale is their calibration).
    """
    if device is None:
        device = wavelengths_um.device
    if camera in _SYNTHETIC_CAMERAS:
        cfa = _build_synthetic_cfa(camera, wavelengths_um.to(device))   # [3, N]
        return cfa.to(device=device, dtype=dtype).view(3, -1, 1, 1)
    if camera not in _CFA_DB:
        raise KeyError(f"Unknown camera '{camera}'. Available: {AVAILABLE_CAMERAS}")

    db_entry = _CFA_DB[camera]
    rows = []
    for ch in ("R", "G", "B"):
        rows.append(_interp_to_wavelengths(db_entry[ch], wavelengths_um.to(device)))
    cfa = torch.stack(rows, dim=0).to(device=device, dtype=dtype)  # [3, N_λ]

    if normalize:
        peaks = cfa.amax(dim=1, keepdim=True).clamp_min(1e-12)
        cfa = cfa / peaks

    return cfa.view(3, -1, 1, 1)


def cfa_summary_table(
    camera: str,
    wavelengths_um: torch.Tensor,
) -> str:
    """Return a printable string summarizing the (R/G/B) × wavelength matrix."""
    cfa = build_cfa_transmission(camera, wavelengths_um)
    arr = cfa.squeeze(-1).squeeze(-1).cpu().numpy()
    info = describe_camera(camera)
    lines = [
        f"Camera: {info.name}  ({info.vendor}, {info.sensor_class})",
        f"  {info.description}",
        "  Wavelengths (nm): " + " ".join(f"{w * 1000:6.0f}" for w in wavelengths_um.tolist()),
    ]
    for ch_idx, ch in enumerate("RGB"):
        row = " ".join(f"{arr[ch_idx, i]:6.3f}" for i in range(arr.shape[1]))
        lines.append(f"  {ch}: [{row}]")
    return "\n".join(lines)
