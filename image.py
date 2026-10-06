#!/usr/bin/env python
"""Image a scene through the published metalenses and score the reconstruction.

    python image.py --scene data/scenes/kodim01.png
    python image.py --scene my_photo.png --designs information --out out/imaging

The scene is resized to the 198x198 active object grid, lifted to the nine
design wavelengths under the scene prior, and every object point is propagated
through the full-Jones forward model onto the RGGB sensor (exact full-scene
propagation, 39,204 points per design, no PSF convolution). The raw frame is
demosaicked, deconvolved by a spatially varying Wiener filter built from an
8x8 field bank of exact PSFs, and colour-corrected by the analytic per-design
3x3 matrix. PSNR, CIEDE2000 and S-CIELAB are reported against the represented
scene. This is the manuscript imaging pipeline.

Stages and their caches under --out:

    psf_bank/<design>.npz         8x8 exact PSF bank (scene independent)
    calibration/<design>.npz      sensor spectral DC and colour matrix
    scenes/<scene>/raw_<design>.npz     truth, rgb_e, raw
    scenes/<scene>/recon_<design>.npz   reconstruction (linear RGB)
    scenes/<scene>/*.png                truth, raw and reconstruction previews
    scenes/<scene>/metrics.json

A stage whose cache exists is skipped, so metrics and previews can be redone
without re-simulating. The scene render needs CUDA: one design takes about
20 min (D4-symmetric map, eight-fold reuse) to 35 min (mirror-symmetric map,
four-fold reuse) on a 48 GB card. The PSF bank and the calibration probe are
65 single-point propagations per design and also run on CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mosaic_metalens.fulljones import scoring as S  # noqa: E402
from mosaic_metalens.fulljones import colour_calibration as C  # noqa: E402
from mosaic_metalens.fulljones import imaging as I  # noqa: E402
from mosaic_metalens.fulljones import imaging_metrics as M  # noqa: E402
from mosaic_metalens.fulljones.fov_psf import render_psf_bank  # noqa: E402
from mosaic_metalens.fulljones.scene_render import (  # noqa: E402
    load_scene_linear_rgb,
    render_scene,
)

DESIGNS = {
    "reference": ("hyperbolic reference", HERE / "designs/hyperbolic_reference.pt"),
    "information": ("information design", HERE / "designs/information_design.pt"),
    "mtf": ("MTF-volume control", HERE / "designs/mtf_volume_control.pt"),
}
NODES = 8
WIENER_REG = I.WIENER_REG
CENTRE_ROI = 64


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def load_widths(key: str, device: torch.device) -> torch.Tensor:
    path = DESIGNS[key][1]
    state = torch.load(path, map_location="cpu", weights_only=False)
    widths = state["width_um"] if "width_um" in state else state["width_full"]
    return widths.to(device, torch.float32).clamp(S.WIDTH_MIN_UM, S.WIDTH_MAX_UM)


def save_png(path: Path, srgb8: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(srgb8).save(path)


def raw_preview(raw: np.ndarray) -> np.ndarray:
    """Raw electrons as a grey image, scaled to the frame's 99.9th percentile."""
    scale = float(np.percentile(raw, 99.9)) or 1.0
    grey = I.to_srgb8(np.repeat((raw / scale)[..., None], 3, axis=-1))
    return grey


class Stages:
    def __init__(self, out: Path, device: torch.device, allow_cpu_render: bool):
        self.out = out
        self.device = device
        self.allow_cpu_render = allow_cpu_render
        self._engine = None
        self._protocol = None

    def engine(self):
        if self._engine is None:
            t = time.time()
            self._engine = S.build_engine(self.device, lut_path=S.LUT_PATH)
            self._protocol = S.load_prior(self.device)
            print(f"engine ready on {self.device.type} ({time.time() - t:.0f}s)", flush=True)
        return self._engine, self._protocol

    def psf_bank(self, key: str) -> dict:
        path = self.out / "psf_bank" / f"{key}.npz"
        if path.is_file():
            return dict(np.load(path, allow_pickle=False))
        engine, protocol = self.engine()
        t = time.time()
        bank = render_psf_bank(engine, protocol, load_widths(key, self.device),
                               nodes=NODES, electron_calibration=S.ELECTRON_CALIBRATION,
                               progress=True)
        arrays = {k: v for k, v in bank.items() if isinstance(v, np.ndarray)}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        print(f"[{key}] PSF bank {NODES}x{NODES} ({time.time() - t:.0f}s) -> {path}", flush=True)
        return arrays

    def calibration(self, key: str, bank: dict) -> np.ndarray:
        path = self.out / "calibration" / f"{key}.npz"
        if path.is_file():
            return np.load(path, allow_pickle=False)["colour_matrix"]
        engine, protocol = self.engine()
        t = time.time()
        sensor_dc, _, _ = C.exact_reference_sensor_spectral_dc(
            engine, load_widths(key, self.device),
            electron_calibration=S.ELECTRON_CALIBRATION)
        cache = I.psf_cache_from_bank(bank, NODES)
        matrix = C.colour_matrix(
            sensor_dc, cache,
            protocol["scene_color_covariance"].cpu().numpy(),
            protocol["target_color_transform"].cpu().numpy(),
            wiener_reg=WIENER_REG, centre_roi_size=CENTRE_ROI)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sensor_dc=sensor_dc, colour_matrix=matrix,
                            wiener_reg=WIENER_REG, centre_roi_size=CENTRE_ROI)
        print(f"[{key}] colour matrix ({time.time() - t:.0f}s) -> {path}", flush=True)
        return matrix

    def raw(self, key: str, scene_dir: Path, scene_path: Path) -> dict:
        path = scene_dir / f"raw_{key}.npz"
        if path.is_file():
            return dict(np.load(path, allow_pickle=False))
        if self.device.type != "cuda" and not self.allow_cpu_render:
            raise SystemExit(
                "the full-scene render needs CUDA (39,204 propagations per "
                "design). Pass --allow-cpu-render to run it anyway.")
        engine, protocol = self.engine()
        scene = load_scene_linear_rgb(scene_path, self.device)
        result = render_scene(engine, protocol, load_widths(key, self.device), scene,
                              electron_calibration=S.ELECTRON_CALIBRATION, progress=True)
        arrays = {k: result[k] for k in ("truth", "rgb_e", "raw")}
        np.savez_compressed(path, **arrays, symmetry=result["symmetry"],
                            wall_s=result["wall_s"])
        print(f"[{key}] raw render, {result['symmetry']} "
              f"({result['wall_s'] / 60:.1f} min) -> {path}", flush=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="RGB image (PNG, JPEG)")
    ap.add_argument("--designs", default="all",
                    help="all, or a comma-separated subset of the design keys")
    ap.add_argument("--out", default="out/imaging")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-cpu-render", action="store_true")
    ap.add_argument("--lut", default=None,
                    help="full-Jones LUT npz for another meta-atom library or pillar height "
                         "(default: the published SiN 1000 nm library)")
    ap.add_argument("--design", action="append", default=None, metavar="NAME=PATH",
                    help="width map checkpoint to image instead of the published designs; "
                         "repeatable. The checkpoint needs a 'width_um' or 'width_full' tensor.")
    ap.add_argument("--electron-calibration", type=float, default=None,
                    help="frozen model-electron calibration scalar for --lut platforms "
                         "(default: the published SiN 1000 nm value)")
    args = ap.parse_args()

    device = resolve_device(args.device)
    if args.lut is not None:
        S.LUT_PATH = Path(args.lut).resolve()
    if args.electron_calibration is not None:
        S.ELECTRON_CALIBRATION = float(args.electron_calibration)
    if args.design:
        DESIGNS.clear()
        for spec in args.design:
            name, _, path = spec.partition("=")
            DESIGNS[name.strip()] = (name.strip(), Path(path.strip()).resolve())
    out = (HERE / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    scene_path = Path(args.scene).resolve()
    scene_dir = out / "scenes" / scene_path.stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    keys = list(DESIGNS) if args.designs == "all" else [k.strip() for k in args.designs.split(",")]
    unknown = [k for k in keys if k not in DESIGNS]
    if unknown:
        raise SystemExit(f"unknown design keys {unknown}; available: {list(DESIGNS)}")
    stages = Stages(out, device, args.allow_cpu_render)

    print("=" * 68)
    print("metalens-information-design  |  image.py  (exact full-scene imaging)")
    print("=" * 68)
    print(f"scene {scene_path}\ndevice {device.type}  out {out}")
    print("-" * 68)

    metrics_path = scene_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    print(f"  {'design':<22s}{'PSNR dB':>9s}{'dE00':>8s}{'S-CIELAB':>10s}{'raw e/px':>10s}")
    for key in keys:
        label = DESIGNS[key][0]
        bank = stages.psf_bank(key)
        matrix = stages.calibration(key, bank)
        raw = stages.raw(key, scene_dir, scene_path)

        cache = I.psf_cache_from_bank(bank, NODES)
        mask = I.rggb_mask(*raw["raw"].shape)
        recon = I.apply_colour_matrix(
            matrix, I.reconstruct(I.demosaic(raw["raw"], mask), cache, WIENER_REG))
        truth = raw["truth"].astype(np.float64)
        t = np.clip(truth, 0.0, 1.0)
        r = np.clip(recon, 0.0, 1.0)
        row = {
            "design": label,
            "psnr_db": M.psnr_db(t, r),
            "delta_e00": M.delta_e00_mean(t, r),
            "scielab_delta_e": M.scielab_delta_e_mean(t, r),
            "mean_raw_electrons_per_pixel": float(raw["raw"].mean()),
            "wiener_reg": WIENER_REG,
            "nodes": NODES,
        }
        metrics[key] = row
        np.savez_compressed(scene_dir / f"recon_{key}.npz", recon=recon.astype(np.float32),
                            colour_matrix=matrix)
        save_png(scene_dir / f"recon_{key}.png", I.to_srgb8(recon))
        save_png(scene_dir / f"raw_{key}.png", raw_preview(raw["raw"]))
        if not (scene_dir / "truth.png").is_file():
            save_png(scene_dir / "truth.png", I.to_srgb8(truth))
        print(f"  {label:<22s}{row['psnr_db']:>9.2f}{row['delta_e00']:>8.2f}"
              f"{row['scielab_delta_e']:>10.2f}{row['mean_raw_electrons_per_pixel']:>10.1f}",
              flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("-" * 68)
    print(f"metrics -> {metrics_path}")
    print("truth is the scene as represented under the prior. PSNR, CIEDE2000 and "
          "S-CIELAB are computed on clipped linear RGB (manuscript convention).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
