#!/usr/bin/env python
"""Optimization worker for the web console.

Runs one I_tar width-map optimization described by <job-dir>/config.json and
reports progress by atomically rewriting <job-dir>/progress.json after every
step. The parent server process only ever reads that file, so the worker can
be killed or crash without corrupting the UI state.

The optical geometry (focal length, aperture diameter, pupil grid, CFA) is
taken from the config and installed into the scoring module before the engine
is built. The sensor sampling contract is NOT configurable: the 64-alias
polyphase operator requires the 0.25 um scene grid and the 1.0 um photosite
pitch (see mosaic_metalens/fulljones/production_multirate.py).

    python webapp/run_job.py --job-dir webapp/jobs/<id>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mosaic_metalens.fulljones import scoring as S
from mosaic_metalens.fulljones.cfa_library import AVAILABLE_CAMERAS
from mosaic_metalens.fulljones.field_local import field_local_sensor_shift_xy_um
from mosaic_metalens.fulljones.optim_widthmap import ProjectedMirrorQuadrantWidthParam
from mosaic_metalens.fulljones.projected_width_optimizer import ProjectedAdam
from mosaic_metalens.fulljones.sensor_objective import _field_object_batch

PREVIEW = 96
FOV_MAX_DEG = 20.0


# ----------------------------------------------------------------- progress
class Progress:
    def __init__(self, job_dir: Path, total_steps: int):
        self.path = job_dir / "progress.json"
        self.job_dir = job_dir
        self.state = {
            "status": "running", "step": 0, "total_steps": total_steps,
            "itar_history": [], "width_preview": None, "eta_s": None,
            "message": "",
        }
        self.t0 = time.time()

    def write(self, **updates) -> None:
        self.state.update(updates)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(tmp, self.path)

    def step(self, n: int) -> None:
        per = (time.time() - self.t0) / max(n, 1)
        eta = per * (self.state["total_steps"] - n)
        self.write(step=n, eta_s=round(eta, 1))

    def cancelled(self) -> bool:
        return (self.job_dir / "cancel").exists()


def width_preview(width_map: torch.Tensor) -> dict:
    w = width_map.detach().to("cpu", torch.float32)[None, None]
    small = torch.nn.functional.interpolate(
        w, size=(PREVIEW, PREVIEW), mode="area")[0, 0]
    return {
        "w": PREVIEW, "h": PREVIEW,
        "min": round(float(small.min()), 4),
        "max": round(float(small.max()), 4),
        "data": [round(float(v), 4) for v in small.reshape(-1)],
    }


# ----------------------------------------------------------------- geometry
def make_field_points_fov(engine, fov_deg: float, n: int = 5):
    """Square-field quadrature capped by BOTH sensor coverage and a user FOV.

    Identical to scoring.make_field_points except that the per-azimuth
    object-space boundary is additionally limited to z_object*tan(fov_deg).
    """
    from mosaic_metalens.fulljones.field_local import reference_half_space_indices
    n_in, n_out = (float(x) for x in reference_half_space_indices(engine))
    z_object = abs(float(engine.spec.object_plane.z_um))
    sensor_distance = float(engine.sensor_grid.z_um - engine.pupil_grid.z_um)
    half_x = 0.5 * engine.sensor_grid.width * float(engine.sensor_grid.pitch_um)
    half_y = 0.5 * engine.sensor_grid.height * float(engine.sensor_grid.pitch_um)
    usable_x = half_x - S.SENSOR_FIELD_MARGIN_UM
    usable_y = half_y - S.SENSOR_FIELD_MARGIN_UM
    object_half = 0.5 * S.OBJ_SIZE_UM
    fov_radius = z_object * math.tan(math.radians(max(fov_deg, 1e-6)))
    azimuths = (0.0, 22.5, 45.0, 67.5, 90.0)
    azimuth_w = (1.0 / 12.0, 4.0 / 12.0, 2.0 / 12.0, 4.0 / 12.0, 1.0 / 12.0)
    points, weights = [], []
    for radial_index in range(n):
        fraction = math.sqrt((radial_index + 0.5) / n)
        for azimuth, qw in zip(azimuths, azimuth_w):
            az = math.radians(azimuth)
            cos_az, sin_az = abs(math.cos(az)), abs(math.sin(az))
            sensor_radial_limit = min(usable_x / max(cos_az, 1e-15),
                                      usable_y / max(sin_az, 1e-15))
            theta_out = math.atan(sensor_radial_limit / sensor_distance)
            sin_in = (n_out / n_in) * math.sin(theta_out)
            sensor_obj_r = z_object * math.tan(math.asin(sin_in))
            obj_r = min(object_half / max(cos_az, 1e-15),
                        object_half / max(sin_az, 1e-15))
            boundary = min(sensor_obj_r, obj_r, fov_radius)
            radius = fraction * boundary
            theta_in = math.degrees(math.atan(radius / z_object))
            points.append(S.FieldPoint(
                f"field_r{radial_index}_az{azimuth:g}deg", theta_in, azimuth))
            weights.append(qw * boundary ** 2 / n)
    w = torch.tensor(weights, dtype=torch.float64)
    return points, w / w.sum()


def load_init_width(pupil_grid: int, device, dtype) -> torch.Tensor:
    """Hyperbolic-reference init, bilinearly resampled to the requested grid."""
    ref = torch.load(ROOT / "designs" / "hyperbolic_reference.pt",
                     map_location="cpu", weights_only=True)["width_um"]
    ref = ref.to(torch.float32)
    if ref.shape != (pupil_grid, pupil_grid):
        ref = torch.nn.functional.interpolate(
            ref[None, None], size=(pupil_grid, pupil_grid),
            mode="bilinear", align_corners=False)[0, 0]
    return ref.to(device, dtype).clamp(S.WIDTH_MIN_UM, S.WIDTH_MAX_UM)


# ----------------------------------------------------------------- figures
def render_psf_mtf(engine, widths, job_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        batch = _field_object_batch(engine, S.FieldPoint("onaxis", 0.0, 0.0),
                                    S.POINT_RADIANCE)
        shift = field_local_sensor_shift_xy_um(engine, batch)
        psf = engine.forward_optics_from_object_batch(
            batch, width_map_override=widths,
            sensor_field_shift_xy_um=shift)          # [n_wl, H, W] intensity
    psf = psf.detach().cpu().to(torch.float64).numpy()
    wl_nm = [w * 1000.0 for w in engine.wavelengths_um.tolist()]
    pitch = float(engine.sensor_grid.pitch_um)

    # --- on-axis spectral PSF tiles (crop the central region) ---
    crop = 96
    h, w = psf.shape[-2:]
    cy, cx = h // 2, w // 2
    sl = np.s_[cy - crop:cy + crop, cx - crop:cx + crop]
    show = [(450.0, "#2B5FD9"), (550.0, "#1F9C4B"), (650.0, "#CC2A1E")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (target, color) in zip(axes, show):
        i = int(np.argmin([abs(v - target) for v in wl_nm]))
        tile = psf[i][sl]
        ax.imshow(tile / max(tile.max(), 1e-30), cmap="inferno")
        ax.set_title(f"{wl_nm[i]:.0f} nm", fontsize=9, color=color)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("on-axis PSF (self-normalized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(job_dir / "psf.png", dpi=120)
    plt.close(fig)

    # --- polychromatic channel MTF (CFA-weighted incoherent PSF sum) ---
    cfa = engine.cfa_transmission.detach().cpu().to(torch.float64).numpy()
    cfa = cfa.reshape(cfa.shape[0], cfa.shape[1])       # [3, n_wl]
    f_max, n_bins = 1.3, 60
    bins = np.linspace(0.0, f_max, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    fy = np.fft.fftfreq(h, d=pitch)
    fx = np.fft.fftfreq(w, d=pitch)
    radius = np.hypot(fy[:, None], fx[None, :]).ravel()
    which = np.digitize(radius, bins) - 1
    valid = (which >= 0) & (which < n_bins)
    counts = np.bincount(which[valid], minlength=n_bins)

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for ch, color, label in [(0, "#CC2A1E", "R"), (1, "#1F9C4B", "G"),
                             (2, "#2B5FD9", "B")]:
        poly = np.tensordot(cfa[ch], psf, axes=(0, 0))   # [H, W]
        otf = np.fft.fft2(np.fft.ifftshift(poly))
        dc = abs(otf[0, 0])
        mag = (np.abs(otf) / max(dc, 1e-30)).ravel()
        sums = np.bincount(which[valid], weights=mag[valid], minlength=n_bins)
        prof = np.divide(sums, counts, out=np.zeros_like(sums),
                         where=counts > 0)
        ax.plot(centres, prof, color=color, lw=1.3, label=label)
    ax.set_xlabel("spatial frequency (cycles/um)")
    ax.set_ylabel("channel MTF (on axis)")
    ax.set_xlim(0, f_max); ax.set_ylim(bottom=0)
    ax.legend(title="channel", fontsize=8, title_fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(job_dir / "mtf.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------- main
def run(job_dir: Path) -> int:
    cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
    steps = int(cfg["steps"])
    progress = Progress(job_dir, steps)
    progress.write(status="running", message="validating configuration")

    # ---- validation -------------------------------------------------
    if cfg.get("library", "SiN") != "SiN":
        raise ValueError("only the SiN library is bundled with this release")
    cfa = str(cfg.get("cfa", "samsung_galaxy_s20"))
    if cfa not in AVAILABLE_CAMERAS:
        raise ValueError(f"unknown CFA '{cfa}'; available: {AVAILABLE_CAMERAS}")
    focal_um = float(cfg["focal_um"])
    diameter_um = float(cfg["diameter_um"])
    pupil_grid = int(cfg["pupil_grid"])
    fov_deg = float(cfg.get("fov_deg", 10.0))
    fields_per_step = int(cfg.get("fields_per_step", 4))
    validate_every = int(cfg.get("validate_every", 10))
    quadrature_n = int(cfg.get("quadrature_n", 5))
    seed = int(cfg.get("seed", 0))
    if focal_um <= 0 or diameter_um <= 0:
        raise ValueError("focal length and diameter must be positive")
    if diameter_um > pupil_grid * S.PILLAR_PITCH_UM:
        raise ValueError(
            f"aperture D = {diameter_um} um does not fit the "
            f"{pupil_grid}x{pupil_grid} lattice "
            f"({pupil_grid * S.PILLAR_PITCH_UM:.1f} um across); "
            "raise the pupil grid")
    if pupil_grid % 2:
        raise ValueError("pupil grid must be even")
    if not 0.0 <= fov_deg <= FOV_MAX_DEG:
        raise ValueError(f"FOV half-angle must be within [0, {FOV_MAX_DEG}] deg")
    if not 1 <= quadrature_n <= 5:
        raise ValueError("quadrature_n must be within [1, 5]")

    requested = str(cfg.get("device", "auto")).lower()
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is unavailable")
        device = torch.device("cuda")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.get("allow_cpu"):
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "no CUDA device found; the full-Jones forward is only practical "
            "on a GPU (set \"device\": \"cpu\" or \"allow_cpu\": true for a "
            "smoke test)")
    dtype = torch.float64

    # ---- install the requested geometry into the scoring module -----
    S.D_UM = diameter_um
    S.F_UM = focal_um
    S.PUPIL_GRID = pupil_grid
    S.CAMERA_NAME = cfa

    progress.write(message="building full-Jones engine")
    engine = S.build_engine(device)
    protocol = S.load_prior(device, dtype)
    scorer = S.TargetInformationScorer(engine, protocol, device, dtype)
    field_points, field_weight = make_field_points_fov(engine, fov_deg,
                                                       quadrature_n)
    field_weight = field_weight.to(device)

    init_map = load_init_width(pupil_grid, device, dtype)
    param = ProjectedMirrorQuadrantWidthParam(
        pupil_grid, pupil_grid, init_w_full_um=init_map,
        width_range_um=(S.WIDTH_MIN_UM, S.WIDTH_MAX_UM),
        pupil_pitch_um=S.PILLAR_PITCH_UM,
        aperture_radius_um=diameter_um * 0.5,
    ).to(device)
    optimizer = ProjectedAdam(param.projected_parameter,
                              S.WIDTH_MIN_UM, S.WIDTH_MAX_UM, lr=1.6e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def full_itar() -> float:
        with torch.no_grad():
            widths = param.expand()
            return sum(float(field_weight[i])
                       * float(scorer.target_information_bits(widths, fp))
                       for i, fp in enumerate(field_points))

    progress.write(message="scoring the initial width map")
    best = full_itar()
    best_state = {k: v.detach().clone() for k, v in param.state_dict().items()}
    history = [[0, round(best, 5)]]
    progress.write(itar_history=history,
                   width_preview=width_preview(param.expand()),
                   message=f"init I_tar = {best:.4f} bit/raw px")

    # ---- optimization loop ------------------------------------------
    def bail_out() -> None:
        """Release the CUDA context promptly so the GPU memory returns."""
        progress.write(status="cancelled", message="cancelled by user")
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    for step in range(1, steps + 1):
        if progress.cancelled():
            bail_out()
            return 0
        optimizer.zero_grad(set_to_none=True)
        widths = param.expand()
        picks = torch.multinomial(field_weight.to("cpu", torch.float64),
                                  fields_per_step, replacement=True,
                                  generator=gen).tolist()
        loss = widths.new_zeros(())
        for pick in picks:
            # A single field evaluation can take seconds to minutes, so honor
            # cancellation between fields as well as between steps.
            if progress.cancelled():
                bail_out()
                return 0
            bits = scorer.target_information_bits(widths, field_points[pick])
            loss = loss - bits / fields_per_step
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % validate_every == 0 or step == steps:
            cur = full_itar()
            if cur > best:
                best = cur
                best_state = {k: v.detach().clone()
                              for k, v in param.state_dict().items()}
            history.append([step, round(cur, 5)])
            progress.write(itar_history=history,
                           width_preview=width_preview(param.expand()),
                           message=f"I_tar = {cur:.4f}  best = {best:.4f}")
        progress.step(step)

    # ---- results -----------------------------------------------------
    progress.write(message="saving the optimized width map")
    param.load_state_dict(best_state)
    final_map = param.expand().detach().cpu().to(torch.float32).contiguous()
    np.save(job_dir / "width.npy", final_map.numpy())
    torch.save({"width_um": final_map, "best_i_tar": float(best)},
               job_dir / "width.pt")

    progress.write(message="rendering PSF and MTF previews")
    render_psf_mtf(engine, final_map.to(device, dtype), job_dir)

    final = {"status": "done",
             "message": f"finished, best I_tar = {best:.4f} bit/raw px"}
    if device.type == "cuda":
        final["peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    progress.write(**final)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    args = ap.parse_args()
    job_dir = Path(args.job_dir)
    try:
        return run(job_dir)
    except Exception as exc:
        (job_dir / "worker.log").write_text(traceback.format_exc(),
                                            encoding="utf-8")
        state = {"status": "error", "step": 0, "total_steps": 0,
                 "itar_history": [], "width_preview": None, "eta_s": None,
                 "message": f"{type(exc).__name__}: {exc}"}
        try:
            prior = json.loads((job_dir / "progress.json").read_text("utf-8"))
            prior.update(status="error",
                         message=f"{type(exc).__name__}: {exc}")
            state = prior
        except Exception:
            pass
        tmp = job_dir / "progress.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, job_dir / "progress.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
