r"""Local web console for I_tar width-map optimization.

Serves the static frontend and a small JSON API. Each optimization job runs in
its own worker subprocess (webapp/run_job.py) that writes <job>/progress.json
atomically after every step; this server polls that file on a slow cadence and
caches the state, so the browser can poll this server cheaply.

Run from the repository root or from webapp/:

    python webapp/server.py            # http://127.0.0.1:8642
    python webapp/server.py --mock     # exercise the UI without a GPU

A mock runner is also available per-job by POSTing {"mock": true}.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
JOBS_DIR = HERE / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="itar-designer")

# ----------------------------------------------------------------- meta
LIBRARIES = [
    {"id": "SiN", "label": "SiN square posts, h = 1000 nm, pitch 290 nm"},
]
_FALLBACK_CFAS = [
    {"id": "samsung_galaxy_s20", "label": "Samsung Galaxy S20 RGGB"},
    {"id": "boxcar", "label": "synthetic boxcar (zero-crosstalk ablation)"},
    {"id": "gaussian", "label": "synthetic Gaussian (smooth ablation)"},
]
DEFAULTS = {
    "library": "SiN", "cfa": "samsung_galaxy_s20",
    "focal_um": 346.7, "diameter_um": 208.0, "pixel_um": 1.0,
    "fov_deg": 10.0, "pupil_grid": 720, "steps": 120,
    "fields_per_step": 4, "seed": 20260908,
}


def cfa_choices() -> list[dict]:
    """CFA list from the vendored library; static fallback without torch."""
    try:
        sys.path.insert(0, str(HERE.parent))
        from mosaic_metalens.fulljones.cfa_library import AVAILABLE_CAMERAS
        return [{"id": name, "label": name.replace("_", " ")}
                for name in AVAILABLE_CAMERAS]
    except Exception:
        return _FALLBACK_CFAS


class JobConfig(BaseModel):
    library: str = "SiN"
    cfa: str = "samsung_galaxy_s20"
    focal_um: float = Field(346.7, gt=10.0)
    diameter_um: float = Field(208.0, gt=10.0)
    pixel_um: float = Field(1.0, ge=1.0, le=1.0,
                            description="fixed by the 64-alias contract")
    fov_deg: float = Field(10.0, ge=0.0, le=20.0)
    pupil_grid: int = Field(720, ge=240, le=2048)
    steps: int = Field(120, ge=1, le=1000)
    fields_per_step: int = Field(4, ge=1, le=25)
    seed: int = 20260908
    validate_every: int = Field(10, ge=1, le=100)
    quadrature_n: int = Field(5, ge=1, le=5)
    device: str = Field("auto", pattern="^(auto|cuda|cpu)$")
    allow_cpu: bool = False
    mock: bool = False


class JobState:
    def __init__(self, job_id: str, config: JobConfig):
        self.job_id = job_id
        self.config = config
        self.status = "queued"
        self.step = 0
        self.total_steps = config.steps
        self.itar_history: list[list[float]] = []
        self.width_preview: dict | None = None
        self.message = ""
        self.started_at = time.time()
        self.eta_s: float | None = None
        self.peak_vram_gb: float | None = None
        self.cancel_requested = False
        self.proc: subprocess.Popen | None = None
        self.dir = JOBS_DIR / job_id
        self.dir.mkdir(exist_ok=True)

    def snapshot(self) -> dict:
        out = {
            "status": self.status, "step": self.step,
            "total_steps": self.total_steps,
            "itar_history": self.itar_history,
            "width_preview": self.width_preview,
            "message": self.message, "started_at": self.started_at,
            "eta_s": self.eta_s,
        }
        if self.peak_vram_gb is not None:
            out["peak_vram_gb"] = self.peak_vram_gb
        return out


JOBS: dict[str, JobState] = {}
FORCE_MOCK = False


# ----------------------------------------------------------------- worker runner
def _worker_runner(job: JobState) -> None:
    """Spawn run_job.py and mirror its progress.json into the job state."""
    worker = HERE / "run_job.py"
    log = open(job.dir / "server_worker.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(worker), "--job-dir", str(job.dir)],
        cwd=str(HERE.parent), stdout=log, stderr=subprocess.STDOUT)
    job.proc = proc
    job.status = "running"
    progress_path = job.dir / "progress.json"
    while True:
        time.sleep(2.0)
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            state = None
        if state:
            job.status = state.get("status", job.status)
            job.step = int(state.get("step", job.step))
            job.total_steps = int(state.get("total_steps", job.total_steps))
            job.itar_history = state.get("itar_history", job.itar_history)
            job.width_preview = state.get("width_preview", job.width_preview)
            job.message = state.get("message", job.message)
            job.eta_s = state.get("eta_s", job.eta_s)
            job.peak_vram_gb = state.get("peak_vram_gb", job.peak_vram_gb)
        if proc.poll() is not None:
            break
    log.close()
    # Final read after exit; if the worker died before writing a terminal
    # status, surface the failure instead of leaving the UI on "running".
    try:
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        job.status = state.get("status", "error")
        job.message = state.get("message", job.message)
        job.itar_history = state.get("itar_history", job.itar_history)
        job.width_preview = state.get("width_preview", job.width_preview)
        job.peak_vram_gb = state.get("peak_vram_gb", job.peak_vram_gb)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if job.status not in ("done", "error", "cancelled"):
        if job.cancel_requested:
            job.status = "cancelled"
            job.message = "cancelled (worker terminated, GPU memory released)"
        else:
            job.status = "error"
            job.message = (f"worker exited with code {proc.returncode} "
                           "before reporting a result (see server_worker.log)")


# ----------------------------------------------------------------- mock runner
def _mock_runner(job: JobState) -> None:
    """Simulated optimization so the UI loop can be tested without a GPU."""
    cfg = job.config
    rng = np.random.default_rng(cfg.seed)
    n = 96
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) / (n / 2)
    width = np.clip(0.17 + 0.05 * np.cos(9 * np.pi * r * r), 0.10, 0.24)
    job.status = "running"
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        if job.cancel_requested or (job.dir / "cancel").exists():
            job.status = "cancelled"
            job.message = "cancelled by user"
            return
        time.sleep(0.15)
        itar = 0.5059 + (0.5860 - 0.5059) * (1 - math.exp(-3.2 * step / cfg.steps))
        itar += float(rng.normal(0, 0.0008))
        width = np.clip(
            width + rng.normal(0, 0.0012, size=width.shape) * (1 - step / cfg.steps),
            0.10, 0.24)
        job.step = step
        if step % 2 == 0 or step == cfg.steps:
            job.itar_history.append([step, round(itar, 5)])
            job.width_preview = {
                "w": n, "h": n, "min": float(width.min()),
                "max": float(width.max()),
                "data": [round(float(v), 4) for v in width.ravel()],
            }
        job.eta_s = (time.time() - t0) / step * (cfg.steps - step)
    np.save(job.dir / "width.npy", width)
    _mock_psf_mtf(job)
    job.status = "done"
    job.message = f"finished, I_tar = {job.itar_history[-1][1]:.4f} bit/raw px"
    job.eta_s = 0.0


def _mock_psf_mtf(job: JobState) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(-6, 6, 241)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (lam, c) in zip(axes, [(450, "#2B5FD9"), (550, "#1F9C4B"),
                                   (650, "#CC2A1E")]):
        s = 1.22 * lam / 550
        psf = (np.sinc(np.outer(x, np.ones_like(x)) / s) ** 2
               * np.sinc(np.outer(np.ones_like(x), x) / s) ** 2)
        ax.imshow(psf, cmap="inferno")
        ax.set_title(f"{lam} nm (mock)", fontsize=8, color=c)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(job.dir / "psf.png", dpi=110)
    plt.close(fig)
    f = np.linspace(0, 1.3, 80)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for lam_eff, c, lab in [(608, "#CC2A1E", "R"), (546, "#1F9C4B", "G"),
                            (480, "#2B5FD9", "B")]:
        cut = 2 * 0.287 / (lam_eff / 1000.0)
        ax.plot(f, np.clip(1 - f / cut, 0, None) * 0.6, color=c, label=lab)
    ax.set_xlabel("f (cycles/um)"); ax.set_ylabel("channel MTF (mock)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(job.dir / "mtf.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------- endpoints
@app.get("/api/meta")
def meta() -> dict:
    return {"libraries": LIBRARIES, "cfas": cfa_choices(),
            "defaults": DEFAULTS}


@app.post("/api/jobs")
def create_job(config: JobConfig) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = JobState(job_id, config)
    JOBS[job_id] = job
    (job.dir / "config.json").write_text(config.model_dump_json(indent=2),
                                         encoding="utf-8")
    runner = _mock_runner if (config.mock or FORCE_MOCK) else _worker_runner
    threading.Thread(target=runner, args=(job,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.snapshot()


def _enforce_cancel(job: JobState, grace_s: float = 15.0) -> None:
    """Give the worker a grace period to exit cleanly, then kill it.

    Killing the worker process releases its CUDA context, so cancelling always
    returns the GPU memory even if the worker is stuck inside a long field
    evaluation.
    """
    proc = job.proc
    if proc is None:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass
        job.status = "cancelled"
        job.message = "cancelled (worker terminated, GPU memory released)"


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.cancel_requested = True
    (job.dir / "cancel").write_text("", encoding="utf-8")
    threading.Thread(target=_enforce_cancel, args=(job,), daemon=True).start()
    return {"ok": True}


@app.get("/api/jobs/{job_id}/{fname}")
def job_file(job_id: str, fname: str):
    if fname not in ("psf.png", "mtf.png", "width.npy", "width.pt"):
        raise HTTPException(404, "unknown file")
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    path = job.dir / fname
    if not path.exists():
        raise HTTPException(404, "not ready")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--mock", action="store_true",
                        help="force every job onto the mock runner")
    args = parser.parse_args()
    FORCE_MOCK = args.mock
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)
