# Web console for I_tar width-map optimization

A local, single-user web interface around the optimization method of the paper:
choose the lens specification (focal length, aperture diameter, CFA spectrum,
field of view), run the target-information (`I_tar`) width-map optimization on
your own GPU, and watch the width map, the `I_tar` trace, and the resulting
on-axis PSF/MTF as the run progresses. The forward model and objective are the
same vendored full-Jones engine used by `optimize.py` in the repository root.

## Install

From the repository root:

```
pip install -r requirements.txt
pip install fastapi uvicorn matplotlib
```

## Run

```
python webapp/server.py
```

then open <http://127.0.0.1:8642>. To exercise the interface without a GPU,
start the server with `--mock` (or append `?mock=1` to the URL for a purely
client-side demo).

Each job runs in its own worker process (`webapp/run_job.py`) and writes its
state to `webapp/jobs/<id>/` (`config.json`, `progress.json`, `width.npy`,
`width.pt`, `psf.png`, `mtf.png`, `worker.log` on failure).

## Hardware requirements

The vectorial forward model (full 2x2 Jones propagation over nine wavelengths,
float64) is only practical on CUDA. Peak GPU memory grows with the pupil grid:

| pupil grid | aperture it can hold | peak VRAM | note |
|---|---|---|---|
| 240 | up to 69 um | to be measured | smoke tests |
| 360 | up to 104 um | to be measured | |
| 480 | up to 139 um | to be measured | |
| 720 | up to 208 um | to be measured | production geometry |

The full production configuration (720 grid, 25-field quadrature, 300 steps)
runs in about 20 minutes on a 48 GB RTX 5880 at roughly 4 s/step; the completed
job reports its measured `peak_vram_gb` in the UI, which is also how the table
above will be filled in. If you run out of memory, reduce the aperture diameter
(the pupil grid follows it) before reducing anything else.

CPU execution is supported only as a smoke test: POST a job with
`"allow_cpu": true` and a small step count, or run the worker directly:

```
python webapp/run_job.py --job-dir <dir with config.json>
```

## What is fixed and what is free

Free per job: focal length, aperture diameter (the pupil grid is derived from
it and can be raised manually), FOV half-angle (up to 20 degrees; it caps the
field quadrature in object space), CFA spectrum (measured camera responses from
the bundled library, plus the boxcar/Gaussian ablations), optimization steps,
stochastic field samples per step, and the seed.

Fixed by the sampling contract (not configurable): the 0.25 um scene grid, the
1.0 um photosite pitch, and the resulting 64-alias polyphase measurement
operator, the nine design wavelengths (420-670 nm), and the SiN meta-atom
library (square posts, height 1000 nm, pitch 290 nm) whose full-Jones response
table ships in `data/fulljones/`. The optimizer recipe follows `optimize.py`:
projected Adam on the mirror-quadrant parameterization, cosine learning rate
1.6e-2 to 1e-5, deterministic full-field validation with best-checkpoint
selection.
