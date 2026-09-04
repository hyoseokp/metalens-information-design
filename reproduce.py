#!/usr/bin/env python
"""Reproduce the paper's target-information numbers (Table 1).

    python reproduce.py                 # all three designs, auto device
    python reproduce.py --device cuda
    python reproduce.py --designs information

Scores the three published width maps through the rigorous full-Jones forward
model and the target-information objective I_tar of the manuscript. It does not
re-run the optimization; the optimizer is described in the paper and the stored
width maps are the published artefact.

The forward model is vectorial (full 2x2 Jones, angle-resolved) and evaluates 25
field points over nine wavelengths, so a full run takes roughly 15 min per design
on CPU with a peak memory of about 14 GB, and is much faster on CUDA. Exit
status is non-zero if any weighted I_tar deviates from records/expected.json by
more than the relative tolerance.

Memory. Two environment variables lower the peak on a host that cannot hold
the default batch. ENGINE2_VEC_SUBBATCH=1 propagates one dipole per pass
(peak about 8 GB) and returns bit-identical numbers. ENGINE2_WAVELENGTH_CHUNK=1
also builds and propagates one wavelength per pass (peak about 6 GB) and agrees
with the default to about 1e-12 relative. A failed allocation exits with status 3
so a wrapper can retry with those settings. With --results-dir each design's
score is written as JSON when it completes and read back on the next run, so a
retry resumes instead of starting over.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

MEMORY_EXIT_STATUS = 3
MEMORY_SETTINGS = ("ENGINE2_VEC_SUBBATCH", "ENGINE2_WAVELENGTH_CHUNK")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mosaic_metalens.fulljones import scoring as S  # noqa: E402

DESIGNS = {
    "reference": ("hyperbolic reference", HERE / "designs/hyperbolic_reference.pt"),
    "information": ("information design", HERE / "designs/information_design.pt"),
    "mtf": ("MTF-volume control", HERE / "designs/mtf_volume_control.pt"),
}
EXPECTED = HERE / "records" / "expected.json"


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def is_memory_error(error: BaseException) -> bool:
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    return "not enough memory" in message or "out of memory" in message


def memory_settings() -> str:
    active = [f"{name}={os.environ[name]}" for name in MEMORY_SETTINGS
              if os.environ.get(name)]
    return " ".join(active) if active else "default"


def score_designs(args, keys, expected, results_dir: Path | None) -> bool:
    device = resolve_device(args.device)
    print(f"python {platform.python_version()}  torch {torch.__version__}  "
          f"device {device.type}  memory settings {memory_settings()}")
    print("-" * 68)

    t0 = time.time()
    engine = S.build_engine(device)
    protocol = S.load_prior(device)
    scorer = S.TargetInformationScorer(engine, protocol, device)
    field_points, field_weight = S.make_field_points(engine, 5)
    print(f"engine + {len(field_points)}-field quadrature ready "
          f"({time.time() - t0:.0f}s)")
    print("-" * 68)

    failed = False
    print(f"  {'design':<22s}{'I_tar':>10s}{'expected':>10s}{'rel':>9s}")
    for key in keys:
        label, path = DESIGNS[key]
        record_path = results_dir / f"i_tar_{key}.json" if results_dir else None
        t = time.time()
        if record_path is not None and record_path.is_file():
            record = json.loads(record_path.read_text())
            total, per_field = record["i_tar"], record["per_field"]
            note = "  (from earlier attempt)"
        else:
            widths = torch.load(path, map_location="cpu", weights_only=True)["width_um"]
            total, per_field = scorer.score(widths, field_points, field_weight)
            note = f"  ({time.time()-t:.0f}s)"
        exp = expected["i_tar"][key] if expected else None
        rel = abs(total - exp) / exp if exp else None
        flag = ""
        if rel is not None and rel > args.tolerance:
            flag, failed = "  FAIL", True
        exp_s = f"{exp:>10.4f}" if exp else f"{'-':>10s}"
        rel_s = f"{100*rel:>8.2f}%" if rel is not None else f"{'-':>9s}"
        print(f"  {label:<22s}{total:>10.4f}{exp_s}{rel_s}{flag}{note}")
        if record_path is not None and not record_path.is_file():
            record_path.write_text(json.dumps({
                "design": label, "key": key, "i_tar": total,
                "per_field": per_field, "expected": exp,
                "relative_deviation": rel, "device": device.type,
                "memory_settings": memory_settings(),
            }, indent=2))
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--designs", default="all",
                    choices=["all", "reference", "information", "mtf"])
    ap.add_argument("--tolerance", type=float, default=1e-2)
    ap.add_argument("--results-dir", default=None,
                    help="write each design's score here as JSON and reuse "
                         "scores already present, so a retry resumes")
    args = ap.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("metalens-information-design  |  reproduce.py  (full-Jones I_tar)")
    print("=" * 68)

    expected = json.loads(EXPECTED.read_text()) if EXPECTED.is_file() else None
    keys = list(DESIGNS) if args.designs == "all" else [args.designs]
    try:
        failed = score_designs(args, keys, expected, results_dir)
    except (MemoryError, RuntimeError) as error:
        if not is_memory_error(error):
            raise
        print("-" * 68)
        print(f"stopped: memory allocation failed ({str(error).splitlines()[0][:120]})")
        print("rerun with ENGINE2_VEC_SUBBATCH=1, then also "
              "ENGINE2_WAVELENGTH_CHUNK=1, to lower the peak memory.")
        return MEMORY_EXIT_STATUS

    print("-" * 68)
    print("I_tar is the delivered target information in bit/raw-pixel "
          "(manuscript Table 1).")
    if expected:
        print(f"comparison tolerance: {args.tolerance:g} relative "
              "(CPU/CUDA float differences are sub-percent).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
