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
on CPU and is much faster on CUDA. Exit status is non-zero if any weighted I_tar
deviates from records/expected.json by more than the relative tolerance.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--designs", default="all",
                    choices=["all", "reference", "information", "mtf"])
    ap.add_argument("--tolerance", type=float, default=1e-2)
    args = ap.parse_args()
    device = resolve_device(args.device)

    print("=" * 68)
    print("metalens-information-design  |  reproduce.py  (full-Jones I_tar)")
    print("=" * 68)
    print(f"python {platform.python_version()}  torch {torch.__version__}  "
          f"device {device.type}")
    print("-" * 68)

    t0 = time.time()
    engine = S.build_engine(device)
    protocol = S.load_prior(device)
    scorer = S.TargetInformationScorer(engine, protocol, device)
    field_points, field_weight = S.make_field_points(engine, 5)
    print(f"engine + {len(field_points)}-field quadrature ready "
          f"({time.time() - t0:.0f}s)")
    print("-" * 68)

    expected = json.loads(EXPECTED.read_text()) if EXPECTED.is_file() else None
    keys = list(DESIGNS) if args.designs == "all" else [args.designs]
    results, failed = {}, False
    print(f"  {'design':<22s}{'I_tar':>10s}{'expected':>10s}{'rel':>9s}")
    for key in keys:
        label, path = DESIGNS[key]
        widths = torch.load(path, map_location="cpu", weights_only=True)["width_um"]
        t = time.time()
        total, _ = scorer.score(widths, field_points, field_weight)
        results[key] = total
        exp = expected["i_tar"][key] if expected else None
        rel = abs(total - exp) / exp if exp else None
        flag = ""
        if rel is not None and rel > args.tolerance:
            flag, failed = "  FAIL", True
        exp_s = f"{exp:>10.4f}" if exp else f"{'-':>10s}"
        rel_s = f"{100*rel:>8.2f}%" if rel is not None else f"{'-':>9s}"
        print(f"  {label:<22s}{total:>10.4f}{exp_s}{rel_s}{flag}  ({time.time()-t:.0f}s)")

    print("-" * 68)
    print("I_tar is the delivered target information in bit/raw-pixel "
          "(manuscript Table 1).")
    if expected:
        print(f"comparison tolerance: {args.tolerance:g} relative "
              "(CPU/CUDA float differences are sub-percent).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
