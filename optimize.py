#!/usr/bin/env python
"""Optimize a meta-atom width map for camera-delivered target information.

Reproduces the optimization method of the paper. Starting from the projected
hyperbolic reference, it maximizes the target information I_tar through the
rigorous full-Jones forward model with:

    - the mirror-quadrant width parameterization on the D=208 um aperture,
    - 300 projected-Adam steps with four stochastic field samples per step,
    - a cosine learning-rate schedule from 1.6e-2 to 1e-5,
    - deterministic 25-field validation every ten steps (best kept), and
    - a five-step full-field refinement at a constant 1e-4.

Re-running does NOT return the published width map exactly: the field draw is
stochastic and the objective is non-convex. The published maps are in designs/.

The vectorial forward is heavy (each field is a full 2x2 Jones propagation over
nine wavelengths), so a full run is only practical on CUDA. Use --steps for a
short smoke test on CPU.

    python optimize.py --device cuda --out out/optimized.pt
    python optimize.py --device cpu --steps 2      # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from mosaic_metalens.fulljones import scoring as S
from mosaic_metalens.fulljones.optim_widthmap import ProjectedMirrorQuadrantWidthParam
from mosaic_metalens.fulljones.projected_width_optimizer import ProjectedAdam

HERE = Path(__file__).resolve().parent


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1.6e-2)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--fields-per-step", type=int, default=4)
    ap.add_argument("--validate-every", type=int, default=10)
    ap.add_argument("--refine-steps", type=int, default=5)
    ap.add_argument("--refine-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--init", default="designs/hyperbolic_reference.pt")
    ap.add_argument("--out", default="out/optimized.pt")
    args = ap.parse_args()
    device = resolve_device(args.device)
    dtype = torch.float64

    print("building full-Jones engine and target-information objective ...", flush=True)
    engine = S.build_engine(device)
    protocol = S.load_prior(device, dtype)
    scorer = S.TargetInformationScorer(engine, protocol, device, dtype)
    field_points, field_weight = S.make_field_points(engine, 5)
    field_weight = field_weight.to(device)

    init_map = torch.load(HERE / args.init, map_location="cpu",
                          weights_only=True)["width_um"].to(device, dtype).clamp(0.10, 0.24)
    param = ProjectedMirrorQuadrantWidthParam(
        S.PUPIL_GRID, S.PUPIL_GRID, init_w_full_um=init_map,
        width_range_um=(S.WIDTH_MIN_UM, S.WIDTH_MAX_UM),
        pupil_pitch_um=S.PILLAR_PITCH_UM, aperture_radius_um=S.D_UM * 0.5,
    ).to(device)
    optimizer = ProjectedAdam(param.projected_parameter,
                              S.WIDTH_MIN_UM, S.WIDTH_MAX_UM, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr_min)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def full_itar() -> float:
        with torch.no_grad():
            widths = param.expand()
            return sum(float(field_weight[i]) * float(scorer.target_information_bits(widths, fp))
                       for i, fp in enumerate(field_points))

    best = full_itar()
    best_state = {k: v.detach().clone() for k, v in param.state_dict().items()}
    print(f"init  I_tar = {best:.4f}  (25-field quadrature)", flush=True)

    def run(n_steps, sched, per_step_fields, label):
        nonlocal best, best_state
        for step in range(1, n_steps + 1):
            t = time.time()
            optimizer.zero_grad(set_to_none=True)
            widths = param.expand()
            if per_step_fields is None:
                picks = list(range(len(field_points)))
                scale = field_weight
            else:
                picks = torch.randint(0, len(field_points), (per_step_fields,),
                                      generator=gen).tolist()
                scale = None
            loss = widths.new_zeros(())
            for j, pick in enumerate(picks):
                bits = scorer.target_information_bits(widths, field_points[pick])
                w = float(scale[pick]) if scale is not None else 1.0 / len(picks)
                loss = loss - w * bits
            loss.backward()
            optimizer.step()
            if sched is not None:
                sched.step()
            if step % args.validate_every == 0 or step == n_steps:
                cur = full_itar()
                if cur > best:
                    best = cur
                    best_state = {k: v.detach().clone() for k, v in param.state_dict().items()}
                lr = optimizer.param_groups[0]["lr"]
                print(f"[{label}] step {step:3d}/{n_steps}  I_tar={cur:.4f}  "
                      f"best={best:.4f}  lr={lr:.2e}  ({time.time()-t:.0f}s/step)", flush=True)

    run(args.steps, scheduler, args.fields_per_step, "main")
    # Five-step full-field refinement at a constant rate.
    param.load_state_dict(best_state)
    for g in optimizer.param_groups:
        g["lr"] = args.refine_lr
    run(args.refine_steps, None, None, "refine")

    param.load_state_dict(best_state)
    final_map = param.expand().detach().cpu().to(torch.float32).contiguous()
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width_um": final_map, "design": "optimized",
                "best_i_tar": float(best)}, out)
    print(f"\nsaved optimized width map to {out}  (best I_tar {best:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
