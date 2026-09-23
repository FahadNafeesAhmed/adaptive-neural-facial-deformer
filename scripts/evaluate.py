"""Head-to-head evaluation: classical ICP vs neural vs hybrid, on identical seeded scans.

    uv run python scripts/evaluate.py --run solver --n 256

Writes runs/<run>/results.json and runs/<run>/results.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from anfd.fitting import REFINE
from anfd.metrics import score
from anfd.solve import Solver
from anfd.synth import CaptureConfig, CaptureSynth, cached_eval_set, yaw_of

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["surface_mm", "expr_mm", "shape_mm", "weight_mae", "rot_deg"]
YAW_BINS = [(0, 10), (10, 20), (20, 35)]


def run_method(solver: Solver, name: str, data: dict, batch: int) -> tuple[dict, float]:
    pts = data["points"]
    if name in ("neural", "hybrid"):
        solver.neural(pts[:batch])  # warm up CUDA kernels so timing reflects steady state
    torch.cuda.synchronize()
    t0 = time.time()
    if name == "icp (true pose given)":  # upper bound: not available in practice
        outs = [solver.fitter.fit(pts[i:i + batch], rot=data["rot"][i:i + batch],
                                  trans=data["trans"][i:i + batch], schedule=REFINE * 4)
                for i in range(0, len(pts), batch)]
        pred = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    else:
        pred = solver.solve(pts, name, batch)
    torch.cuda.synchronize()
    return pred, 1000 * (time.time() - t0) / len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="solver")
    ap.add_argument("--n", type=int, default=256, help="scans per test set (ICP is ~1.5 s each)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-icp", action="store_true", help="skip the slow classical baseline")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    solver = Solver(run_dir / "solver.pt", "cuda")
    methods = ["neural", "hybrid"] + ([] if args.skip_icp else ["icp", "icp (true pose given)"])
    sets = {
        "clean": (CaptureConfig.clean(), 4321, "eval_clean.pt"),
        "noisy": (CaptureConfig(), 1234, "eval_noisy.pt"),
    }

    results: dict = {}
    for set_name, (cfg, seed, fname) in sets.items():
        synth = CaptureSynth(solver.fm, cfg).to(solver.device)
        data = cached_eval_set(synth, 512, seed, ROOT / "data" / fname)
        data = {k: v[: args.n] for k, v in data.items()}
        yaw = yaw_of(data["rot"]).abs()
        for m in methods:
            pred, ms = run_method(solver, m, data, args.batch)
            s = score(solver.fitter, pred, data)
            row = {k: s[k].mean().item() for k in METRICS}
            row["surface_mm_p90"] = s["surface_mm"].quantile(0.9).item()
            row["ms_per_frame"] = ms
            row["by_yaw"] = {f"{lo}-{hi}": s["surface_mm"][(yaw >= lo) & (yaw < hi)].mean().item()
                             for lo, hi in YAW_BINS}
            results.setdefault(set_name, {})[m] = row
            print(f"{set_name:6s} {m:22s} " + " ".join(f"{k}={row[k]:.3f}" for k in METRICS)
                  + f" p90={row['surface_mm_p90']:.2f} {ms:.1f}ms", flush=True)

    # Single-scan latency of the network (what an interactive tool would feel).
    one = data["points"][:1]
    for _ in range(3):
        solver.neural(one)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        solver.neural(one)
    torch.cuda.synchronize()
    results["neural_latency_ms_batch1"] = 1000 * (time.time() - t0) / 20

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [f"Evaluated on {args.n} seeded synthetic scans per set. Errors in mm on the face region.",
             "", "| set | method | surface mm | p90 | expression mm | weight MAE | rot ° | ms/frame |",
             "|---|---|---|---|---|---|---|---|"]
    for set_name in sets:
        for m, r in results[set_name].items():
            lines.append(f"| {set_name} | {m} | {r['surface_mm']:.2f} | {r['surface_mm_p90']:.2f} | "
                         f"{r['expr_mm']:.2f} | {r['weight_mae']:.3f} | {r['rot_deg']:.2f} | "
                         f"{r['ms_per_frame']:.1f} |")
    lines += ["", "Surface error (mm) on noisy scans by head yaw:", "",
              "| method | " + " | ".join(f"yaw {lo}-{hi}°" for lo, hi in YAW_BINS) + " |",
              "|---|" + "---|" * len(YAW_BINS)]
    for m, r in results["noisy"].items():
        lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in r["by_yaw"].values()) + " |")
    lines += ["", f"Network latency, single scan: {results['neural_latency_ms_batch1']:.1f} ms"]
    (run_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
