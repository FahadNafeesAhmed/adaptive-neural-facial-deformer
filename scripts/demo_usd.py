"""Make a demo performance and open it in Blender (File > Import > Universal Scene Description).

    uv run python scripts/demo_usd.py --run solver --method hybrid

Synthesises a smooth 4-second facial performance with head motion, scans every frame with
realistic corruption, solves the scans, and writes to runs/<run>/demo/:
  truth.usda   ground-truth rig + animation
  scan.usda    the raw point clouds the solver saw
  solved.usda  the solved rig (53 editable blendshapes + weight curves + head transform)
  scene.usda   all three side by side  <- import this one
  curves.png   truth vs solved weight curves for the most active shapes (needs matplotlib)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from anfd.metrics import score
from anfd.solve import Solver
from anfd.synth import CaptureConfig, CaptureSynth, euler_to_matrix
from anfd.usd_io import export_point_clouds, export_rig, export_scene

ROOT = Path(__file__).resolve().parents[1]


def performance(synth: CaptureSynth, frames: int, fps: float, g: torch.Generator):
    """Keyframed expressions every 0.5 s with smoothstep blending, plus a slow head turn."""
    dev = synth.centre.device
    step = int(fps // 2)
    keys = synth.sample_expression(frames // step + 2, g)
    keys[0] = 0  # start from neutral
    t = torch.arange(frames, device=dev) / step
    i, f = t.long(), (t % 1)[:, None]
    f = f * f * (3 - 2 * f)
    ex = keys[i] * (1 - f) + keys[i + 1] * f
    sec = torch.arange(frames, device=dev) / fps
    yaw = torch.deg2rad(25 * torch.sin(2 * math.pi * sec / 4))
    pitch = torch.deg2rad(8 * torch.sin(2 * math.pi * sec / 3))
    rot = euler_to_matrix(yaw, pitch, torch.zeros_like(yaw))
    trans = torch.stack([0.5 * torch.sin(2 * math.pi * sec / 2), torch.zeros_like(sec),
                         torch.zeros_like(sec)], -1)
    id_w = torch.randn(1, synth.n_id, device=dev, generator=g).expand(frames, -1)
    return id_w, ex, rot, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="solver")
    ap.add_argument("--method", choices=["neural", "hybrid", "icp"], default="hybrid")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    solver = Solver(ROOT / "runs" / args.run / "solver.pt")
    dev = solver.device
    synth = CaptureSynth(solver.fm, CaptureConfig()).to(dev)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    frames = int(args.seconds * args.fps)
    id_w, ex, rot, trans = performance(synth, frames, args.fps, g)
    verts = synth.pose(synth.rig(id_w, ex), rot, trans)
    clouds = synth.scan(verts, g)

    pred = solver.solve(clouds, args.method)
    gt = {"id_w": id_w, "ex_w": ex, "rot": rot, "trans": trans, "verts": verts}
    s = score(solver.fitter, pred, gt)
    print(f"{args.method}: surface {s['surface_mm'].mean():.2f} mm, expression {s['expr_mm'].mean():.2f} mm, "
          f"weight MAE {s['weight_mae'].mean():.3f}")

    out = ROOT / "runs" / args.run / "demo"
    out.mkdir(parents=True, exist_ok=True)
    fm, centre = solver.fm, synth.centre.cpu().numpy()
    np_ = lambda x: x.detach().cpu().numpy()
    export_rig(out / "truth.usda", fm, np_(id_w[0]), np_(ex), args.fps, (np_(rot), np_(trans), centre))
    export_point_clouds(out / "scan.usda", np_(clouds), args.fps)
    export_rig(out / "solved.usda", fm, np_(pred["id_w"]).mean(0), np_(pred["ex_w"]), args.fps,
               (np_(pred["rot"]), np_(pred["trans"]), centre))
    export_scene(out / "scene.usda", {"Truth": (out / "truth.usda", -30.0),
                                      "Scan": (out / "scan.usda", 0.0),
                                      "Solved": (out / "solved.usda", 30.0)}, args.fps, frames)
    print(f"wrote {out / 'scene.usda'}  (Blender: File > Import > Universal Scene Description)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    top = np_(ex).max(0).argsort()[::-1][:6]
    fig, axes = plt.subplots(len(top), 1, figsize=(9, 1.6 * len(top)), sharex=True)
    sec = np.arange(frames) / args.fps
    for ax, j in zip(axes, top):
        ax.plot(sec, np_(ex)[:, j], color="#333333", lw=2, label="truth")
        ax.plot(sec, np_(pred["ex_w"])[:, j], color="#d9534f", lw=1.5, label=f"solved ({args.method})")
        ax.set_ylabel(fm.ex_names[j], rotation=0, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    print(f"wrote {out / 'curves.png'}")


if __name__ == "__main__":
    main()
