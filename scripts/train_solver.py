"""Train the neural blendshape solver on endless synthetic captures.

    uv run python scripts/train_solver.py --steps 40000 --batch 64   (~2.5 h on an RTX 5060 laptop)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from anfd.facemodel import FaceModel
from anfd.fitting import RigFitter
from anfd.metrics import score
from anfd.model import FaceSolverNet
from anfd.synth import CaptureConfig, CaptureSynth, cached_eval_set

ROOT = Path(__file__).resolve().parents[1]


def losses(fitter: RigFitter, pred: dict, gt: dict) -> dict[str, torch.Tensor]:
    rig = fitter.rig
    posed = fitter.model(pred["id_w"], pred["ex_w"], pred["rot"], pred["trans"])
    canon_p, canon_g = rig(pred["id_w"], pred["ex_w"]), rig(gt["id_w"], gt["ex_w"])
    return {
        "posed": (posed - gt["verts"]).norm(dim=-1).mean(),
        "canon": (canon_p - canon_g).norm(dim=-1).mean(),
        "expr": (rig(gt["id_w"], pred["ex_w"]) - canon_g).norm(dim=-1).mean(),
        "weights": (pred["ex_w"] - gt["ex_w"]).abs().mean(),
        "rot": (pred["rot"] - gt["rot"]).abs().mean(),
    }


LOSS_W = {"posed": 1.0, "canon": 1.0, "expr": 2.0, "weights": 2.0, "rot": 1.0}


@torch.no_grad()
def evaluate(net, fitter, data, bs=64) -> dict[str, float]:
    net.eval()
    out: dict[str, list] = {}
    for i in range(0, data["points"].shape[0], bs):
        gt = {k: v[i : i + bs] for k, v in data.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(gt["points"])
        pred = {k: v.float() for k, v in pred.items()}
        for k, v in score(fitter, pred, gt).items():
            out.setdefault(k, []).append(v)
    net.train()
    return {k: torch.cat(v).mean().item() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--run", default="solver")
    ap.add_argument("--eval-every", type=int, default=2000)
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    fm = FaceModel.load()
    synth = CaptureSynth(fm, CaptureConfig()).to(dev)
    fitter = RigFitter(synth.rig, synth.centre, synth.tris)
    net = FaceSolverNet(fm.n_id, fm.n_ex).to(dev)
    net.centre.copy_(synth.centre)

    evals = {
        "noisy": cached_eval_set(synth, 512, 1234, ROOT / "data" / "eval_noisy.pt"),
        "clean": cached_eval_set(CaptureSynth(fm, CaptureConfig.clean()).to(dev), 512, 4321,
                                 ROOT / "data" / "eval_clean.pt"),
    }
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.03)
    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    def record(rec: dict, tag: str = "") -> None:
        print(tag + json.dumps(rec), flush=True)
        with open(run_dir / "log.jsonl", "a") as log:
            log.write(json.dumps(rec) + "\n")

    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        gt = synth(args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(gt["points"])
        pred = {k: v.float() for k, v in pred.items()}
        parts = losses(fitter, pred, gt)
        loss = sum(LOSS_W[k] * v for k, v in parts.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            rec = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()},
                   "sec": round(time.time() - t0, 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            record(rec)
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, **{f"{n}/{k}": v for n, d in evals.items()
                                    for k, v in evaluate(net, fitter, d).items()}}
            record(rec, "EVAL ")
            torch.save({"model": net.state_dict(), "step": step, "args": vars(args)},
                       run_dir / "solver.pt")


if __name__ == "__main__":
    main()
