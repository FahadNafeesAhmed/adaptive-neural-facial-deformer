"""High-level solver API + `anfd-solve` command-line tool.

    anfd-solve scans/ -o performance.usda --method hybrid
    anfd-solve frame_0001.ply frame_0002.ply -o out.usda --units mm --json weights.json

Scans are point clouds (.ply ascii/binary, .npy, .xyz/.txt) in head-centred coordinates,
Y up, face looking toward +Z. A directory is read as a sequence in sorted filename order.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from anfd.facemodel import ROOT, FaceModel
from anfd.fitting import REFINE, RigFitter
from anfd.model import FaceSolverNet
from anfd.synth import CaptureSynth
from anfd.usd_io import export_rig

DEFAULT_CKPT = ROOT / "runs" / "solver" / "solver.pt"
UNITS_TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0}
PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1",
             "uint8": "u1", "char": "i1", "int8": "i1", "short": "i2", "int16": "i2",
             "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


# ---------------------------------------------------------------- point cloud IO
def read_ply(path: Path) -> np.ndarray:
    """Vertex x/y/z from an ascii or binary PLY (other elements after vertices are ignored)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, n, props, in_vertex = None, 0, [], False
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError("list properties on vertices are not supported")
                props.append((tok[2], PLY_TYPES[tok[1]]))
        if fmt == "ascii":
            rows = [f.readline().split() for _ in range(n)]
            arr = np.array(rows, dtype=np.float64)
            cols = [p for p, _ in props]
            return arr[:, [cols.index(c) for c in "xyz"]].astype(np.float32)
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + t) for name, t in props])
        data = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
        return np.stack([data[c] for c in "xyz"], 1).astype(np.float32)


def read_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        pts = read_ply(path)
    elif suffix == ".npy":
        pts = np.load(path).astype(np.float32)
    elif suffix in {".xyz", ".txt", ".csv"}:
        pts = np.loadtxt(path, delimiter="," if suffix == ".csv" else None, dtype=np.float32)
    else:
        raise ValueError(f"unsupported point cloud format: {path}")
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"{path}: expected an (N, 3) array, got {pts.shape}")
    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 256:
        raise ValueError(f"{path}: only {len(pts)} valid points; need at least 256")
    return pts


def resample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-size input for the network: random subset, or sampling with replacement."""
    return pts[rng.choice(len(pts), n, replace=len(pts) < n)]


# ---------------------------------------------------------------- solver
class Solver:
    """Loads the face model + trained network and solves batches of point clouds."""

    def __init__(self, checkpoint: Path | None = DEFAULT_CKPT, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fm = FaceModel.load()
        region = CaptureSynth(self.fm).to(self.device)
        self.fitter = RigFitter(region.rig, region.centre, region.tris)
        self.net = None
        if checkpoint is not None and Path(checkpoint).exists():
            self.net = FaceSolverNet(self.fm.n_id, self.fm.n_ex).to(self.device).eval()
            self.net.load_state_dict(torch.load(checkpoint, map_location=self.device)["model"])

    @torch.no_grad()
    def neural(self, pts: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.net is None:
            raise RuntimeError("no trained checkpoint loaded; train first or use method='icp'")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.net(pts)
        return {k: v.float() for k, v in out.items()}

    @torch.no_grad()
    def solve(self, pts: torch.Tensor, method: str = "hybrid", batch: int = 32) -> dict[str, torch.Tensor]:
        """pts (B, N, 3) in cm -> {rot, trans, id_w, ex_w}. method: neural | hybrid | icp."""
        outs = []
        for i in range(0, pts.shape[0], batch):
            p = pts[i : i + batch].to(self.device)
            if method == "icp":
                outs.append(self.fitter.fit_multistart(p))
            elif method == "neural":
                outs.append(self.neural(p))
            elif method == "hybrid":
                outs.append(self.fitter.fit(p, **self.neural(p), schedule=REFINE))
            else:
                raise ValueError(f"unknown method {method!r}")
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Solve raw facial scans into ICT/ARKit blendshape weights.")
    ap.add_argument("inputs", nargs="+", type=Path, help="point cloud files or directories (sequence)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output .usda/.usdc")
    ap.add_argument("--method", choices=["neural", "hybrid", "icp"], default="hybrid")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--units", choices=list(UNITS_TO_CM), default="cm", help="units of the input scans")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--json", type=Path, help="also write per-frame weights as JSON")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    files: list[Path] = []
    for p in args.inputs:
        files += sorted(q for q in p.iterdir() if q.is_file()) if p.is_dir() else [p]
    if not files:
        raise SystemExit("no input scans found")
    rng = np.random.default_rng(0)
    clouds = np.stack([resample(read_points(f), 4096, rng) for f in files]) * UNITS_TO_CM[args.units]

    solver = Solver(args.checkpoint, args.device)
    t0 = time.time()
    res = {k: v.cpu().numpy() for k, v in solver.solve(torch.from_numpy(clouds), args.method).items()}
    dt = time.time() - t0
    # One performer per sequence: identity is shared, so average it across frames.
    id_w = res["id_w"].mean(0)
    centre = solver.fitter.centre.cpu().numpy()
    export_rig(args.output, solver.fm, id_w, res["ex_w"], args.fps, (res["rot"], res["trans"], centre))
    if args.json:
        names = [str(n) for n in solver.fm.ex_names]
        frames = [{"file": f.name, "weights": dict(zip(names, map(float, w)))} for f, w in zip(files, res["ex_w"])]
        args.json.write_text(json.dumps({"method": args.method, "frames": frames}, indent=1), encoding="utf-8")
    print(f"solved {len(files)} frame(s) with {args.method} in {dt:.2f}s "
          f"({1000 * dt / len(files):.1f} ms/frame) -> {args.output}")


if __name__ == "__main__":
    main()
