"""ICT-FaceKit linear face model: loading, caching, and evaluation.

The model is  V = neutral + sum_i id_w[i] * id_mode[i] + sum_j ex_w[j] * ex_mode[j]
where each mode is (shape mesh - neutral mesh). Identity weights are ~N(0, 1);
expression weights live in [0, 1] like ARKit blendshapes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ICT_DIR = ROOT / "external" / "ICT-FaceKit" / "FaceXModel"
CACHE = Path(os.environ.get("ANFD_MODEL", ROOT / "data" / "ict_model.npz"))


def read_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Read vertices and polygon faces (0-based vertex indices) from an OBJ."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.asarray(verts, dtype=np.float32), faces


def build_cache(ict_dir: Path = ICT_DIR, out: Path = CACHE) -> Path:
    """Parse the ICT OBJ files once and store everything in a single .npz."""
    cfg = json.loads((ict_dir / "vertex_indices.json").read_text())
    neutral, faces = read_obj(ict_dir / "generic_neutral_mesh.obj")

    def modes(names: list[str]) -> np.ndarray:
        return np.stack([read_obj(ict_dir / f"{n}.obj")[0] - neutral for n in names])

    id_names = [f"identity{i:03d}" for i in range(100)]
    ex_names = cfg["expressions"]
    counts = np.array([len(f) for f in faces], dtype=np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        neutral=neutral,
        id_modes=modes(id_names),
        ex_modes=modes(ex_names),
        ex_names=np.array(ex_names),
        face_counts=counts,
        face_indices=np.concatenate([np.array(f, dtype=np.int32) for f in faces]),
        fitting_verts=np.array(cfg["idx_to_fitting_verts"], dtype=np.int64),
        rigid_verts=np.array(cfg["idx_to_rigid_verts"], dtype=np.int64),
        landmark_verts=np.array(cfg["idx_to_landmark_verts"], dtype=np.int64),
    )
    return out


@dataclass
class FaceModel:
    neutral: np.ndarray  # (V, 3)
    id_modes: np.ndarray  # (100, V, 3)
    ex_modes: np.ndarray  # (53, V, 3)
    ex_names: list[str]
    face_counts: np.ndarray  # (F,) vertices per polygon
    face_indices: np.ndarray  # flat polygon vertex indices
    fitting_verts: np.ndarray
    rigid_verts: np.ndarray
    landmark_verts: np.ndarray

    @classmethod
    def load(cls, path: Path = CACHE) -> FaceModel:
        if not path.exists():
            build_cache(out=path)
        d = np.load(path)
        return cls(**{k: d[k] for k in d.files if k != "ex_names"}, ex_names=list(d["ex_names"]))

    @property
    def n_verts(self) -> int:
        return self.neutral.shape[0]

    @property
    def n_id(self) -> int:
        return self.id_modes.shape[0]

    @property
    def n_ex(self) -> int:
        return self.ex_modes.shape[0]

    def triangles(self) -> np.ndarray:
        """Fan-triangulate the polygon faces -> (T, 3)."""
        tris, start = [], 0
        for c in self.face_counts:
            poly = self.face_indices[start : start + c]
            tris.extend((poly[0], poly[k], poly[k + 1]) for k in range(1, c - 1))
            start += c
        return np.asarray(tris, dtype=np.int64)

    def region_triangles(self, verts: np.ndarray | None = None) -> np.ndarray:
        """Triangles fully inside a vertex subset (default: the fitting region), re-indexed
        into that subset's numbering."""
        verts = self.fitting_verts if verts is None else verts
        remap = -np.ones(self.n_verts, dtype=np.int64)
        remap[verts] = np.arange(len(verts))
        tris = remap[self.triangles()]
        return tris[(tris >= 0).all(1)]

    def evaluate(self, id_w: np.ndarray, ex_w: np.ndarray) -> np.ndarray:
        """Vertices for one identity/expression setting (numpy, CPU)."""
        return (
            self.neutral
            + np.tensordot(id_w, self.id_modes, axes=1)
            + np.tensordot(ex_w, self.ex_modes, axes=1)
        )


class TorchFaceModel(torch.nn.Module):
    """Batched, differentiable version of the linear rig, optionally on a vertex subset."""

    def __init__(self, fm: FaceModel, verts: np.ndarray | None = None):
        super().__init__()
        sel = slice(None) if verts is None else torch.as_tensor(verts)
        self.register_buffer("neutral", torch.from_numpy(fm.neutral)[sel].contiguous())
        self.register_buffer("id_basis", torch.from_numpy(fm.id_modes)[:, sel].flatten(1).T.contiguous())
        self.register_buffer("ex_basis", torch.from_numpy(fm.ex_modes)[:, sel].flatten(1).T.contiguous())

    def forward(self, id_w: torch.Tensor, ex_w: torch.Tensor) -> torch.Tensor:
        """id_w (B, 100), ex_w (B, 53) -> vertices (B, V, 3)."""
        offs = id_w @ self.id_basis.T + ex_w @ self.ex_basis.T
        return self.neutral + offs.view(id_w.shape[0], -1, 3)
