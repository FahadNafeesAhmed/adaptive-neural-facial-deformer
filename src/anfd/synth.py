"""On-the-fly synthetic "raw capture" generator (GPU).

Each sample: random identity, plausible sparse expression, random head pose, then the face
surface is turned into what a depth sensor / photogrammetry rig actually hands you:
an unordered point cloud with no vertex correspondence, noise, holes, outliers, and
self-occlusion. Ground truth (weights, pose, clean vertices) is kept for supervision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from anfd.facemodel import FaceModel, TorchFaceModel


@dataclass
class CaptureConfig:
    n_points: int = 4096
    noise_cm: tuple[float, float] = (0.0, 0.1)  # per-sample Gaussian sigma range (0-1 mm)
    max_holes: int = 3
    hole_radius_cm: tuple[float, float] = (0.8, 2.5)
    outlier_frac: tuple[float, float] = (0.0, 0.02)
    yaw_deg: float = 35.0
    pitch_deg: float = 20.0
    roll_deg: float = 10.0
    trans_cm: float = 2.0
    occlusion_prob: float = 0.7  # single-view capture (back-facing surface missing) vs multi-view

    @classmethod
    def clean(cls) -> CaptureConfig:
        return cls(noise_cm=(0.0, 0.0), max_holes=0, outlier_frac=(0.0, 0.0), occlusion_prob=0.0)


# Pairs of (L, R) expression indices are filled in from names at runtime.
def _lr_pairs(names: list[str]) -> list[tuple[int, int]]:
    return [(i, names.index(n[:-2] + "_R")) for i, n in enumerate(names) if n.endswith("_L")]


def euler_to_matrix(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """(B,) radians -> (B, 3, 3), R = Ry(yaw) @ Rx(pitch) @ Rz(roll) for a Y-up head."""
    cy, sy, cp, sp, cr, sr = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
    o, z = torch.ones_like(yaw), torch.zeros_like(yaw)
    ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    rx = torch.stack([o, z, z, z, cp, -sp, z, sp, cp], -1).view(-1, 3, 3)
    rz = torch.stack([cr, -sr, z, sr, cr, z, z, z, o], -1).view(-1, 3, 3)
    return ry @ rx @ rz


class CaptureSynth(torch.nn.Module):
    def __init__(self, fm: FaceModel, cfg: CaptureConfig | None = None):
        super().__init__()
        self.cfg = cfg or CaptureConfig()
        self.rig = TorchFaceModel(fm, fm.fitting_verts)
        self.register_buffer("tris", torch.from_numpy(fm.region_triangles()))
        self.lr = _lr_pairs(fm.ex_names)
        self.n_id, self.n_ex = fm.n_id, fm.n_ex
        # Head-centric origin: centre of the neutral face region.
        self.register_buffer("centre", self.rig.neutral.mean(0))

    # ---------- parameter sampling ----------
    def sample_expression(self, b: int, g: torch.Generator | None = None) -> torch.Tensor:
        dev = self.centre.device
        active = torch.rand(b, self.n_ex, device=dev, generator=g) < 0.18
        u = torch.rand(b, self.n_ex, device=dev, generator=g)
        vals = (1 - (1 - u) ** (1 / 1.6)) ** (1 / 1.2)  # Kumaraswamy(1.2, 1.6): Beta-shaped, seedable
        w = vals * active
        # Faces are mostly symmetric: mirror L->R for ~60% of samples.
        sym = torch.rand(b, 1, device=dev, generator=g) < 0.6
        for li, ri in self.lr:
            jitter = 0.15 * (torch.rand(b, device=dev, generator=g) - 0.5)
            w[:, ri] = torch.where(sym[:, 0], (w[:, li] + jitter).clamp(0, 1), w[:, ri])
        # A share of fully neutral faces keeps the solver honest near zero.
        w[torch.rand(b, device=dev, generator=g) < 0.05] = 0
        return w

    def sample_pose(self, b: int, g: torch.Generator | None = None):
        c, dev = self.cfg, self.centre.device

        def u(scale):
            return (torch.rand(b, device=dev, generator=g) * 2 - 1) * scale

        rot = euler_to_matrix(
            u(math.radians(c.yaw_deg)), u(math.radians(c.pitch_deg)), u(math.radians(c.roll_deg))
        )
        trans = torch.stack([u(c.trans_cm) for _ in range(3)], -1)
        return rot, trans

    def pose(self, verts: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        return (verts - self.centre) @ rot.transpose(1, 2) + self.centre + trans[:, None]

    # ---------- capture corruption ----------
    def scan(self, verts: torch.Tensor, g: torch.Generator | None = None) -> torch.Tensor:
        """Posed face-region vertices (B, V, 3) -> corrupted point cloud (B, N, 3)."""
        c, (b, _, _), dev = self.cfg, verts.shape, verts.device
        tri = verts[:, self.tris]  # (B, T, 3, 3)
        e1, e2 = tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]
        normal = torch.cross(e1, e2, dim=-1)
        area = normal.norm(dim=-1)
        prob = area.clone()
        if c.occlusion_prob > 0:  # camera looks down -z; drop back-facing surface
            facing = normal[..., 2] / area.clamp_min(1e-9)
            occl = torch.rand(b, 1, device=dev, generator=g) < c.occlusion_prob
            prob = torch.where(occl, prob * (facing > 0.05), prob)
        # Oversample, then carve holes and subsample to exactly n_points.
        m = c.n_points * 2
        idx = torch.multinomial(prob + 1e-12, m, replacement=True, generator=g)
        uv = torch.rand(b, m, 2, device=dev, generator=g)
        flip = uv.sum(-1, keepdim=True) > 1
        uv = torch.where(flip, 1 - uv, uv)
        t = torch.gather(tri, 1, idx[..., None, None].expand(-1, -1, 3, 3))
        pts = t[:, :, 0] + uv[..., :1] * (t[:, :, 1] - t[:, :, 0]) + uv[..., 1:] * (t[:, :, 2] - t[:, :, 0])

        keep = torch.ones(b, m, dtype=torch.bool, device=dev)
        for _ in range(c.max_holes):
            use = torch.rand(b, device=dev, generator=g) < 0.6
            centre = torch.gather(pts, 1, torch.randint(0, m, (b, 1, 1), device=dev, generator=g).expand(-1, -1, 3))
            r = c.hole_radius_cm[0] + torch.rand(b, 1, device=dev, generator=g) * (c.hole_radius_cm[1] - c.hole_radius_cm[0])
            keep &= ~(use[:, None] & ((pts - centre).norm(dim=-1) < r))
        # Pick n_points survivors per sample (random order = no correspondence).
        score = torch.rand(b, m, device=dev, generator=g) + keep.float() * 2
        order = score.argsort(dim=1, descending=True)[:, : c.n_points]
        pts = torch.gather(pts, 1, order[..., None].expand(-1, -1, 3))

        sigma = c.noise_cm[0] + torch.rand(b, 1, 1, device=dev, generator=g) * (c.noise_cm[1] - c.noise_cm[0])
        pts = pts + sigma * torch.randn(pts.shape, device=dev, generator=g)

        frac = c.outlier_frac[0] + torch.rand(b, device=dev, generator=g) * (c.outlier_frac[1] - c.outlier_frac[0])
        lo, hi = verts.amin(1, keepdim=True) - 2, verts.amax(1, keepdim=True) + 2
        junk = lo + torch.rand(pts.shape, device=dev, generator=g) * (hi - lo)
        is_out = torch.rand(b, c.n_points, device=dev, generator=g) < frac[:, None]
        return torch.where(is_out[..., None], junk, pts)

    @torch.no_grad()
    def forward(self, b: int, g: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        dev = self.centre.device
        id_w = torch.randn(b, self.n_id, device=dev, generator=g)
        ex_w = self.sample_expression(b, g)
        rot, trans = self.sample_pose(b, g)
        verts = self.pose(self.rig(id_w, ex_w), rot, trans)
        return {
            "points": self.scan(verts, g),
            "id_w": id_w,
            "ex_w": ex_w,
            "rot": rot,
            "trans": trans,
            "verts": verts,
        }


def cached_eval_set(synth: CaptureSynth, n: int, seed: int, path: Path) -> dict[str, torch.Tensor]:
    """Fixed, seeded evaluation captures cached on disk so every method sees identical data."""
    dev = synth.centre.device
    if path.exists():
        return {k: v.to(dev) for k, v in torch.load(path).items()}
    data = synth(n, torch.Generator(device=dev).manual_seed(seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in data.items()}, path)
    return data


def yaw_of(rot: torch.Tensor) -> torch.Tensor:
    """Yaw (degrees) of R = Ry @ Rx @ Rz."""
    return torch.rad2deg(torch.atan2(rot[:, 0, 2], rot[:, 2, 2]))
