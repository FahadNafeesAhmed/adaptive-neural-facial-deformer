"""Neural solver: raw point cloud -> head pose + identity + 53 blendshape weights.

Architecture (Point-MAE style patch transformer):
  farthest-point-sample G centres -> k-NN patches -> shared mini-PointNet per patch -> tokens
  + learned embedding of each centre -> Transformer encoder -> [CLS] -> heads.
Rotation uses the continuous 6D representation (Zhou et al. 2019) so it trains smoothly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def farthest_point_sample(x: torch.Tensor, g: int) -> torch.Tensor:
    """x (B,N,3) -> indices (B,g). Deterministic start at the point nearest the centroid."""
    b, n, _ = x.shape
    idx = torch.empty(b, g, dtype=torch.long, device=x.device)
    dist = torch.full((b, n), float("inf"), device=x.device)
    cur = (x - x.mean(1, keepdim=True)).norm(dim=-1).argmin(1)
    ar = torch.arange(b, device=x.device)
    for i in range(g):
        idx[:, i] = cur
        dist = torch.minimum(dist, (x - x[ar, cur][:, None]).square().sum(-1))
        cur = dist.argmax(1)
    return idx


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-2)


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, groups: int, k: int):
        super().__init__()
        self.groups, self.k = groups, k
        self.mlp1 = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 256))
        self.mlp2 = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, dim))
        self.pos = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dim))

    def forward(self, x):
        centres = torch.gather(x, 1, farthest_point_sample(x, self.groups)[..., None].expand(-1, -1, 3))
        knn = torch.cdist(centres, x).topk(self.k, largest=False).indices  # (B,G,k)
        patch = torch.gather(x[:, None].expand(-1, self.groups, -1, -1), 2,
                             knn[..., None].expand(-1, -1, -1, 3)) - centres[:, :, None]
        f = self.mlp1(patch)  # (B,G,k,256)
        f = torch.cat([f, f.max(2, keepdim=True).values.expand_as(f)], -1)
        tokens = self.mlp2(f).max(2).values  # (B,G,dim)
        return tokens + self.pos(centres)


class FaceSolverNet(nn.Module):
    def __init__(self, n_id: int = 100, n_ex: int = 53, dim: int = 256, depth: int = 6,
                 heads: int = 8, groups: int = 256, k: int = 32):
        super().__init__()
        self.embed = PatchEmbed(dim, groups, k)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 6 + 3 + n_id + n_ex))
        self.n_id, self.n_ex = n_id, n_ex
        self.register_buffer("centre", torch.zeros(3))

    def forward(self, pts: torch.Tensor) -> dict[str, torch.Tensor]:
        # Normalise translation so the net only sees shape; add the offset back afterwards.
        mean = pts.median(1).values
        x = self.embed(pts - mean[:, None])
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)
        h = self.head(self.norm(self.encoder(x)[:, 0]))
        d6, t, id_w, ex = h.split([6, 3, self.n_id, self.n_ex], -1)
        eye6 = torch.tensor([1.0, 0, 0, 0, 1, 0], device=h.device)
        return {
            "rot": rot6d_to_matrix(d6 + eye6),
            "trans": t + mean - self.centre,
            "id_w": id_w,
            "ex_w": torch.sigmoid(ex),
        }
