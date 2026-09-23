"""Classical optimisation-based rig fitting (the baseline the network must beat), and the
refinement step used by the hybrid solver.

Coarse-to-fine non-rigid ICP, batched on GPU:
  repeat:  closest-vertex correspondences (scan -> model), trimmed to reject outliers
           rigid Procrustes update
           point-to-plane (+ light point-to-point) linear system for identity + expression,
           solved as a box-constrained QP (expression weights in [0, 1]) with FISTA
The schedule starts rigid-only, then heavily regularised shape, then fine shape, because an
unregularised fit from a wrong pose bends identity/expression to absorb the pose error.
"""

from __future__ import annotations

import math

import torch

from anfd.facemodel import TorchFaceModel
from anfd.synth import euler_to_matrix

# (iterations, lam_id, lam_ex); lam_id=None means a rigid-only iteration.
COARSE_TO_FINE = ((8, None, None), (10, 1e-2, 1e-4), (22, 1e-5, 0.0))
REFINE = ((10, 1e-5, 0.0),)
YAW_STARTS = (-40.0, -20.0, 0.0, 20.0, 40.0)


def nearest(a: torch.Tensor, b: torch.Tensor, chunk: int = 8):
    """Nearest point in b for every point in a, batched in chunks to bound GPU memory."""
    ds, js = zip(*(torch.cdist(a[i : i + chunk], b[i : i + chunk]).min(-1)
                   for i in range(0, a.shape[0], chunk)))
    return torch.cat(ds), torch.cat(js)


def vertex_normals(verts: torch.Tensor, tris: torch.Tensor) -> torch.Tensor:
    t = verts[:, tris]
    fn = torch.cross(t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0], dim=-1)
    vn = torch.zeros_like(verts)
    for k in range(3):
        vn.index_add_(1, tris[:, k], fn)
    return torch.nn.functional.normalize(vn, dim=-1)


def procrustes(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor):
    """Weighted rigid fit: find R, t with dst ~ src @ R^T + t. src/dst (B,N,3), w (B,N)."""
    w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
    mu_s, mu_d = (w[..., None] * src).sum(1), (w[..., None] * dst).sum(1)
    cov = ((dst - mu_d[:, None]) * w[..., None]).transpose(1, 2) @ (src - mu_s[:, None])
    u, _, vt = torch.linalg.svd(cov)
    d = torch.det(u @ vt).sign()
    fix = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    rot = u @ fix @ vt
    return rot, mu_d - (mu_s[:, None] @ rot.transpose(1, 2))[:, 0]


def box_qp(h: torch.Tensor, g: torch.Tensor, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
           iters: int = 150) -> torch.Tensor:
    """min 0.5 x^T H x - g^T x  s.t. lo <= x <= hi  (batched FISTA, warm-started)."""
    step = 1.0 / torch.linalg.eigvalsh(h)[:, -1:].clamp_min(1e-9)
    x, y, t = x.clamp(lo, hi), x.clamp(lo, hi), 1.0
    for _ in range(iters):
        x_new = (y - step * ((h @ y[..., None])[..., 0] - g)).clamp(lo, hi)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


class RigFitter:
    def __init__(self, rig: TorchFaceModel, centre: torch.Tensor, tris: torch.Tensor,
                 point_w: float = 0.3, trim: float = 0.9):
        self.rig, self.centre, self.tris = rig, centre, tris
        self.point_w, self.trim = point_w, trim
        self.n_id = rig.id_basis.shape[1]
        self.n_ex = rig.ex_basis.shape[1]
        self.basis = torch.cat([rig.id_basis, rig.ex_basis], 1)  # (3V, K)

    def model(self, id_w, ex_w, rot, trans):
        v = self.rig(id_w, ex_w)
        return (v - self.centre) @ rot.transpose(1, 2) + self.centre + trans[:, None]

    def _correspond(self, pts, verts):
        d, j = nearest(pts, verts)
        k = max(1, int(self.trim * pts.shape[1]))
        thresh = d.kthvalue(k, dim=1, keepdim=True).values
        return j, (d <= thresh).float()

    @torch.no_grad()
    def fit(self, pts, id_w=None, ex_w=None, rot=None, trans=None, schedule=COARSE_TO_FINE):
        """Non-rigid ICP. Use schedule=REFINE to polish an already-good initialisation."""
        b, dev = pts.shape[0], pts.device
        id_w = torch.zeros(b, self.n_id, device=dev) if id_w is None else id_w.clone()
        ex_w = torch.zeros(b, self.n_ex, device=dev) if ex_w is None else ex_w.clone()
        if rot is None:
            rot = torch.eye(3, device=dev).expand(b, 3, 3).clone()
        if trans is None:
            trans = pts.mean(1) - self.rig.neutral.mean(0)
        inf = torch.full((self.n_id,), float("inf"), device=dev)
        lo = torch.cat([-inf, torch.zeros(self.n_ex, device=dev)])
        hi = torch.cat([inf, torch.ones(self.n_ex, device=dev)])
        x = torch.cat([id_w, ex_w], 1)
        ar3 = torch.arange(3, device=dev)

        for n_iter, lam_id, lam_ex in schedule:
            for _ in range(n_iter):
                canon = self.rig(id_w, ex_w)
                posed = (canon - self.centre) @ rot.transpose(1, 2) + self.centre + trans[:, None]
                j, w = self._correspond(pts, posed)
                # Rigid step: unposed model points -> scan
                src = torch.gather(canon, 1, j[..., None].expand(-1, -1, 3)) - self.centre
                rot, t0 = procrustes(src, pts, w)
                trans = t0 - self.centre
                if lam_id is None:
                    continue
                # Shape step in the canonical frame
                local = (pts - self.centre - trans[:, None]) @ rot + self.centre
                nrm = torch.gather(vertex_normals(canon, self.tris), 1, j[..., None].expand(-1, -1, 3))
                a_pt = self.basis[3 * j[..., None] + ar3]  # (B, N, 3, K)
                r_pt = local - self.rig.neutral[j]  # (B, N, 3)
                a_pl = (nrm[..., None] * a_pt).sum(2)  # (B, N, K) point-to-plane rows
                r_pl = (nrm * r_pt).sum(-1)
                wa_pl = a_pl * w[..., None]
                a_pt = a_pt.flatten(1, 2)
                wa_pt = a_pt * w.repeat_interleave(3, 1)[..., None]
                h = wa_pl.transpose(1, 2) @ a_pl + self.point_w**2 * wa_pt.transpose(1, 2) @ a_pt
                g = (wa_pl.transpose(1, 2) @ r_pl[..., None]
                     + self.point_w**2 * wa_pt.transpose(1, 2) @ r_pt.flatten(1)[..., None])[..., 0]
                reg = torch.cat([torch.full((self.n_id,), lam_id), torch.full((self.n_ex,), lam_ex)])
                h = h + torch.diag(reg.to(dev)) * w.sum(1)[:, None, None]
                x = box_qp(h, g, x, lo, hi)
                id_w, ex_w = x[:, : self.n_id], x[:, self.n_id :]
        return {"id_w": id_w, "ex_w": ex_w, "rot": rot, "trans": trans}

    @torch.no_grad()
    def fit_multistart(self, pts, yaws_deg=YAW_STARTS, schedule=COARSE_TO_FINE):
        """Without landmarks the classical fit needs several pose guesses; keep the best
        by the fit's own residual (no ground truth is used to choose)."""
        best, best_err = None, None
        for yaw in yaws_deg:
            y = torch.full((pts.shape[0],), math.radians(yaw), device=pts.device)
            z = torch.zeros_like(y)
            out = self.fit(pts, rot=euler_to_matrix(y, z, z), schedule=schedule)
            err = self.residual(pts, out)
            if best is None:
                best, best_err = out, err
            else:
                better = err < best_err
                best = {k: torch.where(better.view(-1, *[1] * (v.dim() - 1)), out[k], v)
                        for k, v in best.items()}
                best_err = torch.minimum(err, best_err)
        return best

    def residual(self, pts, p):
        """Trimmed mean scan-to-model distance (cm) - the fit's own quality score."""
        d = nearest(pts, self.model(p["id_w"], p["ex_w"], p["rot"], p["trans"]))[0]
        k = max(1, int(self.trim * pts.shape[1]))
        return d.sort(1).values[:, :k].mean(1)
