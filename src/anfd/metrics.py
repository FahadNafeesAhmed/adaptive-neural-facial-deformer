"""Metrics that matter to a rigging team, all in millimetres on the face region."""

from __future__ import annotations

import torch

from anfd.fitting import RigFitter


@torch.no_grad()
def score(fitter: RigFitter, pred: dict, gt: dict) -> dict[str, torch.Tensor]:
    """Per-sample metrics.

    surface_mm    : posed mesh error (what you'd see overlaid on the scan)
    shape_mm      : pose-free mesh error (identity + expression), i.e. rig-space accuracy
    expr_mm       : error caused by the expression weights alone, identity held at truth -
                    this is what an animator inherits when they open the solved rig
    weight_mae    : mean absolute blendshape weight error (0-1 scale)
    rot_deg       : head rotation error
    """
    rig = fitter.rig
    posed = fitter.model(pred["id_w"], pred["ex_w"], pred["rot"], pred["trans"])
    canon_p, canon_g = rig(pred["id_w"], pred["ex_w"]), rig(gt["id_w"], gt["ex_w"])
    expr_p = rig(gt["id_w"], pred["ex_w"])
    rel = pred["rot"].transpose(1, 2) @ gt["rot"]
    cos = ((rel.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
    return {
        "surface_mm": 10 * (posed - gt["verts"]).norm(dim=-1).mean(1),
        "shape_mm": 10 * (canon_p - canon_g).norm(dim=-1).mean(1),
        "expr_mm": 10 * (expr_p - canon_g).norm(dim=-1).mean(1),
        "weight_mae": (pred["ex_w"] - gt["ex_w"]).abs().mean(1),
        "rot_deg": torch.rad2deg(torch.acos(cos)),
    }
