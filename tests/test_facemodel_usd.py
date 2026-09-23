import numpy as np
import pytest
import torch

from anfd.facemodel import FaceModel, TorchFaceModel
from anfd.synth import euler_to_matrix
from anfd.usd_io import evaluate_usd_rig, export_rig


@pytest.fixture(scope="module")
def fm():
    return FaceModel.load()


def test_model_shapes(fm):
    assert fm.neutral.shape == (26719, 3)
    assert fm.id_modes.shape == (100, 26719, 3)
    assert fm.ex_modes.shape == (53, 26719, 3)
    assert fm.landmark_verts.shape == (68,)
    assert fm.face_counts.sum() == fm.face_indices.size
    tris = fm.region_triangles()
    assert tris.min() >= 0 and tris.max() < len(fm.fitting_verts)


def test_torch_matches_numpy(fm):
    rng = np.random.default_rng(0)
    id_w, ex_w = rng.normal(size=100).astype(np.float32), rng.uniform(size=53).astype(np.float32)
    tm = TorchFaceModel(fm, fm.fitting_verts)
    out = tm(torch.from_numpy(id_w)[None], torch.from_numpy(ex_w)[None])[0].numpy()
    ref = fm.evaluate(id_w, ex_w)[fm.fitting_verts]
    assert np.abs(out - ref).max() < 1e-3


def test_usd_roundtrip_with_head_pose(fm, tmp_path):
    rng = np.random.default_rng(1)
    id_w = rng.normal(size=100).astype(np.float32)
    anim = rng.uniform(size=(3, 53)).astype(np.float32)
    rot = euler_to_matrix(*torch.tensor([[0.3, -0.2, 0.0], [0.1, 0.1, 0.05], [0.0, 0.0, 0.0]]).T).numpy()
    trans = rng.normal(size=(3, 3)).astype(np.float32)
    centre = fm.neutral[fm.fitting_verts].mean(0)
    path = export_rig(tmp_path / "face.usda", fm, id_w, anim, head_pose=(rot, trans, centre))
    for t in range(3):
        canon = fm.evaluate(id_w, anim[t])
        expect = (canon - centre) @ rot[t].T + centre + trans[t]
        assert np.abs(evaluate_usd_rig(path, t) - expect).max() < 1e-3
