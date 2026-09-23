import numpy as np
import pytest
import torch

from anfd.facemodel import FaceModel
from anfd.fitting import RigFitter, procrustes
from anfd.metrics import score
from anfd.model import FaceSolverNet
from anfd.solve import read_points
from anfd.synth import CaptureConfig, CaptureSynth, euler_to_matrix, yaw_of

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def fm():
    return FaceModel.load()


def test_synth_shapes_and_determinism(fm):
    synth = CaptureSynth(fm).to(DEV)
    a = synth(4, torch.Generator(device=DEV).manual_seed(3))
    b = synth(4, torch.Generator(device=DEV).manual_seed(3))
    assert a["points"].shape == (4, 4096, 3)
    assert torch.isfinite(a["points"]).all()
    assert ((a["ex_w"] >= 0) & (a["ex_w"] <= 1)).all()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_yaw_roundtrip():
    yaw = torch.tensor([-0.5, 0.0, 0.4])
    rot = euler_to_matrix(yaw, torch.full_like(yaw, 0.2), torch.full_like(yaw, -0.1))
    assert torch.allclose(yaw_of(rot), torch.rad2deg(yaw), atol=1e-4)


def test_procrustes_recovers_rigid_motion():
    g = torch.Generator().manual_seed(0)
    src = torch.randn(2, 500, 3, generator=g)
    rot = euler_to_matrix(torch.tensor([0.5, -1.0]), torch.tensor([0.2, 0.3]), torch.tensor([0.1, 0.0]))
    t = torch.randn(2, 3, generator=g)
    dst = src @ rot.transpose(1, 2) + t[:, None]
    r_hat, t_hat = procrustes(src, dst, torch.ones(2, 500))
    assert torch.allclose(r_hat, rot, atol=1e-4) and torch.allclose(t_hat, t, atol=1e-4)


def test_icp_recovers_clean_scan_from_true_pose(fm):
    synth = CaptureSynth(fm, CaptureConfig.clean()).to(DEV)
    gt = synth(8, torch.Generator(device=DEV).manual_seed(5))
    fitter = RigFitter(synth.rig, synth.centre, synth.tris)
    pred = fitter.fit(gt["points"], rot=gt["rot"], trans=gt["trans"])
    # Blendshape fits have a few hard, ambiguous cases, so check the typical face.
    assert score(fitter, pred, gt)["surface_mm"].median() < 2.0


def test_network_output_contract():
    net = FaceSolverNet(groups=32, k=16, depth=2).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 1024, 3) * 5)
    rot = out["rot"]
    assert out["id_w"].shape == (2, 100) and out["ex_w"].shape == (2, 53)
    assert ((out["ex_w"] >= 0) & (out["ex_w"] <= 1)).all()
    assert torch.allclose(rot @ rot.transpose(1, 2), torch.eye(3).expand(2, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(rot), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("binary", [False, True])
def test_ply_reader(tmp_path, binary):
    pts = np.random.default_rng(0).normal(size=(300, 3)).astype(np.float32)
    path = tmp_path / "scan.ply"
    header = (f"ply\nformat {'binary_little_endian' if binary else 'ascii'} 1.0\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        if binary:
            rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1")])
            rec["x"], rec["y"], rec["z"] = pts.T
            f.write(rec.tobytes())
        else:
            f.write("".join(f"{x} {y} {z} 255\n" for x, y, z in pts).encode())
    assert np.allclose(read_points(path), pts, atol=1e-5)
