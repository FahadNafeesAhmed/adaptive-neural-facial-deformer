"""Write faces to USD as a proper UsdSkel blendshape rig, so DCCs (Blender, Maya, Houdini)
get editable shape targets + animation curves rather than baked vertex caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

from anfd.facemodel import FaceModel

SPARSE_EPS = 1e-6


def _new_stage(out: Path, fps: float, n_frames: int) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # ICT units are centimetres
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(n_frames - 1, 0))
    return stage


def _add_mesh(stage: Usd.Stage, path: str, fm: FaceModel, points: np.ndarray) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(fm.face_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(fm.face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def pose_matrix(rot: np.ndarray, trans: np.ndarray, centre: np.ndarray) -> Gf.Matrix4d:
    """Our head pose  p' = (p - c) R^T + c + t  as a USD (row-vector) 4x4 matrix."""
    m = np.eye(4)
    m[:3, :3] = rot.T
    m[3, :3] = centre - centre @ rot.T + trans
    return Gf.Matrix4d(m.tolist())


def export_rig(
    out: Path,
    fm: FaceModel,
    id_w: np.ndarray,
    ex_weights: np.ndarray | None = None,
    fps: float = 30.0,
    head_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Path:
    """SkelRoot with the identity baked into the base mesh, the 53 expression shapes as
    UsdSkelBlendShapes, an optional weight animation (T, 53), and an optional animated head
    transform `head_pose = (rot (T,3,3), trans (T,3), centre (3,))`."""
    ex_weights = None if ex_weights is None else np.atleast_2d(ex_weights)
    stage = _new_stage(out, fps, 0 if ex_weights is None else len(ex_weights))
    root = UsdSkel.Root.Define(stage, "/Face")
    stage.SetDefaultPrim(root.GetPrim())

    base = fm.neutral + np.tensordot(id_w, fm.id_modes, axes=1)
    mesh = _add_mesh(stage, "/Face/Head", fm, base)

    # One root joint: UsdSkel needs a skeleton to drive blendshapes, even without skinning.
    skel = UsdSkel.Skeleton.Define(stage, "/Face/Skel")
    skel.CreateJointsAttr(["root"])
    ident = Gf.Matrix4d(1.0)
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray([ident]))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray([ident]))

    names = [str(n) for n in fm.ex_names]
    targets = []
    for name, mode in zip(names, fm.ex_modes):
        idx = np.nonzero(np.abs(mode).max(axis=1) > SPARSE_EPS)[0]
        bs = UsdSkel.BlendShape.Define(stage, f"/Face/Head/{name}")
        bs.CreateOffsetsAttr(Vt.Vec3fArray.FromNumpy(mode[idx].astype(np.float32)))
        bs.CreatePointIndicesAttr(Vt.IntArray.FromNumpy(idx.astype(np.int32)))
        targets.append(bs.GetPath())

    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skel.GetPath()])
    binding.CreateBlendShapesAttr(names)
    binding.CreateBlendShapeTargetsRel().SetTargets(targets)
    binding.CreateJointIndicesPrimvar(constant=True, elementSize=1).Set(Vt.IntArray([0]))
    binding.CreateJointWeightsPrimvar(constant=True, elementSize=1).Set(Vt.FloatArray([1.0]))
    binding.CreateGeomBindTransformAttr(ident)

    if ex_weights is not None:
        anim = UsdSkel.Animation.Define(stage, "/Face/Skel/Anim")
        anim.CreateJointsAttr(["root"])
        anim.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f(0)]))
        anim.CreateRotationsAttr(Vt.QuatfArray([Gf.Quatf(1)]))
        anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1)]))
        anim.CreateBlendShapesAttr(names)
        attr = anim.CreateBlendShapeWeightsAttr()
        for t, w in enumerate(ex_weights):
            attr.Set(Vt.FloatArray.FromNumpy(w.astype(np.float32)), Usd.TimeCode(t))
        UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets(
            [anim.GetPath()]
        )

    if head_pose is not None:
        rot, trans, centre = head_pose
        op = UsdGeom.Xformable(root.GetPrim()).AddTransformOp()
        for t, (r, tr) in enumerate(zip(rot, trans)):
            op.Set(pose_matrix(r, tr, centre), Usd.TimeCode(t))

    stage.GetRootLayer().Save()
    return out


def export_point_clouds(out: Path, clouds: np.ndarray, fps: float = 30.0, width_cm: float = 0.15) -> Path:
    """Animated raw scan points (T, N, 3) as UsdGeomPoints."""
    stage = _new_stage(out, fps, len(clouds))
    pts = UsdGeom.Points.Define(stage, "/Scan")
    stage.SetDefaultPrim(pts.GetPrim())
    attr = pts.CreatePointsAttr()
    for t, c in enumerate(clouds):
        attr.Set(Vt.Vec3fArray.FromNumpy(c.astype(np.float32)), Usd.TimeCode(t))
    pts.CreateWidthsAttr(Vt.FloatArray([width_cm]))
    pts.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    pts.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.35, 0.2)]))
    stage.GetRootLayer().Save()
    return out


def export_scene(out: Path, layers: dict[str, tuple[Path, float]], fps: float, n_frames: int) -> Path:
    """Compose several exported files side by side: {name: (file, x_offset_cm)}."""
    stage = _new_stage(out, fps, n_frames)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for name, (path, x) in layers.items():
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}")
        prim.GetPrim().GetReferences().AddReference(Path(path).name)
        prim.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
    stage.GetRootLayer().Save()
    return out


def evaluate_usd_rig(path: Path, time: float) -> np.ndarray:
    """Independently re-evaluate a blendshape rig (+ head transform) from the USD file."""
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Face/Head"))
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    binding = UsdSkel.BindingAPI(mesh.GetPrim())
    names = list(binding.GetBlendShapesAttr().Get())
    targets = [str(t) for t in binding.GetBlendShapeTargetsRel().GetTargets()]
    anim = UsdSkel.Animation(stage.GetPrimAtPath("/Face/Skel/Anim"))
    anim_names = list(anim.GetBlendShapesAttr().Get())
    weights = np.array(anim.GetBlendShapeWeightsAttr().Get(Usd.TimeCode(time)))
    for name, tgt in zip(names, targets):
        bs = UsdSkel.BlendShape(stage.GetPrimAtPath(Sdf.Path(tgt)))
        idx = np.array(bs.GetPointIndicesAttr().Get())
        pts[idx] += weights[anim_names.index(name)] * np.array(bs.GetOffsetsAttr().Get())
    xf = UsdGeom.Xformable(stage.GetPrimAtPath("/Face")).ComputeLocalToWorldTransform(Usd.TimeCode(time))
    m = np.array(xf)
    return pts @ m[:3, :3] + m[3, :3]
