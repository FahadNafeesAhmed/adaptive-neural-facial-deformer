"""Render the demo scene (truth | raw scan | solved rig) with Blender, headless.

    blender -b --factory-startup -P scripts/blender_render.py -- runs/solver/demo/scene.usda runs/solver/demo/frames

Runs inside Blender's own Python (bpy). Imports the composed USD scene, colours the three
columns, adds labels, frames an orthographic camera and renders every frame to PNG.
`scripts/make_media.py` then turns the frames into an MP4 and a GIF.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

LABELS = ("Ground truth", "Raw scan (input)", "Solved rig (output)")
COLOURS = ((0.78, 0.78, 0.80, 1.0), (0.95, 0.45, 0.18, 1.0), (0.45, 0.70, 0.95, 1.0))


def args() -> tuple[Path, Path, dict]:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage: blender -b -P blender_render.py -- scene.usda out_dir [--frames N] [--width W]")
    opts = {"frames": None, "width": 1600, "fps": 30}
    for i, a in enumerate(argv[2:]):
        if a in ("--frames", "--width", "--fps"):
            opts[a[2:]] = int(argv[2 + i + 1])
    return Path(argv[0]).resolve(), Path(argv[1]).resolve(), opts


def points_as_spheres(ob, radius: float) -> None:
    """Workbench does not draw raw point clouds, so instance a small sphere on every scan point
    with geometry nodes and realise the result as an ordinary mesh."""
    ng = bpy.data.node_groups.new("ScanPointsAsSpheres", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    sphere = ng.nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Radius"].default_value, sphere.inputs["Subdivisions"].default_value = radius, 1
    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    real = ng.nodes.new("GeometryNodeRealizeInstances")
    ng.links.new(gin.outputs[0], inst.inputs["Points"])
    ng.links.new(sphere.outputs["Mesh"], inst.inputs["Instance"])
    ng.links.new(inst.outputs["Instances"], real.inputs["Geometry"])
    ng.links.new(real.outputs["Geometry"], gout.inputs[0])
    ob.modifiers.new("scan_spheres", "NODES").node_group = ng


def world_bounds(objs) -> tuple[Vector, Vector]:
    dg = bpy.context.evaluated_depsgraph_get()
    pts = [ob.evaluated_get(dg).matrix_world @ Vector(c) for ob in objs for c in ob.evaluated_get(dg).bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def label(text: str, x: float, z: float, y: float, colour, size: float) -> None:
    curve = bpy.data.curves.new(f"label_{text}", "FONT")
    curve.body, curve.align_x, curve.size = text, "CENTER", size
    ob = bpy.data.objects.new(f"label_{text}", curve)
    ob.location, ob.rotation_euler, ob.color = (x, y, z), (math.radians(90), 0, 0), colour
    bpy.context.scene.collection.objects.link(ob)


def main() -> None:
    scene_path, out_dir, opts = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(scene_path))
    scene = bpy.context.scene
    scene.render.fps = opts["fps"]  # the importer keeps Blender's 24 fps default
    if opts["frames"]:
        scene.frame_end = scene.frame_start + opts["frames"] - 1
    scene.frame_set(scene.frame_start)

    heads = sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.matrix_world.translation.x)
    scans = [o for o in bpy.data.objects if o.type == "POINTCLOUD"]
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.hide_render = True
    if len(heads) != 2 or len(scans) != 1:
        raise SystemExit(f"expected 2 heads and 1 scan, found {len(heads)} and {len(scans)}")
    # The importer puts a 0.01 cm-to-m scale on a parent, so give the radius in the object's own units.
    points_as_spheres(scans[0], 0.0011 / scans[0].matrix_world.to_scale().x)
    columns = [heads[0], scans[0], heads[1]]  # sorted left to right: truth, scan, solved
    for ob, colour in zip(columns, COLOURS):
        ob.color = colour

    lo, hi = world_bounds(columns)
    centre, size = (lo + hi) / 2, hi - lo
    for ob, text, colour in zip(columns, LABELS, COLOURS):
        col_lo, col_hi = world_bounds([ob])
        label(text, (col_lo.x + col_hi.x) / 2, hi.z + 0.03, centre.y, colour, 0.028)

    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.type, cam.data.ortho_scale = "ORTHO", size.x * 1.12
    cam.location = (centre.x, lo.y - 1.5, centre.z + 0.015)
    cam.rotation_euler = (math.radians(90), 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light, shading.color_type = "STUDIO", "OBJECT"
    shading.show_cavity, shading.cavity_type = True, "BOTH"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.035, 0.037, 0.045)
    scene.render.resolution_x = opts["width"]
    # Video codecs want dimensions in multiples of 16.
    scene.render.resolution_y = int(opts["width"] * (size.z + 0.08) / (size.x * 1.12)) // 16 * 16
    scene.render.image_settings.file_format = "PNG"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_dir / "frame_")
    bpy.ops.render.render(animation=True)


main()
