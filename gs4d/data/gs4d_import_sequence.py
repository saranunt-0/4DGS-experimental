"""Load an animated Gaussian-splat PLY sequence into Blender (4.5 LTS / 5.x).

No add-on needed: a Geometry Nodes modifier uses the native *Import PLY* node to
load ``<name>_####.ply`` for the current frame (``####`` = zero-padded index), instances an oriented,
scaled ellipsoid per splat and shades it with an emissive Gaussian-falloff
material.  Nothing runs in Python at playback/render time, so it works on
render farms and in background renders.

For hero-quality splat rendering you can instead use the KIRI Engine
"3DGS Render" add-on on the individual PLY frames; this script is the
dependency-free path.

Usage
-----
Inside Blender: Scripting workspace -> Open this file -> set ``SEQ_DIR`` below
(leave empty to use ``./ply_sequence`` next to this script) -> Run Script.

Command line::

    blender -P gs4d_import_sequence.py -- --dir /path/to/ply_sequence
    blender -b -P gs4d_import_sequence.py -- --dir ./ply_sequence --render //render/frame_#### --engine CYCLES
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import bpy

SEQ_DIR = ""        # folder with <name>_0001.ply, <name>_0002.ply, ...
SPLAT_SIZE = 2.0    # ellipsoid radius in standard deviations (2 = 95% of the Gaussian)
DETAIL = 1          # ico-sphere subdivisions per splat (0 = fastest)
MIN_OPACITY = 0.02  # cull nearly transparent splats
SH_C0 = 0.28209479177387814


def find_sequence(seq_dir: str):
    files = sorted(glob.glob(os.path.join(seq_dir, "*.ply")))
    pat = re.compile(r"^(.*?)(\d+)\.ply$")
    groups: dict[str, list[int]] = {}
    for f in files:
        m = pat.match(os.path.basename(f))
        if m:
            groups.setdefault(m.group(1), []).append(int(m.group(2)))
    if not groups:
        raise FileNotFoundError(f"no numbered .ply files in {seq_dir}")
    prefix = max(groups, key=lambda k: len(groups[k]))
    idx = sorted(groups[prefix])
    return prefix, idx[0], len(idx)


def _socket(ng, name, in_out, stype, default=None, min_value=None):
    s = ng.interface.new_socket(name, in_out=in_out, socket_type=stype)
    if default is not None and hasattr(s, "default_value"):
        s.default_value = default
    if min_value is not None and hasattr(s, "min_value"):
        s.min_value = min_value
    return s


def _math(nodes, op, a=None, b=None, loc=(0, 0)):
    n = nodes.new("ShaderNodeMath")
    n.operation = op
    n.location = loc
    if a is not None and not hasattr(a, "is_output"):
        n.inputs[0].default_value = a
    if b is not None and not hasattr(b, "is_output"):
        n.inputs[1].default_value = b
    return n


def build_node_group(name: str = "GS4D_PLY_Sequence"):
    if name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[name])
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    _socket(ng, "Geometry", "INPUT", "NodeSocketGeometry")
    _socket(ng, "Path Pattern", "INPUT", "NodeSocketString", "")
    _socket(ng, "First Index", "INPUT", "NodeSocketInt", 1)
    _socket(ng, "Frame Count", "INPUT", "NodeSocketInt", 1, 1)
    _socket(ng, "Start Frame", "INPUT", "NodeSocketInt", 1)
    _socket(ng, "Splat Size", "INPUT", "NodeSocketFloat", SPLAT_SIZE, 0.0)
    _socket(ng, "Detail", "INPUT", "NodeSocketInt", DETAIL, 0)
    _socket(ng, "Min Opacity", "INPUT", "NodeSocketFloat", MIN_OPACITY, 0.0)
    _socket(ng, "Material", "INPUT", "NodeSocketMaterial")
    _socket(ng, "Geometry", "OUTPUT", "NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gin = N.new("NodeGroupInput")
    gin.location = (-1800, 0)
    gout = N.new("NodeGroupOutput")
    gout.location = (900, 0)

    # ---- file name for the current frame: prefix + zero-padded(index) + ".ply" (loops)
    t = N.new("GeometryNodeInputSceneTime")
    t.location = (-1600, 400)
    sub = _math(N, "SUBTRACT", loc=(-1400, 400))
    L.new(t.outputs["Frame"], sub.inputs[0])
    L.new(gin.outputs["Start Frame"], sub.inputs[1])
    mod = _math(N, "FLOORED_MODULO", loc=(-1250, 400))
    L.new(sub.outputs[0], mod.inputs[0])
    L.new(gin.outputs["Frame Count"], mod.inputs[1])
    add = _math(N, "ADD", loc=(-1100, 400))
    L.new(mod.outputs[0], add.inputs[0])
    L.new(gin.outputs["First Index"], add.inputs[1])
    pad = _math(N, "ADD", b=10000.0, loc=(-950, 400))
    L.new(add.outputs[0], pad.inputs[0])
    v2s = N.new("FunctionNodeValueToString")
    v2s.location = (-800, 400)
    v2s.inputs["Decimals"].default_value = 0
    L.new(pad.outputs[0], v2s.inputs["Value"])
    sl = N.new("FunctionNodeSliceString")
    sl.location = (-650, 400)
    sl.inputs["Position"].default_value = 1
    sl.inputs["Length"].default_value = 4
    L.new(v2s.outputs["String"], sl.inputs["String"])
    # "<dir>/<name>_####.ply" -> replace "####" with the padded index (no multi-input ordering issues)
    rep = N.new("FunctionNodeReplaceString")
    rep.location = (-450, 400)
    L.new(gin.outputs["Path Pattern"], rep.inputs["String"])
    rep.inputs["Find"].default_value = "####"
    L.new(sl.outputs["String"], rep.inputs["Replace"])
    imp = N.new("GeometryNodeImportPLY")
    imp.location = (-250, 400)
    L.new(rep.outputs["String"], imp.inputs["Path"])

    def attr(name, loc):
        a = N.new("GeometryNodeInputNamedAttribute")
        a.data_type = "FLOAT"
        a.inputs["Name"].default_value = name
        a.location = loc
        return a.outputs["Attribute"]

    # ---- opacity = sigmoid(logit); cull faint splats
    op = attr("opacity", (-600, -50))
    neg = _math(N, "MULTIPLY", b=-1.0, loc=(-450, -50))
    L.new(op, neg.inputs[0])
    ex = _math(N, "EXPONENT", loc=(-300, -50))
    L.new(neg.outputs[0], ex.inputs[0])
    one = _math(N, "ADD", b=1.0, loc=(-150, -50))
    L.new(ex.outputs[0], one.inputs[0])
    sig = _math(N, "DIVIDE", a=1.0, loc=(0, -50))
    L.new(one.outputs[0], sig.inputs[1])
    faint = _math(N, "LESS_THAN", loc=(150, -50))
    L.new(sig.outputs[0], faint.inputs[0])
    L.new(gin.outputs["Min Opacity"], faint.inputs[1])
    dele = N.new("GeometryNodeDeleteGeometry")
    dele.location = (0, 400)
    dele.domain = "POINT"
    L.new(imp.outputs[0], dele.inputs["Geometry"])
    L.new(faint.outputs[0], dele.inputs["Selection"])

    # ---- colour: sRGB = 0.5 + C0 * f_dc  ->  scene-linear (approx. gamma 2.2)
    comb = N.new("ShaderNodeCombineXYZ")
    comb.location = (-150, -300)
    for i, axis in enumerate("XYZ"):
        dc = attr(f"f_dc_{i}", (-900, -250 - 120 * i))
        m1 = _math(N, "MULTIPLY_ADD", loc=(-700, -250 - 120 * i))
        L.new(dc, m1.inputs[0])
        m1.inputs[1].default_value = SH_C0
        m1.inputs[2].default_value = 0.5
        cl = _math(N, "MAXIMUM", b=0.0, loc=(-550, -250 - 120 * i))
        L.new(m1.outputs[0], cl.inputs[0])
        pw = _math(N, "POWER", b=2.2, loc=(-400, -250 - 120 * i))
        L.new(cl.outputs[0], pw.inputs[0])
        L.new(pw.outputs[0], comb.inputs[axis])
    st_col = N.new("GeometryNodeStoreNamedAttribute")
    st_col.location = (200, 400)
    st_col.data_type = "FLOAT_VECTOR"
    st_col.domain = "POINT"
    st_col.inputs["Name"].default_value = "gs_color"
    L.new(dele.outputs[0], st_col.inputs["Geometry"])
    L.new(comb.outputs[0], st_col.inputs["Value"])
    st_a = N.new("GeometryNodeStoreNamedAttribute")
    st_a.location = (350, 400)
    st_a.data_type = "FLOAT"
    st_a.domain = "POINT"
    st_a.inputs["Name"].default_value = "gs_alpha"
    L.new(st_col.outputs[0], st_a.inputs["Geometry"])
    L.new(sig.outputs[0], st_a.inputs["Value"])

    # ---- orientation (w, x, y, z) and anisotropic scale exp(scale_i) * size
    q2r = N.new("FunctionNodeQuaternionToRotation")
    q2r.location = (200, -600)
    for i, sock in enumerate(["W", "X", "Y", "Z"]):
        L.new(attr(f"rot_{i}", (0, -550 - 100 * i)), q2r.inputs[sock])
    sc = N.new("ShaderNodeCombineXYZ")
    sc.location = (200, -950)
    for i, axis in enumerate("XYZ"):
        e = _math(N, "EXPONENT", loc=(-100, -900 - 120 * i))
        L.new(attr(f"scale_{i}", (-300, -900 - 120 * i)), e.inputs[0])
        m = _math(N, "MULTIPLY", loc=(50, -900 - 120 * i))
        L.new(e.outputs[0], m.inputs[0])
        L.new(gin.outputs["Splat Size"], m.inputs[1])
        L.new(m.outputs[0], sc.inputs[axis])

    ico = N.new("GeometryNodeMeshIcoSphere")
    ico.location = (350, -200)
    ico.inputs["Radius"].default_value = 1.0
    L.new(gin.outputs["Detail"], ico.inputs["Subdivisions"])
    inst = N.new("GeometryNodeInstanceOnPoints")
    inst.location = (550, 300)
    L.new(st_a.outputs[0], inst.inputs["Points"])
    L.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    L.new(q2r.outputs["Rotation"], inst.inputs["Rotation"])
    L.new(sc.outputs[0], inst.inputs["Scale"])
    setm = N.new("GeometryNodeSetMaterial")
    setm.location = (720, 300)
    L.new(inst.outputs[0], setm.inputs["Geometry"])
    L.new(gin.outputs["Material"], setm.inputs["Material"])
    L.new(setm.outputs[0], gout.inputs["Geometry"])
    return ng


def build_material(name: str = "GS4D_Splat", splat_size: float = SPLAT_SIZE):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):  # node trees are always on in 5.x (flag deprecated)
        mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = True
    N, L = mat.node_tree.nodes, mat.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    col = N.new("ShaderNodeAttribute")
    col.attribute_type = "INSTANCER"
    col.attribute_name = "gs_color"
    col.location = (-400, 200)
    alp = N.new("ShaderNodeAttribute")
    alp.attribute_type = "INSTANCER"
    alp.attribute_name = "gs_alpha"
    alp.location = (-400, -150)
    emis = N.new("ShaderNodeEmission")
    emis.location = (200, 150)
    L.new(col.outputs["Vector"], emis.inputs["Color"])
    transp = N.new("ShaderNodeBsdfTransparent")
    transp.location = (200, 0)
    # Gaussian falloff across the ellipsoid: exp(-0.5 * k^2 * (1 - (N.I)^2))
    geo = N.new("ShaderNodeNewGeometry")
    geo.location = (-600, -400)
    dot = N.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-400, -400)
    L.new(geo.outputs["Normal"], dot.inputs[0])
    L.new(geo.outputs["Incoming"], dot.inputs[1])
    sq = N.new("ShaderNodeMath")
    sq.operation = "MULTIPLY"
    sq.location = (-200, -400)
    L.new(dot.outputs["Value"], sq.inputs[0])
    L.new(dot.outputs["Value"], sq.inputs[1])
    om = N.new("ShaderNodeMath")
    om.operation = "SUBTRACT"
    om.inputs[0].default_value = 1.0
    om.location = (-50, -400)
    L.new(sq.outputs[0], om.inputs[1])
    k = N.new("ShaderNodeMath")
    k.operation = "MULTIPLY"
    k.inputs[1].default_value = -0.5 * splat_size * splat_size
    k.location = (100, -400)
    L.new(om.outputs[0], k.inputs[0])
    ex = N.new("ShaderNodeMath")
    ex.operation = "EXPONENT"
    ex.location = (250, -400)
    L.new(k.outputs[0], ex.inputs[0])
    fac = N.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    fac.use_clamp = True
    fac.location = (400, -250)
    L.new(alp.outputs["Fac"], fac.inputs[0])
    L.new(ex.outputs[0], fac.inputs[1])
    mix = N.new("ShaderNodeMixShader")
    mix.location = (500, 50)
    L.new(fac.outputs[0], mix.inputs["Fac"])
    L.new(transp.outputs[0], mix.inputs[1])
    L.new(emis.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def setup_sequence(seq_dir: str, object_name: str = "GS4D_Tree", fps: int | None = None,
                   splat_size: float = SPLAT_SIZE, detail: int = DETAIL, set_scene: bool = True):
    seq_dir = os.path.abspath(bpy.path.abspath(seq_dir))
    prefix, first, count = find_sequence(seq_dir)
    ng = build_node_group()
    mat = build_material(splat_size=splat_size)
    mesh = bpy.data.meshes.new(object_name + "_base")
    ob = bpy.data.objects.new(object_name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("GS4D Sequence", "NODES")
    mod.node_group = ng
    values = {
        "Path Pattern": os.path.join(seq_dir, prefix) + "####.ply",
        "First Index": first,
        "Frame Count": count,
        "Start Frame": 1,
        "Splat Size": splat_size,
        "Detail": detail,
        "Min Opacity": MIN_OPACITY,
        "Material": mat,
    }
    for item in ng.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name in values:
            mod[item.identifier] = values[item.name]
    if set_scene:
        sc = bpy.context.scene
        sc.frame_start = 1
        sc.frame_end = count
        if fps:
            sc.render.fps = int(fps)
        try:
            sc.view_settings.view_transform = "Standard"  # 3DGS colours are display-referred
        except TypeError:
            pass
    print(f"[gs4d] {object_name}: {count} frames of {prefix}####.ply from {seq_dir}")
    return ob


def add_camera_and_world(target=(0.0, 0.0, 2.6), distance=11.0, elevation_deg=6.0, background=(0.9, 0.93, 0.97)):
    from mathutils import Vector

    sc = bpy.context.scene
    cam_data = bpy.data.cameras.new("GS4D_Camera")
    cam_data.lens = 40
    cam = bpy.data.objects.new("GS4D_Camera", cam_data)
    sc.collection.objects.link(cam)
    el = math.radians(elevation_deg)
    target = Vector(target)
    cam.location = target + Vector((0.0, -distance * math.cos(el), distance * math.sin(el)))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    world = sc.world or bpy.data.worlds.new("GS4D_World")
    sc.world = world
    if bpy.app.version < (5, 0, 0):
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (*[c ** 2.2 for c in background], 1.0)
        bg.inputs[1].default_value = 1.0
    return cam


def main(argv=None):
    argv = argv if argv is not None else (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    ap = argparse.ArgumentParser(description="Import a Gaussian-splat PLY sequence")
    ap.add_argument("--dir", default=SEQ_DIR)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--size", type=float, default=SPLAT_SIZE)
    ap.add_argument("--detail", type=int, default=DETAIL)
    ap.add_argument("--camera", action="store_true", help="add a camera and a light-grey world")
    ap.add_argument("--render", default="", help="render the animation to this path (e.g. //render/frame_####)")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=720)
    args = ap.parse_args(argv)
    seq_dir = args.dir
    if not seq_dir:
        here = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        text_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else here
        seq_dir = os.path.join(text_dir, "ply_sequence")
    setup_sequence(seq_dir, fps=args.fps, splat_size=args.size, detail=args.detail)
    if args.camera or args.render:
        add_camera_and_world()
    if args.render:
        sc = bpy.context.scene
        sc.render.engine = args.engine
        if args.engine == "CYCLES":
            sc.cycles.samples = args.samples
            sc.cycles.use_denoising = False
        sc.render.resolution_x = sc.render.resolution_y = args.resolution
        sc.render.filepath = args.render
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
