import importlib.util

import numpy as np
import pytest

from gs4d.cleanup import cameras_focus_point, isolate_object, mask_vote
from gs4d.export import export_all, export_ply_sequence
from gs4d.io_ply import load_gaussians
from gs4d.render import orbit_cameras, render_cpu
from gs4d.synthetic import make_captured_scene
from gs4d.wind import WindAnimation, WindParams


def test_isolate_object(small_tree):
    m, _ = small_tree
    scene, gt = make_captured_scene(m, seed=3)
    keep, rep = isolate_object(scene, up="+z", verbose=False)
    recall = (keep & gt).sum() / gt.sum()
    bg_kept = (keep & ~gt).sum() / (~gt).sum()
    assert recall > 0.97, str(rep)
    assert bg_kept < 0.01, str(rep)


def test_mask_vote_and_focus(small_tree):
    m, _ = small_tree
    scene, gt = make_captured_scene(m, seed=4, n_floaters=500)
    lo, hi = m.bounds(0.002)
    c = (lo + hi) / 2
    cams = orbit_cameras(c, 9.0, n=10, elevation_deg=12, width=64, height=64)
    assert np.linalg.norm(cameras_focus_point(cams) - c) < 1e-6
    masks = [render_cpu(m, cam, return_alpha=True)[1] > 0.3 for cam in cams]
    keep = mask_vote(scene, cams, masks, threshold=0.7)
    assert (keep & gt).sum() / gt.sum() > 0.97
    assert (keep & ~gt).sum() / (~gt).sum() < 0.03


@pytest.fixture(scope="module")
def short_anim(small_tree):
    m, sk = small_tree
    return WindAnimation(m, sk, WindParams(speed=6.0, duration=0.5, fps=8), verbose=False)


def test_ply_sequence(tmp_path, short_anim):
    paths = export_ply_sequence(short_anim, tmp_path, "t", progress=False)
    assert [p.name for p in paths] == [f"t_{i:04d}.ply" for i in range(1, len(short_anim) + 1)]
    f2 = load_gaussians(paths[2])
    ref = short_anim.frame(2)
    assert np.allclose(f2.means, ref.means, atol=1e-6)
    assert np.allclose(np.abs(np.sum(f2.quats * ref.quats, 1)), 1.0, atol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="usd-core not installed")
def test_usd_export(tmp_path, short_anim):
    from pxr import Usd, UsdVol

    res = export_all(short_anim, tmp_path, "t", formats=("usd",))
    stage = Usd.Stage.Open(str(res["usd"]))
    prim = UsdVol.ParticleField3DGaussianSplat(stage.GetPrimAtPath("/Tree/Splats"))
    pos = prim.GetPositionsAttr()
    assert len(pos.GetTimeSamples()) == len(short_anim)
    ref = short_anim.frame(1)
    assert np.allclose(np.array(pos.Get(2)), ref.means, atol=1e-6)
    q = prim.GetOrientationsAttr().Get(2)
    assert np.isclose(q[0].GetReal(), ref.quats[0, 0], atol=1e-6)
    assert np.allclose(list(q[0].GetImaginary()), ref.quats[0, 1:], atol=1e-6)
    assert np.allclose(np.array(prim.GetScalesAttr().Get()), ref.scales, rtol=1e-5)
    op = np.array(prim.GetOpacitiesAttr().Get())
    assert op.min() >= 0 and op.max() <= 1
    assert stage.GetEndTimeCode() - stage.GetStartTimeCode() + 1 == len(short_anim)


@pytest.mark.skipif(importlib.util.find_spec("bpy") is None, reason="bpy (Blender as a module) not installed")
def test_blender_import(tmp_path, short_anim):
    import sys

    import bpy

    res = export_all(short_anim, tmp_path, "t", formats=("blender",))
    sys.path.insert(0, str(tmp_path))
    import gs4d_import_sequence as g

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ob = g.setup_sequence(str(tmp_path / "ply_sequence"), detail=0)
    sc = bpy.context.scene
    assert sc.frame_end == len(short_anim)

    def instance_positions(frame):
        sc.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        return np.array([tuple(i.matrix_world.translation) for i in dg.object_instances
                         if i.is_instance and i.parent and i.parent.original.name == ob.name])

    p1, p3 = instance_positions(1), instance_positions(3)
    assert len(p1) == len(short_anim.model)
    assert np.allclose(np.sort(p3[:, 2]), np.sort(short_anim.frame(2).means[:, 2]), atol=1e-5)
    assert np.abs(p1 - p3).max() > 0
