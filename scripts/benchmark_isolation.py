"""Benchmark object isolation on synthetic captures (numbers quoted in docs/object_only_gaussians.md).

    python scripts/benchmark_isolation.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs4d import GaussianModel, TreeParams, generate_tree, isolate_object, mask_vote, orbit_cameras  # noqa: E402
from gs4d.pointcloud import PointCloud, isolate_point_cloud  # noqa: E402
from gs4d.quaternion import quat_to_matrix, rotvec_to_matrix  # noqa: E402
from gs4d.render import render_cpu  # noqa: E402
from gs4d.synthetic import make_captured_scene  # noqa: E402

TREES = {
    "default": TreeParams(),
    "seed 3": TreeParams(seed=3),
    "leaning (seed 11)": TreeParams(seed=11),
    "small/sparse": TreeParams(children=(4, 3, 3, 2), leaves_per_branch=(10, 16), segments=(6, 4, 3, 3, 2)),
}


def sample_points(model, n, seed=0):
    rng = np.random.default_rng(seed)
    s = model.scales
    area = s[:, 0] * s[:, 1] + s[:, 1] * s[:, 2] + s[:, 0] * s[:, 2]
    idx = rng.choice(len(model), n, p=area / area.sum())
    z = np.clip(rng.normal(size=(n, 3)), -1.8, 1.8) * s[idx]
    return model.means[idx] + np.einsum("nij,nj->ni", quat_to_matrix(model.quats[idx].astype(np.float64)), z), idx


def main():
    print("== captured splat scenes (ground, bush, sky shell, floaters): 4 trees x 3 scenes")
    rows, clean = [], []
    for name, tp in TREES.items():
        m, _ = generate_tree(tp)
        plain = GaussianModel(m.means, m.quats, m.log_scales, m.opacity_logits, m.sh_dc, m.sh_rest)
        clean.append(isolate_object(plain, up="+z", verbose=False)[0].mean())
        lo, hi = m.bounds(0.002)
        cams = orbit_cameras((lo + hi) / 2, 9.0, n=16, elevation_deg=12, width=128, height=128)
        masks = [render_cpu(m, cam, return_alpha=True)[1] > 0.3 for cam in cams]
        for seed in (0, 3, 5):
            scene, gt = make_captured_scene(m, seed=seed)
            k1, _ = isolate_object(scene, up="+z", verbose=False)
            k2 = mask_vote(scene, cams, masks, threshold=0.7)
            k3, _ = isolate_object(scene, up="+z", initial_keep=k2, verbose=False)
            rows.append([(k & gt).sum() / gt.sum() for k in (k1, k2, k3)] + [(k & ~gt).sum() / (~gt).sum() for k in (k1, k2, k3)])
    r = np.array(rows)
    for i, n in enumerate(["isolate_object (no cameras)", "mask_vote (16 views)", "mask_vote + isolate_object"]):
        print(f"  {n:30s} tree kept {r[:, i].mean() * 100:5.1f}% (min {r[:, i].min() * 100:5.1f}%)   "
              f"background kept {r[:, 3 + i].mean() * 100:5.2f}% (max {r[:, 3 + i].max() * 100:5.2f}%)")
    print("  clean trees (nothing to remove) kept:", [f"{c * 100:.1f}%" for c in clean])

    m, _ = generate_tree()
    scene, gt = make_captured_scene(m, seed=1)
    for deg in (15, 30):
        R = rotvec_to_matrix(np.array([np.deg2rad(deg), 0.0, 0.0]))
        keep, rep = isolate_object(scene.transformed(R), up="+z", verbose=False)
        err = np.degrees(np.arccos(np.clip(rep.ground["normal"] @ (R @ [0, 0, 1]), -1, 1)))
        print(f"  capture tilted {deg} deg: tree kept {(keep & gt).sum() / gt.sum() * 100:.2f}%, "
              f"background kept {(keep & ~gt).sum() / (~gt).sum() * 100:.2f}%, ground normal error {err:.2f} deg")

    print("== point-cloud scans (200k points, tree + ground): isolate_point_cloud")
    for name in ("default", "small/sparse"):
        m, _ = generate_tree(TREES[name])
        scene, _ = make_captured_scene(m, n_floaters=0, n_sky=0, n_clutter=0, n_ground=6000, ground_radius=1.0)
        pts, idx = sample_points(scene, 200_000)
        is_tree = idx < len(m)
        pc = PointCloud(pts.astype(np.float64), scene.rgb[idx])
        tree_pc, up, _ = isolate_point_cloud(pc, coarse_count=100_000, verbose=False)
        kept = np.zeros(len(pts), bool)
        kept[cKDTree(pts).query(tree_pc.points, k=1)[1]] = True
        print(f"  {name:14s} tree points kept {kept[is_tree].mean() * 100:.1f}%, ground points kept {kept[~is_tree].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
