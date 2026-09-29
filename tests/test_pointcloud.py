import numpy as np
import pytest

from gs4d.gaussians import GaussianModel
from gs4d.io_ply import load_gaussians
from gs4d.pointcloud import PointCloud, isolate_point_cloud, pointcloud_to_gaussians, read_point_cloud
from gs4d.quaternion import quat_to_matrix


def sample_points(model: GaussianModel, n: int, seed: int = 0):
    """Scanner-like points sampled from the surfaces of a splat model."""
    rng = np.random.default_rng(seed)
    s = model.scales
    area = s[:, 0] * s[:, 1] + s[:, 1] * s[:, 2] + s[:, 0] * s[:, 2]
    idx = rng.choice(len(model), n, p=area / area.sum())
    z = np.clip(rng.normal(size=(n, 3)), -1.8, 1.8) * s[idx]
    pts = model.means[idx] + np.einsum("nij,nj->ni", quat_to_matrix(model.quats[idx].astype(np.float64)), z)
    return pts.astype(np.float64), model.rgb[idx]


def write_ply(path, pts, rgb):
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.empty(len(pts), dt)
    a["x"], a["y"], a["z"] = pts.T
    c = (rgb * 255 + 0.5).astype(np.uint8)
    a["red"], a["green"], a["blue"] = c.T
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty double x\nproperty double y\n"
           "property double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(a.tobytes())


@pytest.fixture(scope="module")
def scan(small_tree, tmp_path_factory):
    from gs4d.synthetic import make_captured_scene

    m, _ = small_tree
    scene, _ = make_captured_scene(m, n_floaters=0, n_sky=0, n_clutter=0, n_ground=6000, ground_radius=1.0)
    pts, rgb = sample_points(scene, 200_000)
    utm = np.array([512345.678, 4123456.789, 87.0])
    path = tmp_path_factory.mktemp("scan") / "oak.ply"
    write_ply(path, pts + utm, rgb)
    return path, pts, rgb, m


def test_read_recenters_survey_coordinates(scan):
    path, pts, rgb, _ = scan
    pc = read_point_cloud(path)
    assert np.all(np.abs(pc.offset[:2]) > 1e5) and np.abs(pc.points).max() < 200
    assert np.allclose(pc.points + pc.offset, pts + [512345.678, 4123456.789, 87.0], atol=1e-6)
    assert np.allclose(pc.colors, rgb, atol=1 / 255 + 1e-6)


@pytest.mark.parametrize("mode,target", [("surfel", 50_000), ("voxel", 30_000)])
def test_fit_modes(scan, mode, target):
    path, *_ = scan
    pc = read_point_cloud(path)
    g = pointcloud_to_gaussians(pc, target_count=target, mode=mode, verbose=False)
    assert 0.5 * target < len(g) <= 1.3 * target
    assert np.isfinite(g.means).all() and np.isfinite(g.log_scales).all()
    assert np.allclose(np.linalg.norm(g.quats, axis=1), 1, atol=1e-5)
    # splats hug the points: every Gaussian centre is near the cloud
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pc.points).query(g.means, k=1)
    assert np.quantile(d, 0.99) < 0.05 * np.ptp(pc.points, axis=0).max()


def test_load_gaussians_accepts_point_clouds(scan, tmp_path):
    path, pts, rgb, _ = scan
    g = load_gaussians(path, max_points=20_000)
    assert len(g) <= 26_000 and g.sh_degree == 0
    xyz = tmp_path / "oak.xyz"
    np.savetxt(xyz, np.concatenate([pts[:3000], rgb[:3000] * 255], 1), fmt="%.5f")
    g2 = load_gaussians(xyz)
    assert len(g2) == 3000


def test_isolate_point_cloud(small_tree):
    from gs4d.synthetic import make_captured_scene

    m, _ = small_tree
    scene, _ = make_captured_scene(m, n_floaters=0, n_sky=0, n_clutter=0, n_ground=6000, ground_radius=1.0)
    pts, rgb = sample_points(scene, 120_000)
    # label: points sampled from tree splats vs ground splats
    from scipy.spatial import cKDTree
    _, nearest = cKDTree(scene.means).query(pts, k=1)
    is_tree = nearest < len(m)
    tree_pc, up, rep = isolate_point_cloud(PointCloud(pts, rgb), up="auto", coarse_count=60_000, verbose=False)
    assert np.dot(up, [0, 0, 1]) > 0.99
    kept = np.zeros(len(pts), bool)
    kept[cKDTree(pts).query(tree_pc.points, k=1)[1]] = True
    assert kept[is_tree].mean() > 0.97
    assert kept[~is_tree].mean() < 0.01


def broad_oak_points(seed: int = 0) -> np.ndarray:
    """Old-oak silhouette: short trunk, wide crown off-centre from it, branch curtains hanging to near the ground."""
    rng = np.random.default_rng(seed)
    a, z = rng.uniform(0, 2 * np.pi, 5000), rng.uniform(0.0, 7.0, 5000)
    trunk = np.stack([0.8 * np.cos(a), 0.8 * np.sin(a), z], 1)
    d = rng.normal(size=(100_000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    crown = d * [11.0, 11.0, 7.0] * rng.uniform(0.85, 1.0, (len(d), 1)) + [4.0, 0.0, 12.0]
    rim = crown[(np.linalg.norm(crown[:, :2] - [4.0, 0.0], axis=1) > 9.0) & (crown[:, 2] < 12.0)]
    curtain = rim[rng.choice(len(rim), len(rim) // 2)]
    curtain[:, 2] = rng.uniform(2.0, curtain[:, 2])
    return np.concatenate([trunk, crown, curtain])


@pytest.mark.parametrize("up", ["+z", "-y", "+x"])
def test_guess_up_broad_crown(up):
    """Geometry only (no colour cue): the thin, light trunk end must win over the crown's sides.

    The median-spread cue alone picked a crown side here (as on the real 200-year-old oak scan)."""
    from gs4d.quaternion import rotation_between
    from gs4d.skeleton import guess_up_axis, parse_up

    R = rotation_between([0, 0, 1], parse_up(up))
    pts = broad_oak_points() @ R.T
    m = pointcloud_to_gaussians(PointCloud(pts, np.full((len(pts), 3), 0.5)), 40_000, verbose=False)
    assert guess_up_axis(m) == up


def test_fix_sky_bleed():
    """Blue and grey sky colours inside foliage are repainted; white bark and green leaves are untouched."""
    from gs4d.pointcloud import fix_sky_bleed

    rng = np.random.default_rng(0)
    d = rng.normal(size=(20000, 3))
    crown = d / np.linalg.norm(d, axis=1, keepdims=True) * 2.5 * rng.random((20000, 1)) ** (1 / 3) + [0, 0, 6]
    leaf = np.clip(rng.normal([0.20, 0.30, 0.05], 0.03, (len(crown), 3)), 0, 1)
    sky = rng.random(len(crown)) < 0.1
    leaf[sky] = np.where(rng.random((sky.sum(), 1)) < 0.5, [0.55, 0.70, 0.95], [0.66, 0.68, 0.67])
    a, z = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(0, 3.3, 3000)
    trunk = np.stack([0.4 * np.cos(a), 0.4 * np.sin(a), z], 1)       # white birch bark below the crown
    bark = np.full((len(trunk), 3), 0.85)
    floaters = rng.uniform(-1, 1, (50, 3)) + [8, 8, 8]
    pts = np.concatenate([crown, trunk, floaters])
    col = np.concatenate([leaf, bark, np.tile([0.55, 0.70, 0.95], (50, 1))])
    out = fix_sky_bleed(PointCloud(pts, col), verbose=False)
    assert len(out) == len(crown) + len(trunk)                        # only the sky floaters are removed
    fixed = out.colors[: len(crown)][sky]
    assert np.all(fixed[:, 1] > fixed[:, 2] + 0.1)                    # repainted green, not blue / grey
    assert np.allclose(out.colors[: len(crown)][~sky], leaf[~sky])    # real leaves untouched
    assert np.allclose(out.colors[len(crown):], 0.85)                 # white bark kept
