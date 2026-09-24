import numpy as np
import pytest

from gs4d.gaussians import GaussianModel
from gs4d.io_ply import load_gaussians, save_ply, save_splat
from gs4d.quaternion import (
    matrix_to_quat, quat_from_rotvec, quat_multiply, quat_to_matrix, rotation_between, rotvec_to_matrix,
)


def random_model(n=500, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2 - 1
    return GaussianModel(
        rng.normal(size=(n, 3)), rng.normal(size=(n, 4)), rng.normal(-3, 0.5, (n, 3)),
        rng.normal(size=n), rng.normal(size=(n, 3)), rng.normal(size=(n, k, 3)) * 0.1,
        {"label": rng.integers(0, 3, n).astype(np.float32)},
    )


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    R = rotvec_to_matrix(v)
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-9)
    q = matrix_to_quat(R)
    assert np.allclose(quat_to_matrix(q), R, atol=1e-9)
    # composition: q(a) * q(b) == R(a) @ R(b)
    qa, qb = quat_from_rotvec(v[:100]), quat_from_rotvec(v[100:])
    assert np.allclose(quat_to_matrix(quat_multiply(qa, qb)), R[:100] @ R[100:], atol=1e-9)
    Rb = rotation_between([0, 0, 1], [0, -1, 0])
    assert np.allclose(Rb @ [0, 0, 1], [0, -1, 0])


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_ply_roundtrip(tmp_path, degree):
    m = random_model(degree=degree)
    p = save_ply(m, tmp_path / "a.ply", extra_attributes=["label"])
    r = load_gaussians(p)
    assert len(r) == len(m) and r.sh_degree == degree
    for a in ("means", "quats", "log_scales", "opacity_logits", "sh_dc", "sh_rest"):
        assert np.allclose(getattr(r, a), getattr(m, a), atol=1e-6), a
    assert np.allclose(r.extra["label"], m.extra["label"])
    # SH truncation keeps the DC term
    r0 = load_gaussians(save_ply(m, tmp_path / "b.ply", sh_degree=0))
    assert r0.sh_degree == 0 and np.allclose(r0.sh_dc, m.sh_dc, atol=1e-6)


def test_ply_reference_layout(tmp_path):
    """f_rest must be channel-major like the reference 3DGS exporter."""
    from gs4d.io_ply import read_ply_vertices

    m = random_model(n=10, degree=1)
    cols = read_ply_vertices(save_ply(m, tmp_path / "c.ply"))
    assert np.allclose(cols["f_rest_0"], m.sh_rest[:, 0, 0])  # R, coeff 1
    assert np.allclose(cols["f_rest_3"], m.sh_rest[:, 0, 1])  # G, coeff 1
    assert np.allclose(cols["f_rest_6"], m.sh_rest[:, 0, 2])  # B, coeff 1


def test_splat_format(tmp_path):
    m = random_model(degree=0)
    r = load_gaussians(save_splat(m, tmp_path / "a.splat"))
    assert len(r) == len(m)
    order = np.lexsort(r.means.T)
    order_m = np.lexsort(m.means.T)
    assert np.allclose(r.means[order], m.means[order_m], atol=1e-5)
    assert np.allclose(r.rgb[order], m.rgb[order_m], atol=1 / 255 + 1e-6)


def test_similarity_transform():
    m = random_model(degree=0)
    R = rotvec_to_matrix(np.array([0.3, -0.2, 0.9]))
    t = np.array([1.0, 2.0, 3.0])
    out = m.transformed(R, t, scale=2.0)
    assert np.allclose(out.means, 2.0 * m.means @ R.T + t, atol=1e-4)
    # covariance transforms as s^2 R Sigma R^T
    def cov(mm):
        Rg = quat_to_matrix(mm.quats.astype(np.float64))
        S = np.exp(mm.log_scales.astype(np.float64))
        M = Rg * S[:, None, :]
        return M @ M.transpose(0, 2, 1)
    assert np.allclose(cov(out), 4.0 * R @ cov(m) @ R.T, atol=1e-5)


def test_render_cpu(small_tree):
    from gs4d.render import framing, orbit_cameras, render_cpu

    m, _ = small_tree
    c, d = framing(m)
    cam = orbit_cameras(c, d, n=1, width=96, height=96)[0]
    img, alpha = render_cpu(m, cam, background=(1, 1, 1), return_alpha=True)
    assert img.shape == (96, 96, 3) and np.isfinite(img).all()
    assert 0.02 < (alpha > 0.5).mean() < 0.9
    # object pixels are darker than the white background
    assert img[alpha > 0.9].mean() < 0.6


def test_colmap_text_roundtrip(tmp_path, small_tree):
    from gs4d.colmap import read_colmap, write_colmap_text
    from gs4d.render import orbit_cameras

    cams = orbit_cameras([0, 0, 2], 8.0, n=5, width=64, height=48)
    names = [f"f{i}.png" for i in range(5)]
    write_colmap_text(tmp_path / "m", cams, names, np.zeros((3, 3)))
    sc = read_colmap(tmp_path / "m")
    assert sc.image_names == names and len(sc.points) == 3
    for a, b in zip(cams, sc.cameras):
        assert np.allclose(a.viewmat, b.viewmat, atol=1e-6) and a.width == b.width
