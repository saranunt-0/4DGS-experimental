"""CPU rasteriser (gs4d.torch_raster) and render-to-splat distillation (gs4d.distill)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gs4d.distill import distill_rig, evaluate, load_rendered_dataset, render_points, write_dataset  # noqa: E402
from gs4d.render import look_at, render_cpu  # noqa: E402
from gs4d.torch_raster import rasterize_cpu, render_model  # noqa: E402


@pytest.fixture(autouse=True)
def _one_thread():
    """Tiny tensors: one thread is faster and immune to contention with other jobs."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_forward_matches_numpy_renderer(small_tree):
    m, _ = small_tree
    lo, hi = m.bounds()
    c = 0.5 * (lo + hi)
    cam = look_at(c + np.array([0.0, -2.5, 0.8]) * np.linalg.norm(hi - lo), c, [0, 0, 1], 96, 80)
    a, b = render_cpu(m, cam), render_model(m, cam)
    assert np.abs(a - b).mean() < 1e-3 and np.abs(a - b).max() < 0.05


def test_gradients_and_densification_info():
    gen = torch.Generator().manual_seed(0)
    n = 25
    means = (torch.randn(n, 3, generator=gen) * 0.3).requires_grad_()
    quats = torch.randn(n, 4, generator=gen).requires_grad_()
    scales = (torch.rand(n, 3, generator=gen) * 0.2 + 0.05).requires_grad_()
    op = (torch.rand(n, generator=gen) * 0.8 + 0.1).requires_grad_()
    col = torch.rand(n, 1, 3, generator=gen).requires_grad_()
    cam = look_at([0, -3, 0.3], [0, 0, 0], [0, 0, 1], 40, 40)
    V, K = torch.tensor(cam.viewmat, dtype=torch.float32)[None], torch.tensor(cam.K, dtype=torch.float32)[None]
    target = torch.rand(40, 40, 3, generator=gen)

    def loss():
        r, _, info = rasterize_cpu(means, quats, scales, op, col, V, K, 40, 40, backgrounds=torch.ones(1, 3))
        return ((r[0] - target) ** 2).double().mean(), info

    L, info = loss()
    info["means2d"].retain_grad()
    L.backward()
    assert {"means2d", "radii", "width", "height", "n_cameras", "gaussian_ids"} <= set(info)
    assert info["means2d"].shape == (1, n, 2) and info["radii"].shape == (1, n, 2)
    assert (info["means2d"].grad.abs().sum(-1) > 0).sum() > n // 2
    for p, tol in ((op, 0.02), (col, 0.02), (means, 0.25)):     # geometry: hard 3-sigma cut, like CUDA 3DGS
        j = tuple(np.unravel_index(int(p.grad.abs().argmax()), p.shape))
        eps = 1e-3
        with torch.no_grad():
            old = p[j].item()
            p[j] = old + eps
            lp = loss()[0].item()
            p[j] = old - eps
            lm = loss()[0].item()
            p[j] = old
        num = (lp - lm) / (2 * eps)
        assert abs(p.grad[j].item() - num) <= tol * abs(num) + 1e-6


def test_render_points_zbuffer_and_coverage():
    g = np.stack(np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)), -1).reshape(-1, 2)
    near = np.c_[g * 0.5, np.full(len(g), 0.5)]            # small red square in front
    far = np.c_[g, np.full(len(g), -0.5)]                  # large blue square behind
    pts = np.concatenate([near, far])
    col = np.concatenate([np.tile([1.0, 0, 0], (len(g), 1)), np.tile([0, 0, 1.0], (len(g), 1))])
    cam = look_at([0, 0, 4], [0, 0, 0], [0, 1, 0], 64, 64)
    rgb, a = render_points(pts, col, cam, point_size=0.02)
    assert a[32, 32] > 0.99 and a[1, 1] < 0.01                  # covered centre, empty corner
    assert rgb[32, 32, 0] > 0.9 and rgb[32, 32, 2] < 0.1        # the near square wins the z-test
    assert rgb[32, 14, 2] > 0.9                                # far square visible around it


def test_rig_dataset_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(3000, 3)) * [1, 1, 2] + [5, 5, 3]
    rig = distill_rig(pts, width=32, height=32, rings=((0, 6), (40, 4)), close_rings=((20, 3),), n_test=4)
    assert len(rig.train) == 6 + 4 + 1 + 3 and len(rig.test) == 4
    imgs = [np.full((32, 32, 3), 0.5)] * len(rig.train)
    alphas = [np.ones((32, 32))] * len(rig.train)
    write_dataset(tmp_path / "train", rig.train, imgs, alphas)
    data = load_rendered_dataset(tmp_path / "train")
    assert len(data.images) == len(rig.train)
    for a, b in zip(data.cameras, rig.train):
        assert np.allclose(a.viewmat, b.viewmat, atol=1e-5) and np.isclose(a.fx, b.fx)
    for c in rig.train:                                        # every camera looks at the object
        p = c.viewmat[:3, :3] @ pts.mean(0) + c.viewmat[:3, 3]
        assert p[2] > 0 and abs(c.fx * p[0] / p[2] + c.cx - 16) < 16


def test_train_from_renders_beats_init(small_tree):
    """A short CPU run on point renders: densification respects the budget and held-out PSNR improves."""
    from gs4d.train import TrainConfig, TrainData, train_object_splat

    m, _ = small_tree
    pts, cols = m.means.astype(np.float64), m.rgb
    rig = distill_rig(pts, width=48, height=48, rings=((0, 8), (35, 8)), close_rings=(), n_test=3)
    rend = [render_points(pts, cols, c, point_size=0.04) for c in rig.train + rig.test]
    tr, te = rend[: len(rig.train)], rend[len(rig.train):]
    data = TrainData([(r * 255).astype(np.uint8) for r, _ in tr], [a.astype(np.float32) for _, a in tr],
                     rig.train, np.zeros((0, 3)), np.zeros((0, 3)))
    curve = []
    cfg = TrainConfig(iters=200, sh_degree=0, device="cpu", max_gaussians=1500, init_points=800, log_every=0,
                      eval_every=100, reset_every=10 ** 6, res_schedule=((0, 2), (100, 1)), refine_start=30,
                      refine_every=30, refine_stop_frac=0.8, grow_grad2d=1e-5, init_scale=0.3)
    sizes = []
    train_object_splat(
        data, cfg, rasterize_fn=rasterize_cpu,
        callback=lambda s, mm: (sizes.append(len(mm)), curve.append(
            evaluate({"m": mm}, rig.test, [r for r, _ in te], [a for _, a in te])["m"])),
    )
    assert max(sizes) > 1000 and max(sizes) <= 1500           # densified, but within the budget
    assert curve[-1]["psnr"] > curve[0]["psnr"] + 2.0
