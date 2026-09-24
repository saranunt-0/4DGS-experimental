"""CPU smoke test for the masked training loop.

gsplat's rasteriser needs CUDA, so the renderer is replaced by a tiny
differentiable stand-in; gsplat's real ``DefaultStrategy`` (densification /
pruning bookkeeping) is still exercised end to end.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("gsplat") is None,
    reason="torch + gsplat not installed",
)


def mock_rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                       sh_degree=None, packed=False, backgrounds=None, **kw):
    import torch

    C, N = viewmats.shape[0], means.shape[0]
    R, t = viewmats[:, :3, :3], viewmats[:, :3, 3]
    pc = torch.einsum("cij,nj->cni", R, means) + t[:, None]
    z = pc[..., 2].clamp(min=1e-3)
    means2d = torch.stack([Ks[:, 0, 0, None] * pc[..., 0] / z + Ks[:, 0, 2, None],
                           Ks[:, 1, 1, None] * pc[..., 1] / z + Ks[:, 1, 2, None]], -1)
    col = colors[:, 0, :] * 0.28209479177387814 + 0.5
    w = opacities / (opacities.sum() + 1e-6)
    alpha = opacities.mean().clamp(0, 1)
    img = (w[:, None] * col).sum(0) * alpha + backgrounds[0] * (1 - alpha)
    img = img.expand(C, height, width, 3) + 0.0 * means2d.sum() + 0.0 * scales.sum() + 0.0 * quats.sum()
    alphas = alpha.expand(C, height, width, 1)
    radii = torch.full((C, N, 2), 3, dtype=torch.int32)
    info = {"means2d": means2d, "radii": radii, "width": width, "height": height, "n_cameras": C, "gaussian_ids": None}
    return img, alphas, info


def test_train_loop_cpu(small_tree):
    from gs4d.render import orbit_cameras, render_cpu
    from gs4d.train import TrainConfig, TrainData, train_object_splat

    m, _ = small_tree
    lo, hi = m.bounds(0.002)
    c = (lo + hi) / 2
    cams = orbit_cameras(c, 9.0, n=4, width=24, height=24)
    imgs, masks = [], []
    for cam in cams:
        im, a = render_cpu(m, cam, return_alpha=True)
        imgs.append((im * 255).astype(np.uint8))
        masks.append(a.astype(np.float32))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(m), 300, replace=False)
    data = TrainData(imgs, masks, cams, m.means[idx].astype(np.float64), m.rgb[idx])
    cfg = TrainConfig(iters=40, device="cpu", log_every=0, sh_degree=1, sh_degree_interval=10)
    out = train_object_splat(data, cfg, rasterize_fn=mock_rasterization)
    assert len(out) > 0 and out.sh_degree == 1
    assert np.isfinite(out.means).all() and np.isfinite(out.opacity_logits).all()
