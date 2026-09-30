"""Render-to-splat distillation: photograph a point cloud, then train a 3DGS on the photos.

The image-based alternative to fitting Gaussians to the points directly
(:func:`gs4d.pointcloud.pointcloud_to_gaussians`).  It is the same idea as
Houdini's ``generate_training_data`` or Blender camera-array rigs: the
renders contain only the object (alpha = 0 elsewhere), the camera poses are
known exactly, so no COLMAP and no background are needed.

* :func:`distill_rig` - training and held-out test cameras around the object;
* :func:`render_points` - z-buffered, supersampled RGBA point-cloud renders
  (what a point-cloud viewer shows);
* :func:`write_dataset` / :func:`load_rendered_dataset` - a standard COLMAP
  text model + RGBA PNGs + masks that gsplat, nerfstudio, Postshot or Brush
  can also train on;
* :func:`evaluate` - PSNR / SSIM / silhouette error of models on test views.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .render import Camera, look_at


@dataclass
class Rig:
    train: list
    test: list
    center: np.ndarray
    distance: float


def distill_rig(points: np.ndarray, up=(0, 0, 1), width: int = 512, height: int = 512, fov_deg: float = 45.0,
                rings=((-10, 16), (10, 24), (30, 24), (50, 16), (70, 8)), close_rings=((5, 20), (35, 16)),
                close_factor: float = 0.55, n_test: int = 16, seed: int = 0) -> Rig:
    """Cameras on rings around the object plus closer views of its upper part; test views lie in between.

    ``rings`` are ``(elevation_deg, count)`` at the distance that frames the
    whole object; ``close_rings`` sit at ``close_factor`` x that distance
    and aim at random points of the upper two thirds (crown detail).
    """
    from .skeleton import horizontal_basis

    rng = np.random.default_rng(seed)
    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    lo, hi = np.quantile(points, 0.002, axis=0), np.quantile(points, 0.998, axis=0)
    center = 0.5 * (lo + hi)
    sample = points[rng.choice(len(points), min(len(points), 200_000), replace=False)]
    r = float(np.quantile(np.linalg.norm(sample - center, axis=1), 0.998))   # bounding sphere, not box
    dist = 1.02 * r / np.sin(np.deg2rad(fov_deg) / 2)
    e1, e2 = horizontal_basis(up)
    h = (points - center) @ up
    h_lo, h_hi = np.quantile(h, [0.02, 0.98])

    def cam(el_deg, az_deg, d, target):
        el, az = np.deg2rad(el_deg), np.deg2rad(az_deg)
        direction = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
        return look_at(target + d * direction, target, up, width, height, fov_deg)

    def crown_target():
        off = rng.uniform(-0.3, 0.3, 3) * r
        off -= (off @ up) * up
        return center + off + up * rng.uniform(h_lo + 0.35 * (h_hi - h_lo), h_lo + 0.85 * (h_hi - h_lo))

    train = []
    for k, (el, n) in enumerate(rings):
        train += [cam(el, 360.0 * i / n + 7.0 * k, dist, center) for i in range(n)]
    train.append(cam(89.0, 0.0, dist, center))
    for k, (el, n) in enumerate(close_rings):
        train += [cam(el, 360.0 * i / n + 11.0 * k, dist * close_factor, crown_target()) for i in range(n)]
    test = []
    n_far = n_test - n_test // 4
    for i in range(n_far):
        test.append(cam(rng.uniform(-5, 65), rng.uniform(0, 360), dist * rng.uniform(0.95, 1.05), center))
    for i in range(n_test - n_far):
        test.append(cam(rng.uniform(5, 40), rng.uniform(0, 360), dist * close_factor, crown_target()))
    return Rig(train, test, center, float(dist))


def render_points(points: np.ndarray, colors: np.ndarray, cam: Camera, point_size: float = 0.025,
                  supersample: int = 2, max_radius_px: int = 6, depth_tol: float | None = None):
    """Render a coloured point cloud like a point-cloud viewer, with anti-aliased RGBA output.

    Each point is a disc of ``point_size`` (world units, at least ~1 pixel)
    drawn with a z-buffer at ``supersample`` x resolution; points within
    ``depth_tol`` of the front surface are averaged, then the image is box-
    filtered down.  Returns ``(rgb (H, W, 3) float, alpha (H, W) float)``;
    ``rgb`` is not premultiplied.
    """
    import torch

    s = int(supersample)
    W, H = cam.width * s, cam.height * s
    fx, fy, cx, cy = cam.fx * s, cam.fy * s, cam.cx * s, cam.cy * s
    R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
    p = torch.from_numpy((points @ R.T + t).astype(np.float32))
    col = torch.from_numpy(np.asarray(colors, np.float32))
    z = p[:, 2]
    ok = z > 1e-3
    u = fx * p[:, 0] / z.clamp_min(1e-3) + cx
    v = fy * p[:, 1] / z.clamp_min(1e-3) + cy
    rad = (0.5 * point_size * fx / z.clamp_min(1e-3)).round().clamp(1, max_radius_px).long()
    ok &= (u > -max_radius_px) & (u < W + max_radius_px) & (v > -max_radius_px) & (v < H + max_radius_px)
    idx = torch.nonzero(ok).squeeze(1)
    ui, vi, zi, ri = u[idx].floor().long(), v[idx].floor().long(), z[idx], rad[idx]
    tol = float(point_size if depth_tol is None else depth_tol)

    def passes():
        for r in torch.unique(ri).tolist():
            sel = torch.nonzero(ri == r).squeeze(1)
            for oy in range(-r, r + 1):
                for ox in range(-r, r + 1):
                    if ox * ox + oy * oy > (r + 0.5) ** 2:
                        continue
                    px, py = ui[sel] + ox, vi[sel] + oy
                    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                    yield sel[inside], py[inside] * W + px[inside]

    zbuf = torch.full((H * W,), float("inf"))
    for sel, pix in passes():
        zbuf.scatter_reduce_(0, pix, zi[sel], reduce="amin")
    acc = torch.zeros(H * W, 3)
    cnt = torch.zeros(H * W)
    for sel, pix in passes():
        front = zi[sel] <= zbuf[pix] + tol
        acc.index_add_(0, pix[front], col[idx[sel[front]]])
        cnt.index_add_(0, pix[front], torch.ones(int(front.sum())))
    cov = (cnt > 0).float()
    rgb_pm = acc / cnt.clamp_min(1)[:, None] * cov[:, None]
    rgb_pm = rgb_pm.view(cam.height, s, cam.width, s, 3).mean((1, 3))
    alpha = cov.view(cam.height, s, cam.width, s).mean((1, 3))
    rgb = rgb_pm / alpha.clamp_min(1e-6)[..., None]
    return rgb.clamp(0, 1).numpy(), alpha.numpy()


def write_dataset(scene_dir: str | Path, cams: list, images: list, alphas: list, prefix: str = "view") -> Path:
    """COLMAP text model (known poses) + RGBA PNGs in ``images/`` + masks in ``masks/``."""
    from PIL import Image

    from .colmap import write_colmap_text

    d = Path(scene_dir)
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    names = []
    for i, (rgb, a) in enumerate(zip(images, alphas)):
        name = f"{prefix}_{i:03d}.png"
        rgba = np.concatenate([rgb * (a[..., None] > 0), a[..., None]], -1)
        Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(d / "images" / name)
        Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(d / "masks" / name)
        names.append(name)
    write_colmap_text(d / "sparse" / "0", cams, names)
    return d


def load_rendered_dataset(scene_dir: str | Path):
    """Read a :func:`write_dataset` folder back as :class:`gs4d.train.TrainData` (no SfM points)."""
    from .train import load_dataset

    d = Path(scene_dir)
    return load_dataset(d, d / "masks", max_width=10 ** 6)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    import torch

    from .train import _ssim

    with torch.no_grad():
        return float(_ssim(torch.from_numpy(a.astype(np.float32)), torch.from_numpy(b.astype(np.float32))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(np.mean((a.astype(np.float64) - b) ** 2), 1e-12)))


def evaluate(models: dict, cams: list, gt_rgb: list, gt_alpha: list, background=(1.0, 1.0, 1.0)) -> dict:
    """Per-model mean PSNR / SSIM (RGB over ``background``) and silhouette errors on the given views."""
    from .torch_raster import render_model

    bg = np.asarray(background, np.float32)
    out = {}
    for name, model in models.items():
        ps, ss, amae, iou = [], [], [], []
        for cam, rgb, a in zip(cams, gt_rgb, gt_alpha):
            gt = rgb * a[..., None] + bg * (1 - a[..., None])
            img, al = render_model(model, cam, background=tuple(bg), return_alpha=True)
            ps.append(psnr(img, gt))
            ss.append(ssim(img, gt))
            amae.append(float(np.abs(al - a).mean()))
            m1, m2 = al > 0.5, a > 0.5
            iou.append(float((m1 & m2).sum() / max((m1 | m2).sum(), 1)))
        out[name] = {"gaussians": len(model), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                     "alpha_mae": float(np.mean(amae)), "silhouette_iou": float(np.mean(iou)),
                     "psnr_per_view": [round(x, 2) for x in ps]}
    return out


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
