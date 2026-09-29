"""Preview rendering of Gaussian splats.

* :func:`render` uses **gsplat** on a CUDA GPU when available (fast, exact).
* Otherwise it falls back to :func:`render_cpu`, a vectorised numpy EWA
  splatter: every Gaussian is projected to a 2D ellipse, expanded into pixel
  fragments, and alpha-composited front-to-back per pixel (the same maths as
  the 3DGS rasteriser, minus tiling).  Good enough for previews / GIFs.

Cameras use the OpenCV / COLMAP / gsplat convention: world-to-camera 4x4
``viewmat``, x right, y down, z forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussians import GaussianModel, SH_C0
from .quaternion import quat_to_matrix


@dataclass
class Camera:
    viewmat: np.ndarray  # (4, 4) world -> camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def position(self) -> np.ndarray:
        R = self.viewmat[:3, :3]
        t = self.viewmat[:3, 3]
        return -R.T @ t


def look_at(eye, target, up, width=512, height=512, fov_deg=45.0) -> Camera:
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # points "down" on screen
    R = np.stack([x, y, z], 0)
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    f = 0.5 * height / np.tan(np.deg2rad(fov_deg) / 2)
    return Camera(V, f, f, width / 2.0, height / 2.0, int(width), int(height))


def orbit_cameras(
    center,
    radius: float,
    up=(0, 0, 1),
    n: int = 36,
    elevation_deg: float = 10.0,
    start_deg: float = 0.0,
    sweep_deg: float = 360.0,
    width: int = 512,
    height: int = 512,
    fov_deg: float = 45.0,
) -> list[Camera]:
    from .skeleton import horizontal_basis

    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    e1, e2 = horizontal_basis(up)
    cams = []
    el = np.deg2rad(elevation_deg)
    for i in range(n):
        az = np.deg2rad(start_deg + sweep_deg * i / max(n, 1))
        d = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
        cams.append(look_at(np.asarray(center) + radius * d, center, up, width, height, fov_deg))
    return cams


def framing(model: GaussianModel, up=(0, 0, 1), fov_deg: float = 45.0, margin: float = 1.15):
    """Centre and orbit radius that fit the model in view."""
    lo, hi = model.bounds(quantile=0.005)
    center = 0.5 * (lo + hi)
    r = 0.5 * np.linalg.norm(hi - lo)
    dist = margin * r / np.sin(np.deg2rad(fov_deg) / 2)
    return center, dist


# ----------------------------------------------------------------------------
# SH evaluation (reference 3DGS constants)
# ----------------------------------------------------------------------------
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
       -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def eval_sh(sh_dc: np.ndarray, sh_rest: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """RGB colour of each Gaussian seen along unit ``dirs`` (camera -> Gaussian)."""
    res = SH_C0 * sh_dc.astype(np.float64)
    k = sh_rest.shape[1]
    if k >= 3:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        s = sh_rest.astype(np.float64)
        res = res - _C1 * y * s[:, 0] + _C1 * z * s[:, 1] - _C1 * x * s[:, 2]
        if k >= 8:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + _C2[0] * xy * s[:, 3] + _C2[1] * yz * s[:, 4]
                   + _C2[2] * (2 * zz - xx - yy) * s[:, 5] + _C2[3] * xz * s[:, 6]
                   + _C2[4] * (xx - yy) * s[:, 7])
            if k >= 15:
                res = (res + _C3[0] * y * (3 * xx - yy) * s[:, 8] + _C3[1] * xy * z * s[:, 9]
                       + _C3[2] * y * (4 * zz - xx - yy) * s[:, 10]
                       + _C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * s[:, 11]
                       + _C3[4] * x * (4 * zz - xx - yy) * s[:, 12]
                       + _C3[5] * z * (xx - yy) * s[:, 13] + _C3[6] * x * (xx - 3 * yy) * s[:, 14])
    return np.clip(res + 0.5, 0.0, None)


# ----------------------------------------------------------------------------
# CPU renderer
# ----------------------------------------------------------------------------
def render_cpu(
    model: GaussianModel,
    cam: Camera,
    background=(1.0, 1.0, 1.0),
    max_radius_px: int = 40,
    colors: np.ndarray | None = None,
    use_sh: bool = True,
    chunk: int = 40000,
    return_alpha: bool = False,
):
    """Exact-order alpha compositing of projected Gaussians on the CPU."""
    W, H = cam.width, cam.height
    R = cam.viewmat[:3, :3]
    t = cam.viewmat[:3, 3]
    p = model.means.astype(np.float64) @ R.T + t
    z = p[:, 2]
    vis = z > 0.02
    # frustum cull (generous)
    u = cam.fx * p[:, 0] / np.maximum(z, 1e-6) + cam.cx
    v = cam.fy * p[:, 1] / np.maximum(z, 1e-6) + cam.cy
    vis &= (u > -max_radius_px) & (u < W + max_radius_px) & (v > -max_radius_px) & (v < H + max_radius_px)
    idx = np.flatnonzero(vis)
    idx = idx[np.argsort(z[idx], kind="stable")]  # front to back

    img = np.zeros((H * W, 3))
    Tpix = np.ones(H * W)
    if len(idx) == 0:
        out = np.broadcast_to(np.asarray(background, np.float64), (H, W, 3)).copy()
        return (out, np.zeros((H, W))) if return_alpha else out

    if colors is None:
        if use_sh and model.sh_rest.shape[1] > 0:
            dirs = model.means[idx].astype(np.float64) - cam.position
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            col_all = np.zeros((len(model), 3))
            col_all[idx] = eval_sh(model.sh_dc[idx], model.sh_rest[idx], dirs)
        else:
            col_all = np.clip(model.sh_dc.astype(np.float64) * SH_C0 + 0.5, 0, None)
    else:
        col_all = np.asarray(colors, np.float64)
    opac_all = model.opacities

    for start in range(0, len(idx), chunk):
        ids = idx[start:start + chunk]
        pc = p[ids]
        zc = pc[:, 2]
        Rg = quat_to_matrix(model.quats[ids].astype(np.float64))
        S = np.exp(model.log_scales[ids].astype(np.float64))
        M = Rg * S[:, None, :]
        cov_w = M @ M.transpose(0, 2, 1)
        cov_c = R @ cov_w @ R.T
        J = np.zeros((len(ids), 2, 3))
        J[:, 0, 0] = cam.fx / zc
        J[:, 0, 2] = -cam.fx * pc[:, 0] / zc**2
        J[:, 1, 1] = cam.fy / zc
        J[:, 1, 2] = -cam.fy * pc[:, 1] / zc**2
        cov2 = J @ cov_c @ J.transpose(0, 2, 1)
        a = cov2[:, 0, 0] + 0.3
        b = cov2[:, 0, 1]
        c = cov2[:, 1, 1] + 0.3
        det = a * c - b * b
        ok = det > 1e-12
        mid = 0.5 * (a + c)
        lam = mid + np.sqrt(np.maximum(mid * mid - det, 0.0))
        rad = np.minimum(np.ceil(3.0 * np.sqrt(lam)), max_radius_px).astype(np.int64)
        uc = cam.fx * pc[:, 0] / zc + cam.cx
        vc = cam.fy * pc[:, 1] / zc + cam.cy
        x0 = np.clip(np.floor(uc - rad).astype(np.int64), 0, W - 1)
        x1 = np.clip(np.ceil(uc + rad).astype(np.int64), 0, W - 1)
        y0 = np.clip(np.floor(vc - rad).astype(np.int64), 0, H - 1)
        y1 = np.clip(np.ceil(vc + rad).astype(np.int64), 0, H - 1)
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        ok &= (uc + rad >= 0) & (uc - rad < W) & (vc + rad >= 0) & (vc - rad < H)
        cnt = np.where(ok, bw * bh, 0)
        total = int(cnt.sum())
        if total == 0:
            continue
        g = np.repeat(np.arange(len(ids), dtype=np.int32), cnt)
        offs = np.repeat((np.cumsum(cnt) - cnt).astype(np.int64), cnt)
        local = (np.arange(total, dtype=np.int64) - offs).astype(np.int32)
        bwg = bw[g].astype(np.int32)
        px = x0[g].astype(np.int32) + local % bwg
        py = y0[g].astype(np.int32) + local // bwg
        # per-Gaussian conic, pre-divided by det, in float32 for bandwidth
        ca = (c / det).astype(np.float32)
        cb = (-2.0 * b / det).astype(np.float32)
        cc = (a / det).astype(np.float32)
        dx = (px + 0.5).astype(np.float32) - uc.astype(np.float32)[g]
        dy = (py + 0.5).astype(np.float32) - vc.astype(np.float32)[g]
        power = -0.5 * (ca[g] * dx * dx + cb[g] * dx * dy + cc[g] * dy * dy)
        alpha = np.minimum(np.float32(0.99), opac_all[ids].astype(np.float32)[g] * np.exp(power))
        keep = alpha >= 1.0 / 255.0
        g, alpha = g[keep], alpha[keep]
        pix = (py * W + px)[keep]
        if len(pix) == 0:
            continue
        order = np.argsort(pix, kind="stable")  # stable => keeps depth order inside each pixel
        pix, g, alpha = pix[order], g[order], alpha[order].astype(np.float64)
        log1m = np.log1p(-alpha)
        cs = np.cumsum(log1m)
        excl = cs - log1m
        seg_start = np.ones(len(pix), bool)
        seg_start[1:] = pix[1:] != pix[:-1]
        base = np.maximum.accumulate(np.where(seg_start, np.arange(len(pix)), 0))
        Tk = Tpix[pix] * np.exp(excl - excl[base])
        w = Tk * alpha
        col = col_all[ids]
        for ch in range(3):
            img[:, ch] += np.bincount(pix, weights=w * col[g, ch], minlength=H * W)
        seg_sum = np.bincount(pix, weights=log1m, minlength=H * W)
        Tpix *= np.exp(seg_sum)

    bg = np.asarray(background, np.float64)
    out = img + Tpix[:, None] * bg[None, :]
    out = np.clip(out.reshape(H, W, 3), 0, 1)
    if return_alpha:
        return out, (1.0 - Tpix).reshape(H, W)
    return out


# ----------------------------------------------------------------------------
# GPU renderer (gsplat)
# ----------------------------------------------------------------------------
def gsplat_available() -> bool:
    try:
        import torch  # noqa: F401
        import gsplat  # noqa: F401  (availability check)

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def render_gsplat(model: GaussianModel, cam: Camera, background=(1.0, 1.0, 1.0), colors=None, return_alpha=False):
    import torch
    from gsplat import rasterization

    dev = "cuda"
    means = torch.from_numpy(model.means).to(dev)
    quats = torch.from_numpy(model.quats).to(dev)
    scales = torch.from_numpy(np.exp(model.log_scales)).to(dev)
    opac = torch.sigmoid(torch.from_numpy(model.opacity_logits).to(dev))
    if colors is not None:
        cols = torch.from_numpy(np.asarray(colors, np.float32)).to(dev)
        sh_degree = None
    else:
        sh = np.concatenate([model.sh_dc[:, None, :], model.sh_rest], 1)
        cols = torch.from_numpy(sh).to(dev)
        sh_degree = model.sh_degree
    view = torch.from_numpy(cam.viewmat.astype(np.float32))[None].to(dev)
    K = torch.from_numpy(cam.K.astype(np.float32))[None].to(dev)
    bg = torch.tensor(np.asarray(background, np.float32))[None].to(dev)
    with torch.no_grad():
        img, alpha, _ = rasterization(
            means, quats, scales, opac, cols, view, K, cam.width, cam.height,
            sh_degree=sh_degree, backgrounds=bg, packed=False,
        )
    out = img[0].clamp(0, 1).cpu().numpy()
    if return_alpha:
        return out, alpha[0, ..., 0].cpu().numpy()
    return out


def render(model: GaussianModel, cam: Camera, background=(1.0, 1.0, 1.0), backend: str = "auto", **kw):
    """Render with gsplat on GPU if possible, else the numpy fallback."""
    if backend == "gsplat" or (backend == "auto" and gsplat_available()):
        kw.pop("max_radius_px", None)
        kw.pop("use_sh", None)
        kw.pop("chunk", None)
        return render_gsplat(model, cam, background, **kw)
    return render_cpu(model, cam, background, **kw)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
