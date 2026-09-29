"""Differentiable Gaussian-splat rasteriser in plain PyTorch (runs on the CPU).

Drop-in for ``gsplat.rasterization`` inside :func:`gs4d.train.train_object_splat`
when there is no CUDA GPU: same call signature, same outputs
(``renders [C,H,W,3]``, ``alphas [C,H,W,1]``, ``info``) and the ``info`` keys
gsplat's ``DefaultStrategy`` needs for densification (``means2d`` with a
retained gradient, ``radii``, ``width``, ``height``, ``n_cameras``).

Same image formation as gsplat / the reference 3DGS rasteriser: EWA projection
with a 0.3 px low-pass, 3-sigma footprint, alpha = min(0.99, o * exp(power)),
fragments with alpha < 1/255 skipped, front-to-back compositing.  Instead of
tiles it enumerates every (Gaussian, pixel) fragment, sorts them by pixel and
depth, and composites with a segmented cumulative sum of log-transmittance, so
autograd does the backward pass.  Roughly 1 s per 512x512 view with 150k
splats on 4 CPU cores.
"""

from __future__ import annotations

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz quaternions (any norm) -> (N, 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def eval_sh(coeffs: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """RGB from SH ``coeffs`` (N, K, 3) along unit ``dirs`` (N, 3), gsplat basis order."""
    res = SH_C0 * coeffs[:, 0]
    if degree >= 1 and coeffs.shape[1] >= 4:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        s = coeffs
        res = res - SH_C1 * y * s[:, 1] + SH_C1 * z * s[:, 2] - SH_C1 * x * s[:, 3]
        if degree >= 2 and coeffs.shape[1] >= 9:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * s[:, 4] + SH_C2[1] * yz * s[:, 5] + SH_C2[2] * (2 * zz - xx - yy) * s[:, 6]
                   + SH_C2[3] * xz * s[:, 7] + SH_C2[4] * (xx - yy) * s[:, 8])
            if degree >= 3 and coeffs.shape[1] >= 16:
                res = (res + SH_C3[0] * y * (3 * xx - yy) * s[:, 9] + SH_C3[1] * xy * z * s[:, 10]
                       + SH_C3[2] * y * (4 * zz - xx - yy) * s[:, 11]
                       + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * s[:, 12]
                       + SH_C3[4] * x * (4 * zz - xx - yy) * s[:, 13]
                       + SH_C3[5] * z * (xx - yy) * s[:, 14] + SH_C3[6] * x * (xx - 3 * yy) * s[:, 15])
    return (res + 0.5).clamp_min(0.0)


_BUCKETS = None


def _bucket(n: torch.Tensor) -> torch.Tensor:
    """Round box sizes up to ~12% steps so a few dozen groups cover all splats."""
    global _BUCKETS
    if _BUCKETS is None or _BUCKETS.device != n.device:
        b, sizes = 1, []
        while b <= 4096:
            sizes.append(b)
            b = b + 1 if b < 8 else int(b * 1.125) + 1
        _BUCKETS = torch.tensor(sizes, device=n.device)
    return _BUCKETS[torch.searchsorted(_BUCKETS, n)]


def _render_view(means, quats, scales, opacities, colors, viewmat, K, width, height, sh_degree, background,
                 near_plane, eps2d, max_radius):
    W, H = int(width), int(height)
    dev = means.device
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    pc = means @ R.T + t
    x, y, z = pc.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zc = z.clamp_min(near_plane)
    u = fx * x / zc + cx
    v = fy * y / zc + cy
    means2d = torch.stack([u, v], -1)[None]          # (1, N, 2): gsplat's densification reads its gradient
    m2 = means2d[0]

    # EWA: 2D covariance J W Sigma W^T J^T (+ eps2d low-pass); x/z clamped like gsplat for stable Jacobians
    limx, limy = 1.3 * 0.5 * W / fx, 1.3 * 0.5 * H / fy
    tx = (x / zc).clamp(-limx, limx) * zc
    ty = (y / zc).clamp(-limy, limy) * zc
    Rg = quat_to_rotmat(quats)
    M = Rg * scales[:, None, :]
    S = R[None] @ (M @ M.transpose(1, 2)) @ R.T[None]
    ja, jb = fx / zc, -fx * tx / (zc * zc)
    jc, jd = fy / zc, -fy * ty / (zc * zc)
    cxx = ja * ja * S[:, 0, 0] + 2 * ja * jb * S[:, 0, 2] + jb * jb * S[:, 2, 2] + eps2d
    cxy = ja * (jc * S[:, 0, 1] + jd * S[:, 0, 2]) + jb * (jc * S[:, 2, 1] + jd * S[:, 2, 2])
    cyy = jc * jc * S[:, 1, 1] + 2 * jc * jd * S[:, 1, 2] + jd * jd * S[:, 2, 2] + eps2d
    det = cxx * cyy - cxy * cxy
    det_safe = det.clamp_min(1e-12)
    ca, cb, cc = cyy / det_safe, -cxy / det_safe, cxx / det_safe      # conic

    with torch.no_grad():
        rx = torch.ceil(3.0 * cxx.clamp_min(0).sqrt()).clamp(max=max_radius)
        ry = torch.ceil(3.0 * cyy.clamp_min(0).sqrt()).clamp(max=max_radius)
        valid = ((z > near_plane) & (det > 0) & (u + rx > 0) & (u - rx < W) & (v + ry > 0) & (v - ry < H)
                 & (opacities > 1.0 / 255.0))
        radii = torch.stack([rx, ry], -1).where(valid[:, None], torch.zeros_like(rx)[:, None]).int()[None]
        ids = torch.nonzero(valid).squeeze(1)
        rank = torch.empty_like(ids)
        rank[torch.argsort(z[ids])] = torch.arange(len(ids), device=dev)   # depth rank, 0 = nearest
        x0 = (u[ids] - rx[ids]).floor().clamp(0, W - 1).long()
        y0 = (v[ids] - ry[ids]).floor().clamp(0, H - 1).long()
        bw = ((u[ids] + rx[ids]).ceil().clamp(0, W - 1).long() - x0 + 1)
        bh = ((v[ids] + ry[ids]).ceil().clamp(0, H - 1).long() - y0 + 1)
        # group splats by (bucketed) box size and enumerate each group's pixels by broadcasting:
        # no per-fragment gathers while culling
        bwq, bhq = _bucket(bw), _bucket(bh)
        gkey = bwq * 4096 + bhq
        korder = torch.argsort(gkey)
        keys, counts = torch.unique_consecutive(gkey[korder], return_counts=True)
        gis, pixs, ranks = [], [], []
        start = 0
        for key, n in zip(keys.tolist(), counts.tolist()):
            sel = korder[start:start + n]
            start += n
            gw, gh = key // 4096, key % 4096
            oy, ox = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
            px = x0[sel, None] + ox.reshape(1, -1)
            py = y0[sel, None] + oy.reshape(1, -1)
            gsel = ids[sel]
            ddx = px.float() + 0.5 - u[gsel, None]
            ddy = py.float() + 0.5 - v[gsel, None]
            pw = -0.5 * (ca[gsel, None] * ddx * ddx + cc[gsel, None] * ddy * ddy) - cb[gsel, None] * ddx * ddy
            keep = ((px < W) & (py < H) & (pw <= 0) & (pw > -4.5)
                    & (opacities[gsel, None] * torch.exp(pw) >= 1.0 / 255.0))
            gis.append(gsel[:, None].expand_as(px)[keep])
            pixs.append((py * W + px)[keep])
            ranks.append(rank[sel, None].expand_as(px)[keep])
        gi, pix, rk = torch.cat(gis), torch.cat(pixs), torch.cat(ranks)
        order = torch.argsort(pix * max(len(ids), 1) + rk)   # by pixel, near -> far within a pixel
        gi, pix = gi[order], pix[order]
        dxs = (pix % W).float() + 0.5
        dys = torch.div(pix, W, rounding_mode="floor").float() + 0.5
        newseg = torch.ones_like(pix, dtype=torch.bool)
        newseg[1:] = pix[1:] != pix[:-1]
        seg = torch.cumsum(newseg, 0) - 1
        seg_start = torch.nonzero(newseg).squeeze(1)

    # differentiable part: only the surviving fragments
    ddx = dxs - m2[gi, 0]
    ddy = dys - m2[gi, 1]
    power = -0.5 * (ca[gi] * ddx * ddx + cc[gi] * ddy * ddy) - cb[gi] * ddx * ddy
    alpha = (opacities[gi] * torch.exp(power)).clamp(max=0.99)
    log1m = torch.log1p(-alpha).double()                 # float64: millions of terms in one cumsum
    excl = torch.cumsum(log1m, 0) - log1m
    trans = torch.exp(excl - excl[seg_start][seg]).float()
    w = alpha * trans

    campos = -R.T @ t
    dirs = means - campos
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    deg = colors.shape[1] - 1 if sh_degree is None else sh_degree
    rgb = eval_sh(colors, dirs, deg) if colors.dim() == 3 else colors
    img = torch.zeros(H * W, 3, device=dev, dtype=w.dtype).index_add(0, pix, w[:, None] * rgb[gi])
    acc = torch.zeros(H * W, device=dev, dtype=w.dtype).index_add(0, pix, w)
    if background is not None:
        img = img + (1.0 - acc)[:, None] * background[None]
    return img.view(H, W, 3), acc.view(H, W, 1), means2d, radii


def rasterize_cpu(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree=None,
                  packed=False, backgrounds=None, near_plane=0.01, eps2d=0.3, max_radius=None, **_):
    """``gsplat.rasterization``-compatible CPU rasteriser (one or more cameras, ``packed=False``)."""
    max_radius = float(max_radius or 0.25 * max(width, height))
    renders, alphas, m2s, radii = [], [], [], []
    for c in range(viewmats.shape[0]):
        bg = None if backgrounds is None else backgrounds[c]
        img, a, m2, r = _render_view(means, quats, scales, opacities, colors, viewmats[c], Ks[c], width, height,
                                     sh_degree, bg, near_plane, eps2d, max_radius)
        renders.append(img)
        alphas.append(a)
        m2s.append(m2)
        radii.append(r)
    means2d = m2s[0] if len(m2s) == 1 else torch.cat(m2s, 0)
    info = {"means2d": means2d, "radii": torch.cat(radii, 0), "width": int(width), "height": int(height),
            "n_cameras": int(viewmats.shape[0]), "gaussian_ids": None}
    return torch.stack(renders), torch.stack(alphas), info


def render_model(model, cam, background=(1.0, 1.0, 1.0), return_alpha: bool = False):
    """Render a :class:`~gs4d.gaussians.GaussianModel` with :func:`rasterize_cpu` (no gradients)."""
    import numpy as np

    with torch.no_grad():
        colors = torch.from_numpy(np.concatenate([model.sh_dc[:, None], model.sh_rest], 1).astype(np.float32)) \
            if model.sh_rest is not None and model.sh_rest.shape[1] else \
            torch.from_numpy(model.sh_dc[:, None].astype(np.float32))
        r, a, _ = rasterize_cpu(
            torch.from_numpy(model.means.astype(np.float32)), torch.from_numpy(model.quats.astype(np.float32)),
            torch.from_numpy(np.exp(model.log_scales).astype(np.float32)),
            torch.from_numpy(model.opacities.astype(np.float32)), colors,
            torch.from_numpy(cam.viewmat.astype(np.float32))[None], torch.from_numpy(cam.K.astype(np.float32))[None],
            cam.width, cam.height, backgrounds=torch.tensor(background, dtype=torch.float32)[None],
        )
    img = r[0].numpy().clip(0, 1)
    return (img, a[0, ..., 0].numpy()) if return_alpha else img
