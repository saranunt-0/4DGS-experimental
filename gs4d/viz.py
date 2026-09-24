"""Notebook helpers: image grids, MP4 previews, skeleton overlays, wind plots."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

from .render import Camera, render, to_uint8


def show_grid(images, titles=None, cols: int | None = None, size: float = 4.0):
    import matplotlib.pyplot as plt

    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes.ravel()[i]
        ax.imshow(np.clip(img, 0, 1) if np.asarray(img).dtype != np.uint8 else img)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def render_frames(frames, cam: Camera, background=(1, 1, 1), every: int = 1, backend: str = "auto",
                  progress: bool = True, **kw) -> list[np.ndarray]:
    """Render ``frames`` (a WindAnimation or list of models) to uint8 images."""
    out = []
    idx = list(range(0, len(frames), every))
    t0 = time.time()
    for k, i in enumerate(idx):
        out.append(to_uint8(render(frames[i], cam, background=background, backend=backend, **kw)))
        if progress and (k == 0 or (k + 1) % max(1, len(idx) // 8) == 0 or k == len(idx) - 1):
            el = time.time() - t0
            print(f"  rendered {k + 1}/{len(idx)} frames ({el:.0f}s, ~{el / (k + 1) * (len(idx) - k - 1):.0f}s left)")
    return out


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: float = 24.0, loops: int = 1) -> Path:
    import imageio.v2 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    pad = [(0, h % 2), (0, w % 2), (0, 0)]  # libx264 wants even sizes
    with iio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
                        pixelformat="yuv420p") as wr:
        for _ in range(loops):
            for f in frames:
                wr.append_data(np.pad(f, pad, mode="edge"))
    return path


def html_video(path: str | Path, width: int = 480):
    from IPython.display import HTML

    data = base64.b64encode(Path(path).read_bytes()).decode()
    return HTML(f'<video width="{width}" controls autoplay loop muted playsinline>'
                f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>')


def project(points: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
    p = np.asarray(points, np.float64) @ R.T + t
    z = p[:, 2]
    uv = np.stack([cam.fx * p[:, 0] / np.maximum(z, 1e-6) + cam.cx, cam.fy * p[:, 1] / np.maximum(z, 1e-6) + cam.cy], 1)
    return uv, z > 1e-6


def plot_skeleton(image: np.ndarray, skel, cam: Camera, title: str = "skeleton", ax=None):
    """Overlay the joint hierarchy (parent -> child segments) on a render."""
    import matplotlib.pyplot as plt

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    uv, ok = project(skel.pivots, cam)
    tips = skel.pivots + skel.directions * skel.lengths[:, None]
    uv_tip, ok_tip = project(tips, cam)
    depth_norm = skel.depth / max(skel.depth.max(), 1)
    cmap = plt.get_cmap("plasma")
    for j in range(skel.num_joints):
        p = skel.parents[j]
        a = uv[p] if p >= 0 else uv[j]
        if ok[j] and (p < 0 or ok[p]):
            ax.plot([a[0], uv[j][0]], [a[1], uv[j][1]], color=cmap(depth_norm[j]), lw=1.2)
        if ok[j] and ok_tip[j] and not np.any(skel.parents == j):
            ax.plot([uv[j][0], uv_tip[j][0]], [uv[j][1], uv_tip[j][1]], color=cmap(depth_norm[j]), lw=0.8)
    ax.scatter(uv[ok][:, 0], uv[ok][:, 1], s=3, c=[cmap(d) for d in depth_norm[ok]])
    ax.set_title(title)
    ax.axis("off")
    if own:
        plt.show()


def plot_wind(anim):
    """Wind speed at the crown, and how far a trunk point, a branch and an outer leaf move."""
    import matplotlib.pyplot as plt

    from .skeleton import horizontal_basis

    skel = anim.skel
    t = anim.times
    x = anim.model.means.astype(np.float64)
    up = skel.up
    h = x @ up
    base = x[h <= np.quantile(h, 0.02)].mean(0)
    rel = x - base
    radial = np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)
    hn = (h - h.min()) / max(np.ptp(h), 1e-9)
    leaf = skel.leafness
    wood = (leaf < 0.2) & (skel.bind >= 0)
    probes = {}
    trunk = np.flatnonzero(wood & (np.abs(hn - 0.45) < 0.05))
    if len(trunk):
        probes["trunk (mid-height)"] = trunk[np.argmin(radial[trunk])]
    branch = np.flatnonzero(wood & (hn > 0.5))
    if len(branch):
        probes["branch"] = branch[np.argmin(np.abs(radial[branch] - 0.5 * radial.max()))]
    leaves = np.flatnonzero(leaf > 0.8)
    if len(leaves):
        probes["outer leaf"] = leaves[np.argmax(radial[leaves] + 0.3 * rel[leaves] @ up)]
    e1, e2 = horizontal_basis(up)
    a = np.deg2rad(anim.p.direction_deg)
    wdir = np.cos(a) * e1 + np.sin(a) * e2
    idx = np.array(list(probes.values()), int)
    disp = np.array([(anim.frame(i).means[idx] - x[idx]) @ wdir for i in range(len(anim))]) * anim.to_m * 100
    crown = x[leaves].mean(0) if len(leaves) else x.mean(0)
    v = np.array([np.linalg.norm(anim.wind((crown * anim.to_m)[None], ti)[0]) for ti in t])

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    axes[0].plot(t, v, color="tab:blue")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("wind at crown [m/s]")
    axes[0].set_title("gusty wind (gust fronts travel downwind)")
    for k, name in enumerate(probes):
        axes[1].plot(t, disp[:, k], label=name)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("downwind displacement [cm]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("response: stiff trunk, swaying branches, fluttering leaves")
    plt.tight_layout()
    plt.show()
