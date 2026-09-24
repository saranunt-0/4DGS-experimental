"""Minimal COLMAP sparse-model reader (binary or text), numpy only.

Returns cameras in the OpenCV / gsplat convention (world-to-camera
``viewmat``), which is exactly COLMAP's convention.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .quaternion import quat_to_matrix
from .render import Camera

_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16}
_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
    11: "RAD_TAN_THIN_PRISM_FISHEYE",
}


@dataclass
class ColmapScene:
    cameras: list          # list[Camera], one per registered image (sorted by name)
    image_names: list      # list[str]
    points: np.ndarray     # (P, 3)
    colors: np.ndarray     # (P, 3) in [0, 1]
    camera_models: list    # COLMAP model name per image


def _intrinsics(model: str, params: np.ndarray):
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        return params[0], params[0], params[1], params[2]
    return params[0], params[1], params[2], params[3]


def _read_bin(path: Path):
    cams = {}
    with open(path / "cameras.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            k = _MODEL_NUM_PARAMS[mid]
            params = np.array(struct.unpack("<" + "d" * k, f.read(8 * k)))
            cams[cid] = (_MODEL_NAMES[mid], int(w), int(h), params)
    images = []
    with open(path / "images.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * npts, 1)
            images.append((name.decode(), q, t, cid))
    pts, cols = [], []
    with open(path / "points3D.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            data = f.read(43)
            xyz = struct.unpack("<3d", data[8:32])
            rgb = struct.unpack("<3B", data[32:35])
            (tl,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * tl, 1)
            pts.append(xyz)
            cols.append(rgb)
    return cams, images, np.array(pts).reshape(-1, 3), np.array(cols).reshape(-1, 3) / 255.0


def _read_txt(path: Path):
    cams = {}
    for line in (path / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        cams[int(tok[0])] = (tok[1], int(tok[2]), int(tok[3]), np.array(list(map(float, tok[4:]))))
    images = []
    lines = [ln for ln in (path / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        tok = lines[i].split()
        if len(tok) < 10:
            continue
        images.append((tok[9], np.array(list(map(float, tok[1:5]))), np.array(list(map(float, tok[5:8]))), int(tok[8])))
    pts, cols = [], []
    for line in (path / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        pts.append(list(map(float, tok[1:4])))
        cols.append(list(map(int, tok[4:7])))
    return cams, images, np.array(pts).reshape(-1, 3), np.array(cols).reshape(-1, 3) / 255.0


def read_colmap(model_dir: str | Path) -> ColmapScene:
    """Read ``cameras/images/points3D`` (.bin or .txt) from a COLMAP model folder."""
    p = Path(model_dir)
    if (p / "cameras.bin").exists():
        cams, images, pts, cols = _read_bin(p)
    elif (p / "cameras.txt").exists():
        cams, images, pts, cols = _read_txt(p)
    else:
        raise FileNotFoundError(f"no COLMAP model in {p}")
    images.sort(key=lambda im: im[0])
    out_cams, names, models = [], [], []
    for name, q, t, cid in images:
        model, w, h, params = cams[cid]
        fx, fy, cx, cy = _intrinsics(model, params)
        V = np.eye(4)
        V[:3, :3] = quat_to_matrix(q)
        V[:3, 3] = t
        out_cams.append(Camera(V, fx, fy, cx, cy, w, h))
        names.append(name)
        models.append(model)
    return ColmapScene(out_cams, names, pts, cols, models)


def write_colmap_text(scene_dir: str | Path, cameras: list, names: list, points: np.ndarray | None = None,
                      colors: np.ndarray | None = None) -> Path:
    """Write PINHOLE cameras + poses as a COLMAP text model (useful for synthetic data)."""
    from .quaternion import matrix_to_quat

    d = Path(scene_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cameras.txt", "w") as f:
        for i, c in enumerate(cameras):
            f.write(f"{i + 1} PINHOLE {c.width} {c.height} {c.fx} {c.fy} {c.cx} {c.cy}\n")
    with open(d / "images.txt", "w") as f:
        for i, (c, n) in enumerate(zip(cameras, names)):
            q = matrix_to_quat(c.viewmat[:3, :3])
            t = c.viewmat[:3, 3]
            f.write(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i + 1} {n}\n\n")
    with open(d / "points3D.txt", "w") as f:
        if points is not None:
            cols = np.zeros_like(points) if colors is None else colors
            for i, (p, c) in enumerate(zip(points, cols)):
                r, g, b = (np.clip(c, 0, 1) * 255).astype(int)
                f.write(f"{i + 1} {p[0]} {p[1]} {p[2]} {r} {g} {b} 0\n")
    return d
