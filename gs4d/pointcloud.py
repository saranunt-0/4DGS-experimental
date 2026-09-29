"""Point cloud -> 3D Gaussians, without the original photos.

A true 3DGS model is *optimised* against the images a scene was captured
from; a point cloud alone has nothing to optimise against.  What we can do is
fit one Gaussian to each local cluster of points so the result is a real,
animatable, exportable 3DGS file:

* **surfel mode** (clouds up to 3 x ``target_count`` points, randomly
  thinned to ``target_count``): one Gaussian per point, oriented by a PCA of its neighbours - flat on bark and leaves,
  elongated along twigs - with a size from the local point spacing;
* **voxel mode** (dense scans): points are merged per voxel and each Gaussian
  takes the mean colour and the *covariance* of its points, so its shape
  follows the local structure.  The voxel size is chosen automatically to hit
  ``target_count`` Gaussians.

Survey / georeferenced coordinates (e.g. UTM in the millions) are recentred
in float64 before anything is cast to float32.  View-dependent colour cannot
be recovered from points, so the result has SH degree 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gaussians import GaussianModel, rgb_to_sh0, logit


@dataclass
class PointCloud:
    points: np.ndarray                 # (N, 3) float64, recentred (see ``offset``)
    colors: np.ndarray | None = None   # (N, 3) in [0, 1]
    normals: np.ndarray | None = None  # (N, 3)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # add back for original coordinates

    def __len__(self) -> int:
        return len(self.points)

    def subset(self, idx) -> "PointCloud":
        return PointCloud(self.points[idx], None if self.colors is None else self.colors[idx],
                          None if self.normals is None else self.normals[idx], self.offset)


def _colors_from_columns(cols: dict) -> np.ndarray | None:
    for names in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(n in cols for n in names):
            c = np.stack([cols[n] for n in names], 1).astype(np.float64)
            if np.issubdtype(np.asarray(cols[names[0]]).dtype, np.integer) or c.max() > 1.0:
                c = c / (65535.0 if c.max() > 255.0 else 255.0)
            return np.clip(c, 0.0, 1.0)
    if all(f"f_dc_{i}" in cols for i in range(3)):
        from .gaussians import sh0_to_rgb
        return np.clip(sh0_to_rgb(np.stack([cols[f"f_dc_{i}"] for i in range(3)], 1)), 0, 1)
    return None


def point_cloud_from_columns(cols: dict, recenter: bool = True) -> PointCloud:
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    ok = np.all(np.isfinite(xyz), axis=1)
    offset = np.median(xyz[ok], axis=0) if recenter and ok.any() else np.zeros(3)
    # only recentre survey-sized coordinates, to a round multiple of 100 units (easy to re-align in a DCC);
    # the remainder stays small enough for float32 (sub-millimetre)
    if recenter:
        offset = np.round(offset / 100.0) * 100.0 * (np.abs(offset) > 1000)
    colors = _colors_from_columns(cols)
    normals = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64) if all(
        n in cols for n in ("nx", "ny", "nz")) else None
    pc = PointCloud(xyz - offset, colors, normals, offset)
    return pc.subset(ok) if not ok.all() else pc


def read_point_cloud(path: str | os.PathLike, recenter: bool = True) -> PointCloud:
    """Read ``.ply`` (any numeric property types), ``.xyz/.txt/.pts/.csv`` (x y z [r g b]), ``.npy``,
    or ``.las/.laz`` (needs ``pip install laspy[lazrs]``)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".ply":
        from .io_ply import read_ply_vertices
        return point_cloud_from_columns(read_ply_vertices(path), recenter)
    if ext in (".las", ".laz"):
        import laspy  # type: ignore
        las = laspy.read(str(path))
        cols = {"x": np.asarray(las.x), "y": np.asarray(las.y), "z": np.asarray(las.z)}
        if hasattr(las, "red"):
            cols.update(red=np.asarray(las.red), green=np.asarray(las.green), blue=np.asarray(las.blue))
        return point_cloud_from_columns(cols, recenter)
    if ext == ".npy":
        a = np.load(path)
        cols = {"x": a[:, 0], "y": a[:, 1], "z": a[:, 2]}
        if a.shape[1] >= 6:
            cols.update(red=a[:, 3], green=a[:, 4], blue=a[:, 5])
        return point_cloud_from_columns(cols, recenter)
    if ext in (".xyz", ".txt", ".pts", ".csv"):
        delim = "," if ext == ".csv" else None
        with open(path) as f:
            first = f.readline()
            skip = 1 if (ext == ".pts" and len(first.split()) == 1) or not first.replace(",", " ").split()[0].lstrip("-").replace(".", "", 1).isdigit() else 0
        a = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2)
        cols = {"x": a[:, 0], "y": a[:, 1], "z": a[:, 2]}
        if a.shape[1] >= 6:
            rgb = a[:, -3:]  # .pts has an intensity column before r g b
            cols.update(red=rgb[:, 0], green=rgb[:, 1], blue=rgb[:, 2])
        return point_cloud_from_columns(cols, recenter)
    raise ValueError(f"unsupported point cloud format {ext!r}")


# ---------------------------------------------------------------------------
def _eigh_frames(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition -> (proper rotation matrices, eigenvalues), largest axis first."""
    w, v = np.linalg.eigh(cov)                     # ascending
    w = w[:, ::-1]
    v = v[:, :, ::-1]
    flip = np.linalg.det(v) < 0
    v[flip, :, 2] *= -1.0
    return v, np.maximum(w, 0.0)


def _voxel_keys(p: np.ndarray, voxel: float) -> np.ndarray:
    ijk = np.floor((p - p.min(0)) / voxel).astype(np.int64)
    dims = ijk.max(0) + 1
    return np.ravel_multi_index(ijk.T, dims)


def choose_voxel_size(points: np.ndarray, target: int, iters: int = 14) -> float:
    """Voxel size whose occupied-voxel count is close to ``target``."""
    ext = np.ptp(points, axis=0).max()
    lo, hi = ext / 4096.0, ext / 8.0
    sample = points if len(points) <= 3_000_000 else points[np.random.default_rng(0).choice(len(points), 3_000_000, replace=False)]
    scale = len(points) / len(sample)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        n = len(np.unique(_voxel_keys(sample, mid))) * (scale if scale > 1 else 1)
        if n > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def pointcloud_to_gaussians(
    pc: PointCloud,
    target_count: int = 800_000,
    mode: str = "auto",
    opacity: float = 0.95,
    size_factor: float = 1.0,
    thickness: float = 0.25,
    k_neighbors: int = 12,
    verbose: bool = True,
) -> GaussianModel:
    """Fit Gaussians to a point cloud (see module docstring).

    ``size_factor`` scales every splat (increase if the result looks holey,
    decrease if it looks blobby); ``thickness`` is the normal-axis size of
    surfels relative to their in-plane size.
    """
    from scipy.spatial import cKDTree

    p = np.asarray(pc.points, np.float64)
    col = pc.colors if pc.colors is not None else np.full((len(p), 3), 0.55)
    n = len(p)
    if mode == "auto":
        # voxel merging only pays off with several points per Gaussian; otherwise subsample to surfels
        mode = "voxel" if n >= 3 * target_count else "surfel"
    if mode == "surfel" and n > target_count:
        sel = np.random.default_rng(0).choice(n, target_count, replace=False)
        p, col, n = p[sel], col[sel], target_count

    if mode == "surfel":
        k = min(k_neighbors, n - 1)
        d, idx = cKDTree(p).query(p, k=k + 1, workers=-1)
        spacing = np.median(d[:, 1:7], axis=1)
        nb = p[idx]                                          # (N, k+1, 3)
        c = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", c, c) / (k + 1)
        R, w = _eigh_frames(cov)
        sd = np.sqrt(w)
        # in-plane sizes follow the neighbourhood spread (elongated along twigs), clamped to the spacing
        s1 = np.clip(sd[:, 0] * 0.9, 0.5 * spacing, 2.0 * spacing)
        s2 = np.clip(sd[:, 1] * 0.9, 0.35 * spacing, s1)
        s3 = np.maximum(thickness * s2, 0.05 * spacing)
        scales = np.stack([s1, s2, s3], 1) * size_factor
        means, rgb, op = p, col, np.full(n, opacity)
        extra = {"n_points": np.ones(n, np.float32)}
        info = f"surfel mode: {len(pc):,} points -> {n:,} Gaussians (median spacing {np.median(spacing):.4g})"
    elif mode == "voxel":
        voxel = choose_voxel_size(p, target_count)
        keys = _voxel_keys(p, voxel)
        uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        m = len(uniq)
        cntf = cnt.astype(np.float64)
        mean = np.stack([np.bincount(inv, weights=p[:, a], minlength=m) for a in range(3)], 1) / cntf[:, None]
        rgb = np.stack([np.bincount(inv, weights=col[:, a], minlength=m) for a in range(3)], 1) / cntf[:, None]
        q = p - mean[inv]
        cov = np.empty((m, 3, 3))
        for a in range(3):
            for b in range(a, 3):
                v = np.bincount(inv, weights=q[:, a] * q[:, b], minlength=m) / cntf
                cov[:, a, b] = v
                cov[:, b, a] = v
        sparse_vox = cnt < 4
        cov[sparse_vox] = np.eye(3) * (voxel / 3.5) ** 2      # too few points for a shape: round splat
        cov += np.eye(3) * (0.04 * voxel) ** 2
        R, w = _eigh_frames(cov)
        # uniform points in a slab of width L have std L/sqrt(12); widen so neighbours overlap
        scales = np.clip(np.sqrt(w) * 1.35, 0.03 * voxel, 0.75 * voxel) * size_factor
        means = mean
        op = np.where(cnt >= 3, opacity, opacity * 0.7)
        extra = {"n_points": cnt.astype(np.float32)}
        info = f"voxel mode: {n:,} points -> {m:,} Gaussians (voxel {voxel:.4g}, median {np.median(cnt):.0f} pts/voxel)"
    else:
        raise ValueError("mode must be 'auto', 'surfel' or 'voxel'")

    from .quaternion import matrix_to_quat

    model = GaussianModel(
        means, matrix_to_quat(R), np.log(np.maximum(scales, 1e-9)), logit(op), rgb_to_sh0(rgb), None, extra
    )
    if verbose:
        print(info)
    return model


def isolate_point_cloud(pc: PointCloud, up="auto", coarse_count: int = 200_000, verbose: bool = True, **isolate_kw):
    """Object-only *points*: fit a coarse splat model, isolate the tree on it
    (gs4d.cleanup.isolate_object), then keep the points near surviving splats.

    Refitting only these points spends the whole Gaussian budget on the tree
    instead of the ground.  Returns ``(tree_points, up_vector, report)``; when a
    ground plane is found its normal is returned as the (gravity) up vector.
    """
    from scipy.spatial import cKDTree

    from .cleanup import isolate_object
    from .skeleton import estimate_up, parse_up

    coarse = pointcloud_to_gaussians(pc, target_count=min(coarse_count, len(pc)), verbose=False)
    if isinstance(up, str) and up == "auto":
        up, how = estimate_up(coarse)
        if verbose:
            print(f"up = {np.round(up, 3).tolist()} ({how})")
    up = parse_up(up)
    keep, rep = isolate_object(coarse, up=up, verbose=verbose, **isolate_kw)
    kept = coarse.subset(keep)
    d, j = cKDTree(kept.means.astype(np.float64)).query(pc.points, k=1, workers=-1)
    sel = d <= 2.0 * kept.scales.max(1)[j]
    if rep.ground:
        n, p0 = rep.ground["normal"], rep.ground["point"]
        sel &= (pc.points - p0) @ n > 0.5 * rep.ground.get("thickness", 0.0)
        up = n
    if verbose:
        print(f"object-only points: {sel.sum():,} / {len(pc):,}")
    return pc.subset(sel), up, rep


SCAN_EXTS = (".ply", ".splat", ".xyz", ".txt", ".pts", ".csv", ".npy", ".las", ".laz")


def extract_from_zip(path: str | os.PathLike, out_dir: str | os.PathLike) -> Path:
    """Extract the largest point-cloud / splat file from a zip archive."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        members = [m for m in z.infolist()
                   if Path(m.filename).suffix.lower() in SCAN_EXTS and not m.filename.startswith("__MACOSX")]
        if not members:
            raise ValueError(f"no point cloud / splat file in {path}: {[m.filename for m in z.infolist()][:20]}")
        best = max(members, key=lambda m: m.file_size)
        print(f"zip: {[(m.filename, m.file_size) for m in members]} -> using {best.filename}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return Path(z.extract(best, out_dir))


def is_point_cloud_file(path: str | os.PathLike) -> bool:
    """True for point clouds, False for 3DGS files (.ply with opacity/scale attributes, .splat)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".splat":
        return False
    if ext != ".ply":
        return ext in SCAN_EXTS
    with open(path, "rb") as f:
        head = f.read(16384).split(b"end_header")[0].decode("ascii", "replace")
    return not ("opacity" in head and "scale_0" in head)


def load_scan(path: str | os.PathLike, max_gaussians: int = 600_000, isolate: bool = True, up="auto",
              work_dir: str | os.PathLike | None = None):
    """Scan (zip / point cloud / 3DGS file) -> object-only Gaussians.

    Point clouds are isolated on a coarse fit first, then only the tree's
    points are fitted with the full ``max_gaussians`` budget.  Returns
    ``(model, up_vector, source_path)``.
    """
    from .cleanup import isolate_object
    from .io_ply import load_gaussians
    from .skeleton import estimate_up, parse_up

    path = Path(path)
    if path.suffix.lower() == ".zip":
        path = extract_from_zip(path, work_dir or path.with_suffix(""))
    if is_point_cloud_file(path):
        pc = read_point_cloud(path)
        print(f"point cloud: {len(pc):,} points, colours: {pc.colors is not None}"
              + (f", recentred by {pc.offset.tolist()}" if np.any(pc.offset) else ""))
        if isolate:
            pc, up_vec, _ = isolate_point_cloud(pc, up=up)
        else:
            coarse = pointcloud_to_gaussians(pc, 200_000, verbose=False)
            up_vec = estimate_up(coarse)[0] if (isinstance(up, str) and up == "auto") else parse_up(up)
        model = pointcloud_to_gaussians(pc, target_count=max_gaussians)
    else:
        model = load_gaussians(path)
        up_vec = estimate_up(model)[0] if (isinstance(up, str) and up == "auto") else parse_up(up)
        if isolate:
            keep, rep = isolate_object(model, up=up_vec)
            model = model.subset(keep)
            if rep.ground:
                up_vec = rep.ground["normal"]
    up_vec = np.asarray(up_vec, np.float64)
    return model, up_vec / np.linalg.norm(up_vec), path
