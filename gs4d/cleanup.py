"""Object-only Gaussian models: remove background, ground and floaters.

Two complementary routes (see ``docs/object_only_gaussians.md``):

* **Prevent** background Gaussians at training time - train on masked images
  with a random background colour + alpha loss (``gs4d.train``).  This is the
  single most effective step.
* **Clean up** an existing splat (this module): every function returns a
  boolean *keep* mask so steps can be inspected and combined.

    keep = isolate_object(model, up="+z")          # one-shot heuristic pipeline
    tree = model.subset(keep)

If you have the training cameras and per-image object masks, prefer
:func:`mask_vote` - it is the 3D analogue of the 2D masks and removes almost
everything that is not the object (a lightweight cousin of FlashSplat /
SA3D-style mask lifting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .gaussians import GaussianModel
from .skeleton import parse_up


# ---------------------------------------------------------------- basic masks
def opacity_mask(model: GaussianModel, min_opacity: float = 0.05) -> np.ndarray:
    return model.opacities >= min_opacity


def scale_mask(model: GaussianModel, max_scale: float | None = None, quantile: float = 0.998,
               max_anisotropy: float | None = 200.0) -> np.ndarray:
    """Drop giant splats (typical sky / background blobs) and extreme needles."""
    s = model.scales
    smax = s.max(1)
    limit = max_scale if max_scale is not None else np.quantile(smax, quantile) * 1.5
    keep = smax <= limit
    if max_anisotropy:
        keep &= smax / np.maximum(s.min(1), 1e-12) <= max_anisotropy
    return keep


def box_mask(model: GaussianModel, lo, hi) -> np.ndarray:
    x = model.means
    return np.all((x >= np.asarray(lo)) & (x <= np.asarray(hi)), axis=1)


def sphere_mask(model: GaussianModel, center, radius: float) -> np.ndarray:
    return np.linalg.norm(model.means - np.asarray(center), axis=1) <= radius


def cylinder_mask(model: GaussianModel, center, up, radius: float, h_min: float = -np.inf, h_max: float = np.inf) -> np.ndarray:
    """Vertical cylinder around ``center`` - the natural crop for a tree."""
    up = parse_up(up)
    d = model.means.astype(np.float64) - np.asarray(center, np.float64)
    h = d @ up
    r = np.linalg.norm(d - np.outer(h, up), axis=1)
    return (r <= radius) & (h >= h_min) & (h <= h_max)


def statistical_outlier_mask(model: GaussianModel, k: int = 16, std_ratio: float = 2.5) -> np.ndarray:
    """Remove points whose mean k-NN distance is unusually large (isolated floaters)."""
    x = model.means.astype(np.float64)
    if len(x) <= k:
        return np.ones(len(x), bool)
    d, _ = cKDTree(x).query(x, k=k + 1)
    md = d[:, 1:].mean(1)
    return md <= md.mean() + std_ratio * md.std()


def largest_component_mask(model: GaussianModel, voxel: float | None = None, keep_fraction: float = 0.02,
                           max_grid: int = 256, min_opacity: float = 0.1, seed: np.ndarray | None = None) -> np.ndarray:
    """Keep the biggest spatially connected blob (plus blobs >= keep_fraction of it).

    Occupancy is voxelised (26-connected) using reasonably opaque Gaussians;
    every Gaussian then inherits the component of its voxel.  With ``seed``
    (boolean mask), components are ranked by how many seed Gaussians they
    contain instead of by size - "the blob(s) that the seed region touches".
    """
    x = model.means.astype(np.float64)
    lo, hi = np.quantile(x, 0.0005, axis=0), np.quantile(x, 0.9995, axis=0)
    ext = np.maximum(hi - lo, 1e-9)
    if voxel is None:
        voxel = ext.max() / 128.0
    voxel = max(voxel, ext.max() / max_grid)
    shape = np.minimum(np.ceil(ext / voxel).astype(int) + 1, max_grid)
    idx = np.clip(np.floor((x - lo) / voxel).astype(int), 0, shape - 1)
    solid = model.opacities >= min_opacity
    occ = np.zeros(shape, bool)
    occ[tuple(idx[solid].T)] = True
    occ = ndimage.binary_closing(occ, iterations=1) | occ
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.ones(len(x), bool)
    comp = lab[tuple(idx.T)]
    if seed is not None and np.any(seed):
        sizes = np.bincount(comp[np.asarray(seed, bool) & solid], minlength=n + 1)[1:]
    else:
        sizes = np.bincount(lab.reshape(-1))[1:]
    if sizes.max() <= 0:
        return np.ones(len(x), bool)
    good = np.flatnonzero(sizes >= keep_fraction * sizes.max()) + 1
    inside = np.all((x >= lo - voxel) & (x <= hi + voxel), axis=1)
    return np.isin(comp, good) & inside


def ground_mask(model: GaussianModel, up="+z", band: float = 0.015, iters: int = 300,
                search_fraction: float = 0.25, keep_radius: float = 0.0, seed: int = 0) -> tuple[np.ndarray, dict]:
    """RANSAC-fit the ground plane under the object and drop Gaussians on/under it.

    ``band`` and ``keep_radius`` are fractions of the object height;
    ``keep_radius`` protects a disc around the trunk base (roots).
    Returns ``(keep_mask, info)``.
    """
    up = parse_up(up)
    rng = np.random.default_rng(seed)
    x = model.means.astype(np.float64)
    h = x @ up
    # robust range: ignore sky shells / floaters above *and* junk below the ground
    lo, hi = np.quantile(h, [0.05, 0.9])
    height = max(hi - lo, 1e-9)
    cand = np.flatnonzero(h <= lo + search_fraction * height)
    if len(cand) < 50:
        return np.ones(len(x), bool), {"found": False}
    pts = x[cand]
    thr = band * height
    sel = np.arange(len(x)) if len(x) <= 20000 else rng.choice(len(x), 20000, replace=False)
    probe = x[sel]
    # weight by opacity^2: dense, opaque structure (a trunk) counts, faint floaters/sky barely do
    pw = model.opacities[sel] ** 2
    pw = pw / max(pw.sum(), 1e-12)
    best_score, best_inl, best = -np.inf, 0, None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if abs(n @ up) < 0.85:  # ground must be roughly horizontal (up may be ~30 deg off)
            continue
        if n @ up < 0:
            n = -n
        dist = (pts - s[0]) @ n
        inl = np.sum(np.abs(dist) < thr)
        # real ground has (almost) nothing underneath it; a plane through the
        # bottom of a crown has the trunk and the ground below it
        below = np.sum(dist < -thr)
        score = inl - 2.0 * below
        if score <= best_score:
            continue
        # a valid ground plane has (almost) the whole scene above it; this rejects
        # planes through the lower crown when little or no real ground is left
        if np.sum(pw[(probe - s[0]) @ n < -thr]) > 0.03:
            continue
        if score > best_score:
            best_score, best_inl, best = score, inl, (n, s[0])
    if best is None or best_inl < 0.15 * len(cand):
        return np.ones(len(x), bool), {"found": False}
    n, p0 = best
    # least-squares refit on the inliers; cut at the measured ground thickness
    signed = (x - p0) @ n
    inl = np.abs(signed) < thr
    q = x[inl]
    c = q.mean(0)
    n_fit = np.linalg.svd(q - c, full_matrices=False)[2][2]
    if n_fit @ n < 0:
        n_fit = -n_fit
    if n_fit @ up > 0.85:
        n, p0 = n_fit, c
    signed = (x - p0) @ n
    sigma = float(np.std(signed[np.abs(signed) < thr]))
    cut = min(max(2.5 * sigma, 0.1 * thr), thr)
    keep = signed > cut
    if keep_radius > 0:
        base = x[np.abs(signed) < thr]
        trunk = np.median(x[(signed > thr) & (signed < 5 * thr)], axis=0) if np.any((signed > thr) & (signed < 5 * thr)) else base.mean(0)
        r = np.linalg.norm((x - trunk) - np.outer((x - trunk) @ n, n), axis=1)
        keep |= (r < keep_radius * height) & (signed > -thr)
    return keep, {"found": True, "normal": n, "point": p0, "inliers": int(best_inl), "thickness": cut}


# ------------------------------------------------------------ mask voting
def mask_vote(model: GaussianModel, cameras, masks, threshold: float = 0.7, min_views: int = 3,
              mask_threshold: float = 0.5) -> np.ndarray:
    """Keep Gaussians whose centre falls inside the object mask in most views.

    ``cameras``: list of :class:`gs4d.render.Camera`; ``masks``: list of HxW
    arrays (1 = object) at the camera resolution.  Occlusion is ignored, which
    is conservative for a convex-ish object seen from all around: background
    Gaussians project outside the mask in most views.
    """
    x = model.means.astype(np.float64)
    inside = np.zeros(len(x))
    seen = np.zeros(len(x))
    for cam, m in zip(cameras, masks):
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
        p = x @ R.T + t
        z = p[:, 2]
        ok = z > 1e-6
        u = cam.fx * p[:, 0] / np.where(ok, z, 1) + cam.cx
        v = cam.fy * p[:, 1] / np.where(ok, z, 1) + cam.cy
        H, W = m.shape
        sx = W / cam.width
        sy = H / cam.height
        ui = np.floor(u * sx).astype(np.int64)
        vi = np.floor(v * sy).astype(np.int64)
        ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        seen += ok
        hit = np.zeros(len(x), bool)
        hit[ok] = m[vi[ok], ui[ok]] >= mask_threshold
        inside += hit
    ratio = inside / np.maximum(seen, 1)
    return (seen >= min_views) & (ratio >= threshold)


def cameras_focus_point(cameras) -> np.ndarray:
    """Least-squares point closest to all camera optical axes (the orbit centre)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for cam in cameras:
        R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
        c = -R.T @ t
        d = R[2]  # camera +z (forward) in world coordinates
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def estimate_object_column(model: GaussianModel, up="+z", grid: int = 96, min_opacity: float = 0.1) -> tuple[np.ndarray, float]:
    """Find the vertical column holding the main standing object.

    Works even when background Gaussians outnumber the object: fit the ground
    plane, histogram the *above-ground* mass (weighted by height) on the
    ground plane, take the peak and grow its connected blob to get a radius.
    """
    from .skeleton import horizontal_basis

    up = parse_up(up)
    solid = model.opacities >= min_opacity
    solid &= scale_mask(model, quantile=0.98)
    m = model.subset(np.flatnonzero(solid)) if solid.sum() > 100 else model
    x = m.means.astype(np.float64)
    gkeep, info = ground_mask(m, up, band=0.01, search_fraction=0.3)
    if info.get("found"):
        n, p0 = info["normal"], info["point"]
        h = (x - p0) @ n
    else:
        n = up
        h = x @ up - np.quantile(x @ up, 0.02)
    above = h > 0.01 * (np.quantile(h, 0.99) - np.quantile(h, 0.01))
    e1, e2 = horizontal_basis(n)
    uv = np.stack([x @ e1, x @ e2], 1)
    lo, hi = np.quantile(uv[above], 0.01, axis=0), np.quantile(uv[above], 0.99, axis=0)
    cell = max((hi - lo).max() / grid, 1e-9)
    ij = np.floor((uv[above] - lo) / cell).astype(int)
    ok = np.all((ij >= 0) & (ij < grid), axis=1)
    hist = np.zeros((grid, grid))
    np.add.at(hist, tuple(ij[ok].T), np.clip(h[above][ok], 0, None))
    dens = ndimage.gaussian_filter(hist, 1.5)
    peak = np.unravel_index(np.argmax(dens), dens.shape)
    blob, _ = ndimage.label(dens > 0.08 * dens[peak])
    region = blob == blob[peak]
    cells = np.argwhere(region)
    c_uv = lo + (np.asarray(peak) + 0.5) * cell
    radius = float(np.max(np.linalg.norm((cells + 0.5) * cell + lo - c_uv, axis=1)) + 2 * cell)
    col = above & (np.linalg.norm(uv - c_uv, axis=1) < radius)
    base_h = np.quantile(x[col] @ up, 0.02) if col.any() else np.quantile(x @ up, 0.02)
    center = c_uv[0] * e1 + c_uv[1] * e2 + base_h * up
    # express the centre along ``up`` so that cylinder_mask(h_min=...) is intuitive
    center = center - (center @ up - base_h) * up
    return center, radius * 1.1


# --------------------------------------------------------- one-shot pipeline
@dataclass
class CleanupReport:
    steps: list = field(default_factory=list)
    center: np.ndarray | None = None
    radius: float | None = None
    ground: dict | None = None  # RANSAC plane (normal is a good gravity "up" estimate)

    def add(self, name: str, keep: np.ndarray, total: int):
        self.steps.append((name, int(keep.sum()), total))

    def __str__(self) -> str:
        lines = ["object isolation:"]
        for name, k, n in self.steps:
            lines.append(f"  {name:<28s} kept {k:>9,} / {n:,}  ({100.0 * k / max(n, 1):5.1f}%)")
        return "\n".join(lines)


def isolate_object(
    model: GaussianModel,
    up="+z",
    center=None,
    radius: float | None = None,
    min_opacity: float = 0.05,
    remove_ground: bool = True,
    ground_band: float = 0.015,
    sor_std: float = 4.0,
    keep_fraction: float = 0.02,
    initial_keep: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, CleanupReport]:
    """Heuristic object extraction for a scene that is centred on one object.

    Steps: opacity prune -> giant-splat prune -> cylinder crop around the
    object (``center``/``radius`` in model units; auto = densest opaque region)
    -> ground-plane removal -> statistical outlier removal -> largest component.
    ``initial_keep`` (e.g. from :func:`mask_vote`) is AND-ed in first.
    """
    up = parse_up(up)
    n = len(model)
    rep = CleanupReport()
    keep = np.ones(n, bool) if initial_keep is None else np.asarray(initial_keep, bool).copy()
    if initial_keep is not None:
        rep.add("initial mask (e.g. mask vote)", keep, n)
    keep &= opacity_mask(model, min_opacity)
    rep.add("opacity >= %.2f" % min_opacity, keep, n)
    keep &= scale_mask(model)
    rep.add("no giant / needle splats", keep, n)

    if remove_ground:
        # fit the ground on the whole (pre-crop) scene: many more ground samples
        sub = np.flatnonzero(keep)
        g, info = ground_mask(model.subset(sub), up, band=ground_band, search_fraction=0.3)
        if info.get("found"):
            k2 = np.zeros(n, bool)
            k2[sub[g]] = True
            keep &= k2
            rep.ground = info
            rep.add("ground plane removed", keep, n)

    hard_crop = radius is not None
    if center is None or radius is None:
        c_est, r_est = estimate_object_column(model.subset(np.flatnonzero(keep)), up)
        center = c_est if center is None else center
        radius = r_est if radius is None else radius
    center = np.asarray(center, np.float64)
    # automatic radius: only a *seed*; the object is grown by connectivity inside a 2x safety cylinder
    seed = cylinder_mask(model, center, up, radius)
    keep &= seed if hard_crop else cylinder_mask(model, center, up, 2.0 * radius)
    rep.center, rep.radius = center, radius
    rep.add(("cylinder crop r=%.3g" if hard_crop else "safety cylinder r=%.3g") % (radius if hard_crop else 2 * radius), keep, n)

    sub = np.flatnonzero(keep)
    if len(sub) > 32:
        s = statistical_outlier_mask(model.subset(sub), std_ratio=sor_std)
        k2 = np.zeros(n, bool)
        k2[sub[s]] = True
        keep &= k2
        rep.add("statistical outliers", keep, n)
        sub = np.flatnonzero(keep)
        c = largest_component_mask(model.subset(sub), keep_fraction=keep_fraction, seed=seed[sub])
        k2 = np.zeros(n, bool)
        k2[sub[c]] = True
        keep &= k2
        rep.add("connected object (seeded)", keep, n)
    if verbose:
        print(rep)
    return keep, rep
