"""Tree skeletons for Gaussian splats.

A :class:`TreeSkeleton` is a forest of rotational joints.  Every joint ``j``
rotates the geometry bound to it (and all of its descendants) about its
``pivot``; the rotation is accumulated down the hierarchy exactly like forward
kinematics on a character rig.  Because a child's transform is always
``parent_transform o local_rotation_about(child_pivot)``, positions stay
continuous across joints without explicit skin-weight blending.

Two ways to obtain a skeleton:

* :func:`gs4d.procedural_tree.generate_tree` returns the exact skeleton it grew.
* :func:`extract_skeleton` recovers an approximate branch hierarchy from any
  captured tree (level-set / geodesic-slice skeletonisation, the classic
  approach used for laser-scanned trees: Verroust & Lazarus 2000, Xu et al.
  2007 "Knowledge and heuristic-based modeling of laser-scanned trees").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from .gaussians import GaussianModel


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane orthogonal to ``up``.

    ``e1`` is the world axis most orthogonal to ``up`` (so for ``up=+Z`` the
    basis is ``(+X, +Y)`` and for ``up=+Y`` it is ``(+X, -Z)``).
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    axes = np.eye(3)
    ref = axes[np.argmin(np.abs(axes @ up))]
    e1 = ref - np.dot(ref, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2


@dataclass
class TreeSkeleton:
    pivots: np.ndarray      # (J, 3) rest-pose rotation centres
    parents: np.ndarray     # (J,)   parent joint, -1 = attached to the static world
    directions: np.ndarray  # (J, 3) unit direction of the segment each joint drives
    lengths: np.ndarray     # (J,)   segment length
    bind: np.ndarray        # (N,)   joint per Gaussian, -1 = static (e.g. ground)
    leafness: np.ndarray    # (N,)   0 = wood, 1 = leaf (drives flutter)
    up: np.ndarray          # (3,)   world up vector
    height: float           # characteristic tree height (length unit of the model)

    def __post_init__(self) -> None:
        self.pivots = np.asarray(self.pivots, np.float64).reshape(-1, 3)
        self.parents = np.asarray(self.parents, np.int64).reshape(-1)
        d = np.asarray(self.directions, np.float64).reshape(-1, 3)
        self.directions = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        self.lengths = np.asarray(self.lengths, np.float64).reshape(-1)
        self.bind = np.asarray(self.bind, np.int64).reshape(-1)
        self.leafness = np.clip(np.asarray(self.leafness, np.float64).reshape(-1), 0.0, 1.0)
        self.up = np.asarray(self.up, np.float64) / np.linalg.norm(self.up)
        self.height = float(self.height)
        J = len(self.pivots)
        if np.any(self.parents >= J) or np.any(self.bind >= J):
            raise ValueError("skeleton index out of range")
        self.depth = self._compute_depth()
        order = np.argsort(self.depth, kind="stable")
        self.levels = [order[self.depth[order] == d] for d in range(int(self.depth.max()) + 1)] if J else []

    # ------------------------------------------------------------ topology
    @property
    def num_joints(self) -> int:
        return int(len(self.pivots))

    def _compute_depth(self) -> np.ndarray:
        J = len(self.parents)
        depth = np.full(J, -1, np.int64)
        roots = self.parents < 0
        depth[roots] = 0
        for _ in range(J + 1):
            todo = depth < 0
            if not np.any(todo):
                break
            pd = depth[self.parents[todo]]
            ready = pd >= 0
            idx = np.flatnonzero(todo)[ready]
            depth[idx] = pd[ready] + 1
        if np.any(depth < 0):
            raise ValueError("skeleton parents contain a cycle")
        return depth

    def ancestor_matrix(self, joint_of_item: np.ndarray) -> sparse.csr_matrix:
        """Sparse (J x M) matrix with A[j, m] = 1 iff joint j is item m's joint or an ancestor of it."""
        joint_of_item = np.asarray(joint_of_item, np.int64)
        rows, cols = [], []
        cur = joint_of_item.copy()
        items = np.arange(len(cur))
        while True:
            ok = cur >= 0
            if not np.any(ok):
                break
            rows.append(cur[ok])
            cols.append(items[ok])
            cur = np.where(ok, self.parents[np.maximum(cur, 0)], -1)
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
        else:
            r = c = np.zeros(0, np.int64)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(self.num_joints, len(joint_of_item)))

    def summary(self) -> str:
        leaf_frac = float(np.mean(self.leafness > 0.5)) if len(self.leafness) else 0.0
        return (
            f"TreeSkeleton: {self.num_joints} joints, max depth {int(self.depth.max()) if self.num_joints else 0}, "
            f"{len(self.bind):,} bound Gaussians ({np.mean(self.bind < 0) * 100:.1f}% static, "
            f"{leaf_frac * 100:.1f}% leaf), height {self.height:.3f}"
        )


# --------------------------------------------------------------------------
# Up-axis helpers
# --------------------------------------------------------------------------
AXES = {
    "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "+y": (0, 1, 0), "-y": (0, -1, 0),
    "+z": (0, 0, 1), "-z": (0, 0, -1),
}


def parse_up(up) -> np.ndarray:
    if isinstance(up, str):
        return np.asarray(AXES[up.lower()], np.float64)
    v = np.asarray(up, np.float64)
    return v / np.linalg.norm(v)


def guess_up_axis(model: GaussianModel, return_scores: bool = False):
    """Heuristic up-axis guess for a single tree (one of +x, -x, +y, -y, +z, -z).

    Two cues per signed axis:

    * geometry - a tree stands on a thin trunk and carries a wide crown, so
      the bottom band is much thinner *and much lighter* than the widest /
      heaviest band (the mass term matters for broad trees whose drooping
      branches reach down beside a short trunk, e.g. old oaks);
    * colour   - bark-brown at the bottom, leaf-green at the top.

    The colour cue is very reliable for green foliage and is ignored when the
    colours carry no signal.  Always check the preview.
    """
    x = model.means.astype(np.float64)
    w = model.opacities > 0.2
    sub = model.subset(np.flatnonzero(w)) if w.sum() > 100 else model
    x = sub.means.astype(np.float64)
    leaf = color_leafness(sub)
    bark = bark_score(sub)
    c = np.median(x, axis=0)
    scores = {}
    for name, a in AXES.items():
        a = np.asarray(a, np.float64)
        h = (x - c) @ a
        lo, hi = np.quantile(h, [0.005, 0.995])
        t = (h - lo) / max(hi - lo, 1e-9)
        radial = np.linalg.norm((x - c) - np.outer(h, a), axis=1)
        bands = [radial[(t >= b) & (t < b + 0.12)] for b in np.arange(0.0, 0.96, 0.12)]
        # lower-quartile radius: the trunk core, not the branch tips hanging down beside it
        spreads = np.array([np.quantile(b, 0.25) if len(b) > 20 else np.nan for b in bands])
        counts = np.array([len(b) for b in bands], np.float64)
        if np.isnan(spreads[0]) or np.all(np.isnan(spreads)):
            geo = -10.0
        else:
            geo = float(np.log(np.nanmax(spreads) / (spreads[0] + 1e-9)) + np.log(counts.max() / (counts[0] + 1.0)))
        top, bot = t > 0.6, t < 0.2
        col = 0.0
        if top.any() and bot.any():
            col = float(leaf[top].mean() - leaf[bot].mean() + bark[bot].mean() - bark[top].mean())
        scores[name] = (geo, col)
    colour_informative = max(v[1] for v in scores.values()) > 0.15
    total = {k: g + (2.0 * c_ if colour_informative else 0.0) for k, (g, c_) in scores.items()}
    best = max(total, key=total.get)
    return (best, scores) if return_scores else best


# --------------------------------------------------------------------------
# Leafness
# --------------------------------------------------------------------------
def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, np.float64), 0, 1)
    mx = rgb.max(1)
    mn = rgb.min(1)
    d = mx - mn
    h = np.zeros(len(rgb))
    nz = d > 1e-9
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    i = nz & (mx == r)
    h[i] = ((g[i] - b[i]) / d[i]) % 6
    i = nz & (mx == g) & ~(mx == r)
    h[i] = (b[i] - r[i]) / d[i] + 2
    i = nz & (mx == b) & ~(mx == r) & ~(mx == g)
    h[i] = (r[i] - g[i]) / d[i] + 4
    h = h / 6.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, s, mx], 1)


def _smoothstep(x, a, b):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def color_leafness(model: GaussianModel, hue_center_deg: float = 95.0, hue_width_deg: float = 55.0) -> np.ndarray:
    """0..1 score that a Gaussian is foliage, from its base colour.

    Green-ish hue (default 40..150 deg) with some saturation => leaf.  Change
    ``hue_center_deg`` for autumn foliage (e.g. 35 for orange/yellow).
    """
    hsv = rgb_to_hsv(model.rgb)
    hue = hsv[:, 0] * 360.0
    dist = np.abs((hue - hue_center_deg + 180.0) % 360.0 - 180.0)
    hue_score = 1.0 - _smoothstep(dist, hue_width_deg * 0.6, hue_width_deg)
    sat_score = _smoothstep(hsv[:, 1], 0.08, 0.22)
    return hue_score * sat_score


def bark_score(model: GaussianModel) -> np.ndarray:
    """0..1 score for bark-like colours: dark browns (hue 10..50 deg) or dull greys."""
    hsv = rgb_to_hsv(model.rgb)
    hue = hsv[:, 0] * 360.0
    brown = (1.0 - _smoothstep(np.abs(hue - 30.0), 15.0, 30.0)) * (1.0 - _smoothstep(hsv[:, 2], 0.45, 0.7))
    grey = 1.0 - _smoothstep(hsv[:, 1], 0.08, 0.18)
    return np.maximum(brown, grey * (1.0 - _smoothstep(hsv[:, 2], 0.6, 0.85)))


# --------------------------------------------------------------------------
# Skeleton extraction from an arbitrary Gaussian tree
# --------------------------------------------------------------------------
def _voxel_downsample(x: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = np.floor((x - x.min(0)) / voxel).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cent = np.zeros((len(counts), 3))
    np.add.at(cent, inv, x)
    cent /= counts[:, None]
    return cent, inv, counts


def _knn_graph(pts: np.ndarray, k: int, max_edge: float) -> sparse.csr_matrix:
    n = len(pts)
    k = min(k, n - 1)
    d, idx = cKDTree(pts).query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    w = d[:, 1:].reshape(-1)
    keep = w <= max_edge
    g = sparse.csr_matrix((w[keep] + 1e-9, (rows[keep], cols[keep])), shape=(n, n))
    g = g.maximum(g.T)
    # stitch disconnected pieces to the largest component so every voxel is reachable
    ncomp, lab = csgraph.connected_components(g, directed=False)
    if ncomp > 1:
        sizes = np.bincount(lab)
        main = np.argmax(sizes)
        main_idx = np.flatnonzero(lab == main)
        tree = cKDTree(pts[main_idx])
        extra_r, extra_c, extra_w = [], [], []
        for comp in np.flatnonzero(np.arange(ncomp) != main):
            members = np.flatnonzero(lab == comp)
            dd, ii = tree.query(pts[members], k=1)
            j = np.argmin(dd)
            extra_r.append(members[j])
            extra_c.append(main_idx[ii[j]])
            extra_w.append(dd[j] * 2.0 + 1e-9)  # discourage paths through gaps
        e = sparse.csr_matrix((extra_w, (extra_r, extra_c)), shape=(n, n))
        g = g.maximum(e).maximum(e.T)
    return g.tocsr()


def extract_skeleton(
    model: GaussianModel,
    up="+z",
    n_slices: int = 36,
    voxel_size: float | None = None,
    k_neighbors: int = 10,
    leafness: np.ndarray | None = None,
    min_opacity: float = 0.1,
    pin_below: float = 0.0,
) -> TreeSkeleton:
    """Recover a branch hierarchy from a (cleaned, object-only) Gaussian tree.

    1. voxel-downsample the Gaussian centres and connect them in a k-NN graph;
    2. geodesic (Dijkstra) distance from the root = lowest trunk voxel;
    3. slice the geodesic distance into ``n_slices`` level sets, split each
       slice into connected components -> skeleton joints;
    4. each joint's parent is the joint its shortest paths come from.

    ``pin_below`` (fraction of height) keeps Gaussians below that height static
    (useful when some ground / roots are still attached).
    """
    up = parse_up(up)
    x_all = model.means.astype(np.float64)
    sel = model.opacities >= min_opacity
    if sel.sum() < 50:
        sel = np.ones(len(model), bool)
    x = x_all[sel]
    h = x @ up
    h0, h1 = np.quantile(h, [0.002, 0.998])
    height = max(h1 - h0, 1e-6)
    if voxel_size is None:
        voxel_size = height / 90.0
    vox, _, counts = _voxel_downsample(x, voxel_size)
    g = _knn_graph(vox, k_neighbors, max_edge=voxel_size * 3.5)

    # root: most central voxel within the lowest 3% of the tree
    vh = vox @ up
    low = vh <= h0 + 0.03 * height
    if low.sum() == 0:
        low = vh <= np.quantile(vh, 0.02)
    low_idx = np.flatnonzero(low)
    base_center = np.average(vox[low_idx], axis=0, weights=counts[low_idx])
    root = low_idx[np.argmin(np.linalg.norm(vox[low_idx] - base_center, axis=1))]

    dist, pred = csgraph.dijkstra(g, directed=False, indices=root, return_predecessors=True)
    finite = np.isfinite(dist)
    dist[~finite] = dist[finite].max()
    width = dist.max() / n_slices + 1e-12
    slice_id = np.minimum((dist / width).astype(np.int64), n_slices - 1)

    # connected components inside each slice (edges between same-slice voxels only)
    coo = g.tocoo()
    same = slice_id[coo.row] == slice_id[coo.col]
    gs = sparse.csr_matrix((coo.data[same], (coo.row[same], coo.col[same])), shape=g.shape)
    _, node_of_voxel = csgraph.connected_components(gs, directed=False)
    uniq, node_of_voxel = np.unique(node_of_voxel, return_inverse=True)
    J = len(uniq)

    # parent of each node = majority node of its voxels' Dijkstra predecessors
    has_pred = (pred >= 0) & (node_of_voxel[np.maximum(pred, 0)] != node_of_voxel)
    child_nodes = node_of_voxel[has_pred]
    parent_nodes = node_of_voxel[pred[has_pred]]
    parents = np.full(J, -1, np.int64)
    if len(child_nodes):
        pair = sparse.csr_matrix((np.ones(len(child_nodes)), (child_nodes, parent_nodes)), shape=(J, J))
        pair = pair.tolil()
        for j in range(J):
            row = pair.rows[j]
            if row:
                vals = pair.data[j]
                parents[j] = row[int(np.argmax(vals))]
    root_node = node_of_voxel[root]
    parents[root_node] = -1
    # break any accidental cycles by forcing parents to lie in a lower slice
    node_slice = np.zeros(J, np.int64)
    np.maximum.at(node_slice, node_of_voxel, slice_id)
    node_min_slice = np.full(J, n_slices + 1, np.int64)
    np.minimum.at(node_min_slice, node_of_voxel, slice_id)
    bad = (parents >= 0) & (node_min_slice[np.maximum(parents, 0)] >= node_min_slice)
    parents[bad & (np.arange(J) != root_node)] = -1
    orphans = np.flatnonzero((parents < 0) & (np.arange(J) != root_node))
    if len(orphans):
        node_cent = np.zeros((J, 3))
        np.add.at(node_cent, node_of_voxel, vox)
        node_cent /= np.bincount(node_of_voxel, minlength=J)[:, None]
        for j in orphans:
            cand = np.flatnonzero(node_min_slice < node_min_slice[j])
            parents[j] = cand[np.argmin(np.linalg.norm(node_cent[cand] - node_cent[j], axis=1))] if len(cand) else root_node

    # pivots at the junction with the parent; direction towards the node's far end
    pivots = np.zeros((J, 3))
    far = np.zeros((J, 3))
    boundary = has_pred
    bsum = np.zeros((J, 3))
    bcnt = np.zeros(J)
    bnodes = node_of_voxel[boundary]
    mid = 0.5 * (vox[boundary] + vox[pred[boundary]])
    np.add.at(bsum, bnodes, mid)
    np.add.at(bcnt, bnodes, 1)
    csum = np.zeros((J, 3))
    ccnt = np.zeros(J)
    np.add.at(csum, node_of_voxel, vox * dist[:, None] ** 2)
    np.add.at(ccnt, node_of_voxel, dist**2)
    anyv = np.zeros((J, 3))
    np.add.at(anyv, node_of_voxel, vox)
    nv = np.bincount(node_of_voxel, minlength=J)
    anyv /= np.maximum(nv, 1)[:, None]
    pivots[:] = np.where(bcnt[:, None] > 0, bsum / np.maximum(bcnt, 1)[:, None], anyv)
    pivots[root_node] = vox[root]
    far[:] = np.where(ccnt[:, None] > 0, csum / np.maximum(ccnt, 1e-12)[:, None], anyv)
    dirs = far - pivots
    lens = np.linalg.norm(dirs, axis=1)
    for d in range(3):  # fall back to the parent's direction (or up) for degenerate nodes
        weak = lens < voxel_size * 0.25
        if not np.any(weak):
            break
        p = parents[weak]
        dirs[weak] = np.where((p >= 0)[:, None], dirs[np.maximum(p, 0)], up)
        lens = np.linalg.norm(dirs, axis=1)
    lengths = np.maximum(width, lens)

    # bind every Gaussian (including low-opacity ones) to the nearest voxel's node
    _, nearest = cKDTree(vox).query(x_all, k=1)
    bind = node_of_voxel[nearest].astype(np.int64)
    if pin_below > 0:
        bind[(x_all @ up) < h0 + pin_below * height] = -1

    if leafness is None:
        leafness = auto_leafness(model, parents, bind, counts_per_node=np.bincount(node_of_voxel, weights=counts, minlength=J))

    return TreeSkeleton(pivots, parents, dirs, lengths, bind, leafness, up, height)


def subtree_sizes(parents: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum of ``weights`` over each joint's subtree."""
    parents = np.asarray(parents, np.int64)
    tmp = TreeSkeleton.__new__(TreeSkeleton)
    tmp.parents = parents
    depth = TreeSkeleton._compute_depth(tmp)
    total = np.asarray(weights, np.float64).copy()
    for d in range(int(depth.max()), 0, -1):
        idx = np.flatnonzero(depth == d)
        np.add.at(total, parents[idx], total[idx])
    return total


def auto_leafness(
    model: GaussianModel,
    parents: np.ndarray,
    bind: np.ndarray,
    counts_per_node: np.ndarray | None = None,
    hue_center_deg: float = 95.0,
) -> np.ndarray:
    """Combine a colour cue (green) with a structural cue (thin, peripheral joints)."""
    if counts_per_node is None:
        counts_per_node = np.bincount(np.maximum(bind, 0), minlength=len(parents)).astype(np.float64)
    sub = subtree_sizes(parents, counts_per_node)
    s = np.sqrt(sub / max(sub.max(), 1e-9))  # pipe-model "thickness" 0..1
    thin = 1.0 - _smoothstep(s[np.maximum(bind, 0)], 0.15, 0.45)
    colour = color_leafness(model, hue_center_deg=hue_center_deg)
    bark = bark_score(model) * (1.0 - colour)
    leaf = np.maximum(colour * (1.0 - _smoothstep(s[np.maximum(bind, 0)], 0.5, 0.8)), 0.85 * thin * (1.0 - bark))
    leaf[bind < 0] = 0.0
    return leaf


def subset_skeleton(skel: TreeSkeleton, keep: np.ndarray) -> TreeSkeleton:
    """Keep the joints but restrict the per-Gaussian binding to ``keep``.

    ``keep`` may be a boolean mask longer than the binding (e.g. over a scene
    = [tree, extra splats]); the extra splats become static (bind = -1).
    """
    keep = np.asarray(keep)
    bind, leaf = skel.bind, skel.leafness
    if keep.dtype == bool and len(keep) > len(bind):
        extra = len(keep) - len(bind)
        bind = np.concatenate([bind, np.full(extra, -1)])
        leaf = np.concatenate([leaf, np.zeros(extra)])
    return TreeSkeleton(skel.pivots, skel.parents, skel.directions, skel.lengths,
                        bind[keep], leaf[keep], skel.up, skel.height)


def estimate_up(model: GaussianModel, seed: int = 0) -> tuple[np.ndarray, str]:
    """Up vector for a captured scene or an isolated tree.

    If the scene contains a dominant plane (the ground of an outdoor capture)
    its normal - oriented towards the side holding most of the splats - is
    used; otherwise the tree shape/colour heuristic :func:`guess_up_axis`.
    Returns ``(up_vector, how)``.
    """
    rng = np.random.default_rng(seed)
    w = model.opacities > 0.2
    x = model.means[w if w.sum() > 100 else slice(None)].astype(np.float64)
    if len(x) > 60000:
        x = x[rng.choice(len(x), 60000, replace=False)]
    lo, hi = np.quantile(x, 0.1, axis=0), np.quantile(x, 0.9, axis=0)  # robust to sky shells
    thr = 0.004 * np.linalg.norm(hi - lo)
    best_n, best_inl, best_p, best_score = None, 0, None, -np.inf
    for _ in range(600):
        s = x[rng.choice(len(x), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = (x - s[0]) @ n
        inl = int(np.sum(np.abs(d) < thr))
        # the ground has (almost) everything on one side of it
        minority = min(np.sum(d > thr), np.sum(d < -thr))
        score = inl - 2 * minority
        if score > best_score:
            best_n, best_inl, best_p, best_score = n, inl, s[0], score
    if best_n is not None and best_inl >= 0.1 * len(x) and best_score > 0.05 * len(x):
        side = (x - best_p) @ best_n
        n = best_n if np.sum(side > thr) >= np.sum(side < -thr) else -best_n
        # snap to a world axis only when within ~2 degrees (tilted captures keep the true normal)
        ax = np.eye(3)[np.argmax(np.abs(n))] * np.sign(n[np.argmax(np.abs(n))])
        if n @ ax > np.cos(np.deg2rad(2)):
            n = ax
        return n, f"ground plane ({best_inl / len(x):.0%} of splats)"
    name = guess_up_axis(model)
    return parse_up(name), f"tree shape/colour heuristic ({name})"
