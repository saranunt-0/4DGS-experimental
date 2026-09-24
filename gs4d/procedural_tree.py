"""Procedural broad-leaf tree built directly out of 3D Gaussians.

This gives the demo a guaranteed, object-only (no background) tree that runs
on a CPU in seconds, *and* the exact branch skeleton that grew it, so the wind
animation can use true hierarchical (forward-kinematics) bending.

Bark is made of flattened Gaussians tiled around each branch segment; every
leaf is a thin elliptical "disc" Gaussian (plus a smaller tip Gaussian) whose
normal faces outward/upward.  Baked ambient occlusion and a sun gradient give
the crown depth when rendered with plain splatting (no relighting needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gaussians import GaussianModel
from .quaternion import frame_from_normal, matrix_to_quat, rotvec_to_matrix
from .skeleton import TreeSkeleton, _smoothstep


@dataclass
class TreeParams:
    seed: int = 21
    trunk_length: float = 3.6            # metres, trunk (level 0) total length
    trunk_radius: float = 0.16
    children: tuple = (6, 5, 4, 3)       # child branches spawned per branch, by parent level
    length_ratio: tuple = (0.62, 0.55, 0.5, 0.45)
    branch_angle_deg: tuple = (48.0, 42.0, 45.0, 50.0)
    child_start: tuple = (0.38, 0.2, 0.15, 0.2)   # where children start along the parent (0..1)
    radius_ratio: float = 0.62
    segments: tuple = (10, 7, 5, 4, 3)   # segments per branch, by level
    gnarl: float = 0.16                  # random direction jitter per segment
    upward_tropism: float = 0.10
    gravity_droop: float = 0.06          # extra sag for thin branches
    leaf_levels: tuple = (3, 4)          # branch levels that carry leaves
    leaves_per_branch: tuple = (26, 46)  # per branch at each leaf level
    leaf_length: float = 0.17            # metres ("leaf clumps" read better than real 6 cm leaves)
    leaf_width_ratio: float = 0.55
    bark_spacing: float = 0.045          # target distance between bark Gaussians
    leaf_hue_deg: tuple = (78.0, 122.0)  # yellow-green .. green
    bark_rgb: tuple = (0.33, 0.24, 0.17)
    autumn: float = 0.0                  # 0 = summer green, 1 = orange/red foliage


@dataclass
class _Branch:
    level: int
    points: np.ndarray         # (S+1, 3) polyline
    radii: np.ndarray          # (S+1,)
    joint_ids: list = field(default_factory=list)  # joint per segment


def _perpendicular(d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    v -= v.dot(d) * d
    return v / np.linalg.norm(v)


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return rotvec_to_matrix(axis / np.linalg.norm(axis) * angle) @ v


def generate_tree(params: TreeParams | None = None) -> tuple[GaussianModel, TreeSkeleton]:
    """Grow a tree; returns ``(gaussians, skeleton)`` with +Z up, base at the origin."""
    p = params or TreeParams()
    rng = np.random.default_rng(p.seed)
    up = np.array([0.0, 0.0, 1.0])
    golden = np.deg2rad(137.508)

    branches: list[_Branch] = []
    pivots, parents, dirs, lens = [], [], [], []

    def grow(level: int, start: np.ndarray, direction: np.ndarray, length: float, radius: float, parent_joint: int):
        nseg = p.segments[min(level, len(p.segments) - 1)]
        seg_len = length / nseg
        pts = [start.copy()]
        d = direction / np.linalg.norm(direction)
        for s in range(nseg):
            jitter = rng.normal(size=3) * p.gnarl * (0.35 if level == 0 else 0.6 + 0.4 * level / 4)
            droop = p.gravity_droop * level * (s + 1) / nseg
            d = d + jitter + (p.upward_tropism if level > 0 else 0.02) * up - droop * up
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * seg_len)
        pts = np.array(pts)
        radii = radius * (1.0 - 0.65 * np.linspace(0.0, 1.0, nseg + 1) ** 1.1)
        b = _Branch(level, pts, radii)
        prev = parent_joint
        for s in range(nseg):
            jid = len(pivots)
            seg = pts[s + 1] - pts[s]
            pivots.append(pts[s])
            parents.append(prev)
            dirs.append(seg)
            lens.append(np.linalg.norm(seg))
            b.joint_ids.append(jid)
            prev = jid
        branches.append(b)

        if level >= len(p.children):
            return
        nchild = p.children[level]
        if nchild <= 0:
            return
        t0 = p.child_start[level]
        ts = np.sort(t0 + (0.97 - t0) * (np.arange(nchild) + rng.uniform(0.2, 0.8, nchild)) / nchild)
        az0 = rng.uniform(0, 2 * np.pi)
        for c, t in enumerate(ts):
            f = t * nseg
            s = min(int(f), nseg - 1)
            u = f - s
            pos = pts[s] * (1 - u) + pts[s + 1] * u
            pdir = pts[s + 1] - pts[s]
            pdir /= np.linalg.norm(pdir)
            ang = np.deg2rad(p.branch_angle_deg[level] + rng.normal(0, 7))
            # phyllotaxis: successive children rotate by the golden angle around the parent
            ref = np.array([1.0, 0.0, 0.0]) if abs(pdir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ref = ref - ref.dot(pdir) * pdir
            ref /= np.linalg.norm(ref)
            side = _rotate_about(ref, pdir, az0 + c * golden + rng.normal(0, 0.25))
            cdir = np.cos(ang) * pdir + np.sin(ang) * side
            # round crown: children in the middle of the parent are the longest
            uu = (t - t0) / max(0.97 - t0, 1e-6)
            shape = 0.55 + 0.45 * np.sin(np.pi * np.clip(0.15 + 0.85 * uu, 0, 1))
            clen = length * p.length_ratio[level] * shape * rng.uniform(0.85, 1.15)
            crad = max(radius * (1 - 0.65 * t) * p.radius_ratio, 0.004)
            grow(level + 1, pos, cdir, clen, crad, b.joint_ids[s])

    grow(0, np.zeros(3), up + rng.normal(size=3) * 0.03, p.trunk_length, p.trunk_radius, -1)

    pivots_a = np.array(pivots)
    parents_a = np.array(parents)
    dirs_a = np.array(dirs)
    lens_a = np.array(lens)

    # ------------------------------------------------------------ bark splats
    wood_means, wood_rot, wood_scale, wood_rgb, wood_joint, wood_level = [], [], [], [], [], []
    bark = np.array(p.bark_rgb)
    for b in branches:
        for s, jid in enumerate(b.joint_ids):
            a, c = b.points[s], b.points[s + 1]
            seg = c - a
            L = np.linalg.norm(seg)
            d = seg / L
            r = 0.5 * (b.radii[s] + b.radii[s + 1])
            n_along = max(1, int(np.ceil(L / p.bark_spacing)))
            n_around = int(np.clip(np.round(2 * np.pi * r / p.bark_spacing), 3, 40))
            tt = (np.arange(n_along) + 0.5) / n_along
            phi = (np.arange(n_around) / n_around) * 2 * np.pi
            T, PHI = np.meshgrid(tt, phi, indexing="ij")
            T = T.reshape(-1)
            PHI = PHI.reshape(-1) + rng.uniform(0, 2 * np.pi / n_around, T.size) + T * 0.8
            e1 = _perpendicular(d, np.random.default_rng(jid))
            e2 = np.cross(d, e1)
            radial = np.cos(PHI)[:, None] * e1 + np.sin(PHI)[:, None] * e2
            rr = (b.radii[s] * (1 - T) + b.radii[s + 1] * T)[:, None]
            centers = a + T[:, None] * seg + radial * rr
            # local frame: x along branch, y around, z = surface normal
            frames = np.stack([np.broadcast_to(d, radial.shape), np.cross(radial, d), radial], axis=-1)
            circ = 2 * np.pi * rr[:, 0] / n_around
            sc = np.stack(
                [np.full(T.size, L / n_along * 0.6), circ * 0.6, np.maximum(rr[:, 0] * 0.12, 0.002)], 1
            )
            shade = rng.normal(1.0, 0.13, T.size) * (0.85 + 0.15 * np.cos(PHI * 3 + jid))
            height_tone = 0.75 + 0.25 * np.clip(centers[:, 2] / 3.0, 0, 1)
            col = bark[None, :] * (shade * height_tone)[:, None]
            if b.level >= 3:  # young twigs are a bit greener / lighter
                col = col * 0.8 + np.array([0.30, 0.28, 0.16]) * 0.2
            wood_means.append(centers)
            wood_rot.append(frames)
            wood_scale.append(sc)
            wood_rgb.append(col)
            wood_joint.append(np.full(T.size, jid))
            wood_level.append(np.full(T.size, b.level))
    wood_means = np.concatenate(wood_means)
    wood_quats = matrix_to_quat(np.concatenate(wood_rot))
    wood_scale = np.concatenate(wood_scale)
    wood_rgb = np.clip(np.concatenate(wood_rgb), 0, 1)
    wood_joint = np.concatenate(wood_joint)
    wood_level = np.concatenate(wood_level)

    # ------------------------------------------------------------ leaf splats
    leaf_pos, leaf_frames, leaf_joint, leaf_level = [], [], [], []
    crown_center = pivots_a[parents_a >= 0].mean(0) if len(pivots_a) > 1 else np.zeros(3)
    for b in branches:
        if b.level not in p.leaf_levels:
            continue
        nleaf = p.leaves_per_branch[list(p.leaf_levels).index(b.level)]
        nseg = len(b.joint_ids)
        t = np.sort(rng.uniform(0.2, 1.0, nleaf))
        f = t * nseg
        s = np.minimum(f.astype(int), nseg - 1)
        u = (f - s)[:, None]
        pos = b.points[s] * (1 - u) + b.points[s + 1] * u
        bdir = b.points[s + 1] - b.points[s]
        bdir /= np.linalg.norm(bdir, axis=1, keepdims=True)
        outward = pos - crown_center
        outward[:, 2] *= 0.3
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
        rnd = rng.normal(size=(nleaf, 3))
        petiole = rnd - np.sum(rnd * bdir, 1, keepdims=True) * bdir
        petiole = petiole / np.linalg.norm(petiole, axis=1, keepdims=True) + 0.6 * outward + 0.3 * bdir - 0.25 * up
        petiole /= np.linalg.norm(petiole, axis=1, keepdims=True)
        normal = up * 0.9 + outward * 0.8 + rng.normal(size=(nleaf, 3)) * 0.7
        frames = frame_from_normal(normal, tangent_hint=petiole)
        leaf_pos.append(pos + petiole * p.leaf_length * 0.55 * rng.uniform(0.7, 1.2, (nleaf, 1)))
        leaf_frames.append(frames)
        leaf_joint.append(np.array(b.joint_ids)[s])
        leaf_level.append(np.full(nleaf, b.level))
    leaf_pos = np.concatenate(leaf_pos)
    leaf_frames = np.concatenate(leaf_frames)
    leaf_joint = np.concatenate(leaf_joint)
    leaf_level = np.concatenate(leaf_level)
    nl = len(leaf_pos)

    size = p.leaf_length * rng.uniform(0.75, 1.25, nl)
    blade_scale = np.stack([size * 0.30, size * p.leaf_width_ratio * 0.30, np.full(nl, 0.004)], 1)
    tip_pos = leaf_pos + leaf_frames[:, :, 0] * (size * 0.38)[:, None]
    tip_scale = blade_scale * np.array([0.55, 0.6, 1.0])

    # colour: hue variation, sun-lit top/outer leaves brighter, inner leaves darker (baked AO)
    rel = leaf_pos - crown_center
    extent = np.quantile(np.linalg.norm(rel, axis=1), 0.95)
    outer = np.clip(np.linalg.norm(rel * [1, 1, 0.8], axis=1) / extent, 0, 1.2)
    height = (leaf_pos[:, 2] - leaf_pos[:, 2].min()) / max(np.ptp(leaf_pos[:, 2]), 1e-6)
    ao = 0.35 + 0.65 * _smoothstep(outer, 0.2, 1.0)
    sun = 0.75 + 0.35 * height + 0.25 * np.clip(leaf_frames[:, 2, 2], 0, 1)
    hue = rng.uniform(*p.leaf_hue_deg, nl) / 360.0
    if p.autumn > 0:
        hue = hue * (1 - p.autumn) + rng.uniform(0.02, 0.12, nl) * p.autumn
    sat = rng.uniform(0.55, 0.85, nl)
    val = np.clip(rng.uniform(0.32, 0.52, nl) * ao * sun, 0.04, 1.0)
    leaf_rgb = _hsv_to_rgb(np.stack([hue, sat, val], 1))
    tip_rgb = np.clip(leaf_rgb * rng.uniform(1.0, 1.25, (nl, 1)), 0, 1)

    leaf_quats = matrix_to_quat(leaf_frames)

    means = np.concatenate([wood_means, leaf_pos, tip_pos])
    quats = np.concatenate([wood_quats, leaf_quats, leaf_quats])
    scales = np.concatenate([wood_scale, blade_scale, tip_scale])
    rgb = np.concatenate([wood_rgb, leaf_rgb, tip_rgb])
    opac = np.concatenate([np.full(len(wood_means), 0.97), np.full(nl, 0.93), np.full(nl, 0.88)])
    part = np.concatenate([np.zeros(len(wood_means)), np.ones(2 * nl)])
    joint = np.concatenate([wood_joint, leaf_joint, leaf_joint]).astype(np.int64)
    level = np.concatenate([wood_level, leaf_level, leaf_level])

    model = GaussianModel.from_activated(
        means, quats, scales, opac, rgb, extra={"part": part.astype(np.float32), "branch_level": level.astype(np.float32)}
    )
    leafness = np.where(part > 0.5, 1.0, np.where(level >= 4, 0.35, 0.0))
    height_total = float(np.quantile(means[:, 2], 0.998) - means[:, 2].min())
    skel = TreeSkeleton(pivots_a, parents_a, dirs_a, lens_a, joint, leafness, up, height_total)
    return model, skel


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, 0] % 1.0, hsv[:, 1], hsv[:, 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    pch = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    choices = [
        np.stack([v, t, pch], 1), np.stack([q, v, pch], 1), np.stack([pch, v, t], 1),
        np.stack([pch, q, v], 1), np.stack([t, pch, v], 1), np.stack([v, pch, q], 1),
    ]
    out = np.zeros((len(h), 3))
    for k in range(6):
        out[i == k] = choices[k][i == k]
    return out
