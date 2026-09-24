"""Synthetic "captured scene" around an object, for demos and tests.

A real 3DGS capture of a tree also reconstructs the ground, whatever was
behind it, a shell of huge sky Gaussians and semi-transparent floaters.  This
module fakes those artifacts around a clean model so that the object-only
tools can be demonstrated (and measured) without a real capture.
"""

from __future__ import annotations

import numpy as np

from .gaussians import GaussianModel
from .skeleton import horizontal_basis, parse_up


def _disc_gaussians(center, normal, n, radius, rgb, rng, size, opacity):
    from .quaternion import frame_from_normal, matrix_to_quat

    e1, e2 = horizontal_basis(normal)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    pos = center + (r * np.cos(a))[:, None] * e1 + (r * np.sin(a))[:, None] * e2
    frames = frame_from_normal(np.tile(normal, (n, 1)) + rng.normal(0, 0.15, (n, 3)))
    sc = np.stack([np.full(n, size), np.full(n, size), np.full(n, size * 0.1)], 1) * rng.uniform(0.6, 1.4, (n, 1))
    return pos, matrix_to_quat(frames), sc, np.clip(rgb * rng.normal(1, 0.12, (n, 3)), 0, 1), np.full(n, opacity)


def make_captured_scene(obj: GaussianModel, up="+z", seed: int = 0, ground_radius: float = 2.5,
                        n_ground: int = 12000, n_floaters: int = 2500, n_sky: int = 3000,
                        n_clutter: int = 4000) -> tuple[GaussianModel, np.ndarray]:
    """Return ``(scene, is_object)``; distances are relative to the object height."""
    rng = np.random.default_rng(seed)
    up = parse_up(up)
    x = obj.means.astype(np.float64)
    h = x @ up
    height = np.quantile(h, 0.998) - h.min()
    base = x[h <= h.min() + 0.02 * height].mean(0)
    base = base - (base @ up - h.min()) * up
    e1, e2 = horizontal_basis(up)
    parts = []

    # ground: grass/dirt discs, lying on the plane through the trunk base
    parts.append(_disc_gaussians(base - 0.004 * height * up, up, n_ground, ground_radius * height,
                                 np.array([0.36, 0.42, 0.22]), rng, 0.018 * height, 0.95))
    # clutter: a bush / fence patch off to one side
    c = base + 1.6 * height * e1 + 0.4 * height * e2 + 0.15 * height * up
    pos = c + rng.normal(0, 0.18 * height, (n_clutter, 3)) * [1, 1, 0.6]
    parts.append((pos, rng.normal(size=(n_clutter, 4)), np.full((n_clutter, 3), 0.02 * height),
                  np.clip(np.array([0.25, 0.3, 0.18]) * rng.normal(1, 0.2, (n_clutter, 3)), 0, 1),
                  np.full(n_clutter, 0.9)))
    # sky shell: huge, soft, bluish Gaussians far away
    d = rng.normal(size=(n_sky, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[(d @ up) < -0.1] *= -1
    pos = base + 4.0 * height * d
    parts.append((pos, rng.normal(size=(n_sky, 4)), np.full((n_sky, 3), 0.35 * height) * rng.uniform(0.5, 1.5, (n_sky, 1)),
                  np.clip(np.array([0.72, 0.82, 0.95]) * rng.normal(1, 0.05, (n_sky, 3)), 0, 1),
                  rng.uniform(0.2, 0.7, n_sky)))
    # floaters: semi-transparent specks around the object
    pos = base + rng.normal(0, 0.8 * height, (n_floaters, 3)) + 0.5 * height * up
    parts.append((pos, rng.normal(size=(n_floaters, 4)), np.full((n_floaters, 3), 0.01 * height) * rng.uniform(0.5, 3, (n_floaters, 1)),
                  rng.uniform(0.3, 0.95, (n_floaters, 3)), rng.uniform(0.05, 0.5, n_floaters)))

    extra_models = [
        GaussianModel.from_activated(p, q, s, o, c_) for p, q, s, c_, o in parts if len(p)
    ]
    obj_plain = GaussianModel(obj.means, obj.quats, obj.log_scales, obj.opacity_logits, obj.sh_dc, obj.sh_rest)
    scene = GaussianModel.concat([obj_plain] + extra_models)
    is_object = np.zeros(len(scene), bool)
    is_object[: len(obj)] = True
    return scene, is_object
