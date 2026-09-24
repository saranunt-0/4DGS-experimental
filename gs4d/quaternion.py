"""Vectorised quaternion / rotation helpers.

Convention: quaternions are stored as ``(w, x, y, z)`` (scalar first), which is
the layout used by the original 3DGS PLY files (``rot_0..rot_3``), gsplat and
OpenUSD's ``GfQuatf`` constructor.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q).copy()
    q[..., 1:] *= -1.0
    return q


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Axis-angle vector (angle = norm, radians) -> unit quaternion."""
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(x/2)/x with a Taylor fallback near zero
    small = angle < 1e-8
    k = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.where(small, 1.0, angle))
    return np.concatenate([np.cos(half), v * k], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w >= 0), numerically robust."""
    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    # Four candidate solutions (Shepperd); pick the best-conditioned one.
    t = np.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    best = np.argmax(t, axis=-1)
    q = np.empty((m.shape[0], 4))
    s = np.sqrt(np.maximum(t[np.arange(len(best)), best], 1e-12)) * 2.0

    i = best == 0
    q[i, 0] = 0.25 * s[i]
    q[i, 1] = (m[i, 2, 1] - m[i, 1, 2]) / s[i]
    q[i, 2] = (m[i, 0, 2] - m[i, 2, 0]) / s[i]
    q[i, 3] = (m[i, 1, 0] - m[i, 0, 1]) / s[i]
    i = best == 1
    q[i, 0] = (m[i, 2, 1] - m[i, 1, 2]) / s[i]
    q[i, 1] = 0.25 * s[i]
    q[i, 2] = (m[i, 0, 1] + m[i, 1, 0]) / s[i]
    q[i, 3] = (m[i, 0, 2] + m[i, 2, 0]) / s[i]
    i = best == 2
    q[i, 0] = (m[i, 0, 2] - m[i, 2, 0]) / s[i]
    q[i, 1] = (m[i, 0, 1] + m[i, 1, 0]) / s[i]
    q[i, 2] = 0.25 * s[i]
    q[i, 3] = (m[i, 1, 2] + m[i, 2, 1]) / s[i]
    i = best == 3
    q[i, 0] = (m[i, 1, 0] - m[i, 0, 1]) / s[i]
    q[i, 1] = (m[i, 0, 2] + m[i, 2, 0]) / s[i]
    q[i, 2] = (m[i, 1, 2] + m[i, 2, 1]) / s[i]
    q[i, 3] = 0.25 * s[i]

    q = quat_normalize(q)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(shape + (4,))


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """Rodrigues formula for a batch of axis-angle vectors."""
    return quat_to_matrix(quat_from_rotvec(v))


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:  # opposite vectors: rotate 180 deg about any orthogonal axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return rotvec_to_matrix(axis * np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def frame_from_normal(n: np.ndarray, tangent_hint: np.ndarray | None = None) -> np.ndarray:
    """Build rotation matrices whose 3rd column is the (unit) vector ``n``.

    Columns are (t, b, n); ``t`` follows ``tangent_hint`` projected onto the
    plane orthogonal to ``n`` when given.
    """
    n = np.asarray(n, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    if tangent_hint is None:
        helper = np.where(np.abs(n[..., 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        t = np.cross(helper, n)
    else:
        th = np.broadcast_to(np.asarray(tangent_hint, dtype=np.float64), n.shape)
        t = th - np.sum(th * n, axis=-1, keepdims=True) * n
        bad = np.linalg.norm(t, axis=-1) < 1e-6
        if np.any(bad):
            helper = np.where(np.abs(n[bad][..., 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
            t[bad] = np.cross(helper, n[bad])
    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
    b = np.cross(n, t)
    return np.stack([t, b, n], axis=-1)
