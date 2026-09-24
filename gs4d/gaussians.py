"""In-memory container for a 3D Gaussian Splatting model.

The arrays follow the *raw* conventions of the reference 3DGS PLY format so
that a load -> edit -> save round trip is lossless:

* ``means``          (N, 3)    Gaussian centres
* ``quats``          (N, 4)    rotation quaternion, (w, x, y, z)
* ``log_scales``     (N, 3)    log of the per-axis standard deviation
* ``opacity_logits`` (N,)      inverse-sigmoid opacity
* ``sh_dc``          (N, 3)    degree-0 SH coefficient (RGB)
* ``sh_rest``        (N, K, 3) higher-order SH, K = (degree + 1)^2 - 1
* ``extra``          dict of optional per-Gaussian arrays (labels, weights...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quaternion import quat_multiply, quat_normalize, matrix_to_quat

SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    return np.asarray(sh0, dtype=np.float64) * SH_C0 + 0.5


def num_sh_rest(degree: int) -> int:
    return (degree + 1) ** 2 - 1


def degree_from_num_rest(k: int) -> int:
    for d in range(0, 5):
        if num_sh_rest(d) == k:
            return d
    raise ValueError(f"{k} higher-order SH coefficients per channel is not a valid SH layout")


@dataclass
class GaussianModel:
    means: np.ndarray
    quats: np.ndarray
    log_scales: np.ndarray
    opacity_logits: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.means)
        self.means = np.ascontiguousarray(self.means, dtype=np.float32).reshape(n, 3)
        self.quats = np.ascontiguousarray(quat_normalize(self.quats), dtype=np.float32).reshape(n, 4)
        self.log_scales = np.ascontiguousarray(self.log_scales, dtype=np.float32).reshape(n, 3)
        self.opacity_logits = np.ascontiguousarray(self.opacity_logits, dtype=np.float32).reshape(n)
        self.sh_dc = np.ascontiguousarray(self.sh_dc, dtype=np.float32).reshape(n, 3)
        if self.sh_rest is None:
            self.sh_rest = np.zeros((n, 0, 3), dtype=np.float32)
        rest = np.ascontiguousarray(self.sh_rest, dtype=np.float32)
        k = rest.shape[1] if rest.ndim == 3 else (rest.size // (3 * n) if n else 0)
        self.sh_rest = rest.reshape(n, k, 3)
        degree_from_num_rest(self.sh_rest.shape[1])  # validates the layout
        for k, v in self.extra.items():
            if len(v) != n:
                raise ValueError(f"extra attribute {k!r} has length {len(v)}, expected {n}")

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        return degree_from_num_rest(self.sh_rest.shape[1])

    @property
    def scales(self) -> np.ndarray:
        """Linear (activated) per-axis standard deviations."""
        return np.exp(self.log_scales.astype(np.float64))

    @property
    def opacities(self) -> np.ndarray:
        """Linear opacity in [0, 1]."""
        return sigmoid(self.opacity_logits)

    @property
    def rgb(self) -> np.ndarray:
        """View-independent base colour (degree-0 SH), clipped to [0, 1]."""
        return np.clip(sh0_to_rgb(self.sh_dc), 0.0, 1.0)

    @classmethod
    def from_activated(
        cls,
        means: np.ndarray,
        quats: np.ndarray,
        scales: np.ndarray,
        opacities: np.ndarray,
        rgb: np.ndarray,
        sh_rest: np.ndarray | None = None,
        extra: dict | None = None,
    ) -> "GaussianModel":
        return cls(
            means=means,
            quats=quats,
            log_scales=np.log(np.maximum(np.asarray(scales, dtype=np.float64), 1e-10)),
            opacity_logits=logit(opacities),
            sh_dc=rgb_to_sh0(rgb),
            sh_rest=sh_rest,
            extra=dict(extra or {}),
        )

    def copy(self) -> "GaussianModel":
        return GaussianModel(
            self.means.copy(),
            self.quats.copy(),
            self.log_scales.copy(),
            self.opacity_logits.copy(),
            self.sh_dc.copy(),
            self.sh_rest.copy(),
            {k: np.array(v, copy=True) for k, v in self.extra.items()},
        )

    def subset(self, index: np.ndarray) -> "GaussianModel":
        """Select Gaussians by boolean mask, integer index array or slice."""
        if not isinstance(index, slice):
            index = np.asarray(index)
        return GaussianModel(
            self.means[index],
            self.quats[index],
            self.log_scales[index],
            self.opacity_logits[index],
            self.sh_dc[index],
            self.sh_rest[index],
            {k: np.asarray(v)[index] for k, v in self.extra.items()},
        )

    def with_pose(self, means: np.ndarray, quats: np.ndarray) -> "GaussianModel":
        """Shallow copy that shares appearance arrays but has new means/quats."""
        out = GaussianModel.__new__(GaussianModel)
        out.means = np.ascontiguousarray(means, dtype=np.float32)
        out.quats = np.ascontiguousarray(quats, dtype=np.float32)
        out.log_scales = self.log_scales
        out.opacity_logits = self.opacity_logits
        out.sh_dc = self.sh_dc
        out.sh_rest = self.sh_rest
        out.extra = self.extra
        return out

    def with_sh_degree(self, degree: int) -> "GaussianModel":
        """Truncate (or zero-pad) the SH coefficients to ``degree``."""
        k = num_sh_rest(degree)
        cur = self.sh_rest.shape[1]
        if k <= cur:
            rest = self.sh_rest[:, :k]
        else:
            rest = np.concatenate([self.sh_rest, np.zeros((len(self), k - cur, 3), np.float32)], axis=1)
        out = self.copy()
        out.sh_rest = np.ascontiguousarray(rest, dtype=np.float32)
        return out

    @staticmethod
    def concat(models: list["GaussianModel"]) -> "GaussianModel":
        deg = max(m.sh_degree for m in models)
        models = [m if m.sh_degree == deg else m.with_sh_degree(deg) for m in models]
        keys = set(models[0].extra)
        for m in models[1:]:
            keys &= set(m.extra)
        return GaussianModel(
            np.concatenate([m.means for m in models]),
            np.concatenate([m.quats for m in models]),
            np.concatenate([m.log_scales for m in models]),
            np.concatenate([m.opacity_logits for m in models]),
            np.concatenate([m.sh_dc for m in models]),
            np.concatenate([m.sh_rest for m in models]),
            {k: np.concatenate([np.asarray(m.extra[k]) for m in models]) for k in keys},
        )

    # ------------------------------------------------------------- geometry
    def bounds(self, quantile: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        if quantile > 0:
            return (
                np.quantile(self.means, quantile, axis=0),
                np.quantile(self.means, 1.0 - quantile, axis=0),
            )
        return self.means.min(axis=0), self.means.max(axis=0)

    def transformed(
        self,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        scale: float = 1.0,
    ) -> "GaussianModel":
        """Apply ``x' = scale * R @ x + t`` to the whole model.

        Orientations are rotated and log-scales shifted accordingly.  The
        higher-order SH coefficients are *not* rotated (view-dependent colour
        will be slightly off after large rotations); use ``with_sh_degree(0)``
        if that matters.
        """
        R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        out = self.copy()
        out.means = (scale * (self.means.astype(np.float64) @ R.T) + t).astype(np.float32)
        qR = matrix_to_quat(R)[None, :]
        out.quats = quat_normalize(quat_multiply(qR, self.quats.astype(np.float64))).astype(np.float32)
        out.log_scales = (self.log_scales + np.log(scale)).astype(np.float32)
        return out

    def summary(self) -> str:
        lo, hi = self.bounds()
        return (
            f"GaussianModel: {len(self):,} Gaussians, SH degree {self.sh_degree}, "
            f"bounds min={np.round(lo, 3).tolist()} max={np.round(hi, 3).tolist()}, "
            f"median opacity={float(np.median(self.opacities)):.3f}"
        )
