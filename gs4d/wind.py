"""Wind animation for Gaussian-splat trees.

Model (training-free, runs on a CPU):

1. **Wind field** ``v(x, t)``: mean flow + gust fronts that travel downwind
   (so a gust visibly sweeps across the crown) + smooth 3D turbulence.
2. **Branch dynamics**: every skeleton joint is a damped harmonic oscillator
   (2-DOF bend, twist removed) driven by the aerodynamic drag torque of its
   whole subtree, ``tau_j = sum_i (x_i - p_j) x F_i`` with ``F ~ A |v| v``.
   Natural frequency and compliance are interpolated from trunk to twig using
   a pipe-model thickness proxy, so the trunk sways slowly and a little while
   twigs whip faster and further.  This is the same modelling family as the
   classic real-time tree animation literature (Stam 1997, Habel et al. 2009)
   and the per-part oscillator prior tested in "Wind on Trees" (2026).
3. **Forward kinematics** poses every Gaussian (position *and* orientation).
4. **Leaf flutter**: a spatially coherent random-Fourier field adds fast
   small rotations/jitter to leaf Gaussians (flipping leaves shimmer because
   their anisotropic splats change footprint as they rotate).

With ``loop=True`` all temporal frequencies are multiples of 1/duration and
the residual is distributed over the clip, so the last frame flows back into
the first (seamless loops for games / turntables).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .gaussians import GaussianModel
from .quaternion import matrix_to_quat, quat_from_rotvec, quat_multiply, quat_normalize, rotvec_to_matrix
from .skeleton import TreeSkeleton, horizontal_basis, subtree_sizes

V_REF = 10.0  # m/s; bend amplitudes are specified at this reference wind speed


@dataclass
class WindParams:
    # wind field
    speed: float = 5.0             # mean wind speed in m/s (2 = breeze, 5 = moderate, 10+ = strong)
    direction_deg: float = 0.0     # azimuth around the up axis
    gustiness: float = 0.6         # gust amplitude relative to the mean speed
    gust_period: float = 3.0       # typical seconds between gusts
    turbulence: float = 0.35       # small-scale fluctuation relative to the mean speed
    tree_height_m: float | None = None  # real height of the tree (None: model units are metres)
    # tree response
    flexibility: float = 1.0       # global bend multiplier
    trunk_stiffness: float = 1.0   # >1 = stiffer, faster trunk
    trunk_bend_deg: float = 5.0    # trunk bend over the full height at 10 m/s
    branch_bend_deg: float = 80.0  # bend of thin branches per tree-height of length at 10 m/s
    trunk_freq_hz: float = 0.45
    twig_freq_hz: float = 2.6
    damping: float = 0.12          # damping ratio (0.05 bouncy .. 0.4 sluggish)
    max_joint_deg: float = 14.0    # soft limit per joint
    # leaves
    leaf_flutter_deg: float = 25.0  # leaf rotation amplitude at 10 m/s
    leaf_flutter_hz: float = 4.5
    leaf_shake: float = 0.004       # leaf positional jitter (fraction of tree height) at 10 m/s
    leaf_coherence: float = 0.05    # spatial wavelength of flutter (fraction of tree height)
    # timeline
    fps: float = 24.0
    duration: float = 4.0
    loop: bool = True
    warmup: float = 8.0
    substeps: int = 6
    max_drag_particles: int = 6000
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _quantize(freqs: np.ndarray, period: float | None) -> np.ndarray:
    if not period:
        return freqs
    return np.maximum(np.round(freqs * period), 1.0) / period


class WindField:
    """Analytic, loopable wind velocity field (in m/s, positions in metres)."""

    def __init__(self, p: WindParams, up: np.ndarray, center: np.ndarray, height_m: float, rng: np.random.Generator):
        self.p = p
        self.up = up / np.linalg.norm(up)
        e1, e2 = horizontal_basis(self.up)
        a = np.deg2rad(p.direction_deg)
        self.dir = np.cos(a) * e1 + np.sin(a) * e2
        self.center = np.asarray(center, np.float64)
        period = p.duration if p.loop else None
        # gust fronts: a few low frequencies around 1/gust_period travelling downwind
        base = 1.0 / max(p.gust_period, 0.2)
        self.gf = _quantize(base * np.array([0.45, 0.8, 1.0, 1.35, 2.1]), period)
        self.ga = np.array([0.8, 1.0, 0.9, 0.6, 0.35])
        self.ga /= np.sqrt(0.5 * np.sum(self.ga**2))
        self.gphi = rng.uniform(0, 2 * np.pi, len(self.gf))
        self.convect = max(p.speed, 0.5)
        # turbulence: random Fourier features, wavelength ~ half the tree
        m = 12
        lam = max(0.5 * height_m, 0.3)
        k = rng.normal(size=(m, 3))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        self.tk = k * (2 * np.pi / lam) * rng.uniform(0.6, 1.6, (m, 1))
        self.tf = _quantize(rng.uniform(0.25, 1.4, m), period)
        self.tphi = rng.uniform(0, 2 * np.pi, m)
        b = rng.normal(size=(m, 3))
        b -= 0.5 * np.outer(b @ self.up, self.up)  # weaker vertical gusts
        self.tb = b / np.linalg.norm(b, axis=1, keepdims=True) / np.sqrt(0.5 * m)

    def __call__(self, x_m: np.ndarray, t: float) -> np.ndarray:
        p = self.p
        if p.speed <= 0:
            return np.zeros_like(x_m)
        xi = (x_m - self.center) @ self.dir
        g = np.sin(2 * np.pi * self.gf[None, :] * (t - xi[:, None] / self.convect) + self.gphi[None, :]) @ self.ga
        along = p.speed * np.maximum(1.0 + p.gustiness * g, 0.05)
        turb = np.sin(x_m @ self.tk.T + 2 * np.pi * self.tf[None, :] * t + self.tphi[None, :]) @ self.tb
        return along[:, None] * self.dir[None, :] + (p.speed * p.turbulence) * turb


def _drag_particles(model: GaussianModel, skel: TreeSkeleton, max_particles: int):
    """Aggregate Gaussians into (voxel, joint) drag particles with area weights."""
    s = np.sort(model.scales, axis=1)
    area = np.pi * s[:, 2] * s[:, 1] * model.opacities * (0.3 + 0.7 * skel.leafness)
    x = model.means.astype(np.float64)
    ok = skel.bind >= 0
    x, area, joint = x[ok], area[ok], skel.bind[ok]
    extent = np.ptp(x, axis=0).max() if len(x) else 1.0
    voxel = extent / 16.0
    for _ in range(12):
        key = np.floor((x - x.min(0)) / voxel).astype(np.int64)
        key = np.concatenate([key, joint[:, None]], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        if len(uniq) <= max_particles:
            break
        voxel *= 1.35
    inv = inv.reshape(-1)
    M = len(uniq)
    a = np.bincount(inv, weights=area, minlength=M)
    px = np.zeros((M, 3))
    np.add.at(px, inv, x * area[:, None])
    px /= np.maximum(a, 1e-20)[:, None]
    pj = uniq[:, 3]
    return px, a, pj


class WindAnimation:
    """Simulated wind motion for a Gaussian tree.  Index it to get posed frames."""

    def __init__(self, model: GaussianModel, skel: TreeSkeleton, params: WindParams | None = None, verbose: bool = True):
        if len(model) != len(skel.bind):
            raise ValueError("skeleton binding does not match the model size")
        self.model = model
        self.skel = skel
        self.p = p = params or WindParams()
        rng = np.random.default_rng(p.seed)
        self.n_frames = max(1, int(round(p.duration * p.fps)))
        self.times = np.arange(self.n_frames) / p.fps

        H = skel.height
        height_m = p.tree_height_m or H
        self.to_m = height_m / H  # model units -> metres
        center = np.median(model.means, axis=0).astype(np.float64)
        self.wind = WindField(p, skel.up, center * self.to_m, height_m, rng)

        # ----------------------------------------------------- joint parameters
        px, pa, pj = _drag_particles(model, skel, p.max_drag_particles)
        self._px, self._pa = px, pa
        A = skel.ancestor_matrix(pj).tocoo()
        self._A = A.tocsr()
        J = skel.num_joints
        lever = np.linalg.norm(px[A.col] - skel.pivots[A.row], axis=1)
        self._D = np.bincount(A.row, weights=pa[A.col] * lever, minlength=J) + 1e-20
        sub_area = subtree_sizes(skel.parents, np.bincount(pj, weights=pa, minlength=J))
        s = np.sqrt(sub_area / max(sub_area.max(), 1e-20))
        self.thickness = s
        stiff = max(p.trunk_stiffness, 1e-3)
        kt = np.deg2rad(p.trunk_bend_deg) / stiff
        kb = np.deg2rad(p.branch_bend_deg)
        kappa = kt**s * kb ** (1 - s)
        self.amp = p.flexibility * kappa * (skel.lengths / H)
        ft = p.trunk_freq_hz * np.sqrt(stiff)
        self.omega = 2 * np.pi * (ft**s * p.twig_freq_hz ** (1 - s))

        # --------------------------------------------------------- simulation
        self.theta = self._simulate(verbose)

        # --------------------------------------------------------- leaf flutter
        m = 10
        lam = max(p.leaf_coherence, 1e-3) * H
        k = rng.normal(size=(m, 3))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        self._fk = k * (2 * np.pi / lam) * rng.uniform(0.7, 1.4, (m, 1))
        period = p.duration if p.loop else None
        self._ff = _quantize(p.leaf_flutter_hz * rng.uniform(0.6, 1.5, m), period)
        self._fphi = rng.uniform(0, 2 * np.pi, m)
        u = rng.normal(size=(m, 3))
        self._fu = u / np.linalg.norm(u, axis=1, keepdims=True) / np.sqrt(0.5 * m)
        w = rng.normal(size=(m, 3))
        self._fw = w / np.linalg.norm(w, axis=1, keepdims=True) / np.sqrt(0.5 * m)
        self._fpsi = rng.uniform(0, 2 * np.pi, m)

    # ------------------------------------------------------------------ sim
    def _torque(self, t: float) -> np.ndarray:
        v = self.wind(self._px * self.to_m, t)
        f = v * np.linalg.norm(v, axis=1, keepdims=True) / V_REF**2
        F = f * self._pa[:, None]
        S0 = self._A @ F
        S1 = self._A @ np.cross(self._px, F)
        tau = S1 - np.cross(self.skel.pivots, S0)
        return tau / self._D[:, None]

    def _simulate(self, verbose: bool) -> np.ndarray:
        p = self.p
        skel = self.skel
        J = skel.num_joints
        d = skel.directions
        dt = 1.0 / (p.fps * p.substeps)
        zeta = p.damping
        w = self.omega[:, None]
        amp = self.amp[:, None]
        min_damp_time = 1.0 / max(zeta * self.omega.min(), 1e-3)
        warm = max(p.warmup, min(4.0 * min_damp_time, 30.0))
        n_warm = int(np.ceil(warm * p.fps)) * p.substeps
        n_rec = (self.n_frames + (1 if p.loop else 0)) * p.substeps
        th = np.zeros((J, 3))
        om = np.zeros((J, 3))
        out = np.zeros((self.n_frames + 1, J, 3))
        t0 = -n_warm * dt

        def proj(v):
            return v - np.sum(v * d, axis=1, keepdims=True) * d

        rec = 0
        for step in range(n_warm + n_rec):
            t = t0 + step * dt
            if step >= n_warm and (step - n_warm) % p.substeps == 0:
                out[rec] = th
                rec += 1
            target = proj(amp * self._torque(t))
            om = om + dt * (w**2 * (target - th) - 2 * zeta * w * om)
            om = proj(om)
            th = proj(th + dt * om)
        if rec <= self.n_frames:
            out[rec] = th
        theta = out[: self.n_frames + 1]
        if p.loop:
            # spread the (small) residual so frame N == frame 0 exactly
            resid = theta[self.n_frames] - theta[0]
            ramp = (np.arange(self.n_frames + 1) / self.n_frames)[:, None, None]
            theta = theta - ramp * resid[None]
        theta = theta[: self.n_frames]
        # soft joint limit
        lim = np.deg2rad(self.p.max_joint_deg)
        mag = np.linalg.norm(theta, axis=-1, keepdims=True)
        theta = theta * (lim * np.tanh(mag / lim) / np.maximum(mag, 1e-12))
        if verbose:
            deg = np.rad2deg(np.linalg.norm(theta, axis=-1))
            print(
                f"[wind] simulated {J} joints for {warm:.1f}s warm-up + {p.duration:.1f}s clip; "
                f"mean/max joint bend {deg.mean():.2f}/{deg.max():.2f} deg"
            )
        return theta

    # --------------------------------------------------------------- posing
    def joint_transforms(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """World rotation (J,3,3) and posed pivot position (J,3) of every joint."""
        skel = self.skel
        Rl = rotvec_to_matrix(self.theta[i])
        Rw = np.empty_like(Rl)
        Pw = np.empty((skel.num_joints, 3))
        for lvl in skel.levels:
            par = skel.parents[lvl]
            root = par < 0
            r = lvl[root]
            Rw[r] = Rl[r]
            Pw[r] = skel.pivots[r]
            c = lvl[~root]
            if len(c):
                pp = par[~root]
                Rw[c] = Rw[pp] @ Rl[c]
                Pw[c] = np.einsum("nij,nj->ni", Rw[pp], skel.pivots[c] - skel.pivots[pp]) + Pw[pp]
        return Rw, Pw

    def frame(self, i: int, flutter: bool = True) -> GaussianModel:
        """Posed Gaussians for frame ``i`` (shares appearance arrays with the rest pose)."""
        i = int(i) % self.n_frames
        skel = self.skel
        model = self.model
        x = model.means.astype(np.float64)
        q = model.quats.astype(np.float64)
        Rw, Pw = self.joint_transforms(i)
        qw = matrix_to_quat(Rw)
        b = skel.bind
        moving = b >= 0
        bm = b[moving]
        x_new = x.copy()
        q_new = q.copy()
        x_new[moving] = np.einsum("nij,nj->ni", Rw[bm], x[moving] - skel.pivots[bm]) + Pw[bm]
        q_new[moving] = quat_multiply(qw[bm], q[moving])

        p = self.p
        if flutter and p.speed > 0 and (p.leaf_flutter_deg > 0 or p.leaf_shake > 0):
            leaf = skel.leafness
            sel = np.flatnonzero(leaf > 1e-3)
            if len(sel):
                t = self.times[i]
                # local wind strength at each Gaussian's joint (gusts modulate the flutter)
                vj = self.wind(skel.pivots * self.to_m, t)
                speed = np.linalg.norm(vj, axis=1) / V_REF
                gain = leaf[sel] * np.minimum(speed[np.maximum(b[sel], 0)], 2.5)
                ph = x[sel] @ self._fk.T + 2 * np.pi * self._ff[None, :] * t + self._fphi[None, :]
                rot = np.sin(ph) @ self._fu * (np.deg2rad(p.leaf_flutter_deg) * gain)[:, None]
                disp = np.cos(ph + self._fpsi[None, :]) @ self._fw * (p.leaf_shake * skel.height * gain)[:, None]
                q_new[sel] = quat_multiply(quat_from_rotvec(rot), q_new[sel])
                x_new[sel] += disp
        return model.with_pose(x_new.astype(np.float32), quat_normalize(q_new).astype(np.float32))

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, i: int) -> GaussianModel:
        return self.frame(i)

    def __iter__(self):
        for i in range(self.n_frames):
            yield self.frame(i)

    def summary(self) -> str:
        deg = np.rad2deg(np.linalg.norm(self.theta, axis=-1))
        return (
            f"WindAnimation: {self.n_frames} frames @ {self.p.fps:g} fps, {self.skel.num_joints} joints, "
            f"joint bend mean {deg.mean():.2f} deg / max {deg.max():.2f} deg, loop={self.p.loop}"
        )


def animate_tree(model: GaussianModel, skel: TreeSkeleton, **kwargs) -> WindAnimation:
    """Convenience wrapper: ``animate_tree(model, skel, speed=6, duration=4)``."""
    return WindAnimation(model, skel, WindParams(**kwargs))
