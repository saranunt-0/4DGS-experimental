import numpy as np
import pytest

from gs4d.gaussians import GaussianModel
from gs4d.skeleton import extract_skeleton, guess_up_axis, parse_up
from gs4d.wind import WindAnimation, WindParams


def test_procedural_tree(small_tree):
    m, sk = small_tree
    assert len(m) == len(sk.bind) and len(m) > 2000
    assert sk.parents[0] == -1 and np.all(sk.parents[1:] < np.arange(1, sk.num_joints))
    assert set(np.unique(m.extra["part"])) == {0.0, 1.0}
    assert np.all(sk.leafness[m.extra["part"] > 0.5] == 1.0)
    assert 3.0 < sk.height < 8.0


def test_zero_wind_is_identity(small_tree):
    m, sk = small_tree
    anim = WindAnimation(m, sk, WindParams(speed=0.0, duration=0.5, fps=4), verbose=False)
    for i in range(len(anim)):
        f = anim.frame(i)
        assert np.allclose(f.means, m.means) and np.allclose(np.abs(np.sum(f.quats * m.quats, 1)), 1, atol=1e-5)


def test_wind_bends_downwind_and_loops(small_tree):
    m, sk = small_tree
    p = WindParams(speed=8.0, direction_deg=0.0, duration=2.0, fps=12, loop=True)
    anim = WindAnimation(m, sk, p, verbose=False)
    top = m.means[:, 2] > np.quantile(m.means[:, 2], 0.9)
    disp = np.stack([anim.frame(i, flutter=False).means[top] - m.means[top] for i in range(len(anim))])
    # direction 0 deg = +X for a +Z-up tree: the crown leans downwind on average
    assert disp[..., 0].mean() > 0.01
    assert np.abs(disp[..., 0].mean()) > 3 * np.abs(disp[..., 1].mean())
    # seamless loop: last->first step is no bigger than a typical frame step
    steps = np.abs(np.diff(disp, axis=0)).max(axis=(1, 2))
    wrap = np.abs(disp[0] - disp[-1]).max()
    assert wrap <= 2.0 * steps.max() + 1e-6
    # quaternions stay normalised, scales untouched
    f = anim.frame(3)
    assert np.allclose(np.linalg.norm(f.quats, axis=1), 1, atol=1e-5)
    assert f.log_scales is m.log_scales


def test_leaf_flutter_only_moves_leaves(small_tree):
    m, sk = small_tree
    anim = WindAnimation(m, sk, WindParams(speed=6.0, duration=1.0, fps=8), verbose=False)
    a, b = anim.frame(2), anim.frame(2, flutter=False)
    wood = sk.leafness == 0
    assert np.allclose(a.means[wood], b.means[wood])
    assert np.abs(a.means[~wood] - b.means[~wood]).max() > 0


@pytest.mark.parametrize("up", ["+z", "-y", "+x"])
def test_guess_up_and_extract_skeleton(small_tree, up):
    from gs4d.quaternion import rotation_between

    m, _ = small_tree
    plain = GaussianModel(m.means, m.quats, m.log_scales, m.opacity_logits, m.sh_dc, m.sh_rest)
    R = rotation_between([0, 0, 1], parse_up(up))
    rot = plain.transformed(R)
    assert guess_up_axis(rot) == up
    sk = extract_skeleton(rot, up=up, n_slices=24)
    assert 20 < sk.num_joints < 2000
    assert (sk.parents == -1).sum() == 1
    leaf = m.extra["part"] > 0.5
    assert sk.leafness[leaf].mean() > 0.8 and sk.leafness[~leaf].mean() < 0.3
    anim = WindAnimation(rot, sk, WindParams(speed=6.0, duration=1.0, fps=6), verbose=False)
    f = anim.frame(3)
    d = np.linalg.norm(f.means - rot.means, axis=1)
    h = (rot.means @ parse_up(up))
    # the base barely moves, the crown moves more
    assert d[h < np.quantile(h, 0.05)].mean() < d[h > np.quantile(h, 0.8)].mean()
