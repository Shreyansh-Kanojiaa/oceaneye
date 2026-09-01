"""Acceptance checks for CLAUDE.md section 6.1."""

import numpy as np
import pytest

from oceaneye.config import Config
from oceaneye.fields import (
    DIFFUSIVITY_ERR,
    DOMAIN_M,
    SPEED_ERR,
    WIND_SHEAR,
    WIND_SPEED_MS,
    current,
    member_params,
    wind,
)

CFG = Config()
T = 3600.0 * 5


def sample_xy(n=500, seed=0):
    """A spread of positions covering a couple of gyre cells, in metres."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * DOMAIN_M, size=(n, 2))


def speeds(uv):
    return np.linalg.norm(uv, axis=-1)


def test_member_0_reproducible():
    xy = sample_xy()
    assert np.array_equal(current(xy, T, 0, CFG), current(xy, T, 0, CFG))
    assert np.array_equal(wind(xy, T, 0, CFG), wind(xy, T, 0, CFG))


def test_member_params_deterministic_and_seed_dependent():
    assert member_params(3, CFG) == member_params(3, CFG)
    assert member_params(3, CFG) != member_params(3, Config(seed=CFG.seed + 1))


def test_members_differ():
    xy = sample_xy()
    fields = [current(xy, T, m, CFG) for m in range(CFG.n_members)]
    winds = [wind(xy, T, m, CFG) for m in range(CFG.n_members)]
    for m in range(1, CFG.n_members):
        assert not np.allclose(fields[0], fields[m])
        assert not np.allclose(winds[0], winds[m])


def test_members_perturb_windage_and_diffusivity():
    ws = {member_params(m, CFG).windage for m in range(CFG.n_members)}
    ks = {member_params(m, CFG).diffusivity for m in range(CFG.n_members)}
    assert len(ws) == CFG.n_members
    assert len(ks) == CFG.n_members
    assert all(abs(w / CFG.windage - 1.0) <= SPEED_ERR for w in ws)
    assert all(abs(k / CFG.diffusivity - 1.0) <= DIFFUSIVITY_ERR for k in ks)


@pytest.mark.parametrize("member", range(0, 30, 7))
def test_current_magnitude_range(member):
    """~0.1-0.4 m/s. Stagnation points exist, so bound the bulk, not every sample."""
    s = speeds(current(sample_xy(2000), T, member, CFG))
    assert 0.1 <= np.median(s) <= 0.4
    # The design range bounds the bulk; the box corners run faster (0.513 unperturbed),
    # and a member may be SPEED_ERR fast on top of that. Derived, not fitted: the old
    # flat 0.6 was calibrated against a +/-15% scale draw and does not survive +/-30%.
    assert s.max() < 0.55 * (1.0 + SPEED_ERR)
    assert np.mean(s > 0.1) > 0.75


@pytest.mark.parametrize("member", range(0, 30, 7))
def test_wind_magnitude_range(member):
    s = speeds(wind(sample_xy(2000), T, member, CFG))
    assert s.min() >= WIND_SPEED_MS * (1.0 - SPEED_ERR) - WIND_SHEAR
    assert s.max() <= WIND_SPEED_MS * (1.0 + SPEED_ERR) + WIND_SHEAR
    assert 5.0 <= np.median(s) <= 10.0          # CLAUDE.md 6.1's stated range


def test_current_vectorised_over_n2():
    xy = sample_xy(37)
    out = current(xy, T, 2, CFG)
    assert out.shape == xy.shape
    for i in range(0, 37, 9):
        assert np.allclose(out[i], current(xy[i], T, 2, CFG))


def test_wind_vectorised_over_n2():
    xy = sample_xy(37)
    out = wind(xy, T, 2, CFG)
    assert out.shape == xy.shape
    for i in range(0, 37, 9):
        assert np.allclose(out[i], wind(xy[i], T, 2, CFG))


def test_time_dependence_is_slow():
    """The fields move, but not much over one 10-minute drift step."""
    xy = sample_xy()
    dt = CFG.dt_minutes * 60.0
    c0, c1 = current(xy, T, 1, CFG), current(xy, T + dt, 1, CFG)
    w0, w1 = wind(xy, T, 1, CFG), wind(xy, T + dt, 1, CFG)
    assert not np.allclose(c0, c1)
    assert np.abs(c1 - c0).max() < 0.05
    assert np.abs(w1 - w0).max() < 0.5
