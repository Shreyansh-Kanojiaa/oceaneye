"""Acceptance checks for CLAUDE.md section 6.2."""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from oceaneye.config import Config
from oceaneye.drift import Particles, advect, seed_line_source
from oceaneye.fields import member_params

CFG = Config()
T0 = 1_725_000_000.0   # unix seconds UTC


@dataclass
class StubTrack:
    """Minimal stand-in for ais.Track (6.5). Same field names, straight line at 8 kn."""

    mmsi: str
    times: np.ndarray
    xy: np.ndarray


def straight_track(speed_ms=4.1, heading_rad=0.6, hours=12.0, n_knots=25) -> StubTrack:
    times = T0 + np.linspace(0.0, hours * 3600.0, n_knots)
    d = (times - T0) * speed_ms
    xy = np.stack([d * np.cos(heading_rad), d * np.sin(heading_rad)], axis=-1)
    return StubTrack(mmsi="000000001", times=times, xy=xy)


def zero_fields(monkeypatch):
    """Silence both fields so only diffusivity can move a particle."""
    monkeypatch.setattr("oceaneye.drift.current", lambda xy, t, m, c: np.zeros_like(xy))
    monkeypatch.setattr("oceaneye.drift.wind", lambda xy, t, m, c: np.zeros_like(xy))


# --- seeding -------------------------------------------------------------------


def test_seeds_lie_along_the_track_not_at_a_point():
    track = straight_track()
    tau = (T0 + 3600.0, T0 + 4 * 3600.0)
    p = seed_line_source(track, tau, 400, np.random.default_rng(0))

    # Every seed sits on the track at its own seed time.
    expected = np.stack(
        [np.interp(p.t_seed, track.times, track.xy[:, i]) for i in (0, 1)], axis=-1
    )
    assert np.allclose(p.xy, expected)

    # The cloud spans the distance the vessel covered during tau, not a single point.
    span = np.linalg.norm(p.xy.max(axis=0) - p.xy.min(axis=0))
    sailed = 4.1 * (tau[1] - tau[0])
    assert span == pytest.approx(sailed, rel=0.02)

    midpoint = np.interp(np.mean(tau), track.times, track.xy[:, 0])
    assert np.std(p.xy[:, 0]) > 0.2 * abs(sailed * np.cos(0.6))
    assert not np.allclose(p.xy[:, 0], midpoint)


def test_seed_times_cover_tau_uniformly():
    tau = (T0 + 3600.0, T0 + 5 * 3600.0)
    p = seed_line_source(straight_track(), tau, 500, np.random.default_rng(1))
    assert len(p) == 500
    assert p.t_seed.min() >= tau[0]
    assert p.t_seed.max() <= tau[1]
    # Stratified draws: each decile of the window gets ~10% of the particles.
    counts, _ = np.histogram(p.t_seed, bins=10, range=tau)
    assert counts.min() >= 45 and counts.max() <= 55


def test_seeding_is_reproducible_and_rng_driven():
    track, tau = straight_track(), (T0 + 3600.0, T0 + 4 * 3600.0)
    a = seed_line_source(track, tau, 100, np.random.default_rng(3))
    b = seed_line_source(track, tau, 100, np.random.default_rng(3))
    c = seed_line_source(track, tau, 100, np.random.default_rng(4))
    assert np.array_equal(a.xy, b.xy)
    assert not np.array_equal(a.xy, c.xy)


def test_zero_duration_tau_is_a_point_release():
    t = T0 + 3600.0
    p = seed_line_source(straight_track(), (t, t), 50, np.random.default_rng(0))
    assert np.allclose(p.xy, p.xy[0])


def test_tau_outside_track_range_raises():
    track = straight_track()
    with pytest.raises(ValueError, match="outside the track"):
        seed_line_source(track, (T0 - 3600.0, T0 + 3600.0), 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="outside the track"):
        seed_line_source(track, (T0, track.times[-1] + 1.0), 10, np.random.default_rng(0))


# --- advection -----------------------------------------------------------------


def test_particle_count_conserved():
    p = seed_line_source(straight_track(), (T0, T0 + 3600.0), 400, np.random.default_rng(0))
    out = advect(p, T0, T0 + 6 * 3600.0, 0, CFG)
    assert len(out) == len(p) == 400
    assert out.xy.shape == p.xy.shape
    assert np.array_equal(out.t_seed, p.t_seed)


def test_dead_ocean_does_not_move_particles(monkeypatch):
    """Zero current + zero wind + zero diffusivity => particles do not move."""
    zero_fields(monkeypatch)
    cfg = Config(diffusivity=0.0)
    p = seed_line_source(straight_track(), (T0, T0 + 3600.0), 400, np.random.default_rng(0))
    out = advect(p, T0, T0 + 8 * 3600.0, 0, cfg)
    assert np.array_equal(out.xy, p.xy)


def test_zero_diffusivity_alone_still_advects():
    """Guards against the dead-ocean test passing for the wrong reason."""
    cfg = Config(diffusivity=0.0)
    p = seed_line_source(straight_track(), (T0, T0 + 3600.0), 200, np.random.default_rng(0))
    out = advect(p, T0, T0 + 6 * 3600.0, 0, cfg)
    assert np.linalg.norm(out.xy - p.xy, axis=-1).min() > 100.0


def test_unreleased_particles_stay_put():
    """A particle does not drift before its own seed time."""
    tau = (T0 + 2 * 3600.0, T0 + 6 * 3600.0)
    p = seed_line_source(straight_track(), tau, 400, np.random.default_rng(0))
    stop = T0 + 4 * 3600.0
    out = advect(p, tau[0], stop, 0, CFG)
    moved = np.linalg.norm(out.xy - p.xy, axis=-1)
    assert np.all(moved[p.t_seed > stop] == 0.0)
    assert np.all(moved[p.t_seed <= stop - 3600.0] > 0.0)


def test_later_release_means_shorter_trajectory():
    """The line-source point: drift time varies along tau, so path length must too.

    Compared from a single start position -- particles seeded kilometres apart sit in
    different parts of the gyre, where net displacement is not a proxy for drift time.
    """
    cfg = Config(diffusivity=0.0)   # isolate advection from the random walk
    delays = np.array([0.0, 2.0, 4.0, 6.0, 8.0]) * 3600.0
    p = Particles(xy=np.full((5, 2), 5000.0), t_seed=T0 + delays)

    xy, path, t, t_obs = p.xy.copy(), np.zeros(5), T0, T0 + 10 * 3600.0
    while t < t_obs:
        nxt = min(t + 600.0, t_obs)
        stepped = advect(Particles(xy, p.t_seed), t, nxt, 0, cfg)
        path += np.linalg.norm(stepped.xy - xy, axis=1)
        xy, t = stepped.xy, nxt

    assert np.all(np.diff(path) < 0.0)
    assert path[0] > 5.0 * path[-1]


def test_advection_is_reproducible_and_member_dependent():
    p = seed_line_source(straight_track(), (T0, T0 + 3600.0), 300, np.random.default_rng(0))
    t1 = T0 + 6 * 3600.0
    a = advect(p, T0, t1, 0, CFG)
    b = advect(p, T0, t1, 0, CFG)
    c = advect(p, T0, t1, 1, CFG)
    assert np.array_equal(a.xy, b.xy)
    assert not np.allclose(a.xy, c.xy)
    # An explicit rng overrides the derived one (non-negotiable 5: no global state).
    d = advect(p, T0, t1, 0, CFG, rng=np.random.default_rng(99))
    assert not np.allclose(a.xy, d.xy)


def test_displacement_is_physically_plausible():
    """6 h under a ~0.2 m/s ocean plus 3% windage should move oil a few kilometres."""
    p = seed_line_source(straight_track(), (T0, T0 + 600.0), 400, np.random.default_rng(0))
    out = advect(p, T0, T0 + 6 * 3600.0, 0, CFG)
    d = np.linalg.norm(out.xy - p.xy, axis=-1)
    assert 1_000.0 < np.median(d) < 15_000.0


def test_diffusion_matches_random_walk_theory():
    """Over a short window, before shear matters, sigma ~ sqrt(2*K*T) per axis."""
    cfg = Config(diffusivity=12.0)
    k = member_params(0, cfg).diffusivity
    n, hours = 20_000, 1.0
    p = Particles(xy=np.zeros((n, 2)), t_seed=np.full(n, T0))
    out = advect(p, T0, T0 + hours * 3600.0, 0, cfg)
    expected = np.sqrt(2.0 * k * hours * 3600.0)
    assert np.allclose(np.std(out.xy, axis=0), expected, rtol=0.15)


def test_diffusion_keeps_spreading_and_scales_with_k():
    n = 4000
    p = Particles(xy=np.zeros((n, 2)), t_seed=np.full(n, T0))
    spread = lambda o: np.linalg.norm(np.std(o.xy, axis=0))  # noqa: E731
    short = spread(advect(p, T0, T0 + 2 * 3600.0, 0, Config(diffusivity=12.0)))
    long = spread(advect(p, T0, T0 + 8 * 3600.0, 0, Config(diffusivity=12.0)))
    strong = spread(advect(p, T0, T0 + 2 * 3600.0, 0, Config(diffusivity=48.0)))
    assert long > short
    assert strong > short


def test_no_time_travel():
    p = seed_line_source(straight_track(), (T0, T0 + 3600.0), 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="forward only"):
        advect(p, T0 + 3600.0, T0, 0, CFG)


def test_zero_length_advection_is_a_no_op():
    p = seed_line_source(straight_track(), (T0, T0 + 3600.0), 50, np.random.default_rng(0))
    out = advect(p, T0, T0, 0, CFG)
    assert np.array_equal(out.xy, p.xy)


def test_vectorised_no_python_loop_over_particles():
    """100k particles x 36 steps. A per-particle loop is 3.6M iterations and blows this."""
    n = 100_000
    p = Particles(xy=np.zeros((n, 2)), t_seed=np.full(n, T0))
    start = time.perf_counter()
    out = advect(p, T0, T0 + 6 * 3600.0, 0, CFG)
    elapsed = time.perf_counter() - start
    assert len(out) == n
    assert elapsed < 3.0, f"advect took {elapsed:.2f}s for {n} particles"
