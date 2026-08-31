"""Acceptance checks for 6.6: simulate-and-recover scenarios."""

import dataclasses

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import ConvexHull

from oceaneye.config import Config, ensemble_members, truth_member
from oceaneye.drift import advect, seed_line_source
from oceaneye.fields import DOMAIN_M
from oceaneye.slick import morphology, to_mask
from oceaneye.truth import FIELD_VALID_CELLS, Scenario, make_scenario

CFG = Config()
STRUCT = np.ones((3, 3), bool)     # 8-connectivity


# --- 6.6 acceptance ---

def test_observed_mask_is_non_empty_and_plausibly_sized():
    for seed in range(8):
        sc = make_scenario(seed)
        assert sc.obs_mask.any()
        area_km2 = morphology(sc.obs_mask, sc.grid)["area_m2"] / 1e6
        assert 1.0 < area_km2 < 200.0, f"seed {seed}: {area_km2:.1f} km^2 is not a slick"


def test_hidden_polluter_is_absent_from_tracks():
    for seed in range(8):
        shown, hidden = make_scenario(seed), make_scenario(seed, hide_polluter=True)
        assert shown.true_mmsi is not None
        assert hidden.true_mmsi is None
        assert shown.true_mmsi not in {t.mmsi for t in hidden.tracks}
        assert len(hidden.tracks) == len(shown.tracks) - 1


def test_visible_polluter_is_present_and_named():
    for seed in range(8):
        sc = make_scenario(seed)
        assert sc.true_mmsi in {t.mmsi for t in sc.tracks}


def test_hiding_the_polluter_does_not_change_the_observation():
    """The slick is the same; only the candidate list differs."""
    for seed in range(4):
        shown, hidden = make_scenario(seed), make_scenario(seed, hide_polluter=True)
        assert np.array_equal(shown.obs_mask, hidden.obs_mask)
        assert shown.true_tau == hidden.true_tau


def test_scenarios_are_reproducible_and_differ_across_seeds():
    a, b = make_scenario(5), make_scenario(5)
    assert np.array_equal(a.obs_mask, b.obs_mask) and a.true_tau == b.true_tau
    c = make_scenario(6)
    assert not np.array_equal(a.obs_mask, c.obs_mask)


# --- point 1: the held-out member is structural ---

def test_truth_member_is_outside_the_range_attribution_averages_over():
    """Not a convention -- the observation's member is not in `ensemble_members`."""
    for cfg in (Config(), Config(n_members=8), Config(n_members=1)):
        assert truth_member(cfg) not in ensemble_members(cfg)


def test_observation_differs_from_every_member_attribution_can_use():
    """A member attribution averages over must not reproduce the observation exactly."""
    sc = make_scenario(2)
    track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
    rng = np.random.default_rng(0)
    for e in list(ensemble_members(CFG))[:8]:
        p = seed_line_source(track, sc.true_tau, CFG.n_particles, rng)
        pred = to_mask(advect(p, sc.true_tau[0], sc.t_obs, e, CFG), sc.grid)
        assert not np.array_equal(pred, sc.obs_mask)


# --- point 2: same mask-formation path as 6.7 predictions ---

def test_observed_mask_uses_the_configured_particle_count():
    """Truth is not better resolved than a prediction: both use cfg.n_particles."""
    lean = make_scenario(1, cfg=Config(n_particles=200))
    rich = make_scenario(1, cfg=Config(n_particles=6400))
    assert rich.obs_mask.sum() > 2 * lean.obs_mask.sum()


def test_observed_and_predicted_masks_share_the_scenario_grid():
    sc = make_scenario(4)
    assert sc.obs_mask.shape == tuple(sc.grid.shape)
    track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
    p = seed_line_source(track, sc.true_tau, CFG.n_particles, np.random.default_rng(1))
    pred = to_mask(advect(p, sc.true_tau[0], sc.t_obs, 0, CFG), sc.grid)
    assert pred.shape == sc.obs_mask.shape


# --- releases sit where the field is valid ---

def test_release_is_on_the_track_and_before_the_observation():
    for seed in range(8):
        sc = make_scenario(seed)
        track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
        t0, t1 = sc.true_tau
        assert track.times[0] <= t0 < t1 <= track.times[-1]
        assert t1 < sc.t_obs


def test_slick_stays_in_the_well_behaved_field_region():
    for seed in range(12):
        sc = make_scenario(seed)
        xs = sc.grid.centres_of(*np.nonzero(sc.obs_mask))[:, 0]
        assert np.abs(xs).max() < FIELD_VALID_CELLS * DOMAIN_M


# --- point 3: the mask is a slick, not confetti ---

def test_observed_mask_is_mostly_one_connected_blob(capsys):
    fills, comps, larges = [], [], []
    for seed in range(8):
        sc = make_scenario(seed)
        m = sc.obs_mask
        lab, n = ndimage.label(m, structure=STRUCT)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        pts = np.argwhere(m).astype(float)
        fills.append(m.sum() / ConvexHull(pts).volume)
        comps.append(n)
        larges.append(sizes.max() / m.sum())

    with capsys.disabled():
        print(f"\n  observed mask at n_particles={CFG.n_particles}, 8 seeds:")
        print(f"    fill fraction (of convex hull)  {np.mean(fills):.1%}")
        print(f"    connected components            {np.mean(comps):.1f}")
        print(f"    cells in the largest component  {np.mean(larges):.1%}")

    assert np.mean(fills) > 0.55
    assert np.mean(larges) > 0.90


def test_scenario_is_frozen():
    assert Scenario.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_scenario(0).t_obs = 0.0
