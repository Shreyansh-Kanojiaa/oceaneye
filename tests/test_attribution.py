"""6.7 acceptance. Most tests run a reduced ensemble for speed; the printed acceptance
test runs the real Config, since a number that came from a toy configuration is not a
number about this system."""

import numpy as np
import pytest

from oceaneye.ais import interpolate
from oceaneye.attribution import (
    GATE_SPEED_MS,
    AttributionResult,
    footprint_frames,
    gate,
    likelihood,
    posterior,
    predicted_footprint,
    tau_grid,
)
from oceaneye.config import T_START, Config, ensemble_members, truth_member
from oceaneye.slick import Grid, morphology
from oceaneye.truth import make_scenario

# Reduced ensemble: the contracts under test (normalisation, gating, tau bounds) do not
# depend on member count, and 30 members x ~70 hypotheses is 10 s a scenario.
FAST = Config(n_members=4, n_particles=300)
SEEDS = (3, 7, 11, 19)


def run(seed, cfg=FAST, hide=False):
    sc = make_scenario(seed, hide_polluter=hide, cfg=cfg)
    return sc, posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)


def overlap_s(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("hide", [False, True])
def test_vessel_mass_plus_h0_is_exactly_one(seed, hide):
    """The non-negotiable: H0 is normalised alongside the vessels, not bolted on after."""
    _, r = run(seed, hide=hide)
    assert abs(r.table["p"].sum() + r.p_unknown - 1.0) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_vessel_probabilities_do_not_sum_to_one(seed):
    _, r = run(seed)
    assert r.table["p"].sum() < 1.0
    assert r.p_unknown > 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_vessel_is_top_ranked(seed):
    """Ranked by marginal probability -- tau is a nuisance parameter for this question."""
    sc, r = run(seed)
    assert r.by_vessel.index[0] == sc.true_mmsi


@pytest.mark.parametrize("seed", SEEDS)
def test_recovered_tau_overlaps_the_true_tau(seed):
    """Best window on the top-ranked vessel, not the best row over all vessels."""
    sc, r = run(seed)
    rows = r.table[r.table["mmsi"] == r.by_vessel.index[0]]
    best = rows.iloc[0]
    assert overlap_s((best["tau_start"], best["tau_end"]), sc.true_tau) > 0.0


@pytest.mark.parametrize("seed", (3, 7))
def test_hidden_polluter_puts_h0_on_top(seed):
    """The feature, not a bug to tune away: with no correct answer present, decline.

    Runs the real Config, unlike its neighbours. How far H0 clears the best surviving
    vessel depends directly on ensemble spread, so a 4-member ensemble is measuring a
    different system, not a cheaper version of this one -- it flips seed 7. Seed 7 is the
    hard case either way: the planted vessel is the confuser, so hiding it leaves its
    parallel lead a few km abeam as a genuinely good stand-in.
    """
    sc, r = run(seed, cfg=Config(), hide=True)
    assert sc.true_mmsi is None
    assert r.p_unknown > r.by_vessel.max()
    assert r.p_unknown > r.table["p"].max()


@pytest.mark.parametrize("seed", SEEDS)
def test_gate_never_drops_the_true_vessel(seed):
    """A gate that can drop the answer is not a cost saver, it is a second classifier."""
    sc = make_scenario(seed, cfg=FAST)
    kept = gate(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, FAST)
    assert sc.true_mmsi in [t.mmsi for t in kept]


def test_gate_actually_rejects_something():
    """If it never rejects, it is dead code and every scenario pays for all five tracks."""
    rejected = 0
    for seed in SEEDS:
        sc = make_scenario(seed, cfg=FAST)
        rejected += len(sc.tracks) - len(gate(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, FAST))
    assert rejected > 0


def test_empty_observation_raises():
    sc = make_scenario(3, cfg=FAST)
    with pytest.raises(ValueError, match="empty observed mask"):
        gate(np.zeros_like(sc.obs_mask), sc.tracks, sc.t_obs, sc.grid, FAST)


def test_slick_out_of_reach_of_every_track_gives_p_unknown_one():
    """Not 'nearly 1' -- with no candidate hypotheses the sum is empty and H0 takes all."""
    sc = make_scenario(3, cfg=FAST)
    far = Grid(origin=(2_000_000.0, 2_000_000.0), shape=(20, 20), res_m=FAST.grid_res_m)
    mask = np.zeros(far.shape, dtype=bool)
    mask[8:12, 8:12] = True
    r = posterior(mask, sc.tracks, sc.t_obs, far, FAST)
    assert r.table.empty
    assert r.best is None
    assert r.p_unknown == 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_tau_grid_stays_inside_the_track_and_ends_by_observation(seed):
    sc = make_scenario(seed, cfg=FAST)
    for track in sc.tracks:
        for t0, t1 in tau_grid(track, sc.t_obs):
            assert track.times[0] <= t0 < t1 <= min(track.times[-1], sc.t_obs)


def test_gate_speed_exceeds_the_fastest_parcel_the_fields_can_produce():
    """The gate radius is only safe if nothing can outrun it."""
    from oceaneye.fields import SPEED_ERR, WIND_SHEAR, WIND_SPEED_MS, current, member_params
    cfg = Config()
    xy = np.stack(np.meshgrid(np.linspace(-6e4, 1.2e5, 60), np.linspace(-6e4, 6e4, 60)),
                  axis=-1).reshape(-1, 2)
    peak = max(
        np.hypot(*current(xy, T_START + h * 3600.0, m, cfg).T).max()
        + member_params(m, cfg).windage * (WIND_SPEED_MS * (1 + SPEED_ERR) + WIND_SHEAR)
        for m in range(cfg.n_members) for h in (0, 3, 6)
    )
    assert peak < GATE_SPEED_MS


def test_posterior_is_reproducible():
    _, a = run(7)
    _, b = run(7)
    assert np.allclose(a.table["p"], b.table["p"])
    assert a.p_unknown == b.p_unknown


def test_likelihood_never_touches_the_held_out_member():
    """Structural check: the observation's member is outside what likelihood averages."""
    cfg = FAST
    assert truth_member(cfg) not in ensemble_members(cfg)
    sc = make_scenario(3, cfg=cfg)
    track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
    assert 0.0 <= likelihood(track, sc.true_tau, sc.obs_mask, sc.t_obs, sc.grid, cfg) <= 1.0


def test_full_fidelity_acceptance_report(capsys):
    """Printed acceptance at the real Config, one normal scene and its hidden twin."""
    cfg = Config()
    sc = make_scenario(7, cfg=cfg)
    r = posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)
    hid = make_scenario(7, hide_polluter=True, cfg=cfg)
    rh = posterior(hid.obs_mask, hid.tracks, hid.t_obs, hid.grid, cfg)
    mins = lambda t: (np.asarray(t) - T_START) / 60.0  # noqa: E731

    with capsys.disabled():
        print(f"\n  seed 7, {cfg.n_members} members, {cfg.n_particles} particles")
        print(f"  truth: mmsi {sc.true_mmsi}, tau "
              f"{mins(sc.true_tau)[0]:.1f}-{mins(sc.true_tau)[1]:.1f} min after window start")
        print(f"  sum(p) + p_unknown = {r.table['p'].sum() + r.p_unknown:.12f}")
        print(f"  p_unknown          = {r.p_unknown:.4f}")
        print("  per vessel:")
        for mmsi, p in r.by_vessel.items():
            print(f"    {mmsi}  {p:.4f}{'   <-- planted' if mmsi == sc.true_mmsi else ''}")
        print("  top (vessel, tau) rows:")
        for _, row in r.table.head(6).iterrows():
            print(f"    {row['mmsi']}  tau {mins(row['tau_start']):6.1f}-"
                  f"{mins(row['tau_end']):6.1f}  L {row['likelihood']:.4f}  p {row['p']:.4f}")
        print(f"  hidden polluter: p_unknown {rh.p_unknown:.4f} vs best vessel "
              f"{rh.by_vessel.max():.4f}")

    assert r.by_vessel.index[0] == sc.true_mmsi
    assert rh.p_unknown > rh.by_vessel.max()


def test_result_type():
    _, r = run(3)
    assert isinstance(r, AttributionResult)


def test_frames_travel_from_the_release_site_to_the_observed_slick():
    """The drift clock's whole claim: the ocean moves the oil off the track that laid it.

    If the first and last frames sat in the same place the scrubber would be decoration,
    and geometry-only attribution would be right.
    """
    sc = make_scenario(7, cfg=FAST)
    track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
    times, frames = footprint_frames(track, sc.true_tau, sc.t_obs, sc.grid, FAST, n_frames=8)

    assert times.shape == (8,) and frames.shape == (8, *sc.grid.shape)
    assert times[0] > sc.true_tau[0] and times[-1] == pytest.approx(sc.t_obs)
    assert frames.min() >= 0.0 and frames.max() <= 1.0

    obs = morphology(sc.obs_mask, sc.grid)["centroid_xy"]
    release = interpolate(track, np.array([sc.true_tau[0]]))[0]
    first = morphology(frames[0] > 0, sc.grid)["centroid_xy"]
    last = morphology(frames[-1] > 0, sc.grid)["centroid_xy"]

    assert np.hypot(*(first - release)) < np.hypot(*(last - release))   # it travelled
    assert np.hypot(*(last - obs)) < np.hypot(*(first - obs))           # towards the slick


def test_predicted_footprint_is_the_last_frame():
    sc = make_scenario(3, cfg=FAST)
    track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
    one = predicted_footprint(track, sc.true_tau, sc.t_obs, sc.grid, FAST)
    assert one == pytest.approx(footprint_frames(
        track, sc.true_tau, sc.t_obs, sc.grid, FAST)[1][0])
