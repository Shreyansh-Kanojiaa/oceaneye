"""Acceptance checks for 6.4: predicted vs observed slick similarity."""

import numpy as np
import pytest
from test_slick import ellipse_particles

from oceaneye.config import Config
from oceaneye.similarity import CENTROID_SCALE_M, score
from oceaneye.slick import Grid, to_mask

CFG = Config()
# One fixed grid, big enough to hold every translation these tests apply.
GRID = Grid(origin=(-30_000.0, -30_000.0), shape=(300, 300), res_m=200.0)


def blob(a_m=3_000.0, b_m=1_000.0, angle_deg=30.0, centre=(0.0, 0.0), seed=1, n=20_000):
    return to_mask(ellipse_particles(a_m, b_m, angle_deg, n, seed, centre), GRID)


# --- 6.4 acceptance ---

def test_identical_masks_score_exactly_one():
    """Exactly 1.0, no tolerance. Every term is exactly 1.0 and the weights self-normalise."""
    m = blob()
    assert score(m, m, GRID, CFG) == 1.0


def test_identical_masks_score_one_for_any_weights():
    """The weight sum is divided out, so exactness does not depend on weights summing to 1."""
    m = blob()
    odd = Config(sim_weights=(0.3, 0.9, 0.11, 0.07))
    assert score(m, m, GRID, CFG.__class__(sim_weights=(1.0, 1.0, 1.0, 1.0))) == 1.0
    assert score(m, m, GRID, odd) == 1.0


def test_disjoint_masks_score_about_zero():
    a = blob(centre=(0.0, 0.0))
    b = blob(centre=(25_000.0, 0.0))
    assert not (a & b).any(), "masks must actually be disjoint for this test to mean anything"
    assert score(a, b, GRID, CFG) < 0.01


def test_score_decreases_at_every_translation_step(capsys):
    """The check that matters: slide one mask away and watch the score fall, step by step."""
    obs = blob(centre=(0.0, 0.0))
    offsets = np.arange(0.0, 20_001.0, 1_000.0)
    scores = [score(blob(centre=(float(dx), 0.0)), obs, GRID, CFG) for dx in offsets]

    with capsys.disabled():
        print(f"\n  centroid offset -> score   (lambda = {CENTROID_SCALE_M:,.0f} m,"
              f" slick is 3000 x 1000 m)")
        for dx, s in zip(offsets, scores, strict=True):
            bar = "#" * int(round(s * 50))
            print(f"  {dx:7,.0f} m   {s:.6f}  {bar}")

    diffs = np.diff(scores)
    assert (diffs < 0).all(), f"score did not fall at every step: {diffs}"
    assert scores[0] == 1.0


def test_score_is_symmetric():
    a, b = blob(centre=(0.0, 0.0)), blob(centre=(2_000.0, 1_000.0))
    assert score(a, b, GRID, CFG) == pytest.approx(score(b, a, GRID, CFG))


def test_score_stays_in_unit_interval():
    obs = blob()
    for dx in (0.0, 500.0, 4_000.0, 20_000.0):
        for ang in (0.0, 45.0, 90.0):
            s = score(blob(angle_deg=ang, centre=(dx, 0.0)), obs, GRID, CFG)
            assert 0.0 <= s <= 1.0


# --- the individual terms behave ---

def test_rotating_a_mask_lowers_the_score():
    obs = blob(angle_deg=30.0)
    assert score(blob(angle_deg=90.0), obs, GRID, CFG) < score(blob(angle_deg=40.0), obs,
                                                               GRID, CFG) < 1.0


def test_orientation_term_is_modulo_180():
    """A slick rotated by exactly 180 degrees is the same slick, so only IoU should move."""
    obs = blob(angle_deg=10.0)
    s_same = score(blob(angle_deg=10.0), obs, GRID, CFG)
    s_flipped = score(blob(angle_deg=190.0), obs, GRID, CFG)
    assert s_flipped == pytest.approx(s_same, abs=0.02)


def test_size_mismatch_lowers_the_score():
    obs = blob(a_m=3_000.0, b_m=1_000.0)
    same_place_bigger = blob(a_m=6_000.0, b_m=2_000.0)
    assert score(same_place_bigger, obs, GRID, CFG) < 1.0


def test_empty_prediction_scores_zero():
    """Particles that all drifted off the grid predict nothing, and nothing matches nothing."""
    empty = np.zeros(GRID.shape, dtype=bool)
    assert score(empty, blob(), GRID, CFG) == 0.0
    assert score(blob(), empty, GRID, CFG) == 0.0
    assert score(empty, empty, GRID, CFG) == 0.0


def test_prediction_drifted_off_grid_scores_zero_not_nan():
    """A NaN here would silently poison every likelihood it entered in 6.7."""
    gone = blob(centre=(50_000.0, 0.0))
    assert not gone.any()
    s = score(gone, blob(centre=(0.0, 0.0)), GRID, CFG)
    assert s == 0.0 and not np.isnan(s)


def test_masks_on_different_grids_raise():
    other = Grid(origin=(0.0, 0.0), shape=(50, 50), res_m=200.0)
    with pytest.raises(ValueError, match="must share one grid"):
        score(blob(), np.zeros(other.shape, dtype=bool), GRID, CFG)
    with pytest.raises(ValueError, match="do not match grid"):
        score(np.zeros(other.shape, dtype=bool), np.zeros(other.shape, dtype=bool), GRID, CFG)
