"""6.8 acceptance. The scoring rule is under test here as much as the arithmetic: a
hidden-polluter trial is correct when H0 ranks top, so H0 is a scoreable answer and not a
residual."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from oceaneye.calibrate import (  # noqa: E402
    PRESENT_FRACTION,
    bin_edges,
    ece,
    fit_isotonic,
    hidden_flags,
    reliability_diagram,
    run_trials,
    score_scenario,
)
from oceaneye.config import Config  # noqa: E402

FAST = Config(n_members=4, n_particles=300)


def synthetic(conf, correct):
    return pd.DataFrame({"confidence": np.asarray(conf, dtype=float),
                         "correct": np.asarray(correct, dtype=bool)})


def test_perfectly_calibrated_scores_near_zero_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.05, 0.95, 20_000)
    df = synthetic(conf, rng.random(20_000) < conf)
    assert ece(df, bins=10) < 0.01


def test_maximally_overconfident_scores_one():
    """Always says 1.0, always wrong -> ECE is exactly 1."""
    assert ece(synthetic(np.ones(100), np.zeros(100))) == pytest.approx(1.0)


@pytest.mark.parametrize("quantile", [False, True])
def test_ece_is_a_real_number_in_the_unit_interval(quantile):
    rng = np.random.default_rng(1)
    df = synthetic(rng.uniform(0.3, 0.6, 300), rng.random(300) < 0.4)
    v = ece(df, bins=10, quantile=quantile)
    assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_quantile_bins_populate_where_equal_width_bins_collapse():
    """The clustered-confidence case: 10 equal-width bins give one populated bin."""
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.44, 0.49, 500)
    eq = bin_edges(conf, 10, quantile=False)
    qu = bin_edges(conf, 10, quantile=True)
    assert int((np.histogram(conf, bins=eq)[0] > 0).sum()) == 1
    assert int((np.histogram(conf, bins=qu)[0] > 0).sum()) >= 8


def test_isotonic_map_is_monotone_and_bounded():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.0, 1.0, 500)
    iso = fit_isotonic(synthetic(conf, rng.random(500) < conf**2))
    out = iso.predict(np.linspace(0.0, 1.0, 50))
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("quantile", [False, True])
def test_reliability_diagram_draws_and_is_labelled(quantile):
    rng = np.random.default_rng(4)
    conf = rng.uniform(0.35, 0.65, 200)
    fig, ax = plt.subplots()
    reliability_diagram(synthetic(conf, rng.random(200) < conf), ax, quantile=quantile)
    assert ax.get_xlabel() and ax.get_ylabel()
    assert ("quantile bins" in ax.get_title()) == quantile
    assert len(ax.lines) >= 2          # diagonal plus the observed curve
    plt.close(fig)


def test_trial_mix_is_about_thirty_percent_hidden():
    """If H0 is never the right answer, the behaviour we most want to defend is unscored."""
    flags = hidden_flags(2000, Config())
    assert 0.25 < flags.mean() < 0.35
    assert abs(flags.mean() - (1.0 - PRESENT_FRACTION)) < 0.03


def test_hidden_flags_are_reproducible():
    assert np.array_equal(hidden_flags(50, Config()), hidden_flags(50, Config()))


def test_hidden_trial_is_correct_exactly_when_h0_ranks_top():
    t = score_scenario(3, hidden=True, cfg=FAST)
    assert t.hidden
    assert t.correct == (t.top == "H0")
    assert 0.0 <= t.confidence <= 1.0


def test_present_trial_is_correct_only_if_the_planted_vessel_ranks_top():
    from oceaneye.truth import make_scenario
    t = score_scenario(3, hidden=False, cfg=FAST)
    assert t.correct == (t.top == make_scenario(3, cfg=FAST).true_mmsi)


def test_confidence_is_the_top_bar_the_ui_shows():
    """Not the vessel probability and not p_unknown -- whichever ranks first."""
    for hidden in (False, True):
        t = score_scenario(7, hidden=hidden, cfg=FAST)
        assert t.confidence == pytest.approx(max(t.p_h0, t.p_top_vessel))
        assert (t.top == "H0") == (t.p_h0 >= t.p_top_vessel)


def test_run_trials_returns_the_scored_pair():
    df = run_trials(3, FAST)
    assert {"confidence", "correct"} <= set(df.columns)
    assert len(df) == 3
    assert df["confidence"].between(0.0, 1.0).all()
    assert df["correct"].dtype == bool


def test_run_trials_is_reproducible():
    a, b = run_trials(3, FAST), run_trials(3, FAST)
    assert np.allclose(a["confidence"], b["confidence"])
    assert list(a["correct"]) == list(b["correct"])
