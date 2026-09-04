"""The scene panel has to be readable, not just correct.

Only the two things that were actually wrong on screen are tested here: MMSI labels landing
off-panel, and the flow overlay. The rest of plotting.py is matplotlib calls whose output is
a picture, and a test that asserts a picture has some number of artists in it protects
nothing.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from oceaneye.config import Config  # noqa: E402
from oceaneye.plotting import plot_scene  # noqa: E402
from oceaneye.truth import make_scenario  # noqa: E402

FAST = Config(n_members=4, n_particles=300)
SEEDS = (1, 3, 7, 9)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_visible_track_is_labelled_on_screen(ax, seed):
    """The label used to sit at the track's own midpoint, which at this zoom is usually
    off-panel -- so most vessels in the scene rendered as an unlabelled grey line."""
    sc = make_scenario(seed, cfg=FAST)
    plot_scene(ax, sc, t=sc.t_obs, cfg=FAST)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()

    inside = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1  # noqa: E731
    # `.xy` is the annotation's anchor point; the plain Text captions do not have one.
    labelled = {a.get_text() for a in ax.texts
                if hasattr(a, "xy") and inside(*a.xy)}

    visible = {t.mmsi[-4:] for t in sc.tracks
               if any(inside(*p) for p in t.xy / 1e3)}
    assert visible, f"seed {seed} put no track on screen at all"
    assert visible <= labelled, f"unlabelled on screen: {visible - labelled}"


def test_flow_overlay_draws_current_and_wind_at_one_scale(ax):
    """Two quivers, same scale: the wind arrow is only comparable to the current arrows
    if it is drawn through the same length-per-m/s."""
    sc = make_scenario(7, cfg=FAST)
    plot_scene(ax, sc, t=sc.t_obs, cfg=FAST)

    quivers = [c for c in ax.collections if hasattr(c, "scale")]
    assert len(quivers) == 2
    assert quivers[0].scale == quivers[1].scale

    # Wind is drawn at the windage it contributes (~0.2 m/s), not at wind speed (~7.5),
    # so the two arrow sets stay within a small factor of each other.
    cur, wnd = (float(np.mean(np.hypot(q.U, q.V))) for q in quivers)
    assert 0.1 < wnd / cur < 10.0


def test_scene_without_a_time_draws_no_flow(ax):
    """`t=None` is the no-overlay path the calibration figures use."""
    sc = make_scenario(3, cfg=FAST)
    plot_scene(ax, sc)
    assert [c for c in ax.collections if hasattr(c, "scale")] == []
