"""Acceptance checks for 6.5: Track, interpolation, and confusable synthetic AIS."""

import numpy as np
import pytest

from oceaneye.ais import (
    CONFUSER_ABEAM_M,
    KNOT_MS,
    SPEED_RANGE_KN,
    Track,
    interpolate,
    synthetic_tracks,
)
from oceaneye.config import (
    SCENARIO_EXTENT_M,
    T_END,
    T_START,
    Config,
)
from oceaneye.fields import DOMAIN_M as FIELD_DOMAIN_M
from oceaneye.slick import Grid

CFG = Config()


def closest_approach_m(a: Track, b: Track) -> float:
    """Minimum distance between any sampled point of `a` and any of `b`, metres."""
    return float(np.hypot(*(a.xy[:, None, :] - b.xy[None, :, :]).T).min())


def same_time_separation_m(a: Track, b: Track) -> float:
    """Minimum distance between the two vessels at the *same* instant, metres."""
    assert np.array_equal(a.times, b.times)
    return float(np.hypot(*(a.xy - b.xy).T).min())


# --- 6.5 acceptance ---

def test_interpolation_is_exact_at_knots():
    track = synthetic_tracks(4, CFG, seed=3)[0]
    got = interpolate(track, track.times)
    assert np.array_equal(got, track.xy)


def test_interpolation_is_linear_between_knots():
    track = synthetic_tracks(4, CFG, seed=3)[0]
    mid = 0.5 * (track.times[5] + track.times[6])
    assert interpolate(track, mid) == pytest.approx(0.5 * (track.xy[5] + track.xy[6]))


def test_scalar_time_gives_one_position_array_time_gives_many():
    track = synthetic_tracks(3, CFG, seed=1)[0]
    assert interpolate(track, float(track.times[2])).shape == (2,)
    assert interpolate(track, track.times[2:7]).shape == (5, 2)


def test_interpolate_raises_outside_the_span_rather_than_extrapolating():
    """Loud, not clamped: an out-of-span tau is a bug in 6.7's tau grid."""
    track = synthetic_tracks(3, CFG, seed=1)[0]
    for bad in (track.times[0] - 1.0, track.times[-1] + 1.0, T_START - 86_400.0):
        with pytest.raises(ValueError, match="outside the track"):
            interpolate(track, bad)
    with pytest.raises(ValueError, match="outside the track"):
        interpolate(track, [float(track.times[0]), float(track.times[-1] + 0.5)])


def test_non_monotonic_times_are_rejected():
    t = np.array([0.0, 10.0, 5.0, 20.0])
    with pytest.raises(ValueError, match="strictly ascending"):
        Track(mmsi="1", times=t, xy=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="strictly ascending"):
        Track(mmsi="1", times=np.array([0.0, 10.0, 10.0]), xy=np.zeros((3, 2)))


def test_track_shape_is_validated():
    with pytest.raises(ValueError, match=r"xy must be"):
        Track(mmsi="1", times=np.array([0.0, 1.0]), xy=np.zeros((3, 2)))


def test_tracks_differ_across_seeds():
    a = synthetic_tracks(4, CFG, seed=1)
    b = synthetic_tracks(4, CFG, seed=2)
    assert {t.mmsi for t in a}.isdisjoint({t.mmsi for t in b})
    assert not np.allclose(a[0].xy, b[0].xy)


def test_tracks_are_reproducible_for_one_seed():
    a = synthetic_tracks(4, CFG, seed=9)
    b = synthetic_tracks(4, CFG, seed=9)
    assert all(np.array_equal(x.xy, y.xy) for x, y in zip(a, b, strict=True))


# --- the shared scenario frame (point 3) ---

def test_tracks_are_centred_on_the_box_and_span_the_window():
    """Vessels transit *through* the start box, they are not confined to it.

    Over a 3 h window a merchant ship covers 33-78 km against a 40 km box, so containment
    would need loops or hard turns. What must hold is that mid-window -- where releases sit
    and where the current field has to be valid -- is inside the box.
    """
    box = np.asarray(SCENARIO_EXTENT_M)
    slack = CONFUSER_ABEAM_M[1]          # the confuser is offset abeam of its lead
    for seed in range(6):
        for tr in synthetic_tracks(5, CFG, seed=seed):
            assert tr.times[0] == T_START and tr.times[-1] <= T_END
            mid = tr.xy[len(tr) // 2]
            assert (mid >= -slack).all() and (mid <= box + slack).all()


def test_track_excursion_stays_within_the_measured_field_region():
    """Guards the scale coupling: tracks must not wander where the gyre stops being valid.

    Only x is at risk. The double gyre is genuinely periodic in y -- sin(pi*y) and
    cos(pi*y) continue into a mirrored, counter-rotating cell -- but in x the streamfunction
    carries f = a*x^2 + b*x, whose dfdx term grows linearly and pushes the current out of
    its 0.1-0.4 m/s design range several cells out. Attribution can then sweep tau along a
    whole track without a geometric in-gyre restriction.
    """
    worst = max(float((np.abs(t.xy[:, 0]) / FIELD_DOMAIN_M).max())
                for seed in range(12) for t in synthetic_tracks(5, CFG, seed=seed))
    assert worst < 2.0, (
        f"tracks reach {worst:.1f} gyre cells out in x; the current field is unphysical there")


def test_tracks_share_one_time_base_so_a_grid_covers_them_all():
    tracks = synthetic_tracks(5, CFG, seed=4)
    assert all(np.array_equal(t.times, tracks[0].times) for t in tracks)
    grid = Grid.covering(np.concatenate([t.xy for t in tracks]), CFG)
    for tr in tracks:
        row, col = grid.cell_of(tr.xy)
        h, w = grid.shape
        assert ((row >= 0) & (row < h) & (col >= 0) & (col < w)).all()


def test_speeds_are_plausible_merchant_speeds():
    """Speed between AIS samples is *speed made good* -- the chord, not the arc.

    A turning vessel makes good less than its through-water speed, so the 6-14 kn range of
    6.5 is asserted on the track mean, plus the hard physical bound that no chord can
    exceed the through-water maximum.
    """
    lo, hi = SPEED_RANGE_KN[0] * KNOT_MS, SPEED_RANGE_KN[1] * KNOT_MS
    for seed in range(6):
        for tr in synthetic_tracks(5, CFG, seed=seed):
            made_good = np.hypot(*np.diff(tr.xy, axis=0).T) / np.diff(tr.times)
            assert made_good.max() <= hi + 1e-6
            assert lo <= made_good.mean() <= hi
            assert made_good.min() > 0.9 * lo


def test_course_changes_are_gentle():
    """No sharp turns within one AIS sample interval."""
    for seed in range(6):
        for tr in synthetic_tracks(5, CFG, seed=seed):
            legs = np.diff(tr.xy, axis=0)
            head = np.arctan2(legs[:, 1], legs[:, 0])
            turn = np.abs(np.rad2deg(np.arctan2(np.sin(np.diff(head)), np.cos(np.diff(head)))))
            assert turn.max() < 45.0, f"seed {seed} {tr.mmsi}: {turn.max():.0f} deg in 15 min"


# --- the confuser (point 4) ---

def test_every_scene_contains_a_genuine_confuser():
    """Track 1 shadows track 0: a few km abeam, near-parallel, over the same hours."""
    for seed in range(8):
        tracks = synthetic_tracks(5, CFG, seed=seed)
        sep = same_time_separation_m(tracks[0], tracks[1])
        assert sep < 6_000.0, f"seed {seed}: confuser {sep:,.0f} m away is not confusable"
        head = [np.arctan2(*(t.xy[-1] - t.xy[0])[::-1]) for t in tracks[:2]]
        dtheta = abs(np.rad2deg(np.arctan2(np.sin(head[0] - head[1]),
                                           np.cos(head[0] - head[1]))))
        assert dtheta < 45.0, f"seed {seed}: confuser heading differs by {dtheta:.0f} deg"


def test_confuser_is_not_a_copy_of_the_track_it_shadows():
    tracks = synthetic_tracks(5, CFG, seed=0)
    assert same_time_separation_m(tracks[0], tracks[1]) > 500.0


def test_scene_geometry(capsys):
    """Print the scene so the geometry can be eyeballed, not just asserted."""
    tracks = synthetic_tracks(5, CFG, seed=7)
    box = np.asarray(SCENARIO_EXTENT_M)
    pts = np.concatenate([t.xy for t in tracks])
    lo, hi = pts.min(axis=0) - 2_000.0, pts.max(axis=0) + 2_000.0
    rows, cols = 22, 76

    def cell(xy):
        f = (np.asarray(xy) - lo) / (hi - lo)
        return int((1.0 - f[1]) * (rows - 1)), int(f[0] * (cols - 1))

    canvas = [[" "] * cols for _ in range(rows)]
    # The start box, so the transit-through-the-gyre geometry is visible.
    for f in np.linspace(0.0, 1.0, 400):
        for edge in ((f * box[0], 0.0), (f * box[0], box[1]),
                     (0.0, f * box[1]), (box[0], f * box[1])):
            r, c = cell(edge)
            canvas[r][c] = "."
    for i, tr in enumerate(tracks):
        letter = chr(ord("A") + i)
        for xy in tr.xy:
            r, c = cell(xy)
            canvas[r][c] = letter
        for xy, mark in ((tr.xy[0], letter.lower()), (tr.xy[-1], ">")):
            r, c = cell(xy)
            canvas[r][c] = mark

    with capsys.disabled():
        print(f"\n  view {hi[0] - lo[0]:,.0f} x {hi[1] - lo[1]:,.0f} m, "
              f"{(T_END - T_START) / 3600:.0f} h window, seed 7")
        print(f"  dotted = {box[0]:,.0f} x {box[1]:,.0f} m start box (the gyre cell)")
        print("  lowercase = start, '>' = end.  A and B are the confuser pair.")
        print("  +" + "-" * cols + "+")
        for row in canvas:
            print("  |" + "".join(row) + "|")
        print("  +" + "-" * cols + "+")

        print("\n  closest approach between tracks, metres (geometric):")
        names = [chr(ord("A") + i) for i in range(len(tracks))]
        print("        " + "".join(f"{n:>10}" for n in names))
        for i, a in enumerate(tracks):
            cells = "".join("         ." if i == j else
                            f"{closest_approach_m(a, b):>10,.0f}"
                            for j, b in enumerate(tracks))
            print(f"     {names[i]}  {cells}")
        print("\n  same-time separation A-B (the confuser pair): "
              f"{same_time_separation_m(tracks[0], tracks[1]):,.0f} m")
        def mean_kn(t):
            return np.hypot(*np.diff(t.xy, axis=0).T).mean() / np.diff(t.times).mean() / KNOT_MS

        print("  speeds: " + ", ".join(
            f"{n}={mean_kn(t):.1f} kn" for n, t in zip(names, tracks, strict=True)))

    assert same_time_separation_m(tracks[0], tracks[1]) < 6_000.0
