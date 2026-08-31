"""Acceptance checks for 6.3: rasterising particles and recovering slick shape."""

import numpy as np
import pytest

from oceaneye.config import Config
from oceaneye.drift import Particles
from oceaneye.slick import Grid, morphology, to_mask

CFG = Config()


def ellipse_particles(a_m, b_m, angle_deg, n, seed, centre=(0.0, 0.0)):
    """`n` particles filling an ellipse with semi-axes (a_m, b_m) rotated by `angle_deg`."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2 * np.pi, n)
    xy = np.stack([a_m * r * np.cos(th), b_m * r * np.sin(th)], axis=-1)
    c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    rot = np.array([[c, -s], [s, c]])
    return Particles(xy=xy @ rot.T + np.asarray(centre), t_seed=np.zeros(n))


def angle_error_deg(got_rad, want_deg):
    """Smallest angle between two orientations, in degrees, **modulo 180**."""
    d = (np.rad2deg(got_rad) - want_deg) % 180.0
    return min(d, 180.0 - d)


# --- 6.3 acceptance: known ellipse recovers area within 10% and orientation within 10 deg ---

def test_ellipse_recovers_area_and_orientation(capsys):
    a_m, b_m, angle_deg = 3_000.0, 1_000.0, 30.0
    true_area = np.pi * a_m * b_m
    p = ellipse_particles(a_m, b_m, angle_deg, n=20_000, seed=1)
    grid = Grid.covering(p.xy, CFG)
    m = morphology(to_mask(p, grid), grid)

    area_err = abs(m["area_m2"] - true_area) / true_area
    ang_err = angle_error_deg(m["orientation_rad"], angle_deg)
    with capsys.disabled():
        print(f"\n  grid            {grid.shape} cells @ {grid.res_m:.0f} m,"
              f" origin ({grid.origin[0]:.0f}, {grid.origin[1]:.0f})")
        print(f"  area            true {true_area:12,.0f} m^2   "
              f"recovered {m['area_m2']:12,.0f} m^2   err {area_err:6.2%}")
        print(f"  orientation     true {angle_deg:6.2f} deg        "
              f"recovered {np.rad2deg(m['orientation_rad']):6.2f} deg        "
              f"err {ang_err:5.2f} deg (mod 180)")
        print(f"  elongation      true {a_m / b_m:6.2f}            "
              f"recovered {m['elongation']:6.2f}")
        print(f"  centroid        {m['centroid_xy'].round(1)} m")
        print(f"  perimeter       {m['perimeter_m']:,.0f} m (staircase)")
    assert area_err < 0.10
    assert ang_err < 10.0


@pytest.mark.parametrize("angle_deg", [0.0, 10.0, 30.0, 75.0, 120.0, 170.0])
def test_orientation_recovered_at_many_angles(angle_deg):
    p = ellipse_particles(3_000.0, 1_000.0, angle_deg, n=20_000, seed=2)
    grid = Grid.covering(p.xy, CFG)
    m = morphology(to_mask(p, grid), grid)
    assert angle_error_deg(m["orientation_rad"], angle_deg) < 10.0


def test_orientation_is_modulo_180():
    """A slick at 10 deg and one at 190 deg are the same orientation."""
    grid = Grid(origin=(-6_000.0, -6_000.0), shape=(60, 60), res_m=200.0)
    a = morphology(to_mask(ellipse_particles(3_000.0, 800.0, 10.0, 20_000, 3), grid), grid)
    b = morphology(to_mask(ellipse_particles(3_000.0, 800.0, 190.0, 20_000, 3), grid), grid)
    assert 0.0 <= a["orientation_rad"] < np.pi
    assert a["orientation_rad"] == pytest.approx(b["orientation_rad"], abs=1e-9)


def test_area_scales_with_ellipse_size():
    """Both ellipses are >= 10 cells across the minor axis, i.e. actually resolved."""
    grid = Grid(origin=(-24_000.0, -24_000.0), shape=(240, 240), res_m=200.0)
    small = morphology(to_mask(ellipse_particles(3_000.0, 1_000.0, 45.0, 40_000, 4), grid), grid)
    big = morphology(to_mask(ellipse_particles(6_000.0, 2_000.0, 45.0, 40_000, 4), grid), grid)
    assert big["area_m2"] / small["area_m2"] == pytest.approx(4.0, rel=0.10)


def test_area_bias_shrinks_with_cell_size(capsys):
    """Residual area bias is a resolution floor, ~O(res / minor axis), not a fixed offset.

    A slick thinner than ~10 cells across is not resolved: the same 3x1 km ellipse reads
    +17% at 400 m cells and +1% at 100 m. Pins the boundary correction against regression.
    """
    a_m, b_m = 3_000.0, 1_000.0
    true_area = np.pi * a_m * b_m
    p = ellipse_particles(a_m, b_m, 45.0, n=60_000, seed=9)
    errs = []
    for res_m in (400.0, 200.0, 100.0):
        n = int(24_000 / res_m)
        grid = Grid(origin=(-12_000.0, -12_000.0), shape=(n, n), res_m=res_m)
        area = morphology(to_mask(p, grid), grid)["area_m2"]
        errs.append((area - true_area) / true_area)
        with capsys.disabled():
            print(f"\n  res {res_m:5.0f} m  ({2 * b_m / res_m:4.0f} cells across minor axis)"
                  f"   area {area:12,.0f} m^2   err {errs[-1]:+7.2%}")
    assert errs[0] > errs[1] > errs[2] > 0.0
    assert errs[2] < 0.03


def test_disc_is_not_elongated():
    p = ellipse_particles(2_000.0, 2_000.0, 0.0, n=20_000, seed=5)
    grid = Grid.covering(p.xy, CFG)
    assert morphology(to_mask(p, grid), grid)["elongation"] == pytest.approx(1.0, abs=0.05)


# --- grid and rasterising ---

def test_grid_is_fixed_not_derived_per_cloud():
    """Two clouds rasterised on one grid stay in the same coordinate frame."""
    grid = Grid(origin=(0.0, 0.0), shape=(50, 50), res_m=200.0)
    here = Particles(xy=np.array([[1_000.0, 1_000.0]]), t_seed=np.zeros(1))
    there = Particles(xy=np.array([[5_000.0, 3_000.0]]), t_seed=np.zeros(1))
    assert np.argwhere(to_mask(here, grid)).tolist() == [[5, 5]]
    assert np.argwhere(to_mask(there, grid)).tolist() == [[15, 25]]


def test_particles_off_grid_are_dropped_not_clamped():
    grid = Grid(origin=(0.0, 0.0), shape=(10, 10), res_m=200.0)
    p = Particles(xy=np.array([[-5_000.0, 500.0], [500.0, 500.0], [9e4, 9e4]]),
                  t_seed=np.zeros(3))
    mask = to_mask(p, grid)
    assert mask.sum() == 1
    assert mask[2, 2]


def test_covering_contains_every_particle():
    p = ellipse_particles(4_000.0, 900.0, 60.0, n=5_000, seed=6, centre=(20_000.0, -8_000.0))
    grid = Grid.covering(p.xy, CFG)
    row, col = grid.cell_of(p.xy)
    h, w = grid.shape
    assert ((row >= 0) & (row < h) & (col >= 0) & (col < w)).all()
    assert to_mask(p, grid).sum() > 0


def test_empty_mask_is_zero_area_and_nan_shape():
    grid = Grid(origin=(0.0, 0.0), shape=(10, 10), res_m=200.0)
    m = morphology(np.zeros((10, 10), dtype=bool), grid)
    assert m["area_m2"] == 0.0
    assert m["perimeter_m"] == 0.0
    assert np.isnan(m["orientation_rad"])
    assert np.isnan(m["centroid_xy"]).all()


def test_centroid_matches_ellipse_centre():
    centre = (7_000.0, -3_000.0)
    p = ellipse_particles(2_500.0, 1_200.0, 20.0, n=20_000, seed=7, centre=centre)
    grid = Grid.covering(p.xy, CFG)
    got = morphology(to_mask(p, grid), grid)["centroid_xy"]
    assert got == pytest.approx(np.asarray(centre), abs=0.05 * 2_500.0)


def test_perimeter_grows_with_area():
    grid = Grid(origin=(-12_000.0, -12_000.0), shape=(120, 120), res_m=200.0)
    small = morphology(to_mask(ellipse_particles(1_500.0, 1_500.0, 0.0, 40_000, 8), grid), grid)
    big = morphology(to_mask(ellipse_particles(3_000.0, 3_000.0, 0.0, 40_000, 8), grid), grid)
    assert big["perimeter_m"] > small["perimeter_m"]
