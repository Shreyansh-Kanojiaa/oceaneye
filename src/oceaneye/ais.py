"""Vessel tracks: the `Track` record, interpolation, and synthetic AIS.

Positions are projected metres in the shared scenario frame (`config.SCENARIO_EXTENT_M`,
origin at (0, 0)); times are unix seconds UTC inside `[config.T_START, config.T_END]`.
Everything downstream -- the drift domain, the `Grid` from 6.3, the observation time --
uses that same frame, so tracks and masks are directly comparable.

Synthetic tracks are deliberately *confusable*. A scene of well-separated vessels makes
6.7's top-1 check trivial and proves nothing, so every scene contains at least one
confuser: a track shadowing another a few km abeam, on a similar heading, overlapping in
time. Attribution has to separate those on drift physics, which is the point of the project.
"""

from dataclasses import dataclass

import numpy as np

from .config import (
    SCENARIO_EXTENT_M,
    SCENARIO_MARGIN_M,
    T_END,
    T_START,
    Config,
)

# Calibration knobs, module-level as in fields.py.
KNOT_MS = 0.514_444                  # one knot in m/s
SPEED_RANGE_KN = (6.0, 14.0)         # merchant transit speeds
SAMPLE_S = 900.0                     # AIS resample interval, seconds (15 min)
INTEGRATE_S = 60.0                   # internal integration step, seconds
TURN_STD_DEG_PER_STEP = 0.8          # gentle course changes, degrees per integration step
CONFUSER_ABEAM_M = (2_000.0, 5_000.0)   # lateral offset of the confuser, metres


@dataclass(frozen=True)
class Track:
    """One vessel's AIS history. `times` unix seconds UTC ascending, `xy` (n, 2) metres.

    This is the only track type in the repo: `drift.seed_line_source` takes it directly.
    """

    mmsi: str
    times: np.ndarray
    xy: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        xy = np.asarray(self.xy, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError(f"times must be 1-D with >= 2 knots, got shape {times.shape}")
        if xy.shape != (times.size, 2):
            raise ValueError(f"xy must be ({times.size}, 2), got {xy.shape}")
        if not np.all(np.diff(times) > 0):
            raise ValueError(f"times must be strictly ascending (mmsi {self.mmsi})")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "xy", xy)

    def __len__(self) -> int:
        return self.times.size


def interpolate(track: Track, t) -> np.ndarray:
    """Position(s) on `track` at time(s) `t`, unix seconds UTC. Linear between knots.

    Returns (2,) for scalar `t`, (n, 2) for array `t`.

    **Raises** on any `t` outside the track's span. It does not extrapolate and it does not
    clamp: a tau outside a track's time range is a bug in 6.7's tau grid, and a clamped
    position would quietly seed oil at the vessel's first or last known point as though
    that were observed. Loud is correct here.
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    lo, hi = float(track.times[0]), float(track.times[-1])
    bad = (t_arr < lo) | (t_arr > hi)
    if bad.any():
        raise ValueError(
            f"time(s) {np.unique(t_arr[bad])[:4]} fall outside the track's time range "
            f"({lo}, {hi}) for mmsi {track.mmsi}; interpolate does not extrapolate"
        )
    xy = np.stack([np.interp(t_arr, track.times, track.xy[:, i]) for i in (0, 1)], axis=-1)
    return xy[0] if np.ndim(t) == 0 else xy


def _walk(start_xy, heading_rad, speed_ms, times, rng):
    """Integrate one vessel at INTEGRATE_S, then sample it at `times`.

    A transit with a slow random walk on heading and no boundary handling at all: the
    vessel goes where it is pointed. Nothing turns it back at the edge of the start box,
    because a ship that loops to stay inside a 40 km box is not a merchant ship.
    """
    turn_std = np.deg2rad(TURN_STD_DEG_PER_STEP)

    fine = np.arange(times[0], times[-1] + INTEGRATE_S, INTEGRATE_S)
    xy = np.empty((fine.size, 2))
    xy[0] = start_xy
    heading = float(heading_rad)
    for k in range(1, fine.size):
        heading += rng.normal(0.0, turn_std)
        step = speed_ms * (fine[k] - fine[k - 1])
        xy[k] = xy[k - 1] + step * np.array([np.cos(heading), np.sin(heading)])

    return np.stack([np.interp(times, fine, xy[:, i]) for i in (0, 1)], axis=-1)


def _shadow(lead_xy, rng):
    """A track running parallel to `lead_xy`, a few km abeam -- the confuser.

    Offset along the lead's local normal, so the two are near-parallel *everywhere* rather
    than merely starting on the same heading. An independent walk seeded with the lead's
    heading diverges as both wander: measured 109 degrees apart by the end of a 12 h window,
    which is not a confuser at all.
    """
    tangent = np.gradient(lead_xy, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    abeam = rng.uniform(*CONFUSER_ABEAM_M) * rng.choice([-1.0, 1.0])
    return lead_xy + abeam * normal


def synthetic_tracks(n: int, cfg: Config, seed: int) -> list[Track]:
    """`n` plausible vessel tracks sharing one time base over the scenario window.

    Speeds 6-14 kn, gentle course changes, tracks that cross. Track index 1 is always a
    **confuser**: it shadows track 0 a few km abeam on a near-parallel heading over the
    same times, so 6.7 has to discriminate on drift rather than on proximity alone.
    """
    if n < 2:
        raise ValueError(f"need at least 2 tracks to place a confuser, got {n}")
    rng = np.random.default_rng([cfg.seed, seed])
    box = np.asarray(SCENARIO_EXTENT_M)
    times = np.arange(T_START, T_END + 1.0, SAMPLE_S)

    tracks: list[Track] = []
    for i in range(n):
        if i == 1:
            # The confuser: same water, same hours, a few km abeam of track 0.
            xy = _shadow(tracks[0].xy, rng)
        else:
            speed = rng.uniform(*SPEED_RANGE_KN) * KNOT_MS
            # Place the vessel in the box at the MIDDLE of the window and back the start
            # out along its heading, so the transit is centred on the gyre rather than
            # running away from it. Releases sit near mid-window, and that is the part of
            # the track that has to be in a valid current field.
            midpoint = rng.uniform(SCENARIO_MARGIN_M, box - SCENARIO_MARGIN_M)
            heading = rng.uniform(0.0, 2 * np.pi)
            back = speed * (times[-1] - times[0]) / 2.0
            start = midpoint - back * np.array([np.cos(heading), np.sin(heading)])
            xy = _walk(start, heading, speed, times, rng)
        tracks.append(Track(mmsi=f"{419_000_000 + 1000 * seed + i:09d}", times=times, xy=xy))
    return tracks
