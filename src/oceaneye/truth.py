"""Simulate-and-recover scenarios: plant a release, drift it, hand back only what a
satellite would have seen.

No public dataset pairs a SAR slick with a confirmed MMSI, so ground truth has to be
simulated. That constraint is what makes calibration possible at all -- we know the answer
by construction -- and it is why every user-facing surface says SYNTHETIC.

Two things keep the recovery honest:

* The observation is drifted under `config.truth_member(cfg)`, an ensemble index outside
  the `config.ensemble_members(cfg)` range that attribution averages over. Held out
  structurally, not by convention.
* The observed mask is built by exactly the path 6.7 uses for predictions -- same
  `n_particles`, same `to_mask`, same `Grid` -- so IoU is not biased by a truth that is
  better resolved than every candidate.
"""

from dataclasses import dataclass

import numpy as np

from .ais import Track, synthetic_tracks
from .config import (
    SCENARIO_HOURS,
    T_START,
    Config,
    truth_member,
)
from .drift import advect, seed_line_source
from .fields import DOMAIN_M
from .slick import Grid, to_mask

N_TRACKS = 5                      # vessels in a scene, one of which is the confuser
RELEASE_MIN_MINUTES = 20.0        # shortest discharge
RELEASE_MAX_MINUTES = 60.0        # longest discharge
RELEASE_EARLIEST_FRAC = 0.20      # tau starts no earlier than this far into the window
RELEASE_LATEST_FRAC = 0.50        # tau ends no later than this, leaving >= half for drift
GRID_PAD_M = 20_000.0             # raster margin around the observed slick
FIELD_VALID_CELLS = 2.0           # |x| beyond this many gyre cells leaves the design range


@dataclass(frozen=True)
class Scenario:
    """One synthetic scene. `true_mmsi` is None when the polluter has been hidden."""

    tracks: list[Track]
    true_mmsi: str | None
    true_tau: tuple[float, float]
    obs_mask: np.ndarray
    t_obs: float
    grid: Grid


def make_scenario(seed: int, hide_polluter: bool = False, cfg: Config | None = None
                  ) -> Scenario:
    """Plant a release on one vessel, drift it to `t_obs`, return the observed mask.

    With `hide_polluter=True` the responsible vessel is dropped from `.tracks` and
    `.true_mmsi` is None: the scene then genuinely has no correct answer, which is what
    6.7's H0 has to detect rather than guess around.
    """
    cfg = cfg or Config()
    rng = np.random.default_rng([cfg.seed, seed, 0xC0FFEE])
    tracks = synthetic_tracks(N_TRACKS, cfg, seed)
    window = SCENARIO_HOURS * 3600.0
    t_obs = T_START + window

    culprit = tracks[int(rng.integers(len(tracks)))]
    duration = rng.uniform(RELEASE_MIN_MINUTES, RELEASE_MAX_MINUTES) * 60.0
    tau_start = T_START + window * rng.uniform(
        RELEASE_EARLIEST_FRAC, RELEASE_LATEST_FRAC - duration / window)
    tau = (float(tau_start), float(tau_start + duration))

    # Same path 6.7 takes for a predicted slick, at the same n_particles, but under the
    # held-out member -- so the observation is not a draw attribution can average over.
    particles = seed_line_source(culprit, tau, cfg.n_particles, rng)
    drifted = advect(particles, tau[0], t_obs, truth_member(cfg), cfg)

    if np.abs(drifted.xy[:, 0]).max() > FIELD_VALID_CELLS * DOMAIN_M:
        raise ValueError(
            f"seed {seed}: slick drifted to {np.abs(drifted.xy[:, 0]).max() / DOMAIN_M:.1f} "
            "gyre cells in x, outside the current field's design range"
        )

    grid = Grid.covering(drifted.xy, cfg, pad_m=GRID_PAD_M)
    obs_mask = to_mask(drifted, grid)

    if hide_polluter:
        return Scenario(tracks=[t for t in tracks if t.mmsi != culprit.mmsi],
                        true_mmsi=None, true_tau=tau, obs_mask=obs_mask,
                        t_obs=t_obs, grid=grid)
    return Scenario(tracks=tracks, true_mmsi=culprit.mmsi, true_tau=tau,
                    obs_mask=obs_mask, t_obs=t_obs, grid=grid)
