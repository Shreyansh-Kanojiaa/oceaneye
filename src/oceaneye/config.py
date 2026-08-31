"""Run configuration. Every stochastic function takes a seed or rng derived from this."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable, hashable run parameters. Units are stated per field."""

    seed: int = 42
    n_members: int = 30
    # 1600, not 400: at 400 the observed mask is 37.7% filled and breaks into 27.8
    # components, only 81.2% of cells in the largest -- confetti, not a slick, and the
    # orientation and extent terms read shape off that. 1600 gives 64.6% fill and 96.2% in
    # the largest component for 26% more runtime. Raised rather than closing the mask
    # morphologically, which would distort the very shape those terms measure.
    n_particles: int = 1600
    dt_minutes: int = 10
    windage: float = 0.03            # fraction of wind speed added to drift
    diffusivity: float = 12.0        # m^2/s
    grid_res_m: float = 200.0
    sim_weights: tuple = (0.45, 0.25, 0.15, 0.15)   # iou, centroid, orientation, extent
    l0_unknown: float = 0.15         # H0 likelihood floor
    prior_unknown: float = 0.20      # pi_0


# --- Shared scenario frame -------------------------------------------------------------
# Every module that places something in space or time uses these. Tracks, the drift domain,
# the Grid from 6.3 and the observation time all live in this one frame; if they drift
# apart nothing downstream lines up.
#
# The box is one full double-gyre cell of fields.py (DOMAIN_M = 20 km spans the cell's unit
# height, and the classic double gyre is defined on [0, 2] x [0, 1]), so tracks sit inside
# the structured part of the current field rather than off in the uniform mean flow.
# Vessels START inside this box and transit; they are not confined to it. A merchant ship
# covers 33-78 km in the window, so nothing plausible stays inside a 40 km box -- confining
# them needs either repeated loops or hard turns, and neither is merchant behaviour. The
# box bounds where releases happen, which is what has to sit inside the gyre; Grid.covering
# sizes the raster from the tracks and slick that a scenario actually uses.
SCENARIO_EXTENT_M = (40_000.0, 20_000.0)   # (x, y) start box, metres, origin at (0, 0)
SCENARIO_MARGIN_M = 2_000.0                # start points stay this far inside the box
T_START = 1_788_220_800                    # unix seconds UTC = 2026-09-01T00:00:00Z
# 6 h, not 3: measured deformation of a 3 km pair over the window is 6.6 deg of rotation
# and 311 m of differential displacement at 3 h it is 3.0 deg and 177 m, which is lost
# under the 509 m diffusive spread. 6 h keeps track excursion at 1.3 gyre cells.
SCENARIO_HOURS = 6.0                       # length of the AIS window
T_END = T_START + SCENARIO_HOURS * 3600.0


# --- Ensemble bookkeeping --------------------------------------------------------------
# The held-out member is structural, not a convention. `truth.py` drifts the observed slick
# under `truth_member(cfg)`, which is deliberately OUTSIDE `ensemble_members(cfg)` -- the
# range `attribution.py` averages over. Nobody can accidentally average over the member
# that generated the observation without changing the range itself, and a test asserts the
# two are disjoint. If the observation came from a member attribution also uses, the
# simulate-and-recover result would be optimistically biased.


def ensemble_members(cfg: "Config") -> range:
    """Members attribution averages over. `truth_member` is never one of these."""
    return range(cfg.n_members)


def truth_member(cfg: "Config") -> int:
    """The held-out member that generates observations. Outside `ensemble_members`."""
    return cfg.n_members
