"""Run configuration. Every stochastic function takes a seed or rng derived from this."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable, hashable run parameters. Units are stated per field."""

    seed: int = 42
    n_members: int = 30
    n_particles: int = 400
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
# 3 h, not 12: over 3 h a 0.25 m/s current displaces oil ~2.7 km against a slick of ~3 km,
# so drift is a real signal, while a vessel transits the gyre once rather than circling it.
SCENARIO_HOURS = 3.0                       # length of the AIS window
T_END = T_START + SCENARIO_HOURS * 3600.0
