"""Analytic ocean: a double-gyre current plus a slowly rotating near-uniform wind.

Positions are projected metres, times are unix seconds UTC. Both fields are pure
functions of (position, time, ensemble member) -- no global random state.

The ensemble is the uncertainty model: each member perturbs the current (rotation and
scale), the windage coefficient, and the diffusivity. Member 0 is not special; drift and
attribution average over members, and `truth.py` will hold one out.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from .config import Config

# Calibration knobs. Kept module-level rather than in Config because only this module
# reads them; promote to Config if drift or the UI ever needs to vary them.
DOMAIN_M = 20_000.0          # length scale of one gyre cell, metres
GYRE_AMPLITUDE = 0.18        # peak gyre speed, m/s
GYRE_PERIOD_S = 12 * 3600.0  # gyre oscillation period, seconds
MEAN_FLOW = (0.16, 0.05)     # background current, m/s -- keeps speed off zero at stagnation
GYRE_EPS = 0.25              # cross-cell oscillation amplitude, nondimensional

WIND_PERIOD_S = 36 * 3600.0  # wind direction rotation period, seconds
WIND_SPEED_RANGE = (6.0, 9.0)  # per-member base wind speed, m/s
WIND_SHEAR = 0.4             # spatial variation of wind speed across the domain, m/s


@dataclass(frozen=True)
class MemberParams:
    """Per-member perturbations of the physics. Deterministic given (cfg.seed, member)."""

    rotation_rad: float
    scale: float
    windage: float
    diffusivity: float       # m^2/s
    wind_speed: float        # m/s
    wind_phase_rad: float


@cache
def member_params(member: int, cfg: Config) -> MemberParams:
    """Perturbations for one ensemble member. Same (seed, member) -> same numbers."""
    rng = np.random.default_rng([cfg.seed, member])
    lo, hi = WIND_SPEED_RANGE
    return MemberParams(
        rotation_rad=np.deg2rad(rng.uniform(-12.0, 12.0)),
        scale=rng.uniform(0.85, 1.15),
        windage=cfg.windage * rng.uniform(0.7, 1.3),
        diffusivity=cfg.diffusivity * rng.uniform(0.5, 2.0),
        wind_speed=rng.uniform(lo, hi),
        wind_phase_rad=rng.uniform(0.0, 2 * np.pi),
    )


def _rotate(uv: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * uv[..., 0] - s * uv[..., 1], s * uv[..., 0] + c * uv[..., 1]], axis=-1)


def current(xy: np.ndarray, t: float, member: int, cfg: Config) -> np.ndarray:
    """Surface current at positions `xy` (..., 2) in metres and time `t` in seconds.

    Returns (..., 2) in m/s. Double-gyre streamfunction with a slow time dependence,
    plus a steady mean flow, rotated and scaled per ensemble member.
    """
    xy = np.asarray(xy, dtype=float)
    p = member_params(member, cfg)

    # Nondimensionalise onto the classic [0, 2] x [0, 1] double-gyre cell.
    x = xy[..., 0] / DOMAIN_M
    y = xy[..., 1] / DOMAIN_M

    eps = GYRE_EPS * np.sin(2 * np.pi * t / GYRE_PERIOD_S)
    a = eps
    b = 1.0 - 2.0 * eps
    f = a * x**2 + b * x
    dfdx = 2.0 * a * x + b

    u = -np.pi * GYRE_AMPLITUDE * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * GYRE_AMPLITUDE * np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx

    uv = np.stack([u, v], axis=-1) / np.pi  # peak |gyre| ~= GYRE_AMPLITUDE
    uv = uv + np.asarray(MEAN_FLOW)
    return p.scale * _rotate(uv, p.rotation_rad)


def wind(xy: np.ndarray, t: float, member: int, cfg: Config) -> np.ndarray:
    """10 m wind at positions `xy` (..., 2) in metres and time `t` in seconds.

    Returns (..., 2) in m/s. Near-uniform in space, direction rotating slowly in time.
    """
    xy = np.asarray(xy, dtype=float)
    p = member_params(member, cfg)

    theta = p.wind_phase_rad + 2 * np.pi * t / WIND_PERIOD_S
    # Gentle spatial shear so the field is near-uniform rather than exactly uniform.
    speed = p.wind_speed + WIND_SHEAR * np.sin(np.pi * xy[..., 1] / DOMAIN_M)
    return np.stack([speed * np.cos(theta), speed * np.sin(theta)], axis=-1)
