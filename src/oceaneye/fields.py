"""Analytic ocean: a double-gyre current plus a slowly rotating near-uniform wind.

Positions are projected metres, times are unix seconds UTC. Both fields are pure
functions of (position, time, ensemble member) -- no global random state.

The ensemble is the uncertainty model. Members perturb the *forecast* -- current speed and
direction, wind speed and direction, windage, diffusivity -- by the amount a few-hour
forecast is actually wrong by. They are not independent draws of a different ocean. Member 0
is not special; drift and attribution average over members, and `truth.py` holds one out.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from .config import Config

# Calibration knobs. Kept module-level rather than in Config because only this module
# reads them; promote to Config if drift or the UI ever needs to vary them.
# 60 km, not 20: at 20 km a 3 h transit ran 1.9 gyre cells out, where the current field's
# linear dfdx term leaves its design range, which would force attribution to restrict its
# tau grid on a geometric criterion. At 60 km a 6 h transit stays inside 1.3 cells.
DOMAIN_M = 60_000.0          # length scale of one gyre cell, metres
# 0.35, not 0.18: shear scales as amplitude / DOMAIN_M, so tripling the domain cut the
# deformation of a 3 km slick to 92 m over 3 h -- below the 509 m diffusive spread, leaving
# the flow to translate the slick without deforming it and collapsing the demo's contrast.
# Raising the amplitude restores 6.6 deg rotation of a 3 km pair while mean and peak speed
# (0.173 / 0.373 m/s) stay inside the 0.1-0.4 m/s design range.
GYRE_AMPLITUDE = 0.35        # peak gyre speed, m/s
GYRE_PERIOD_S = 12 * 3600.0  # gyre oscillation period, seconds
MEAN_FLOW = (0.16, 0.05)     # background current, m/s -- keeps speed off zero at stagnation
GYRE_EPS = 0.25              # cross-cell oscillation amplitude, nondimensional

WIND_PERIOD_S = 36 * 3600.0  # wind direction rotation period, seconds
WIND_SPEED_MS = 7.5          # base wind speed, m/s
WIND_DIR_RAD = 2.0           # base wind direction at T_START, radians
WIND_SHEAR = 0.4             # spatial variation of wind speed across the domain, m/s

# Perturbation magnitudes, set from forecast-error scale BEFORE seeing any posterior and
# frozen from 2 Sept. A few-hour surface-current or wind forecast is wrong by roughly a
# third in speed and 15 deg in direction; windage is uncertain to +/-30% of 0.03; eddy
# diffusivity to within a factor of ~1.5. Do not adjust these to move a result.
SPEED_ERR = 0.30             # fractional, on current speed, wind speed and windage
DIR_ERR_DEG = 15.0           # on current and wind direction
DIFFUSIVITY_ERR = 0.50       # fractional


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
    err = lambda f: rng.uniform(1.0 - f, 1.0 + f)  # noqa: E731
    return MemberParams(
        rotation_rad=np.deg2rad(rng.uniform(-DIR_ERR_DEG, DIR_ERR_DEG)),
        scale=err(SPEED_ERR),
        windage=cfg.windage * err(SPEED_ERR),
        diffusivity=cfg.diffusivity * err(DIFFUSIVITY_ERR),
        wind_speed=WIND_SPEED_MS * err(SPEED_ERR),
        # Perturbed about a SHARED base direction, not drawn over the full circle. A
        # uniform(0, 2pi) phase made every member blow a 0.2 m/s windage drift a different
        # way, scattering member centroids over 12 km against a 3 km displacement and a
        # 4.6 km slick -- so no member ever overlapped the observation and the mean IoU of
        # a *correct* prediction was 0.094. That is not forecast uncertainty, it is noise.
        wind_phase_rad=WIND_DIR_RAD + np.deg2rad(rng.uniform(-DIR_ERR_DEG, DIR_ERR_DEG)),
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
