"""Forward drift of a discharge: line-source seeding, then advection + diffusion.

A discharging vessel is a *moving line source*, not a point release. Particles are seeded
uniformly in time across the release window tau, each at the vessel's position at its own
seed time, and each stays put until that time arrives. Oil released early has therefore
drifted for longer than oil released late -- collapsing this to a point release at the
midpoint of tau throws away the signal attribution depends on.

Positions are projected metres, times unix seconds UTC. Nothing here touches global random
state: the noise comes from an explicit rng, or from one derived from (cfg.seed, member,
t0, t1) so that repeated calls for the same candidate reproduce exactly.
"""

from dataclasses import dataclass, replace

import numpy as np

from .ais import Track, interpolate
from .config import Config
from .fields import current, member_params, wind


@dataclass(frozen=True, eq=False)
class Particles:
    """A parcel cloud. `xy` is (n, 2) metres, `t_seed` is (n,) unix seconds UTC."""

    xy: np.ndarray
    t_seed: np.ndarray

    def __len__(self) -> int:
        return self.xy.shape[0]


def seed_line_source(track: Track, tau: tuple[float, float], n: int, rng) -> Particles:
    """Seed `n` particles along `track` over the release window `tau` = (start, end).

    Seed times are stratified uniform across tau -- one draw per equal sub-interval, which
    covers the window evenly for the few hundred particles we can afford. Each particle
    starts at the vessel's interpolated position at its own seed time.
    """
    t0, t1 = float(tau[0]), float(tau[1])
    if t1 < t0:
        raise ValueError(f"tau must be (start, end) with end >= start, got {tau}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    # Checked on tau itself, not left to ais.interpolate: seed times fall strictly inside
    # tau, so a tau endpoint hanging off the end of the track would never be interpolated
    # and would pass silently. A tau outside the track's span is a bug in 6.7's tau grid.
    if t0 < track.times[0] or t1 > track.times[-1]:
        raise ValueError(
            f"tau ({t0}, {t1}) falls outside the track's time range "
            f"({track.times[0]}, {track.times[-1]})"
        )

    edges = np.linspace(t0, t1, n + 1)
    t_seed = edges[:-1] + rng.uniform(0.0, 1.0, n) * (edges[1] - edges[0])

    # ais.interpolate raises rather than clamping if tau runs off the end of the track.
    return Particles(xy=interpolate(track, t_seed), t_seed=t_seed)


def advect(
    p: Particles, t0: float, t1: float, member: int, cfg: Config, rng=None
) -> Particles:
    """Drift `p` from `t0` to `t1` under ensemble member `member`. Returns a new Particles.

    Per step: x <- x + (u_current + alpha*u_wind)*dt + sqrt(2*K*dt)*N(0,1), with the
    windage alpha and diffusivity K taken from this member's perturbation. A particle is
    frozen until its own seed time, so one call can drift a whole line source.

    `rng` defaults to a generator seeded from (cfg.seed, member, t0, t1): reproducible, and
    no global state. Pass one explicitly to draw an independent realisation.
    """
    t0, t1 = float(t0), float(t1)
    if t1 < t0:
        raise ValueError(f"advection runs forward only, got t0={t0} > t1={t1}")
    if rng is None:
        rng = np.random.default_rng([cfg.seed, member, int(t0), int(t1)])

    mp = member_params(member, cfg)
    dt_full = cfg.dt_minutes * 60.0
    xy = p.xy.copy()
    t = t0
    while t < t1:
        dt = min(dt_full, t1 - t)
        u = current(xy, t, member, cfg) + mp.windage * wind(xy, t, member, cfg)
        noise = rng.standard_normal(xy.shape) * np.sqrt(2.0 * mp.diffusivity * dt)
        # Particles not yet released hold position; no Python loop over particles.
        active = (p.t_seed <= t)[:, None]
        xy = xy + active * (u * dt + noise)
        t += dt
    return replace(p, xy=xy)
