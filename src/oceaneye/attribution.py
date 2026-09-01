"""Which vessel most plausibly released the observed slick -- and is it any of them?

Output is a ranked probability for INVESTIGATION. Nothing here asserts that a vessel caused
a spill, and no user-facing surface may present it that way.

    P(v, tau | S_obs)  proportional to  L(v, tau) * P(tau | v) * P(v)
    L(v, tau)          = mean over ensemble members of Sim(F(v, tau, e), S_obs)
    p_unknown          = pi_0 * L_0 / (pi_0 * L_0 + sum over (v, tau) of the numerators)

H0 -- "none of these vessels" -- is a hypothesis with its own likelihood floor `l0_unknown`
and prior `prior_unknown`, normalised alongside the vessels. Vessel probabilities therefore
do NOT sum to 1; if they ever do, that is a bug. A scene whose polluter is absent should
concentrate mass on H0 rather than spreading it over whoever happened to be nearest.

Signatures take `grid` on top of what CLAUDE.md 6.7 lists: every mask must be rasterised
onto the observation's own Grid (see 6.3), so it cannot be implicit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ais import Track
from .config import Config, ensemble_members
from .drift import advect, seed_line_source
from .similarity import score
from .slick import Grid, morphology, to_mask

# Coarse tau search. 30 min starts x three durations is ~30 hypotheses on a 6 h track;
# finer costs (hypotheses x members) advections per vessel and buys little, since the
# ensemble spread is kilometres and 30 min of steaming is ~3 km.
TAU_DURATIONS_S = (20 * 60.0, 40 * 60.0, 60 * 60.0)
TAU_START_STEP_S = 30 * 60.0
# P(tau|v) proportional to exp(-duration / this): shorter discharges are more likely a
# priori. Set from the release durations that occur, not fitted to a posterior.
TAU_DURATION_SCALE_S = 30 * 60.0
# Fastest a parcel can travel: measured peak of current + windage*wind over every member
# and every hour of the window is 1.142 m/s (test_gate_speed_exceeds_...). Rounded up with
# room to spare -- the gate must never drop the true vessel, and a loose gate costs only
# runtime, while a tight one silently becomes a second classifier.
GATE_SPEED_MS = 1.3


@dataclass(frozen=True)
class AttributionResult:
    """Posterior over (vessel, tau) plus H0. `table` is sorted by descending p."""

    table: pd.DataFrame          # [mmsi, tau_start, tau_end, likelihood, p]
    p_unknown: float

    @property
    def best(self) -> pd.Series | None:
        """Highest-probability (vessel, tau) row, or None if every track was gated out."""
        return None if self.table.empty else self.table.iloc[0]

    @property
    def by_vessel(self) -> pd.Series:
        """Probability per vessel, tau marginalised out. Does not sum to 1 -- H0 has mass."""
        if self.table.empty:
            return pd.Series(dtype=float)
        return self.table.groupby("mmsi")["p"].sum().sort_values(ascending=False)


def tau_grid(track: Track, t_obs: float) -> list[tuple[float, float]]:
    """Candidate release windows on `track`, all inside its span and ending by `t_obs`."""
    end = min(float(track.times[-1]), float(t_obs))
    out = []
    for duration in TAU_DURATIONS_S:
        starts = np.arange(float(track.times[0]), end - duration + 1.0, TAU_START_STEP_S)
        out += [(float(s), float(s + duration)) for s in starts]
    return out


def gate(obs_mask: np.ndarray, tracks: list[Track], t_obs: float, grid: Grid,
         cfg: Config) -> list[Track]:
    """Tracks that could physically have reached the slick. Cheap, before any drift run.

    A parcel released at time t cannot be more than GATE_SPEED_MS * (t_obs - t) from where
    it was released, so a track that never comes within that of the observed slick is out.
    Purely a cost saver: it must not decide anything a likelihood would decide differently.
    """
    if not obs_mask.any():
        raise ValueError("cannot attribute an empty observed mask")
    m = morphology(obs_mask, grid)
    centroid = m["centroid_xy"]
    radius = float(np.sqrt(m["area_m2"] / np.pi))

    keep = []
    for track in tracks:
        sel = track.times <= t_obs
        if not sel.any():
            continue
        reach = GATE_SPEED_MS * (t_obs - track.times[sel]) + radius
        d = np.hypot(*(track.xy[sel] - centroid).T)
        if np.any(d <= reach):
            keep.append(track)
    return keep


def likelihood(track: Track, tau: tuple[float, float], obs_mask: np.ndarray,
               t_obs: float, grid: Grid, cfg: Config) -> float:
    """Mean similarity over the ensemble of the slick this (vessel, tau) would have made.

    Particles are seeded once -- the vessel's track is known, only the ocean is uncertain --
    then drifted separately under each member of `ensemble_members(cfg)`. The member that
    generated the observation is not in that range (see config.truth_member).
    """
    rng = np.random.default_rng([cfg.seed, int(track.mmsi), int(tau[0]), int(tau[1])])
    p0 = seed_line_source(track, tau, cfg.n_particles, rng)
    sims = [
        score(to_mask(advect(p0, tau[0], t_obs, m, cfg), grid), obs_mask, grid, cfg)
        for m in ensemble_members(cfg)
    ]
    return float(np.mean(sims))


def predicted_footprint(track: Track, tau: tuple[float, float], t_obs: float,
                        grid: Grid, cfg: Config) -> np.ndarray:
    """Fraction of ensemble members that put oil in each cell, for one (vessel, tau).

    The honest thing to draw next to an observation: not one member's mask pretending to
    be a forecast, but how much of the ensemble agrees. Returns float (H, W) in [0, 1].
    """
    rng = np.random.default_rng([cfg.seed, int(track.mmsi), int(tau[0]), int(tau[1])])
    p0 = seed_line_source(track, tau, cfg.n_particles, rng)
    members = list(ensemble_members(cfg))
    total = np.zeros(grid.shape, dtype=float)
    for m in members:
        total += to_mask(advect(p0, tau[0], t_obs, m, cfg), grid)
    return total / len(members)


def posterior(obs_mask: np.ndarray, tracks: list[Track], t_obs: float, grid: Grid,
              cfg: Config) -> AttributionResult:
    """Rank (vessel, tau) hypotheses against each other and against H0.

    With every track gated out, or no track carrying a usable tau, `p_unknown` is exactly
    1.0 -- the honest answer to "which of these vessels", when the answer is none of them.
    """
    candidates = [(t, tau_grid(t, t_obs)) for t in gate(obs_mask, tracks, t_obs, grid, cfg)]
    candidates = [(t, taus) for t, taus in candidates if taus]

    rows = []
    p_vessel = (1.0 - cfg.prior_unknown) / len(candidates) if candidates else 0.0
    for track, taus in candidates:
        # P(tau|v), normalised per vessel so no vessel gains prior mass by having more
        # hypotheses on its track than another.
        durations = np.array([t1 - t0 for t0, t1 in taus])
        p_tau = np.exp(-durations / TAU_DURATION_SCALE_S)
        p_tau /= p_tau.sum()
        for (t0, t1), pt in zip(taus, p_tau, strict=True):
            lik = likelihood(track, (t0, t1), obs_mask, t_obs, grid, cfg)
            rows.append({"mmsi": track.mmsi, "tau_start": t0, "tau_end": t1,
                         "likelihood": lik, "_num": lik * pt * p_vessel})

    num_unknown = cfg.prior_unknown * cfg.l0_unknown
    table = pd.DataFrame(rows, columns=["mmsi", "tau_start", "tau_end", "likelihood", "_num"])
    z = float(table["_num"].sum()) + num_unknown
    table["p"] = table["_num"] / z
    table = table.drop(columns="_num").sort_values("p", ascending=False, ignore_index=True)
    return AttributionResult(table=table, p_unknown=num_unknown / z)
