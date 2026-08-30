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
