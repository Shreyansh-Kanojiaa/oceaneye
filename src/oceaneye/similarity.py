"""How well does a predicted slick match the observed one?

    Sim = w_iou*IoU + w_cen*C + w_ori*|cos(dtheta)|*C + w_ext*(min(A)/max(A))*C
    C   = exp(-d_centroid / CENTROID_SCALE_M)

Weights come from `cfg.sim_weights` in the order (iou, centroid, orientation, extent) and
are normalised by their own sum, so identical masks score exactly 1.0 for any weights.

The proximity factor C on the orientation and extent terms SUPERSEDES the bare weighted sum
written in CLAUDE.md 6.4. That formula cannot pass its own "disjoint masks score ~0"
acceptance check -- measured 0.3046 for identical slicks 20 km apart -- and reverting to it
silently breaks 6.7's H0 behaviour. Confirmed by Shreyansh, 31 Aug; read PROXIMITY GATE
below before "fixing" it back. Both
masks must live on the same `Grid`; a score between masks on different rasters is
meaningless and raises.
"""

import numpy as np

from .config import Config
from .slick import Grid, morphology

# Calibration knob, as in fields.py: the centroid offset at which the proximity term
# falls to 1/e. Set from the scale of drift error we expect over a few hours, not tuned
# against the reliability diagram.
CENTROID_SCALE_M = 5_000.0


def score(pred: np.ndarray, obs: np.ndarray, grid: Grid, cfg: Config) -> float:
    """Similarity of predicted mask `pred` to observed mask `obs`, in [0, 1].

    1.0 exactly iff the masks are identical. 0.0 if either is empty -- a cloud that drifted
    off the grid predicts no slick, which matches nothing rather than matching perfectly.

    PROXIMITY GATE: the orientation and extent terms are multiplied by the same proximity
    factor C as the centroid term. Under the bare sum in 6.4, two identical-shaped,
    identically-oriented slicks on opposite sides of the domain still collect
    w_ori + w_ext = 0.30, because they genuinely do share shape and size. That floor sits
    above cfg.l0_unknown (0.15), so in 6.7 every gated candidate would out-score H0 by
    construction and `p_unknown` could never dominate -- breaking the hide_polluter
    acceptance check. IoU and the centroid term already vanish with distance; gating the
    two shape terms makes the whole score vanish with them.
    """
    pred = np.asarray(pred, dtype=bool)
    obs = np.asarray(obs, dtype=bool)
    if pred.shape != obs.shape:
        raise ValueError(
            f"masks must share one grid, got shapes {pred.shape} and {obs.shape}"
        )
    if pred.shape != tuple(grid.shape):
        raise ValueError(f"masks {pred.shape} do not match grid {tuple(grid.shape)}")
    if not pred.any() or not obs.any():
        return 0.0

    union = int((pred | obs).sum())
    iou = float((pred & obs).sum()) / union

    mp, mo = morphology(pred, grid), morphology(obs, grid)
    d = float(np.hypot(*(mp["centroid_xy"] - mo["centroid_xy"])))
    proximity = float(np.exp(-d / CENTROID_SCALE_M))

    # |cos| because orientation is only defined modulo pi (see slick.morphology): an axis
    # at 10 degrees and one at 190 are the same orientation and must score identically.
    orientation = abs(float(np.cos(mp["orientation_rad"] - mo["orientation_rad"])))
    areas = (mp["area_m2"], mo["area_m2"])
    extent = min(areas) / max(areas)

    terms = np.array([iou, proximity, orientation * proximity, extent * proximity])
    w = np.asarray(cfg.sim_weights, dtype=float)
    if w.shape != (4,):
        raise ValueError(f"cfg.sim_weights must have 4 entries, got {w.shape}")
    # Dividing by the weight sum makes identical masks (terms all exactly 1.0) score
    # exactly 1.0 by construction, for any weights -- not just ones that happen to sum to 1.
    return float(w @ terms / w.sum())
