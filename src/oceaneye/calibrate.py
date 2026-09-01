"""Do the probabilities mean what they say?

A trial plants a scenario, runs the posterior, and records the one number a user would
act on -- the top-ranked probability as the UI shows it -- alongside whether that answer
was right. Repeated over many scenes, that pair is what a reliability diagram reads.

H0 is a scoreable answer here, not a residual. About 30% of trials hide the polluter, and
those count as correct when H0 ranks top. Calibrating only the vessel-naming case would
leave the behaviour we most want to defend -- declining to accuse -- unmeasured.

Nothing in this module tunes anything. Whatever ECE comes out is the result.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .attribution import posterior
from .config import Config
from .truth import make_scenario

PRESENT_FRACTION = 0.70   # the rest hide the polluter, so H0 is the correct answer


@dataclass(frozen=True)
class Trial:
    """One scored scenario. `confidence` is what the UI would display as the top bar."""

    seed: int
    hidden: bool
    top: str              # mmsi, or "H0"
    confidence: float
    correct: bool
    # Recorded, not scored. The headline pair is (confidence, correct) exactly as above.
    # These let the vessel-naming case be inspected separately without a second 40-minute
    # run -- necessary because H0 currently out-ranks the true vessel on most scenes, so
    # the headline accuracy alone cannot distinguish "names the wrong ship" from
    # "declines to name any ship".
    p_h0: float
    p_top_vessel: float
    top_vessel_correct: bool


def score_scenario(seed: int, hidden: bool, cfg: Config) -> Trial:
    """Run one scenario end to end and score its top-ranked answer."""
    sc = make_scenario(seed, hide_polluter=hidden, cfg=cfg)
    r = posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)

    # The ranking the UI shows: one bar per vessel (tau marginalised out) plus one for H0.
    ranked = pd.concat([r.by_vessel, pd.Series({"H0": r.p_unknown})]).sort_values(
        ascending=False)
    top = str(ranked.index[0])
    vessels = r.by_vessel
    return Trial(
        seed=seed, hidden=hidden, top=top, confidence=float(ranked.iloc[0]),
        correct=(top == "H0") if hidden else (top == sc.true_mmsi),
        p_h0=float(r.p_unknown),
        p_top_vessel=float(vessels.iloc[0]) if len(vessels) else 0.0,
        top_vessel_correct=bool(len(vessels) and vessels.index[0] == sc.true_mmsi),
    )


def hidden_flags(n: int, cfg: Config) -> np.ndarray:
    """Which of `n` trials hide the polluter. Drawn once from cfg.seed, so the mix is
    reproducible and roughly 1 - PRESENT_FRACTION of trials."""
    return np.random.default_rng([cfg.seed, 0xCA11B]).random(n) >= PRESENT_FRACTION


def run_trials(n: int, cfg: Config | None = None, progress: bool = False) -> pd.DataFrame:
    """`n` scored scenarios as a DataFrame [seed, hidden, top, confidence, correct].

    Which trials hide the polluter is drawn once from `cfg.seed`, so the same n gives the
    same mix. A scenario that raises -- a slick drifting outside the current field's design
    range -- is skipped and reported, rather than killing a 40-minute run at trial 137.
    """
    cfg = cfg or Config()
    hidden = hidden_flags(n, cfg)

    rows, skipped = [], []
    for i in range(n):
        try:
            rows.append(score_scenario(i + 1, bool(hidden[i]), cfg))
        except ValueError as exc:
            skipped.append((i + 1, str(exc)))
            continue
        if progress:
            t = rows[-1]
            print(f"  trial {i + 1:4d}/{n}  {'H0-case' if t.hidden else 'vessel '}  "
                  f"top {t.top:>9}  conf {t.confidence:.3f}  "
                  f"{'correct' if t.correct else 'WRONG'}", flush=True)

    if skipped:
        print(f"  skipped {len(skipped)} scenario(s): "
              + "; ".join(f"seed {s}: {m}" for s, m in skipped[:3]), flush=True)
    return pd.DataFrame([vars(r) for r in rows])


def bin_edges(confidence: np.ndarray, bins: int, quantile: bool) -> np.ndarray:
    """Equal-width edges over [0, 1], or quantile edges when confidence is clustered."""
    if not quantile:
        return np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, bins + 1)))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def ece(df: pd.DataFrame, bins: int = 10, quantile: bool = False) -> float:
    """Expected calibration error: mean |accuracy - confidence| weighted by bin count."""
    conf = df["confidence"].to_numpy(dtype=float)
    correct = df["correct"].to_numpy(dtype=float)
    edges = bin_edges(conf, bins, quantile)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges) - 2)
    return float(sum(
        (m := idx == b).mean() * abs(correct[m].mean() - conf[m].mean())
        for b in range(len(edges) - 1) if (idx == b).any()
    ))


def fit_isotonic(df: pd.DataFrame) -> IsotonicRegression:
    """Map raw confidence to observed accuracy, monotonically. Fit on trials, not tuned."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(df["confidence"].to_numpy(dtype=float), df["correct"].to_numpy(dtype=float))
    return iso


def reliability_diagram(df: pd.DataFrame, ax, bins: int = 10,
                        quantile: bool = False) -> None:
    """Accuracy against confidence, per bin, with the ideal diagonal and bin counts."""
    conf = df["confidence"].to_numpy(dtype=float)
    correct = df["correct"].to_numpy(dtype=float)
    edges = bin_edges(conf, bins, quantile)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges) - 2)

    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))

    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="perfectly calibrated")
    ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.6, ms=6, label="observed")
    for x, y, n in zip(xs, ys, ns, strict=True):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="0.35")
    lo = min(0.0, min(xs, default=0.0))
    ax.set(xlim=(lo, 1.0), ylim=(0.0, 1.0),
           xlabel="confidence (top-ranked probability)", ylabel="accuracy")
    kind = "quantile bins" if quantile else "equal-width bins"
    ax.set_title(f"Reliability, {len(df)} synthetic trials\n"
                 f"ECE {ece(df, bins, quantile):.3f} over {len(xs)} {kind} "
                 f"(counts above points)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
