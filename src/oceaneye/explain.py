"""Plain-text explanation of the top-ranked attribution answer.

Three sentences, generated only from an already-computed AttributionResult (and the
planted tau, when there is one worth comparing against) -- no new drift runs, no new
similarity scores, no new posterior. Every number quoted here already lives in
result.table or result.p_unknown; this module only selects and formats it.
"""

from .attribution import AttributionResult


def _mins(t: float, t0: float) -> float:
    return (t - t0) / 60.0


def why_this_answer(result: AttributionResult, true_tau: tuple[float, float] | None,
                    t0: float) -> list[str]:
    """Three sentences: top-vs-runner-up likelihood, the MAP tau window, and p_unknown.

    `true_tau` is the planted release window, or None when it should not be compared
    against -- the polluter is hidden, so no candidate could ever recover it. That is
    the same condition under which the app already passes `None` to `plotting.plot_tau`.
    """
    vessels = result.by_vessel
    if vessels.empty:
        return [
            "No candidate vessel was gated within reach of the observed slick, so "
            "there is no top vessel or runner-up to compare.",
            "There is no MAP release window to report, since no candidate produced one.",
            f"H0 (unknown source) holds the entire posterior, p_unknown = "
            f"{result.p_unknown:.3f}: no candidate explains the slick decisively, and "
            "the system is not committing to a vessel.",
        ]

    top_mmsi = str(vessels.index[0])
    top_p = float(vessels.iloc[0])
    # table is sorted by p descending, so the first row for a given mmsi is already that
    # vessel's own highest-p (MAP) row -- no extra sort or computation needed.
    top_row = result.table[result.table["mmsi"] == top_mmsi].iloc[0]
    top_lik = float(top_row["likelihood"])
    top_tau = (float(top_row["tau_start"]), float(top_row["tau_end"]))

    if len(vessels) > 1:
        runner_mmsi = str(vessels.index[1])
        runner_row = result.table[result.table["mmsi"] == runner_mmsi].iloc[0]
        runner_lik = float(runner_row["likelihood"])
        s1 = (f"{top_mmsi} has the highest likelihood at {top_lik:.3f}, ahead of "
              f"runner-up {runner_mmsi} at {runner_lik:.3f} "
              f"(+{top_lik - runner_lik:.3f}).")
    else:
        s1 = (f"{top_mmsi} is the only gated candidate, with likelihood {top_lik:.3f}; "
              "there is no runner-up to compare against.")

    start_min, end_min = _mins(top_tau[0], t0), _mins(top_tau[1], t0)
    if true_tau is not None:
        overlap = top_tau[0] <= true_tau[1] and true_tau[0] <= top_tau[1]
        planted_start, planted_end = _mins(true_tau[0], t0), _mins(true_tau[1], t0)
        s2 = (f"Its MAP release window is {start_min:.1f}-{end_min:.1f} min, which "
              f"{'overlaps' if overlap else 'does not overlap'} the planted window "
              f"{planted_start:.1f}-{planted_end:.1f} min.")
    else:
        s2 = (f"Its MAP release window is {start_min:.1f}-{end_min:.1f} min; the "
              "polluter is hidden in this run, so there is no planted window to "
              "compare against.")

    if result.p_unknown > top_p:
        s3 = (f"p_unknown = {result.p_unknown:.3f} exceeds {top_mmsi}'s posterior "
              f"{top_p:.3f}: no candidate explains the slick decisively, and the "
              "system is not committing to a vessel.")
    else:
        s3 = (f"p_unknown = {result.p_unknown:.3f} is below {top_mmsi}'s posterior "
              f"{top_p:.3f}, so the posterior favours naming {top_mmsi} over the "
              "unknown-source hypothesis.")

    return [s1, s2, s3]
