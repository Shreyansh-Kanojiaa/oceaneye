"""explain.why_this_answer: three sentences, every number traceable to the table."""

import pandas as pd
import pytest

from oceaneye.attribution import AttributionResult
from oceaneye.explain import why_this_answer

T0 = 0.0


def _result(rows, p_unknown):
    table = pd.DataFrame(rows, columns=["mmsi", "tau_start", "tau_end", "likelihood", "p"])
    table = table.sort_values("p", ascending=False, ignore_index=True)
    return AttributionResult(table=table, p_unknown=p_unknown)


def test_three_sentences_top_beats_runner_up_and_overlaps_truth():
    r = _result([
        ("A", 100.0, 200.0, 0.80, 0.50),
        ("B", 100.0, 200.0, 0.40, 0.20),
    ], p_unknown=0.30)
    lines = why_this_answer(r, true_tau=(150.0, 250.0), t0=T0)
    assert len(lines) == 3
    assert "A" in lines[0] and "B" in lines[0]
    assert "0.800" in lines[0] and "0.400" in lines[0]
    assert "overlaps" in lines[1] and "does not overlap" not in lines[1]
    assert "0.300" in lines[2] and "0.500" in lines[2]
    assert "not committing" not in lines[2]   # 0.30 < 0.50, top vessel should win here


def test_p_unknown_exceeding_top_vessel_declines_to_commit():
    r = _result([("A", 100.0, 200.0, 0.30, 0.20)], p_unknown=0.60)
    lines = why_this_answer(r, true_tau=None, t0=T0)
    assert "not committing" in lines[2]
    assert "0.600" in lines[2] and "0.200" in lines[2]


def test_no_overlap_is_reported_as_such():
    r = _result([("A", 100.0, 200.0, 0.5, 0.5)], p_unknown=0.1)
    lines = why_this_answer(r, true_tau=(500.0, 600.0), t0=T0)
    assert "does not overlap" in lines[1]


def test_hidden_polluter_skips_the_planted_tau_comparison():
    r = _result([("A", 100.0, 200.0, 0.5, 0.5)], p_unknown=0.1)
    lines = why_this_answer(r, true_tau=None, t0=T0)
    assert "hidden" in lines[1]
    assert "overlap" not in lines[1]


def test_single_candidate_has_no_runner_up():
    r = _result([("A", 100.0, 200.0, 0.5, 0.5)], p_unknown=0.5)
    lines = why_this_answer(r, true_tau=None, t0=T0)
    assert "no runner-up" in lines[0]


def test_no_gated_candidates_at_all():
    r = _result([], p_unknown=1.0)
    lines = why_this_answer(r, true_tau=None, t0=T0)
    assert len(lines) == 3
    assert "1.000" in lines[2]
    assert "not committing" in lines[2]


def test_always_returns_exactly_three_sentences():
    r = _result([
        ("A", 0.0, 60.0, 0.9, 0.6),
        ("B", 0.0, 60.0, 0.3, 0.1),
        ("C", 0.0, 60.0, 0.2, 0.05),
    ], p_unknown=0.25)
    lines = why_this_answer(r, true_tau=(10.0, 50.0), t0=T0)
    assert len(lines) == 3
    assert all(isinstance(s, str) and s for s in lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
