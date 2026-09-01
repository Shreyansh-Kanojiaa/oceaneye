"""6.9 acceptance: the UI actually runs, and shows what CLAUDE.md 6.9 requires.

Slow by design -- it runs the real pipeline twice, which is the only way to check that the
hidden-polluter path renders. Everything cheaper is tested elsewhere.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=300).run()


def test_app_runs_without_exceptions(app):
    assert [e.value for e in app.exception] == []


def test_synthetic_banner_is_present(app):
    """Non-negotiable 3: never let a synthetic scene leave the room unlabelled."""
    assert any("SYNTHETIC SCENARIO" in str(m.value) for m in app.markdown)


def test_h0_is_shown_alongside_the_vessels(app):
    labels = [m.label for m in app.metric]
    assert "unknown source H₀" in labels
    assert any("planted" in lb for lb in labels)
    assert len(app.dataframe) == 1


def test_hidden_polluter_path_renders_and_h0_ranks_first(app):
    hid = app.toggle[0].set_value(True).run()
    assert [e.value for e in hid.exception] == []
    h0 = next(m for m in hid.metric if m.label == "unknown source H₀")
    assert h0.delta == "ranks first"
    truth = next(m for m in hid.metric if "planted" in m.label)
    assert truth.value == "not in the AIS"
