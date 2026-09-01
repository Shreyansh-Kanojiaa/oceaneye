"""OCEAN-EYE — attribute a synthetic slick to a candidate vessel, with an H0 hypothesis.

    streamlit run app/streamlit_app.py

Nothing here is a real detection. The banner stays on screen for the life of the session.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oceaneye.attribution import posterior, predicted_footprint  # noqa: E402
from oceaneye.calibrate import ece, reliability_diagram  # noqa: E402
from oceaneye.config import T_START, Config  # noqa: E402
from oceaneye.plotting import (  # noqa: E402
    BANNER,
    DISCLAIMER,
    plot_posterior,
    plot_scene,
    plot_tau,
)
from oceaneye.truth import make_scenario  # noqa: E402

TRIALS = ROOT / "outputs" / "trials.csv"


@st.cache_data(show_spinner="Drifting particles through the ensemble…")
def attribute(seed: int, hidden: bool):
    cfg = Config()
    sc = make_scenario(seed, hide_polluter=hidden, cfg=cfg)
    r = posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)
    top = r.by_vessel.index[0] if len(r.by_vessel) else None
    footprint = None
    if top is not None:
        best = r.table[r.table["mmsi"] == top].iloc[0]
        track = next(t for t in sc.tracks if t.mmsi == top)
        footprint = predicted_footprint(
            track, (best["tau_start"], best["tau_end"]), sc.t_obs, sc.grid, cfg)
    return sc, r, top, footprint


st.set_page_config(page_title="OCEAN-EYE", layout="wide")
st.markdown(
    f"<div style='background:#b00020;color:white;padding:9px 14px;border-radius:6px;"
    f"font-weight:700;letter-spacing:.4px'>{BANNER}</div>",
    unsafe_allow_html=True,
)
st.caption(DISCLAIMER)

with st.sidebar:
    st.header("Scenario")
    seed = st.number_input("seed", min_value=1, max_value=999, value=7, step=1,
                           help="Seed 7 plants the release on the confuser — the hard case.")
    hidden = st.toggle("hide the polluter", value=False,
                       help="Drops the responsible vessel from the AIS. There is then no "
                            "correct vessel, and H₀ should take the mass.")
    st.divider()
    cfg = Config()
    st.caption(f"{cfg.n_members} ensemble members · {cfg.n_particles} particles · "
               f"π₀ {cfg.prior_unknown} · L₀ {cfg.l0_unknown}")

sc, r, top, footprint = attribute(int(seed), bool(hidden))
mins = lambda t: (np.asarray(t, dtype=float) - T_START) / 60.0  # noqa: E731

c1, c2, c3 = st.columns(3)
c1.metric("top-ranked vessel", top or "—",
          f"p = {r.by_vessel.iloc[0]:.3f}" if top else None)
c2.metric("unknown source H₀", f"{r.p_unknown:.3f}",
          "ranks first" if (not len(r.by_vessel) or r.p_unknown > r.by_vessel.iloc[0])
          else "ranks below top vessel")
c3.metric("planted vessel (ground truth)", sc.true_mmsi or "not in the AIS",
          "hidden — H₀ is the right answer" if hidden
          else ("recovered" if top == sc.true_mmsi else "MISSED"))

left, right = st.columns([1.15, 1])
with left:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    plot_scene(ax, sc, footprint=footprint, highlight_mmsi=top)
    fig.tight_layout()
    st.pyplot(fig)
with right:
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    plot_posterior(ax, r, true_mmsi=sc.true_mmsi)
    fig.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    plot_tau(ax, r, None if hidden else sc.true_tau, T_START, top)
    fig.tight_layout()
    st.pyplot(fig)

st.subheader("Posterior over (vessel, release window)")
st.caption(f"Vessels sum to {r.table['p'].sum():.4f}; H₀ holds {r.p_unknown:.4f}. "
           "They sum to 1 together — vessel probabilities alone never do.")
table = r.table.head(12).copy()
table["tau_start_min"] = mins(table["tau_start"]).round(1)
table["tau_end_min"] = mins(table["tau_end"]).round(1)
st.dataframe(table[["mmsi", "tau_start_min", "tau_end_min", "likelihood", "p"]],
             width="stretch", hide_index=True)

st.subheader("Calibration")
if TRIALS.exists():
    df = pd.read_csv(TRIALS)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    reliability_diagram(df, ax, quantile=True)
    fig.tight_layout()
    st.columns([1, 1])[0].pyplot(fig)
    st.caption(f"{len(df)} synthetic trials · ECE {ece(df, quantile=True):.4f} "
               f"(quantile bins) · {ece(df):.4f} (equal-width) · "
               f"{int(df.hidden.sum())} of them hide the polluter.")
else:
    st.info("Run `python scripts/run_calibration.py --trials 200` to produce the "
            "reliability diagram.")
