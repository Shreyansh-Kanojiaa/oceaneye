"""OCEAN-EYE — attribute a synthetic slick to a candidate vessel, with an H0 hypothesis.

    streamlit run app/streamlit_app.py

Nothing here is a real detection. The synthetic-data notice stays on screen for the life
of the session, and is not something a viewer can dismiss.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oceaneye.attribution import footprint_frames, posterior  # noqa: E402
from oceaneye.calibrate import ece, reliability_diagram  # noqa: E402
from oceaneye.config import T_START, Config  # noqa: E402
from oceaneye.explain import why_this_answer  # noqa: E402
from oceaneye.plotting import (  # noqa: E402
    ALERT,
    BANNER,
    DISCLAIMER,
    INK,
    LINE,
    MARINE,
    PANEL,
    SLATE,
    plot_posterior,
    plot_scene,
    plot_tau,
)
from oceaneye.truth import make_scenario  # noqa: E402

TRIALS = ROOT / "outputs" / "trials.csv"
N_FRAMES = 12          # drift checkpoints between the release window and the observation

st.set_page_config(page_title="OCEAN-EYE — slick attribution", layout="wide")

# Streamlit's own dev chrome (hamburger menu, "Made with Streamlit" footer) is the fastest
# tell that this is a prototype rather than a fielded tool; the rest is layout, not colour.
st.markdown(f"""
<style>
/* Only the dev-chrome buttons, never the whole toolbar: the sidebar's reopen arrow
   (stExpandSidebarButton) renders inside the SAME [data-testid="stToolbar"] container
   once the sidebar is collapsed, so hiding that container wholesale -- an earlier version
   of this rule did exactly that -- silently makes the sidebar impossible to reopen. */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"] {{
    visibility: hidden; height: 0;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
/* padding-top clears Streamlit's own (position: absolute) header row -- that's where the
   sidebar reopen arrow lives, so this needs to stay tall enough for it to have its own
   clear space above .oe-appbar rather than crowding against the title underneath it. */
.block-container {{ padding-top: 3.25rem; max-width: 1180px; }}

.oe-appbar {{
    background: {PANEL}; color: {INK};
    margin: 0 -1rem 1.1rem -1rem; padding: 16px 24px 14px 24px;
    border-bottom: 3px solid {MARINE};
    display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap;
}}
.oe-appbar-name {{ font-size: 1.2rem; font-weight: 700; letter-spacing: .01em; }}
.oe-appbar-sub {{ font-size: .85rem; color: {SLATE}; margin-left: .8rem; }}
.oe-appbar-status {{
    font-size: .72rem; font-weight: 600; letter-spacing: .06em; color: {MARINE};
    border: 1px solid {MARINE}; padding: 3px 9px; white-space: nowrap;
}}

.oe-notice {{
    border: 1px solid {LINE}; border-left: 3px solid {ALERT};
    padding: 10px 14px; margin-bottom: 1.2rem; font-size: .87rem; line-height: 1.5;
}}
.oe-notice b {{ color: {ALERT}; }}
.oe-notice span {{ display: block; color: {SLATE}; margin-top: .15rem; }}

.oe-label {{
    font-size: .72rem; font-weight: 600; letter-spacing: .07em; color: {SLATE};
    text-transform: uppercase; margin: .3rem 0 .35rem 0;
}}
table.oe-readout {{ font-family: "IBM Plex Mono", monospace; font-size: .82rem;
                    width: 100%; color: {INK}; }}
table.oe-readout td {{ padding: 2px 0; }}
table.oe-readout td:last-child {{ text-align: right; color: {SLATE}; }}

/* st.metric's delta chip ignores theme.baseRadius; flatten it to match everything else. */
[data-testid="stMetricDelta"] {{
    background: transparent !important; border-radius: 0 !important; padding-left: 0 !important;
}}
</style>

<div class="oe-appbar">
  <div><span class="oe-appbar-name">OCEAN-EYE</span>
       <span class="oe-appbar-sub">Satellite oil-slick source attribution
       — SIH26143 prototype</span></div>
  <div class="oe-appbar-status">SYNTHETIC DATA MODE</div>
</div>

<div class="oe-notice">
  <b>{BANNER}</b>
  <span>{DISCLAIMER}</span>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Drifting particles through the ensemble…")
def attribute(seed: int, hidden: bool):
    cfg = Config()
    sc = make_scenario(seed, hide_polluter=hidden, cfg=cfg)
    r = posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)
    top = r.by_vessel.index[0] if len(r.by_vessel) else None
    times = frames = tau = None
    if top is not None:
        best = r.table[r.table["mmsi"] == top].iloc[0]
        track = next(t for t in sc.tracks if t.mmsi == top)
        tau = (best["tau_start"], best["tau_end"])
        times, frames = footprint_frames(track, tau, sc.t_obs, sc.grid, cfg, N_FRAMES)
    return sc, r, top, tau, times, frames


with st.sidebar:
    st.markdown('<div class="oe-label">Scenario controls</div>', unsafe_allow_html=True)
    seed = st.number_input("Seed", min_value=1, max_value=999, value=7, step=1,
                           help="Seed 7 plants the release on the confuser — the hard case.")
    hidden = st.toggle("Hide the polluter", value=False,
                       help="Drops the responsible vessel from the AIS. There is then no "
                            "correct vessel, and H₀ should take the mass.")
    st.divider()
    st.markdown('<div class="oe-label">System parameters</div>', unsafe_allow_html=True)
    cfg = Config()
    st.markdown(f"""
    <table class="oe-readout">
      <tr><td>Ensemble members</td><td>{cfg.n_members}</td></tr>
      <tr><td>Particles / member</td><td>{cfg.n_particles}</td></tr>
      <tr><td>Prior &pi;&#8320; (unknown)</td><td>{cfg.prior_unknown:.2f}</td></tr>
      <tr><td>Floor L&#8320; (unknown)</td><td>{cfg.l0_unknown:.2f}</td></tr>
    </table>
    """, unsafe_allow_html=True)

sc, r, top, tau, times, frames = attribute(int(seed), bool(hidden))
mins = lambda t: (np.asarray(t, dtype=float) - T_START) / 60.0  # noqa: E731

FLAT = {"delta_color": "off", "delta_arrow": "off"}   # a plain readout, no dashboard chrome

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("top-ranked vessel", top or "—",
              f"p = {r.by_vessel.iloc[0]:.3f}" if top else None, **FLAT)
    c2.metric("unknown source H₀", f"{r.p_unknown:.3f}",
              "ranks first" if (not len(r.by_vessel) or r.p_unknown > r.by_vessel.iloc[0])
              else "ranks below top vessel", **FLAT)
    c3.metric("planted vessel (ground truth)", sc.true_mmsi or "not in the AIS",
              "hidden — H₀ is the right answer" if hidden
              else ("recovered" if top == sc.true_mmsi else "MISSED"), **FLAT)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Scene")
    with st.container(border=True):
        i = len(times) - 1 if times is not None else None
        if times is not None:
            # The scrub is the argument: at the release the oil sits on the track, and by
            # t_obs the ocean has carried and turned it onto the observed slick. Geometry
            # alone only ever sees the last frame.
            i = st.select_slider(
                "Drift clock", options=range(len(times)), value=len(times) - 1,
                format_func=lambda k: f"+{(times[k] - tau[0]) / 60:.0f} min after release",
                help="Forward drift of the top candidate's discharge, ensemble agreement "
                     "shaded. The observed slick (orange) stays fixed at t_obs.")
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        plot_scene(ax, sc, footprint=None if i is None else frames[i], highlight_mmsi=top,
                   t=sc.t_obs if i is None else times[i])
        if i is not None:
            ax.set_title(f"predicted slick at +{(times[i] - tau[0]) / 60:.0f} min "
                         f"(blue) vs observed at +{(sc.t_obs - tau[0]) / 60:.0f} min "
                         "(orange)", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
with right:
    st.subheader("Posterior probability")
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(5.4, 2.5))
        plot_posterior(ax, r, true_mmsi=sc.true_mmsi)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Why this answer")
    with st.container(border=True):
        for sentence in why_this_answer(r, None if hidden else sc.true_tau, T_START):
            st.markdown(f"- {sentence}")

    st.subheader("Release window recovery")
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(5.4, 2.5))
        plot_tau(ax, r, None if hidden else sc.true_tau, T_START, top)
        fig.tight_layout()
        st.pyplot(fig)

st.subheader("Ranked hypotheses — vessel, release window (τ)")
with st.container(border=True):
    st.caption(f"Vessel probabilities sum to {r.table['p'].sum():.4f}; H₀ holds "
               f"{r.p_unknown:.4f}. Together they sum to 1 — vessel probabilities alone "
               "never do.")
    table = r.table.head(12).copy()
    table["tau_start_min"] = mins(table["tau_start"]).round(1)
    table["tau_end_min"] = mins(table["tau_end"]).round(1)
    st.dataframe(
        table[["mmsi", "tau_start_min", "tau_end_min", "likelihood", "p"]],
        width="stretch", hide_index=True,
        column_config={
            "mmsi": st.column_config.TextColumn("MMSI"),
            "tau_start_min": st.column_config.NumberColumn("τ start (min)", format="%.1f"),
            "tau_end_min": st.column_config.NumberColumn("τ end (min)", format="%.1f"),
            "likelihood": st.column_config.NumberColumn("Likelihood", format="%.3f"),
            "p": st.column_config.ProgressColumn("Posterior p", format="%.3f",
                                                 min_value=0.0, max_value=1.0),
        },
    )

st.subheader("Calibration")
with st.container(border=True):
    if TRIALS.exists():
        df = pd.read_csv(TRIALS)
        fig, ax = plt.subplots(figsize=(5.0, 4.6))
        reliability_diagram(df, ax, quantile=True)
        fig.tight_layout()
        plot_col, stat_col = st.columns([1.2, 1])
        plot_col.pyplot(fig)
        with stat_col:
            st.markdown(f"""
            <table class="oe-readout">
              <tr><td>Trials</td><td>{len(df)}</td></tr>
              <tr><td>ECE, quantile bins</td><td>{ece(df, quantile=True):.4f}</td></tr>
              <tr><td>ECE, equal-width bins</td><td>{ece(df):.4f}</td></tr>
              <tr><td>Hidden-polluter cases</td><td>{int(df.hidden.sum())}</td></tr>
            </table>
            """, unsafe_allow_html=True)
    else:
        st.info("Run `python scripts/run_calibration.py --trials 200` to produce the "
                "reliability diagram.")
