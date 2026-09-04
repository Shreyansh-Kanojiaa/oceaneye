"""Shared matplotlib helpers, so the demo script and the Streamlit app draw the same thing.

Every figure carries the synthetic banner. That is not decoration: the ocean is analytic,
the AIS is generated and the slick is drifted rather than segmented from SAR, and a figure
that leaves the room without saying so is a figure that can be mistaken for a detection.
"""

import matplotlib as mpl
import numpy as np

from .config import Config, ensemble_members
from .fields import current, member_params, wind

# Shared palette. Named once here and reused by the Streamlit chrome (app/streamlit_app.py
# imports these directly) so the plots and the surrounding UI are one consistent instrument,
# not a chart library's defaults sitting inside someone else's colour scheme.
# Dark, control-room tones -- kept flat and desaturated throughout: nothing here is meant
# to glow, so accents get their contrast from value against PAPER/PANEL, never saturation.
INK = "#e7ecf0"        # primary text
SLATE = "#8b96a3"      # secondary text, axis labels, muted tracks and bars
LINE = "#2b333d"       # hairline borders, gridlines -- subtle, not a fill colour
PAPER = "#0c1016"      # page / figure background
PANEL = "#141a22"      # panel / sidebar background, one step up from PAPER
MARINE = "#5b9bd5"     # single accent colour: the true/highlighted vessel, primary emphasis
ALERT = "#c1554c"      # H0 / miss / caution — used sparingly, never as decoration
CONFIRM = "#5f9468"    # recovered / correct
FLOW = "#41566b"       # current/wind arrows: below the tracks in value, never competing

BANNER = "SYNTHETIC SCENARIO — simulated ocean, generated AIS, drifted slick"
BANNER_COLOUR = ALERT
PAD_KM = 18.0          # half-width of the scene panel around the observed slick
DISCLAIMER = ("Ranked probability for investigation. This is not a determination that any "
              "vessel caused a discharge.")

# Applied once at import: every figure in the project -- the app, run_demo.py, the
# calibration script, the head-to-head benchmark -- draws with the same restrained,
# gridded, chart-like style instead of matplotlib's defaults.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Public Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "axes.facecolor": PANEL,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": LINE,
    "grid.linewidth": 0.6,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "legend.frameon": False,
})


def banner(fig) -> None:
    """Stamp the synthetic-scenario banner across the top of a figure."""
    fig.text(0.5, 0.985, BANNER, ha="center", va="top", fontsize=9, weight="bold",
             color=BANNER_COLOUR)


def _km(v):
    return np.asarray(v) / 1000.0


def _mean_flow(xy_m, t, cfg):
    """Ensemble-MEAN current, wind and windage at `xy_m` (..., 2) metres, time `t`.

    The mean is what to draw: individual members perturb speed by ~30% and direction by
    ~15 deg, and drawing one member's field would present a single draw as the ocean.
    """
    members = list(ensemble_members(cfg))
    cur = np.mean([current(xy_m, t, m, cfg) for m in members], axis=0)
    wnd = np.mean([wind(xy_m, t, m, cfg) for m in members], axis=0)
    alpha = float(np.mean([member_params(m, cfg).windage for m in members]))
    return cur, wnd, alpha


def plot_flow(ax, t, cfg=None, n=9) -> None:
    """Ensemble-mean current quiver plus the wind arrow, at time `t`, over `ax`'s limits.

    Current gets the quiver because it is the field with structure -- the gyre is what
    turns and stretches a slick. Wind is near-uniform by construction, so a full quiver of
    it would be nine copies of one arrow; it gets a single arrow and its windage readout.

    There is no wave field to draw. Drift here is advection + windage + diffusion and
    nothing else (CLAUDE.md 10), and an arrow for a quantity the model does not have would
    be a claim this repo cannot support.
    """
    cfg = cfg or Config()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    cur, wnd, alpha = _mean_flow(np.stack([gx, gy], axis=-1) * 1e3, t, cfg)

    # scale_units="xy" with scale=1/KM_PER_MS: arrow length is physical, so a 0.3 m/s
    # current reads as the same length in every panel and between scrub frames.
    # Arrow length per m/s. Set so the fastest current the field produces (~0.4 m/s) draws
    # about one grid spacing -- any longer and neighbouring arrows overlap into a smear,
    # any shorter (this was /12) and a 0.2 m/s current is an unreadable stub.
    km_per_ms = (x1 - x0) / 3.0
    quiver = dict(angles="xy", scale_units="xy", scale=1.0 / km_per_ms,
                  width=0.004, headwidth=3.5, headlength=4.0)
    ax.quiver(gx, gy, cur[..., 0], cur[..., 1], color=FLOW, zorder=1, **quiver)

    speed = float(np.hypot(*cur.reshape(-1, 2).T).mean())
    w_mean = wnd.reshape(-1, 2).mean(axis=0)
    w_speed = float(np.hypot(*w_mean))
    # Wind through the SAME quiver scale as the current, and at the windage drift it
    # actually contributes rather than at wind speed -- 7.5 m/s drawn against a 0.2 m/s
    # current would be a 30x arrow for the smaller of the two effects on the oil. Same
    # scale means the two arrows can be compared by eye, which is the only reason to draw
    # them together.
    tx, ty = x0 + 0.10 * (x1 - x0), y1 - 0.10 * (y1 - y0)
    ax.quiver([tx], [ty], *(alpha * w_mean)[:, None], color=INK, zorder=4, **quiver)
    # Both captions sit on top of whatever track happens to run under them, so they carry
    # their own ground rather than relying on the panel being empty there.
    plate = dict(fontsize=6.5, va="bottom", zorder=5,
                 bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.75, pad=1.5))
    ax.text(tx, ty + 0.028 * (y1 - y0), color=INK, s=(
        f"wind {w_speed:.1f} m/s → {alpha * w_speed:.2f} m/s windage"), **plate)
    ax.text(x0 + 0.04 * (x1 - x0), y0 + 0.03 * (y1 - y0), color=SLATE, s=(
        f"ensemble-mean flow · current {speed:.2f} m/s (blue-grey arrows) · "
        "no wave field modelled"), **plate)


def plot_scene(ax, scenario, footprint=None, highlight_mmsi=None, t=None, cfg=None) -> None:
    """Tracks, the observed slick, and optionally a predicted ensemble footprint.

    With `t` set (unix seconds), the ensemble-mean flow at that time is drawn underneath.
    """
    g = scenario.grid
    h, w = g.shape
    extent = (g.origin[0] / 1e3, (g.origin[0] + w * g.res_m) / 1e3,
              g.origin[1] / 1e3, (g.origin[1] + h * g.res_m) / 1e3)

    # Zoom to the slick plus context: at full track extent (~120 km) a 4 km slick is a
    # dot, and the whole point of the panel is that the ocean has carried the oil off the
    # track that released it. Set FIRST -- the flow quiver and the track labels below both
    # need the limits to know what is actually on screen.
    ys, xs = np.nonzero(scenario.obs_mask)
    cx = g.origin[0] / 1e3 + (xs.mean() + 0.5) * g.res_m / 1e3
    cy = g.origin[1] / 1e3 + (ys.mean() + 0.5) * g.res_m / 1e3
    ax.set(xlim=(cx - PAD_KM, cx + PAD_KM), ylim=(cy - PAD_KM, cy + PAD_KM),
           xlabel="x (km)", ylabel="y (km)", aspect="equal")

    if t is not None:
        plot_flow(ax, t, cfg)

    if footprint is not None:
        ax.contourf(footprint, levels=[0.15, 0.35, 0.65, 1.01], extent=extent,
                    colors=["#9ecae1", "#4292c6", "#08519c"], alpha=0.45, origin="lower")
    obs = np.ma.masked_where(~scenario.obs_mask, scenario.obs_mask.astype(float))
    ax.imshow(obs, extent=extent, origin="lower", cmap="autumn_r", vmin=0, vmax=1,
              alpha=0.9, interpolation="nearest")

    for track in scenario.tracks:
        hit = track.mmsi == highlight_mmsi
        colour = MARINE if hit else SLATE
        kx, ky = _km(track.xy).T
        ax.plot(kx, ky, lw=2.0 if hit else 1.0, color=colour, zorder=3 if hit else 2)
        # First and last AIS fix. A track is exactly the SCENARIO_HOURS window, and a
        # merchant ship covers far more than the panel is wide, so a line that stops
        # mid-panel is a window that ended, not a line that broke. Marking the ends says so.
        ax.plot(kx[[0, -1]], ky[[0, -1]], ls="none", marker="o", ms=3.0, color=colour,
                zorder=3 if hit else 2)
        # Label on the visible stretch, not at the track's own midpoint -- at this zoom the
        # midpoint is usually off-panel, which left most tracks unlabelled.
        seen = np.flatnonzero((kx >= cx - PAD_KM) & (kx <= cx + PAD_KM)
                              & (ky >= cy - PAD_KM) & (ky <= cy + PAD_KM))
        j = seen[seen.size // 2] if seen.size else len(kx) // 2
        ax.annotate(track.mmsi[-4:], (kx[j], ky[j]), fontsize=7, color=colour,
                    weight="bold" if hit else "normal", zorder=4,
                    textcoords="offset points", xytext=(5, 3))

    ax.set_title("observed slick (orange) vs predicted ensemble footprint (blue)",
                 fontsize=9)


def plot_posterior(ax, result, true_mmsi=None) -> None:
    """One bar per vessel plus H0. The bars do not sum to 1 -- H0 carries mass."""
    labels = list(result.by_vessel.index) + ["H₀ unknown source"]
    values = list(result.by_vessel.values) + [result.p_unknown]
    colours = [MARINE if m == true_mmsi else
               ALERT if m.startswith("H₀") else SLATE for m in labels]
    ax.barh(labels, values, color=colours, edgecolor=PAPER, linewidth=0.6)
    ax.invert_yaxis()
    ax.set(xlim=(0, max(max(values) * 1.25, 0.1)), xlabel="posterior probability")
    for y, v in enumerate(values):
        ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title(f"posterior — vessels sum to {sum(values[:-1]):.3f}, H₀ holds the rest",
                 fontsize=9)


def plot_tau(ax, result, true_tau, t0, top_mmsi=None) -> None:
    """Recovered release windows against the planted one, in minutes after window start."""
    mins = lambda t: (np.asarray(t, dtype=float) - t0) / 60.0  # noqa: E731
    top = top_mmsi or (result.by_vessel.index[0] if len(result.by_vessel) else None)
    rows = result.table[result.table["mmsi"] == top].head(6) if top is not None else []

    if true_tau is not None:
        ax.axvspan(*mins(true_tau), color=MARINE, alpha=0.15, label="planted τ")
    for y, (_, row) in enumerate(rows.iterrows()):
        ax.plot(mins([row["tau_start"], row["tau_end"]]), [y, y], lw=6,
                color=ALERT, alpha=min(1.0, 0.25 + float(row["p"]) * 4))
        ax.text(mins(row["tau_end"]) + 3, y, f"p={row['p']:.3f}", va="center", fontsize=7)
    ax.set(xlabel="minutes after AIS window start", yticks=[],
           title=f"recovered τ for {top}" if top else "no candidate τ")
    ax.invert_yaxis()
    if true_tau is not None:
        ax.legend(fontsize=7, loc="lower right")
