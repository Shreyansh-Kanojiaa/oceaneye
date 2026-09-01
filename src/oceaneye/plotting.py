"""Shared matplotlib helpers, so the demo script and the Streamlit app draw the same thing.

Every figure carries the synthetic banner. That is not decoration: the ocean is analytic,
the AIS is generated and the slick is drifted rather than segmented from SAR, and a figure
that leaves the room without saying so is a figure that can be mistaken for a detection.
"""

import numpy as np

BANNER = "SYNTHETIC SCENARIO — simulated ocean, generated AIS, drifted slick"
BANNER_COLOUR = "#b00020"
PAD_KM = 18.0          # half-width of the scene panel around the observed slick
DISCLAIMER = ("Ranked probability for investigation. This is not a determination that any "
              "vessel caused a discharge.")


def banner(fig) -> None:
    """Stamp the synthetic-scenario banner across the top of a figure."""
    fig.text(0.5, 0.985, BANNER, ha="center", va="top", fontsize=9, weight="bold",
             color=BANNER_COLOUR)


def _km(v):
    return np.asarray(v) / 1000.0


def plot_scene(ax, scenario, footprint=None, highlight_mmsi=None) -> None:
    """Tracks, the observed slick, and optionally a predicted ensemble footprint."""
    g = scenario.grid
    h, w = g.shape
    extent = (g.origin[0] / 1e3, (g.origin[0] + w * g.res_m) / 1e3,
              g.origin[1] / 1e3, (g.origin[1] + h * g.res_m) / 1e3)

    if footprint is not None:
        ax.contourf(footprint, levels=[0.15, 0.35, 0.65, 1.01], extent=extent,
                    colors=["#9ecae1", "#4292c6", "#08519c"], alpha=0.45, origin="lower")
    obs = np.ma.masked_where(~scenario.obs_mask, scenario.obs_mask.astype(float))
    ax.imshow(obs, extent=extent, origin="lower", cmap="autumn_r", vmin=0, vmax=1,
              alpha=0.9, interpolation="nearest")

    for track in scenario.tracks:
        hit = track.mmsi == highlight_mmsi
        ax.plot(*_km(track.xy).T, lw=2.0 if hit else 1.0,
                color="#111111" if hit else "#888888", zorder=3 if hit else 2)
        ax.annotate(track.mmsi[-4:], _km(track.xy[len(track.xy) // 2]), fontsize=7,
                    color="#111111" if hit else "#888888",
                    weight="bold" if hit else "normal",
                    textcoords="offset points", xytext=(4, 0))

    # Zoom to the slick plus context: at full track extent (~120 km) a 4 km slick is a
    # dot, and the whole point of the panel is that the ocean has carried the oil off the
    # track that released it.
    ys, xs = np.nonzero(scenario.obs_mask)
    cx = g.origin[0] / 1e3 + (xs.mean() + 0.5) * g.res_m / 1e3
    cy = g.origin[1] / 1e3 + (ys.mean() + 0.5) * g.res_m / 1e3
    ax.set(xlim=(cx - PAD_KM, cx + PAD_KM), ylim=(cy - PAD_KM, cy + PAD_KM),
           xlabel="x (km)", ylabel="y (km)", aspect="equal")
    ax.set_title("observed slick (orange) vs predicted ensemble footprint (blue)",
                 fontsize=9)


def plot_posterior(ax, result, true_mmsi=None) -> None:
    """One bar per vessel plus H0. The bars do not sum to 1 -- H0 carries mass."""
    labels = list(result.by_vessel.index) + ["H₀ unknown source"]
    values = list(result.by_vessel.values) + [result.p_unknown]
    colours = ["#08519c" if m == true_mmsi else
               "#b00020" if m.startswith("H₀") else "#9ecae1" for m in labels]
    ax.barh(labels, values, color=colours)
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
        ax.axvspan(*mins(true_tau), color="#08519c", alpha=0.20, label="planted τ")
    for y, (_, row) in enumerate(rows.iterrows()):
        ax.plot(mins([row["tau_start"], row["tau_end"]]), [y, y], lw=6,
                color="#b00020", alpha=min(1.0, 0.25 + float(row["p"]) * 4))
        ax.text(mins(row["tau_end"]) + 3, y, f"p={row['p']:.3f}", va="center", fontsize=7)
    ax.set(xlabel="minutes after AIS window start", yticks=[],
           title=f"recovered τ for {top}" if top else "no candidate τ")
    ax.invert_yaxis()
    if true_tau is not None:
        ax.legend(fontsize=7, loc="lower right")
