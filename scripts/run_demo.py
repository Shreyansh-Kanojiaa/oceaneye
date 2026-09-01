"""One synthetic scenario, attributed, as a four-panel figure and a printed table.

    python scripts/run_demo.py --seed 7
    python scripts/run_demo.py --seed 7 --hide-polluter

Seed 7 is the demo default because its planted vessel IS the confuser -- a near-parallel
track a few km abeam of another -- so the answer is not obvious from geometry.

Output is a ranked probability for investigation, never a determination that a vessel
caused a discharge.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oceaneye.attribution import posterior, predicted_footprint  # noqa: E402
from oceaneye.calibrate import reliability_diagram  # noqa: E402
from oceaneye.config import T_START, Config  # noqa: E402
from oceaneye.plotting import DISCLAIMER, banner, plot_posterior, plot_scene, plot_tau  # noqa: E402
from oceaneye.truth import make_scenario  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hide-polluter", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    sc = make_scenario(args.seed, hide_polluter=args.hide_polluter, cfg=cfg)
    r = posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)
    mins = lambda t: (np.asarray(t, dtype=float) - T_START) / 60.0  # noqa: E731

    print(f"SYNTHETIC SCENARIO  seed {args.seed}"
          f"{'  (polluter hidden)' if args.hide_polluter else ''}")
    print(f"truth: {sc.true_mmsi or 'not present in the AIS'}, "
          f"tau {mins(sc.true_tau)[0]:.1f}-{mins(sc.true_tau)[1]:.1f} min\n")
    print(f"p_unknown (H0) {r.p_unknown:.4f}")
    print("vessel marginals:")
    for mmsi, p in r.by_vessel.items():
        print(f"  {mmsi}  {p:.4f}{'   <-- planted' if mmsi == sc.true_mmsi else ''}")
    print(f"\nsum(vessels) + p_unknown = {r.table['p'].sum() + r.p_unknown:.12f}")
    print("\ntop (vessel, tau) rows:")
    for _, row in r.table.head(6).iterrows():
        print(f"  {row['mmsi']}  tau {mins(row['tau_start']):6.1f}-"
              f"{mins(row['tau_end']):6.1f} min  L {row['likelihood']:.4f}  p {row['p']:.4f}")
    print(f"\n{DISCLAIMER}")

    top = r.by_vessel.index[0] if len(r.by_vessel) else None
    footprint = None
    if top is not None:
        best = r.table[r.table["mmsi"] == top].iloc[0]
        track = next(t for t in sc.tracks if t.mmsi == top)
        footprint = predicted_footprint(
            track, (best["tau_start"], best["tau_end"]), sc.t_obs, sc.grid, cfg)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=130)
    plot_scene(axes[0, 0], sc, footprint=footprint, highlight_mmsi=top)
    plot_posterior(axes[0, 1], r, true_mmsi=sc.true_mmsi)
    plot_tau(axes[1, 0], r, sc.true_tau if not args.hide_polluter else None, T_START, top)

    trials = OUT / "trials.csv"
    if trials.exists():
        reliability_diagram(pd.read_csv(trials), axes[1, 1], quantile=True)
    else:
        axes[1, 1].text(0.5, 0.5, "run scripts/run_calibration.py\nfor the reliability "
                        "diagram", ha="center", va="center", fontsize=9)
        axes[1, 1].set_axis_off()

    banner(fig)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    OUT.mkdir(exist_ok=True)
    name = f"demo_seed{args.seed}{'_hidden' if args.hide_polluter else ''}.png"
    fig.savefig(OUT / name)
    print(f"\nwrote {OUT / name}")


if __name__ == "__main__":
    main()
