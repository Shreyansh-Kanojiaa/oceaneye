"""Run N simulate-and-recover trials and write outputs/reliability.png.

    python scripts/run_calibration.py --trials 200

Everything here is SYNTHETIC: the ocean is analytic, the AIS is generated, and the slick
is drifted rather than segmented from SAR. That is what makes ground truth -- and so
calibration -- possible at all.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oceaneye.calibrate import (  # noqa: E402
    bin_edges,
    ece,
    fit_isotonic,
    reliability_diagram,
    run_trials,
)
from oceaneye.config import Config  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"SYNTHETIC SCENARIOS -- {args.trials} trials, {cfg.n_members} members, "
          f"{cfg.n_particles} particles", flush=True)
    df = run_trials(args.trials, cfg, progress=not args.quiet)
    df.to_csv(OUT / "trials.csv", index=False)
    print(f"\n{len(df)} trials in {(time.time() - t0) / 60:.1f} min", flush=True)

    conf = df["confidence"]
    print(f"\nconfidence: min {conf.min():.3f}  max {conf.max():.3f}  "
          f"mean {conf.mean():.3f}  sd {conf.std():.3f}")
    print(f"accuracy overall {df['correct'].mean():.3f}   "
          f"H0-cases {df[df.hidden]['correct'].mean():.3f} (n={int(df.hidden.sum())})   "
          f"vessel-cases {df[~df.hidden]['correct'].mean():.3f} "
          f"(n={int((~df.hidden).sum())})")

    print("\nhistogram of confidence, 10 equal-width bins over [0, 1]:")
    edges = bin_edges(conf.to_numpy(), 10, quantile=False)
    counts, _ = __import__("numpy").histogram(conf, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True):
        print(f"  [{lo:.1f}, {hi:.1f})  {c:4d}  {'#' * int(50 * c / max(counts.max(), 1))}")
    populated = int((counts > 0).sum())
    quantile = populated < 4
    print(f"  {populated}/10 equal-width bins populated -> "
          f"{'QUANTILE bins' if quantile else 'equal-width bins'} for the diagram")

    print(f"\nECE (equal-width, {args.bins} bins) {ece(df, args.bins, False):.4f}")
    print(f"ECE (quantile,    {args.bins} bins) {ece(df, args.bins, True):.4f}")

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=140)
    reliability_diagram(df, ax, bins=args.bins, quantile=quantile)
    fig.text(0.5, 0.005, "SYNTHETIC SCENARIOS -- simulated ocean, generated AIS",
             ha="center", fontsize=7, color="#b00020")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "reliability.png")
    print(f"wrote {OUT / 'reliability.png'} and {OUT / 'trials.csv'}")

    iso = fit_isotonic(df)
    print("isotonic map, raw -> calibrated: "
          + "  ".join(f"{v:.2f}->{iso.predict([v])[0]:.2f}"
                      for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)))


if __name__ == "__main__":
    main()
