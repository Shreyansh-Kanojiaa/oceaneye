"""Geometry vs physics: does forward drift beat geometric matching, and does geometry
degrade as the ocean carries the oil further from the track?

    python experiments/head_to_head.py --tune                  # baseline weights, seeds 1001-1100
    python experiments/head_to_head.py --run --n 100 --start 201
    python experiments/head_to_head.py --analyse               # tables + outputs/*.png

Polluter-present scenarios only: a hidden polluter has no correct vessel and does not test
ranking. Both attributors see the same gated candidates; only the scoring differs.

Everything here is SYNTHETIC -- analytic ocean, generated AIS, drifted slick. The
comparison is between two methods on the same synthetic truth, not a claim about either
method on real data.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oceaneye import baseline  # noqa: E402
from oceaneye.ais import interpolate  # noqa: E402
from oceaneye.attribution import posterior  # noqa: E402
from oceaneye.config import Config  # noqa: E402
from oceaneye.slick import morphology  # noqa: E402
from oceaneye.truth import make_scenario  # noqa: E402

OUT = ROOT / "outputs"
CSV = OUT / "head_to_head.csv"
TUNE_SEEDS = range(1001, 1101)
BANNER = "SYNTHETIC SCENARIOS — simulated ocean, generated AIS, drifted slick"


def displacement_m(sc) -> float:
    """How far the ocean carried the oil: release-line centroid to observed-mask centroid.

    The observed mask IS the truth-member drift from truth.py; nothing is re-derived from
    the posterior. The release centroid is the vessel's mean position over the planted tau.
    """
    track = next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)
    release = interpolate(track, np.linspace(*sc.true_tau, 50)).mean(axis=0)
    observed = morphology(sc.obs_mask, sc.grid)["centroid_xy"]
    return float(np.hypot(*(observed - release)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def tune(cfg: Config) -> None:
    t0 = time.time()
    scenarios = [make_scenario(s, cfg=cfg) for s in TUNE_SEEDS]
    table = baseline.tune(scenarios, cfg)
    print(f"tuned on seeds {TUNE_SEEDS.start}-{TUNE_SEEDS.stop - 1} "
          f"({len(scenarios)} scenarios) in {time.time() - t0:.0f}s, "
          f"{len(table)} (weights, scale) combinations\n")
    print("top 12 combinations by the baseline's own top-1 accuracy:")
    print(table.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = table.iloc[0]
    print(f"\nbest: parity {best.parity:.1f}  proximity {best.proximity:.1f}  "
          f"temporality {best.temporality:.1f}  scale {best.scale_m:.0f} m  "
          f"top-1 {best.top1:.3f}  MRR {best.mrr:.3f}")
    frozen = dict(baseline.WEIGHTS, scale_m=baseline.PROXIMITY_SCALE_M)
    match = (abs(best.parity - frozen["parity"]) < 1e-9
             and abs(best.proximity - frozen["proximity"]) < 1e-9
             and abs(best.temporality - frozen["temporality"]) < 1e-9
             and best.scale_m == frozen["scale_m"])
    print(f"frozen in baseline.py: {frozen}  -> {'MATCHES' if match else 'DIFFERS from'} "
          "this run")
    # How much the choice matters: accuracy of a few fixed alternatives.
    print("\nfor scale, single-component and equal-weight baselines:")
    for label, q in [("proximity only", "parity==0 and temporality==0"),
                     ("parity only", "proximity==0 and temporality==0"),
                     ("temporality only", "parity==0 and proximity==0"),
                     ("equal weights", "abs(parity-0.3)<0.05 and abs(proximity-0.3)<0.05")]:
        sub = table.query(q)
        if len(sub):
            print(f"  {label:18s} best top-1 {sub.top1.max():.3f}")


def run(cfg: Config, n: int, start: int) -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    print(f"{BANNER}\n{n} polluter-present scenarios, seeds {start}-{start + n - 1}, "
          f"{cfg.n_members} members, {cfg.n_particles} particles", flush=True)
    for i, seed in enumerate(range(start, start + n)):
        try:
            sc = make_scenario(seed, cfg=cfg)
        except ValueError as exc:
            print(f"  seed {seed}: skipped ({exc})", flush=True)
            continue
        b = baseline.baseline_attribute(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)
        r = posterior(sc.obs_mask, sc.tracks, sc.t_obs, sc.grid, cfg)
        ours = r.by_vessel
        row = {
            "seed": seed,
            "true_mmsi": sc.true_mmsi,
            "true_is_confuser": sc.tracks.index(
                next(t for t in sc.tracks if t.mmsi == sc.true_mmsi)) == 1,
            "n_candidates": len(b.table),
            "displacement_m": displacement_m(sc),
            "base_top": b.best["mmsi"] if b.best is not None else None,
            "base_score_top": float(b.best["score"]) if b.best is not None else np.nan,
            "ours_top": ours.index[0] if len(ours) else None,
            "ours_p_top": float(ours.iloc[0]) if len(ours) else np.nan,
            "ours_p_true": float(ours.get(sc.true_mmsi, 0.0)),
            "p_unknown": float(r.p_unknown),
        }
        row["base_correct"] = row["base_top"] == sc.true_mmsi
        row["ours_correct"] = row["ours_top"] == sc.true_mmsi
        rows.append(row)
        pd.DataFrame(rows).to_csv(CSV, index=False)      # checkpoint every scenario
        print(f"  {i + 1:3d}/{n} seed {seed}  disp {row['displacement_m'] / 1e3:5.2f} km  "
              f"base {'OK  ' if row['base_correct'] else 'MISS'}  "
              f"ours {'OK  ' if row['ours_correct'] else 'MISS'}  "
              f"p_top {row['ours_p_top']:.3f}  {(time.time() - t0) / (i + 1):4.1f} s/it",
              flush=True)
    print(f"\nwrote {CSV}  ({len(rows)} rows, {(time.time() - t0) / 60:.1f} min)")


def analyse(bins: int) -> None:
    df = pd.read_csv(CSV)
    n = len(df)
    print(f"{BANNER}\n{n} polluter-present scenarios, seeds {df.seed.min()}-{df.seed.max()}\n")

    print("per-seed comparison:")
    show = df[["seed", "true_mmsi", "true_is_confuser", "n_candidates", "displacement_m",
               "base_top", "base_correct", "ours_top", "ours_correct", "ours_p_top",
               "p_unknown"]].copy()
    show["displacement_m"] = (show["displacement_m"] / 1e3).round(2)
    show = show.rename(columns={"displacement_m": "disp_km"})
    print(show.to_string(index=False))

    kb, ko = int(df.base_correct.sum()), int(df.ours_correct.sum())
    lb, ub = wilson(kb, n)
    lo, uo = wilson(ko, n)
    print(f"\ntop-1 accuracy, 95% Wilson CI, N={n}:")
    print(f"  geometry baseline  {kb / n:.3f}  [{lb:.3f}, {ub:.3f}]  width {ub - lb:.3f}")
    print(f"  physics (ours)     {ko / n:.3f}  [{lo:.3f}, {uo:.3f}]  width {uo - lo:.3f}")
    b = int((df.ours_correct & ~df.base_correct).sum())
    c = int((~df.ours_correct & df.base_correct).sum())
    both = int((df.ours_correct & df.base_correct).sum())
    neither = int((~df.ours_correct & ~df.base_correct).sum())
    p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(f"  paired: both right {both}, neither {neither}, ours-only {b}, baseline-only {c}"
          f"  -> exact McNemar p = {p_mc:.4f}")
    if df.true_is_confuser.any():
        sub = df[df.true_is_confuser]
        print(f"  planted vessel is the confuser in {len(sub)} scenes: baseline "
              f"{sub.base_correct.mean():.3f}, ours {sub.ours_correct.mean():.3f}")

    d = df.displacement_m / 1e3
    print(f"\ndisplacement across seeds (km): min {d.min():.2f}  p25 {d.quantile(.25):.2f}  "
          f"median {d.median():.2f}  p75 {d.quantile(.75):.2f}  max {d.max():.2f}")
    edges = np.unique(np.quantile(d, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(d, edges[1:-1]), 0, len(edges) - 2)
    print(f"\ndegradation, {len(edges) - 1} quantile bins of displacement:")
    print(f"  {'bin (km)':>14s} {'n':>4s}  {'baseline':>18s}  {'ours':>18s}")
    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        nb, nk = int(m.sum()), int(df.base_correct[m].sum())
        no = int(df.ours_correct[m].sum())
        rows.append((d[m].mean(), nb, nk / nb, wilson(nk, nb), no / nb, wilson(no, nb)))
        print(f"  {edges[k]:5.2f} – {edges[k + 1]:5.2f} {nb:4d}  "
              f"{nk / nb:.3f} [{wilson(nk, nb)[0]:.2f},{wilson(nk, nb)[1]:.2f}]  "
              f"{no / nb:.3f} [{wilson(no, nb)[0]:.2f},{wilson(no, nb)[1]:.2f}]")
    for label, col in [("baseline", "base_correct"), ("ours", "ours_correct")]:
        right, wrong = d[df[col]], d[~df[col]]
        if len(right) and len(wrong):
            p = mannwhitneyu(right, wrong, alternative="two-sided").pvalue
            print(f"  {label}: displacement when right median {right.median():.2f} km, "
                  f"when wrong {wrong.median():.2f} km; Mann-Whitney p = {p:.4f}")
        else:
            print(f"  {label}: no {'wrong' if not len(wrong) else 'right'} cases, "
                  "no trend testable")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
    accs = [kb / n, ko / n]
    errs = [[max(0.0, accs[0] - lb), max(0.0, accs[1] - lo)],
            [max(0.0, ub - accs[0]), max(0.0, uo - accs[1])]]
    ax.bar(["geometry\n(Cerulean-style)", "physics\n(forward drift)"], accs,
           yerr=errs, capsize=6, color=["#9ecae1", "#08519c"], width=0.55)
    for i, (a, top) in enumerate(zip(accs, (ub, uo), strict=True)):
        ax.text(i, top + 0.025, f"{a:.2f}", ha="center", fontsize=9)   # clear of the cap
    ax.set(ylim=(0, 1.08), ylabel="top-1 accuracy",
           title=f"Head-to-head, N={n} synthetic scenes\n95% Wilson CIs; "
                 f"McNemar p={p_mc:.3f}")
    ax.title.set_fontsize(9)
    fig.text(0.5, 0.005, BANNER, ha="center", fontsize=6.5, color="#b00020")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "head_to_head.png")

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=140)
    xs = [r[0] for r in rows]
    for j, (label, colour) in enumerate([("geometry baseline", "#9ecae1"),
                                         ("physics (ours)", "#08519c")]):
        ys = [r[2 + 2 * j] for r in rows]
        ci = [r[3 + 2 * j] for r in rows]
        # max(0, .): a Wilson bound can land a rounding error past the point estimate.
        ax.errorbar(xs, ys, yerr=[[max(0.0, y - c[0]) for y, c in zip(ys, ci, strict=True)],
                                  [max(0.0, c[1] - y) for y, c in zip(ys, ci, strict=True)]],
                    fmt="o-", color=colour, capsize=4, lw=1.6, label=label)
    for x, r in zip(xs, rows, strict=True):
        ax.text(x, 1.03, f"n={r[1]}", ha="center", fontsize=7, color="0.35")
    ax.set(xlabel="displacement of the slick centroid, release → observation (km)",
           ylabel="top-1 accuracy", ylim=(0, 1.1))
    ax.set_title("Accuracy vs how far the ocean carried the oil "
                 f"({len(rows)} quantile bins, 95% Wilson CIs)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.5, 0.005, BANNER, ha="center", fontsize=6.5, color="#b00020")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "degradation.png")
    print(f"\nwrote {OUT / 'head_to_head.png'} and {OUT / 'degradation.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=201)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()
    cfg = Config()
    if args.tune:
        tune(cfg)
    if args.run:
        run(cfg, args.n, args.start)
    if args.analyse:
        analyse(args.bins)
    if not (args.tune or args.run or args.analyse):
        ap.print_help()


if __name__ == "__main__":
    main()
