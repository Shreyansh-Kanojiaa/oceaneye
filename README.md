# OCEAN-EYE

**Attribute a satellite-observed oil slick to the vessel most likely to have released it,
as a calibrated probability with an explicit unknown-source hypothesis.**

Smart India Hackathon 2026 · Problem Statement **SIH26143** (NTRO, Space Technology, Software)
Team **Kernel Panic**, SRMIST

> Every scenario in this repo is **synthetic** — a simulated ocean, generated AIS, and a
> slick produced by drifting particles, never a real detection. Output is a **ranked
> probability for investigation**, never a claim that a vessel caused a spill. See
> [CLAUDE.md](CLAUDE.md) for the full set of project non-negotiables and build history.

---

## The idea

Operational slick-attribution systems today (e.g. SkyTruth Cerulean) work by **geometry**:
is the observed slick shaped like a candidate vessel's track, and near it? That only holds
if the ocean hasn't moved the oil since release. Over a few hours, it usually has.

OCEAN-EYE attributes by **physics** instead: it drifts oil forward from each candidate
vessel's track, through an ensemble of plausible ocean states, and asks which candidate's
drifted slick best reproduces what was actually observed. Because a discharging ship is a
moving **line source**, not a point, it also infers *when along the track* — the release
window τ — not just *which vessel*.

The output is a Bayesian posterior over `(vessel, τ)` plus an explicit **H₀ "unknown
source"** hypothesis that always carries some probability mass. When the true polluter
isn't even in the candidate list, the system should say so — decline to accuse — rather
than confidently name the nearest ship.

`experiments/head_to_head.py` measures this claim directly: it reconstructs Cerulean's
published geometry-only method as a baseline and compares its top-1 accuracy against ours
as the ocean displaces the slick further from the release track.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                              # full test suite

# one synthetic scenario, attributed, saved as a 4-panel figure
python scripts/run_demo.py --seed 7
python scripts/run_demo.py --seed 7 --hide-polluter   # H0 should dominate

# 200 simulate-and-recover trials -> outputs/reliability.png + outputs/trials.csv
python scripts/run_calibration.py --trials 200

# interactive UI (reads outputs/trials.csv if present, for the calibration panel)
streamlit run app/streamlit_app.py

# geometry vs physics benchmark (optional, needs run_calibration-scale time)
python experiments/head_to_head.py --run --n 100 --start 201
python experiments/head_to_head.py --analyse
```

Dependencies are deliberately minimal: `numpy scipy pandas matplotlib scikit-learn
streamlit`, plus `pytest ruff` for dev. Nothing else gets added without discussion — see
CLAUDE.md §5 and §3 ("Out — do not start these").

---

## How it works

One synthetic scenario flows through the pipeline in this order:

```
config.py        run parameters (Config), the shared scenario frame, the ensemble split
fields.py         │  analytic ocean: double-gyre current + rotating wind, perturbed per member
ais.py            │  synthetic vessel tracks (Track, interpolation) — one is always a confuser
truth.py          │  plants a release on one vessel, drifts it under a HELD-OUT member,
                   │  hands back only what a satellite would see (the observed mask)
drift.py          │  seed_line_source + advect: the forward physics, shared by truth
                   │  and by every candidate hypothesis attribution tests
slick.py          │  particles -> boolean mask on a fixed Grid, + shape statistics
                   │  (area, centroid, orientation, elongation)
similarity.py      │  Sim(predicted, observed) in [0, 1]: IoU + centroid proximity +
                   │  orientation + extent, proximity-gated so distant matches vanish
attribution.py     ▼  gate -> likelihood -> Bayesian posterior over (vessel, τ) + H0
calibrate.py          simulate many scenarios, check whether posterior confidence
                       matches observed accuracy (ECE, reliability diagram, isotonic fit)
baseline.py           geometry-only reconstruction of Cerulean's method, for comparison
plotting.py           shared matplotlib panels (scene, posterior bars, τ, reliability)
app/streamlit_app.py  interactive UI wiring the above together
scripts/*.py          batch runners: one demo scenario, or N calibration trials
experiments/          head-to-head benchmark: physics vs geometry, accuracy vs displacement
```

The critical invariant that makes calibration meaningful: `truth.py` drifts the *observed*
slick under one ensemble member that `attribution.py` never averages over
(`config.truth_member` vs `config.ensemble_members`). Attribution is always scoring itself
against a forecast error it didn't get to see in advance.

**Per-component detail:** each module above has a matching write-up under `team-docs/`
(generated locally, gitignored — see below) with the design decisions, key functions, and
demo talking points for whoever presents that piece.

---

## Repo layout

```
oceaneye/
├── CLAUDE.md              project brief, non-negotiables, build order, schedule
├── data/scenarios/        saved demo scenarios (JSON, committed)
├── outputs/                figures + trial CSVs produced by scripts (gitignored)
├── src/oceaneye/
│   ├── config.py            Config, seeds, shared scenario frame, ensemble split
│   ├── fields.py            currents + wind, ensemble members
│   ├── drift.py             advection, line-source seeding
│   ├── slick.py             particles -> mask, morphology
│   ├── similarity.py        predicted vs observed score
│   ├── ais.py                Track, interpolation, synthetic tracks
│   ├── attribution.py       gate, likelihood, posterior with H0
│   ├── truth.py              simulate-and-recover scenarios
│   ├── calibrate.py         isotonic, ECE, reliability diagram
│   ├── baseline.py          geometry-only attribution (Cerulean-style), for benchmarking
│   └── plotting.py          shared matplotlib helpers
├── app/streamlit_app.py     interactive demo UI
├── scripts/
│   ├── run_demo.py           one scenario -> figure + printed table
│   └── run_calibration.py    N trials -> reliability diagram + ECE
├── experiments/
│   └── head_to_head.py       physics vs geometry benchmark
└── tests/                    one test file per module above
```

## Testing and linting

```bash
pytest                                 # full suite
pytest -m "not slow"                    # skip the Streamlit end-to-end test
ruff check src app scripts tests        # must be clean before every commit
```

## Team documentation (not in this repo)

Running `pip install -e ".[dev]"` alone won't produce them — ask whoever last ran the docs
generation, or regenerate by asking Claude Code to update `README.md` and `team-docs/`
again. A `team-docs/` folder holds one write-up per core component, meant to be split
across the team so each person can explain their piece to the judges. It's listed in
`.gitignore` on purpose — internal prep material, not part of the deliverable — so pull it
from whoever generated it locally rather than expecting it after a fresh clone.

## Licence

Apache License 2.0 — see [LICENSE](LICENSE).
