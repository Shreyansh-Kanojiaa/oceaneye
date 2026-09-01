# OCEAN-EYE — Prototype

Smart India Hackathon 2026 · Problem Statement **SIH26143** (NTRO, Space Technology, Software)
Team **Kernel Panic**, SRMIST · Owner: Shreyansh · **Demo: 4 September 2026**

> Attribute a satellite-observed oil slick to the vessel most likely to have released it,
> as a **calibrated probability** with an explicit **unknown-source** hypothesis.

---

## 0. How to work in this repo

You are the primary builder. Shreyansh integrates, demos, and makes product decisions.

**Working rules — these matter more than speed.**

1. **Run everything you write.** Never hand over code you have not executed. After each
   module, run its acceptance checks and paste the actual output.
2. **Build in the order in §6.** Each module is testable alone. If time runs out, working
   pieces beat one integrated thing that does not run.
3. **One module per turn.** Write it, test it, report, stop. Do not chain three modules and
   present a wall of untested code.
4. **Small commits**, one per module, message like `feat(drift): line-source seeding + advection`.
5. **When a test fails, fix the code, not the test.** If a test is genuinely wrong, say so
   explicitly and explain why before changing it.
6. **No silent scope growth.** If something in §3 tempts you, it is out. Say so and move on.

**Stop and ask Shreyansh when:**
- A design choice changes what the demo shows or claims
- You want a dependency not listed in §5
- An acceptance check cannot be made to pass and you are considering weakening it
- You think the schedule in §7 is slipping

Report progress against the §7 checklist at the end of every session.

---

## 1. What this is, and what it proves

Five days of runway. The scope below is deliberately small. It is **not** the system in the
architecture diagram — it is the thin vertical slice that proves the one idea worth building.

**The idea:** operational systems today (SkyTruth Cerulean) attribute slicks by **geometry** —
is the slick shaped like the track, and near it? That holds only when the ocean has not moved
the oil. We attribute by **physics**: drift oil forward from candidate tracks through a flow
field and ask which candidate reproduces the observed slick. And we infer **when along the
track** the release happened, because a discharging ship is a moving **line source**, not a point.

If only one thing gets demoed, demo that contrast.

---

## 2. Non-negotiables

Project values, not style preferences. Deadline pressure does not override these.

1. **Never output a claim that a vessel caused a spill.** Output is a ranked probability for
   investigation. Every user-facing surface says so.
2. **Never fabricate a result.** No hardcoded metrics, no invented accuracy, no
   "representative output" that did not come from a run.
3. **Label synthetic data as synthetic, on screen.** The ocean is simulated and the AIS
   probably is too. Simulate-and-recover is the *right* method here — it is the only route to
   ground truth in this domain — but it must never be passed off as a real detection.
4. **H₀ always carries probability mass.** Vessel probabilities do not sum to 1. If they ever
   do, that is a bug.
5. **Seeded and reproducible.** Every stochastic function takes an explicit seed or rng.
   Same seed, same numbers. No global random state.

---

## 3. Scope

### In
Analytic ocean (currents + wind) perturbed into an ensemble · line-source forward drift ·
particles to slick mask · similarity scoring · Bayesian posterior over (vessel, τ) with H₀ ·
simulate-and-recover ground truth · isotonic calibration, ECE, reliability diagram ·
Streamlit UI.

### Out — do not start these
- **Any ML training.** No U-Net, no segmentation, no PyTorch.
- **OpenDrift.** Install risk too high for five days; we write our own advection.
- Real Sentinel-1 download or SAR preprocessing.
- CMEMS / ERA5 API integration (registration alone can take days).
- PostGIS, FastAPI, Docker, tile server, React.
- Dark-vessel detection.

The architecture diagram is the **target design**; this repo is **Phase 1**. That distinction
gets stated in the demo. It is a much better answer than pretending otherwise.

### Fallback ladder
If something slips, drop a rung rather than stalling:
1. Real DMA AIS becomes `ais.synthetic_tracks`
2. 30 ensemble members becomes 8
3. Live Streamlit becomes a recorded screen capture
4. Full pipeline becomes simulate-and-recover script + reliability diagram only

**Rung 4 alone is a defensible result. Protect it first.**

---

## 4. Repo layout

```
oceaneye/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── data/scenarios/           # saved demo scenarios (JSON, committed)
├── outputs/                  # figures produced by runs (gitignored)
├── src/oceaneye/
│   ├── config.py             # Config dataclass, seeds, grid
│   ├── fields.py             # currents + wind, ensemble members
│   ├── drift.py              # advection, line-source seeding
│   ├── slick.py              # particles to mask, morphology
│   ├── similarity.py         # predicted vs observed score
│   ├── ais.py                # Track, interpolation, synthetic tracks
│   ├── attribution.py        # gate, likelihood, posterior with H₀
│   ├── truth.py              # simulate-and-recover scenarios
│   ├── calibrate.py          # isotonic, ECE, reliability diagram
│   └── plotting.py           # shared matplotlib helpers
├── app/streamlit_app.py
├── scripts/run_demo.py
├── scripts/run_calibration.py
└── tests/
```

---

## 5. Environment

Python 3.11+. These dependencies, and nothing else without asking:

```
numpy scipy pandas matplotlib scikit-learn streamlit pytest ruff
```

```bash
pip install -e ".[dev]"
```

---

## 6. Build order, contracts, and acceptance checks

Build in this order. Do not start a module until the previous one's checks pass.

### 6.1 `config.py` + `fields.py`

```python
@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_members: int = 30
    n_particles: int = 400
    dt_minutes: int = 10
    windage: float = 0.03
    diffusivity: float = 12.0        # m^2/s
    grid_res_m: float = 200.0
    sim_weights: tuple = (0.45, 0.25, 0.15, 0.15)
    l0_unknown: float = 0.15         # H0 likelihood floor
    prior_unknown: float = 0.20      # pi_0

def current(xy: np.ndarray, t: float, member: int, cfg: Config) -> np.ndarray   # (n,2) m/s
def wind(xy: np.ndarray, t: float, member: int, cfg: Config) -> np.ndarray      # (n,2) m/s
```

Use a double-gyre style analytic current with slow time dependence, magnitude ~0.1-0.4 m/s.
Wind is a slowly rotating near-uniform field, ~5-10 m/s. Members perturb windage, diffusivity,
and a rotation/scale on the current.

**Acceptance:** member 0 reproducible across calls; different members give different fields;
magnitudes in the ranges above; `current` vectorised over `(n,2)` input.

### 6.2 `drift.py`

```python
def seed_line_source(track: Track, tau: tuple[float, float], n: int, rng) -> Particles
def advect(p: Particles, t0: float, t1: float, member: int, cfg: Config) -> Particles
```

Per step: `x <- x + (u_current + alpha*u_wind)*dt + sqrt(2*K*dt)*N(0,1)`.
Seed particles **uniformly in time across τ**, each at the vessel's interpolated position at
its own seed time. Do not collapse to a point release at the midpoint — this is the whole point.

**Acceptance:** particle count conserved; zero current + zero wind + zero diffusivity means
particles do not move; seeded particles lie along the track rather than at one point;
vectorised, no Python loop over particles.

### 6.3 `slick.py`

```python
def to_mask(p: Particles, grid: Grid) -> np.ndarray                 # bool (H,W)
def morphology(mask, grid) -> dict
    # area_m2, centroid_xy, orientation_rad, elongation, perimeter_m
```

**Acceptance:** a known synthetic ellipse of particles recovers its area within ~10% and its
orientation within ~10 degrees.

### 6.4 `similarity.py`

```python
def score(pred, obs, grid, cfg) -> float    # [0,1]
```

```
Sim = w_iou*IoU + w_cen*exp(-d_centroid/lambda) + w_ori*|cos(dtheta)| + w_ext*min(A)/max(A)
```

**Acceptance:** `score(m, m) == 1.0`; disjoint masks give ~0.0; monotonic, so sliding one mask
away from the other decreases the score.

### 6.5 `ais.py`

```python
@dataclass
class Track:
    mmsi: str
    times: np.ndarray        # unix seconds UTC, ascending
    xy: np.ndarray           # (n,2) projected metres

def interpolate(track, t) -> np.ndarray
def synthetic_tracks(n, cfg, seed) -> list[Track]
```

Synthetic tracks: plausible headings, speeds 6-14 kn, gentle course changes, some crossing.

**Acceptance:** interpolation exact at knots; monotonic times enforced; tracks differ across seeds.

### 6.6 `truth.py`

```python
def make_scenario(seed, hide_polluter: bool = False) -> Scenario
    # .tracks, .true_mmsi (None if hidden), .true_tau, .obs_mask, .t_obs
```

Pick a vessel, pick τ, drift under **one held-out ensemble member** — not the mean, and not a
member the attribution averages over. The observation must not come from the same draw.
With `hide_polluter=True`, drop the true vessel from `.tracks`.

**Acceptance:** observed mask non-empty and plausibly sized; with `hide_polluter=True` the true
MMSI is absent from `.tracks`.

### 6.7 `attribution.py` — the critical path

```python
def gate(obs_mask, tracks, t_obs, cfg) -> list[Track]
def likelihood(track, tau, obs_mask, t_obs, cfg) -> float
def posterior(obs_mask, tracks, t_obs, cfg) -> AttributionResult
    # .table: DataFrame[mmsi, tau_start, tau_end, p]
    # .p_unknown: float
    # .best: row
```

```
P(v, tau | S_obs)  proportional to  L(v, tau) * P(tau|v) * P(v)
L(v, tau) = (1/N) * sum_e Sim( F(v, tau, e), S_obs )
p_unknown = pi_0*L_0 / ( pi_0*L_0 + sum_{v,tau} P(v,tau) )
```

Search τ over a coarse grid of start times and durations along each track. `P(τ|v)` prefers
shorter discharges. Normalise across all `(v, τ)` **and** H₀ together.

**Acceptance:** `table.p.sum() + p_unknown == 1.0` within 1e-9; on a normal scenario the planted
vessel is top-1 and the recovered τ overlaps the true τ; with `hide_polluter=True`, `p_unknown`
is the single largest value. That last behaviour is a **feature to demonstrate**, not something
to tune away.

### 6.8 `calibrate.py`

```python
def run_trials(n, cfg) -> pd.DataFrame      # [confidence, correct]
def fit_isotonic(df) -> IsotonicRegression
def ece(df, bins=10) -> float
def reliability_diagram(df, ax) -> None
```

**Acceptance:** 200 trials run without error and produce `outputs/reliability.png`; ECE is a
real number in [0,1]. **Report whatever ECE comes out.** Do not tune weights until the diagram
looks flattering — that is fitting the demo, not the problem.

### 6.9 `scripts/` and `app/streamlit_app.py`

UI shows: map with tracks, observed vs predicted slick, posterior bars **including
`p_unknown`**, recovered τ vs true τ, reliability diagram, and a persistent
**"SYNTHETIC SCENARIO"** banner.

```bash
python scripts/run_demo.py --seed 7
python scripts/run_demo.py --seed 7 --hide-polluter
python scripts/run_calibration.py --trials 200
streamlit run app/streamlit_app.py
```

---

## 7. Schedule and status

| Day | Target | Status |
|---|---|---|
| 30 Aug | Repo, env, `config.py`, `fields.py` | [x] |
| 31 Aug | `drift.py`, `slick.py`, `similarity.py` | [x] |
| 1 Sept | `ais.py`, `truth.py` | [x] |
| 2 Sept | `attribution.py` — critical path | [x] |
| 3 Sept | `calibrate.py`, Streamlit, rehearsal, backup recording | [ ] |
| 4 Sept | Demo | [ ] |

If 2 September ends without a working posterior, drop to an ensemble of 8 and move on.
If 3 September morning arrives with no posterior, go to fallback rung 4.

---

## 8. Conventions

- Positions in **projected metres** (local equal-area). Convert lat/lon once, at the edges.
- Times **unix seconds UTC**. No naive datetimes.
- Type hints on public functions; docstrings state units.
- Vectorise with NumPy; no Python loops over particles.
- Pure functions where possible, so they test without the UI.
- `ruff check src app scripts tests` clean before each commit.

---

## 9. Demo script (90 seconds)

1. "Detecting the slick is largely solved. Naming the ship is not."
2. Show the scene: tracks, one observed slick.
3. Show the contrast: geometry picks the nearest or most parallel track; the ocean has
   displaced and rotated the slick since release.
4. Run attribution: posterior bars, top candidate, recovered τ, `p_unknown`.
5. Reveal ground truth: recovered vs planted vessel and τ.
6. Re-run with the polluter hidden. `p_unknown` dominates. "The system declines to accuse."
7. Reliability diagram and ECE over 200 trials. "Our probabilities mean what they say."

---

## 10. State these limitations out loud

Volunteering them reads as competence; hiding them reads as naivety, and judges find them anyway.

- Analytic ocean, not CMEMS. Real currents are an adapter, not a redesign.
- Slick is simulated, not segmented from Sentinel-1. Detection is Phase 2.
- Ground truth is synthetic because no public dataset pairs a SAR slick with a confirmed MMSI.
  That constraint is also what makes calibration possible.
- Drift is advection + diffusion + windage. No weathering, no waves, no vertical structure.

---

## 11. Do not

- Do not train a model.
- Do not `pip install opendrift`.
- Do not add a database, REST API, or Docker.
- Do not tune weights to flatter the reliability diagram.
- Do not remove the synthetic-scenario banner.
- Do not put a number on a slide that did not come from a run in this repo.
