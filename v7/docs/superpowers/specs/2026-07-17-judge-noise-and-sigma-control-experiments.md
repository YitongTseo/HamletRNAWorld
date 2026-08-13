# Judge-noise & σ-control experiments

- **Date:** 2026-07-17
- **Status:** Experiment 1 in progress; Experiment 2 designed (runs after Exp 1 picks a judge)
- **Author:** Yitong + Claude
- **Related:** `server/judge.py`, `server/evolution.py`, `server/gardener.py`; memory `project_v7_learned_embedding`

## Why we're doing this

Over 28 generations the worms are **not learning**: `best_score` is a directionless
random walk (F1 18↔34, F2 17↔38) and mean worm fitness *drifts down* (F1 18.5→14.4,
F2 18.8→14.0). Root analysis found two coupled causes:

1. **σ runaway.** The Rechenberg 1/5 rule's success baseline compares each fresh child
   to the *previous generation's mean child fitness* (`evolution.py` `adapt_sigma`
   call, `parent_fitness=state.prev_fresh_mean`). "% of children above the mean" ≈ 50%
   by construction, which is far above the 0.2 target, so σ grows almost every
   generation and pins at `SIGMA_MAX = 3.0`. At σ=3 offspring are near-random redraws,
   so nothing accumulates. (v6 had the opposite bug: it compared against the champion
   *max*, giving ≈0% success → σ floored. Both are the same class of error — comparing
   to a population order-statistic that is nearly constant in σ — at opposite rails.)
2. **A noisy judge.** The LLM judge is non-stationary; even at a good σ the selection
   signal may be dominated by judge noise. If so, *no* σ controller will make this
   learn, and the real fix is a less noisy judge.

These two experiments untangle them, in order:

- **Experiment 1** measures how noisy the judge is under different scoring methods
  (current/absolute vs. pairwise→Bradley-Terry) and representations, and picks the
  least-noisy one.
- **Experiment 2** uses that judge to find a σ-control scheme that actually converges.

## Background: how production scores today (the incumbent)

From `server/judge.py` (verified 2026-07-17):

| property | value |
|---|---|
| model | `claude-haiku-4-5` |
| window | 15 tokens, **non-overlapping** (stride 15); trailing partial kept |
| sampling | `SAMPLE_FRACTION = 0.25` — 25% of windows, seeded by generation number |
| windows/worm | **~36–43** (varies with poem length; gen-28 flask_1 range) |
| call shape | **one batched call per worm** — all sampled windows in a single prompt, one output line `idx,emotional,coherence` per window; rubric is prompt-cached |
| **temperature** | **unset → API default 1.0** (a major, near-free noise lever) |
| fitness | `fitness()` = Σ over scored windows of `repetition_factor · (1.5·(E/100)^1.5 + (C/100)^1.5)` |

Key consequences for the experiment:
- Production noise is **one batched draw per worm** at temp 1.0 — the ~40 window scores
  are correlated (same call), and a whole worm's fitness is a single noisy sample.
- Because seeding is by generation number, production shows the *same windows* every
  time within a generation; the only re-run variability is the LLM sampling itself.
- The historical text we need is on disk: `poem_clean.txt` (full poem → reconstruct all
  windows via `make_windows`) and `scores.jsonl` (the 25% sampled windows + their one
  historical score). **No simulation rerun is needed — we only re-call the judge.**

## Experiment 1 — judge-noise shootout

### Data (4 pools)

Four independent 16-worm pools, each = one flask's generation-28 population (chosen for
being deep into the σ-runaway regime, where noise vs. signal matters most):

- `poetry-1/flask_1/gen-0028`
- `poetry-1/flask_2/gen-0028`
- `poetry-2/flask_1/gen-0028`
- `poetry-2/flask_2/gen-0028`

Four pools give cross-flask replication → statistical significance on the "which config
is least noisy" conclusion. Each worm's candidate window set = its full `make_windows`
(from `poem_clean.txt`), so `m` can range up to ~40+.

### Factors swept

| factor | values | notes |
|---|---|---|
| **method** | (0) production-batch-absolute @ temp 1.0 — *incumbent* · (0b) same @ temp 0 · (P) pairwise→Bradley-Terry @ temp 0 | temp axis isolates sampling-noise |
| **sampling** | random · stratified | how the *m* representative windows are chosen |
| **size m** | 5 · 15 · all(~40) | at m=all, random≡stratified (the "use everything"/production-like point) |

"Stratified" = sort a worm's windows by their historical score and pick `m` spread
across low→mid→high, so the sample can't land all-mediocre; guards the GAMMA-weighted
top windows that carry the fitness signal. "Random" = uniform random windows.

### Cheap-by-construction (the reuse trick)

For **absolute** methods we replicate production's batching: one Haiku call scores *all*
of a worm's candidate windows. Do that **K times** per (worm, temp). Then every
(sampling × size) absolute config is a **free re-aggregation** of those per-window
scores — no extra calls. Only **pairwise** needs fresh calls per (sampling, size),
because it shows different text each comparison; pairwise is therefore the cost driver
and is run on a representative subset (sizes {5, all}, both samplings, single presentation
order + a separate order-swap position-bias pass).

### Repeats

`K = 10` independent judge passes per config. Reproducibility estimates tighten ~1/√K;
extend to K=20 only for configs that come out borderline.

### Metrics

Primary:
- **Rank reproducibility** = mean Kendall τ between the worm-rankings of every pair of
  the K independent passes (also report Spearman ρ). Higher = less noisy. Computed per
  (pool, config), then averaged across the 4 pools with a spread.

Supporting:
- **Absolute:** ICC(2,1) test-retest; signal-to-noise = (between-worm SD)/(same-worm
  repeat SD); score-clustering (effective range used of 1–100).
- **Pairwise:** per-pair agreement across passes; **position-bias flip rate** (A-first
  vs B-first disagreement); transitivity-violation rate.
- **Cost:** judge-call count per config (the axis to trade against stability).
- **Validity proxy (not ground truth):** correlation of each config's mean ranking with
  the grand-mean ranking over *all* passes and methods (best available estimate of
  "true" order), so we don't crown a config that's stable but stably wrong.

### Deliverable

A **stability-vs-cost frontier**: for every config, mean Kendall τ (with cross-pool
error bars) vs. judge-calls. Recommendation = the config on the efficient frontier —
most reproducible ranking per call — with a note on whether pairwise's extra cost buys
enough stability over the best absolute config to be worth it, and how much of the
incumbent's noise was just temperature.

### Call budget

Per 16-worm pool, K=10:
- Absolute (both temps, batched): 16 worms × 10 × 2 temps ≈ **320 calls** (all sampling×size free).
- Pairwise: 120 pairs × 10 × (2 sampling × 2 size) ≈ **4,800 calls** + a ~240-call bias pass.
- ≈ **5.3k calls/pool** → **~21k calls across 4 pools**. Haiku, bounded concurrency ≈ 8 → ~30–45 min wall-clock. Runs alongside the live flasks (their judge fires only at rollover, hours apart), so the public site stays up.

## Experiment 2 — σ-control shootout (design; run after Exp 1)

Using the least-noisy judge from Exp 1, find a σ-scheme that converges. Candidates:

| scheme | success baseline | expectation |
|---|---|---|
| v6 (control) | vs champion max | σ → floor (reproduce the collapse) |
| v7 (control, live) | vs previous-gen mean | σ → ceiling (reproduce the runaway) |
| 1/5-vs-centroid | vs parent centroid θ, **evaluated this gen** (one eps=0 slot, same judge pass) | σ settles at interior fixed point |
| (1+λ)-vs-elite | vs reigning elite, children sampled *around the elite* | σ settles; simpler hill-climb |
| xNES/SNES native | *no baseline* — natural-gradient step-size from ranked utilities | σ self-adapts; no 1/5 rule at all |

Run **offline first** on a sphere-with-injected-judge-noise simulator (cheap,
reproducible, no API) to see which schemes keep σ at a stable interior value AND make
rank-based fitness climb. Promote the 1–2 winners to a live A/B on the flasks. Metrics:
σ trajectory stability, mean/best fitness trend over generations, and whether the
centroid's fitness actually rises. Detailed spec to be appended once Exp 1 completes.

## Code & artifact layout

```
v7/experiments/judge_noise/
  pool_builder.py   # load 4 pools from historical data; make_windows; random/stratified sampling
  judges.py         # absolute-batch + pairwise Haiku calls; Bradley-Terry aggregation
  run_experiment.py # orchestrate configs × K × pools; bounded-concurrency async; resumable cache
  analyze.py        # Kendall τ / ICC / position-bias / cost → results table (JSON + CSV)
  visualize.py      # stability-vs-cost frontier, τ heatmaps, score-clustering, bias plots → HTML report
  results/          # raw judge outputs (cached, resumable) + computed metrics + report.html
```

## How to read the results later

- `results/metrics.json` — per (pool, config): mean/σ Kendall τ, ICC, S/N, position-bias,
  call-count, validity-corr.
- `results/report.html` — the frontier plot is the headline: pick the point that is
  highest-τ for the fewest calls. If temp=0 alone closes most of the gap to pairwise,
  the cheap win is "just set the judge temperature." If pairwise is materially above the
  best absolute config even after temp=0, pairwise earns its O(N²) cost.
- The chosen judge (method + sampling + size + temp) is the input to Experiment 2.

## Non-goals

- Not changing the live evolution loop in this experiment (Exp 1 is pure measurement).
- Not establishing absolute "truth" of poem quality — we measure **reproducibility**
  (and a weak validity proxy), which is what selection actually needs.
- Not solving production O(N²) here; that's a follow-up (sparse/tournament comparison)
  only if pairwise wins.
