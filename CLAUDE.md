# HamletRNAWorld — working notes for Claude

Read this first. It is the accumulated set of things that are easy to get
wrong here, most of which were learned by getting them wrong.

New to the repo and trying to get it running? → **`v7/docs/SETUP.md`**
Want to know how the simulation works? → **`v7/docs/ARCHITECTURE.md`**

## What this is

An art/research simulation. A population of simulated *C. elegans* worms whose
300-neuron connectomes drive a body crawling over a scrolling field of words
from **Hamlet**. A worm "eats" a word by touching it; the eaten words are its
poem. Each on-screen word emits a 12-channel chemosensory signal into the
worm's amphid neurons, so the worm *tastes* meaning and steers toward it. The
connectome weights evolve under a Natural Evolution Strategy, judged for poetic
quality by a Claude model.

Public site: **https://wordswordsworms.org**

## Hard rules

1. **`v1`..`v7` are evolutionary snapshots. Never modify an older version.**
   Each forks its predecessor. **v7 is active.** v6 is kept runnable as the
   comparison baseline — treat it as read-only.

2. **Git is CODE-ONLY.** The whole `data/` tree is ignored. Four servers racing
   to push per-generation artifacts to one branch failed non-fast-forward and
   bloated history with ~19M lines of JSON.

3. **Do NOT `git rm` or `git reset` the `data/` files that are still tracked**
   from before that change. They exist *only on this disk* and are waiting on
   an off-disk backup. When the generation purge deletes one, leave the
   deletion unstaged. There are ~215 unpushed data commits for the same reason
   — do not "clean them up".

4. **There is no pytest in the venv.** Run tests with:
   ```
   /home/web/.venv/bin/python tests/run_all.py            # all
   /home/web/.venv/bin/python tests/run_all.py evolution  # substring filter
   ```
   Several test modules are pytest-style with no `__main__` block, so running
   them directly executes **nothing and exits 0**. `run_all.py` imports each
   module and calls every `test_*` function. Don't trust a bare
   `python tests/test_foo.py` that prints nothing.

5. **`/home/web/.wormlet.env` must stay secrets-only** (`ANTHROPIC_API_KEY`,
   `WORMLET_DEBUG_SECRET`). systemd's `EnvironmentFile=` *shadows* `Environment=`,
   so any operational var in there silently overrides every unit file.

6. **Anthropic structured-output schemas reject `maxItems` / `minItems` /
   `minimum` / `maximum` / `minLength`.** Enforce bounds in the prompt and
   client-side, not in the schema, or the call 400s.

## Analysing whether the worms are learning

**Fitness regime changed 2026-08-15** (`fitness.json` carries `window_floor`
when the new regime scored it): it is now the per-window MEAN with the divisor
floored at 12, so volume no longer pays — hunger enforces eating instead.
**Every generation scored before then used a SUM over judged windows, which
rewarded eating VOLUME.** When analysing old generations, always normalise by
`windows_scored` before concluding anything about poem quality; a flask could
gain fitness purely by eating more. Never compare fitness values across the
regime boundary.

Also check for scoring-regime changes before reading a trend: `GAMMA` changed
2.5 → 1.5 at v6 generation 13 and jumped fitness 3.7 → 27.3 in a single
generation. Several "learning" curves in this project's history are config
changes or degenerate exploits, not learning. Per-worm fitness lives at
`data/<datadir>/generations/<flask>/gen-NNNN/<worm>/fitness.json`; σ and the
σ-scheme are in that generation's `metadata.json`; the parent step size is
`delta_norm` in `selection.json`.

## Current state (2026-08-13)

- **Chemosensory encoder: frozen UMAP** (`WORMLET_ENCODER=umap`, the default).
  The v7 learned encoder is still present and tested under
  `WORMLET_ENCODER=learned`, but it went nearly blind in production — channel
  std 0.027 over the range 0.40-0.55, against UMAP's 0.131 over the full 0-1 —
  and its (1+1)-ES accepted 2-7 mutations in 101 generations.
- **PC11 is a deterministic POS grammar** (`server/pos_grammar.py`): an
  interpolated POS trigram over Hamlet's dialogue, max-normalised per context.
  Not learned.
- **The NES step is decoupled from σ.** It used to carry a `1/σ` factor while
  the sampling radius scaled with σ, so at σ=0.02 the parent stepped 43×
  further than it had sampled. Now the true natural gradient plus a trust
  region; step/sampling-radius is a constant 0.170 at every σ.
- **No repetition penalty.** Fitness is the judge's rating alone.
- **Fitness is quality-per-window since 2026-08-15** (mean with divisor
  floored at `FITNESS_WINDOW_FLOOR=12`). Volume is enforced by hunger
  (starvation), not by the judge. The lifelike lineage on droog
  (`data-lifelike-2`, flask_1) crossed the regime at gen 8.
- **Plasticity saturation is observable:** `plasticity_capped`/
  `plasticity_edges` in worm snapshots and `wormlet_plasticity_capped` in
  VictoriaMetrics. If capped/edges trends toward 1 across generations, the
  NES has found "pin everything at the cap" as a strategy and plasticity is
  an amplifier, not learning. **It did, at gen 3 of `data-trio-3`** (five of
  ten beowulf worms, `eta` +0.047 → +0.330 in one generation), and the
  worms crawled in circles.
- **The lifelike genes are LOG-SCALED time constants since 2026-08-16**, and
  every lineage was cold-started for it (`data-trio-4`). Before that they
  were linear numbers sharing the NES's single additive σ with the synapse
  weights: at σ=0.149 against a `baseline_pull` default of 0.02, **54% of
  all worm-generations sat pinned at zero forgetting and 33% at zero
  plasticity, from generation 1 in all three flasks** — the rule genes were
  never under selection, they were noise. Now the genome stores log(rate),
  so one σ is a ~16% change whatever the units (mutational effects on rates
  are multiplicative in nature), and zero is unreachable. Measured pinning
  after the change: 0%. Phenotype defaults are unchanged (`tau_forget_s`
  40 s = the STM equivalent for a 112-minute life).
- **Mutations are heavy-tailed since 2026-08-16** (`MUTATION_DF=3`,
  `server/evolution.py`): children are drawn from a Student-t, not a
  Gaussian, because measured mutational-effect distributions are L-shaped —
  mostly tiny, with a rare large tail — and a Gaussian makes neither, it
  makes middling changes to every coordinate every time. Same σ, same
  variance; P(|ε|>5σ) goes 0.000% → 0.285%. This is safe ONLY because the
  update now uses the sampling law's score function (`mutation_score`), not
  the raw perturbation: for a Gaussian the two are identical, for the t the
  score REDESCENDS, so a 10σ child informs the step a fifth as much as a 1σ
  one. `MUTATION_DF=inf` restores the Gaussian exactly and is how the A/B is
  written. The t is MULTIVARIATE — one chi-square scale per child, not per
  coordinate — because at d=3,696 per-coordinate tails vanish into
  concentration of measure: per-child ||ε||/√d spans 0.98-1.04 under a
  Gaussian, 0.91-2.90 under independent t, and 0.35-29.5 with a shared
  scale. The L-shape is a fact about individuals, not coordinates.
  **What is NOT modelled:** real mutations are also SPARSE (a handful of
  loci per genome per generation); dense perturbation of all ~3,700
  dimensions every generation is a deliberate compression of evolutionary
  time. Do not "fix" this: at a realistic few loci per child and ~6 fresh
  children a generation, each of the 3,689 weights would be touched well
  under once across a 100-generation run — the search would not move at
  all. Real evolution buys sparsity with population sizes of 10⁴-10⁶ and
  thousands of generations; this experiment has 10 worms and ~100.
- **The delta cap is PROPORTIONAL since 2026-08-16:**
  `|delta| <= min(DELTA_CAP, DELTA_CAP_FRAC * |w|)`, so a synapse may at
  most double and learning can't reorder the connectome. The old flat +10
  amplified the median synapse (|w| 2.0) 6x and the weakest 11x while the
  strongest moved 2.7x — saturation flattened every weight to magnitude
  ~10-12 and erased the structure. Measured at full saturation, same seed:
  drift/min 204 (flat) vs 985 (proportional), i.e. spinning on the spot
  versus travelling. `DELTA_CAP_FRAC = inf` restores the old rule exactly
  and is how the A/B is written in `tests/test_lifelike.py`.
- **Habituation exists but is OFF everywhere** (`WORMLET_HABITUATION=1` to
  enable): per-neuron chemosensory adaptation — a static smell fades
  (subtractive EMA baseline, `adapt_rate` gene), silence recovers it, eating
  dishabituates (`dishab_relief` gene). Snapshot key `habituation` is the
  baseline L1: pinned at 0 across a lineage means adapt_rate evolved to 0.
  Like the other lifelike genes, only a cold-started lineage evolves the two
  new params — every existing lineage (including `data-lifelike-2`) runs the
  clipped defaults.
- **Open question:** neither v6 nor v7 has ever demonstrably learned to write
  better poetry once fitness is normalised by volume. The above fixes remove
  the known mechanical reasons it *couldn't*; whether it now does is untested.
- The 8 live poetry flasks carry 101 generations of connectome evolved against
  the old near-blind smell field, so expect a transient.

## Ops quick reference

Four systemd processes, 2 flasks × 16 worms each (8 flasks / 128 worms total),
each with its **own** `WORMLET_DATA_DIR`:

| unit | port | data dir | flasks |
|---|---|---|---|
| `wormlet-poetry-1` | 8000 | `data/poetry-1` | 1–2 |
| `wormlet-poetry-2` | 8010 | `data/poetry-2` | 3–4 |
| `wormlet-poetry-3` | 8020 | `data/poetry-3` | 5–6 |
| `wormlet-poetry-4` | 8030 | `data/poetry-4` | 7–8 |

```bash
sudo -n systemctl restart wormlet-poetry-1        # canary one first
curl -s localhost:8000/healthz | python3 -m json.tool
sudo -n journalctl -u wormlet-poetry-1 -f
```

`/healthz` reports the live encoder mode — trust it over assumptions about
which code is loaded. Restarting is safe between rollovers: generations persist
on disk. Because the four processes have separate data dirs, the generations
viewer reads its siblings' dirs read-only to show all 8 flasks; flask ids are
`poetry-N:flask_M` and the separator **must not** be `/` or the
`/{flask}/{gen}` route swallows it.

## Style

Match the surrounding code: dense explanatory comments that say *why*,
especially recording what was measured and what was ruled out. This codebase
documents its own dead ends on purpose — keep that up.
