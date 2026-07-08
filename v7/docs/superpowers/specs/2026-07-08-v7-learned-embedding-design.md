# v7 — Learned, context-aware word embedding + evolution fixes + 8-flask scale-up

Status: **approved** (user delegated autonomous implementation 2026-07-08)
Supersedes: v6 (frozen-UMAP embedding, 2-flask poetry, per-flask-only NES)

## Motivation

v6 drove chemosensation with a **frozen** 12-dim UMAP reduction of
`nomic-embed-text-v1.5` (512-dim) embeddings. Two problems:

1. **The embedding never learns.** UMAP is fixed; only the connectome evolves.
   The worm's "sense of taste" can't adapt to the task.
2. **Evolution is effectively dead.** The gardener logged the same two failures
   for 8+ epochs (gens 95–103): σ pinned at the 0.020 floor with *no genome ever
   held*, and the Claude judge handing perfect scores to raw repetition
   ("God God God", "strew'd strew'd strew'd", "yet … yet … yet").

Root cause of the σ collapse (confirmed in `server/generations.py:324`): the
Rechenberg 1/5 success test compared every fresh child against the *previous
generation's single best (max)* score — the reigning champion — instead of the
parent. Almost no child beats the champion under a noisy judge, so
success_rate ≈ 0 ≪ 0.2 every generation → σ ×0.82 → monotonic collapse to
`SIGMA_MIN`. Once floored, children are near-identical and it can never recover.

v7 addresses all of this and scales the flagship poetry run 4×.

## Summary of changes

1. **Learned, context-aware embedding** replaces frozen UMAP. A small NN,
   shared globally across all poetry worms/flasks, co-evolved by the same NES.
2. **Memory folded into the embedding.** The decaying-residual afterglow is
   removed; the worm's last-5 eaten words are inputs to the embedding, so the
   same on-screen word tastes different depending on what was recently eaten.
3. **Learned POS syntax signal** as the 12th chemosensory dimension (also a
   shared, co-evolved net).
4. **σ-collapse fix** — parent-probe reference for the 1/5 rule.
5. **Deterministic repetition penalty** in fitness.
6. **8 poetry flasks across 4 cores** (was 2 flasks on 1 core), with a
   cross-process coordinator for the shared nets.
7. **All experiments restart** (8 poetry + 4 sanity) on the new embedding; v6
   run history is archived intact.

## 1. The learned embedding (shared / global)

### Inputs (per candidate on-screen word, per worm, per tick)
- The candidate word's frozen `nomic-embed-text-v1.5` **512-dim** vector.
- The worm's **last 5 eaten** words' 512-dim vectors (its memory; zero-padded
  when fewer than 5 have been eaten).

### Network
All weights **shared across every worm in every poetry flask** (one global
instance). Each *sanity* experiment gets its own independent copy (different
objectives can't share a gradient).

```
        last 5 EATEN words                      CURRENT on-screen word
       (nomic 512 each)                             (nomic 512)
             │ each                                      │
        ┌────▼──────┐  E: Linear(512→11)+ReLU  (SAME weights for all words)
        │     E     │◄─────────────────────────────────┐
        └────┬──────┘                                   │
        5 × [11]                                   ┌────▼──────┐
             │ concat → 55                          │     E     │
        ┌────▼───────────────┐                     └────┬──────┘
        │ Hh: 55→11 +ReLU     │  history summary          │ E(current) = 11
        └────┬───────────────┘                            │
             │ 11 ─────────────────┐          concat 11+11 = 22
                                    ▼          ┌───────────────────────┐
                                               │ Hf: 22→11 → sigmoid    │
                                               └───────────┬───────────┘
                                                           ▼
                                          11-dim chemosensory drive → PC0–PC10
```

- `E`  : Linear(512→11) + ReLU     — 512·11 + 11 = **5,643** params
- `Hh` : Linear(55→11) + ReLU      — 55·11 + 11 = **616** params
- `Hf` : Linear(22→11) → sigmoid   — 22·11 + 11 = **253** params
- Embedding subtotal: **6,512** params

Sigmoid on `Hf` keeps each dim in `[0,1]`, matching what
`compute_pca_activation` expects (it treated the UMAP 12-vector as `[0,1]`).

### POS syntax net (shared / global; separate from the embedding)
Gives the worm a learned "does this next word fit, grammatically, given what I
just ate" signal — the sequential memory the substrate most lacks.

- Input: one-hot POS of the current word ⊕ one-hot POS of each of the last 5
  eaten words. Canonical universal tagset, **T = 12**
  (`NOUN VERB ADJ ADV DET ADP PRON PRT CONJ NUM X .`). Input width 6·12 = **72**
  (zero-pad missing history).
- `P1`: Linear(72→16) + ReLU
- `P2`: Linear(16→1) → sigmoid → **PC11** (the AFD pair, formerly thermosensory)
- Params: 72·16+16 + 16·1+1 = **1,185**

**Total shared genome ≈ 7,697 params**, evolved by pooled NES over all 128
poetry worms/gen (vs. the original 33.5k-param proposal over 16 worms — the
global sharing + smaller net is the noise reduction the design targets).

### Chemosensory assembly
The 12-vector `[emb₀…emb₁₀, pos₁₁]` feeds `compute_pca_activation` exactly as
the UMAP 12-vector did (per-neuron L/R via `direction_factor`, spatial
`intensity`). **No residual term** — see §2.

## 2. Removing the residual

Delete `sim/world.py`'s decaying-afterglow machinery (`RESIDUAL_TAU_BRAIN_TICKS`,
the `_residual_*` buffers, `compute_residual_chemo`, residual viewer payloads).
Each tick, chemosensory state is computed purely from in-range on-screen words,
each embedded with the worm's **current eaten-history** as context. Memory of
eaten words now lives entirely in the history inputs to `E`/`Hh` and the POS net.
The per-worm last-5-eaten buffer (words + their 512 vectors + their POS tags) is
maintained on eat.

## 3. Evolution engine changes

### (a) σ-collapse fix — parent probe
Replace "beat previous-gen best (max)" with the textbook Rechenberg reference:
`success_rate = fraction of fresh children whose fitness > the PARENT's`. Get the
parent's fitness by evaluating the **unperturbed parent centroid** each
generation under the *same* judge sampling seed as the children (a "parent
probe"). σ then adapts by the existing 1/5 rule. Applies to per-flask brain σ
**and** the global shared-net σ. This lets σ actually grow again.

### (b) Deterministic repetition penalty
In fitness aggregation, down-weight windows dominated by token repetition so
pure repetition can't score high. Diversity factor over the scored window,
e.g. `f = (unique_tokens / total_tokens)` (or an n-gram repeat rate), multiplied
into the window's contribution. Deterministic, free. Test: a repetitive window
scores strictly less than a diverse window with identical judge E/C.

### (c) Global shared-net NES + coordinator
The shared embedding+POS nets evolve from the **pooled** contributions of all
128 poetry worms:
- Each spawned worm carries its own perturbation of the global net
  `θ_g + σ_g·eps_g` (persisted eps_g) alongside its per-flask brain
  perturbation. Its single fitness is attributed to both.
- At each generation boundary each process writes its worms'
  `(eps_g, fitness)` to `data/poetry_shared/gen-NNNN/contrib_<flask>.json`
  (atomic tmp+rename). A **leader** waits for all 8 flasks (with a
  timeout/quorum — proceed with whoever reported; never freeze), computes the
  pooled NES update on `θ_g`, adapts `σ_g` via a pooled parent-probe 1/5, writes
  `θ_g` for the next gen, and bumps the shared generation counter. Every process
  reads the new `θ_g` before spawning the next generation.
- **Determinism**: contributions processed in fixed `(flask, worm)` order; eps
  from seeded RNGs.

Per-flask **brains** keep evolving independently (no cross-process barrier).

## 4. Orchestration — 8 poetry flasks on 4 cores

- Python's GIL ⇒ one process ≈ one core for the sim loop, so 4 poetry
  processes (systemd `wormlet-poetry-1..4`), one per core, each
  `WORMLET_N_FLASKS=2`, `WORMLET_N_WORMS_PER_FLASK=16`, ports 8000/8010/8020/8030.
- Global flask identity `flask_1..flask_8` (process *p* owns `flask_{2p-1}`,
  `flask_{2p}`); unique seeds across all.
- Shared coordination dir `data/poetry_shared/`.
- Viewer dropdown: 8 poetry + 4 sanity entries. Reuses the existing
  experiment-dropdown mechanism (`server/experiments.py` + `all_for_dropdown`).
- **Sanity experiments** restart on the new embedding, each with its OWN
  shared-net copy (single flask/process, no coordinator). Their deterministic
  POS scorers are unchanged and now also drive their embedding's evolution.
- Follows the proven per-core independent-process pattern already used for the
  4 sanity experiments (`deploy/wormlet-*.service`).

## 5. Prerequisite — 512-dim nomic cache

`scripts/build_corpus_nomic512.py`: run `nomic-embed-text-v1.5` once over the
4,563 Hamlet word types → `cache/corpus_nomic512.json`
(`{model, prefix, n_words, n_dims:512, words:[...], nomic512:[[...]]}`). Runtime
loads it (numpy) and applies the current genome's embedding net; forward-pass
outputs are cached per `(word, history)` within a generation (weights are fixed
during a generation). `torch`+`transformers` are available in `/home/web/.venv`
(build-time only; runtime needs only numpy).

## 6. Archive & restart

- Tag v6 HEAD `v6-epoch-0103-archive`; v6's `data/generations` (712 MB) stays
  in-repo as the archive.
- v7 cold-starts fresh. Cutover: repoint systemd + cloudflared at v7 **after**
  local verification. (NB: v6 currently has duplicate `wormlet-app.service` and
  `wormlet.service` both bound to port 8000 — reconcile during cutover.)

## 7. Tests
- `tests/test_embedding_net.py` — forward-pass shapes, determinism, sigmoid
  range, history zero-padding, flatten/unflatten round-trip.
- `tests/test_evolution.py` — parent-probe success_rate; σ can grow; repetition
  penalty monotonicity.
- `tests/test_coordinator.py` — pooled-update determinism; timeout/quorum
  proceeds on partial input.
- `tests/test_determinism.py` — same-seed-same-trajectory preserved.

## Implementation status (2026-07-08)

**Done + tested (all green; determinism preserved):**
- `server/embedding.py` — learned context-aware embedding + POS net, genome
  (de)serialize, per-generation cache. `tests/test_embedding_net.py` (7/7).
- `server/evolution.py` — deterministic `repetition_factor` folded into
  `fitness()`; σ-collapse fix (population-baseline 1/5 rule via
  `parent_fitness`/`fresh_mean`, replacing the "beat the champion" bug).
  `tests/test_evolution.py` (6/6).
- `server/generations.py` — both rollover paths pass `parent_fitness=
  state.prev_fresh_mean` and persist `ng.fresh_mean`.
- `sim/world.py` — chemosensation now uses the learned embedding with the
  worm's last-`HISTORY` eaten words as context; **residual removed**.
  `tests/test_determinism.py`, `test_generations.py`, `test_rollover_wiring.py`
  still green.
- `server/shared_evolution.py` — pooled NES for the shared net +
  `SharedNetCoordinator` (filesystem barrier, timeout/quorum, atomic writes,
  deterministic ordering). `tests/test_coordinator.py` (6/6).
- `scripts/build_corpus_nomic512.py` + `cache/corpus_nomic512.json`
  (4,919 words × 512-dim, Matryoshka-truncated from 768 + renormalized).

**Note on dims:** nomic-embed-text-v1.5 emits 768-dim; we Matryoshka-truncate
to 512 (a first-class feature of this model) to match the design. `D_IN=512`.

## Remaining integration (not yet wired into the live server)

The algorithmic core above is complete and unit-tested. What remains is
threading it through the running server + orchestration + deploy. Each worm
must SENSE with its own perturbation of the shared genome for the pooled NES
to have signal:

1. **Per-worm embedding perturbation.**
   - `Worm` gains an `embedding_genome: np.ndarray` (+ its `eps` vs the shared
     parent). `World` builds a per-worm `EmbeddingModel` from it (falling back
     to `embedding.get_model()` when None, as today).
   - `orchestrator.load_flasks` / `_ensure_flask_worm_dir` load/create a
     per-worm `embedding.json` (genome + eps).
2. **Rollover wiring** (`app.py:_run_all_flask_rollovers_sync` +
   `generations.run_generation_rollover`):
   - After scoring, gather each worm's `(embedding_eps, fitness)`; call
     `SharedNetCoordinator.contribute(gen, flask, eps, scores)`.
   - `barrier_update(gen)` → new shared θ_g/σ_g; then respawn every worm with a
     fresh `θ_g + σ_g·eps` (new eps per worm), writing `embedding.json`.
   - Sanity experiments use a per-experiment `SharedNetState` (single flask, no
     coordinator) — verifiable headlessly (deterministic scorer, no API).
3. **8 poetry flasks / 4 processes.** `deploy/wormlet-poetry-{1..4}.service`
   (ports 8000/8010/8020/8030, `N_FLASKS=2`, `N_WORMS_PER_FLASK=16`, shared
   `WORMLET_POETRY_SHARED_DIR=data/poetry_shared`, global flask ids
   `flask_1..8`, unique seeds). Extend `experiments.py` EXPERIMENTS + the
   viewer dropdown with the 8 poetry entries.
4. **Archive + cutover.** Tag v6 HEAD `v6-epoch-0103-archive`; point systemd +
   cloudflared at v7; reconcile the duplicate `wormlet-app.service` /
   `wormlet.service` (both bound to :8000). **Do end-to-end headless
   verification (a few sanity-path generations, confirm σ breathes and the
   shared genome drifts) BEFORE flipping the public site.**

Recommended completion order: 1 → verify headless (sanity path) → 2 → 3 → 4.

## Risks / non-goals
- **Coordinator** is the main new risk → timeout/quorum + atomic writes + tests.
- **CPU** — 128 worms / 4 processes; past freezes wiped generations. Carry v6's
  `sim_loop` yield fix; watch with `py-spy` on first boot; be ready to dial
  worms/flask down.
- Non-goals: changing the judge model or the connectome topology.
