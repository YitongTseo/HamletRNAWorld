# v7.1 — Per-flask shared embedder: precompute + batch + decouple

Status: **implemented autonomously** (user delegated, away, 2026-07-08)
Supersedes the "global shared embedding + cross-process coordinator" parts of
`2026-07-08-v7-learned-embedding-design.md` §3(c) and §4.

## Motivation

v7 shipped a *global* shared embedding genome co-evolved across all 8 poetry
flasks via a filesystem barrier (`SharedNetCoordinator`). Two problems the user
hit in practice:

1. **Slow.** Each worm carried its *own* perturbation of the shared genome, so
   every worm had a different `EmbeddingModel`. The 512→11 projection could not
   be shared, and `World._compute_smells` called `embed()` once per in-range
   word, recomputing the 5-word history summary on every call. Profiling (one
   worm, `full` passage) put the embedding forward-pass at **~21–24 % of
   per-tick cost**.
2. **Coupled.** The cross-process barrier made a fast process block up to
   1800 s waiting for slower ones every epoch.

## Decision (user's directive)

Revert to a **per-flask embedder**: one embedder genome per flask, **shared by
every worm in that flask** and **fixed for the whole generation**. 8 flasks →
8 independent embedders that diverge and hyper-specialize (more of the search
space explored in parallel), and flasks no longer wait on each other.

Because the embedder is identical across a flask's worms and constant within a
generation, we can:

- **Precompute** the `E: 512→11` projection over all 4,919 corpus words **once
  per generation per flask** → an `(N, 11)` table.
- Per brain-tick per worm, run only the cheap **context** (`Hh: 55→11`) and
  **combination** (`Hf: 22→11`) nets (plus the POS net), reading the E-table.
- **Batch** every in-range word for a worm into ~2 matmuls, computing the
  history summary **once** per worm per brain-tick.

Measured: **19.4× faster** on the embedding smell-pass, numerically identical
to the old per-word path (verified to 1e-9). That ~20 % per-tick saving is what
pays for going **12 → 16 worms/flask** at roughly flat generation wall-time.

## Architecture

### Embedding (`server/embedding.py`)
`EmbeddingModel` gains:
- `prime()` — builds `self._E_table` (N,11) from the current `W_E,b_E`, plus a
  process-cached, genome-independent `POS one-hot` table and word→row index.
  Called whenever params change (once per generation per flask).
- `embed_batch(current_words, history_words) -> (out (K,12), valid (K,))` —
  vectorized: history summary once, all K words through `Hf`/`P2` in single
  matmuls. `embed()` (single word) retained for tests/back-compat and now
  delegates to the table.

### World (`sim/world.py`)
- `World` takes a **shared** `embedding_model` (the flask's one model), not a
  per-worm `embedding_genome`. Falls back to the process-wide default model.
- `_compute_smells` collects in-range edible words, calls `embed_batch` **once**,
  then does the per-word geometry (distance/direction) loop.

### Per-flask embedder evolution (`server/flask_embedder.py`, new)
One embedder genome per flask, evolved by a **(1+1)-ES with Rechenberg's 1/5
rule** — no per-worm perturbation, no cross-process coordination:

- State (persisted `data/generations/<flask>/embedder.json`): `theta` (incumbent
  genome), `sigma`, `generation`, `incumbent_score`, recent success history.
- Each generation the flask senses with a **candidate** `theta + sigma·eps`
  (deterministic `eps` from flask-seed + generation). All worms in the flask
  share it (→ one precomputed table).
- At rollover: `candidate_score = mean(flask worm fitnesses)` — averaged over the
  flask's brains, so a single embedder sample is evaluated ~16×, low variance.
  If `candidate_score > incumbent_score`: accept (`theta ← candidate`,
  `incumbent_score ← candidate_score`), else keep incumbent. σ adapts by the 1/5
  rule over a success window. Draw the next candidate; rebuild the flask model.

Rationale: the precompute *requires* an identical `W_E` across a flask's worms,
which rules out per-worm gradient for the embedder; flask-granularity evolution
is the consequence. (1+1)-ES is the simplest faithful fit and, because each
candidate is scored by all ~16 worms, the accept/reject is reliable.
**Upgrade path if stronger signal is wanted:** antithetic 2-point (split the
flask, `θ±σ·eps`, gradient ∝ `(f₊−f₋)·eps`) — two tables, still cheap.

### Orchestration (`orchestrator.py`, `worm_group.py`, `app.py`)
- `WormGroup` owns its `FlaskEmbedderState` + shared `EmbeddingModel`.
- `load_flasks` builds each flask's model and passes it to every worm's World.
- **Removed:** `SharedNetCoordinator`, the `_poetry_shared_coevolve` barrier,
  per-worm `embedding.json` / `embedding_eps`, and all `WORMLET_POETRY_SHARED_*`
  env vars. Each flask's brain rollover now also steps its own embedder.
- Sanity experiments (1 flask/process) use the same per-flask embedder — their
  deterministic scorer's fitness drives the (1+1)-ES.

### Deploy
- `deploy/wormlet-poetry-{1..4}.service`: drop the coordinator env vars, set
  `WORMLET_N_WORMS_PER_FLASK=16`. Flasks stay 2/process, 4 processes, 1 core
  each (8 flasks total). No shared dir.
- Repo pushed to GitHub so wormfarm2 can `git pull` + enable the units.

## Tests
- `test_embedding_net.py`: `embed_batch` matches `embed` (incl. OOV/padding);
  table rebuild on param change; determinism.
- `test_flask_embedder.py`: (1+1) accept-on-improve, reject-on-worse, σ 1/5
  adaptation, deterministic eps, persistence round-trip.
- `test_determinism.py`, `test_generations.py`: still green (shared-model path).
- `test_coordinator.py`: removed (coordinator deleted).

## Performance work (2026-07-08, same day) — read this if you're optimizing

Three stacked optimizations landed after the per-flask embedder. Measured on
one core, `full` passage. **Every one is bit-identical / determinism-preserving**
— verified against a captured reference (sha256 of head-trace + midline + eaten
over multiple seeds), not just "looks fine".

1. **Embedding batch** (`server/embedding.py`, `sim/world.py`). `EmbeddingModel`
   precomputes the 512→11 `E`-table once per generation per flask (`prime()`);
   `embed_batch()` does every in-range word for a worm in ~2 matmuls with the
   history summary computed **once** (it had been recomputed per word).
   `World._compute_smells` distance-filters first, then one batched call.
   ~19× on the smell-pass; it's now a tiny slice of the tick.

2. **IK chain vectorized** (`sim/worm.py`). `IKChain` stores per-segment head/
   tail as parallel numpy arrays; the spring relaxation is ONE vectorized pass
   (each segment reads/writes only its own head/tail — no neighbour coupling).
   ~4× on the chain. The exact scalar op order is preserved (incl. the
   `sqrt(...) or 1e-9` exact-zero replacement and the seeded init draw order),
   so it's bit-identical.

3. **Batched per-flask body** (`sim/flask_body.py`). **Key insight: the IK chain
   is COSMETIC** — `_check_food`/`_check_walls`/`_compute_smells` all use the
   head point (`target_x/y`); only the viewer's `midline()` reads the chain. So
   a whole flask's chains relax together in ONE numpy pass over `(W, N)` arrays.
   `FlaskBody` owns those arrays and repoints each `IKChain` at a row VIEW
   (zero-copy); `WormBody.step()` only snaps the head when `batched` (defers the
   relax), and `sim_loop` calls `flask.body.relax()` once per flask per tick.
   Built at startup and rebuilt in `_respawn_flask`. ~4.5× on the chain →
   ~1.15–1.5× on flask wall-time depending on scene density. Fails safe: no
   `FlaskBody` ⇒ worms relax their own chains (correct, just unbatched).

Cumulative: per-body-tick ~0.324 → ~0.20 ms standalone; a 16-worm flask runs
generations meaningfully faster than the old 12-worm setup. **Next lever if ever
needed:** the per-tick `Food` rebuild in `World.tick` (allocates dataclasses
every tick) and the connectome brain (`sim/connectome.py`). Lowering the body
tick-rate would work too but is LOSSY (changes trajectories + viewer smoothness)
— prefer vectorization.

## Non-goals
- Judge model and connectome topology unchanged.
- Body tick-rate reduction (lossy) deliberately not used.
