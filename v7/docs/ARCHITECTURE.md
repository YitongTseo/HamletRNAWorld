# HamletRNAWorld — Architecture (v7)

Orientation doc for anyone (human or Claude) picking this up cold. For the
*why* behind v7's changes see `docs/superpowers/specs/2026-07-08-v7-learned-embedding-design.md`.

## What this is

An art/research simulation: a population of simulated *C. elegans* worms whose
300-neuron connectomes drive a body that crawls over a scrolling field of words
from **Hamlet**. A worm "eats" a word by touching it; the eaten words become the
worm's poem. Words emit a chemosensory signal into the worm's amphid sensory
neurons, so the worm literally *tastes* meaning and steers toward it. The
connectome weights — and, new in v7, the word→taste embedding — **evolve** under
a Natural Evolution Strategy (NES) judged by a Claude model for poetic quality.

Public site: **https://wordswordsworms.org**. Not a shared codebase — a personal
project. `v1`..`v7` are evolutionary snapshots; each forks its predecessor and
adds capability. **Never modify an older version.** v7 is active.

## The pipeline (v7)

```
Hamlet corpus ──► nomic-embed-text-v1.5 (512-dim, FROZEN, cached offline)
                        │
   on-screen word ──────┤        worm's last-5 EATEN words ──────┐
                        ▼                                        ▼
              ┌───────────────────────────────────────────────────────┐
              │  LEARNED, SHARED embedding net (co-evolved by NES)      │
              │  E:512→11 (per word) → Hh:55→11 (history) →             │
              │  Hf:22→11→sigmoid  ⇒ 11-dim chemo drive (PC0–PC10)      │
              │  POS net: one-hot(cur⊕last5) 72→16→1→sigmoid ⇒ PC11     │
              └───────────────────────────────────────────────────────┘
                        │ 12-dim vector, per in-range word
                        ▼
      compute_pca_activation → per-neuron L/R firing (spatial + direction)
                        ▼
      connectome (sim/connectome.py) → muscle body (sim/muscle_body.py) → motion
                        ▼
      eats words → poem → (per generation) Claude judge → fitness → NES update
```

Key v7 differences from v6: the embedding is **learned & shared** (not frozen
UMAP); the eaten-word **residual afterglow is gone** — memory is the last-5-eaten
input to the embedding; PC11 is a learned **POS syntax** signal.

## Layout (`v7/`)

- `main.py` — uvicorn entrypoint. `python main.py --host --port`.
- `server/`
  - `app.py` — FastAPI + WebSocket + the `sim_loop` (ticks all worms, broadcasts,
    triggers rollovers). Reads `WORMLET_*` env. **Has the load-bearing
    `await asyncio.sleep(0)` yield fix — keep it** (else the viewer starves).
  - `orchestrator.py` — builds/loads worms; `load_flasks(n_flasks, n_worms_per_flask)`;
    per-worm/-flask data layout; `drain_and_persist`.
  - `worm_group.py` — a `WormGroup` = one flask's worms.
  - `generations.py` — per-generation rollover: score → NES → write artifacts →
    git commit. Holds `GenerationState` (parent vector, sigma, children eps).
  - `evolution.py` — NES math: `flatten/unflatten_weights`, `fitness`,
    `nes_update`, `adapt_sigma` (Rechenberg 1/5), `evolve_generation`.
  - `embedding.py` *(v7 new)* — the shared learned embedding + POS nets:
    numpy forward pass, param flatten/unflatten, genome (de)serialize.
  - `shared_evolution.py` *(v7 new)* — pooled NES + cross-process coordinator
    for the global nets (`data/poetry_shared/`).
  - `judge.py` — Claude judge (`ScoredWindow` E/C per window). Needs
    `ANTHROPIC_API_KEY`.
  - `gardener.py` — every N gens, a Claude "gardener" writes prose commentary to
    `data/generations/meta/gen-NNNN/gardeners_log.md`. Read these to see what the
    run is actually doing.
  - `experiments.py` — the `EXPERIMENTS` registry powering the viewer dropdown
    (8 poetry flasks + 4 POS sanity checks); per-mode ports/subdomains/scorers.
  - `pos_scorers.py` — deterministic NLTK POS fitness for the sanity experiments
    (and the POS-tag source for the v7 POS net). Uses `.venv/nltk_data`.
- `sim/`
  - `world.py` — the world: word field (`TextScroller`), in-range detection,
    chemosensory assembly (v7: calls `server/embedding.py`, no residual).
  - `connectome.py` — neuron dynamics from the weight dict.
  - `muscle_body.py`, `world.py`, `worm.py` — body + integration.
  - `chemosensory_mapping.py` — `PC_NEURON_PAIRS` (canonical PC→neuron order) +
    `compute_pca_activation`.
  - `weights.json` — default connectome (≈3,689 edges = the per-worm brain genome).
- `cache/`
  - `corpus_nomic512.json` *(v7 new)* — frozen 512-dim nomic vectors per word.
  - `corpus_umap.json` / `corpus_pca.json` — legacy 12-dim reductions (kept for
    reference / the `WORMLET_EMBEDDING` fallback).
- `data/`
  - `flasks/<flask>/<worm>/` — live per-worm state (seed, weights, poem). Ignored by git.
  - `generations/<flask>/gen-NNNN/<worm>/` + `generations/meta/gen-NNNN/` —
    tracked archival record (poems, weights, scores, gardener logs).
  - `poetry_shared/` *(v7 new)* — global shared-net parent + per-gen contributions.
  - `experiments/<mode>/` — per-sanity-experiment data roots.
- `deploy/` — systemd units + cloudflared template. `bin/` — setup scripts.
- `tests/` — `test_determinism.py` (same-seed⇒same-trajectory — sacred),
  `test_generations.py`, `test_embedding_net.py`, `test_evolution.py`,
  `test_coordinator.py`, `test_smoke_multi.py`.

## Evolution model (v7)

Two genomes evolve together:
1. **Per-worm brain** (~3,689 connectome weights) — NES per flask, independent.
2. **Global shared nets** (~7,697 embedding+POS params) — ONE instance across all
   8 poetry flasks, pooled NES over all 128 worms/gen via the coordinator. Each
   sanity experiment has its own copy.

Each generation: worms crawl a corpus pass → judge scores windows (poetry) or a
deterministic scorer runs (sanity) → **repetition penalty** applied → NES updates
both genomes. σ adapts by Rechenberg's 1/5 rule using a **parent probe** (fraction
of fresh children beating the unperturbed parent — NOT the champion; that bug
floored σ in v6).

## Runtime / ops

- Interpreter: **`/home/web/.venv/bin/python`** (numpy, fastapi, uvicorn; torch +
  transformers for the offline embedding build). `v7/.venv/` holds only
  `nltk_data`.
- Determinism is enforced by `tests/test_determinism.py`. **Never** add wallclock
  or unseeded `random.*` to sim logic.
- Runs under systemd; fronted by a named cloudflared tunnel. Env/secrets in
  `/home/web/.wormlet.env` (secrets ONLY — operational `WORMLET_*` vars go in the
  unit files, or `EnvironmentFile` shadows them). `ANTHROPIC_API_KEY` is required
  or the judge/gardener bail silently.
- Env flags: `WORMLET_GENERATIONS_ENABLED`, `WORMLET_N_FLASKS`,
  `WORMLET_N_WORMS_PER_FLASK`, `WORMLET_EXPERIMENT_MODE`, `WORMLET_EMBEDDING`
  (v7 default `learned`; `umap`/`pca` for the legacy fallback), `WORMLET_DATA_DIR`.
- Useful: `journalctl -u wormlet-poetry-1 -f`, `curl 127.0.0.1:8000/healthz`.

## Gotchas

- Truncated `.wormlet.env` silently drops the app into legacy single-group mode
  (no evolution) while still returning HTTP 200.
- CPU: 128 worms across 4 processes on a 4-core box is near the edge; freezes
  have wiped in-progress generations. Watch with `py-spy`.
- The coordinator must never hard-block on a stalled flask (timeout/quorum).
