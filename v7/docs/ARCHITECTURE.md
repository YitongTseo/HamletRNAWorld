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
Hamlet corpus ──► UMAP-12 over the vocabulary (FROZEN, cache/corpus_umap.json)
                        │
   on-screen word ──────┤        worm's last-2 EATEN words ──────┐
                        ▼                                        ▼
        ┌───────────────────────────┐        ┌──────────────────────────────┐
        │ PC0–PC10: frozen UMAP     │        │ PC11: POS grammar            │
        │ coordinates, dims 0..10   │        │ P(tag_cur | tag_prev1,prev2) │
        │ (no learning, no context) │        │ from Hamlet, max-normalised  │
        └───────────────────────────┘        └──────────────────────────────┘
                        │ 12-dim vector in [0,1], per in-range word
                        ▼
      compute_pca_activation → per-neuron L/R firing (spatial + direction)
                        ▼
      connectome (sim/connectome.py) → muscle body (sim/muscle_body.py) → motion
                        ▼
      eats words → poem → (per generation) Claude judge → fitness → NES update
```

**Only the connectome evolves.** The chemosensory encoder is entirely frozen:
UMAP coordinates for meaning, a corpus-derived POS trigram for syntax. Neither
is learned.

Key differences from v6: the eaten-word **residual afterglow is gone** — the
worm's memory now lives in the POS channel's dependence on the last two eaten
words; **PC11 is a grammatical-fit signal** rather than v6's 12th UMAP
dimension; and there is **no repetition penalty** in fitness.

### The learned encoder (v7.0–v7.1), and why it's off

v7 originally replaced UMAP with a small net co-evolved by the same NES:
`E:512→11` per word, `Hh:55→11` history summary, `Hf:22→11→sigmoid`, over
frozen 512-dim nomic vectors. It is still in the tree and still tested, behind
`WORMLET_ENCODER=learned`. It is **off by default** because measurement showed
it had gone nearly blind:

- per-channel std across 4000 words was **0.027 over the range 0.40–0.55**,
  against UMAP's **0.131 over the full 0–1**;
- `compute_pca_activation` feeds the channel value straight to neuron firing
  with no renormalisation, so word identity was ~4% of the drive and the rest a
  constant offset — the worms could barely smell one word from another;
- two causes: He init assumes unit-variance inputs but nomic-512 is
  L2-normalised (per-component std 0.044, a 23× deficit), and different nomic
  word vectors have mean pairwise cosine **0.700** with the shared mean vector
  at 84% of a typical norm, so a random linear projection preserves exactly the
  common component that UMAP exists to strip;
- the per-flask (1+1)-ES that was supposed to learn out of this accepted **2–7
  mutations in 101 generations**, and zero in the last 30 for six of eight
  flasks.

See `server/embedding.py` for the full write-up.

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
    `nes_update` (natural gradient + trust region), `adapt_sigma`
    (Rechenberg 1/5), `evolve_generation`.
  - `sigma_controllers.py` — pluggable σ schemes (`vs_mean`, `vs_elite`,
    `xnes`, `sigma_anneal`) selected by `WORMLET_SIGMA_SCHEME`.
  - `embedding.py` — the chemosensory encoder. Dispatches on
    `WORMLET_ENCODER`: frozen UMAP (default) or the v7 learned net.
  - `pos_grammar.py` — the PC11 grammar channel: interpolated POS trigram over
    Hamlet's dialogue plus an in-context lexicon, cached in
    `cache/pos_transitions.json`.
  - `flask_embedder.py` — per-flask (1+1)-ES for the learned encoder. Inert
    unless `WORMLET_ENCODER=learned`.
  - `board_publish.py` — publishes each generation's winner outside the repo,
    served at `/board` (ESP32 boards read it).
  - `judge.py` — Claude judge (`ScoredWindow` E/C per window). Needs
    `ANTHROPIC_API_KEY`. `JUDGE_TEMPERATURE=0` — leaving it at the API default
    of 1.0 was the single biggest noise source (Kendall τ 0.31 → 0.60).
  - `gardener.py` — every N gens, a Claude "gardener" writes prose commentary to
    `data/generations/meta/gen-NNNN/gardeners_log.md`. Read these to see what the
    run is actually doing.
  - `experiments.py` — the `EXPERIMENTS` registry powering the viewer dropdown
    (8 poetry flasks + 4 POS sanity checks); per-mode ports/subdomains/scorers.
  - `pos_scorers.py` — deterministic NLTK POS fitness for the sanity
    experiments, and the OOV fallback tagger for `pos_grammar`. Uses
    `.venv/nltk_data`. Tags the **lowercased** word: it used to cache on
    `word.lower()` while tagging the original casing, and NLTK gives
    `king`→NOUN but `King`→VERB, so tags depended on which capitalisation was
    seen first.
- `sim/`
  - `world.py` — the world: word field (`TextScroller`), in-range detection,
    chemosensory assembly (v7: calls `server/embedding.py`, no residual).
  - `connectome.py` — neuron dynamics from the weight dict.
  - `muscle_body.py`, `world.py`, `worm.py` — body + integration.
  - `chemosensory_mapping.py` — `PC_NEURON_PAIRS` (canonical PC→neuron order) +
    `compute_pca_activation`.
  - `weights.json` — default connectome (≈3,689 edges = the per-worm brain genome).
- `cache/` — all committed, nothing to precompute. `corpus_umap.json` is the
  **live encoder**; `corpus_nomic512.json` (25 MB) is used only by the learned
  encoder; `pos_transitions.json` is the POS grammar table + lexicon;
  `corpus_pca.json` is the legacy v6.1 encoder.
- `data/` — **entirely gitignored** (see Gotchas). `<datadir>/flasks/<flask>/<worm>/`
  is live per-worm state; `<datadir>/generations/<flask>/gen-NNNN/<worm>/` and
  `generations/meta/gen-NNNN/` are the on-disk archival record. Each process
  points `WORMLET_DATA_DIR` at its own subtree (`data/poetry-N`).
- `deploy/` — systemd units + cloudflared template. `bin/` — setup scripts.
- `tests/` — **run via `python tests/run_all.py`**, not pytest (not a
  dependency; several modules have no `__main__` and silently do nothing when
  run directly). `test_determinism.py` (same-seed ⇒ same-trajectory — sacred),
  `test_evolution.py`, `test_embedding_net.py` (both encoder modes),
  `test_pos_grammar.py`, `test_generations_viewer.py`, `test_smoke_multi.py`.

## Evolution model

**One genome evolves: the per-worm brain** (~3,689 connectome weights), by NES
per flask, independently. The encoder is frozen. (Under
`WORMLET_ENCODER=learned` a second per-flask genome evolves by a (1+1)-ES —
that is the only case where anything else learns.)

Each generation: worms crawl a corpus pass → the judge scores a 25% sample of
15-token windows (poetry) or a deterministic POS scorer runs (sanity) → fitness
→ NES update, top-5 genomes carried verbatim as elites.

**Fitness is a SUM over judged windows, so it rewards eating volume.** Normalise
by `windows_scored` before drawing conclusions about poem quality.

The NES step is `θ' = θ + clip(lr·σ/n · Σ rw_i·eps_i)` with the step bounded to
`|Δθ| ≤ 0.5·σ·√d`. That σ factor is load-bearing: the step used to carry `1/σ`
(the plain gradient, not the natural one), so the step size and the sampling
radius moved in **opposite** directions — at σ=0.02 the parent stepped 43×
further than it had sampled, and at σ=3.0 it barely moved at all. No σ was ever
sane, and no σ controller could compensate. Now step/sampling-radius is a
constant 0.170 at every σ.

σ adapts by Rechenberg's 1/5 rule via `sigma_controllers.py`. Note that both
prior baselines were structurally wrong: v6 compared children to the champion
**max** (success ≈ 0 → σ floors), v7.0 to the population **mean** (success ≈ 0.5
→ σ ceilings). `vs_elite` compares to the current-gen incumbent.

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
- Env flags: full table in `docs/SETUP.md` §8. The encoder switch is
  **`WORMLET_ENCODER`** (`umap` default / `learned`) — an earlier draft of this
  doc called it `WORMLET_EMBEDDING`, which does not exist.
- Useful: `journalctl -u wormlet-poetry-1 -f`, `curl 127.0.0.1:8000/healthz`.
  `/healthz` reports the **live** encoder mode — trust it over assumptions
  about which code a running process has loaded. (It was a hardcoded string
  until 2026-08-13 and would happily report `learned` on a UMAP process.)

## Gotchas

- Truncated `.wormlet.env` silently drops the app into legacy single-group mode
  (no evolution) while still returning HTTP 200.
- `EnvironmentFile=` **shadows** `Environment=`, so any operational var in
  `.wormlet.env` overrides every unit file. Keep it secrets-only.
- CPU: 128 worms across 4 processes on a 4-core box is near the edge; freezes
  have wiped in-progress generations. Watch with `py-spy`.
- **Git is code-only.** `data/` is gitignored — four processes racing to push
  per-generation JSON to one branch failed non-fast-forward and bloated history
  with ~19M lines. `WORMLET_GIT_COMMIT` still defaults to `1`; the units set it
  to `0`.
- Some `data/` files are **still tracked** from before that change and exist
  only on this disk. Do **not** `git rm`/reset them, and leave purge-driven
  deletions unstaged, until an off-disk backup exists.
- Anthropic structured-output schemas reject `maxItems` / `minItems` /
  `minimum` / `maximum` / `minLength` — enforce bounds in the prompt and
  client-side, or the call 400s.
- The four processes have separate data dirs, so the `/generations` viewer
  reads its siblings' dirs read-only to show all 8 flasks. Flask ids are
  `poetry-N:flask_M`; the separator must **not** be `/` or the
  `/{flask}/{gen}` route swallows it.

## Where the science currently stands

Neither v6 nor v7 has demonstrably learned to write better poetry once fitness
is normalised by eating volume. v6's apparent climb was a `GAMMA` change
(2.5 → 1.5) at generation 13 plus increased eating; one v6 flask's spectacular
run was a repetition exploit. The 2026-08-13 changes — restoring a
high-contrast encoder, giving PC11 real grammatical content, and decoupling the
NES step from σ — remove the known mechanical reasons it *couldn't* learn.
Whether it now does is an open question, and the live populations carry 101
generations of connectome adapted to the old near-blind input, so expect a
transient before the answer means anything.
