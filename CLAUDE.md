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

**Fitness is a SUM over judged windows, so it rewards eating VOLUME.** Always
normalise by `windows_scored` before concluding anything about poem quality. A
flask can gain fitness purely by eating more.

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
