# Setting up HamletRNAWorld (v7) on your own server

From a clean machine to a running instance. Every command here was run and
verified on 2026-08-13 against a fresh `git clone` on Ubuntu 24.04 /
Python 3.12.

Working with Claude Code? Point it at this repo and it will read `CLAUDE.md`
at the root, which carries the project's hard rules and current state. This
file is the runbook; `docs/ARCHITECTURE.md` explains how the simulation works.

---

## 0. What you're installing

A FastAPI server that ticks a population of simulated *C. elegans* worms at
~60 Hz, streams them to a browser over WebSocket, and — if you enable
generations — periodically asks a Claude model to judge the poems the worms
have eaten, then evolves their connectomes.

Two modes, and you should start with the first:

| | LLM judge | cost | use it for |
|---|---|---|---|
| **Viewer only** (`WORMLET_GENERATIONS_ENABLED=0`) | no | free | seeing it work, development |
| **Generations** (`=1`) | yes, per rollover | Anthropic API usage | actual evolution runs |

You do **not** need an API key for viewer-only mode.

---

## 1. Requirements

- **Python 3.10+** (3.12 is what this runs on)
- **~1 GB disk** for the repo (the frozen embedding caches are ~28 MB, git
  history is larger)
- **~400 MB RAM per process**, one CPU core per process
- An **Anthropic API key** — only for generations mode

## 2. Clone and create a venv

```bash
git clone https://github.com/YitongTseo/HamletRNAWorld.git
cd HamletRNAWorld/v7

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e .
```

`pip install -e .` reads `pyproject.toml`: fastapi, uvicorn[standard],
anthropic, numpy, nltk.

> **Only ever work inside `v7/`.** `v1`..`v6` are frozen historical snapshots.
> v6 is kept runnable as the comparison baseline; don't modify it.

## 3. Download the NLTK tagger data

The POS grammar channel needs NLTK's perceptron tagger. The code looks in
`v7/.venv/nltk_data` first, so put it there — that keeps it project-local and
out of your home directory:

```bash
./.venv/bin/python -c "
import nltk
for p in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','universal_tagset'):
    nltk.download(p, download_dir='.venv/nltk_data')
"
```

Verify:

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from server import pos_grammar as G
g = G.get_grammar()
print('lexicon:', len(g._lexicon))
print([(w, g.tag(w)) for w in ('king','speak','the','and')])
print(sorted(g.context_distribution(['the']).items(), key=lambda kv: -kv[1])[:3])
"
```

Expected, exactly:

```
lexicon: 5148
[('king', 'NOUN'), ('speak', 'VERB'), ('the', 'DET'), ('and', 'CONJ')]
[('NOUN', 1.0), ('VERB', 0.3497623099328682), ('ADJ', 0.29076486650632044)]
```

If tags come back `X`, the tagger data isn't being found.

> **Different NLTK versions tag differently.** In isolation one version gives
> `speak` → VERB and another gives NOUN. That does **not** affect the
> simulation, because the PC11 grammar channel reads the *committed* lexicon in
> `cache/pos_transitions.json` (built by tagging Hamlet's lines in context), and
> only falls back to the live tagger for words absent from the play. The numbers
> above are therefore identical on every install — which is why you should
> **not** rerun `python -m server.pos_grammar` unless you intend to rebuild the
> lexicon, since doing so on a different NLTK version will change it.

## 4. What you do NOT need to build

The expensive artifacts are committed, so there is nothing to precompute:

| `cache/` file | what it is |
|---|---|
| `corpus_umap.json` | 12-dim UMAP over the Hamlet vocabulary — **the live chemosensory encoder** |
| `corpus_nomic512.json` | 4919 × 512 frozen nomic embeddings (25 MB); only used by `WORMLET_ENCODER=learned` |
| `pos_transitions.json` | POS trigram table + in-context lexicon for the PC11 grammar channel |
| `corpus_pca.json` | legacy PCA encoder (v6.1 lineage) |
| `neuron_body_coords.json` | neuron positions for the viewer |

The scripts that regenerate them live in `scripts/` and need extra
dependencies (`umap-learn`, a local embedding model). You only need those if
you change the corpus.

`pos_transitions.json` can be rebuilt with no extra dependencies —
`./.venv/bin/python -m server.pos_grammar` — but **don't**, unless you mean
to. It is committed precisely so the chemosensory input is identical on every
install; rebuilding it under a different NLTK version silently changes the
lexicon and therefore what every worm tastes.

## 5. Run the tests

```bash
./.venv/bin/python tests/run_all.py
# expect: 116/116 passed
```

**Use `run_all.py`, not pytest** — pytest isn't a dependency, and several test
modules have no `__main__` block, so running them directly executes nothing
and exits 0 (silently "passing"). `run_all.py` imports each module and calls
every `test_*` function. It takes a substring filter:
`./.venv/bin/python tests/run_all.py evolution`.

## 6. Run it (viewer only)

```bash
WORMLET_GENERATIONS_ENABLED=0 WORMLET_N_WORMS=6 \
  ./.venv/bin/python main.py --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. You should see worms crawling over scrolling
Hamlet text and eating words. Check `http://127.0.0.1:8000/healthz`:

```json
{"tick": 1287, "embedding": "umap", "pos_channel": "corpus_grammar",
 "generations_enabled": false, "n_worms_total": 6}
```

`embedding` reports the **live** encoder — trust it over your assumptions
about which code is loaded.

Pages: `/` overview · `/focus` single worm with neuron activity ·
`/poems` live poems · `/generations` evolution dashboard (needs rollovers).

## 7. Enable evolution (costs API money)

Put your key in a secrets file:

```bash
cat > ~/.wormlet.env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
WORMLET_DEBUG_SECRET=some-random-string
EOF
chmod 600 ~/.wormlet.env
```

> **Keep this file secrets-only.** systemd's `EnvironmentFile=` *shadows*
> `Environment=`, so any operational variable in here silently overrides every
> unit file. That has bitten this project before — a leftover worm count in
> `.wormlet.env` quietly loaded a completely different configuration than the
> units specified.

```bash
set -a; . ~/.wormlet.env; set +a
WORMLET_GENERATIONS_ENABLED=1 WORMLET_N_FLASKS=1 WORMLET_N_WORMS_PER_FLASK=8 \
WORMLET_PASSAGE=opening WORMLET_DATA_DIR=$PWD/data/dev WORMLET_GIT_COMMIT=0 \
  ./.venv/bin/python main.py --host 127.0.0.1 --port 8000
```

`WORMLET_GIT_COMMIT=0` matters: it defaults to `1`, so without it the rollover
tries to commit generation artifacts into your working tree. See §8.

A rollover fires when the worms finish the passage. `opening` is short —
minutes. `full` is the whole play and takes hours per generation, which is
what production uses. Artifacts land in
`data/dev/generations/flask_1/gen-NNNN/`.

## 8. Environment variables

Defaults below are read out of the code, not from memory — check them again if
you change versions.

| variable | default | meaning |
|---|---|---|
| `WORMLET_DATA_DIR` | `<v7>/data` | **per-process** data root. Give each process its own. |
| `WORMLET_GENERATIONS_ENABLED` | `0` | `1` turns on evolution + the LLM judge |
| `WORMLET_ENCODER` | `umap` | `umap` (frozen, default) or `learned` (v7 co-evolved net) |
| `WORMLET_N_FLASKS` | `4` (`1` in experiment mode) | independent populations in this process |
| `WORMLET_N_WORMS_PER_FLASK` | `10` | worms per flask (production sets 16) |
| `WORMLET_N_WORMS` | `0` → orchestrator default | worm count in legacy/no-generations mode; `main.py --n-worms` sets it |
| `WORMLET_PASSAGE` | `opening` | `opening`, `act1`, or `full` |
| `WORMLET_SIGMA_SCHEME` | `vs_mean` | σ controller: `vs_mean`, `vs_elite`, `xnes`, `sigma_anneal` |
| `WORMLET_EMBEDDING_SEED` | `0` | seeds each flask's embedder lineage (learned mode) |
| `WORMLET_GIT_COMMIT` | **`1`** | commit generation artifacts. **Set this to `0`** — see below |
| `WORMLET_GARDENER` | `1` | the per-epoch LLM commentary pass |
| `WORMLET_VIEW_MODE` | `poetry` | which dropdown entry this process highlights |
| `WORMLET_EXPERIMENT_MODE` | unset | POS-scorer sanity runs (`words`, `nouns`, `pos_chain`, `semantic`) — no LLM, free |
| `WORMLET_CHECKPOINT_INTERVAL_S` | `60` | worm-state checkpoint cadence |
| `WORMLET_BOARD_PUBLISH_DIR` | `/home/web/board_publish` | where generation winners are published |
| `ANTHROPIC_API_KEY` | unset | required only in generations mode |

### Set `WORMLET_GIT_COMMIT=0`

**It defaults to `1`, which is not what you want.** This repo is now
**code-only**: `data/` is gitignored because four processes racing to push
per-generation JSON to one branch failed non-fast-forward and bloated history
with ~19M lines. Every unit file in `deploy/` sets it to `0` explicitly, and
experiment mode forces it to `0` — but a bare `python main.py` with
generations enabled will try to commit. If you want artifacts off the box, use
the board-publish path (`server/board_publish.py`) or your own backup.

## 9. Production: multiple processes

Production runs **4 processes × 2 flasks × 16 worms = 8 flasks / 128 worms**,
one CPU core each. Unit files are in `deploy/`. To adapt them, change
`User`/`WorkingDirectory`/`ExecStart` paths and give each process a distinct
`WORMLET_DATA_DIR`, port, `CPUAffinity`, and `WORMLET_EMBEDDING_SEED`.

```bash
sudo cp deploy/wormlet-poetry-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wormlet-poetry-1     # canary ONE first
curl -s localhost:8000/healthz | python3 -m json.tool
sudo systemctl enable --now wormlet-poetry-{2,3,4}
```

Scale to your box: 4 processes need 4 cores and ~1.6 GB. On a smaller machine
run one process with more flasks, or fewer worms per flask.

**Health checks:** `deploy/wormlet-healthcheck{,.service,.timer}` probe
`/healthz` for every wormlet unit, skip inactive ones, and restart only a
wedged process (15-minute threshold; `/healthz` is rollover-safe).

```bash
sudo cp deploy/wormlet-healthcheck /usr/local/bin/
sudo cp deploy/wormlet-healthcheck.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now wormlet-healthcheck.timer
```

Because each process has its own data dir, the `/generations` viewer reads its
**sibling** processes' dirs read-only so all 8 flasks appear in the dropdown
from any port. That only kicks in when the data dirs are named `poetry-N`
under a shared parent.

## 10. Public access (optional)

Production fronts this with a Cloudflare tunnel — no open inbound ports.
`deploy/cloudflared.yml.template` has the ingress map (one hostname per
process). `deploy/setup_cloudflared.sh` is the installer. Any reverse proxy
works; the app binds `127.0.0.1` by design.

---

## Troubleshooting

**Worms don't move / eat nothing.** Check `/healthz` shows
`"embedding": "umap"`. If chemosensation is silent, `cache/corpus_umap.json`
is missing or unreadable — the model degrades to an empty map rather than
crashing. Note a *single* worm eating nothing over a short run is normal
variance, not a bug; judge it across the population.

**`tag_word` returns `X` for everything.** NLTK data isn't installed where the
code looks — redo step 3.

**Tests "pass" but print nothing.** You ran a module directly instead of
`tests/run_all.py`. See step 5.

**Rollovers never fire.** Generations only roll over when the worms finish the
passage. With `WORMLET_PASSAGE=full` that's hours. Use `opening` to iterate.

**Judge calls 400.** Anthropic structured-output schemas reject `maxItems`,
`minItems`, `minimum`, `maximum`, `minLength`. Enforce bounds in the prompt and
client-side instead.

**A unit ignores its `Environment=` settings.** Something operational is in
`~/.wormlet.env`. `EnvironmentFile=` shadows `Environment=`. Keep it
secrets-only.

**Fresh clone has a `data/` directory.** Some pre-`code-only` generation
artifacts are still tracked. Harmless. Don't `git rm` them in the upstream
repo — see `CLAUDE.md`.
