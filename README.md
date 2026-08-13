# HamletRNAWorld

Simulated *C. elegans* worms crawl over a scrolling field of words from
**Hamlet**. A worm eats a word by touching it; the eaten words are its poem.
Each on-screen word emits a 12-channel chemosensory signal into the worm's
amphid neurons, so the worm *tastes* meaning and steers toward it. Their
300-neuron connectomes evolve under a Natural Evolution Strategy, judged for
poetic quality by a Claude model.

Live at **https://wordswordsworms.org**

## Start here

| | |
|---|---|
| **Get it running on your own server** | [`v7/docs/SETUP.md`](v7/docs/SETUP.md) |
| **How the simulation works** | [`v7/docs/ARCHITECTURE.md`](v7/docs/ARCHITECTURE.md) |
| **Working with Claude Code** | [`CLAUDE.md`](CLAUDE.md) — loaded automatically; carries the project's hard rules |

Quick local run (no API key needed):

```bash
cd v7
python3 -m venv .venv && ./.venv/bin/pip install -e .
./.venv/bin/python -c "
import nltk
for p in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','universal_tagset'):
    nltk.download(p, download_dir='.venv/nltk_data')"
WORMLET_GENERATIONS_ENABLED=0 WORMLET_N_WORMS=6 ./.venv/bin/python main.py --port 8000
```

Then open http://127.0.0.1:8000.

## Repository layout

`v1`..`v7` are evolutionary snapshots; each forks its predecessor and adds
capability. **`v7` is the active version — never modify an older one.** v6 is
kept runnable as the comparison baseline.

## Status

The chemosensory encoder is frozen (UMAP coordinates for meaning, a
corpus-derived POS trigram for grammatical fit); only the connectomes evolve.
An honest caveat, documented at length in `CLAUDE.md` and the architecture doc:
**no version of this has yet demonstrably learned to write better poetry** once
fitness is normalised by how much each worm ate. Recent work removed several
mechanical reasons it couldn't. Whether it now does is open.
