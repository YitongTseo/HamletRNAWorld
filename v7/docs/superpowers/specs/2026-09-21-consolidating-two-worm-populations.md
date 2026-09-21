# Consolidating two worm populations

**Status:** options, not a decision. Written 2026-09-21, after the two code
lines were reconciled (PRs #8, #9, #1, #7, #10) and `main` became a truthful
description of the project for the first time since 2026-08-15.

**The goal:** jtrotsky's worms currently run on droog; the intent is to move
them onto YitongTseo's server, which is already running a different, larger
population.

---

## What is actually running, on each side

Read from the launch scripts, not from documentation — this repo's own ops
section described a fleet that no longer matched droog, which is exactly the
mistake this section exists to prevent.

| | droog (jtrotsky) | YitongTseo's host |
|---|---|---|
| lineage | `/var/wormlet/data-trio-5`, generation 188 | per this repo's CLAUDE.md ops section — **unverified, see below** |
| shape | 3 flasks x 10 worms = 30 | 4 units, 8 flasks x 16 worms = 128 |
| texts | hamlet, beowulf, daodejing (one per flask) | poetry-1..4 |
| judge | `anthropic`, Haiku, `SAMPLE_FRACTION=0.25` | anthropic |
| gardener | **off** (`WORMLET_GARDENER=0`) | on |
| lifelike | plasticity + hunger + habituation + punct smell, `COMMON_SEED=1` | 1-3 lifelike, 4 is the stock control arm |
| launch | Bastille jail, `daemon(8)`, `restart-worms.sh` | systemd, `/home/web/.venv` |
| public | worms.droog.nz (Cloudflare tunnel) | wordswordsworms.org |

**Verify the right-hand column before planning against it.** droog had already
drifted from what CLAUDE.md claimed — different flask count, different texts,
gardener disabled — and the docs did not know. Read the live systemd units and
`/healthz` on that host.

## The constraint that shapes every option

A lineage is its data directory, and **fitness histories are only comparable
within one scoring regime**. The rules this repo already enforces elsewhere:

* a different judge model or backend is a **different critic**; fitness across
  the splice is not one history (`judge_description()` is recorded per
  generation for exactly this reason);
* the rubric names the text, so a flask reading Beowulf is judged by a
  different critic than one reading Hamlet — cross-corpus fitness is never one
  comparable series;
* the fitness regime changed on 2026-08-15 (per-window mean, `window_floor`);
  values either side of that are not comparable;
* `GAMMA` and the sigma scheme shift the scale without changing the name.

So "consolidate" cannot mean "put both sets of numbers in one table". It has
to mean one of the following.

---

## Option A — Cohabit

Both lineages run on one host, in separate data dirs, under separate
processes. Nothing is merged; the move is pure logistics.

* **Preserves:** everything. Both histories stay intact and separately
  readable. Reversible at any point.
* **Costs:** one host now carries 158 worms' worth of tick load and the known
  ~44 MB/hour leak on each process; the leak's ceiling (~2.2 GB over two days)
  has to fit in the host's memory twice over.
* **Open:** whether the corpora and worm-name sets collide in the viewer, and
  whether both can publish to one board dir.

This is the only option that requires no decision about the artwork. It is
also the only one that can be done first and decided later.

## Option B — Merge the populations

Worms from both sides compete in shared flasks: one population, one selection
pressure.

* **Preserves:** a single continuing artwork, which is arguably the point.
* **Costs:** the histories end. A merged flask is a new lineage with a new
  generation counter, whatever the two parents were.
* **Required first:** one judge (model AND backend), one corpus per flask, one
  fitness regime, one sigma scheme. Any mismatch and the first generation's
  ranking is an artefact of the mismatch rather than of the worms.
* **Genome compatibility:** both sides now run the same code, but a genome's
  layout is fixed at cold start. A flask whose `parent_keys` lack the
  `_lifelike` block cannot be ranked against one that has it without the
  retrofit running first (`migrate_genome_layout`, ported to log coordinates
  2026-09-21). Check `parent_keys` on both sides before mixing.

## Option C — Retire one side

One population continues; the other is archived — data dir kept, process
stopped, poems and winners preserved as a record.

* **Preserves:** the surviving lineage's history, unambiguously.
* **Costs:** whichever is retired stops being a living thing. If it is the
  188-generation trio, that is the longer-running population; if it is the
  128-worm fleet, that is the larger one.
* **Cheapest to operate**, and honest — it does not pretend two histories are
  one.

---

## What to settle before choosing

1. **Verify the other host's live config** — units, flags, flask count, data
   dirs, judge, generation numbers. Not from this repo's docs.
2. **Whose `_lifelike` design survives long-term.** Log coordinates won
   provisionally on 2026-09-21; the argument and the alternative are recorded
   in the merge note in `sim/lifelike.py`. A consolidation is the natural
   moment to settle it, because both populations have to agree.
3. **One judge or two.** If the merged fleet keeps per-corpus rubrics, fitness
   stays per-flask and cross-flask comparison stays meaningless — which is
   fine, as long as nobody later reads a leaderboard across them.
4. **Memory headroom.** The leak is unresolved (~175 MB fresh to ~2.2 GB over
   two days, tick-driven, not explained by retained Worlds). Two populations
   on one host doubles the exposure; `wormlet_rss_bytes` in VictoriaMetrics is
   the series to read before committing.
5. **Who owns the public URL** after the move, and whether both keep one.

## Recommendation

Do Option A first regardless of the eventual answer. It is reversible, it
requires no decision about the artwork, and it puts both populations where a
merge could later happen. Treat B and C as decisions about the work rather
than about infrastructure, and make them deliberately once both sides are
running in one place and the leak's behaviour under double load is known.
