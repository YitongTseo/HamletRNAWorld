# Two UIs behind one switch, and the lifelike relaunch

**Date:** 2026-09-01
**Status:** implemented and deployed

## Why

Five merged PRs brought two separable things: worm/algorithm work we want
(within-lifetime plasticity, hunger/satiety, habituation, quality-per-window
fitness, pluggable judge backends, three new corpora) and a complete visual
restyle ("Tobacco & Ochre") that we do not want as the default.

The decision: keep the algorithms, keep the old look as default, keep the
restyle available and unmodified, and take the one part of the restyle that
earns its place — the **desire layer**, which haloes the words the worm
currently wants.

## The UI seam

Two complete viewer trees, neither aware of the other:

| tree | variant | what it is |
|---|---|---|
| `viewer/` | `classic` | black/green, thumbnail-card overview, `ui-monospace` |
| `viewer_vivarium/` | `vivarium` | upstream's restyle, byte-identical to `origin/main` |

Selection, in priority order:

1. `?ui=classic` / `?ui=vivarium` on any page — per-request, no restart
2. `WORMLET_UI` — the process default, set in a systemd drop-in
3. `classic` — fallback, including for an unrecognised value

All of it lives in `server/ui_variant.py`. `app.py`'s six page routes call
`ui_variant.page(request, "<file>.html")`; `/static` is replaced by
`ui_variant.mount_all(app)`.

### Why two trees rather than a theme

The front-ends differ *structurally*, not just chromatically: the overview is
a grid of thumbnail cards in one and a tray of petri dishes in the other, and
the focus page grows a specimen card. No stylesheet swap expresses that. A
single tree carrying both shapes behind `if (VARIANT === …)` would make every
future edit a two-branch edit inside one file, and would guarantee that
`viewer_vivarium/` drifts from upstream and stops merging cleanly.

The cost is duplication. 19 of 26 files genuinely differ; **6 are identical
and must be edited in both places**: `poems.js`, `focus/dock.js`,
`focus/state.js`, `focus/magnifier.js`, `focus/panel-chrome.js`,
`focus/responsive.js`. If that set grows, promote it to `viewer_common/`
mounted at `/static/common/` rather than letting the trees drift apart.

### How `/static` resolves

Every `/static/…` reference in either tree lives **only in the five HTML
files** — there are none in any `.js` or `.css`. So `page()` rewrites
`/static/` to `/static/<variant>/` as it serves the HTML, and both trees mount
side by side. Neither tree contains a single variant-aware line.

`tests/test_ui_variant.py::test_no_static_refs_outside_html` enforces this. If
a `.js` file ever hardcodes `/static/…`, `page()` cannot reach it and that
asset silently loads from the *default* variant — invisible in classic,
broken only under `?ui=vivarium`.

Freshness is already handled by `app.disable_cdn_cache`, which sets
`no-cache, no-store` on everything under `/static/`. Do not add a second
caching layer.

## The desire layer

Ported into classic from the vivarium tree, with colours read from the palette
CSS vars (`--accent`, `--fg`) instead of upstream's hardcoded ochre — the file
is shared by four experiment modes, each with its own accent.

- additive radial glows under words, radius and alpha scaling with pull
- per-word tint: white blended toward the accent as pull rises
- a breathing ring on the single strongest pull
- `d` toggles the layer; smell lines now default off (`o` restores them)
- **not** ported: upstream's press-and-hold gating of the PCA popup

"Pull" is the summed chemosensory activation the word currently produces in
the worm's nose — the sim's own numbers, not a client-side recomputation. It
is salience, not steering: the turn decision uses the L/R split.

### Bug found and fixed: the tint never worked

Upstream joined smells to the food snapshot on rounded coordinates, commented
*"snapshot food and smells round identically"*. They do not.
`World._compute_smells` runs on **brain ticks (2 Hz)** while the food snapshot
is taken **every frame (60 Hz)**, so the scroller has already moved every word
by the time both reach the client.

Measured against a live production frame: a constant **5.25 px** drift in y,
and **0 of 4** smells joined. Small enough to read as a rounding quirk, fatal
to an equality join, and silent — the fallback is "draw the word normally", so
it looks like the feature is simply subtle.

`sensed_smells` is already keyed by `f"{line_id}_{word_idx}"`, the exact key
`wordFoodMap` uses. The fix was to send it: smells now carry `line_id` and
`word_idx` at both serialisation sites (`sim/world.py`, `server/app.py`), and
both viewers join on identity. The heatmap and halo also now position from the
live food snapshot rather than the stale smell coords, removing the same 5 px
offset from the glow itself.

Re-verified on the fixed server: 1/1 join, 0.00 px drift, tint firing.
`test_desire_layer_joins_on_identity_in_both_trees` fails if either tree
reintroduces a coordinate join.

## The relaunch (2026-09-01)

All 8 flasks were at **generation 137** and stayed there — every process
resumes from its own `state.json`. Backups: `state.json.bak-prelifelike-*`.

| unit | port | lifelike | σ scheme | role |
|---|---|---|---|---|
| `wormlet-poetry-1` | 8000 | plasticity + hunger + habituation | `vs_elite` | |
| `wormlet-poetry-2` | 8010 | plasticity + hunger + habituation | `vs_elite` | |
| `wormlet-poetry-3` | 8020 | plasticity + hunger + habituation | `vs_elite` | |
| `wormlet-poetry-4` | 8030 | **none — stock** | `sigma_anneal` | control arm |

Drop-ins: `ui.conf` (all four, `WORMLET_UI=classic`) and `lifelike.conf`
(1–3 only). **Do not add `lifelike.conf` to poetry-4** without recording why —
the comparison is the point.

`WORMLET_FLASK_TEXTS` is deliberately unset, so every flask stays on Hamlet.
The Laozi, Beowulf and Daodejing corpora ship but stay dark.

### Two caveats that are live, not hypothetical

**The lifelike genes are frozen for these lineages.** `parent_keys` is fixed at
cold start, so a lineage predating the feature carries no `_lifelike` gene
block and runs the clipped defaults forever. This run tests whether the
*behaviour* helps, not whether evolution can tune it. Only a lineage
cold-started with the flags on searches the learning rules themselves.

**Hunger means worms can starve to death.** At deploy all 32 worms on poetry-1
sat at satiety 0.89–1.00 with zero deaths, but `deaths.jsonl` per flask is now
a thing to watch.

### Verified at deploy

- gen 137 preserved on all 8 flasks; encoder `umap`; 16 worms/flask
- lifelike keys live on 1–3 (`satiety`, `plasticity_delta`, `habituation`),
  absent on 4
- `plasticity_capped: 0` — not saturated (see CLAUDE.md on what saturation
  would mean)
- tick rate 46 Hz (lifelike) vs 57 Hz (control) — lifelike costs ~19%
- load 2.7/4 cores, 1.96 GB / 7.8 GB
- `www`, `poetry-2/3/4` subdomains all 200; no tracebacks
- 182/182 tests green
