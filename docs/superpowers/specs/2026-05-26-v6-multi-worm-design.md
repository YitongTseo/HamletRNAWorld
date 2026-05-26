# v6: multi-worm, headless, publicly-hosted

Status: approved 2026-05-26.

## Goal

Turn the single-worm, visualization-coupled v5 into a multi-worm headless server with a separate visualization layer, hosted publicly via Cloudflare. Each worm writes its own "poem" by eating words from its private Hamlet stream. Sims keep running whether anyone is watching or not.

## Non-goals

- Weight evolution / Hebbian / reward learning. v6 saves per-worm weights but does not mutate them. That's v7+.
- Real-time interactivity from public visitors. v6 is view-only public; mutation is debug-door only.
- Backwards compatibility with v5. v6 lives in its own directory; v5 stays untouched.

## Architecture

```
v6/
├── pyproject.toml
├── main.py                   # CLI entry: --port, --n-worms, --secret
├── worms.json                # canonical worm lineup: [{name, seed}]
├── data/                     # gitignored; created at boot
│   └── worms/<name>/
│       ├── seed.txt
│       ├── weights.json      # per-worm copy (frozen in v6)
│       └── poem.txt          # append-only, one eaten word per line
├── sim/                      # forked from v5/sim with determinism patches
├── server/
│   └── app.py                # FastAPI with multi-worm orchestrator
└── viewer/
    ├── index.html            # grid of thumbnails
    ├── focus.html            # detail view for one worm
    ├── poems.html            # side-by-side columns
    └── *.js
```

## Components

### Worm identity (`server/orchestrator.py`)

A `Worm` dataclass binds: `name`, `seed`, `world: World`, `poem_path: Path`.

On boot, the orchestrator reads `worms.json`. For each entry:
1. Ensure `data/worms/<name>/` exists.
2. If `weights.json` is missing, copy `sim/weights.json` into it.
3. Create a `World` seeded with `random.Random(seed)` and the per-worm weights.
4. Open `poem.txt` in append mode.

`worms.json` ships with 6 entries (Alice/Bob/Carol/Dave/Eve/Frank, seeds 1..6) but is fully user-editable. A `--n-worms` CLI flag overrides by truncating or extending the list.

### Deterministic sim (`sim/world.py`, `sim/connectome.py`)

Changes from v5:

- `Connectome.__init__` takes an optional `rng: random.Random`; falls back to a fresh `Random()` if absent. `rand_excite` uses `self.rng.choice` instead of `random.choice`.
- `World.step()` becomes `World.tick()` and takes **no arguments**. Internal `self.tick_count: int` increments by 1. Brain fires every `BRAIN_TICK_PERIOD = 30` ticks (=> 2 Hz at 60-tick body). Stim-linger is measured in ticks, not seconds.
- The text scroller's per-tick scroll distance is `SCROLL_SPEED / 60` (deterministic; was wallclock-dt-driven).
- No `time.monotonic()` calls inside any sim file.
- Given the same `(weights, seed, sequence-of-debug-inputs)`, two runs produce identical poem sequences. A determinism test enforces this.

### Orchestrator + broadcast (`server/app.py`)

One asyncio task ticks **all worms** at 60 Hz:

```python
async def sim_loop():
    next_t = time.monotonic()
    while True:
        for worm in worms:
            worm.world.tick()
            for word in worm.world.drain_eaten_words():
                worm.poem_file.write(word + "\n")
                worm.poem_file.flush()
        # broadcast (see below)
        next_t += 1/60
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0: await asyncio.sleep(sleep_for)
        else: next_t = time.monotonic()  # resync if we fell behind
```

`World.drain_eaten_words()` is a new method: returns the words eaten since the last drain and clears the internal buffer. The poem file is the source of truth.

Two WebSocket endpoints:

- `/ws/overview` — every 6 ticks (10 Hz), sends `{worms: [{name, head: [x,y], midline_compact, word_count, recent_words: [last 3]}, ...]}`. `midline_compact` is the midline subsampled to ~20 points to keep payloads small.
- `/ws/focus/<name>` — every tick (60 Hz), sends the full v5-style snapshot for one named worm.

HTTP endpoints:

- `GET /` → `viewer/index.html` (grid)
- `GET /focus/<name>` → `viewer/focus.html`
- `GET /poems` → `viewer/poems.html`
- `GET /api/poems` → `{name: [words...]}` for all worms (used to bootstrap the poems page)
- `GET /api/poems/<name>` → list of words for one worm
- `GET /api/graph` → static connectome topology (same as v5)
- `GET /api/embeddings` → DistilBERT cache passthrough (same as v5)

WS endpoint for poem streaming:
- `/ws/poems` — every time a word is eaten by any worm, broadcast `{worm: name, word: str, idx: int}`. Used by the poems page to push new words live.

### Debug door

A `require_debug_secret` dependency on every mutation route. The secret comes from `WORMLET_DEBUG_SECRET` env var (set in a `.env` file or shell). Clients must send `Authorization: Bearer <secret>` or a `?token=<secret>` query param (for hand-typed WS URLs).

Debug endpoints:
- `POST /debug/<name>/pause` `{paused: bool}`
- `POST /debug/<name>/reset` — re-seed the worm from its seed file, archive old poem to `poem.<timestamp>.txt`
- `POST /debug/<name>/set_head` `{x, y, facing}`
- `POST /debug/<name>/add_food` `{x, y}` (still useful for testing the food/smell pipeline)

If `WORMLET_DEBUG_SECRET` is unset at boot, the server logs a generated random secret and uses it for the session — never auto-disables auth.

### Viewer

**`/index.html` — grid:**
- Subscribes to `/ws/overview`.
- Responsive CSS grid (auto-fill, 220px min). Each cell is a `<canvas>` rendering the worm's midline + a few scrolling words, plus a small caption: name + word count + last eaten word.
- Click → navigate to `/focus/<name>`.

**`/focus.html` — detail:**
- Reads `<name>` from the URL.
- Subscribes to `/ws/focus/<name>` + fetches `/api/graph` and `/api/embeddings` (same as v5).
- Renders everything v5 renders: full midline, neurons firing, food, smells, HUD, chemosensory panel. We port v5's `viewer/main.js` essentially verbatim, just retargeted at the per-worm WS.
- A "← All worms" link in the corner.

**`/poems.html` — columns:**
- Fetches `/api/poems` once on load, then subscribes to `/ws/poems`.
- Renders N vertical columns (CSS grid), one per worm. Newest word appears at the bottom of each column; older words scroll up. Long poems use `overflow-y: auto` with auto-scroll-to-bottom.

### Hosting (Cloudflare Tunnel)

- Install `cloudflared` on the server.
- Run `cloudflared tunnel --url http://127.0.0.1:8000` in a background process; capture the printed `https://<random>.trycloudflare.com` URL.
- That URL works for both HTTP (pages) and WebSocket (Cloudflare proxies WS automatically).
- The debug door secret is **never** transmitted over plain HTTP; Cloudflare terminates TLS, so the Authorization header is encrypted in transit.
- Upgrade path: a "named tunnel" with a custom domain. Requires a Cloudflare account and either a domain we own or a Cloudflare-provided subdomain. Out of scope for v6 v1.

## Data flow

```
[ orchestrator ]
    ├── tick all worms 60 Hz
    ├── drain eaten words → append to per-worm poem.txt
    └── broadcast
         ├── /ws/overview (10 Hz, compact, all worms)
         ├── /ws/focus/<n> (60 Hz, full, one worm)
         └── /ws/poems (event, on every eaten word)

[ viewer ]
    /         → /ws/overview        → grid canvases
    /focus/N  → /ws/focus/N         → v5-style detail
    /poems    → /api/poems + /ws/poems → columns
```

## Testing

1. **Determinism test** (pytest): run a worm for 600 ticks with `seed=1`, save the poem; run again with the same seed; assert poem sequences are identical. This is the load-bearing claim.
2. **Multi-worm smoke test**: instantiate 6 worms, tick 600 times, assert each produced ≥1 eaten word and all poem files exist with non-zero size.
3. **End-to-end manual**: start server, hit `/`, `/focus/alice`, `/poems` via the SSH-forwarded port. Then bring up `cloudflared` and confirm the public URL serves the same pages and that WS upgrades work through the proxy.

## Out of scope (notes for later)

- Worm weight mutation. v6 saves weights but doesn't change them. The persistence layout already supports it.
- Selectable per-worm corpora. v6 hard-codes Hamlet for everyone.
- Public read-write interactions. The orchestrator's debug routes are reusable for opt-in public actions later.
- Stable Cloudflare URL. Quick tunnel for v6 v1; named tunnel + custom domain is a v6.1 upgrade.
