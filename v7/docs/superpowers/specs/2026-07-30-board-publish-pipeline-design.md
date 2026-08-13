# Board publish pipeline — design

**Date:** 2026-07-30
**Status:** approved, implementing
**Author:** brainstormed with Yitong

## Problem

The ESP32 boards need to pull each generation's winning worm (evolved embedder
genome + connectome weights) so they can render a worm that matches the server.
Today the sim commits per-generation artifacts to the shared `HamletRNAWorld`
git repo and pushes to GitHub. That has failed in two ways:

1. **The push races and loses.** Multiple servers push per-generation artifacts
   to the same `main` branch. `server/generations.py::_git_commit` does
   `add → commit → pull --rebase → push` under a *per-process* lock, but nothing
   coordinates across *servers*. Whichever server pushes first wins; the others
   hit a non-fast-forward rejection, `rebase --abort`, and strand the commit in
   local history. As of 2026-07-30 this box was **215 commits ahead of
   origin/main**, none pushed.
2. **It bloats history.** Those 215 commits are 100% data — 42,483 files /
   ~18.8M insertions of JSON. None of it is code. Pushing it would permanently
   bloat the GitHub repo, and cloning it onto a board is absurd.

Root cause: **a single shared mutable git branch is being used as a
multi-writer append log.** The artifacts don't actually collide (each process
writes a disjoint `data/poetry-N/` subtree); the conflict is purely git's
single-writer-per-ref model.

## Approach

**Git goes back to code-only. Evolving artifacts are self-hosted over the
existing Cloudflare tunnel.**

Each poetry process writes the board-facing subset of each generation into a
static tree under its **own** subdirectory, and the existing FastAPI app serves
that tree at `/board`. Because every writer owns a disjoint key namespace, the
cross-server race is *structurally impossible* — no lock, no rebase, no shared
mutable pointer.

Chosen over object storage (Cloudflare R2) and a separate artifacts git repo
because it needs **no new account, no new service, and no new hostname** — it
reuses the tunnel and app already running. Accepted trade-off: board-serving is
coupled to the app being up, which is fine because a board only needs *new*
weights when a generation completes, and that requires the app to be up anyway.

## Served tree

Root: `/home/web/board_publish/` (outside the repo; git never touches it).
Overridable via `WORMLET_BOARD_PUBLISH_DIR`.

```
board_publish/
  index.json                       # {processes: {poetry-1: {epoch, ts}, ...}}
  poetry-1/
    latest.json                    # manifest (see below)
    gen-0061/
      embedder.json                # evolved embedder genome (winner's flask)
      weights.json                 # winning worm's connectome genome
      winner.json                  # {epoch, flask, worm, fitness, ts}
      seed.txt                     # winner seed (board determinism input)
    gen-0060/ ...                  # keep last 20, prune older
  poetry-2/ ... poetry-3/ ... poetry-4/ ...
```

`latest.json`:
```json
{
  "process": "poetry-1",
  "epoch": 61,
  "winner": {"flask": "flask_2", "worm": "Alice", "fitness": 23.48},
  "seed": 1234567,
  "sha256": "<hash of embedder.json + weights.json bytes>",
  "files": {"embedder": "gen-0061/embedder.json",
            "weights":  "gen-0061/weights.json",
            "winner":   "gen-0061/winner.json",
            "seed":     "gen-0061/seed.txt"},
  "timestamp": "2026-07-30T04:00:00Z"
}
```

The board polls `latest.json`, compares `sha256` (or `epoch`), and fetches the
referenced files only when they change.

## Write protocol (atomicity)

A poller must never see a `latest.json` that points at half-written files:

1. `mkdir board_publish/<process>/gen-NNNN/`.
2. Copy `embedder.json`, `weights.json`, `winner.json`, `seed.txt` each via
   temp-file + `os.replace` (atomic per file).
3. Compute `sha256` over `embedder.json` + `weights.json` bytes.
4. Write `latest.json` **last**, via temp-file + `os.replace`.
5. Prune `gen-*` dirs beyond the newest 20.
6. Rewrite `index.json` via temp-file + `os.replace`.

Failure at any step is logged and returns `False` — non-fatal, exactly like the
old git commit. The sim keeps running.

## Serving

In `server/app.py`, after `app = FastAPI(...)`:
```python
BOARD_PUBLISH_DIR = Path(os.environ.get("WORMLET_BOARD_PUBLISH_DIR",
                                        "/home/web/board_publish"))
BOARD_PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/board", StaticFiles(directory=str(BOARD_PUBLISH_DIR)), name="board")
```
All four poetry procs mount the same shared dir, so any proc can serve any
process's subtree (e.g. `wordswordsworms.org/board/poetry-3/latest.json` works
even though poetry-1's app answers it).

## Integration point

`server/app.py::_run_all_flask_rollovers_sync`, the block at lines ~559–589 that
currently builds `flask_paths` and calls `_git_commit(...)`. Replace with:

```python
published = False
if os.environ.get("WORMLET_BOARD_PUBLISH", "1") != "0" and global_winner_flask and epoch_num >= 1:
    from server.board_publish import publish_board_bundle, PROCESS_ID
    published = publish_board_bundle(
        process_id=PROCESS_ID, generations_root=GENERATIONS_ROOT,
        epoch=epoch_num, winner_flask=global_winner_flask,
        winner_worm=global_winner_worm, winner_score=global_winner_score,
        keepalive=_generation_keepalive)
```

The purge block's gate changes from `if committed or purge_anyway:` to
`if published or purge_anyway:` — preserving the invariant "only purge local
files after the winner is safely published."

## Git changes (forward only — NO history surgery)

- Add `data/` to `v7/.gitignore`; drop the now-moot `!data/poetry-*/generations/`
  re-include. Prevents future accidental tracking.
- Remove the `_git_commit` data-commit call from the rollover.
- **Do NOT** `git rm --cached`, reset, or rewrite history yet: the 215 tracked
  data commits hold files that are *only* on this disk, so any history rewrite
  risks deleting them. Deferred until after the flash-drive backup exists. The
  215 local commits are harmless as long as they are never pushed; GitHub stays
  clean because we simply stop pushing data.

## Out of scope (separate workstreams)

- **Portable/deterministic math** so the board matches the server bit-for-bit.
  Independent: this pipeline publishes the exact JSON the server used; bit
  exactness is the C-port's problem, designed separately.
- **Baking to a flat `.hwrm`.** MVP publishes raw JSON. Baking happens later,
  either ported onto the server or done board/Mac-side from this raw pull.
- **`.git` de-bloat / second-server reconciliation.** After backup.

## Verification

1. Unit-ish: run `publish_board_bundle` against the current on-disk latest gen
   for poetry-1; assert the gen dir, all four files, and a well-formed
   `latest.json` (with a real sha256) appear under `board_publish/poetry-1/`.
2. Serving: `curl 127.0.0.1:8000/board/poetry-1/latest.json` → 200 + valid JSON;
   `curl .../board/poetry-1/gen-NNNN/weights.json` → 200.
3. Public: `curl https://wordswordsworms.org/board/poetry-1/latest.json` → 200.
4. Prune: publish 21 synthetic gens, assert only the newest 20 remain.
