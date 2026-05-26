"""FastAPI server for v6.

Runs N worms headless at 60 Hz. Each worm has its own World, its own seed,
and its own poem file. Three viewer pages:
- /            : grid of thumbnails (lightweight overview, 10 Hz)
- /focus/<n>   : full v5-style detail view for one worm (60 Hz)
- /poems       : side-by-side columns, streamed live

Public visitors can subscribe to any of the read-only feeds. The /debug/*
routes (pause, reset, set_head, add_food) require a Bearer token from the
WORMLET_DEBUG_SECRET env var.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

LOG = logging.getLogger("wormlet")
if not LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False

from fastapi import (
    Depends, FastAPI, Header, HTTPException, Request, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from sim.connectome import (
    MUSCLE_PREFIXES, FOOD_SENSE_NEURONS, NOSE_TOUCH_NEURONS, HUNGER_NEURONS,
    SENSORY_NEURONS, CHEMOSENSORY_NEURONS, MOTOR_NEURONS,
)
from corpus.hamlet import get_sentences
from sim.world import World

from server.orchestrator import (
    load_worms, load_flasks, drain_and_persist, reset_worm, Worm,
)
from server.worm_group import WormGroup
from server import generations as gens_mod
from server.generations import (
    GenerationProgress, GenerationState, run_generation_rollover,
    PHASE_RUNNING, PHASE_CORPUS_DRAINING,
)

V6_ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = V6_ROOT / "viewer"

TICK_HZ = 60
TICK_DT = 1.0 / TICK_HZ
OVERVIEW_EVERY = 6  # 60/6 = 10 Hz overview broadcasts

# Process-wide state, populated in lifespan().
# When generations are enabled, FLASKS holds the 4 WormGroup containers and
# WORMS is a flattened view across all flasks (for convenience). When
# disabled, FLASKS is empty and WORMS is the legacy single-group lineup.
FLASKS: list[WormGroup] = []
WORMS: list[Worm] = []
WORM_BY_KEY: dict[tuple[str, str], Worm] = {}  # (flask_name, worm_name) → Worm
OVERVIEW_CLIENTS: set[WebSocket] = set()
# Focus subscribers keyed by "flask/worm" so different flasks' Alices don't collide.
FOCUS_CLIENTS: dict[str, set[WebSocket]] = {}
POEM_CLIENTS: set[WebSocket] = set()

_STARTED_AT: float = time.monotonic()
_LAST_TICK_AT: float = time.monotonic()  # updated each sim_loop iteration; read by /healthz

# Generational evolution state. Off by default for backward compat — set
# WORMLET_GENERATIONS_ENABLED=1 in /home/web/.wormlet.env to turn it on.
GENERATIONS_ENABLED: bool = os.environ.get("WORMLET_GENERATIONS_ENABLED", "0") == "1"
N_FLASKS: int = int(os.environ.get("WORMLET_N_FLASKS", "4"))
N_WORMS_PER_FLASK: int = int(os.environ.get("WORMLET_N_WORMS_PER_FLASK", "10"))
GENERATION_PROGRESS = GenerationProgress()
GENERATION_GRACE_S = 3.0           # seconds after corpus exhausts before scoring fires


def _focus_key(flask: str, worm: str) -> str:
    return f"{flask}/{worm}"


def _find_worm(flask: str, worm: str) -> Worm | None:
    return WORM_BY_KEY.get((flask, worm))


def _find_flask(name: str) -> WormGroup | None:
    for f in FLASKS:
        if f.name == name:
            return f
    return None


def _ensure_debug_secret() -> str:
    s = os.environ.get("WORMLET_DEBUG_SECRET")
    if s:
        return s
    generated = secrets.token_urlsafe(24)
    os.environ["WORMLET_DEBUG_SECRET"] = generated
    print(f"\n[WORMLET] No WORMLET_DEBUG_SECRET set; generated for this session:\n    {generated}\n")
    return generated


def require_debug(authorization: str | None = Header(default=None),
                  token: str | None = None) -> None:
    """Verify Bearer token. Accepts Authorization: Bearer <token> or ?token=."""
    expected = os.environ.get("WORMLET_DEBUG_SECRET", "")
    if not expected:
        raise HTTPException(status_code=500, detail="debug secret not set")
    presented = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    elif token:
        presented = token.strip()
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="bad debug token")


def _compact_midline(midline: list[tuple[float, float]], n: int = 24) -> list[list[float]]:
    if not midline:
        return []
    if len(midline) <= n:
        return [[round(x, 1), round(y, 1)] for x, y in midline]
    step = (len(midline) - 1) / (n - 1)
    out = []
    for i in range(n):
        x, y = midline[round(i * step)]
        out.append([round(x, 1), round(y, 1)])
    return out


def _worm_overview_dict(w: Worm, flask_name: str) -> dict:
    return {
        "flask": flask_name,
        "name": w.name,
        "head": [round(w.world.worm.target_x, 1),
                 round(w.world.worm.target_y, 1)],
        "facing": round(w.world.worm.facing_dir, 3),
        "midline": _compact_midline(w.world.worm.midline()),
        "food": [
            {"x": round(f.x, 1), "y": round(f.y, 1), "word": f.word}
            for f in w.world.food
        ],
        "word_count": w.word_count,
        "recent_words": w.recent_words[-3:],
        "paused": w.world.paused,
    }


def _overview_payload() -> dict:
    """When generations are enabled, broadcast the per-flask structure so
    the frontend can render 4 sections side by side. In legacy mode, wrap
    the single worm list as a single 'default' flask so the frontend
    contract stays the same."""
    tick = WORMS[0].world.tick_count if WORMS else 0
    if FLASKS:
        return {
            "type": "overview",
            "tick": tick,
            "flasks": [
                {
                    "name": f.name,
                    "display": f.display,
                    "generation": f.state.generation if f.state else 0,
                    "worms": [_worm_overview_dict(w, f.name) for w in f.worms],
                }
                for f in FLASKS
            ],
        }
    return {
        "type": "overview",
        "tick": tick,
        "flasks": [{
            "name": "default",
            "display": "Worms",
            "generation": 0,
            "worms": [_worm_overview_dict(w, "default") for w in WORMS],
        }],
    }


def _build_snapshot(worm: Worm) -> dict:
    """Same shape as v5's World.snapshot(), kept inline so we don't add
    snapshot() back to the sim package."""
    w = worm.world
    midline = w.worm.midline()
    VIS_THRESHOLD = 1.0
    raw = w.brain.activity()
    neurons_active = {n: round(v, 1) for n, v in raw.items() if v > VIS_THRESHOLD}
    return {
        "world": {"w": 1600.0, "h": 1000.0},
        "body_kind": w.body_kind,
        "midline": [[round(x, 2), round(y, 2)] for x, y in midline],
        "head": [round(w.worm.target_x, 2), round(w.worm.target_y, 2)],
        "facing": round(w.worm.facing_dir, 4),
        "speed": round(w.worm.speed, 4),
        "food": [
            {"x": round(f.x, 2), "y": round(f.y, 2), "word": f.word,
             "line_id": f.line_id, "word_idx": f.word_idx}
            for f in w.food
        ],
        "smells": [
            {"word": s["word"], "x": round(s["x"], 2), "y": round(s["y"], 2),
             "distance": round(s["distance"], 2),
             "pca": [round(x, 4) for x in s["pca"]],
             "neurons": s["neurons"],
             "direction_factor": round(s["direction_factor"], 3),
             "distance_factor": round(s["distance_factor"], 3)}
            for s in w.sensed_smells.values()
        ],
        "residual": {
            "pca": [round(x, 4) for x in getattr(w, "_last_residual_pca", [0.0] * 12)],
            "words": [{"word": wd, "decay": dc}
                      for wd, dc in getattr(w, "_last_residual_words", [])],
        },
        "motor": {"L": round(w.brain.accum_left, 2),
                  "R": round(w.brain.accum_right, 2)},
        "stim": {"hunger": w.stim_hunger, "nose_touch": w.stim_nose_touch,
                 "food_sense": w.stim_food_sense},
        "paused": w.paused,
        "neurons": neurons_active,
    }


def _focus_payload(worm: Worm, flask_name: str = "default") -> dict:
    snap = _build_snapshot(worm)
    snap["type"] = "state"
    snap["name"] = worm.name
    snap["flask"] = flask_name
    snap["word_count"] = worm.word_count
    return snap


_BROADCAST_SEND_TIMEOUT = 1.0  # seconds; per-client cap so one slow consumer can't stall sim_loop


async def _broadcast(clients: set[WebSocket], payload: dict) -> None:
    if not clients:
        return
    msg = json.dumps(payload)

    async def _send(ws: WebSocket):
        try:
            await asyncio.wait_for(ws.send_text(msg), timeout=_BROADCAST_SEND_TIMEOUT)
            return None
        except Exception:
            return ws

    snapshot = list(clients)
    results = await asyncio.gather(*(_send(ws) for ws in snapshot), return_exceptions=True)
    for r in results:
        if isinstance(r, WebSocket):
            clients.discard(r)


def _atomic_write_json(path, payload) -> None:
    """Write JSON to `path` via temp-file + rename so a mid-write crash
    leaves either the OLD or the NEW content on disk, never a partial.
    POSIX rename is atomic on the same filesystem."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _mutate_seed(s: int) -> int:
    """Numerical Recipes LCG step — deterministic, well-mixed, keeps the
    result a positive 31-bit int. Each generation gets a fresh seed
    while staying fully reproducible from (base_seed, gen_count)."""
    return (s * 1664525 + 1013904223) & 0x7FFFFFFF


def _respawn_flask(flask: WormGroup, new_weights: dict) -> None:
    """Install new per-worm weights + mutate per-worm seeds for the next
    generation, and reset poem state. Caller has already persisted the
    previous gen's artifacts (including pre-mutation seed.txt) to disk."""
    for w in flask.worms:
        wt = new_weights.get(w.name)
        if wt is None:
            continue
        # Mutate this worm's seed for the new generation. Persist atomically
        # so a crash mid-respawn leaves either the OLD or NEW seed on disk,
        # never a partial. The per-generation snapshot in
        # data/generations/<flask>/gen-NNNN/<worm>/seed.txt records which
        # seed was used FOR THAT generation (it was already written by
        # run_generation_rollover BEFORE this respawn, capturing w.seed at
        # the moment the previous generation ran).
        w.seed = _mutate_seed(w.seed)
        tmp = w.poem_path.parent / "seed.txt.tmp"
        tmp.write_text(str(w.seed))
        tmp.replace(w.poem_path.parent / "seed.txt")
        _atomic_write_json(w.poem_path.parent / "weights.json", wt)
        try:
            w.poem_file.close()
        except Exception:
            pass
        w.world = World(seed=w.seed, weights=wt)
        w.poem_path.write_text("")  # truncate previous-generation poem
        w.poem_file = open(w.poem_path, "a", buffering=1)
        w.word_count = 0
        w.recent_words = []
    flask.corpus_exhausted_at = None


def _generation_keepalive() -> None:
    """Bump _LAST_TICK_AT so the tick watchdog doesn't fire while sim_loop
    is paused for scoring."""
    global _LAST_TICK_AT
    _LAST_TICK_AT = time.monotonic()


def _run_all_flask_rollovers_sync() -> None:
    """Synchronously roll over every flask, then call the meta-gardener,
    then commit + purge everything atomically. Runs in a worker thread
    so the event loop stays responsive for the progress overlay.

    Atomicity matters: we want either ALL six flasks' data + the meta
    log in git for an epoch, or none. So per-flask rollovers no longer
    commit themselves (they pass run_gardener=False which now also
    skips commit+purge); a single git commit at the end covers
    everything."""
    from server.gardener import maybe_write_meta_log
    from server.generations import GENERATIONS_ROOT, _git_commit, _purge_gen_dir

    # Initialize the cumulative progress totals BEFORE the per-flask loop
    # so run_generation_rollover doesn't reset them per-flask (it skips
    # the reset when worms_total is already > 0). This gives the overlay
    # one smooth 0→N bar instead of one that jumps back to 0 each flask.
    GENERATION_PROGRESS.worms_total = sum(len(f.worms) for f in FLASKS)
    GENERATION_PROGRESS.worms_done = 0
    GENERATION_PROGRESS.started_at = time.time()

    new_weights_by_flask: dict[str, dict] = {}
    per_flask_scoring: dict[str, dict] = {}

    for flask in FLASKS:
        LOG.info("rollover starting for %s gen=%d",
                 flask.name, (flask.state.generation if flask.state else 0) + 1)
        try:
            # run_generation_rollover handles judging + NES + per-worm
            # artifact writing + (optionally) git commit for this flask.
            new_weights = run_generation_rollover(
                flask.worms, flask.state, GENERATION_PROGRESS, _generation_keepalive,
                run_gardener=False,  # meta-gardener fires once after all flasks
            )
            new_weights_by_flask[flask.name] = new_weights
            per_flask_scoring[flask.name] = {
                "best_score": (flask.state.best_score_history[-1]
                               if flask.state.best_score_history else 0.0),
                "generation": flask.state.generation,
            }
        except Exception:
            LOG.exception("flask %s rollover raised; continuing", flask.name)

    # Meta-gardener observes all flasks at once and writes one log per
    # epoch (after all six flasks have rolled over).
    try:
        epoch_num = max((f.state.generation if f.state else 0) for f in FLASKS) if FLASKS else 0
        log_path = maybe_write_meta_log(
            flasks=FLASKS,
            generation_num=epoch_num,
            keepalive=_generation_keepalive,
        )
        if log_path:
            LOG.info("meta-gardener wrote %s", log_path)
        else:
            LOG.info("meta-gardener PASSed or was skipped")
    except Exception:
        LOG.exception("meta-gardener raised; continuing without log")

    # --- Find the GLOBAL winner across all 36 worms for this epoch ---
    # Each per-flask metadata.json already has the flask's best_score +
    # ranked worms. Cross-flask comparison picks the single (flask, worm)
    # pair with the highest fitness — that's the only worm whose full
    # weights+poem we keep locally after purge.
    global_winner_flask: str | None = None
    global_winner_worm: str | None = None
    global_winner_score = float("-inf")
    for flask in FLASKS:
        gen = flask.state.generation if flask.state else 0
        if gen < 1:
            continue
        gen_dir = GENERATIONS_ROOT / flask.name / f"gen-{gen:04d}"
        try:
            md = json.loads((gen_dir / "metadata.json").read_text())
            score = md.get("best_score", float("-inf"))
            if score > global_winner_score and md.get("ranks"):
                global_winner_score = score
                global_winner_flask = flask.name
                global_winner_worm = md["ranks"][0]
        except Exception:
            LOG.exception("couldn't read metadata for %s gen-%04d", flask.name, gen)

    # Record the global winner alongside the meta-gardener log so the epoch's
    # winner is part of the archival record. The file is small and shows
    # provenance: which flask, which worm, which epoch, what they scored.
    if global_winner_flask and epoch_num >= 1:
        meta_dir = GENERATIONS_ROOT / "meta" / f"gen-{epoch_num:04d}"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "winner.json").write_text(json.dumps({
            "epoch": epoch_num,
            "flask": global_winner_flask,
            "worm": global_winner_worm,
            "fitness": global_winner_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        LOG.info("epoch %d global winner: %s/%s (fitness=%.3f)",
                 epoch_num, global_winner_flask, global_winner_worm, global_winner_score)

    # --- One atomic commit covers all six flasks' data + the meta log + winner.json ---
    GENERATION_PROGRESS.phase = gens_mod.PHASE_COMMITTING
    _generation_keepalive()
    committed = False
    if os.environ.get("WORMLET_GIT_COMMIT", "1") != "0":
        flask_paths = []
        commit_lines = []
        for flask in FLASKS:
            gen = flask.state.generation if flask.state else 0
            if gen < 1:
                continue
            gen_dir = GENERATIONS_ROOT / flask.name / f"gen-{gen:04d}"
            if gen_dir.exists():
                flask_paths.append(gen_dir)
                best = flask.state.best_score_history[-1] if flask.state.best_score_history else 0.0
                commit_lines.append(f"{flask.name}: best={best:.3f} σ={flask.state.sigma:.3f}")
        meta_dir = GENERATIONS_ROOT / "meta" / f"gen-{epoch_num:04d}"
        if meta_dir.exists():
            flask_paths.append(meta_dir)
        if flask_paths:
            winner_tag = (f" winner={global_winner_flask}/{global_winner_worm} "
                          f"({global_winner_score:.3f})") if global_winner_flask else ""
            msg = (f"epoch {epoch_num:04d}:{winner_tag} · "
                   + ", ".join(commit_lines))
            committed = _git_commit(msg, flask_paths, keepalive=_generation_keepalive)
            if committed:
                LOG.info("epoch %d committed to git", epoch_num)
            else:
                LOG.warning("epoch %d git commit failed; data still on disk", epoch_num)
    else:
        LOG.info("WORMLET_GIT_COMMIT=0: skipping commit; nothing purged")

    # --- Purge bulky files only after the data is safely in git ---
    # Global-winner policy: only the (global_winner_flask, global_winner_worm)
    # pair gets its weights.json + poem_clean.txt kept. Every other worm in
    # every flask is fully purged (still keeping fitness/seed/scores for the
    # gardener to read).
    purge_anyway = os.environ.get("WORMLET_PURGE_ANYWAY", "0") == "1"
    if committed or purge_anyway:
        for flask in FLASKS:
            gen = flask.state.generation if flask.state else 0
            if gen < 1:
                continue
            gen_dir = GENERATIONS_ROOT / flask.name / f"gen-{gen:04d}"
            kept_worm = (global_winner_worm
                         if flask.name == global_winner_flask else None)
            try:
                _purge_gen_dir(gen_dir, kept_worm=kept_worm)
            except Exception:
                LOG.exception("purge for %s gen-%04d raised; continuing", flask.name, gen)

    # Install new weights into each flask only after all rollovers + the
    # meta-gardener + commit + purge have completed. Doing this in a separate
    # pass keeps each flask's poem.txt and gen-NNNN files intact until the
    # gardener has read them.
    GENERATION_PROGRESS.phase = gens_mod.PHASE_RESPAWNING
    for flask in FLASKS:
        new_weights = new_weights_by_flask.get(flask.name)
        if new_weights:
            _respawn_flask(flask, new_weights)


async def _trigger_generation_rollover() -> None:
    """Drive one end-of-generation cycle across all flasks. Runs the
    synchronous heavy lifting in a thread so the event loop stays
    responsive for /healthz and the progress overlay."""
    LOG.info("starting epoch rollover (%d flasks)", len(FLASKS))
    try:
        await asyncio.to_thread(_run_all_flask_rollovers_sync)
        LOG.info("epoch rollover complete")
    except Exception as e:
        LOG.exception("epoch rollover failed")
        GENERATION_PROGRESS.error = str(e)
    finally:
        GENERATION_PROGRESS.phase = PHASE_RUNNING
        _generation_keepalive()


def _iter_flask_worms():
    """Yield (flask_name, worm) for every worm in every flask. In legacy
    single-group mode, FLASKS is empty and we yield ('default', w) for
    every worm in WORMS instead."""
    if FLASKS:
        for f in FLASKS:
            for w in f.worms:
                yield f.name, w
    else:
        for w in WORMS:
            yield "default", w


async def sim_loop():
    """Tick all worms at 60 Hz; broadcast overview at 10 Hz, focus at 60 Hz."""
    global _LAST_TICK_AT
    next_t = time.monotonic()
    overview_counter = 0
    epoch_exhausted_at: float | None = None
    while True:
        # If a rollover is in progress, skip ticking entirely — the rollover
        # path keeps the watchdog alive via _generation_keepalive.
        if GENERATION_PROGRESS.phase not in (PHASE_RUNNING, PHASE_CORPUS_DRAINING):
            await asyncio.sleep(0.25)
            continue

        try:
            for flask_name, worm in _iter_flask_worms():
                try:
                    worm.world.tick()
                    for word in drain_and_persist(worm):
                        await _broadcast(POEM_CLIENTS, {
                            "type": "eaten",
                            "flask": flask_name,
                            "worm": worm.name,
                            "word": word,
                            "word_count": worm.word_count,
                        })
                except Exception:
                    LOG.exception("worm %s/%s tick failed; continuing", flask_name, worm.name)

            _LAST_TICK_AT = time.monotonic()

            # Focus broadcasts every tick for any (flask, worm) with subscribers.
            for key, clients in list(FOCUS_CLIENTS.items()):
                if not clients:
                    continue
                # key shape: "flask_name/worm_name"
                if "/" in key:
                    flask_name, worm_name = key.split("/", 1)
                else:
                    flask_name, worm_name = "default", key
                w = _find_worm(flask_name, worm_name)
                if w is None:
                    continue
                await _broadcast(clients, _focus_payload(w, flask_name))

            overview_counter += 1
            if overview_counter >= OVERVIEW_EVERY:
                overview_counter = 0
                if OVERVIEW_CLIENTS:
                    await _broadcast(OVERVIEW_CLIENTS, _overview_payload())

            # End-of-epoch detection. An "epoch" is one full corpus pass
            # across all flasks. Sim_loop waits until every flask reports
            # corpus_exhausted, then enforces a brief grace period, then
            # hands off to the joint rollover.
            if GENERATIONS_ENABLED and FLASKS:
                all_exhausted = all(f.corpus_exhausted for f in FLASKS)
                if all_exhausted:
                    if epoch_exhausted_at is None:
                        epoch_exhausted_at = time.monotonic()
                        GENERATION_PROGRESS.phase = PHASE_CORPUS_DRAINING
                        LOG.info("epoch corpus exhausted; %.1fs grace before rollover",
                                 GENERATION_GRACE_S)
                    elif time.monotonic() - epoch_exhausted_at >= GENERATION_GRACE_S:
                        epoch_exhausted_at = None
                        await _trigger_generation_rollover()
                else:
                    epoch_exhausted_at = None
                    if GENERATION_PROGRESS.phase == PHASE_CORPUS_DRAINING:
                        GENERATION_PROGRESS.phase = PHASE_RUNNING
        except Exception:
            LOG.exception("sim_loop iteration raised; watchdog will recover if persistent")

        next_t += TICK_DT
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_t = time.monotonic()


def _start_tick_watchdog(stall_seconds: float = 20.0, poll_seconds: float = 2.0) -> None:
    # Background thread: if WORMS[0].world.tick_count doesn't advance for stall_seconds,
    # exit so systemd's Restart=on-failure brings us back. Thread-based (not asyncio) so
    # it still fires if the event loop itself deadlocks.
    def loop() -> None:
        last_tick = -1
        last_change = time.monotonic()
        while True:
            time.sleep(poll_seconds)
            if not WORMS:
                last_change = time.monotonic()
                continue
            # Don't kill the process during a deliberate rollover freeze.
            # Reset the timer instead so we re-arm when ticking resumes.
            if GENERATION_PROGRESS.phase not in (PHASE_RUNNING, PHASE_CORPUS_DRAINING):
                last_change = time.monotonic()
                continue
            try:
                t = WORMS[0].world.tick_count
            except Exception:
                continue
            now = time.monotonic()
            if t != last_tick:
                last_tick = t
                last_change = now
                continue
            if now - last_change > stall_seconds:
                LOG.critical(
                    "watchdog: tick frozen at %d for %.1fs; exiting for systemd restart",
                    t, now - last_change,
                )
                for h in LOG.handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass
                os._exit(1)

    threading.Thread(target=loop, daemon=True, name="wormlet-watchdog").start()


def _asyncio_exception_handler(loop, context: dict) -> None:
    exc = context.get("exception")
    msg = context.get("message", "unhandled asyncio exception")
    if exc is not None:
        LOG.error("asyncio: %s", msg, exc_info=(type(exc), exc, exc.__traceback__))
    else:
        LOG.error("asyncio: %s | context=%r", msg, context)


def _truncate_partial_gen_poems(roots: list[Path]) -> None:
    """Clear each worm's poem.txt at startup across the given roots. The
    new generation always starts from empty poems regardless of whether
    this is a fresh enable, a clean restart between generations, or a
    crash recovery mid-generation. Voluntary restarts during a
    generation will lose the partial poem — intentional for consistency:
    a generation must correspond to a single contiguous run through the
    corpus, not a spliced one. Per-generation poem.txt files are
    already persisted to data/generations/<flask>/gen-NNNN/<worm>/poem_raw.txt
    by the rollover step, so this truncation never destroys anything
    that completed."""
    for root in roots:
        if not root.exists():
            continue
        for poem in root.rglob("poem.txt"):
            try:
                if poem.stat().st_size > 0:
                    poem.write_text("")
                    LOG.info("truncated partial-gen poem: %s", poem)
            except Exception:
                LOG.exception("failed to truncate %s", poem)


def _load_initial_default_weights() -> dict:
    """The starter connectome every flask's parent vector is seeded from."""
    from server.orchestrator import DEFAULT_WEIGHTS
    return json.loads(DEFAULT_WEIGHTS.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global FLASKS, WORMS, WORM_BY_KEY, _STARTED_AT, _LAST_TICK_AT
    _STARTED_AT = time.monotonic()
    _LAST_TICK_AT = time.monotonic()

    from server.orchestrator import DATA_DIR, FLASKS_DIR

    if GENERATIONS_ENABLED:
        _truncate_partial_gen_poems([DATA_DIR, FLASKS_DIR])
        # Build 4 flasks of 10 worms each. Each flask's GenerationState is
        # loaded from disk (or initialized cold) so the lineage survives
        # process restarts.
        per_flask_worms = load_flasks(n_flasks=N_FLASKS, n_worms_per_flask=N_WORMS_PER_FLASK)
        default_weights = _load_initial_default_weights()
        FLASKS = []
        for fi, worms in enumerate(per_flask_worms):
            flask_name = f"flask_{fi + 1}"
            state = GenerationState.load_or_init(flask_name, default_weights)
            FLASKS.append(WormGroup(
                name=flask_name,
                display=f"Flask {fi + 1}",
                worms=worms,
                state=state,
            ))
            WORMS.extend(worms)
            for w in worms:
                WORM_BY_KEY[(flask_name, w.name)] = w
        LOG.info("loaded %d flasks × %d worms = %d total",
                 len(FLASKS), len(FLASKS[0].worms) if FLASKS else 0, len(WORMS))
    else:
        # Legacy single-group mode for backward compat.
        n = int(os.environ.get("WORMLET_N_WORMS", "0")) or None
        WORMS = load_worms(n_worms=n)
        for w in WORMS:
            WORM_BY_KEY[("default", w.name)] = w
        LOG.info("loaded %d worms (legacy single-group mode): %s",
                 len(WORMS), [w.name for w in WORMS])

    _ensure_debug_secret()
    asyncio.get_running_loop().set_exception_handler(_asyncio_exception_handler)
    _start_tick_watchdog()
    task = asyncio.create_task(sim_loop())
    try:
        yield
    finally:
        task.cancel()
        for w in WORMS:
            w.close()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def disable_cdn_cache(request, call_next):
    """Tell Cloudflare and the browser not to cache our HTML/JS/CSS. This
    is a dev project iterating quickly; we'd rather pay the origin hit
    than serve stale code. `CDN-Cache-Control` is the directive Cloudflare
    respects independently of the browser's `Cache-Control`."""
    response = await call_next(request)
    path = request.url.path
    if (path.startswith("/static/") or path.startswith("/focus/")
            or path in ("/", "/poems", "/about")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["CDN-Cache-Control"] = "no-store"
    return response


# --- HTML pages ---

@app.get("/")
async def index():
    return FileResponse(VIEWER_DIR / "index.html")


@app.get("/focus/{name}")
async def focus_page(name: str):
    return FileResponse(VIEWER_DIR / "focus.html")


@app.get("/focus/{flask}/{name}")
async def focus_page_flask(flask: str, name: str):
    return FileResponse(VIEWER_DIR / "focus.html")


@app.get("/poems")
async def poems_page():
    return FileResponse(VIEWER_DIR / "poems.html")


@app.get("/about")
async def about_page():
    return FileResponse(VIEWER_DIR / "about.html")


@app.get("/healthz")
async def healthz():
    from sim.world import EMBEDDING_MODE
    now = time.monotonic()
    tick = WORMS[0].world.tick_count if WORMS else 0
    return JSONResponse({
        "tick": tick,
        "uptime_s": round(now - _STARTED_AT, 1),
        "last_tick_advance_s_ago": round(now - _LAST_TICK_AT, 2),
        "embedding": EMBEDDING_MODE,
        "generations_enabled": GENERATIONS_ENABLED,
        "flasks": [
            {"name": f.name, "display": f.display,
             "generation": (f.state.generation if f.state else 0),
             "sigma": (f.state.sigma if f.state else None),
             "n_worms": len(f.worms)}
            for f in FLASKS
        ],
        "n_worms_total": len(WORMS),
        "clients": {
            "overview": len(OVERVIEW_CLIENTS),
            "focus": {name: len(s) for name, s in FOCUS_CLIENTS.items() if s},
            "poems": len(POEM_CLIENTS),
        },
    })


@app.get("/api/generation_status")
async def generation_status():
    """Frontend polls this every ~500ms during rollover to drive the
    progress overlay. Returns the live GenerationProgress as JSON; phase
    is 'running' the rest of the time."""
    p = GENERATION_PROGRESS
    return JSONResponse({
        "enabled": GENERATIONS_ENABLED,
        "phase": p.phase,
        "group": p.group,
        "generation": p.generation,
        "worms_total": p.worms_total,
        "worms_done": p.worms_done,
        "started_at": p.started_at,
        "elapsed_s": round(time.time() - p.started_at, 1) if p.started_at else 0,
        "error": p.error,
    })


# --- API ---

_GRAPH_PAYLOAD: dict | None = None


def _build_graph_payload() -> dict:
    worm = WORMS[0]
    brain = worm.world.brain
    neurons = brain.neurons
    idx = {n: i for i, n in enumerate(neurons)}
    edges = [
        [idx[pre], idx[post], w]
        for pre, targets in brain.weights.items()
        for post, w in targets.items()
    ]
    positions_path = V6_ROOT / "sim" / "neuron_positions.json"
    raw_pos = json.loads(positions_path.read_text()) if positions_path.exists() else {}
    positions = [raw_pos.get(n) for n in neurons]
    muscle_prefixes = set(MUSCLE_PREFIXES)
    food_set = set(FOOD_SENSE_NEURONS)
    nose_set = set(NOSE_TOUCH_NEURONS)
    hunger_set = set(HUNGER_NEURONS)
    return {
        "neurons": neurons,
        "edges": edges,
        "positions": positions,
        "fire_threshold": 30,
        "muscle_indices": [i for i, n in enumerate(neurons) if n[:3] in muscle_prefixes],
        "sensory_indices": [i for i, n in enumerate(neurons) if n in SENSORY_NEURONS],
        "chemosensory_indices": [i for i, n in enumerate(neurons) if n in CHEMOSENSORY_NEURONS],
        "motor_indices": [i for i, n in enumerate(neurons) if n in MOTOR_NEURONS],
        "food_indices": [i for i, n in enumerate(neurons) if n in food_set],
        "nose_indices": [i for i, n in enumerate(neurons) if n in nose_set],
        "hunger_indices": [i for i, n in enumerate(neurons) if n in hunger_set],
    }


@app.get("/api/graph")
async def graph():
    global _GRAPH_PAYLOAD
    if _GRAPH_PAYLOAD is None:
        _GRAPH_PAYLOAD = _build_graph_payload()
    return JSONResponse(_GRAPH_PAYLOAD)


_CORPUS_PCA_FILE_CACHE: dict | None = None
_CORPUS_UMAP_FILE_CACHE: dict | None = None


@app.get("/api/corpus_pca")
async def corpus_pca():
    """Serves the precomputed Hamlet PCA artifact (built once by
    scripts/build_corpus_pca.py). Includes:
      - words[]            ordered list of word types
      - pca12[][]          (N, 12) — each word's chemosensory-driving vector
      - pca2[][]           (N, 2) — for the focus-page hover scatter
      - pc_neuron_pairs[]  PC index → (L_neuron, R_neuron)
      - explained_variance_ratio_12d
      - model, prefix, n_words, n_pcs
    """
    global _CORPUS_PCA_FILE_CACHE
    if _CORPUS_PCA_FILE_CACHE is None:
        from sim.chemosensory_mapping import PC_NEURON_PAIRS
        path = V6_ROOT / "cache" / "corpus_pca.json"
        if not path.exists():
            raise HTTPException(503, "corpus_pca.json missing — run scripts/build_corpus_pca.py")
        data = json.loads(path.read_text())
        data["pc_neuron_pairs"] = [list(p) for p in PC_NEURON_PAIRS]
        _CORPUS_PCA_FILE_CACHE = data
    return JSONResponse(_CORPUS_PCA_FILE_CACHE)


@app.get("/api/corpus_umap")
async def corpus_umap():
    """Serves the precomputed Hamlet UMAP artifact (built once by
    scripts/build_corpus_umap.py). Same shape as /api/corpus_pca but with
    `umap12`/`umap12_sparse`/`umap2` instead of the `pca*` keys. The focus
    page reads `umap2` for the hover scatter; the sim still uses the PCA
    artifact for chemosensation (will swap when generational evolution
    starts)."""
    global _CORPUS_UMAP_FILE_CACHE
    if _CORPUS_UMAP_FILE_CACHE is None:
        path = V6_ROOT / "cache" / "corpus_umap.json"
        if not path.exists():
            raise HTTPException(503, "corpus_umap.json missing — run scripts/build_corpus_umap.py")
        _CORPUS_UMAP_FILE_CACHE = json.loads(path.read_text())
    return JSONResponse(_CORPUS_UMAP_FILE_CACHE)


_NEURON_BODY_CACHE: dict | None = None


@app.get("/api/neuron_body_coords")
async def neuron_body_coords():
    """Per-neuron 2D body-plan coordinates (axial in [0,1], lateral in
    [-1,+1]) for the x-ray visualization. Built once by
    scripts/build_neuron_body_coords.py from sim/neuron_positions.json."""
    global _NEURON_BODY_CACHE
    if _NEURON_BODY_CACHE is None:
        path = V6_ROOT / "cache" / "neuron_body_coords.json"
        if not path.exists():
            raise HTTPException(503, "neuron_body_coords.json missing — run scripts/build_neuron_body_coords.py")
        _NEURON_BODY_CACHE = json.loads(path.read_text())
    return JSONResponse(_NEURON_BODY_CACHE)


@app.get("/api/worms")
async def list_worms():
    """Return worms grouped by flask. In legacy mode FLASKS is empty and
    we synthesize a single 'default' flask wrapping WORMS."""
    if FLASKS:
        return JSONResponse([
            {
                "flask": f.name,
                "display": f.display,
                "worms": [{"name": w.name, "seed": w.seed, "word_count": w.word_count}
                          for w in f.worms],
            }
            for f in FLASKS
        ])
    return JSONResponse([{
        "flask": "default",
        "display": "Worms",
        "worms": [{"name": w.name, "seed": w.seed, "word_count": w.word_count}
                  for w in WORMS],
    }])


@app.get("/api/poems")
async def all_poems():
    """All poems flattened by (flask, worm). Same shape across legacy and
    flask modes; the flask key is 'default' when generations are off."""
    out: dict = {}
    for flask_name, w in _iter_flask_worms():
        if w.poem_path.exists():
            lines = w.poem_path.read_text().splitlines()
            content = [l for l in lines if l]
        else:
            content = []
        out.setdefault(flask_name, {})[w.name] = content
    return JSONResponse(out)


@app.get("/api/poems/{flask}/{name}")
async def one_poem_flask(flask: str, name: str):
    w = _find_worm(flask, name)
    if w is None:
        raise HTTPException(404, f"unknown worm {flask}/{name}")
    if w.poem_path.exists():
        return PlainTextResponse(w.poem_path.read_text())
    return PlainTextResponse("")


# --- WebSockets ---

@app.websocket("/ws/overview")
async def ws_overview(ws: WebSocket):
    await ws.accept()
    OVERVIEW_CLIENTS.add(ws)
    try:
        await ws.send_text(json.dumps(_overview_payload()))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        OVERVIEW_CLIENTS.discard(ws)


@app.websocket("/ws/focus/{flask}/{name}")
async def ws_focus_flask(ws: WebSocket, flask: str, name: str):
    w = _find_worm(flask, name)
    if w is None:
        await ws.close(code=4004)
        return
    key = _focus_key(flask, name)
    await ws.accept()
    FOCUS_CLIENTS.setdefault(key, set()).add(ws)
    try:
        await ws.send_text(json.dumps(_focus_payload(w, flask)))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        FOCUS_CLIENTS.get(key, set()).discard(ws)


@app.websocket("/ws/poems")
async def ws_poems(ws: WebSocket):
    await ws.accept()
    POEM_CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        POEM_CLIENTS.discard(ws)


# --- Debug door ---

def _get_worm_or_404(name: str) -> Worm:
    """Legacy debug-door lookup by bare worm name. In multi-flask mode
    'Alice' exists in every flask — we return the first match. Use
    /debug/<flask>/<name>/... routes if you need to target a specific
    flask. The debug door is operator-only via the bearer token, so this
    ambiguity is acceptable."""
    for (_flask, wname), w in WORM_BY_KEY.items():
        if wname == name:
            return w
    raise HTTPException(404, "unknown worm")


@app.post("/debug/{name}/pause", dependencies=[Depends(require_debug)])
async def debug_pause(name: str, request: Request):
    w = _get_worm_or_404(name)
    body = await request.json() if (await request.body()) else {}
    w.world.paused = bool(body.get("paused", True))
    return {"name": name, "paused": w.world.paused}


@app.post("/debug/{name}/reset", dependencies=[Depends(require_debug)])
async def debug_reset(name: str):
    w = _get_worm_or_404(name)
    reset_worm(w)
    return {"name": name, "reset": True}


@app.post("/debug/{name}/set_head", dependencies=[Depends(require_debug)])
async def debug_set_head(name: str, request: Request):
    w = _get_worm_or_404(name)
    body = await request.json()
    w.world.worm.target_x = float(body["x"])
    w.world.worm.target_y = float(body["y"])
    w.world.worm.facing_dir = float(body.get("facing", 0.0))
    return {"name": name, "head": [w.world.worm.target_x, w.world.worm.target_y]}


@app.post("/debug/{name}/add_food", dependencies=[Depends(require_debug)])
async def debug_add_food(name: str, request: Request):
    w = _get_worm_or_404(name)
    body = await request.json()
    w.world.add_food(float(body["x"]), float(body["y"]))
    return {"name": name, "ok": True}


# Static viewer files (JS/CSS) — mounted under /static so / and /ws take precedence.
app.mount("/static", StaticFiles(directory=str(VIEWER_DIR)), name="viewer")
