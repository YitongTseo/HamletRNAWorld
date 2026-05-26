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
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

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

from server.orchestrator import load_worms, drain_and_persist, reset_worm, Worm

V6_ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = V6_ROOT / "viewer"

TICK_HZ = 60
TICK_DT = 1.0 / TICK_HZ
OVERVIEW_EVERY = 6  # 60/6 = 10 Hz overview broadcasts

# Process-wide state, populated in lifespan().
WORMS: list[Worm] = []
WORM_BY_NAME: dict[str, Worm] = {}
OVERVIEW_CLIENTS: set[WebSocket] = set()
FOCUS_CLIENTS: dict[str, set[WebSocket]] = {}
POEM_CLIENTS: set[WebSocket] = set()


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


def _overview_payload() -> dict:
    return {
        "type": "overview",
        "tick": WORMS[0].world.tick_count if WORMS else 0,
        "worms": [
            {
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
            for w in WORMS
        ],
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


def _focus_payload(worm: Worm) -> dict:
    snap = _build_snapshot(worm)
    snap["type"] = "state"
    snap["name"] = worm.name
    snap["word_count"] = worm.word_count
    return snap


async def _broadcast(clients: set[WebSocket], payload: dict) -> None:
    if not clients:
        return
    msg = json.dumps(payload)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def sim_loop():
    """Tick all worms at 60 Hz; broadcast overview at 10 Hz, focus at 60 Hz."""
    next_t = time.monotonic()
    overview_counter = 0
    while True:
        for worm in WORMS:
            worm.world.tick()
            for word in drain_and_persist(worm):
                await _broadcast(POEM_CLIENTS, {
                    "type": "eaten", "worm": worm.name,
                    "word": word, "word_count": worm.word_count,
                })

        # Focus broadcasts every tick for any worm with subscribers.
        for name, clients in list(FOCUS_CLIENTS.items()):
            if not clients:
                continue
            w = WORM_BY_NAME.get(name)
            if w is None:
                continue
            await _broadcast(clients, _focus_payload(w))

        overview_counter += 1
        if overview_counter >= OVERVIEW_EVERY:
            overview_counter = 0
            if OVERVIEW_CLIENTS:
                await _broadcast(OVERVIEW_CLIENTS, _overview_payload())

        next_t += TICK_DT
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_t = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global WORMS, WORM_BY_NAME
    n = int(os.environ.get("WORMLET_N_WORMS", "0")) or None
    WORMS = load_worms(n_worms=n)
    WORM_BY_NAME = {w.name: w for w in WORMS}
    print(f"[WORMLET] loaded {len(WORMS)} worms: {[w.name for w in WORMS]}")
    _ensure_debug_secret()
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


@app.get("/poems")
async def poems_page():
    return FileResponse(VIEWER_DIR / "poems.html")


@app.get("/about")
async def about_page():
    return FileResponse(VIEWER_DIR / "about.html")


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


@app.get("/api/worms")
async def list_worms():
    return JSONResponse([
        {"name": w.name, "seed": w.seed, "word_count": w.word_count}
        for w in WORMS
    ])


@app.get("/api/poems")
async def all_poems():
    out = {}
    for w in WORMS:
        if w.poem_path.exists():
            lines = w.poem_path.read_text().splitlines()
            out[w.name] = [l for l in lines if l]
        else:
            out[w.name] = []
    return JSONResponse(out)


@app.get("/api/poems/{name}")
async def one_poem(name: str):
    w = WORM_BY_NAME.get(name)
    if w is None:
        raise HTTPException(404, "unknown worm")
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


@app.websocket("/ws/focus/{name}")
async def ws_focus(ws: WebSocket, name: str):
    if name not in WORM_BY_NAME:
        await ws.close(code=4004)
        return
    await ws.accept()
    FOCUS_CLIENTS.setdefault(name, set()).add(ws)
    try:
        await ws.send_text(json.dumps(_focus_payload(WORM_BY_NAME[name])))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        FOCUS_CLIENTS.get(name, set()).discard(ws)


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
    w = WORM_BY_NAME.get(name)
    if w is None:
        raise HTTPException(404, "unknown worm")
    return w


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
