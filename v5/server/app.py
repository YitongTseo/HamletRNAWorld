"""FastAPI server: ticks the world in a background task and ships state to
all connected viewers over a WebSocket. Also serves the static viewer.

Endpoints:
- GET /                    → viewer/index.html
- GET /<file>              → static file from viewer/
- WS  /ws                  → bidirectional channel:
    server → client: {"type": "state", ...snapshot}
    client → server: {"type": "add_food", "x": ..., "y": ...}
                     {"type": "clear_food"}
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import json as _json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from sim.world import World
from sim.connectome import (
    MUSCLE_PREFIXES, FOOD_SENSE_NEURONS, NOSE_TOUCH_NEURONS, HUNGER_NEURONS,
    SENSORY_NEURONS, CHEMOSENSORY_NEURONS, MOTOR_NEURONS,
)
from embeddings.embedder import embed_and_pca
from corpus.hamlet import get_sentences

VIEWER_DIR = Path(__file__).parent.parent / "viewer"

# Shared simulation singleton + connection set.
WORLD = World()
CLIENTS: set[WebSocket] = set()

# Sim ticks at 60 Hz; viewer broadcast piggybacks on the same loop.
TICK_HZ = 60
TICK_DT = 1.0 / TICK_HZ


async def sim_loop():
    """Drive the world forward in real time; broadcast each frame."""
    next_tick = time.monotonic()
    while True:
        now = time.monotonic()
        WORLD.step(now)
        if CLIENTS:
            payload = json.dumps({"type": "state", **WORLD.snapshot()})
            dead: list[WebSocket] = []
            for ws in CLIENTS:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                CLIENTS.discard(ws)

        next_tick += TICK_DT
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            # We fell behind — resync to "now" so we don't spiral.
            next_tick = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sim_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(VIEWER_DIR / "index.html")


# Graph topology (computed once and cached).
_GRAPH_PAYLOAD: dict | None = None
_POSITIONS_PATH = Path(__file__).parent.parent / "sim" / "neuron_positions.json"


def _build_graph_payload() -> dict:
    """Build static connectome graph: neurons + edges + positions + classifications."""
    brain = WORLD.brain
    neurons = brain.neurons  # sorted list of all neuron names
    idx = {n: i for i, n in enumerate(neurons)}

    # Edge list as [pre_idx, post_idx, weight] triples.
    edges = [
        [idx[pre], idx[post], w]
        for pre, targets in brain.weights.items()
        for post, w in targets.items()
    ]

    # Load anatomical positions (x, y, z in micrometers from c302).
    # Missing entries get None; client handles fallback.
    if _POSITIONS_PATH.exists():
        raw_pos = _json.loads(_POSITIONS_PATH.read_text())
    else:
        raw_pos = {}
    # Build positions list: [[x, y, z] or None] indexed by neuron index.
    positions = [raw_pos.get(n) for n in neurons]

    # Neuron classification sets.
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


@app.get("/graph")
async def graph():
    """Fetch the static connectome graph topology (neurons + edges)."""
    global _GRAPH_PAYLOAD
    if _GRAPH_PAYLOAD is None:
        _GRAPH_PAYLOAD = _build_graph_payload()
    return JSONResponse(_GRAPH_PAYLOAD)


# Embeddings (computed once and cached).
_EMBEDDING_PAYLOAD: dict | None = None


@app.get("/embeddings")
async def embeddings():
    """Fetch DistilBERT embeddings + PCA for Hamlet words."""
    global _EMBEDDING_PAYLOAD
    if _EMBEDDING_PAYLOAD is None:
        sentences = get_sentences()
        _EMBEDDING_PAYLOAD = embed_and_pca(sentences)
    return JSONResponse(_EMBEDDING_PAYLOAD)


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        # Send an initial snapshot immediately so the viewer doesn't show a
        # blank canvas while waiting for the next sim tick.
        await ws.send_text(json.dumps({"type": "state", **WORLD.snapshot()}))
        while True:
            msg = await ws.receive_text()
            try:
                cmd = json.loads(msg)
            except json.JSONDecodeError:
                continue
            t = cmd.get("type")
            if t == "add_food":
                WORLD.add_food(float(cmd["x"]), float(cmd["y"]))
            elif t == "clear_food":
                WORLD.clear_food()
            elif t == "set_paused":
                WORLD.paused = bool(cmd.get("paused", False))
            elif t == "toggle_paused":
                WORLD.paused = not WORLD.paused
            elif t == "set_head":
                # Debug-only: jump the worm head to a specified spot, useful
                # for taking visual-tuning screenshots without waiting for the
                # worm to wander into frame.
                WORLD.worm.target_x = float(cmd["x"])
                WORLD.worm.target_y = float(cmd["y"])
                WORLD.worm.facing_dir = float(cmd.get("facing", 0.0))
                # Re-seed the IK chain so the body follows immediately.
                from sim.worm import IKChain, WormBody
                if isinstance(WORLD.worm, WormBody):
                    WORLD.worm.chain = IKChain(
                        WORLD.worm.n_segments, WORLD.worm.segment_size,
                        WORLD.worm.target_x, WORLD.worm.target_y,
                        facing=WORLD.worm.facing_dir,
                    )
                # MuscleBody reconstructs its midline from `bend` each frame,
                # so no chain reseed needed.
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(ws)


# Static viewer files (main.js, etc.) — mounted last so /ws and / take precedence.
app.mount("/", StaticFiles(directory=str(VIEWER_DIR)), name="viewer")
