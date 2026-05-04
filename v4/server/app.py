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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sim.world import World

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
