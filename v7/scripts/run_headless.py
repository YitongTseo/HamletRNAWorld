#!/usr/bin/env python3
"""Run the simulation with no HTTP server.

The control runs (`nouns`, `pos_chain`, ...) are scored by a deterministic POS
tagger, so nobody needs to watch them in a browser — the answer is in the data
dir. Driving the FastAPI lifespan directly means the whole run needs neither
uvicorn nor a listening socket, which is what lets a control run on the host
beside the jail instead of inside it.

This is NOT a second copy of the startup logic: it enters the very same
`lifespan()` the server does, so the worms, flasks, embedders, checkpointing and
rollover barrier are wired identically. If app.lifespan changes, this follows.

    v7/.venv/bin/python scripts/run_headless.py

Ctrl-C / SIGTERM shuts down through the same lifespan exit path as the server,
so the final checkpoint is written.
"""
from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> int:
    from server.app import app  # imported late: env must be set before app import

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # The same context manager uvicorn enters. Startup spawns sim_loop as a
    # task; exit runs the shutdown half, which flushes the last checkpoint.
    async with app.router.lifespan_context(app):
        print("headless sim running — Ctrl-C to stop", flush=True)
        await stop.wait()
        print("shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
