"""Run the v7 wormlet server.

    python main.py                  # default 127.0.0.1:8000, 6 worms
    python main.py --port 9000
    python main.py --n-worms 3
    WORMLET_DEBUG_SECRET=hunter2 python main.py
"""
from __future__ import annotations

import argparse
import os

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n-worms", type=int, default=None,
                    help="Override worm count (otherwise uses worms.json).")
    args = ap.parse_args()

    if args.n_worms is not None:
        os.environ["WORMLET_N_WORMS"] = str(args.n_worms)

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
