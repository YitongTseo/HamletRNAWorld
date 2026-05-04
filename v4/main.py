"""Run the v4 worm sim + viewer.

  python main.py            # default: http://127.0.0.1:8000
  python main.py --port 9000
"""
from __future__ import annotations

import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
