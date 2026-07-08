"""One-shot migration to the 2-flask x 20-worm layout with reset sigma.

What this script does, for flask_1 and flask_2 only:
  1. Loads the existing GenerationState (preserving parent_vector,
     generation counter, and best_score_history — 121+ gens of evolution
     we don't want to throw away).
  2. Resets state.sigma to SIGMA_INIT (currently 0.3) and saves.
  3. For every worm slot (Alice .. Trent, 20 names total) on each flask:
       - Ensures data/flasks/<flask>/<worm>/ exists with the right seed.
       - Regenerates that worm's weights.json from
             parent_vector + sigma_new * N(0, I)
         (the elite, worm 0, gets parent_vector exactly).
       - This means existing 6 worms (Alice..Frank) get FRESH children
         under the new sigma, and the 14 new worms (Grace..Trent) get
         born from the same parent. No worm keeps its old weights —
         the old children were spawned under sigma=2.4, way outside
         the new [0.05, 1.0] band, so keeping them would be cheating.

Existing flask_3..flask_6 directories are left untouched. The new server
config (WORMLET_N_FLASKS=2) just stops touching them; the historical data
stays on disk and in git for the generations viewer.

Run from the v6 directory:
    /home/web/.venv/bin/python scripts/migrate_to_2x20.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

V6 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V6))

from server.evolution import SIGMA_INIT, spawn_children  # noqa: E402
from server.orchestrator import FLASK_WORM_NAMES        # noqa: E402

DATA = V6 / "data"
FLASKS_DIR = DATA / "flasks"
GENS_DIR = DATA / "generations"
DEFAULT_WEIGHTS = V6 / "sim" / "weights.json"

TARGET_FLASKS = ["flask_1", "flask_2"]
N_WORMS = 20


def regenerate_flask(flask_name: str) -> None:
    state_path = GENS_DIR / flask_name / "state.json"
    if not state_path.exists():
        print(f"[{flask_name}] no state.json — skipping (cold-start will handle it)")
        return
    state = json.loads(state_path.read_text())

    old_sigma = state.get("sigma", 0.0)
    state["sigma"] = SIGMA_INIT
    state_path.write_text(json.dumps(state))
    print(f"[{flask_name}] sigma {old_sigma:.3f} -> {SIGMA_INIT:.3f}, "
          f"generation={state.get('generation')}, history_len={len(state.get('best_score_history', []))}")

    parent_vec = np.array(state["parent_vector"], dtype=np.float64)
    parent_keys = [tuple(k) for k in state["parent_keys"]]

    rng = np.random.default_rng(state.get("generation", 0) * 7919 + hash(flask_name) % 1000)
    children, _eps = spawn_children(parent_vec, n=N_WORMS, sigma=SIGMA_INIT, rng=rng)

    fi = int(flask_name.removeprefix("flask_")) - 1

    for wi, child_vec in enumerate(children):
        wname = FLASK_WORM_NAMES[wi]
        seed = fi * 1000 + wi + 1
        wdir = FLASKS_DIR / flask_name / wname
        wdir.mkdir(parents=True, exist_ok=True)

        seed_file = wdir / "seed.txt"
        seed_file.write_text(str(seed))

        rounded = np.round(child_vec).astype(int)
        weights: dict[str, dict[str, int]] = {}
        for (src, tgt), w in zip(parent_keys, rounded):
            weights.setdefault(src, {})[tgt] = int(w)
        (wdir / "weights.json").write_text(json.dumps(weights))

        # Reset poem so the new lineage starts clean against the full play.
        poem_path = wdir / "poem.txt"
        if poem_path.exists():
            import time as _t
            poem_path.rename(poem_path.with_name(f"poem.{int(_t.time())}.txt"))
        poem_path.write_text("")
        print(f"  wrote {wname} (seed={seed}) -> {len(weights)} src neurons")


def main() -> None:
    for fname in TARGET_FLASKS:
        regenerate_flask(fname)
    print("\nDone. Restart uvicorn so the new env (N_FLASKS=2, "
          "N_WORMS_PER_FLASK=20, PASSAGE=full) takes effect.")


if __name__ == "__main__":
    main()
