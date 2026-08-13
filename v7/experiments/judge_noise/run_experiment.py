"""Experiment 1 runner — makes and caches every judge call.

Two independent bodies of calls, both resumable (re-running skips cached rows):

  ABSOLUTE  score every window of every worm, K_ABS times, at temp 0 and 1.
            (sampling x size subsets are re-aggregated offline in analyze.py)
  PAIRWISE  for a few representations, judge every worm pair in both orders,
            K_PW times, at temp 0.

Raw rows are appended to results/raw_abs.jsonl and results/raw_pw.jsonl.
Progress is mirrored to results/progress.json.

Run (from v7, with the API key sourced):
    PYTHONPATH=.:experiments/judge_noise python experiments/judge_noise/run_experiment.py
"""
from __future__ import annotations

import json
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pool_builder import POOLS, load_pool, represent, pool_id
from judges import absolute_batch, pairwise

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# --- knobs -------------------------------------------------------------------
K_ABS = 10                       # repeats for absolute (cheap)
K_PW = 8                         # repeats for pairwise (cost driver)
TEMPS_ABS = [0.0, 1.0]
PW_TEMP = 0.0                    # pairwise judged at temp 0 (temp axis studied on absolute)
PW_REPRS = [("random", 5), ("stratified", 5), ("all", None)]  # ("all", None) == every window
MAX_WORKERS = 16

_lock = threading.Lock()


def stable_seed(*parts) -> int:
    return zlib.crc32("::".join(map(str, parts)).encode())


def _append(path: Path, row: dict) -> None:
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _load_done(path: Path, keyfn) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:  # tolerate a truncated final line from an earlier kill
                done.add(keyfn(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# --- build the task lists ----------------------------------------------------

def build_abs_tasks(pools) -> list[dict]:
    tasks = []
    for (proc, flask, gen) in pools:
        pid = pool_id(proc, flask, gen)
        worms = load_pool(proc, flask, gen)
        for worm, wins in worms.items():
            for temp in TEMPS_ABS:
                for r in range(K_ABS):
                    tasks.append({"pool": pid, "worm": worm, "temp": temp, "rep": r,
                                  "_wins": wins})
    return tasks


def build_pw_tasks(pools) -> list[dict]:
    tasks = []
    for (proc, flask, gen) in pools:
        pid = pool_id(proc, flask, gen)
        worms = load_pool(proc, flask, gen)
        names = sorted(worms)
        for (sampling, size) in PW_REPRS:
            # Fix each worm's representation ONCE per config (stable across reps).
            repr_of = {}
            for n in names:
                if size is None:
                    repr_of[n] = sorted(worms[n], key=lambda w: w.idx)
                else:
                    rng = random.Random(stable_seed(pid, n, sampling, size))
                    repr_of[n] = represent(worms[n], sampling, size, rng)
            for r in range(K_PW):
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = names[i], names[j]
                        for order in ("ab", "ba"):
                            tasks.append({
                                "pool": pid, "sampling": sampling,
                                "size": (size if size is not None else "all"),
                                "rep": r, "a": a, "b": b, "order": order,
                                "_ra": repr_of[a], "_rb": repr_of[b],
                            })
    return tasks


# --- keys for resumability ---------------------------------------------------
def abs_key(row):
    return (row["pool"], row["worm"], row["temp"], row["rep"])


def pw_key(row):
    return (row["pool"], row["sampling"], row["size"], row["rep"], row["a"], row["b"], row["order"])


# --- workers -----------------------------------------------------------------
def run_abs(t):
    scores = absolute_batch(t["_wins"], t["temp"])
    return {"pool": t["pool"], "worm": t["worm"], "temp": t["temp"], "rep": t["rep"],
            "scores": {str(k): v for k, v in scores.items()}}


def run_pw(t):
    if t["order"] == "ab":
        winner_side = pairwise(t["_ra"], t["_rb"], PW_TEMP)
    else:
        winner_side = pairwise(t["_rb"], t["_ra"], PW_TEMP)
    return {"pool": t["pool"], "sampling": t["sampling"], "size": t["size"],
            "rep": t["rep"], "a": t["a"], "b": t["b"], "order": t["order"],
            "winner_side": winner_side}


def run_body(name, out_path, tasks, keyfn, worker):
    done = _load_done(out_path, keyfn)
    pending = [t for t in tasks if keyfn(t) not in done]
    total = len(tasks)
    print(f"[{name}] {total} total, {len(done)} cached, {len(pending)} to run", flush=True)
    n_done = len(done)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, t): t for t in pending}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                _append(out_path, fut.result())
            except Exception as e:
                print(f"[{name}] task failed: {e}", flush=True)
            n_done += 1
            if k % 100 == 0 or k == len(pending):
                rate = k / max(1e-9, time.time() - t0)
                eta = (len(pending) - k) / max(1e-9, rate)
                with _lock:
                    (RESULTS / "progress.json").write_text(json.dumps({
                        "body": name, "done": n_done, "total": total,
                        "rate_per_s": round(rate, 2), "eta_s": round(eta),
                    }))
                print(f"[{name}] {n_done}/{total}  {rate:.1f}/s  eta {eta/60:.1f}m", flush=True)


def main():
    print("building task lists...", flush=True)
    abs_tasks = build_abs_tasks(POOLS)
    pw_tasks = build_pw_tasks(POOLS)
    print(f"absolute calls: {len(abs_tasks)} | pairwise calls: {len(pw_tasks)} | "
          f"total {len(abs_tasks)+len(pw_tasks)}", flush=True)
    run_body("ABS", RESULTS / "raw_abs.jsonl", abs_tasks, abs_key, run_abs)
    run_body("PW", RESULTS / "raw_pw.jsonl", pw_tasks, pw_key, run_pw)
    (RESULTS / "DONE").write_text(str(time.time()))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
