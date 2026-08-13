"""Does the '5 windows beats all' result survive RESAMPLING the windows each rep?

analyze.py held each worm's window subset FIXED across the K repeats, so its
reproducibility = judge noise on a fixed input. Production RESAMPLES windows every
generation, so the honest measure must draw fresh windows each rep — which adds
sampling variance that hits small m hardest. This recomputes tau both ways.
Free: raw_abs already scored every window K times at temp 0.
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path
import numpy as np

from pool_builder import POOLS, load_pool, pool_id
from run_experiment import stable_seed, K_ABS
from analyze import _q, mean_pairwise_tau

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    scores = {}
    for l in (RESULTS / "raw_abs.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r["temp"] == 0.0:
            scores[(r["pool"], r["worm"], r["rep"])] = {int(k): tuple(v) for k, v in r["scores"].items()}

    sizes = [5, 15, "all"]
    fixed = defaultdict(list)      # size -> [tau per pool]
    resamp = defaultdict(list)
    for (proc, flask, gen) in POOLS:
        pid = pool_id(proc, flask, gen)
        worms = load_pool(proc, flask, gen)
        names = sorted(worms)
        idxs = {n: [w.idx for w in worms[n]] for n in names}
        for size in sizes:
            m = {n: (len(idxs[n]) if size == "all" else size) for n in names}
            # --- FIXED: one subset per worm, reused every rep ---
            fixed_sub = {}
            for n in names:
                rng = random.Random(stable_seed(pid, n, "random", size))
                fixed_sub[n] = idxs[n] if size == "all" else sorted(rng.sample(idxs[n], min(m[n], len(idxs[n]))))
            vecs_f = []
            for rep in range(K_ABS):
                vecs_f.append([sum(_q(*scores[(pid, n, rep)][i]) for i in fixed_sub[n]
                                   if i in scores[(pid, n, rep)]) for n in names])
            fixed[size].append(mean_pairwise_tau(vecs_f)[0])
            # --- RESAMPLED: fresh subset per worm per rep ---
            vecs_r = []
            for rep in range(K_ABS):
                vec = []
                for n in names:
                    rng = random.Random(stable_seed(pid, n, "random", size, rep))
                    sub = idxs[n] if size == "all" else sorted(rng.sample(idxs[n], min(m[n], len(idxs[n]))))
                    sc = scores[(pid, n, rep)]
                    vec.append(sum(_q(*sc[i]) for i in sub if i in sc))
                vecs_r.append(vec)
            resamp[size].append(mean_pairwise_tau(vecs_r)[0])

    print("temp 0, random windows, 4 pools — mean Kendall tau\n")
    print(f"  {'m':>4s}  {'FIXED windows':>14s}  {'RESAMPLED windows':>18s}   (resampled = production-faithful)")
    for size in sizes:
        f = np.mean(fixed[size]); r = np.mean(resamp[size])
        print(f"  {str(size):>4s}  {f:>14.3f}  {r:>18.3f}   Δ={r-f:+.3f}")
    print("\n  If RESAMPLED flips the order (all >= 5), the '5 wins' result was a fixed-window artifact.")


if __name__ == "__main__":
    main()
