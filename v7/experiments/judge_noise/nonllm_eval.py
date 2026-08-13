"""Do we need the LLM to choose which windows represent a worm?

The absolute run already scored EVERY window K times at temp 0, so we can test any
window-selection rule for FREE by re-aggregating — no new LLM calls. Compare:
  random        - uniform (zero cost, zero info)
  llm_strat     - stratify by HISTORICAL LLM quality score (needs an LLM upfront)
  ttr_strat     - stratify by type-token ratio (lexical diversity; classical)
  rep_strat     - stratify by the engine's repetition_factor (classical)
  diversity     - farthest-point on token Jaccard (max coverage; classical)
All at m=5, temp 0, across the 4 pools. Metric: mean Kendall tau (rank stability).
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from pool_builder import POOLS, load_pool, pool_id
from run_experiment import stable_seed, K_ABS
from analyze import _q, mean_pairwise_tau
from server.evolution import repetition_factor

RESULTS = Path(__file__).resolve().parent / "results"
M = 5

# --- selection rules: (windows, m, rng) -> chosen subset (list of idx) -------

def _by_idx(ws):
    return [w.idx for w in sorted(ws, key=lambda w: w.idx)]

def sel_random(ws, m, rng):
    return _by_idx(rng.sample(ws, m) if m < len(ws) else ws)

def _stratify(ws, m, key, rng):
    if m >= len(ws):
        return _by_idx(ws)
    ordered = sorted(ws, key=key)
    n = len(ordered)
    picked = []
    for b in range(m):
        lo, hi = (b * n) // m, ((b + 1) * n) // m
        hi = max(hi, lo + 1)
        picked.append(ordered[rng.randrange(lo, min(hi, n))])
    return _by_idx(picked)

def sel_llm_strat(ws, m, rng):
    return _stratify(ws, m, lambda w: w.hist_q, rng)

def sel_ttr_strat(ws, m, rng):
    return _stratify(ws, m, lambda w: len(set(w.tokens)) / max(1, len(w.tokens)), rng)

def sel_rep_strat(ws, m, rng):
    return _stratify(ws, m, lambda w: repetition_factor(list(w.tokens)), rng)

def sel_diversity(ws, m, rng):
    """Farthest-point sampling on 1 - Jaccard(token sets) — maximal coverage."""
    if m >= len(ws):
        return _by_idx(ws)
    sets = [set(w.tokens) for w in ws]
    start = rng.randrange(len(ws))
    chosen = [start]
    while len(chosen) < m:
        best, best_d = None, -1.0
        for i in range(len(ws)):
            if i in chosen:
                continue
            d = min(1 - len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for j in chosen)
            if d > best_d:
                best_d, best = d, i
        chosen.append(best)
    return _by_idx([ws[i] for i in chosen])

RULES = {"random": sel_random, "llm_strat": sel_llm_strat, "ttr_strat": sel_ttr_strat,
         "rep_strat": sel_rep_strat, "diversity": sel_diversity}


def main():
    # temp-0 absolute scores: (pool, worm, rep) -> {idx:(e,c)}
    scores = {}
    for l in (RESULTS / "raw_abs.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r["temp"] == 0.0:
            scores[(r["pool"], r["worm"], r["rep"])] = {int(k): tuple(v) for k, v in r["scores"].items()}

    per_pool_tau = defaultdict(list)  # rule -> [tau per pool]
    for (proc, flask, gen) in POOLS:
        pid = pool_id(proc, flask, gen)
        worms = load_pool(proc, flask, gen)
        names = sorted(worms)
        for rule, fn in RULES.items():
            subset = {}
            for n in names:
                rng = random.Random(stable_seed(pid, n, rule, M))
                subset[n] = fn(worms[n], M, rng)
            vectors = []
            for rep in range(K_ABS):
                vec = []
                for n in names:
                    sc = scores.get((pid, n, rep), {})
                    vec.append(sum(_q(*sc[i]) for i in subset[n] if i in sc))
                vectors.append(vec)
            tau, _ = mean_pairwise_tau(vectors)
            per_pool_tau[rule].append(tau)

    print(f"Non-LLM window selection @ m={M}, temp=0, 4 pools\n")
    print(f"  {'rule':12s} {'per-pool tau':40s} {'mean':>6s} {'sd':>6s}")
    base = np.array(per_pool_tau["random"])
    for rule in RULES:
        v = np.array(per_pool_tau[rule])
        print(f"  {rule:12s} {str([round(x,3) for x in v]):40s} {v.mean():6.3f} {v.std():6.3f}")
    print("\n  paired-t vs random (does any beat plain random?):")
    for rule in RULES:
        if rule == "random":
            continue
        v = np.array(per_pool_tau[rule]); d = v - base
        p = ttest_rel(v, base).pvalue
        print(f"    {rule:12s} Δ={d.mean():+.3f}  same-sign={bool(np.all(np.sign(d)==np.sign(d[0])))}  p={p:.3f}")


if __name__ == "__main__":
    main()
