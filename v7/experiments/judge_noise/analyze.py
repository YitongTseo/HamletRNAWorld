"""Experiment 1 analysis — turn raw judge rows into ranking-stability metrics.

Reads results/raw_abs.jsonl + results/raw_pw.jsonl, produces results/metrics.json
(list of per-config records, averaged across the 4 pools) and metrics.csv.

Config record fields:
  method      "absolute" | "pairwise"
  sampling    "random" | "stratified" | "all"
  size        5 | 15 | "all"
  temp        0.0 | 1.0            (absolute only; pairwise fixed at 0.0)
  mean_tau    primary: mean Kendall tau between independent repeats' rankings
  mean_rho    mean Spearman rho (same)
  sn_ratio    signal-to-noise = between-worm SD / mean within-worm repeat SD (absolute)
  pos_bias    P(judge picks the first-shown side)  (pairwise)
  flip_rate   fraction of pairs whose winner flips when order is swapped (pairwise)
  calls       judge calls per pool per full ranking pass
  validity    Spearman of this config's consensus ranking vs the grand consensus
"""
from __future__ import annotations

import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from pool_builder import POOLS, load_pool, represent, pool_id
from judges import bradley_terry
from run_experiment import stable_seed, K_ABS, K_PW, TEMPS_ABS, PW_REPRS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ABS_SAMPLINGS = ["random", "stratified"]
ABS_SIZES = [5, 15, "all"]


def _q(e: int, c: int) -> float:
    """Per-window quality in the fitness shape (GAMMA=1.5, emotional weight 1.5)."""
    return 1.5 * (e / 100.0) ** 1.5 + (c / 100.0) ** 1.5


def load_raw():
    abs_rows = [json.loads(l) for l in (RESULTS / "raw_abs.jsonl").read_text().splitlines() if l.strip()]
    pw_rows = [json.loads(l) for l in (RESULTS / "raw_pw.jsonl").read_text().splitlines() if l.strip()]
    return abs_rows, pw_rows


def mean_pairwise_tau(rank_vectors: list[list[float]]) -> tuple[float, float]:
    """Mean Kendall tau and Spearman rho over all pairs of repeat score-vectors."""
    taus, rhos = [], []
    for a, b in itertools.combinations(rank_vectors, 2):
        t = kendalltau(a, b).statistic
        r = spearmanr(a, b).statistic
        if not np.isnan(t):
            taus.append(t)
        if not np.isnan(r):
            rhos.append(r)
    return (float(np.mean(taus)) if taus else float("nan"),
            float(np.mean(rhos)) if rhos else float("nan"))


# --- absolute ---------------------------------------------------------------

def analyze_absolute(abs_rows):
    """Returns per (pool, sampling, size, temp) -> dict of metrics + the per-repeat
    worm-score vectors (for consensus)."""
    # index scores: (pool, worm, temp, rep) -> {idx: (e,c)}
    idx = {}
    for r in abs_rows:
        idx[(r["pool"], r["worm"], r["temp"], r["rep"])] = {int(k): tuple(v) for k, v in r["scores"].items()}

    out = {}
    for (proc, flask, gen) in POOLS:
        pid = pool_id(proc, flask, gen)
        worms = load_pool(proc, flask, gen)
        names = sorted(worms)
        for sampling in ABS_SAMPLINGS:
            for size in ABS_SIZES:
                for temp in TEMPS_ABS:
                    # Production-faithful: RESAMPLE the window subset every repeat
                    # (fresh seed per rep), so reproducibility includes the variance
                    # of WHICH windows are drawn — not just judge noise on a fixed
                    # set. (size=="all" has no resampling variance.)
                    vectors = []
                    per_worm_scores = defaultdict(list)  # worm -> [agg per rep]
                    ok = True
                    for rep in range(K_ABS):
                        vec = []
                        for n in names:
                            sc = idx.get((pid, n, temp, rep))
                            if sc is None:
                                ok = False
                                break
                            if size == "all":
                                sub = [w.idx for w in worms[n]]
                            else:
                                rng = random.Random(stable_seed(pid, n, sampling, size, rep))
                                sub = [w.idx for w in represent(worms[n], sampling, size, rng)]
                            agg = sum(_q(*sc[i]) for i in sub if i in sc)
                            vec.append(agg)
                            per_worm_scores[n].append(agg)
                        if not ok:
                            break
                        vectors.append(vec)
                    if not ok or len(vectors) < 2:
                        continue
                    tau, rho = mean_pairwise_tau(vectors)
                    # signal-to-noise: between-worm SD of mean scores / mean within-worm SD
                    means = np.array([np.mean(per_worm_scores[n]) for n in names])
                    within = np.array([np.std(per_worm_scores[n]) for n in names])
                    sn = float(np.std(means) / (np.mean(within) + 1e-12))
                    consensus = means  # mean score per worm = best estimate of "true" order
                    out[(pid, sampling, size, temp)] = {
                        "method": "absolute", "pool": pid, "sampling": sampling,
                        "size": size, "temp": temp, "mean_tau": tau, "mean_rho": rho,
                        "sn_ratio": sn, "calls": len(names) * K_ABS,  # 1 batched call/worm/rep
                        "_consensus": {n: float(consensus[k]) for k, n in enumerate(names)},
                    }
    return out


# --- pairwise ---------------------------------------------------------------

def analyze_pairwise(pw_rows):
    # index: (pool, sampling, size, rep) -> list of rows
    grp = defaultdict(list)
    for r in pw_rows:
        grp[(r["pool"], r["sampling"], str(r["size"]), r["rep"])].append(r)

    out = {}
    for (proc, flask, gen) in POOLS:
        pid = pool_id(proc, flask, gen)
        names = sorted(load_pool(proc, flask, gen))
        for (sampling, size) in PW_REPRS:
            skey = "all" if size is None else str(size)
            samp = sampling
            vectors = []
            bias_hits = bias_n = flips = flip_n = 0
            per_worm_strength = defaultdict(list)
            for rep in range(K_PW):
                rows = grp.get((pid, samp, skey, rep), [])
                if not rows:
                    continue
                # winner side per (a,b,order)
                res = {(r["a"], r["b"], r["order"]): r["winner_side"] for r in rows}
                wins = defaultdict(int)  # (winner, loser) -> games
                for a, b in itertools.combinations(names, 2):
                    ab = res.get((a, b, "ab"))
                    ba = res.get((a, b, "ba"))
                    if ab is None or ba is None:
                        continue
                    # position bias: did the judge pick the first side (A)?
                    bias_hits += (ab == "A") + (ba == "A"); bias_n += 2
                    # debias: a wins from ab if 'A', from ba if 'B'
                    a_wins = (ab == "A") + (ba == "B")
                    b_wins = (ab == "B") + (ba == "A")
                    wins[(a, b)] += a_wins
                    wins[(b, a)] += b_wins
                    # flip: swapping order changes the actual winner
                    win_ab = a if ab == "A" else b
                    win_ba = b if ba == "A" else a  # in 'ba', A=b
                    flips += (win_ab != win_ba); flip_n += 1
                bt = bradley_terry(names, wins)
                vec = [bt[n] for n in names]
                vectors.append(vec)
                for n in names:
                    per_worm_strength[n].append(bt[n])
            if len(vectors) < 2:
                continue
            tau, rho = mean_pairwise_tau(vectors)
            consensus = {n: float(np.mean(per_worm_strength[n])) for n in names}
            out[(pid, samp, skey)] = {
                "method": "pairwise", "pool": pid, "sampling": samp, "size": skey,
                "temp": 0.0, "mean_tau": tau, "mean_rho": rho,
                "pos_bias": bias_hits / max(1, bias_n),
                "flip_rate": flips / max(1, flip_n),
                "calls": (len(names) * (len(names) - 1) // 2) * 2 * K_PW,  # pairs*orders*reps
                "_consensus": consensus,
            }
    return out


def aggregate_across_pools(records: list[dict]) -> list[dict]:
    """Collapse the per-pool records into one row per config, mean +/- sd over pools."""
    groups = defaultdict(list)
    for r in records:
        key = (r["method"], r["sampling"], str(r["size"]), r.get("temp"))
        groups[key].append(r)
    agg = []
    for key, rs in groups.items():
        method, sampling, size, temp = key
        row = {"method": method, "sampling": sampling, "size": size, "temp": temp,
               "n_pools": len(rs), "calls": rs[0]["calls"]}
        for m in ("mean_tau", "mean_rho", "sn_ratio", "pos_bias", "flip_rate", "validity"):
            vals = [r[m] for r in rs if m in r and r[m] == r[m]]
            if vals:
                row[m] = float(np.mean(vals))
                row[m + "_sd"] = float(np.std(vals))
        agg.append(row)
    agg.sort(key=lambda r: (-r.get("mean_tau", 0)))
    return agg


def main():
    abs_rows, pw_rows = load_raw()
    print(f"loaded {len(abs_rows)} abs rows, {len(pw_rows)} pw rows")
    a = analyze_absolute(abs_rows)
    p = analyze_pairwise(pw_rows)
    records = list(a.values()) + list(p.values())

    # validity: per pool, grand consensus = mean rank across all configs; correlate.
    by_pool = defaultdict(list)
    for r in records:
        by_pool[r["pool"]].append(r)
    for pool, rs in by_pool.items():
        names = sorted(rs[0]["_consensus"].keys())
        # rank each config's consensus, average ranks -> grand order
        rank_mat = []
        for r in rs:
            order = np.argsort(np.argsort([r["_consensus"][n] for n in names]))
            rank_mat.append(order)
        grand = np.mean(rank_mat, axis=0)
        for r in rs:
            v = [r["_consensus"][n] for n in names]
            r["validity"] = float(spearmanr(v, grand).statistic)

    for r in records:
        r.pop("_consensus", None)

    agg = aggregate_across_pools(records)
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"per_pool": records, "aggregated": agg}, indent=2))
    # CSV of the aggregated view
    cols = ["method", "sampling", "size", "temp", "mean_tau", "mean_tau_sd",
            "mean_rho", "sn_ratio", "pos_bias", "flip_rate", "validity", "calls", "n_pools"]
    with open(RESULTS / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in agg:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k, "")) for k in cols})
    print(f"wrote metrics.json ({len(records)} per-pool records) and metrics.csv "
          f"({len(agg)} configs)")
    # quick console peek
    print("\ntop configs by mean_tau:")
    for r in agg[:8]:
        print(f"  {r['method']:8s} samp={r['sampling']:10s} size={str(r['size']):4s} "
              f"temp={r['temp']}  tau={r.get('mean_tau', float('nan')):.3f}  "
              f"calls={r['calls']}")


if __name__ == "__main__":
    main()
