"""Experiment 2 readout — compare the four σ-control schemes live.

Each poetry process runs one scheme on 2 replicate flasks. This reads the
per-generation metadata written since the A/B started (rows carrying a
`sigma_scheme` field) and summarises, per scheme:
  - σ trajectory: is it stabilising at an interior value, or still railed at
    SIGMA_MAX (runaway) / SIGMA_MIN (collapse)?
  - fitness trend: is best_score / mean actually CLIMBING over generations?
  - success_rate: is the 1/5 signal sitting near 0.2 (healthy) for the 1/5 schemes?

A scheme "wins" if σ settles interior AND fitness trends up. Run:
    python experiments/sigma_ab/readout.py
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path

V7 = Path(__file__).resolve().parent.parent.parent
PROC_SCHEME = {"poetry-1": "vs_mean (control)", "poetry-2": "vs_elite",
               "poetry-3": "xnes", "poetry-4": "sigma_anneal"}
SIGMA_MAX, SIGMA_MIN = 3.0, 0.02


def _trend(xs):
    """Crude slope sign over the series (last-half mean minus first-half mean)."""
    if len(xs) < 4:
        return "—"
    h = len(xs) // 2
    d = sum(xs[h:]) / len(xs[h:]) - sum(xs[:h]) / len(xs[:h])
    return f"{'↑' if d > 0.5 else '↓' if d < -0.5 else '→'} {d:+.1f}"


def load_flask(proc, flask):
    rows = []
    for g in sorted(glob.glob(str(V7 / "data" / proc / "generations" / flask / "gen-*"))):
        try:
            m = json.loads((Path(g) / "metadata.json").read_text())
        except Exception:
            continue
        if "sigma_scheme" in m:  # only post-A/B generations
            rows.append(m)
    return rows


def main():
    print("Experiment 2 — σ-control A/B readout\n" + "=" * 60)
    any_data = False
    for proc, label in PROC_SCHEME.items():
        print(f"\n### {proc}  —  {label}")
        for flask in ("flask_1", "flask_2"):
            rows = load_flask(proc, flask)
            if not rows:
                print(f"  {flask}: (no post-A/B generations yet)")
                continue
            any_data = True
            sig = [r.get("sigma_used", 0) for r in rows]
            best = [r.get("best_score", 0) for r in rows]
            srs = [r["success_rate"] for r in rows if r.get("success_rate") is not None]
            g0, g1 = rows[0]["generation"], rows[-1]["generation"]
            railed = "AT-MAX" if sig[-1] >= SIGMA_MAX * 0.98 else "AT-MIN" if sig[-1] <= SIGMA_MIN * 1.5 else "interior"
            print(f"  {flask}: gen {g0}->{g1} ({len(rows)} gens)")
            print(f"     σ:    {sig[-1]:.3f} [{railed}]   trend {_trend(sig)}   (recent {[round(s,2) for s in sig[-5:]]})")
            print(f"     best: {best[-1]:.1f}   trend {_trend(best)}   (recent {[round(b,1) for b in best[-5:]]})")
            if srs:
                print(f"     success_rate mean {sum(srs)/len(srs):.2f} (target 0.20)")
    if not any_data:
        print("\n(No generations have rolled over under the new schemes yet — each takes ~7h.\n"
              " Re-run in a day or two.)")
    else:
        print("\nPick the scheme whose σ is INTERIOR (not railed) and whose best/mean is trending ↑.")


if __name__ == "__main__":
    main()
