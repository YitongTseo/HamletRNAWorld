#!/usr/bin/env python3
"""Revive an archived lineage into a v7 flask so it resumes at generation N
instead of cold-starting at 0.

    revive_lineage.py --src v6/data/generations/flask_2 \
                      --data-dir v7/data/poetry-1 --flask flask_1 [--n-worms 16]

`--src` is any archived generations/<flask> tree holding gen-NNNN/ dirs with
metadata.json + per-worm weights.json (the v6 lineages tracked in git, or a v7
poetry-N/generations/<flask>). The newest generation there becomes the
resume point:

  <data-dir>/generations/<flask>/state.json   generation=N, parent = that
                                              generation's winner, sigma =
                                              SIGMA_INIT (or --sigma), history
                                              = every archived best_score
  <data-dir>/flasks/<flask>/<worm>/           weights.json + seed.txt for the
                                              v7 lineup (same-named archived
                                              worm, else the winner)
  <data-dir>/generations/<flask>/gen-NNNN/    trimmed copy (what the purge
                                              leaves: scores + winner genome)
                                              of the newest generation, or of
                                              EVERY archived one with --all-gens
                                              so /generations shows the whole
                                              history
  <data-dir>/generations/<flask>/provenance.json

--archive-only skips state.json and the live worm dirs (safe while the
process is running) and only lays down the trimmed gen-NNNN copies.

Stop the owning wormlet process first; start it after. The archived genomes
have no _lifelike block — migrate_genome_layout appends it on the first
rollover of a flask with the lifelike flags on, as designed for pre-lifelike
lineages. Fitness regimes changed on 2026-08-15, so the carried-over score
history is the lineage's record, not a series comparable with new scores."""
from __future__ import annotations

import argparse, json, shutil, sys, time
from pathlib import Path

V7 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V7))
from server.evolution import flatten_weights            # noqa: E402
from server.generations import SIGMA_INIT               # noqa: E402
from server.orchestrator import FLASK_WORM_NAMES        # noqa: E402

KEEP_WORM = ("fitness.json", "seed.txt", "scores.jsonl")
KEEP_WINNER = ("weights.json", "poem_clean.txt")
KEEP_FLASK = ("metadata.json", "selection.json")


def gen_dirs(src: Path) -> list[tuple[int, Path]]:
    out = []
    for d in src.iterdir():
        if d.is_dir() and d.name.startswith("gen-") and (d / "metadata.json").exists():
            out.append((int(d.name[4:]), d))
    return sorted(out)


def copy_archive(gens: list[tuple[int, Path]], gen_root: Path) -> int:
    """Lay a trimmed copy of each (n, gen_dir) under gen_root: the files the
    post-commit purge leaves behind (scores, per-worm fitness, flask metadata)
    plus the winner's genome + poem. Skips a gen whose source IS the target.
    Byte-identical copies, so git stores them as the blobs it already has."""
    n_copied = 0
    for _, gen_dir in gens:
        dst_gen = gen_root / gen_dir.name
        if gen_dir.resolve() == dst_gen.resolve():
            continue
        meta = json.loads((gen_dir / "metadata.json").read_text())
        winner = (meta.get("ranks") or [None])[0]
        if dst_gen.exists():
            shutil.rmtree(dst_gen)
        dst_gen.mkdir(parents=True)
        for f in KEEP_FLASK:
            if (gen_dir / f).exists():
                shutil.copy2(gen_dir / f, dst_gen / f)
        for wd in gen_dir.iterdir():
            if not wd.is_dir():
                continue
            (dst_gen / wd.name).mkdir()
            keep = KEEP_WORM + (KEEP_WINNER if wd.name == winner else ())
            for f in keep:
                if (wd / f).exists():
                    shutil.copy2(wd / f, dst_gen / wd.name / f)
        n_copied += 1
    return n_copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--flask", required=True)
    ap.add_argument("--n-worms", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--all-gens", action="store_true",
                    help="copy a trimmed record of every archived generation, not just the newest")
    ap.add_argument("--archive-only", action="store_true",
                    help="only copy gen-NNNN records; leave state.json and live worms alone")
    a = ap.parse_args()

    gens = gen_dirs(a.src)
    if not gens:
        print(f"no gen-NNNN dirs with metadata.json under {a.src}", file=sys.stderr)
        return 1
    N, gen_dir = gens[-1]
    meta = json.loads((gen_dir / "metadata.json").read_text())
    ranks = meta.get("ranks") or []
    winner = ranks[0] if ranks else None
    if not winner or not (gen_dir / winner / "weights.json").exists():
        print(f"{gen_dir}: no winner weights (ranks={ranks[:3]})", file=sys.stderr)
        return 1
    history = []
    for g, d in gens:
        try:
            history.append(float(json.loads((d / "metadata.json").read_text())["best_score"]))
        except Exception:
            history.append(0.0)

    winner_w = json.loads((gen_dir / winner / "weights.json").read_text())
    winner_w.pop("_lifelike", None)         # retrofitted on the first rollover
    vec, keys = flatten_weights(winner_w)
    sigma = a.sigma if a.sigma is not None else SIGMA_INIT

    gen_root = a.data_dir / "generations" / a.flask
    gen_root.mkdir(parents=True, exist_ok=True)
    if a.archive_only:
        n_copied = copy_archive(gens if a.all_gens else gens[-1:], gen_root)
        print(f"{a.data_dir.name}/{a.flask}: archived {n_copied} generation record(s) "
              f"from {a.src} (gen {gens[0][0]}..{N}); state + worms untouched")
        return 0
    state = {
        "group_name": a.flask,
        "generation": N,
        "sigma": sigma,
        "best_score_history": history,
        "parent_vector": vec.tolist(),
        "parent_keys": [list(k) for k in keys],
        "children": None,            # first rollover reconstructs eps from genomes
        "prev_fresh_mean": None,
    }
    (gen_root / "state.json").write_text(json.dumps(state))

    # Live worm dirs: the v7 lineup, each carrying its same-named archived
    # genome (or the winner's), fresh poem, no checkpoint.
    names = FLASK_WORM_NAMES[: a.n_worms]
    flask_dir = a.data_dir / "flasks" / a.flask
    if flask_dir.exists():
        shutil.rmtree(flask_dir)
    n_same = 0
    for i, name in enumerate(names):
        wdir = flask_dir / name
        wdir.mkdir(parents=True)
        src_w = gen_dir / name / "weights.json"
        if src_w.exists():
            w = json.loads(src_w.read_text()); w.pop("_lifelike", None); n_same += 1
        else:
            w = winner_w
        (wdir / "weights.json").write_text(json.dumps(w))
        src_seed = gen_dir / name / "seed.txt"
        (wdir / "seed.txt").write_text(src_seed.read_text() if src_seed.exists() else str(1000 + i))

    # Trimmed archive of the resume generation (or of every archived one).
    copy_archive(gens if a.all_gens else gens[-1:], gen_root)

    (gen_root / "provenance.json").write_text(json.dumps({
        "revived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(a.src),
        "resumed_generation": N,
        "parent_from": f"{gen_dir.name}/{winner}",
        "archived_best_score": meta.get("best_score"),
        "archived_judge_model": meta.get("judge_model"),
        "archived_sigma_next": meta.get("sigma_next"),
        "sigma_used": sigma,
        "note": a.note,
    }, indent=1))
    print(f"{a.data_dir.name}/{a.flask}: resumes at gen {N} from {a.src} "
          f"(parent={winner} {meta.get('best_score', 0):.1f}, {len(vec)} genes, "
          f"{n_same}/{len(names)} worms carry their own archived genome, sigma={sigma})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
