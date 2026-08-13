"""Publish each generation's winning worm to a static tree served at /board.

This replaces the old git-commit publish path. Why not git: multiple servers
pushing per-generation artifacts to one shared `main` branch raced (every push
after the first hit a non-fast-forward rejection) and bloated history with
~19M lines of JSON. Here each process writes its OWN subtree
(``board_publish/<process>/``), so writers never contend — the cross-server
race is structurally impossible, no lock and no rebase required.

The board polls ``board_publish/<process>/latest.json``, compares ``sha256``
(or ``epoch``), and fetches the referenced files only when they change.

The published JSON is byte-for-byte what the server used; making a board
*reproduce* the server's trajectory from it is the C-port's job (portable math),
designed separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

# Root of the served tree. Lives OUTSIDE the repo so git operations never touch
# it. All poetry processes share this parent and write disjoint children.
BOARD_PUBLISH_DIR = Path(
    os.environ.get("WORMLET_BOARD_PUBLISH_DIR", "/home/web/board_publish")
)

# This process's subtree name, e.g. "poetry-1". The systemd units set
# WORMLET_DATA_DIR=.../data/poetry-N, so its basename is a stable per-process id.
PROCESS_ID = Path(os.environ.get("WORMLET_DATA_DIR", "")).name or "default"

# How many generations to retain per process in the served tree.
KEEP_LAST = 20


def _atomic_replace(dst: Path, write: Callable[[Path], None]) -> None:
    """Write via a temp file in the same dir, then os.replace onto dst.
    Same-directory rename is atomic, so a reader never sees a partial file."""
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_copy(src: Path, dst: Path) -> bytes:
    """Atomically copy src -> dst. Returns the bytes copied (for hashing)."""
    data = src.read_bytes()
    _atomic_replace(dst, lambda t: t.write_bytes(data))
    return data


def _epoch_of(gen_dir: Path) -> int:
    """gen-0061 -> 61; unparseable -> -1 (sorts oldest, pruned first)."""
    try:
        return int(gen_dir.name.split("-")[1])
    except (IndexError, ValueError):
        return -1


def _prune(process_dir: Path, keep: int) -> None:
    gens = sorted(
        (d for d in process_dir.glob("gen-*") if d.is_dir()),
        key=_epoch_of,
    )
    for stale in gens[:-keep] if keep > 0 else []:
        shutil.rmtree(stale, ignore_errors=True)


def _update_index(root: Path, process_id: str, epoch: int, ts: str) -> None:
    index_path = root / "index.json"
    try:
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        index = {}
    if not isinstance(index, dict) or "processes" not in index:
        index = {"processes": {}}
    index["processes"][process_id] = {"epoch": epoch, "timestamp": ts}
    _atomic_replace(
        index_path,
        lambda t: t.write_text(json.dumps(index, indent=2, sort_keys=True)),
    )


def publish_board_bundle(
    process_id: str,
    generations_root: Path,
    epoch: int,
    winner_flask: str,
    winner_worm: str,
    winner_score: float,
    keepalive: Callable[[], None] | None = None,
) -> bool:
    """Copy epoch `epoch`'s winning worm into the served tree and repoint
    latest.json. Non-fatal: logs and returns False on any error, exactly like
    the git commit it replaces. Returns True only when latest.json is live and
    points at fully-written files.

    Source layout (written earlier in the rollover):
        <generations_root>/<winner_flask>/embedder.json            (shared genome)
        <generations_root>/<winner_flask>/gen-NNNN/<worm>/weights.json
        <generations_root>/<winner_flask>/gen-NNNN/<worm>/seed.txt
        <generations_root>/meta/gen-NNNN/winner.json
    """
    generations_root = Path(generations_root)
    gen_tag = f"gen-{epoch:04d}"
    flask_gen = generations_root / winner_flask / gen_tag / winner_worm

    src_embedder = generations_root / winner_flask / "embedder.json"
    src_weights = flask_gen / "weights.json"
    src_seed = flask_gen / "seed.txt"
    src_winner = generations_root / "meta" / gen_tag / "winner.json"

    # embedder + weights are the load-bearing genome; both must exist.
    missing = [str(p) for p in (src_embedder, src_weights) if not p.exists()]
    if missing:
        print(f"[BOARD_PUBLISH] {process_id} {gen_tag}: missing {missing}; skipped",
              flush=True)
        return False

    try:
        process_dir = BOARD_PUBLISH_DIR / process_id
        dst_gen = process_dir / gen_tag
        dst_gen.mkdir(parents=True, exist_ok=True)

        h = hashlib.sha256()
        h.update(_atomic_copy(src_embedder, dst_gen / "embedder.json"))
        h.update(_atomic_copy(src_weights, dst_gen / "weights.json"))
        if keepalive:
            keepalive()

        files = {
            "embedder": f"{gen_tag}/embedder.json",
            "weights": f"{gen_tag}/weights.json",
        }
        seed_val: int | None = None
        if src_seed.exists():
            raw = _atomic_copy(src_seed, dst_gen / "seed.txt")
            files["seed"] = f"{gen_tag}/seed.txt"
            try:
                seed_val = int(raw.decode().strip())
            except ValueError:
                seed_val = None
        if src_winner.exists():
            _atomic_copy(src_winner, dst_gen / "winner.json")
            files["winner"] = f"{gen_tag}/winner.json"

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest = {
            "process": process_id,
            "epoch": epoch,
            "winner": {
                "flask": winner_flask,
                "worm": winner_worm,
                "fitness": float(winner_score),
            },
            "seed": seed_val,
            "sha256": h.hexdigest(),
            "files": files,
            "timestamp": ts,
        }
        # latest.json is written LAST, atomically, so a poller only ever sees a
        # manifest whose referenced files are already fully on disk.
        _atomic_replace(
            process_dir / "latest.json",
            lambda t: t.write_text(json.dumps(manifest, indent=2)),
        )

        _prune(process_dir, KEEP_LAST)
        _update_index(BOARD_PUBLISH_DIR, process_id, epoch, ts)
        print(f"[BOARD_PUBLISH] {process_id} {gen_tag}: published "
              f"{winner_flask}/{winner_worm} ({winner_score:.3f})", flush=True)
        return True
    except Exception as e:  # non-fatal, like the old git path
        print(f"[BOARD_PUBLISH] {process_id} {gen_tag}: FAILED {e!r}", flush=True)
        return False


def publish_latest_on_disk(process_id: str, generations_root: Path) -> bool:
    """Publish whatever the newest committed generation on disk is. Used for the
    one-time backfill (so the tree is populated immediately instead of empty
    until the next ~7h rollover) and the module CLI."""
    generations_root = Path(generations_root)
    meta = generations_root / "meta"
    if not meta.exists():
        print(f"[BOARD_PUBLISH] {process_id}: no meta/ under {generations_root}",
              flush=True)
        return False
    gens = sorted((d for d in meta.glob("gen-*") if d.is_dir()), key=_epoch_of)
    for gen_dir in reversed(gens):  # newest first; skip gens missing winner.json
        winner_json = gen_dir / "winner.json"
        if not winner_json.exists():
            continue
        try:
            w = json.loads(winner_json.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        return publish_board_bundle(
            process_id=process_id,
            generations_root=generations_root,
            epoch=int(w["epoch"]),
            winner_flask=w["flask"],
            winner_worm=w["worm"],
            winner_score=float(w.get("fitness", 0.0)),
        )
    print(f"[BOARD_PUBLISH] {process_id}: no publishable generation found",
          flush=True)
    return False


if __name__ == "__main__":
    import sys

    # Usage: python -m server.board_publish <data_dir> [<data_dir> ...]
    # e.g.   python -m server.board_publish data/poetry-1 data/poetry-2 ...
    if len(sys.argv) < 2:
        print("usage: python -m server.board_publish <data_dir> [...]")
        raise SystemExit(2)
    ok_all = True
    for data_dir in sys.argv[1:]:
        pid = Path(data_dir).name or "default"
        groot = Path(data_dir) / "generations"
        ok = publish_latest_on_disk(pid, groot)
        ok_all = ok_all and ok
    raise SystemExit(0 if ok_all else 1)
