"""Multi-worm orchestrator: owns N Worm instances, ticks them all at 60 Hz,
persists poems to disk, and exposes broadcast hooks for the WS layer.

Per-worm data layout:
    data/worms/<name>/
        seed.txt        # the int seed used to start this worm
        weights.json    # copy of sim/weights.json (frozen in v6)
        poem.txt        # append-only, one eaten word per line
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable

from sim.world import World

V6_ROOT = Path(__file__).resolve().parent.parent

# Override with WORMLET_DATA_DIR to run a dev instance whose poems / weights
# don't pollute prod. Default keeps existing v6/data layout intact.
_DATA_ROOT = Path(os.environ.get("WORMLET_DATA_DIR", V6_ROOT / "data"))
DATA_DIR = _DATA_ROOT / "worms"
# Config file lives next to the data dir so a dev instance can use its own
# (smaller, different-seeded) worm lineup.
WORMS_CONFIG = _DATA_ROOT / "worms.json" if _DATA_ROOT != V6_ROOT / "data" else V6_ROOT / "worms.json"
DEFAULT_WEIGHTS = V6_ROOT / "sim" / "weights.json"


@dataclass
class Worm:
    name: str
    seed: int
    world: World
    poem_path: Path
    poem_file: IO = field(repr=False)
    word_count: int = 0
    recent_words: list[str] = field(default_factory=list)  # tail of poem

    def close(self) -> None:
        try:
            self.poem_file.close()
        except Exception:
            pass


def _load_or_create_config(n_worms: int | None) -> list[dict]:
    """Load worms.json. If it's missing, create the default 6-worm lineup.
    If n_worms is given, truncate/extend the list to match."""
    DEFAULT_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
                     "Grace", "Heidi", "Ivan", "Judy", "Kara", "Liam"]
    if not WORMS_CONFIG.exists():
        default = [{"name": n, "seed": i + 1} for i, n in enumerate(DEFAULT_NAMES[:6])]
        WORMS_CONFIG.write_text(json.dumps(default, indent=2))
    cfg = json.loads(WORMS_CONFIG.read_text())
    if n_worms is not None and n_worms != len(cfg):
        if n_worms <= len(cfg):
            cfg = cfg[:n_worms]
        else:
            for i in range(len(cfg), n_worms):
                name = DEFAULT_NAMES[i % len(DEFAULT_NAMES)]
                cfg.append({"name": name, "seed": i + 1})
    return cfg


def _ensure_worm_dir(name: str, seed: int) -> tuple[Path, dict]:
    """Create data/worms/<name>/ if needed; return (dir, weights)."""
    wdir = DATA_DIR / name
    wdir.mkdir(parents=True, exist_ok=True)
    seed_file = wdir / "seed.txt"
    if not seed_file.exists():
        seed_file.write_text(str(seed))
    weights_path = wdir / "weights.json"
    if not weights_path.exists():
        shutil.copyfile(DEFAULT_WEIGHTS, weights_path)
    weights = json.loads(weights_path.read_text())
    return wdir, weights


def _tail_words(path: Path, n: int) -> list[str]:
    """Cheap tail-N for the recent_words display. For huge files we'd want
    a smarter approach, but poems grow at a few words per minute so this is
    fine for now."""
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return []
    return [l for l in lines[-n:] if l]


def load_worms(n_worms: int | None = None) -> list[Worm]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_or_create_config(n_worms)
    worms: list[Worm] = []
    for entry in cfg:
        name = entry["name"]
        seed = int(entry["seed"])
        wdir, weights = _ensure_worm_dir(name, seed)
        world = World(seed=seed, weights=weights)
        poem_path = wdir / "poem.txt"
        # Initialize word count from existing poem (if any).
        existing = _tail_words(poem_path, n=100_000)  # full file; cheap for our sizes
        # Re-read for accurate count, separately, since _tail_words trims:
        if poem_path.exists():
            with open(poem_path) as f:
                word_count = sum(1 for line in f if line.strip())
        else:
            word_count = 0
        recent_words = _tail_words(poem_path, n=20)
        pf = open(poem_path, "a", buffering=1)  # line-buffered
        worms.append(Worm(
            name=name, seed=seed, world=world,
            poem_path=poem_path, poem_file=pf,
            word_count=word_count, recent_words=recent_words,
        ))
    return worms


def drain_and_persist(worm: Worm) -> Iterable[str]:
    """Drain newly-eaten words, append to disk, update bookkeeping. Yields
    each new word so the caller can broadcast them."""
    for line_id, word_idx, word in worm.world.drain_eaten_words():
        worm.poem_file.write(word + "\n")
        worm.word_count += 1
        worm.recent_words.append(word)
        if len(worm.recent_words) > 20:
            worm.recent_words = worm.recent_words[-20:]
        yield word


def reset_worm(worm: Worm, archive_poem: bool = True) -> None:
    """Re-seed a worm from its seed.txt and (optionally) archive the old
    poem. Called from the debug door."""
    import time as _time
    if archive_poem and worm.poem_path.exists():
        worm.poem_file.close()
        ts = int(_time.time())
        worm.poem_path.rename(worm.poem_path.with_name(f"poem.{ts}.txt"))
        worm.poem_file = open(worm.poem_path, "a", buffering=1)
    weights = json.loads((worm.poem_path.parent / "weights.json").read_text())
    worm.world = World(seed=worm.seed, weights=weights)
    worm.word_count = 0
    worm.recent_words = []
