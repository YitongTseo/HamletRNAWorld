"""Multi-worm orchestrator: owns N Worm instances, ticks them all at 60 Hz,
persists poems to disk, and exposes broadcast hooks for the WS layer.

Per-worm data layout (legacy single-group mode):
    data/worms/<name>/
        seed.txt        # the int seed used to start this worm
        weights.json    # copy of sim/weights.json (frozen in v6)
        poem.txt        # append-only, one eaten word per line

Multi-flask layout (generational evolution mode):
    data/flasks/<flask_name>/<worm_name>/
        seed.txt
        weights.json
        poem.txt
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable

from sim.world import World
from sim import lifelike

V6_ROOT = Path(__file__).resolve().parent.parent

# Override with WORMLET_DATA_DIR to run a dev instance whose poems / weights
# don't pollute prod. Default keeps existing v6/data layout intact.
_DATA_ROOT = Path(os.environ.get("WORMLET_DATA_DIR", V6_ROOT / "data"))

# v7.1: base seed for each flask's independent embedder lineage. Each flask
# derives its own seed (base + flask index) so the 8 embedders diverge.
_EMBEDDING_SEED = int(os.environ.get("WORMLET_EMBEDDING_SEED", "0"))

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
    # v7.1: this worm's fitness from the most recent rollover (set by the
    # rollover functions; read to score the flask's shared embedder candidate).
    last_fitness: float = 0.0

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


# ----- Multi-flask loader ---------------------------------------------------
# 20 worm names per flask. Each flask gets the same names; uniqueness comes
# from the (flask, name) pair. Seeds are unique across all flasks so worm
# trajectories don't repeat across flasks. The list MUST be at least as long
# as WORMLET_N_WORMS_PER_FLASK — if it's shorter, load_flasks cycles names
# and multiple worms collide onto one directory (shared poem.txt/weights.json).
FLASK_WORM_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve",
                    "Frank", "Grace", "Heidi", "Ivan", "Judy",
                    "Kara", "Liam", "Mallory", "Niaj", "Oscar",
                    "Peggy", "Quentin", "Ruth", "Sybil", "Trent"]
# WORMLET_FLASK_WORM_NAMES: comma-separated override of the lineup above, so a
# deployment can name its own worms without editing code (legacy no-generations
# mode already allows this via worms.json; flask mode had no equivalent). The
# same collision rule applies — load_flasks still raises if the list is shorter
# than WORMLET_N_WORMS_PER_FLASK. Names become directory names; spaces are fine.
_ENV_NAMES = os.environ.get("WORMLET_FLASK_WORM_NAMES", "").strip()
# ';' splits per-flask groups (flask 1 gets group 0, etc.); a flat comma list
# keeps the historical semantics: every flask draws from the same lineup.
FLASK_NAME_GROUPS: list[list[str]] | None = None
if _ENV_NAMES:
    if ";" in _ENV_NAMES:
        FLASK_NAME_GROUPS = [[n.strip() for n in grp.split(",") if n.strip()]
                             for grp in _ENV_NAMES.split(";")]
        FLASK_WORM_NAMES = [n for grp in FLASK_NAME_GROUPS for n in grp]
    else:
        FLASK_WORM_NAMES = [n.strip() for n in _ENV_NAMES.split(",") if n.strip()]
    # Names become directory names and (flask, name) identity keys: duplicates
    # silently collide two worms onto one poem/weights/checkpoint file, and a
    # slash becomes a path separator. Fail loudly at startup instead.
    if len(set(FLASK_WORM_NAMES)) != len(FLASK_WORM_NAMES):
        raise ValueError(
            f"WORMLET_FLASK_WORM_NAMES contains duplicates: {_ENV_NAMES!r}")
    if any("/" in n for n in FLASK_WORM_NAMES):
        raise ValueError(
            f"WORMLET_FLASK_WORM_NAMES names must not contain '/': {_ENV_NAMES!r}")
    # With '/' banned a name is a single path segment, so the only remaining
    # traversal spellings are the segments '.' and '..' themselves ("wat..son"
    # is harmless). '..' as a worm name would write poem/weights/checkpoint
    # files into the flask dir's parent (2026-08-15 security audit).
    if any(n in (".", "..") for n in FLASK_WORM_NAMES):
        raise ValueError(
            f"WORMLET_FLASK_WORM_NAMES names must not be '.' or '..': {_ENV_NAMES!r}")
FLASKS_DIR = _DATA_ROOT / "flasks"

# WORMLET_FLASK_TEXTS: comma-separated corpus per flask (e.g.
# "hamlet,laozi,beowulf"). Unset/short list → hamlet (stock). Unknown names
# fail loudly at import, same policy as the worm-name guards above.
FLASK_TEXTS = [t.strip() for t in
               os.environ.get("WORMLET_FLASK_TEXTS", "").split(",") if t.strip()]
if FLASK_TEXTS:
    from corpus import library as _library
    _unknown = [t for t in FLASK_TEXTS if t not in _library.TITLES]
    if _unknown:
        raise ValueError(f"WORMLET_FLASK_TEXTS unknown corpora: {_unknown}")


def flask_corpus(flask_index: int) -> str | None:
    """Corpus for flask N (0-based); None = legacy hamlet/env-passage."""
    return FLASK_TEXTS[flask_index] if flask_index < len(FLASK_TEXTS) else None


def corpus_for_flask_name(flask_name: str) -> str | None:
    """Same, from a 'flask_N' name — for respawn/reset sites that don't
    carry the index."""
    try:
        return flask_corpus(int(flask_name.rsplit("_", 1)[-1]) - 1)
    except ValueError:
        return None


def _ensure_flask_worm_dir(flask_name: str, worm_name: str, seed: int) -> tuple[Path, dict]:
    wdir = FLASKS_DIR / flask_name / worm_name
    wdir.mkdir(parents=True, exist_ok=True)
    seed_file = wdir / "seed.txt"
    if not seed_file.exists():
        seed_file.write_text(str(seed))
    weights_path = wdir / "weights.json"
    if not weights_path.exists():
        # Fresh worm dir. In lifelike mode the _lifelike gene block is
        # PERSISTED into the new weights.json so the learning-rule params
        # genuinely enter the NES search space (rollovers flatten the on-disk
        # genome). Existing files are NEVER retro-injected: a lineage whose
        # parent_keys predate the block would flatten to a different length
        # and livelock the rollover — old lineages keep running the clipped
        # defaults instead. Must stay in lockstep with the cold-start
        # injection in app._load_initial_default_weights.
        weights = json.loads(DEFAULT_WEIGHTS.read_text())
        if lifelike.any_enabled():
            lifelike.ensure_params(weights)
        weights_path.write_text(json.dumps(weights))
        return wdir, weights
    return wdir, json.loads(weights_path.read_text())


def load_flasks(n_flasks: int = 4, n_worms_per_flask: int = 10) -> list[list[Worm]]:
    """Build N flasks of M worms each. Returns a list of lists — outer is
    flasks (in order flask_1, flask_2, ...), inner is the worms of that
    flask. Caller wraps each inner list in a WormGroup and attaches the flask's
    shared embedder (v7.1). Seeds are unique across all worms in all flasks
    (flask_idx * 1000 + worm_idx + 1) so no two worms ever share trajectories.

    The worlds are built with the process-default embedding model; the caller
    (app.lifespan) then attaches each flask's OWN shared EmbeddingModel via
    `attach_flask_model`, so the precomputed E-table is shared across the flask
    and independent of every other flask."""
    _min_names = (min(len(g) for g in FLASK_NAME_GROUPS)
                  if FLASK_NAME_GROUPS else len(FLASK_WORM_NAMES))
    if n_worms_per_flask > _min_names:
        raise ValueError(
            f"n_worms_per_flask={n_worms_per_flask} exceeds the {_min_names} "
            f"available worm names; worms would collide onto shared directories. "
            f"Add more names to FLASK_WORM_NAMES."
        )
    FLASKS_DIR.mkdir(parents=True, exist_ok=True)
    flasks: list[list[Worm]] = []
    for fi in range(n_flasks):
        flask_name = f"flask_{fi + 1}"
        worms: list[Worm] = []
        for wi in range(n_worms_per_flask):
            names = (FLASK_NAME_GROUPS[fi]
                     if FLASK_NAME_GROUPS and fi < len(FLASK_NAME_GROUPS)
                     else FLASK_WORM_NAMES)
            wname = names[wi % len(names)]
            seed = fi * 1000 + wi + 1
            wdir, weights = _ensure_flask_worm_dir(flask_name, wname, seed)
            world = World(seed=seed, weights=weights, corpus=flask_corpus(fi))
            poem_path = wdir / "poem.txt"
            if poem_path.exists():
                with open(poem_path) as f:
                    word_count = sum(1 for line in f if line.strip())
            else:
                word_count = 0
            recent_words = _tail_words(poem_path, n=20)
            pf = open(poem_path, "a", buffering=1)
            worms.append(Worm(
                name=wname, seed=seed, world=world,
                poem_path=poem_path, poem_file=pf,
                word_count=word_count, recent_words=recent_words,
            ))
        flasks.append(worms)
    return flasks


def attach_flask_model(worms: list[Worm], model) -> None:
    """Point every worm in a flask at the flask's single shared EmbeddingModel.
    Called once at startup and re-used across generations (the model's genome is
    swapped in place at rollover via `EmbeddingModel.set_genome`)."""
    for w in worms:
        w.world.embedding_model = model


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
    # Flask dir layout: .../flasks/<flask_name>/<worm_name>/poem.txt
    worm.world = World(seed=worm.seed, weights=weights,
                       corpus=corpus_for_flask_name(worm.poem_path.parent.parent.name))
    worm.word_count = 0
    worm.recent_words = []
