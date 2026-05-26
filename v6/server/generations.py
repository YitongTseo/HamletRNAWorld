"""End-of-generation orchestrator.

Responsibilities (in order):
  1. Read each worm's eaten-token stream from its poem.txt.
  2. Apply punctuation cleanup → poem_clean.txt.
  3. Sample + LLM-judge → per-window scores.
  4. Aggregate to per-worm fitness, rank, NES update.
  5. Write everything to data/generations/gen-NNNN/<group>/<worm>/.
  6. git add → commit → push.
  7. Return the new per-worm WeightDicts for the caller to install.

The caller (sim_loop) is responsible for actually installing the new
weights on the running Worm instances and resetting the TextScroller.
This module never touches the live sim state — it only reads poems and
returns plans.

State (per group): the parent vector, the NES sigma, and the generation
counter persist in data/generations/<group>/state.json so a restart can
resume the lineage. The first generation reads the worms' current
weights.json as the initial parent (cold-start path).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from corpus.hamlet import is_non_reactive  # punctuation set is in PUNCTUATION below
from server.evolution import (
    SIGMA_INIT, GAMMA, EMOTIONAL_WEIGHT,
    adapt_sigma, fitness, flatten_weights, nes_update,
    rank_weights, spawn_children, unflatten_weights, WeightDict,
)
from server.gardener import maybe_write_log
from server.judge import judge_poem, ScoredWindow
from server.orchestrator import Worm
from server.poem_clean import clean as clean_punctuation

V6_ROOT = Path(__file__).resolve().parent.parent
GENERATIONS_ROOT = V6_ROOT / "data" / "generations"


# Phase strings exposed via /api/generation_status so the frontend can show
# what's happening during the freeze.
PHASE_RUNNING = "running"          # normal simulation, no rollover in progress
PHASE_CORPUS_DRAINING = "corpus_draining"  # corpus exhausted, letting words scroll off
PHASE_JUDGING = "judging"          # LLM scoring in progress
PHASE_EVOLVING = "evolving"        # NES update + writing artifacts
PHASE_COMMITTING = "committing"    # git add/commit/push
PHASE_RESPAWNING = "respawning"    # installing new weights into worms

KEEPALIVE_FN = Callable[[], None]  # callback that ticks the watchdog


@dataclass
class GenerationState:
    """Per-group evolution state. Persisted to JSON between generations
    so the lineage survives process restarts."""
    group_name: str
    generation: int = 0          # number of completed generations
    sigma: float = SIGMA_INIT
    best_score_history: list[float] = field(default_factory=list)
    # parent_vector and parent_keys regenerate from disk if missing.
    parent_vector: list[float] | None = None
    parent_keys: list[list[str]] | None = None  # JSON-friendly [src, tgt] pairs

    def state_path(self) -> Path:
        return GENERATIONS_ROOT / self.group_name / "state.json"

    @classmethod
    def load_or_init(cls, group_name: str, default_weights: WeightDict) -> "GenerationState":
        path = GENERATIONS_ROOT / group_name / "state.json"
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        # Cold start: flatten the default connectome as our starting parent.
        vec, keys = flatten_weights(default_weights)
        return cls(
            group_name=group_name,
            parent_vector=vec.tolist(),
            parent_keys=[list(k) for k in keys],
        )

    def save(self) -> None:
        self.state_path().parent.mkdir(parents=True, exist_ok=True)
        self.state_path().write_text(json.dumps(asdict(self)))


@dataclass
class GenerationProgress:
    """Shared by sim_loop and exposed via /api/generation_status for the
    frontend progress overlay. Updated in-place during a rollover."""
    phase: str = PHASE_RUNNING
    group: str | None = None
    generation: int = 0
    worms_total: int = 0
    worms_done: int = 0
    started_at: float = 0.0   # time.time(), 0 = not started
    error: str | None = None


def _read_eaten_tokens(worm: Worm) -> list[str]:
    """Return every token the worm has eaten in this generation, in order.
    Drains poem.txt — one token per nonblank line — and ignores empties.
    We trust the file contents because drain_and_persist wrote them via
    a line-buffered handle, so no token can be partial here."""
    if not worm.poem_path.exists():
        return []
    with open(worm.poem_path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _write_worm_artifacts(
    out_dir: Path,
    worm: Worm,
    raw_tokens: list[str],
    clean_tokens: list[str],
    scored: list[ScoredWindow],
    fit: float,
    rank: int,
    weights: WeightDict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "poem_raw.txt").write_text("\n".join(raw_tokens))
    (out_dir / "poem_clean.txt").write_text(" ".join(clean_tokens))
    (out_dir / "weights.json").write_text(json.dumps(weights))
    (out_dir / "seed.txt").write_text(str(worm.seed))
    with (out_dir / "scores.jsonl").open("w") as f:
        for s in scored:
            f.write(json.dumps({
                "idx": s.idx,
                "tokens": s.tokens,
                "emotional": s.emotional,
                "coherence": s.coherence,
            }) + "\n")
    (out_dir / "fitness.json").write_text(json.dumps({
        "fitness": fit,
        "rank": rank,
        "windows_scored": len(scored),
        "gamma": GAMMA,
        "emotional_weight": EMOTIONAL_WEIGHT,
    }))


def _git_commit(msg: str, paths: list[Path], keepalive: KEEPALIVE_FN | None) -> bool:
    """Add + commit + push the listed paths. Returns True on success.
    Failure is logged but non-fatal — the simulation continues even if
    the remote is unreachable."""
    try:
        rel = [str(p.relative_to(V6_ROOT.parent)) for p in paths]
        subprocess.run(["git", "-C", str(V6_ROOT.parent), "add", *rel],
                       check=True, capture_output=True, timeout=30)
        if keepalive: keepalive()
        subprocess.run(["git", "-C", str(V6_ROOT.parent), "commit", "-m", msg],
                       check=True, capture_output=True, timeout=30)
        if keepalive: keepalive()
        subprocess.run(["git", "-C", str(V6_ROOT.parent), "push"],
                       check=True, capture_output=True, timeout=60)
        return True
    except subprocess.CalledProcessError as e:
        # Most likely: nothing to commit (idempotent rollover), or no remote
        # configured. Either way, don't crash the sim.
        print(f"[GENERATIONS] git commit failed: {e.stderr.decode(errors='replace')[:500]}",
              flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[GENERATIONS] git commit timed out", flush=True)
        return False


def _purge_gen_dir(gen_dir: Path, kept_worm: str | None) -> None:
    """Trim bulky files from a generation directory AFTER a successful git
    commit. The git history retains everything; what we keep on local disk
    is the subset the gardener regularly reads + (optionally) one worm's
    full record. Pass `kept_worm=None` to purge every worm in this flask
    equally (used for non-winning flasks under the global-winner policy).

    Kept locally (for every worm in the generation):
      - fitness.json   (per-worm score + rank — gardener reads this)
      - seed.txt       (small, lineage reconstruction)
      - scores.jsonl   (per-window scores — gardener samples these)
    Kept locally (flask-level):
      - metadata.json, selection.json
    Kept locally (only the named `kept_worm`, if any):
      - weights.json     (the genome we want to point at later)
      - poem_clean.txt   (the literary artifact worth keeping)
    Deleted (recoverable from git):
      - poem_raw.txt    for every worm
      - weights.json    for every worm except `kept_worm`
      - poem_clean.txt  for every worm except `kept_worm`

    Under the global-winner policy: across all six flasks for an epoch,
    exactly one (flask, worm) pair is the global winner. That flask's
    purge call passes `kept_worm=winner_name`; the other five flasks
    pass `kept_worm=None`."""
    if not gen_dir.exists():
        return
    n_removed = 0
    for worm_dir in gen_dir.iterdir():
        if not worm_dir.is_dir():
            continue
        is_kept = (kept_worm is not None and worm_dir.name == kept_worm)
        targets = ["poem_raw.txt"]
        if not is_kept:
            targets.extend(["weights.json", "poem_clean.txt"])
        for fname in targets:
            f = worm_dir / fname
            if f.exists():
                try:
                    f.unlink()
                    n_removed += 1
                except Exception:
                    pass
    label = f"kept worm={kept_worm}" if kept_worm else "no worm fully kept"
    print(f"[GENERATIONS] purged {n_removed} files from {gen_dir.name} ({label})", flush=True)


def run_generation_rollover(
    worms: list[Worm],
    state: GenerationState,
    progress: GenerationProgress,
    keepalive: KEEPALIVE_FN | None = None,
    run_gardener: bool = True,
) -> dict[str, WeightDict]:
    """Run one full end-of-generation cycle and return the new per-worm
    WeightDicts. Caller installs them. Caller must NOT tick worms during
    this call — the function blocks for the duration of LLM scoring and
    git push.

    `keepalive` is invoked at key checkpoints; have it bump the watchdog's
    _LAST_TICK_AT so the freeze doesn't trip the kill-and-restart.
    """
    group = state.group_name
    progress.group = group
    progress.generation = state.generation + 1
    progress.error = None
    if not progress.started_at:
        progress.started_at = time.time()
    # If the caller hasn't pre-set the totals (multi-flask runs do; legacy
    # single-group callers don't), initialize them here. We then increment
    # worms_done across the per-worm loop below — never reset it — so the
    # overlay bar advances continuously when stitched across multiple
    # flasks.
    if progress.worms_total == 0:
        progress.worms_total = len(worms)
        progress.worms_done = 0

    # --- Phase: judging ---
    progress.phase = PHASE_JUDGING
    poems_raw: dict[str, list[str]] = {}
    poems_clean: dict[str, list[str]] = {}
    scored_by_worm: dict[str, list[ScoredWindow]] = {}
    fitness_by_worm: dict[str, float] = {}

    for w in worms:
        if keepalive: keepalive()
        raw = _read_eaten_tokens(w)
        clean = clean_punctuation(raw)
        poems_raw[w.name] = raw
        poems_clean[w.name] = clean
        try:
            scored = judge_poem(clean, worm_name=w.name, seed=state.generation * 1000 + hash(w.name) % 1000)
        except Exception as e:
            print(f"[GENERATIONS] judge failed for {w.name}: {e}", flush=True)
            scored = []
        scored_by_worm[w.name] = scored
        fitness_by_worm[w.name] = fitness(scored)
        progress.worms_done += 1

    # --- Phase: evolving ---
    progress.phase = PHASE_EVOLVING
    if keepalive: keepalive()

    parent_vec = np.array(state.parent_vector, dtype=np.float64)
    parent_keys = [tuple(k) for k in state.parent_keys]  # type: ignore[arg-type]

    # Reconstruct each child's eps from the current weight files vs parent.
    # On first run there are no children yet (cold-start path); skip the
    # NES step and just promote the worms' current weights as the new parent.
    eps_list: list[np.ndarray] = []
    scores_list: list[float] = []
    worm_order: list[str] = []
    for w in worms:
        worm_order.append(w.name)
        cur_vec, _ = flatten_weights(json.loads(w.poem_path.parent.joinpath("weights.json").read_text()))
        eps = (cur_vec - parent_vec) / state.sigma if state.sigma > 0 else np.zeros_like(parent_vec)
        eps_list.append(eps)
        scores_list.append(fitness_by_worm[w.name])

    new_parent_vec = nes_update(parent_vec, eps_list, scores_list, sigma=state.sigma)
    best_score = max(scores_list) if scores_list else 0.0
    improved = (
        not state.best_score_history
        or best_score > state.best_score_history[-1]
    )
    new_sigma = adapt_sigma(state.sigma, improved=improved)

    # Rank ordering for write-out (highest fitness = rank 0).
    ranked = sorted(range(len(worms)), key=lambda i: -scores_list[i])
    rank_of = {worm_order[idx]: r for r, idx in enumerate(ranked)}

    # Spawn next-generation children: one elite + (N-1) perturbed.
    rng = np.random.default_rng(state.generation + 1)
    next_children, next_eps = spawn_children(new_parent_vec, n=len(worms), sigma=new_sigma, rng=rng)
    new_weights: dict[str, WeightDict] = {}
    for w, child_vec in zip(worms, next_children):
        new_weights[w.name] = unflatten_weights(child_vec, parent_keys)

    # --- Phase: writing artifacts ---
    gen_dir = GENERATIONS_ROOT / group / f"gen-{state.generation + 1:04d}"
    for w in worms:
        wdir = gen_dir / w.name
        _write_worm_artifacts(
            wdir, w,
            raw_tokens=poems_raw[w.name],
            clean_tokens=poems_clean[w.name],
            scored=scored_by_worm[w.name],
            fit=fitness_by_worm[w.name],
            rank=rank_of[w.name],
            weights=new_weights[w.name],
        )
    (gen_dir / "metadata.json").write_text(json.dumps({
        "group": group,
        "generation": state.generation + 1,
        "judge_model": "claude-haiku-4-5",
        "sigma_used": state.sigma,
        "sigma_next": new_sigma,
        "best_score": best_score,
        "started_at": progress.started_at,
        "ended_at": time.time(),
        "ranks": [worm_order[i] for i in ranked],
    }))
    (gen_dir / "selection.json").write_text(json.dumps({
        "winner": worm_order[ranked[0]] if ranked else None,
        "new_parent_norm": float(np.linalg.norm(new_parent_vec)),
        "delta_norm": float(np.linalg.norm(new_parent_vec - parent_vec)),
    }))

    # --- Gardener's log (optional, free-form, may PASS) ---
    # Skipped in multi-flask mode; the caller runs a single meta-gardener
    # observing all flasks AND handles the atomic commit+purge for the
    # whole epoch (all six flasks + meta log) in one pass.
    if not run_gardener:
        state.generation += 1
        state.sigma = new_sigma
        state.best_score_history.append(best_score)
        state.parent_vector = new_parent_vec.tolist()
        state.save()
        return new_weights

    if keepalive: keepalive()
    try:
        log_path = maybe_write_log(
            gen_dir=gen_dir,
            scored_by_worm=scored_by_worm,
            fitness_by_worm=fitness_by_worm,
            generation_num=state.generation + 1,
            group_name=group,
            # state.best_score_history is updated AFTER this call, so it
            # currently holds scores from gens 1..N-1, which is exactly
            # what the gardener needs for the catalog of prior generations.
            best_score_history=state.best_score_history,
        )
        if log_path:
            print(f"[GENERATIONS] gardener wrote {log_path.name}", flush=True)
        else:
            print(f"[GENERATIONS] gardener PASSed (or skipped) for gen-{state.generation + 1:04d}",
                  flush=True)
    except Exception:
        # Gardener is a nice-to-have; never let it break the rollover.
        print("[GENERATIONS] gardener log raised; continuing", flush=True)

    # --- Phase: committing ---
    progress.phase = PHASE_COMMITTING
    if keepalive: keepalive()
    committed = False
    if os.environ.get("WORMLET_GIT_COMMIT", "1") != "0":
        msg = f"gen {state.generation + 1:04d} [{group}]: best={best_score:.3f} σ={state.sigma:.3f}→{new_sigma:.3f}"
        committed = _git_commit(msg, [gen_dir], keepalive=keepalive)
    else:
        # Local-only mode: artifacts already written to disk, just skip git.
        print(f"[GENERATIONS] WORMLET_GIT_COMMIT=0: skipping commit for gen-{state.generation + 1:04d}",
              flush=True)

    # --- Post-commit purge ---
    # Only purge after the data is safely in git, otherwise we'd lose it.
    # Set WORMLET_PURGE_ANYWAY=1 to purge even without a successful commit
    # (useful for local-only dev where you don't need archival).
    purge_anyway = os.environ.get("WORMLET_PURGE_ANYWAY", "0") == "1"
    if committed or purge_anyway:
        winner_name = worm_order[ranked[0]] if ranked else None
        try:
            _purge_gen_dir(gen_dir, winner_name=winner_name)
        except Exception:
            # Purge is opportunistic — never let it break the rollover.
            print(f"[GENERATIONS] purge for gen-{state.generation + 1:04d} raised; continuing",
                  flush=True)

    # --- Update persistent state ---
    state.generation += 1
    state.sigma = new_sigma
    state.best_score_history.append(best_score)
    state.parent_vector = new_parent_vec.tolist()
    # parent_keys unchanged across generations (same connectome topology).
    state.save()

    return new_weights
