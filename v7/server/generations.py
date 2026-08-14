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

import contextlib
import fcntl
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
    SIGMA_INIT, GAMMA, EMOTIONAL_WEIGHT, N_ELITES,
    evolve_generation, fitness, flatten_weights, unflatten_weights, WeightDict,
)
from server.gardener import maybe_write_log
from server.judge import judge_poem, ScoredWindow
from server.orchestrator import Worm
from server.poem_clean import clean as clean_punctuation

V6_ROOT = Path(__file__).resolve().parent.parent
# Honor WORMLET_DATA_DIR so each experiment process (which sets its own
# WORMLET_DATA_DIR) lands its generation artifacts under that process's
# isolated data tree. Default unchanged — prod (no env var set) still writes
# to v6/data/generations/.
_DATA_ROOT = Path(os.environ.get("WORMLET_DATA_DIR", V6_ROOT / "data"))
GENERATIONS_ROOT = _DATA_ROOT / "generations"

# Experiment 2 (2026-07-17): the σ-control scheme is chosen per process via
# WORMLET_SIGMA_SCHEME for the live A/B — vs_mean (control) · vs_elite · xnes ·
# sigma_anneal. Default vs_mean = the prior behaviour, so an unset env is a no-op.
SIGMA_SCHEME = os.environ.get("WORMLET_SIGMA_SCHEME", "vs_mean")


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
    # Change 2: the spawn record for the CURRENTLY-LIVE generation, keyed by
    # worm name: {name: {"eps": [float]|None, "is_elite": bool}}. The next
    # rollover reads the TRUE eps from here for the NES gradient instead of
    # reconstructing it from (rounded) weight files. None until the first
    # rollover under the new engine (cold-start / upgrade fallback path).
    children: dict[str, dict] | None = None
    # v7 σ-collapse fix: the mean fitness of the PREVIOUS generation's fresh
    # children, used as the parent-fitness proxy for Rechenberg's 1/5 rule.
    # None until the first rollover records it.
    prev_fresh_mean: float | None = None

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


# Repo-wide lock so the 4 parallel poetry processes (which share one git repo)
# serialize their add/commit/rebase/push instead of racing on index.lock and
# clobbering each other's pushes. Each process only stages its own gen_dir, so
# there's no file overlap — the lock only guards the git operations.
_COMMIT_LOCK = V6_ROOT.parent / ".git" / "wormlet-commit.lock"
_LOCK_ACQUIRE_TIMEOUT = 120.0  # seconds to wait for the lock before giving up


@contextlib.contextmanager
def _repo_commit_lock(keepalive: KEEPALIVE_FN | None):
    """Bounded, keepalive-friendly exclusive lock over the shared repo.
    Yields True if the lock was acquired, False if it timed out."""
    fh = open(_COMMIT_LOCK, "w")
    deadline = time.monotonic() + _LOCK_ACQUIRE_TIMEOUT
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                if keepalive: keepalive()
                time.sleep(0.5)
        yield acquired
    finally:
        if acquired:
            fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _git_commit(msg: str, paths: list[Path], keepalive: KEEPALIVE_FN | None) -> bool:
    """Add + commit + rebase + push the listed paths. Returns True on success.
    Failure is logged but non-fatal — the simulation continues even if the
    remote is unreachable. Serialized across processes via _repo_commit_lock."""
    repo = str(V6_ROOT.parent)

    def _git(args, timeout):
        return subprocess.run(["git", "-C", repo, *args],
                              check=True, capture_output=True, timeout=timeout)

    try:
        with _repo_commit_lock(keepalive) as got_lock:
            if not got_lock:
                print("[GENERATIONS] git commit skipped: could not acquire repo lock "
                      f"within {_LOCK_ACQUIRE_TIMEOUT:.0f}s", flush=True)
                return False

            rel = [str(p.relative_to(V6_ROOT.parent)) for p in paths]
            _git(["add", *rel], 30)
            if keepalive: keepalive()

            # Nothing staged → nothing to do (idempotent rollover). git diff
            # --cached exits 1 when there ARE staged changes.
            if subprocess.run(["git", "-C", repo, "diff", "--cached", "--quiet"],
                              capture_output=True, timeout=30).returncode == 0:
                print("[GENERATIONS] git commit: nothing to commit", flush=True)
                return False

            _git(["commit", "-m", msg], 30)
            if keepalive: keepalive()
            # Land the other processes' (and the 2nd server's) commits under us
            # before pushing, so the push fast-forwards instead of racing.
            _git(["pull", "--rebase"], 60)
            if keepalive: keepalive()
            _git(["push"], 60)
            return True
    except subprocess.CalledProcessError as e:
        # Most likely: no remote configured, or a rebase conflict (the 2nd
        # server co-publishes). Abort any half-finished rebase so the working
        # tree isn't left wedged for the next rollover. The commit itself, if
        # it landed, stays in local history — artifacts are safe on disk.
        subprocess.run(["git", "-C", repo, "rebase", "--abort"],
                       capture_output=True, timeout=30)
        print(f"[GENERATIONS] git commit failed: {e.stderr.decode(errors='replace')[:500]}",
              flush=True)
        return False
    except subprocess.TimeoutExpired:
        subprocess.run(["git", "-C", repo, "rebase", "--abort"],
                       capture_output=True, timeout=30)
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
            # Change 3: same sampling seed for every worm in a generation, so
            # all worms are judged under the SAME window-sampling protocol —
            # otherwise one worm might get its good windows sampled and another
            # not, making the cross-worm fitness comparison (i.e. selection)
            # unfair. Seed varies per generation for audit-trail variety.
            scored = judge_poem(clean, worm_name=w.name, seed=state.generation)
        except Exception as e:
            print(f"[GENERATIONS] judge failed for {w.name}: {e}", flush=True)
            scored = []
        scored_by_worm[w.name] = scored
        fitness_by_worm[w.name] = fitness(scored)
        progress.worms_done += 1

    # Judge-outage guard: if EVERY worm came back unscored, the judge is down
    # (local endpoint offline, API key missing/unreachable) or every poem was
    # empty — either way there is nothing to select on. Evolving would rank an
    # all-zero field (arbitrary elites) and adapt σ on noise, silently
    # corrupting the lineage. Raise instead: the caller logs and continues,
    # state stays untouched, the flask never respawns, so its corpus stays
    # exhausted and the rollover re-fires — i.e. the lineage WAITS for the
    # judge to come back rather than taking a garbage step.
    if not any(scored_by_worm.values()):
        raise RuntimeError(
            f"judge produced zero scored windows across all {len(worms)} worms "
            f"in {group} — aborting rollover; will retry when the corpus "
            f"exhaustion re-fires"
        )

    # --- Phase: evolving ---
    progress.phase = PHASE_EVOLVING
    if keepalive: keepalive()

    parent_vec = np.array(state.parent_vector, dtype=np.float64)
    parent_keys = [tuple(k) for k in state.parent_keys]  # type: ignore[arg-type]

    # Gather, per live worm: its float genome, the TRUE eps it was spawned with
    # (Change 2 — read from state.children, not reconstructed from rounded
    # weights), whether it was carried as an elite, and its fitness.
    genomes: list[np.ndarray] = []
    epses: list[np.ndarray | None] = []
    is_elite_flags: list[bool] = []
    scores_list: list[float] = []
    worm_order: list[str] = []
    records = state.children or {}
    for w in worms:
        worm_order.append(w.name)
        w.last_fitness = fitness_by_worm[w.name]  # v7: for the poetry shared-net coordinator
        cur_vec, _ = flatten_weights(json.loads(w.poem_path.parent.joinpath("weights.json").read_text()))
        genomes.append(cur_vec)
        scores_list.append(fitness_by_worm[w.name])
        rec = records.get(w.name)
        if rec is not None:
            eps_rec = rec.get("eps")
            epses.append(np.array(eps_rec, dtype=np.float64) if eps_rec is not None else None)
            is_elite_flags.append(bool(rec.get("is_elite", False)))
        else:
            # Cold-start / first-rollover-after-upgrade fallback: no spawn record
            # on file, so reconstruct eps from the genome and treat it as fresh.
            eps = (cur_vec - parent_vec) / state.sigma if state.sigma > 0 else np.zeros_like(parent_vec)
            epses.append(eps)
            is_elite_flags.append(False)

    rng = np.random.default_rng(state.generation + 1)
    ng = evolve_generation(
        parent_vec, state.sigma, genomes, epses, is_elite_flags, scores_list,
        n_elites=N_ELITES, rng=rng, parent_fitness=state.prev_fresh_mean,
        scheme=SIGMA_SCHEME,
    )
    new_parent_vec = ng.new_parent
    new_sigma = ng.new_sigma
    best_score = max(scores_list) if scores_list else 0.0

    # Rank ordering for write-out (highest fitness = rank 0).
    ranked = ng.ranked_indices
    rank_of = {worm_order[idx]: r for r, idx in enumerate(ranked)}

    # Assign next-generation genomes to worm slots (elites first, then fresh)
    # and record each slot's TRUE eps + elite flag for the next rollover.
    new_weights: dict[str, WeightDict] = {}
    new_children: dict[str, dict] = {}
    for w, child_vec, child_eps, child_elite in zip(
        worms, ng.next_genomes, ng.next_epses, ng.next_is_elite
    ):
        new_weights[w.name] = unflatten_weights(child_vec, parent_keys)
        new_children[w.name] = {
            "eps": child_eps.tolist() if child_eps is not None else None,
            "is_elite": bool(child_elite),
        }

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
        "sigma_scheme": ng.scheme,
        "success_rate": ng.success_rate,
        "sigma_baseline": ng.sigma_baseline,
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
        state.children = new_children
        state.prev_fresh_mean = ng.fresh_mean
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
        # Stage the whole flask group, not just gen_dir, so the group-level
        # embedder.json (embedder weights) and state.json (NES/connectome state)
        # are pushed alongside this generation's per-worm weights.json snapshots.
        committed = _git_commit(msg, [GENERATIONS_ROOT / group], keepalive=keepalive)
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
    state.children = new_children
    state.prev_fresh_mean = ng.fresh_mean
    # parent_keys unchanged across generations (same connectome topology).
    state.save()

    return new_weights


# =====================================================================
# Experiment-mode rollover
# =====================================================================
# A simpler, cheaper variant of run_generation_rollover for the four
# sanity-check experiments (words / nouns / adj_noun / pos_chain). Key
# differences from the prod path:
#   - Scoring is a free, deterministic POS function (no Claude call)
#   - Storage is light (metrics.json + best_worm.json + lineage), no
#     per-worm poem dumps or scores.jsonl
#   - No git commits — sim experiments are local-only
#   - Gardener runs every N gens (configurable per experiment), not every gen
#   - One flask per process (10 worms); the multi-flask global-winner /
#     meta-gardener dance does not apply
# Same NES math as prod: float genome, true-eps gradient, top-5 elitism,
# Rechenberg 1/5-rule σ adaptation.


def run_experiment_rollover(
    worms: list[Worm],
    state: GenerationState,
    progress: GenerationProgress,
    experiment,                     # server.experiments.Experiment
    keepalive: KEEPALIVE_FN | None = None,
) -> dict[str, WeightDict]:
    """Score with experiment.scorer, NES-evolve, write light artifacts,
    return new weights. No Claude judge, no git commit."""
    from server.pos_scorers import pos_breakdown, longest_valid_chain

    group = state.group_name
    progress.group = group
    progress.generation = state.generation + 1
    progress.error = None
    if not progress.started_at:
        progress.started_at = time.time()
    if progress.worms_total == 0:
        progress.worms_total = len(worms)
        progress.worms_done = 0

    # --- Phase: scoring (no LLM) ---
    progress.phase = PHASE_JUDGING
    eaten_by_worm: dict[str, list[str]] = {}
    fitness_by_worm: dict[str, float] = {}
    pos_by_worm: dict[str, dict[str, int]] = {}
    longest_chain_by_worm: dict[str, int] = {}
    for w in worms:
        if keepalive: keepalive()
        eaten = _read_eaten_tokens(w)
        eaten_by_worm[w.name] = eaten
        fitness_by_worm[w.name] = experiment.scorer(eaten) if experiment.scorer else 0.0
        w.last_fitness = fitness_by_worm[w.name]  # v7.1: scores the flask embedder
        pos_by_worm[w.name] = pos_breakdown(eaten)
        longest_chain_by_worm[w.name] = longest_valid_chain(eaten)
        progress.worms_done += 1

    # --- Phase: evolving (same NES math as prod) ---
    progress.phase = PHASE_EVOLVING
    if keepalive: keepalive()

    parent_vec = np.array(state.parent_vector, dtype=np.float64)
    parent_keys = [tuple(k) for k in state.parent_keys]  # type: ignore[arg-type]

    genomes: list[np.ndarray] = []
    epses: list[np.ndarray | None] = []
    is_elite_flags: list[bool] = []
    scores_list: list[float] = []
    worm_order: list[str] = []
    records = state.children or {}
    for w in worms:
        worm_order.append(w.name)
        cur_vec, _ = flatten_weights(
            json.loads(w.poem_path.parent.joinpath("weights.json").read_text())
        )
        genomes.append(cur_vec)
        scores_list.append(fitness_by_worm[w.name])
        rec = records.get(w.name)
        if rec is not None:
            eps_rec = rec.get("eps")
            epses.append(np.array(eps_rec, dtype=np.float64) if eps_rec is not None else None)
            is_elite_flags.append(bool(rec.get("is_elite", False)))
        else:
            eps = (cur_vec - parent_vec) / state.sigma if state.sigma > 0 else np.zeros_like(parent_vec)
            epses.append(eps)
            is_elite_flags.append(False)

    rng = np.random.default_rng(state.generation + 1)
    ng = evolve_generation(
        parent_vec, state.sigma, genomes, epses, is_elite_flags, scores_list,
        n_elites=N_ELITES, rng=rng, parent_fitness=state.prev_fresh_mean,
        scheme=SIGMA_SCHEME,
    )
    new_parent_vec = ng.new_parent
    new_sigma = ng.new_sigma
    best_score = max(scores_list) if scores_list else 0.0
    ranked = ng.ranked_indices

    # --- Lineage: which prev-gen worm is each new slot descended from? ---
    # Elites (slots 0..N_ELITES-1) carry verbatim from ranked[0..N_ELITES-1].
    # Fresh worms (slots N_ELITES..) descend from the updated parent, but
    # for visualization we point them at the prev-gen winner (rank 0) since
    # that's the worm whose ε contributed most strongly to the gradient.
    winner_name = worm_order[ranked[0]] if ranked else None
    parent_name_for_slot: list[str | None] = []
    for slot_i in range(len(worms)):
        if slot_i < min(N_ELITES, len(ranked)):
            parent_name_for_slot.append(worm_order[ranked[slot_i]])
        else:
            parent_name_for_slot.append(winner_name)

    # Per-worm fitness + lineage record (the data the new generations
    # graph reads to draw cross-generation lines).
    rank_of = {worm_order[idx]: r for r, idx in enumerate(ranked)}
    per_worm = []
    for w in worms:
        per_worm.append({
            "name": w.name,
            "fitness": fitness_by_worm[w.name],
            "rank": rank_of[w.name],
            "words_eaten": len(eaten_by_worm[w.name]),
            "pos_breakdown": pos_by_worm[w.name],
            "longest_chain": longest_chain_by_worm[w.name],
            # Set at evolve time (after we know which slot each worm fills
            # in gen N+1). Below we re-zip to attach next-gen lineage info.
        })

    # Assign next-gen genomes to slots and record children records.
    new_weights: dict[str, WeightDict] = {}
    new_children: dict[str, dict] = {}
    next_gen_lineage: list[dict] = []
    for slot_i, (w, child_vec, child_eps, child_elite) in enumerate(zip(
        worms, ng.next_genomes, ng.next_epses, ng.next_is_elite
    )):
        new_weights[w.name] = unflatten_weights(child_vec, parent_keys)
        new_children[w.name] = {
            "eps": child_eps.tolist() if child_eps is not None else None,
            "is_elite": bool(child_elite),
        }
        next_gen_lineage.append({
            "slot": slot_i,
            "name": w.name,
            "is_elite": bool(child_elite),
            "parent_name_in_this_gen": parent_name_for_slot[slot_i],
        })

    # --- Write light artifacts ---
    gen_dir = GENERATIONS_ROOT / group / f"gen-{state.generation + 1:04d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment_mode": experiment.mode,
        "experiment_label": experiment.label,
        "scorer": experiment.scorer.__name__ if experiment.scorer else None,
        "generation": state.generation + 1,
        "sigma_used": state.sigma,
        "sigma_next": new_sigma,
        "best_score": best_score,
        "avg_score": float(np.mean(scores_list)) if scores_list else 0.0,
        "success_rate": ng.success_rate,
        "sigma_scheme": ng.scheme,
        "sigma_baseline": ng.sigma_baseline,
        "delta_parent_norm": float(np.linalg.norm(new_parent_vec - parent_vec)),
        "new_parent_norm": float(np.linalg.norm(new_parent_vec)),
        "started_at": progress.started_at,
        "ended_at": time.time(),
        "per_worm": per_worm,
        "next_gen_lineage": next_gen_lineage,
        "passage": os.environ.get("WORMLET_PASSAGE", "act1"),
    }
    (gen_dir / "metrics.json").write_text(json.dumps(metrics))

    # The "best worm" record: name, score, eaten-word sample for the gardener
    # and the public viewer (so we keep something tangible without storing
    # every worm's full poem).
    if winner_name and eaten_by_worm.get(winner_name):
        sample = eaten_by_worm[winner_name][:200]
        (gen_dir / "best_worm.json").write_text(json.dumps({
            "name": winner_name,
            "fitness": best_score,
            "words_eaten_sample": sample,
            "total_words_eaten": len(eaten_by_worm[winner_name]),
            "pos_breakdown": pos_by_worm[winner_name],
            "longest_chain": longest_chain_by_worm[winner_name],
        }))

    # --- Optional gardener (every N gens) ---
    new_generation = state.generation + 1
    if experiment.gardener_every and new_generation % experiment.gardener_every == 0:
        if keepalive: keepalive()
        try:
            from server.gardener import maybe_write_experiment_log
            log_path = maybe_write_experiment_log(
                gen_dir=gen_dir,
                experiment=experiment,
                metrics=metrics,
                eaten_by_worm=eaten_by_worm,
                generations_root=GENERATIONS_ROOT / group,
                keepalive=keepalive,
            )
            if log_path:
                print(f"[EXPERIMENT] gardener wrote {log_path.name}", flush=True)
        except Exception:
            print("[EXPERIMENT] gardener raised; continuing", flush=True)

    # v7.1: the flask's shared embedder is stepped by the caller
    # (app._run_experiment_rollover_sync) after this returns, using each worm's
    # last_fitness set above — see server/flask_embedder.py.

    # --- Persist state (no git commit) ---
    state.generation = new_generation
    state.sigma = new_sigma
    state.best_score_history.append(best_score)
    state.parent_vector = new_parent_vec.tolist()
    state.children = new_children
    state.prev_fresh_mean = ng.fresh_mean
    state.save()

    return new_weights
