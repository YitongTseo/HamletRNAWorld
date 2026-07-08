"""v7 co-evolution of the SHARED embedding+POS nets (server/embedding.py).

The connectome brain evolves per-flask (server/evolution.py + generations.py).
The embedding+POS nets are ONE genome (θ_g, ~7,697 params) shared across every
worm — globally across all 8 poetry flasks (each sanity experiment has its own
copy). It is evolved by the SAME NES, but pooled over every worm's fitness:

  * Each worm is spawned with its own perturbation  θ_g + σ_g · eps_g.
  * Its single fitness is attributed to that perturbation.
  * At each generation the NES gradient is estimated from ALL worms' (eps_g,
    fitness) — a far larger, less noisy sample than any one flask's brain NES.
  * There is NO elitism here: θ_g is a single centroid, not a population.
  * σ_g adapts by the same population-baseline 1/5 rule as the brains
    (fraction of worms beating the previous generation's mean fitness).

Two layers:
  1. `pooled_update()` — the pure NES math (reuses server.evolution primitives).
  2. `SharedNetCoordinator` — a filesystem barrier so the 4 poetry PROCESSES
     (2 flasks each) agree on θ_g each generation. One leader collects every
     flask's contribution (with a timeout/quorum so a stalled flask can never
     freeze evolution), updates θ_g, and publishes it; all processes read the
     new θ_g before spawning the next generation. Determinism: contributions
     are processed in sorted (flask, worm) order.

See docs/superpowers/specs/2026-07-08-v7-learned-embedding-design.md §3(c).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from server import embedding
from server.evolution import nes_update, adapt_sigma, SIGMA_INIT, SUCCESS_TARGET


# --- pure pooled NES update -------------------------------------------------

@dataclass
class SharedUpdate:
    new_parent: np.ndarray
    new_sigma: float
    fresh_mean: float
    success_rate: float


def pooled_update(
    parent: np.ndarray,
    eps_list: list[np.ndarray],
    scores: list[float],
    sigma: float,
    parent_fitness: float | None,
) -> SharedUpdate:
    """One NES step for the shared net from ALL worms' (eps, score).

    parent_fitness: the previous generation's pooled fresh-mean (the parent-
    fitness proxy for Rechenberg's 1/5 rule). None -> hold sigma.
    """
    if not eps_list:
        return SharedUpdate(parent.copy(), sigma, 0.0, SUCCESS_TARGET)
    new_parent = nes_update(parent, eps_list, scores, sigma=sigma)
    fresh_mean = float(np.mean(scores))
    if parent_fitness is None:
        success_rate = SUCCESS_TARGET
    else:
        success_rate = sum(s > parent_fitness for s in scores) / len(scores)
    new_sigma = adapt_sigma(sigma, success_rate)
    return SharedUpdate(new_parent, new_sigma, fresh_mean, success_rate)


# --- persistent shared-net state -------------------------------------------

@dataclass
class SharedNetState:
    """The shared embedding+POS genome lineage. One per experiment family
    (all poetry flasks share ONE; each sanity experiment has its own)."""
    name: str
    generation: int = 0
    sigma: float = SIGMA_INIT
    parent_vector: list[float] | None = None
    prev_fresh_mean: float | None = None
    best_mean_history: list[float] = field(default_factory=list)

    @classmethod
    def load_or_init(cls, path: Path, name: str, seed: int = 0) -> "SharedNetState":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        parent = embedding.EmbeddingParams.random_init(seed).flatten()
        return cls(name=name, parent_vector=parent.tolist())

    def parent(self) -> np.ndarray:
        return np.array(self.parent_vector, dtype=np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(asdict(self)))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def spawn_perturbations(
    parent: np.ndarray, sigma: float, n: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (genomes, epses): n fresh Gaussian samples θ_g + σ·eps."""
    genomes, epses = [], []
    for _ in range(n):
        eps = rng.standard_normal(parent.shape)
        epses.append(eps)
        genomes.append(parent + sigma * eps)
    return genomes, epses


# --- cross-process coordinator ---------------------------------------------

class SharedNetCoordinator:
    """Filesystem barrier for the global shared net across N poetry processes.

    Layout under `root` (e.g. data/poetry_shared/):
        state.json                       # published SharedNetState (θ_g, σ_g, gen)
        gen-NNNN/contrib_<flask>.json    # each flask's {eps, score} per worm
        gen-NNNN/done                    # written by the leader after publishing

    Every process calls `contribute()` at its rollover, then `barrier_update()`.
    The leader (the process owning the lexicographically-first expected flask,
    or whoever wins the create-race on the lock) waits for all expected flasks
    (up to `timeout_s`, then proceeds with a quorum) and publishes the new
    state. Non-leaders poll for `done`. If nobody can lead, each process falls
    back to updating from its own worms so evolution never stalls.
    """

    def __init__(self, root: Path, expected_flasks: list[str],
                 timeout_s: float = 300.0, seed: int = 0):
        self.root = root
        self.expected = sorted(expected_flasks)
        self.timeout_s = timeout_s
        self.seed = seed
        self.state_path = root / "state.json"

    def state(self) -> SharedNetState:
        return SharedNetState.load_or_init(self.state_path, "poetry_shared", self.seed)

    def _gen_dir(self, gen: int) -> Path:
        return self.root / f"gen-{gen:04d}"

    def contribute(self, gen: int, flask: str,
                   epses: list[list[float]], scores: list[float]) -> None:
        """Publish this flask's per-worm (eps, score) for generation `gen`."""
        d = self._gen_dir(gen)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write(d / f"contrib_{flask}.json",
                      json.dumps({"flask": flask, "eps": epses, "scores": scores}))

    def _collect(self, gen: int) -> dict[str, dict]:
        d = self._gen_dir(gen)
        out: dict[str, dict] = {}
        for f in self.expected:
            p = d / f"contrib_{f}.json"
            if p.exists():
                try:
                    out[f] = json.loads(p.read_text())
                except Exception:
                    pass
        return out

    def barrier_update(self, gen: int, *, now=time.monotonic,
                       sleep=time.sleep, keepalive=None) -> SharedNetState:
        """Wait for all expected flasks' contributions (quorum after timeout),
        then (as leader) compute + publish the new shared state. Returns the
        published state for the NEXT generation. Idempotent: if `done` already
        exists, just loads and returns it.

        `keepalive` (if given) is called each poll so a long wait doesn't trip
        the process watchdog — generations are minutes-long, so a fast process
        may block here a while waiting for a slow one."""
        def _tick():
            if keepalive:
                try:
                    keepalive()
                except Exception:
                    pass
        gd = self._gen_dir(gen)
        done = gd / "done"
        if done.exists():
            return self.state()

        # Try to become leader by atomically creating the lock dir.
        lock = gd / "leader.lock"
        is_leader = False
        try:
            lock.mkdir(parents=True, exist_ok=False)
            is_leader = True
        except FileExistsError:
            is_leader = False

        if not is_leader:
            # Follower: poll for the leader to finish, then read published state.
            deadline = now() + self.timeout_s + 30.0
            while now() < deadline:
                if done.exists():
                    return self.state()
                _tick()
                sleep(0.5)
            # Leader vanished — take over.
            is_leader = True

        # Leader path: wait for all expected flasks, or quorum after timeout.
        deadline = now() + self.timeout_s
        while now() < deadline:
            got = self._collect(gen)
            if len(got) >= len(self.expected):
                break
            _tick()
            sleep(0.5)
        contribs = self._collect(gen)
        published = self._publish(gen, contribs)
        _atomic_write(done, "ok")
        return published

    def manager(self) -> "SharedNetManager":
        return SharedNetManager(self.root, "poetry_shared", self.seed)

    def _publish(self, gen: int, contribs: dict[str, dict]) -> SharedNetState:
        st = self.state()
        parent = st.parent()
        # Pool eps/scores in sorted (flask, worm-index) order for determinism.
        eps_list: list[np.ndarray] = []
        scores: list[float] = []
        for flask in sorted(contribs):
            c = contribs[flask]
            for eps, sc in zip(c["eps"], c["scores"]):
                eps_list.append(np.asarray(eps, dtype=np.float64))
                scores.append(float(sc))
        if eps_list:
            upd = pooled_update(parent, eps_list, scores, st.sigma, st.prev_fresh_mean)
            st.parent_vector = upd.new_parent.tolist()
            st.sigma = upd.new_sigma
            st.prev_fresh_mean = upd.fresh_mean
            st.best_mean_history.append(upd.fresh_mean)
        st.generation = gen + 1
        st.save(self.state_path)
        return st


# --- per-worm genome management (used by orchestrator + both rollover paths) -

def _worm_eps(worm_seed: int, generation: int) -> np.ndarray:
    """Deterministic, per-(worm, generation) Gaussian eps for the shared net.
    Same worm+gen always yields the same perturbation, so a restart mid-gen
    reconstructs identical worlds."""
    seed = (int(worm_seed) * 1_000_003 + int(generation) + 1) % (2**32)
    return np.random.default_rng(seed).standard_normal(embedding.GENOME_SIZE)


class SharedNetManager:
    """State I/O + per-worm perturbation for one shared-net lineage.

    Poetry: all 8 flasks share ONE manager (shared_dir=data/poetry_shared),
    and the pooled update is done by SharedNetCoordinator across processes.
    Sanity experiments: each has its own manager (shared_dir under its data
    tree) and updates directly via `update()` (single flask, no coordinator).

    Per-worm embedding lives in <worm_dir>/embedding.json = {genome, eps}.
    """

    def __init__(self, shared_dir: Path, name: str, seed: int = 0):
        self.dir = shared_dir
        self.name = name
        self.seed = seed
        self.state_path = shared_dir / "state.json"

    def state(self) -> SharedNetState:
        return SharedNetState.load_or_init(self.state_path, self.name, self.seed)

    def save_state(self, st: SharedNetState) -> None:
        st.save(self.state_path)

    def assign_worm(self, worm_dir: Path, worm_seed: int) -> tuple[list[float], list[float]]:
        """Load this worm's embedding genome, or spawn one from the current
        shared parent+sigma using the deterministic per-worm eps. Writes
        <worm_dir>/embedding.json. Returns (genome, eps) as lists."""
        ej = worm_dir / "embedding.json"
        if ej.exists():
            try:
                d = json.loads(ej.read_text())
                return d["genome"], d["eps"]
            except Exception:
                pass
        st = self.state()
        eps = _worm_eps(worm_seed, st.generation)
        genome = st.parent() + st.sigma * eps
        self._write_worm(worm_dir, genome, eps)
        return genome.tolist(), eps.tolist()

    def respawn_worm(self, worm_dir: Path, worm_seed: int,
                     st: SharedNetState) -> tuple[list[float], list[float]]:
        """Spawn a FRESH perturbation from the given (updated) state for the
        next generation and overwrite embedding.json. Returns (genome, eps)."""
        eps = _worm_eps(worm_seed, st.generation)
        genome = st.parent() + st.sigma * eps
        self._write_worm(worm_dir, genome, eps)
        return genome.tolist(), eps.tolist()

    def _write_worm(self, worm_dir: Path, genome, eps) -> None:
        worm_dir.mkdir(parents=True, exist_ok=True)
        g = genome.tolist() if hasattr(genome, "tolist") else list(genome)
        e = eps.tolist() if hasattr(eps, "tolist") else list(eps)
        _atomic_write(worm_dir / "embedding.json", json.dumps({"genome": g, "eps": e}))

    def read_worm_eps(self, worm_dir: Path) -> list[float] | None:
        ej = worm_dir / "embedding.json"
        if not ej.exists():
            return None
        try:
            return json.loads(ej.read_text())["eps"]
        except Exception:
            return None

    def update(self, eps_list: list, scores: list[float]) -> SharedNetState:
        """Single-flask pooled NES update (sanity path). Advances + persists
        the state and returns it."""
        st = self.state()
        eps_arr = [np.asarray(e, dtype=np.float64) for e in eps_list]
        if eps_arr:
            upd = pooled_update(st.parent(), eps_arr, scores, st.sigma, st.prev_fresh_mean)
            st.parent_vector = upd.new_parent.tolist()
            st.sigma = upd.new_sigma
            st.prev_fresh_mean = upd.fresh_mean
            st.best_mean_history.append(upd.fresh_mean)
        st.generation += 1
        self.save_state(st)
        return st
