"""Integration test for run_generation_rollover's wiring to the elitist-NES
engine. Uses fake worms + a fake judge so it runs with NO Claude API call.

Guards the generations.py side of Changes 2 (true-eps persistence in
GenerationState), 5 (top-N elites carried verbatim into the next gen), and 1
(genome stays float end-to-end)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server.generations as G
from server.generations import GenerationState, GenerationProgress, run_generation_rollover
from server.evolution import flatten_weights, N_ELITES
from server.judge import ScoredWindow


BASE_WEIGHTS = {
    "ASEL": {"AVAL": 5.0, "AVAR": -3.0},
    "ASER": {"AVAL": -7.0, "AVAR": 2.0, "PVCL": 4.0},
    "AVAL": {"PVCL": 6.0},
}
N_WORMS = 8  # need > N_ELITES so there are fresh children too


class FakeWorm:
    def __init__(self, name: str, wdir: Path, seed: int):
        self.name = name
        self.seed = seed
        self.poem_path = wdir / "poem.txt"


def _make_worms(root: Path) -> list[FakeWorm]:
    """Each worm gets its own dir with a poem.txt and a slightly perturbed
    float weights.json, so genomes differ and fitnesses can be ranked."""
    parent_vec, keys = flatten_weights(BASE_WEIGHTS)
    worms = []
    for i in range(N_WORMS):
        name = f"W{i}"
        wdir = root / name
        wdir.mkdir(parents=True)
        # distinct fractional perturbation per worm -> proves float survives
        vec = parent_vec + 0.1 * (i + 1) * np.ones_like(parent_vec)
        wd: dict = {}
        for (s, t), w in zip(keys, vec):
            wd.setdefault(s, {})[t] = float(w)
        (wdir / "weights.json").write_text(json.dumps(wd))
        (wdir / "poem.txt").write_text("\n".join(["to", "be", "or", "not"] * 5))
        worms.append(FakeWorm(name, wdir, seed=100 + i))
    return worms


def _fake_judge(tokens, worm_name, seed=None, corpus="hamlet"):
    """W0 is clearly the best, W1 second, rest mediocre — a stable ranking."""
    score = {"W0": 90, "W1": 75}.get(worm_name, 25)
    return [ScoredWindow(0, tokens[:3], score, score)]


def _fresh_state(root: Path) -> GenerationState:
    vec, keys = flatten_weights(BASE_WEIGHTS)
    st = GenerationState(
        group_name="flask_test",
        generation=5,
        sigma=0.2,
        best_score_history=[1.0, 1.2, 1.1, 1.3, 1.25],
        parent_vector=vec.tolist(),
        parent_keys=[list(k) for k in keys],
    )
    return st


def _run_once(tmp_path: Path, monkeypatch) -> tuple[GenerationState, dict]:
    root = tmp_path / "worms"
    root.mkdir()
    worms = _make_worms(root)
    state = _fresh_state(tmp_path)
    monkeypatch(G, "GENERATIONS_ROOT", tmp_path / "generations")
    monkeypatch(G, "judge_poem", _fake_judge)
    progress = GenerationProgress()
    new_weights = run_generation_rollover(
        worms, state, progress, run_gardener=False,
    )
    return state, new_weights, worms


# --- a tiny monkeypatch shim so we don't need pytest -----------------------

class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def __call__(self, obj, attr, value):
        self._undo.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    def undo(self):
        for obj, attr, val in reversed(self._undo):
            setattr(obj, attr, val)


def _tmp(name: str) -> Path:
    import tempfile
    return Path(tempfile.mkdtemp(prefix=f"wormtest_{name}_"))


# --- tests -----------------------------------------------------------------

def test_rollover_persists_true_eps_and_elites():
    """Change 2: after a rollover, state.children records each next-gen worm's
    TRUE eps (or None for elites) and an is_elite flag. Change 5: exactly
    N_ELITES worms are marked elite."""
    mp = _MonkeyPatch()
    try:
        state, new_weights, worms = _run_once(_tmp("eps"), mp)
    finally:
        mp.undo()
    assert state.children is not None, "rollover must persist child spawn records"
    assert set(state.children.keys()) == {w.name for w in worms}
    n_elite = sum(1 for rec in state.children.values() if rec["is_elite"])
    assert n_elite == N_ELITES, f"expected {N_ELITES} elites, got {n_elite}"
    # Fresh (non-elite) children must carry a real eps vector; elites carry None.
    for rec in state.children.values():
        if rec["is_elite"]:
            assert rec["eps"] is None
        else:
            assert rec["eps"] is not None and len(rec["eps"]) > 0


def test_rollover_carries_best_genome_verbatim():
    """Change 5: the best worm's genome (W0) must reappear unchanged as one of
    the next generation's genomes."""
    mp = _MonkeyPatch()
    try:
        state, new_weights, worms = _run_once(_tmp("elite"), mp)
    finally:
        mp.undo()
    w0_dir = worms[0].poem_path.parent
    w0_genome = json.loads((w0_dir / "weights.json").read_text())
    w0_vec, _ = flatten_weights(w0_genome)
    survivors = [flatten_weights(g)[0] for g in new_weights.values()]
    assert any(np.allclose(w0_vec, s) for s in survivors), \
        "best worm's genome was not carried into the next generation"


def test_rollover_keeps_weights_float():
    """Change 1: returned genomes are float (fractional values survive)."""
    mp = _MonkeyPatch()
    try:
        state, new_weights, worms = _run_once(_tmp("float"), mp)
    finally:
        mp.undo()
    saw_fractional = False
    for wd in new_weights.values():
        for targets in wd.values():
            for v in targets.values():
                assert isinstance(v, float)
                if abs(v - round(v)) > 1e-6:
                    saw_fractional = True
    assert saw_fractional, "no fractional weight survived — rounding still happening?"


def test_rollover_second_pass_uses_persisted_eps():
    """The persisted-eps path (not the cold-start fallback) must work: running
    a second rollover off the state from the first must succeed and move the
    parent via the stored eps."""
    tmp = _tmp("twopass")
    mp = _MonkeyPatch()
    try:
        root = tmp / "worms"
        root.mkdir()
        worms = _make_worms(root)
        state = _fresh_state(tmp)
        mp(G, "GENERATIONS_ROOT", tmp / "generations")
        mp(G, "judge_poem", _fake_judge)
        progress = GenerationProgress()

        new_weights = run_generation_rollover(worms, state, progress, run_gardener=False)
        assert state.children is not None
        parent_after_first = np.array(state.parent_vector)

        # Install returned genomes into the live worm dirs (as app.py would),
        # then roll over again — now state.children drives the gradient.
        for w in worms:
            (w.poem_path.parent / "weights.json").write_text(json.dumps(new_weights[w.name]))
        run_generation_rollover(worms, state, progress, run_gardener=False)
        parent_after_second = np.array(state.parent_vector)
    finally:
        mp.undo()
    assert state.generation == 7  # started at 5, +2 rollovers
    assert not np.allclose(parent_after_first, parent_after_second), \
        "second rollover did not move the parent via persisted eps"


if __name__ == "__main__":
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
