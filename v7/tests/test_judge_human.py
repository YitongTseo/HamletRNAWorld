"""Tests for WORMLET_JUDGE_BACKEND=human — the ballot/verdict protocol.

No network and no model: the human backend is pure filesystem. Each test
points WORMLET_DATA_DIR at a fresh temp dir, so nothing here can touch a
real lineage.

The two properties worth guarding, because breaking either corrupts a
lineage silently:

  1. No verdict => JudgeAwaitingHuman, and it must NOT be swallowed into a
     fallback. A human lineage that quietly paid an API for every generation
     nobody had got round to judging would defeat the whole point.
  2. fitness() ordering == the person's stated order. Selection reads only
     the order (evolution.rank_weights), so the synthesised scores have one
     job and it is this one.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server.judge as judge
from server.evolution import fitness, rank_weights

WORMS = [f"worm_{i}" for i in range(10)]
# ~56 windows so the 0.25 sample fraction yields ~14 sampled windows — the
# production figure (42,920 judged windows / 101 gens / 30 worms = 14.2) and,
# importantly, above fitness()'s FITNESS_WINDOW_FLOOR of 12, so the floor is
# not quietly dividing every score in these tests.
TOKENS = (("him night of him apparition come story speak we all from it "
           "that's Horatio dead and them Stand to you Give farewell mouse "
           "stirring").split() * 35)


class _Env:
    """Point the backend at a throwaway data dir for the duration."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = {k: os.environ.get(k) for k in
                     ("WORMLET_DATA_DIR", "WORMLET_JUDGE_BACKEND",
                      "WORMLET_JUDGE_FALLBACK", "WORMLET_JUDGE_HUMAN")}
        os.environ["WORMLET_DATA_DIR"] = self._tmp.name
        os.environ["WORMLET_JUDGE_BACKEND"] = "human"
        os.environ.pop("WORMLET_JUDGE_FALLBACK", None)
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()
        return False


def _judge_all(generation=256, flask="flask_1"):
    """Run every worm through judge_poem the way a rollover does — including
    absorbing JudgeAwaitingHuman per worm, which is what generations.py does
    so that every worm still files its ballot."""
    out = {}
    for w in WORMS:
        try:
            out[w] = judge.judge_poem(TOKENS, worm_name=w, seed=generation,
                                      flask=flask, generation=generation)
        except judge.JudgeAwaitingHuman:
            out[w] = []
    return out


def _write_verdict(root, top, flask="flask_1", generation=256, **extra):
    p = root / "verdicts" / flask / f"gen-{generation:04d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"flask": flask, "generation": generation,
                             "judge": "test", "top": top, **extra}))


def test_no_verdict_raises_awaiting_and_files_a_ballot():
    with _Env() as root:
        try:
            judge.judge_poem(TOKENS, worm_name="worm_0", seed=256,
                             flask="flask_1", generation=256)
            raise AssertionError("expected JudgeAwaitingHuman")
        except judge.JudgeAwaitingHuman as e:
            assert e.flask == "flask_1" and e.generation == 256
        # The ballot is the point: it must exist even though judging failed,
        # or the bench has nothing to show and the lineage deadlocks.
        ballot = root / "ballots" / "flask_1" / "gen-0256" / "worm_0.json"
        assert ballot.exists(), "ballot not filed on the awaiting path"
        body = json.loads(ballot.read_text())
        assert body["worm"] == "worm_0" and body["windows"]
        # Prompt index and true token offset are both recorded; the verdict's
        # `starred` list refers to the latter. The offsets are a SAMPLE, so
        # they are multiples of STRIDE but not consecutive ones.
        idxs = [w["idx"] for w in body["windows"]]
        assert [w["i"] for w in body["windows"]] == list(range(len(idxs)))
        assert all(i % judge.STRIDE == 0 for i in idxs)
        assert idxs == sorted(set(idxs))
        assert len(idxs) >= 12, f"only {len(idxs)} windows — below the floor"


def test_awaiting_never_falls_through_to_a_paid_backend():
    """With a fallback configured, a merely-late verdict must still wait."""
    with _Env():
        os.environ["WORMLET_JUDGE_FALLBACK"] = "anthropic"
        try:
            judge.judge_poem(TOKENS, worm_name="worm_0", seed=256,
                             flask="flask_1", generation=256)
            raise AssertionError("expected JudgeAwaitingHuman")
        except judge.JudgeAwaitingHuman:
            pass  # correct: no attempt on the paid backend


def test_fitness_order_matches_the_humans_order():
    with _Env() as root:
        _judge_all()  # file the ballots
        top = ["worm_4", "worm_7", "worm_1", "worm_9", "worm_2"]
        _write_verdict(root, top)
        scored = _judge_all()
        fits = {w: fitness(s) for w, s in scored.items()}
        ranked = sorted(fits, key=lambda w: -fits[w])
        assert ranked[:5] == top, f"{ranked[:5]} != {top}"
        # Everyone unranked ties, which is correct: rank_weights gives
        # ranks 6-10 identical weight, so there is nothing to distinguish.
        rest = [fits[w] for w in WORMS if w not in top]
        assert len(set(round(f, 12) for f in rest)) == 1
        assert max(rest) < min(fits[w] for w in top)


def test_rank_weights_justify_only_ranking_five():
    """The design claim, asserted so a future change to the NES weights
    cannot quietly invalidate the UI's 'pick your top five' affordance."""
    w = rank_weights(10)
    assert all(abs(x - w[5]) < 1e-12 for x in w[5:]), "losers no longer tie"
    assert abs(w[:3].sum() - w[w > 0].sum()) < 1e-12, \
        "top three no longer carry all the positive weight"


def test_coherence_comes_from_the_verdict_when_present():
    with _Env() as root:
        _judge_all()
        ballot = json.loads((root / "ballots" / "flask_1" / "gen-0256"
                             / "worm_4.json").read_text())
        first = ballot["windows"][0]["idx"]
        _write_verdict(root, ["worm_4"],
                       coherence={"worm_4": {str(first): 11}})
        scored = _judge_all()["worm_4"]
        by_idx = {s.idx: s for s in scored}
        assert by_idx[first].coherence == 11, "verdict coherence ignored"
        # Unlisted windows mirror the emotional score rather than inventing
        # a number.
        other = [s for s in scored if s.idx != first][0]
        assert other.coherence == other.emotional


def test_coherence_mirrors_emotional_without_a_local_model():
    with _Env() as root:
        _judge_all()
        _write_verdict(root, ["worm_4", "worm_7"])
        for s in _judge_all()["worm_4"]:
            assert s.coherence == s.emotional


def test_ballot_is_not_rewritten_on_re_fire():
    """The rollover re-fires on a cooldown until the verdict lands. Churning
    the file the bench is reading would be a race for no reason."""
    with _Env() as root:
        _judge_all()
        ballot = root / "ballots" / "flask_1" / "gen-0256" / "worm_0.json"
        ballot.write_text(json.dumps({"sentinel": True}))
        _judge_all()
        assert json.loads(ballot.read_text()) == {"sentinel": True}


def test_corrupt_verdict_is_an_error_not_a_wait():
    """A verdict that will never parse must stop the lineage loudly rather
    than leaving it waiting forever for a file that is already there."""
    with _Env() as root:
        p = root / "verdicts" / "flask_1" / "gen-0256.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ this is not json")
        try:
            judge.judge_poem(TOKENS, worm_name="worm_0", seed=256,
                             flask="flask_1", generation=256)
            raise AssertionError("expected a hard error")
        except judge.JudgeAwaitingHuman:
            raise AssertionError("corrupt verdict reported as 'not yet judged'")
        except RuntimeError:
            pass


def test_human_backend_refuses_without_a_flask():
    """Rather than guessing a ballot path and filing it somewhere nobody
    looks."""
    with _Env():
        try:
            judge.judge_poem(TOKENS, worm_name="worm_0", seed=256)
            raise AssertionError("expected a RuntimeError")
        except judge.JudgeAwaitingHuman:
            raise AssertionError("should not reach the ballot path")
        except RuntimeError as e:
            assert "flask=" in str(e)


def test_provenance_names_the_person():
    with _Env():
        os.environ["WORMLET_JUDGE_HUMAN"] = "joe"
        assert judge.judge_description() == "human:joe"
        os.environ.pop("WORMLET_JUDGE_HUMAN")
        assert judge.judge_description() == "human:unnamed"
