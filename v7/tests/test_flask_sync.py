"""Flasks in one process must stay in step, and a flask waiting on the
rollover barrier must not starve.

Both of these were learned the hard way on 2026-08-16. The beowulf flask
correctly rejected a checkpoint written against the pre-window corpus and
restarted its pass; its two siblings restored theirs at ~95% of the way
through, exhausted their corpora minutes later, and then sat on empty dishes
because rollover is a JOINT barrier. With hunger on that is a death sentence
— satiety falls ~0.08/minute, so ~13 minutes of empty dish kills a worm.
Four died before the flasks were resynced by hand.

Two independent guards, tested here:
  1. checkpoint restore is all-or-nothing across the process, so one
     unusable checkpoint can never leave one flask at line 0 and another at
     line 1400;
  2. metabolism stops once a flask's corpus pass is spent, so however the
     flasks desync, waiting on the barrier is idling and not dying.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.text_scroller import TextScroller

SENTENCES = [[f"line{i}", "words", "here"] for i in range(40)]


class _FakeWorld:
    """Just the surface server.app touches during startup recovery."""

    def __init__(self, sentences):
        self.sentences = sentences
        self.text_scroller = TextScroller(sentences, loop=False)
        self.corpus_epoch = 0
        self.lifelike_restored = None

    def restart_pass(self):
        self.text_scroller = TextScroller(self.sentences, loop=False)

    def set_corpus_epoch(self, epoch):
        self.corpus_epoch = epoch

    def restore_lifelike(self, data):
        self.lifelike_restored = data


class _FakeWorm:
    def __init__(self, wdir: Path, sentences):
        wdir.mkdir(parents=True, exist_ok=True)
        self.name = wdir.name
        self.poem_path = wdir / "poem.txt"
        self.poem_path.write_text("some words already eaten\n")
        self.poem_file = open(self.poem_path, "a", buffering=1)
        self.word_count = 5
        self.recent_words = ["some", "words"]
        self.world = _FakeWorld(sentences)


class _FakeState:
    def __init__(self, generation):
        self.generation = generation


class _FakeFlask:
    def __init__(self, worms, generation=3):
        self.worms = worms
        self.state = _FakeState(generation)


def _advance(worm, steps=60):
    for _ in range(steps):
        worm.world.text_scroller.step(0.5)


def _write_checkpoint(worm, generation=3, scroller=None):
    (worm.poem_path.parent / "checkpoint.json").write_text(json.dumps({
        "version": 1,
        "generation": generation,
        "scroller": scroller or worm.world.text_scroller.snapshot(),
        "lifelike": {"satiety": 0.5},
    }))


def _fresh_flasks(tmp: Path, n_flasks=3, n_worms=2):
    return [
        _FakeFlask([_FakeWorm(tmp / f"flask_{f}" / f"w{w}", SENTENCES)
                    for w in range(n_worms)])
        for f in range(n_flasks)
    ]


def test_one_unusable_checkpoint_restarts_every_flask():
    """The desync guard. One worm in one flask has a checkpoint from the
    wrong corpus window; every worm in every flask must restart the
    generation, not just that one."""
    import server.app as A
    with tempfile.TemporaryDirectory() as td:
        flasks = _fresh_flasks(Path(td))
        for f in flasks:
            for w in f.worms:
                _advance(w)                       # everyone is mid-pass
                _write_checkpoint(w)
        # ... except one worm, whose checkpoint points past the end of the
        # line list it now has (exactly what a pre-window beowulf checkpoint
        # looks like against a 1500-line window).
        victim = flasks[1].worms[0]
        stale = dict(victim.world.text_scroller.snapshot(), sent_idx=9999)
        _write_checkpoint(victim, scroller=stale)

        A._restore_or_reset_all(flasks)

        for f in flasks:
            for w in f.worms:
                snap = w.world.text_scroller.snapshot()
                assert snap["sent_idx"] == 0, (
                    f"{w.name} resumed mid-pass while a sibling restarted — "
                    "this is the desync that starves the siblings")
                assert snap["elapsed"] == 0.0
                assert w.word_count == 0
                assert w.poem_path.read_text() == ""
                assert not (w.poem_path.parent / "checkpoint.json").exists()


def test_all_usable_checkpoints_still_resume_together():
    """The guard must not cost the feature: when every checkpoint is good,
    every worm resumes where it was."""
    import server.app as A
    with tempfile.TemporaryDirectory() as td:
        flasks = _fresh_flasks(Path(td))
        expected = {}
        for f in flasks:
            for w in f.worms:
                _advance(w)
                _write_checkpoint(w)
                expected[w.name] = w.world.text_scroller.snapshot()["sent_idx"]
                # Rewind the live scroller so a successful restore is
                # visible: startup builds worlds from scratch.
                w.world.text_scroller = TextScroller(SENTENCES, loop=False)

        A._restore_or_reset_all(flasks)

        for f in flasks:
            for w in f.worms:
                assert w.world.text_scroller.snapshot()["sent_idx"] == expected[w.name]
                assert w.word_count == 5                    # poem kept
                assert w.world.lifelike_restored == {"satiety": 0.5}
        assert all(w.world.corpus_epoch == 3 for f in flasks for w in f.worms)


def test_missing_checkpoints_reset_everyone_without_error():
    """First boot of a lineage: no checkpoints anywhere. Everyone starts
    fresh, and nothing raises on the missing files."""
    import server.app as A
    with tempfile.TemporaryDirectory() as td:
        flasks = _fresh_flasks(Path(td))
        A._restore_or_reset_all(flasks)
        for f in flasks:
            for w in f.worms:
                assert w.word_count == 0
                assert w.poem_path.read_text() == ""


def test_worm_waiting_on_the_barrier_does_not_starve():
    """Metabolism stops when the corpus pass is spent. A flask that finishes
    first can wait an hour or more for the slowest sibling; at ~0.08 satiety
    per minute it would otherwise be dead in 13."""
    saved = os.environ.get("WORMLET_HUNGER")
    os.environ["WORMLET_HUNGER"] = "1"
    try:
        from sim.world import World
        from sim import lifelike

        w = World(seed=1)
        assert w._hunger_on, "test needs hunger on to mean anything"
        w.satiety = 0.5

        # Still scrolling: metabolism runs.
        w.text_scroller = TextScroller(SENTENCES, loop=False)
        for _ in range(200):
            w.tick()
        assert w.satiety < 0.5, "a fed flask must still burn satiety"
        assert not w.dead

        # Pass spent (every line scrolled off): metabolism stops.
        drained = TextScroller([], loop=False)
        assert drained.corpus_exhausted
        w.text_scroller = drained
        w.satiety = 0.02          # one tick from death under the old rule
        for _ in range(5000):     # ~100 s of sim, 40x the old time-to-die
            w.tick()
        assert not w.dead, "worm starved while waiting on the rollover barrier"
        assert w.satiety == 0.02, "satiety drifted while the dish was empty"

        # And the pause is not a free lunch: hunger resumes with the corpus.
        w.text_scroller = TextScroller(SENTENCES, loop=False)
        for _ in range(200):
            w.tick()
        assert w.satiety < 0.02
    finally:
        if saved is None:
            os.environ.pop("WORMLET_HUNGER", None)
        else:
            os.environ["WORMLET_HUNGER"] = saved
