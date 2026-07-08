"""Mid-generation checkpoint test: a TextScroller's corpus progress must
survive a snapshot()/restore() round-trip exactly, so a process restart
resumes mid-sentence/mid-chew instead of restarting the generation.

This is the load-bearing property for Approach C (lightweight checkpoint):
the worm body/brain are intentionally NOT preserved, but the scroll position,
the eaten set, and the on-screen words must be byte-for-byte identical."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.text_scroller import TextScroller

SENTENCES = [
    ["to", "be", "or", "not", "to", "be"],
    ["that", "is", "the", "question"],
    ["whether", "tis", "nobler", "in", "the", "mind"],
    ["to", "suffer", "the", "slings", "and", "arrows"],
]


def _advance(scroller: TextScroller, steps: int, dt: float = 0.05,
             eat: set | None = None) -> None:
    """Step the scroller, eating any word whose (line_id, word_idx) is in
    `eat` the moment it appears on screen."""
    for _ in range(steps):
        scroller.step(dt)
        if eat:
            for w in scroller.alive_words():
                if (w.line_id, w.word_idx) in eat:
                    scroller.mark_eaten(w.line_id, w.word_idx)


def test_snapshot_restore_roundtrip_is_exact():
    a = TextScroller(SENTENCES, loop=False)
    _advance(a, steps=200, eat={(0, 0), (0, 1), (1, 2)})
    snap = a.snapshot()

    b = TextScroller(SENTENCES, loop=False)
    b.restore(snap)

    assert b.snapshot() == snap, "restore() did not reproduce the snapshot"
    assert b._dead == a._dead, "eaten set diverged across round-trip"
    assert b.corpus_exhausted == a.corpus_exhausted
    # On-screen words must match in identity-relevant fields.
    aw = [(w.line_id, w.word_idx, round(w.x, 6), round(w.y, 6)) for w in a.alive_words()]
    bw = [(w.line_id, w.word_idx, round(w.x, 6), round(w.y, 6)) for w in b.alive_words()]
    assert aw == bw, "alive_words diverged across round-trip"


def test_restored_scroller_continues_identically():
    """After restore, stepping forward must track an uninterrupted scroller
    tick-for-tick (the scroller has no RNG, so resume is deterministic)."""
    a = TextScroller(SENTENCES, loop=False)
    _advance(a, steps=150)
    snap = a.snapshot()

    b = TextScroller(SENTENCES, loop=False)
    b.restore(snap)

    for i in range(150):
        a.step(0.05)
        b.step(0.05)
        aw = [(w.line_id, w.word_idx) for w in a.alive_words()]
        bw = [(w.line_id, w.word_idx) for w in b.alive_words()]
        assert aw == bw, f"divergence at step {i} after restore"
    assert a.corpus_exhausted == b.corpus_exhausted


def test_eaten_words_not_respawned_after_restore():
    """A word eaten before the checkpoint must stay eaten after restore —
    otherwise the poem would gain duplicate words on every restart."""
    a = TextScroller(SENTENCES, loop=False)
    _advance(a, steps=60, eat={(0, 0)})
    assert (0, 0) in a._dead
    snap = a.snapshot()

    b = TextScroller(SENTENCES, loop=False)
    b.restore(snap)
    # Keep stepping; (0,0) must never reappear among alive words.
    for _ in range(300):
        b.step(0.05)
        assert (0, 0) not in {(w.line_id, w.word_idx) for w in b.alive_words()}


if __name__ == "__main__":
    test_snapshot_restore_roundtrip_is_exact()
    print("PASS: snapshot/restore round-trip is exact")
    test_restored_scroller_continues_identically()
    print("PASS: restored scroller continues identically")
    test_eaten_words_not_respawned_after_restore()
    print("PASS: eaten words stay eaten after restore")
