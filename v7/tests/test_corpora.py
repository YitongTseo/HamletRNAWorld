"""Tests for the laozi/beowulf corpus loaders and the library dispatcher.
The corpus contract (matching hamlet): get_sentences_with_flags returns
TOKEN LISTS per line, plus parallel edible flags. No network, no LLM."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus import beowulf, hamlet, laozi, library


def test_laozi_token_lists_and_mostly_edible():
    lines, flags = laozi.get_sentences_with_flags("full")
    assert 800 < len(lines) < 1600          # ~ hamlet act1 scale
    assert len(lines) == len(flags)
    assert all(isinstance(ln, list) for ln in lines)  # the corpus contract
    assert sum(flags) / len(flags) > 0.9    # nearly all poem
    toks = {t for ln in lines for t in ln}
    assert "Gutenberg" not in toks          # envelope stripped
    assert "Tao" in toks


def test_laozi_headings_inedible():
    lines, flags = laozi.get_sentences_with_flags("full")
    part_lines = [(ln, ed) for ln, ed in zip(lines, flags)
                  if ln and ln[0] == "PART"]
    assert part_lines and not any(ed for _, ed in part_lines)


def test_beowulf_verse_edible_apparatus_not():
    lines, flags = beowulf.get_sentences_with_flags("full")
    assert 3000 < len(lines) < 7000
    assert all(isinstance(ln, list) for ln in lines)
    # Known verse line is edible ("the Spear-Danes' glory"):
    assert any("glory" in ln and "Spear" in ln and ed
               for ln, ed in zip(lines, flags))
    # Fitt heading is visible but inedible:
    heading = [(ln, ed) for ln, ed in zip(lines, flags)
               if ln[:2] == ["THE", "LIFE"]]
    assert heading and not heading[0][1]
    # Hall's {gloss} prose is inedible ("{The famous race of Spear-Danes.}"):
    gloss = [(ln, ed) for ln, ed in zip(lines, flags)
             if "famous" in ln and "race" in ln]
    assert gloss and not any(ed for _, ed in gloss)
    toks = {t for ln in lines for t in ln}
    assert "Glossary" not in toks           # front matter cut entirely
    assert "ADDENDA" not in toks            # end matter cut


def test_beowulf_line_numbers_and_brackets_never_tokenize():
    lines, _ = beowulf.get_sentences_with_flags("full")
    for ln in lines:
        for t in ln:
            assert not t.isdigit()
            assert "[" not in t and "{" not in t


def test_library_dispatch_and_titles():
    h = library.get_sentences_with_flags("hamlet", "opening")
    assert h == hamlet.get_sentences_with_flags("opening")  # byte-identical
    assert set(library.TITLES) == {"hamlet", "laozi", "beowulf", "daodejing"}
    assert set(library.DISPLAY_TITLES) == set(library.TITLES)
    try:
        library.get_sentences_with_flags("iliad")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        library.get_sentences_with_flags("laozi", "act1")
        assert False, "expected ValueError for non-full passage"
    except ValueError:
        pass


def test_daodejing_character_atoms():
    from corpus import daodejing
    lines, flags = daodejing.get_sentences_with_flags("full")
    toks = [t for ln in lines for t in ln]
    assert 5000 < len(toks) < 7000
    assert 700 < len({t for t in toks if t.isalpha()}) < 900  # ~810 chars
    assert all(len(t) == 1 for t in toks)      # character atoms
    assert not any("a" <= t.lower() <= "z" for t in toks)  # no Latin leaks
    assert "道" in toks and "。" in toks       # text + CJK punct both present
    # 81 chapter-marker lines out of 323 are inedible set-dressing:
    assert 0.7 < sum(flags) / len(flags) < 0.85


def test_daodejing_vertical_column_layout():
    """The scroll delivers reading order: char 0 topmost per column, columns
    right-to-left, siblings sharing one x. Latin corpora stay horizontal."""
    from sim.world import World
    w = World(seed=1, corpus="daodejing")
    for _ in range(60 * 60):
        w.tick()
    cols = {}
    for f in w.food:
        cols.setdefault(round(f.x), []).append((f.word_idx, f.y))
    assert len(cols) >= 2                      # multiple columns on field
    for chars in cols.values():
        chars.sort()
        ys = [y for _, y in chars]
        assert ys == sorted(ys)                # idx order == top-to-bottom
    h = World(seed=1, corpus="laozi")
    for _ in range(60 * 30):
        h.tick()
    ys = {round(f.y) for f in h.food}
    xs = {round(f.x) for f in h.food}
    assert len(xs) > len(ys)                   # horizontal: many x, few y


def test_beowulf_epoch_windows_tile_the_poem():
    """Three contiguous 1500-line windows, cycling, covering every line. The
    last one backs up to end on the final line rather than running short —
    a short epoch would strand the flask on an empty dish waiting for its
    sibling (joint rollover barrier)."""
    total = len(beowulf.get_sentences_with_flags("full")[0])
    assert total == 4286
    windows = [library.epoch_slice("beowulf", total, e) for e in range(3)]
    assert windows == [(0, 1500), (1500, 3000), (2786, 4286)]
    assert all(stop - start == 1500 for start, stop in windows)   # equal epochs
    covered = {i for start, stop in windows for i in range(start, stop)}
    assert covered == set(range(total))                           # nothing unread
    assert library.epoch_slice("beowulf", total, 3) == windows[0]  # cycles
    # Corpora without a cap are untouched, whatever the generation.
    assert library.epoch_slice("laozi", 1072, 7) == (0, 1072)
    assert library.epoch_slice("daodejing", 322, 7) == (0, 322)


def test_beowulf_line_pitch_matches_hamlet():
    """Beowulf's dish spawns at hamlet's 4.5 s (67.5 world units between
    lines), not the 1.57 s / 23.6 units that cramming all 4286 lines into
    one epoch produced. Every corpus still fills the same ~112-minute
    epoch, so the joint rollover barrier is unmoved."""
    from sim.text_scroller import SCROLL_SPEED, SPAWN_INTERVAL
    from sim.world import World
    w = World(seed=1, corpus="beowulf")
    assert w.text_scroller.spawn_interval == SPAWN_INTERVAL
    assert w.text_scroller.spawn_interval * SCROLL_SPEED == 67.5
    epoch_s = len(w.text_scroller.sentences) * w.text_scroller.spawn_interval
    for corpus in ("laozi", "daodejing"):
        other = World(seed=1, corpus=corpus)
        other_s = len(other.text_scroller.sentences) * other.text_scroller.spawn_interval
        assert abs(other_s - epoch_s) / epoch_s < 0.02   # same epoch length
    # The window moves with the generation, and only for capped corpora.
    w.set_corpus_epoch(1)
    assert w.text_scroller.sentences[0] != World(seed=1, corpus="beowulf").text_scroller.sentences[0]
    laozi_w = World(seed=1, corpus="laozi")
    first = laozi_w.text_scroller.sentences[0]
    laozi_w.set_corpus_epoch(2)
    assert laozi_w.text_scroller.sentences[0] == first


def test_corpus_window_is_recorded_for_provenance():
    """The window rides on the world so each generation's metadata.json can
    record which lines it read: two beowulf generations are two different
    texts, and a fitness trend across them is not a trend in the worms."""
    from sim.world import World
    w = World(seed=1, corpus="beowulf")
    assert w.corpus_window == (0, 1500)
    w.set_corpus_epoch(2)
    assert w.corpus_window == (2786, 4286)
    # Whole-text corpora record nothing — absent key = read it all.
    assert World(seed=1, corpus="laozi").corpus_window is None
    assert World(seed=1).corpus_window is None


def test_stale_window_checkpoint_is_rejected_not_silently_exhausted():
    """A checkpoint written against the old full-corpus scroller indexes
    past the new window. Restoring it would report the corpus exhausted
    immediately (empty dish + hunger until the sibling flask finishes), so
    restore() must refuse and leave the scroller untouched."""
    from sim.text_scroller import TextScroller
    lines, flags = library.get_sentences_with_flags("beowulf")
    start, stop = library.epoch_slice("beowulf", len(lines), 0)
    # loop=False explicitly: generation mode, the only mode where the index
    # is absolute and can run off the end.
    ts = TextScroller(lines[start:stop], loop=False, edible_flags=flags[start:stop])
    good = ts.snapshot()
    stale = dict(good, sent_idx=3000)          # old 4286-line generation
    try:
        ts.restore(stale)
    except ValueError:
        pass
    else:
        raise AssertionError("stale checkpoint restored")
    assert not ts.corpus_exhausted
    ts.restore(good)                           # in-window snapshots still load
