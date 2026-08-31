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
