"""Corpus registry — the one place that knows which texts exist.

get_sentences_with_flags(corpus, passage) dispatches to the per-text
module; hamlet keeps its passage vocabulary (opening/soliloquy/act1/full),
the others have only "full". TITLES feed the judge prompt and viewer
labels, so a flask's critic and its dish header always name the same text.
"""
from __future__ import annotations

from corpus import beowulf, daodejing, hamlet, laozi

TITLES = {
    "hamlet": "Shakespeare's Hamlet",
    "laozi": "the Tao Teh King of Lao-Tse (Legge translation)",
    "beowulf": "the Anglo-Saxon epic Beowulf (Hall translation)",
    "daodejing": "the Daodejing of Laozi, in the original Classical Chinese",
}

# Short viewer labels. Decoupled from TITLES because TITLES feed the judge
# rubric, and editing a live flask's rubric mid-lineage is a critic change
# (regime rule) — display can modernise romanisation freely, the rubric
# cannot.
DISPLAY_TITLES = {
    "hamlet": "Hamlet",
    "laozi": "Tao Te Ching (Legge)",
    "beowulf": "Beowulf (Hall)",
    "daodejing": "道德經",
}

# Spatial layout per corpus: "vertical" = column layout (CJK reading
# order delivered by the scroll); default horizontal.
LAYOUTS = {"daodejing": "vertical"}

# Lines a flask reads per generation, for corpora too long to fit one epoch
# at hamlet's pacing. The rollover is a joint barrier, so epoch LENGTH is
# fixed (~1500 lines x 4.5 s); the only free variable is how many lines you
# cram into it, and lines-per-epoch sets the vertical pitch directly
# (pitch = SCROLL_SPEED / line rate). Beowulf's full 4286 lines compressed
# into one epoch gave a 23.6-unit pitch against 18px glyphs — a wall of
# text, three times denser than the hamlet flasks' 67.5 units. Capped here,
# beowulf gets hamlet's pitch and hamlet's feeding economics, and reads the
# epic in three contiguous windows across three generations instead of all
# of it every generation. Corpora absent from this map are unchanged.
EPOCH_LINES = {"beowulf": 1500}

_LOADERS = {
    "hamlet": hamlet.get_sentences_with_flags,
    "laozi": laozi.get_sentences_with_flags,
    "beowulf": beowulf.get_sentences_with_flags,
    "daodejing": daodejing.get_sentences_with_flags,
}


def get_sentences_with_flags(corpus: str, passage: str = "full",
                             n: int | None = None):
    try:
        loader = _LOADERS[corpus]
    except KeyError:
        raise ValueError(f"unknown corpus {corpus!r}; have {sorted(_LOADERS)}")
    return loader(passage, n=n) if n is not None else loader(passage)


def epoch_slice(corpus: str, total: int, epoch: int) -> tuple[int, int]:
    """Half-open [start, stop) line window this corpus shows in generation
    `epoch`. Windows are contiguous, equal-length and cycle, so a lineage
    reads the whole text across ceil(total / EPOCH_LINES) generations.

    The final window of a cycle backs up to end on the last line rather than
    running short: a short epoch would drain this flask's dish early and
    leave its worms starving on an empty field while the sibling flask (same
    process, joint rollover barrier) finishes. It therefore overlaps the
    previous window — 214 lines for beowulf — which is the cheap end of the
    trade.

    Corpora with no cap, or shorter than their cap, get the whole text."""
    n = EPOCH_LINES.get(corpus)
    if not n or total <= n:
        return 0, total
    n_windows = -(-total // n)          # ceil
    start = min((epoch % n_windows) * n, total - n)
    return start, start + n
