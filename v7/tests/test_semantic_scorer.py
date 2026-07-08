"""Tests for the LLM-free semantic-correctness scorer (needs the nomic cache)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import pos_scorers as ps


def _have_cache():
    return bool(ps._load_sem_vectors())


def test_related_beats_unrelated():
    if not _have_cache():
        print("SKIP (no nomic cache)"); return
    related = ["king", "queen", "death", "grave", "father", "son", "night", "dark"]
    unrelated = ["the", "of", "and", "to", "a", "in", "that", "it"]
    assert ps.score_semantic(related) > ps.score_semantic(unrelated)


def test_repetition_scores_zero():
    if not _have_cache():
        print("SKIP"); return
    # identical adjacent words are excluded → no semantic credit for chanting
    assert ps.score_semantic(["love"] * 10) == 0.0


def test_oov_is_transparent():
    if not _have_cache():
        print("SKIP"); return
    # punctuation between two related words shouldn't break the adjacency
    with_punct = ps.score_semantic(["king", "!", "queen"])
    without = ps.score_semantic(["king", "queen"])
    assert with_punct == without == 1.0


def test_short_input():
    assert ps.score_semantic([]) == 0.0
    assert ps.score_semantic(["king"]) == 0.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
