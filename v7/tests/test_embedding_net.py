"""Tests for the v7 learned embedding + POS net (server/embedding.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import embedding as E
from server.embedding import EmbeddingParams, EmbeddingModel, GENOME_SIZE, D_EMB


def _fake_nomic(words, seed=1):
    rng = np.random.default_rng(seed)
    return {w: rng.standard_normal(E.D_IN) for w in words}


def test_genome_size_matches_layout():
    # 6512 embedding + 1185 POS = 7697
    assert GENOME_SIZE == 512*11+11 + 55*11+11 + 22*11+11 + 72*16+16 + 16*1+1
    p = EmbeddingParams.random_init(0)
    assert p.flatten().shape == (GENOME_SIZE,)


def test_flatten_unflatten_roundtrip():
    p = EmbeddingParams.random_init(3)
    vec = p.flatten()
    p2 = EmbeddingParams.from_flat(vec)
    assert np.allclose(vec, p2.flatten())
    for name, _shape in E._LAYOUT:
        assert np.allclose(getattr(p, name), getattr(p2, name))


def test_forward_shape_and_range():
    nomic = _fake_nomic(["the", "king", "dead", "sleep", "crown"])
    m = EmbeddingModel(EmbeddingParams.random_init(0), nomic=nomic)
    out = m.embed("king", ["the", "dead"])
    assert out is not None
    assert out.shape == (12,)
    # embedding dims are sigmoid outputs -> (0,1); POS dim too.
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert out.shape[0] == D_EMB + 1


def test_oov_current_word_returns_none():
    nomic = _fake_nomic(["the", "king"])
    m = EmbeddingModel(EmbeddingParams.random_init(0), nomic=nomic)
    assert m.embed("nonexistentword", ["the"]) is None


def test_history_padding_and_determinism():
    nomic = _fake_nomic(["the", "king", "dead"])
    m = EmbeddingModel(EmbeddingParams.random_init(0), nomic=nomic)
    a = m.embed("king", [])                 # empty history (all padded)
    b = m.embed("king", [])
    assert np.allclose(a, b)                # deterministic
    # padding to 5 vs explicit short history of OOV words behaves the same
    c = m.embed("king", ["the", "dead"])
    assert not np.allclose(a, c)            # history changes the output


def test_history_actually_changes_embedding():
    nomic = _fake_nomic(["the", "king", "dead", "queen"])
    m = EmbeddingModel(EmbeddingParams.random_init(7), nomic=nomic)
    e1 = m.embed("king", ["dead"])
    e2 = m.embed("king", ["queen"])
    assert not np.allclose(e1, e2)          # same word, different memory -> different taste


def test_set_genome_clears_cache():
    nomic = _fake_nomic(["the", "king"])
    m = EmbeddingModel(EmbeddingParams.random_init(0), nomic=nomic)
    before = m.embed("king", ["the"]).copy()
    m.set_genome(EmbeddingParams.random_init(99).flatten())
    after = m.embed("king", ["the"])
    assert not np.allclose(before, after)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
