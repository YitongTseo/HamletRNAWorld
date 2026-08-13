"""Tests for the chemosensory encoder (server/embedding.py), BOTH modes.

Default mode is "umap" (frozen v6 coordinates); "learned" keeps the v7
co-evolved net alive for A/B. PC11 is the deterministic POS grammar in both.

Every test that exercises the learned net must pass mode=ENCODER_LEARNED
explicitly — otherwise it silently runs the UMAP path and the injected fake
nomic table is ignored.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import embedding as E
from server.embedding import (EmbeddingParams, EmbeddingModel, GENOME_SIZE,
                              D_EMB, ENCODER_LEARNED, ENCODER_UMAP)


def _fake_nomic(words, seed=1):
    rng = np.random.default_rng(seed)
    return {w: rng.standard_normal(E.D_IN) for w in words}


def _learned(seed=0, words=("the", "king")):
    """A learned-mode model over a fake nomic table."""
    return EmbeddingModel(EmbeddingParams.random_init(seed),
                          nomic=_fake_nomic(list(words)), mode=ENCODER_LEARNED)


def test_genome_size_matches_layout():
    # The POS sub-net (72*16+16 + 16*1+1 = 1185) was removed from the genome
    # when PC11 became the deterministic corpus grammar.
    assert GENOME_SIZE == 512*11+11 + 55*11+11 + 22*11+11
    assert GENOME_SIZE == 6512
    p = EmbeddingParams.random_init(0)
    assert p.flatten().shape == (GENOME_SIZE,)


def test_legacy_genome_is_accepted_and_tail_ignored():
    """Old embedder.json files are 7697 long (POS sub-net appended). Those
    blocks were last in the layout, so the prefix is still a valid genome."""
    p = EmbeddingParams.random_init(0)
    legacy = np.concatenate([p.flatten(), np.random.default_rng(0)
                             .standard_normal(E.LEGACY_GENOME_SIZE - GENOME_SIZE)])
    assert legacy.shape == (E.LEGACY_GENOME_SIZE,)
    restored = EmbeddingParams.from_flat(legacy)
    assert np.allclose(restored.flatten(), p.flatten())


def test_bad_genome_size_still_rejected():
    try:
        EmbeddingParams.from_flat(np.zeros(123))
    except ValueError:
        return
    raise AssertionError("expected ValueError on a wrong-sized genome")


def test_flatten_unflatten_roundtrip():
    p = EmbeddingParams.random_init(3)
    vec = p.flatten()
    p2 = EmbeddingParams.from_flat(vec)
    assert np.allclose(vec, p2.flatten())
    for name, _shape in E._LAYOUT:
        assert np.allclose(getattr(p, name), getattr(p2, name))


def test_forward_shape_and_range():
    m = _learned(0, ["the", "king", "dead", "sleep", "crown"])
    out = m.embed("king", ["the", "dead"])
    assert out is not None
    assert out.shape == (12,)
    # embedding dims are sigmoid outputs -> (0,1); POS dim too.
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert out.shape[0] == D_EMB + 1


def test_oov_current_word_returns_none():
    m = _learned(0, ["the", "king"])
    assert m.embed("nonexistentword", ["the"]) is None


def test_history_padding_and_determinism():
    m = _learned(0, ["the", "king", "dead"])
    a = m.embed("king", [])                 # empty history (all padded)
    b = m.embed("king", [])
    assert np.allclose(a, b)                # deterministic
    # padding to 5 vs explicit short history of OOV words behaves the same
    c = m.embed("king", ["the", "dead"])
    assert not np.allclose(a, c)            # history changes the output


def test_history_actually_changes_embedding():
    m = _learned(7, ["the", "king", "dead", "queen"])
    e1 = m.embed("king", ["dead"])
    e2 = m.embed("king", ["queen"])
    assert not np.allclose(e1, e2)          # same word, different memory -> different taste


def test_set_genome_clears_cache():
    m = _learned(0, ["the", "king"])
    before = m.embed("king", ["the"]).copy()
    m.set_genome(EmbeddingParams.random_init(99).flatten())
    after = m.embed("king", ["the"])
    assert not np.allclose(before, after)


def test_prime_builds_table():
    words = ["the", "king", "queen", "sword", "ghost"]
    m = EmbeddingModel(EmbeddingParams.random_init(0), _fake_nomic(words), mode=ENCODER_LEARNED)
    assert m._E_table is not None
    assert m._E_table.shape == (len(words), D_EMB)
    # table row == ReLU(vec @ W_E + b_E), i.e. == _encode()
    for w in words:
        assert np.allclose(m._E_table[m._widx[w]], m._encode(w))


def test_embed_batch_matches_per_word():
    words = [f"w{i}" for i in range(30)]
    m = EmbeddingModel(EmbeddingParams.random_init(2), _fake_nomic(words), mode=ENCODER_LEARNED)
    cur = words[:12] + ["OOV_not_a_word"]          # include an OOV current word
    for hist in ([], ["w3"], ["w3", "w7", "OOVhist", "w1", "w9"]):
        out, valid = m.embed_batch(cur, hist)
        assert out.shape == (len(cur), D_EMB + 1)
        for i, w in enumerate(cur):
            single = m.embed(w, hist)
            if single is None:
                assert not valid[i]
            else:
                assert valid[i]
                assert np.allclose(out[i], single, atol=1e-9)


def test_embed_batch_empty_and_rebuild():
    words = ["alpha", "beta", "gamma"]
    m = EmbeddingModel(EmbeddingParams.random_init(5), _fake_nomic(words), mode=ENCODER_LEARNED)
    out, valid = m.embed_batch([], ["alpha"])
    assert out.shape == (0, D_EMB + 1) and valid.shape == (0,)
    # after a genome swap, the batched path reflects the new table
    a, _ = m.embed_batch(["alpha"], [])
    m.set_genome(EmbeddingParams.random_init(6).flatten())
    b, _ = m.embed_batch(["alpha"], [])
    assert not np.allclose(a, b)


# --- UMAP mode (the default) ----------------------------------------------

def _umap_model():
    return EmbeddingModel(EmbeddingParams.random_init(0), mode=ENCODER_UMAP)


def test_default_mode_is_umap():
    """With no override, the encoder must default to UMAP. If the suite is
    being run under an explicit WORMLET_ENCODER (the learned-path A/B), assert
    that the override is honoured instead."""
    import os
    override = (os.environ.get("WORMLET_ENCODER") or "").strip().lower()
    if override in ("", "umap"):
        assert E.encoder_mode() == ENCODER_UMAP
    else:
        assert E.encoder_mode() == override
    # an unrecognised value must fall back to UMAP, never crash
    prev = os.environ.get("WORMLET_ENCODER")
    os.environ["WORMLET_ENCODER"] = "nonsense"
    try:
        assert E.encoder_mode() == ENCODER_UMAP
    finally:
        if prev is None:
            del os.environ["WORMLET_ENCODER"]
        else:
            os.environ["WORMLET_ENCODER"] = prev


def test_umap_forward_shape_and_range():
    m = _umap_model()
    out = m.embed("king", ["the"])
    assert out is not None and out.shape == (D_EMB + 1,)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_umap_oov_returns_none():
    m = _umap_model()
    assert m.embed("zzqxwvnotaword", ["the"]) is None


def test_umap_channels_use_the_full_range():
    """The regression that motivated the revert: the learned encoder's
    channels had std ~0.027 across the vocabulary, over the range 0.40-0.55.
    UMAP must restore real between-word contrast."""
    m = _umap_model()
    words = list(m._umap.keys())[:2000]
    rows = np.array([m._umap[w] for w in words])
    stds = rows.std(axis=0)
    assert stds.mean() > 0.08, f"channel contrast collapsed: {stds.mean():.4f}"
    assert rows.min() < 0.05 and rows.max() > 0.95


def test_umap_word_identity_dominates_not_a_dc_offset():
    """Two different words must be far apart relative to the channel spread."""
    m = _umap_model()
    a, b = m.embed("king", []), m.embed("sword", [])
    assert a is not None and b is not None
    assert np.abs(a[:D_EMB] - b[:D_EMB]).max() > 0.1


def test_umap_history_changes_only_the_pos_channel():
    """The UMAP coordinates are frozen per word, so history moves PC11 alone —
    the semantic channels are context-free by construction in this mode."""
    m = _umap_model()
    a = m.embed("king", ["the"])
    b = m.embed("king", ["sword"])
    assert np.allclose(a[:D_EMB], b[:D_EMB])
    assert not np.isclose(a[D_EMB], b[D_EMB])


def test_umap_set_genome_is_a_noop():
    m = _umap_model()
    before = m.embed("king", ["the"]).copy()
    m.set_genome(EmbeddingParams.random_init(99).flatten())
    assert np.allclose(before, m.embed("king", ["the"]))


def test_umap_embed_batch_matches_per_word():
    m = _umap_model()
    words = list(m._umap.keys())[:20]
    cur = words[:12] + ["zzqxwvnotaword"]
    for hist in ([], ["the"], ["the", "king", "sword"]):
        out, valid = m.embed_batch(cur, hist)
        assert out.shape == (len(cur), D_EMB + 1)
        for i, w in enumerate(cur):
            single = m.embed(w, hist)
            if single is None:
                assert not valid[i]
            else:
                assert valid[i]
                assert np.allclose(out[i], single, atol=1e-9)


def test_umap_pc11_is_the_pos_grammar():
    from server import pos_grammar
    m = _umap_model()
    g = pos_grammar.get_grammar()
    for w in ("king", "the", "sword"):
        for hist in ([], ["the"], ["the", "in"]):
            assert np.isclose(m.embed(w, hist)[D_EMB], g.fit_word(w, hist))


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
