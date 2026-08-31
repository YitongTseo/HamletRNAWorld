"""Tests for WORMLET_PUNCT_SMELL: punctuation as a grammar-only smell.
Default off must match stock exactly; on, marks carry PC11 and nothing else."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import embedding
from server.pos_grammar import get_grammar


def _with_flag(value):
    old = os.environ.get("WORMLET_PUNCT_SMELL")
    if value is None:
        os.environ.pop("WORMLET_PUNCT_SMELL", None)
    else:
        os.environ["WORMLET_PUNCT_SMELL"] = value
    return old


def _restore(old):
    if old is None:
        os.environ.pop("WORMLET_PUNCT_SMELL", None)
    else:
        os.environ["WORMLET_PUNCT_SMELL"] = old


def test_flag_off_punctuation_stays_invisible():
    old = _with_flag(None)
    try:
        m = embedding.get_model()
        out, valid = m.embed_batch([",", ".", "king"], ["the"])
        assert not valid[0] and not valid[1]  # marks invisible, stock
        assert valid[2]
        assert m.embed(".", ["the"]) is None
    finally:
        _restore(old)


def test_flag_on_marks_smell_of_pure_grammar():
    old = _with_flag("1")
    try:
        m = embedding.get_model()
        hist = ["king", "the"]  # most-recent-first: "... the king"
        out, valid = m.embed_batch([".", "king"], hist)
        assert valid[0], "punctuation should be smellable"
        assert np.all(out[0, :11] == 0.0), "semantic channels must stay zero"
        g = get_grammar()
        expected = g.context_distribution(hist).get(g.tag("."), 0.0)
        assert abs(out[0, 11] - expected) < 1e-12
        assert expected > 0.9, "a full stop after 'the king' should fit well"
    finally:
        _restore(old)


def test_flag_on_real_words_unchanged():
    hist = ["king", "the"]
    old = _with_flag(None)
    try:
        m = embedding.get_model()
        off, _ = m.embed_batch(["king", "night"], hist)
    finally:
        _restore(old)
    old = _with_flag("1")
    try:
        on, _ = m.embed_batch(["king", "night"], hist)
        assert np.array_equal(off, on)
    finally:
        _restore(old)


def test_embed_single_matches_batch_for_punctuation():
    old = _with_flag("1")
    try:
        m = embedding.get_model()
        hist = ["sleep", "to"]
        single = m.embed(",", hist)
        batch, valid = m.embed_batch([","], hist)
        assert single is not None and valid[0]
        assert np.allclose(single, batch[0])
    finally:
        _restore(old)


def test_oov_non_punctuation_still_invisible_with_flag_on():
    old = _with_flag("1")
    try:
        m = embedding.get_model()
        out, valid = m.embed_batch(["zzgrobnak"], [])
        assert not valid[0]  # OOV gibberish is not punctuation; stays skipped
    finally:
        _restore(old)
