"""Tests for the gardener's cost controls (2026-09-21). No Claude API calls.

Two things are being defended here:
  * the epoch cadence (WORMLET_GARDENER_EVERY_N_EPOCHS) — three Opus calls
    carrying every worm's metrics are the most expensive thing a rollover
    does, and nothing selects on their output;
  * the cache breakpoint on the per-epoch context — the three rounds share a
    byte-identical prefix, and anything varying per round must sit AFTER the
    breakpoint or the prefix stops caching and the saving silently vanishes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import gardener


def _set(n: str | None):
    if n is None:
        os.environ.pop("WORMLET_GARDENER_EVERY_N_EPOCHS", None)
    else:
        os.environ["WORMLET_GARDENER_EVERY_N_EPOCHS"] = n


def test_gardener_cadence_default_is_every_epoch():
    """Default must be the old behaviour — the cost knob is opt-in."""
    _set(None)
    assert gardener._gardener_every_n() == 1
    assert all(gardener._gardener_epoch_due(e) for e in range(1, 10))


def test_gardener_cadence_every_second_epoch():
    _set("2")
    try:
        due = [e for e in range(1, 11) if gardener._gardener_epoch_due(e)]
        # Epoch 1 always writes — the first log of a lineage is the one worth
        # having — then every second epoch after that.
        assert due == [1, 2, 4, 6, 8, 10]
    finally:
        _set(None)


def test_gardener_cadence_junk_never_silences_the_gardener():
    """A typo in a drop-in must not stop the log for good."""
    for junk in ("", "banana", "0", "-4"):
        _set(junk)
        try:
            assert gardener._gardener_every_n() == 1
            assert gardener._gardener_epoch_due(7)
        finally:
            _set(None)


def test_deferred_epoch_makes_no_api_calls_and_is_not_recorded_as_a_PASS():
    """The whole point is not spending: a deferred epoch must reach no client
    at all. And the marker must be distinguishable from the gardener's PASS —
    `_read_meta_log` only reads gardeners_log.md, and a `.skipped` file is the
    gardener's own decision to rest, which this is not."""
    import tempfile

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("deferred epoch must not construct a client")

    class _FakeFlask:
        name = "flask_1"
        worms: list = []
        state = None

    real_client = gardener.anthropic.Anthropic
    real_root = gardener._generations_root
    _set("2")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-a-real-key")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "generations"
        gardener.anthropic.Anthropic = _Boom
        gardener._generations_root = lambda: root
        try:
            out = gardener.maybe_write_meta_log([_FakeFlask()], generation_num=3)
        finally:
            gardener.anthropic.Anthropic = real_client
            gardener._generations_root = real_root
            _set(None)
        assert out is None
        epoch_dir = root / "meta" / "gen-0003"
        assert (epoch_dir / "gardeners_log.deferred").exists()
        assert not (epoch_dir / "gardeners_log.skipped").exists()
        assert not (epoch_dir / "gardeners_log.md").exists()


def test_meta_gardener_caches_the_shared_prefix_and_not_the_tail():
    """Round 1 writes the cache, rounds 2 and 3 must read it: the metrics +
    logs prefix is one block with cache_control, and the poems/instructions
    that change per round land in a second, uncached block."""
    calls: list[dict] = []

    class _FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            raise RuntimeError("stop after recording the request")

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    class _FakeWorm:
        name = "Alice"
        last_fitness = 1.0
        word_count = 100
        recent_words: list[str] = []

    class _FakeFlask:
        name = "flask_1"
        worms = [_FakeWorm()]
        state = None

    real_client = gardener.anthropic.Anthropic
    real_metrics = gardener._format_all_flasks_full_metrics
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-a-real-key")
    gardener.anthropic.Anthropic = _FakeClient
    gardener._format_all_flasks_full_metrics = lambda flasks: "METRICS BLOCK"
    try:
        # Rounds 1 and 2 swallow their own exceptions and round 3 returns None,
        # so this records three requests and writes nothing.
        gardener.maybe_write_meta_log([_FakeFlask()], generation_num=3)
    finally:
        gardener.anthropic.Anthropic = real_client
        gardener._format_all_flasks_full_metrics = real_metrics

    assert len(calls) == 3, f"expected 3 rounds, recorded {len(calls)}"
    prefixes = set()
    for kw in calls:
        blocks = kw["messages"][0]["content"]
        assert isinstance(blocks, list), "user content must be blocks, not a string"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "METRICS BLOCK" in blocks[0]["text"]
        assert "cache_control" not in blocks[-1], "the varying tail must not be cached"
        prefixes.add(blocks[0]["text"])
    assert len(prefixes) == 1, "the cached prefix differed between rounds — no cache hits"
