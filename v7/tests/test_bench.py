"""Tests for the tasting bench — ballot discovery, verdict validation, and
the round trip into fitness.

No network: OLLAMA_MODEL is left unset throughout, so the coherence pass is
skipped and the bench falls back to its deterministic random shortlist.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server.judge as judge
from server.evolution import fitness

TOKENS = (("him night of him apparition come story speak we all from it "
           "that's Horatio dead and them Stand to you Give farewell mouse "
           "stirring").split() * 35)
WORMS = [f"worm_{i}" for i in range(10)]


class _Bench:
    """A temp lineage with one generation of ballots already filed, and the
    bench module imported against it."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = {k: os.environ.get(k) for k in
                     ("WORMLET_DATA_DIR", "WORMLET_JUDGE_BACKEND",
                      "WORMLET_BENCH_OLLAMA_MODEL", "WORMLET_BENCH_SHOW")}
        os.environ["WORMLET_DATA_DIR"] = self._tmp.name
        os.environ["WORMLET_JUDGE_BACKEND"] = "human"
        os.environ.pop("WORMLET_BENCH_OLLAMA_MODEL", None)
        os.environ["WORMLET_BENCH_SHOW"] = "5"
        for w in WORMS:
            try:
                judge.judge_poem(TOKENS, worm_name=w, seed=256,
                                 flask="flask_1", generation=256)
            except judge.JudgeAwaitingHuman:
                pass
        import bench.app as bench_app
        self.bench = importlib.reload(bench_app)
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()
        return False


def _post(bench, payload):
    """Call the verdict route without an HTTP server."""
    import asyncio

    class _Req:
        async def json(self):
            return payload

    return asyncio.run(bench.verdict(_Req()))


def _status(resp):
    return getattr(resp, "status_code", 200)


def test_pending_finds_the_unjudged_generation():
    with _Bench():
        from bench.app import _pending
        flask, gen, gen_dir = _pending()
        assert (flask, gen) == ("flask_1", 256)
        assert len(list(gen_dir.glob("*.json"))) == len(WORMS)


def test_pending_prefers_the_oldest_unjudged():
    """Applying a verdict to the wrong generation's genomes is silent and
    unrecoverable, so order matters even though it should never happen."""
    with _Bench():
        for w in WORMS[:2]:
            try:
                judge.judge_poem(TOKENS, worm_name=w, seed=257,
                                 flask="flask_1", generation=257)
            except judge.JudgeAwaitingHuman:
                pass
        from bench.app import _pending
        assert _pending()[1] == 256


def test_page_renders_blind_and_shows_the_configured_count():
    with _Bench() as root:
        from bench.app import bench, SHOW_N
        page = bench().body.decode()
        assert "generation 256" in page
        # Blind: letters shown, no worm name in a heading.
        assert "<h2>A</h2>" in page and "<h2>J</h2>" in page
        assert "<h2>worm_0</h2>" not in page
        # Exactly SHOW_N lines offered per worm.
        assert page.count("<li data-idx=") == SHOW_N * len(WORMS)
        # The full ballot is still on disk regardless of what was shown.
        b = json.loads((root / "ballots" / "flask_1" / "gen-0256"
                        / "worm_0.json").read_text())
        assert len(b["windows"]) > SHOW_N


def test_shortlist_is_not_just_the_opening_lines():
    """Showing the first N would judge every worm on how its poem starts."""
    with _Bench():
        from bench.app import _shortlist, _load_ballots, _pending
        _f, _g, gen_dir = _pending()
        firsts = []
        for b in _load_ballots(gen_dir):
            short = _shortlist(b, {}, 256)
            firsts.append(short[0]["idx"] == b["windows"][0]["idx"])
        assert not all(firsts), "shortlist always starts at the poem's start"


def test_shortlist_is_deterministic_across_reloads():
    with _Bench():
        from bench.app import _shortlist, _load_ballots, _pending
        _f, _g, gen_dir = _pending()
        b = _load_ballots(gen_dir)[0]
        assert ([w["idx"] for w in _shortlist(b, {}, 256)]
                == [w["idx"] for w in _shortlist(b, {}, 256)])


def test_verdict_round_trips_into_fitness_order():
    with _Bench() as root:
        import bench.app as bench
        top = ["worm_3", "worm_8", "worm_0"]
        # Star two windows that are genuinely on worm_3's ballot. Offsets are
        # a 25% SAMPLE of the strides, so a plausible-looking 15 or 45 is
        # usually absent — which is exactly the bug this now guards.
        b3 = json.loads((root / "ballots" / "flask_1" / "gen-0256"
                         / "worm_3.json").read_text())
        keep = [b3["windows"][0]["idx"], b3["windows"][3]["idx"]]
        r = _post(bench, {"flask": "flask_1", "generation": 256, "top": top,
                          "starred": {"worm_3": keep}})
        assert r["ok"] and r["revealed"] == top
        scored = {}
        for w in WORMS:
            scored[w] = judge.judge_poem(TOKENS, worm_name=w, seed=256,
                                         flask="flask_1", generation=256)
        fits = {w: fitness(s) for w, s in scored.items()}
        assert sorted(fits, key=lambda w: -fits[w])[:3] == top
        # Starred lines land in the commonplace book with their text.
        rows = [json.loads(l) for l in
                (root / "commonplace.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(r["text"] for r in rows)
        assert {r["idx"] for r in rows} == set(keep)


def test_verdict_rejects_a_star_not_on_the_ballot():
    with _Bench():
        import bench.app as bench
        r = _post(bench, {"flask": "flask_1", "generation": 256,
                          "top": ["worm_3"], "starred": {"worm_3": [999999]}})
        assert _status(r) == 400


def test_verdict_rejects_a_worm_not_on_the_ballot():
    """A stale or mistyped name would score nobody and silently flatten the
    whole field into a tie."""
    with _Bench():
        import bench.app as bench
        r = _post(bench, {"flask": "flask_1", "generation": 256,
                          "top": ["worm_3", "ghost"]})
        assert _status(r) == 400


def test_verdict_rejects_duplicates_and_empties():
    with _Bench():
        import bench.app as bench
        assert _status(_post(bench, {"flask": "flask_1", "generation": 256,
                                     "top": ["worm_3", "worm_3"]})) == 400
        assert _status(_post(bench, {"flask": "flask_1", "generation": 256,
                                     "top": []})) == 400


def test_verdict_refuses_to_overwrite():
    """Re-judging a generation the rollover may already have selected on."""
    with _Bench():
        import bench.app as bench
        assert _post(bench, {"flask": "flask_1", "generation": 256,
                             "top": ["worm_3"]})["ok"]
        assert _status(_post(bench, {"flask": "flask_1", "generation": 256,
                                     "top": ["worm_8"]})) == 409


def test_coherence_path_with_a_model_configured():
    """Exercises the branch that only runs when a local model IS set.

    Added after that branch was left with a `return out, err` whose `err`
    had been deleted — a NameError that would have fired the first time the
    Ollama box was turned on, and that no test could reach while every test
    left the model unset. Stubs the HTTP POST, so still no network.
    """
    with _Bench() as root:
        os.environ["WORMLET_BENCH_OLLAMA_MODEL"] = "gemma3:4b"
        import bench.app as bench_app
        bench = importlib.reload(bench_app)

        calls = []

        def _fake_post(url, body, headers, timeout):
            calls.append(body["model"])
            n = len(body["messages"][1]["content"].splitlines()) - 1
            return {"choices": [{"message": {"content": "\n".join(
                f"{i},50,{60 + i}" for i in range(n))}}]}

        real, judge._http_post_json = judge._http_post_json, _fake_post
        try:
            page = bench.bench().body.decode()
        finally:
            judge._http_post_json = real
        assert "No local model" not in page, "banner shown despite a working model"
        assert calls and set(calls) == {"gemma3:4b"}, \
            f"model not passed through: {calls}"
        # Scores cached so a reload neither re-asks nor gives a second answer.
        assert (root / "ballots" / "flask_1" / "gen-0256"
                / ".coherence.json").exists()


def test_bench_degrades_when_the_model_is_unreachable():
    """A sleeping desktop must cost a banner, not the page."""
    with _Bench():
        os.environ["WORMLET_BENCH_OLLAMA_MODEL"] = "gemma3:4b"
        import bench.app as bench_app
        bench = importlib.reload(bench_app)

        def _boom(*a, **k):
            raise OSError("timed out")

        real, judge._http_post_json = judge._http_post_json, _boom
        try:
            page = bench.bench().body.decode()
        finally:
            judge._http_post_json = real
        assert "No local model" in page, "no banner for an unreachable model"
        assert page.count("<li data-idx=") == 5 * len(WORMS), \
            "page did not fall back to a shortlist"


def test_judged_generation_disappears_from_pending():
    with _Bench():
        import bench.app as bench
        _post(bench, {"flask": "flask_1", "generation": 256,
                      "top": ["worm_3"]})
        assert bench._pending() is None
