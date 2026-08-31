"""The graveyard endpoint: deaths.jsonl + the judge's own favourite window.

Two files have to line up for an obituary — the death record in the LIVE tree
(<data>/flasks/<flask>/deaths.jsonl) and the scored poem in the generations
tree (<data>/generations/<flask>/gen-NNNN/<worm>/scores.jsonl). These pin that
join, and the fallbacks for the case that matters most in practice: a worm
that died in a generation the judge has not scored yet.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@contextlib.contextmanager
def _fixture(tree):
    """Reimport server.app against a temp data dir built by `tree(root)`."""
    saved = {k: os.environ.get(k) for k in ("WORMLET_DATA_DIR", "WORMLET_GENERATIONS")}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        tree(root)
        os.environ["WORMLET_DATA_DIR"] = str(root)
        os.environ["WORMLET_GENERATIONS"] = "0"
        try:
            import server.generations as g
            importlib.reload(g)
            import server.app as A
            importlib.reload(A)
            yield A
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            import server.generations as g
            importlib.reload(g)
            import server.app as A
            importlib.reload(A)


def _death(worm, generation, ts, **kw):
    rec = {
        "ts": ts, "flask": "flask_1", "worm": worm, "generation": generation,
        "cause": "starvation", "died_at_tick": 113163, "starved_s": 600.0,
        "word_count": 64, "x": 1224.1, "y": 451.1,
        "recent_eaten": ["若", "兮", "豫", "：", "容"],
        "plasticity": {"edges": 2962, "capped": 1674},
    }
    rec.update(kw)
    return json.dumps(rec)


def _tree(root: Path):
    flasks = root / "flasks" / "flask_1"
    gens = root / "generations" / "flask_1"
    (flasks).mkdir(parents=True, exist_ok=True)
    (flasks / "deaths.jsonl").write_text("\n".join([
        _death("shen kuo", 3, "2026-08-15T20:04:13Z"),
        _death("cai lun", 1, "2026-08-15T12:32:19Z"),
        _death("ghost", 9, "2026-08-16T01:00:00Z"),      # generation never judged
    ]) + "\n")
    # gen 3 was judged: three windows, the middle one rated highest.
    wd = gens / "gen-0003" / "shen kuo"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "scores.jsonl").write_text("\n".join([
        json.dumps({"idx": 0, "tokens": ["dull", "words"], "emotional": 3, "coherence": 4}),
        json.dumps({"idx": 30, "tokens": ["the", "welkin", "-", "sent", "terror"],
                    "emotional": 23, "coherence": 27}),
        json.dumps({"idx": 60, "tokens": ["middling"], "emotional": 10, "coherence": 10}),
    ]) + "\n")
    (gens / "gen-0003" / "metadata.json").write_text(json.dumps({"corpus": "beowulf"}))
    (gens / "gen-0001").mkdir(parents=True, exist_ok=True)


def _graves(A):
    from fastapi.testclient import TestClient
    return TestClient(A.app).get("/api/graveyard").json()


def test_epitaph_is_the_judges_highest_rated_window():
    """"Most poetic" needs no new judgement from us: the critic already rated
    every window, so the epitaph is simply its favourite."""
    with _fixture(_tree) as A:
        by_worm = {g["worm"]: g for g in _graves(A)["graves"]}
        ep = by_worm["shen kuo"]["epitaph"]
        assert ep["tokens"] == ["the", "welkin", "-", "sent", "terror"]
        assert (ep["emotional"], ep["coherence"]) == (23, 27)


def test_epitaph_falls_back_to_an_earlier_generation_then_to_the_final_lines():
    """The case that matters most: nearly every fresh grave is a worm that
    died in the generation now running, which the judge only scores at
    rollover. The obvious lookup finds nothing for exactly the newest graves,
    so the chain has to reach back — and say which rung it used, since the
    critic never saw the unjudged ones."""
    def tree(root: Path):
        _tree(root)
        # A worm scored in gen 1 but not in the gen it died in (3).
        wd = root / "generations" / "flask_1" / "gen-0001" / "revenant"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "scores.jsonl").write_text(json.dumps(
            {"idx": 0, "tokens": ["remembered", "lines"],
             "emotional": 9, "coherence": 9}) + "\n")
        (root / "flasks" / "flask_1" / "deaths.jsonl").write_text("\n".join([
            _death("revenant", 3, "2026-08-16T02:00:00Z"),
            _death("ghost", 9, "2026-08-16T01:00:00Z"),
        ]) + "\n")
        # The unjudged worm still has the poem it was writing when it died.
        pd = root / "flasks" / "flask_1" / "ghost"
        pd.mkdir(parents=True, exist_ok=True)
        (pd / "poem.txt").write_text("and then the light went out of it\n")

    with _fixture(tree) as A:
        by_worm = {g["worm"]: g for g in _graves(A)["graves"]}
        rev = by_worm["revenant"]["epitaph"]
        assert rev["source"] == "judged-earlier" and rev["generation"] == 1
        assert rev["tokens"] == ["remembered", "lines"]

        ghost = by_worm["ghost"]["epitaph"]
        assert ghost["source"] == "final-lines"
        assert ghost["tokens"][-3:] == ["out", "of", "it"]
        assert ghost["emotional"] is None, "unjudged lines must not carry a score"
        assert by_worm["ghost"]["last_words"] == ["若", "兮", "豫", "：", "容"]


def test_graves_are_newest_first_with_provenance():
    with _fixture(_tree) as A:
        data = _graves(A)
        assert data["total"] == 3
        assert [g["worm"] for g in data["graves"]] == ["ghost", "shen kuo", "cai lun"]
        # Corpus comes from that generation's metadata, not from this
        # process's env — a sibling flask's text must not read as hamlet.
        assert by_gen(data, "shen kuo")["corpus"] == "beowulf"
        g = by_gen(data, "shen kuo")
        assert g["words_eaten"] == 64 and g["cause"] == "starvation"
        assert g["plasticity_capped"] == 1674 and g["plasticity_edges"] == 2962


def by_gen(data, worm):
    return next(g for g in data["graves"] if g["worm"] == worm)


def test_no_deaths_is_an_empty_register_not_an_error():
    def empty(root: Path):
        (root / "flasks" / "flask_1").mkdir(parents=True, exist_ok=True)
        (root / "generations" / "flask_1" / "gen-0001").mkdir(parents=True, exist_ok=True)

    with _fixture(empty) as A:
        data = _graves(A)
        assert data == {"total": 0, "graves": []}
