"""Tests for the generations-viewer flask resolution (server/app.py).

The 8 poetry flasks live in 4 separate processes, each with its own
WORMLET_DATA_DIR. The viewer used to resolve flask names under only its OWN
GENERATIONS_ROOT, so each port could page through just its 2 flasks. These
tests pin the cross-process resolution, the 1..8 labelling, and the fact that
a qualified name survives URL routing.
"""
import contextlib
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _app_with_data_dir(tmp: Path):
    """(Re)import server.app with WORMLET_DATA_DIR pointed at a fixture tree."""
    os.environ["WORMLET_DATA_DIR"] = str(tmp)
    os.environ["WORMLET_GENERATIONS"] = "0"
    import server.generations as g
    importlib.reload(g)
    import server.app as A
    importlib.reload(A)
    return A


@contextlib.contextmanager
def _fixture(procs=4, flasks=2, gens=3, sub="poetry-1", extra=None):
    """A temp data tree + an app bound to it, with the process environment and
    the reloaded modules restored on exit.

    Without the restore these tests would leave WORMLET_DATA_DIR pointing at a
    deleted temp dir, which quietly breaks whatever runs next in the suite."""
    saved = {k: os.environ.get(k) for k in ("WORMLET_DATA_DIR", "WORMLET_GENERATIONS")}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        _make_tree(root, procs=procs, flasks=flasks, gens=gens)
        if extra:
            extra(root)
        try:
            yield _app_with_data_dir(root / sub), root
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


def _make_tree(root: Path, procs=4, flasks=2, gens=3):
    """data/poetry-N/generations/flask_M/gen-XXXX/"""
    for p in range(1, procs + 1):
        for f in range(1, flasks + 1):
            fd = root / f"poetry-{p}" / "generations" / f"flask_{f}"
            for n in range(1, gens + 1):
                (fd / f"gen-{n:04d}").mkdir(parents=True, exist_ok=True)
            (fd / "state.json").write_text('{"generation": %d, "sigma": 0.1}' % gens)


def test_sees_all_eight_flasks_from_one_process():
    with _fixture() as (A, _root):
        names = A._gv_flask_names()
        assert len(names) == 8, names
        assert names[0] == f"poetry-1{A.FLASK_SEP}flask_1"
        assert names[-1] == f"poetry-4{A.FLASK_SEP}flask_2"


def test_all_eight_visible_from_every_process():
    """Any of the four ports must show the whole run, not just its own pair."""
    for sub in ("poetry-1", "poetry-2", "poetry-3", "poetry-4"):
        with _fixture(sub=sub) as (A, _root):
            assert len(A._gv_flask_names()) == 8, sub


def test_labels_number_flasks_one_to_eight():
    with _fixture() as (A, _root):
        labels = [A._gv_label(n) for n in A._gv_flask_names()]
        for i, lab in enumerate(labels, start=1):
            assert lab.startswith(f"flask {i} "), (i, lab)


def test_qualified_name_has_no_slash():
    """A '/' in the id would be eaten by the /{flask}/{gen} route."""
    with _fixture() as (A, _root):
        assert A.FLASK_SEP != "/"
        assert all("/" not in n for n in A._gv_flask_names())


def test_each_name_resolves_to_its_own_dir():
    with _fixture() as (A, _root):
        for n in A._gv_flask_names():
            tag, local = n.split(A.FLASK_SEP)
            d = A._gv_dir(n)
            assert d.is_dir()
            assert d.name == local and d.parent.parent.name == tag


def test_unknown_and_traversal_names_resolve_nowhere():
    with _fixture() as (A, _root):
        for bad in ("nope", "../../etc", "..", "poetry-9:flask_1", ""):
            assert not A._gv_dir(bad).exists(), bad


def test_non_poetry_process_stays_local():
    """Experiment-mode runs have their own data dir and must NOT fan out."""
    def add_exp(root):
        (root / "words-exp" / "generations" / "words" / "gen-0001").mkdir(parents=True)
    with _fixture(sub="words-exp", extra=add_exp) as (A, _root):
        assert A._gv_flask_names() == ["words"]
        assert A._gv_dir("words").is_dir()


def test_http_routes_accept_qualified_names():
    from fastapi.testclient import TestClient
    with _fixture() as (A, _root):
        c = TestClient(A.app)
        r = c.get("/api/generations")
        assert r.status_code == 200
        flasks = r.json()["flasks"]
        assert len(flasks) == 8
        assert {"name", "label", "current_generation", "sigma"} <= set(flasks[0])
        for f in flasks:
            assert c.get(f"/api/generations/{f['name']}").status_code == 200


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
