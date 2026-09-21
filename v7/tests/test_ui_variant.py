"""The two-viewer seam (server/ui_variant.py).

The load-bearing invariant is the one in test_no_static_refs_outside_html:
`page()` rewrites `/static/` -> `/static/<variant>/` only in the HTML it
serves, so if a .js or .css file ever hardcodes `/static/...` it would
silently load the OTHER variant's asset. That failure is invisible in
classic (the default alias resolves) and only shows up under ?ui=vivarium,
so it is worth a test rather than a comment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import ui_variant  # noqa: E402

V7 = Path(__file__).resolve().parent.parent


class _FakeRequest:
    def __init__(self, **params):
        self.query_params = params


def test_both_trees_exist_and_carry_every_page():
    pages = ["index.html", "focus.html", "poems.html", "about.html",
             "generations.html"]
    for variant, tree in ui_variant.TREES.items():
        assert tree.is_dir(), f"{variant} tree missing: {tree}"
        for page in pages:
            assert (tree / page).is_file(), f"{variant} is missing {page}"


def test_no_static_refs_outside_html():
    """/static/ may appear ONLY in HTML, where page() can rewrite it."""
    offenders = []
    for tree in ui_variant.TREES.values():
        for f in tree.rglob("*"):
            if f.suffix not in (".js", ".css") or not f.is_file():
                continue
            if f.name == "chart.umd.js":       # vendored bundle, not ours
                continue
            if "/static/" in f.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(f.relative_to(V7)))
    assert not offenders, (
        "hardcoded /static/ outside HTML — page() cannot rewrite these, so "
        f"they would load the default variant's asset: {offenders}")


def test_resolve_prefers_query_param_then_default():
    assert ui_variant.resolve(_FakeRequest(ui="vivarium")) == "vivarium"
    assert ui_variant.resolve(_FakeRequest(ui="classic")) == "classic"
    # Unknown / absent both fall back to the process default.
    assert ui_variant.resolve(_FakeRequest(ui="nope")) == ui_variant.DEFAULT
    assert ui_variant.resolve(_FakeRequest()) == ui_variant.DEFAULT


def test_page_rewrites_asset_prefix_to_its_own_variant():
    for variant in ui_variant.TREES:
        html = ui_variant.page(_FakeRequest(ui=variant), "index.html").body.decode()
        assert f"/static/{variant}/" in html, f"{variant}: no rewritten assets"
        # No bare /static/ left over — every occurrence must be namespaced.
        for other in ui_variant.TREES:
            if other != variant:
                assert f"/static/{other}/" not in html, (
                    f"{variant} page references {other} assets")
        assert "/static/palette.js" not in html


def test_variants_are_actually_different_pages():
    classic = ui_variant.page(_FakeRequest(ui="classic"), "index.html").body
    vivarium = ui_variant.page(_FakeRequest(ui="vivarium"), "index.html").body
    assert classic != vivarium


def test_pages_are_not_shared_cached():
    """A per-request variant behind a CDN must not be cached for everyone."""
    resp = ui_variant.page(_FakeRequest(ui="classic"), "index.html")
    assert resp.headers["cache-control"] == "no-store"


def test_classic_is_the_stock_default():
    assert ui_variant.CLASSIC == "classic"
    assert ui_variant.dir_for("classic") == V7 / "viewer"
    assert ui_variant.dir_for("vivarium") == V7 / "viewer_vivarium"
    assert ui_variant.dir_for("bogus") == V7 / "viewer"


def test_smells_carry_the_viewer_join_key():
    """Smells must ship line_id/word_idx, not just coordinates.

    The desire layer tints each word by how strongly the worm wants it,
    which requires joining the smell list to the food snapshot. A
    coordinate join looks correct and matches NOTHING: _compute_smells runs
    on brain ticks (2 Hz) while the food snapshot is taken per frame
    (60 Hz), so the scroller has moved every word by the time both reach
    the client. Measured drift was a constant 5.25 px in y — small enough
    to look like a rounding issue, fatal to an equality join.
    """
    import inspect
    from sim import world as world_mod
    from server import app as app_mod

    src = inspect.getsource(world_mod.World._compute_smells)
    assert '"line_id": f.line_id' in src and '"word_idx": f.word_idx' in src, \
        "sensed_smells entries must carry the word's identity"

    # Both serialisation sites must forward it.
    for mod, label in ((world_mod, "sim/world.py"), (app_mod, "server/app.py")):
        text = inspect.getsource(mod)
        smells_at = text.index('"smells": [')
        block = text[smells_at:smells_at + 700]
        assert "line_id" in block and "word_idx" in block, \
            f"{label} drops the join key from the smells payload"


def test_desire_layer_joins_on_identity_in_both_trees():
    """Neither viewer may fall back to the coordinate join."""
    for variant, tree in ui_variant.TREES.items():
        js = (tree / "focus" / "text-canvas.js").read_text()
        assert "desireByWord" in js, f"{variant}: desire layer missing"
        assert "${smell.line_id}_${smell.word_idx}" in js, \
            f"{variant}: heatmap must key on word identity"
        assert "${item.line_id}_${item.word_idx}" in js, \
            f"{variant}: word tint must key on word identity"
        assert "`${smell.x},${smell.y}`" not in js, \
            f"{variant}: coordinate join reintroduced — it silently matches nothing"
        assert "`${item.x},${item.y}`" not in js, \
            f"{variant}: coordinate join reintroduced — it silently matches nothing"
