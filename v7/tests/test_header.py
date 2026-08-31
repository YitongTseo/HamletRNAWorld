"""One header, every page.

The header used to be copy-pasted into each page's <head> and the copies
drifted: the overview styled itself from --bench/--ivory/--ochre (names
palette.js never sets, so it alone ignored the theme), poems/about/generations
used --accent, and the focus view had no header at all — just a hand-rolled
#nav with hardcoded colours and non-uppercase 12px type, so the chrome visibly
changed the moment you clicked into a worm.

These tests are the guard: they fail if a page starts styling its own header
again, or if a new page forgets to load the shared one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

V7 = Path(__file__).resolve().parent.parent
VIEWER = V7 / "viewer"
PAGES = sorted(VIEWER.glob("*.html"))


def test_every_page_loads_the_shared_header():
    assert PAGES, "no viewer pages found"
    for p in PAGES:
        s = p.read_text()
        assert "/static/header.js" in s, f"{p.name} does not load header.js"
        assert re.search(r'<body[^>]*\sdata-page="', s), \
            f"{p.name} has no data-page, so no nav item is marked current"


def test_no_page_defines_its_own_header_markup_or_css():
    """The drift was CSS, so this is the rule that actually holds the line."""
    for p in PAGES:
        s = p.read_text()
        assert "<header" not in s, f"{p.name} hand-rolls header markup"
        offenders = [ln.strip() for ln in s.splitlines()
                     if re.match(r'\s*(#site-)?header[\s{,]', ln)]
        assert not offenders, f"{p.name} styles the header itself: {offenders[:3]}"


def test_focus_no_longer_hand_rolls_a_nav():
    """The specimen view is the page that changed. Its chrome is now the same
    overlay header as everywhere else, and #wormtitle is all that's left."""
    html = (VIEWER / "focus.html").read_text()
    css = (VIEWER / "focus.css").read_text()
    assert 'data-header="overlay"' in html
    assert '<div id="nav">' not in html
    assert not re.search(r'^\s*#nav[\s{,]', css, re.M), "focus.css still styles #nav"
    assert "wormtitle" in html, "the specimen still needs its name"


def test_header_carries_the_same_nav_everywhere_plus_the_tombstone():
    js = (VIEWER / "header.js").read_text()
    for label in ("worms", "poems", "generations", "about"):
        assert f'label: "{label}"' in js, f"{label} missing from the shared nav"
    # The graveyard is found by the stone in the corner, not by a nav word.
    assert 'href="/graveyard"' in js
    assert '<svg' in js and "graveyard-link" in js
    assert 'label: "graveyard"' not in js, \
        "the graveyard is meant to be discovered, not listed"


def test_header_styles_only_from_palette_variables():
    """No hardcoded hex in the header's own rules: that is exactly how the
    focus nav ended up a different colour from every other page. The one
    exception is the overlay scrim, which needs an alpha ramp."""
    js = (VIEWER / "header.js").read_text()
    css_block = js[js.index("var CSS = ["):js.index("function build()")]
    hexes = re.findall(r"#[0-9a-fA-F]{3,6}", css_block)
    assert not hexes, f"hardcoded colours in the shared header: {hexes}"


def test_overview_reshades_with_the_theme():
    """index.html names its colours after the bench; those names must be
    aliases of the palette, or the overview stays tobacco on every other
    experiment subdomain while the rest of the site reshades."""
    s = (VIEWER / "index.html").read_text()
    for private, shared in (("--bench", "--bg"), ("--ivory", "--fg"),
                            ("--ochre", "--accent")):
        assert re.search(rf"{private}:\s*var\({shared}", s), \
            f"{private} is not an alias of {shared}"
