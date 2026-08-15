"""The tab mark (viewer/favicon.svg).

Nothing here judges how it looks — these pin the wiring, which is the part
that silently rots: the icon is injected by viewer/palette.js rather than
written into each page's <head>, so a new page that forgets palette.js loses
the favicon (and the fonts, and the theme) without any error.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

V7 = Path(__file__).resolve().parent.parent
VIEWER = V7 / "viewer"


def test_favicon_svg_is_valid_and_square():
    root = ET.parse(VIEWER / "favicon.svg").getroot()
    assert root.tag.endswith("svg")
    # Square viewBox: browsers scale the tile to 16/32/180 px, and a
    # non-square box would letterbox the worm instead of filling the tab.
    assert root.get("viewBox") == "0 0 32 32"
    body = (VIEWER / "favicon.svg").read_text()
    # House palette, not an ad-hoc colour (see viewer/palette.js "poetry").
    assert "#292118" in body and "#cfa348" in body


def test_every_page_loads_palette_js():
    """palette.js carries the favicon, the fonts and the theme. Any viewer
    page that doesn't load it renders untitled and untinted."""
    pages = sorted(VIEWER.glob("*.html"))
    assert pages, "no viewer pages found"
    for p in pages:
        assert "palette.js" in p.read_text(), f"{p.name} does not load palette.js"


def test_palette_injects_the_icon_link():
    js = (VIEWER / "palette.js").read_text()
    assert re.search(r'icon\.href\s*=\s*"/static/favicon\.svg', js)
    assert 'icon.type = "image/svg+xml"' in js


def test_server_answers_favicon_ico():
    """Root-level /favicon.ico is requested whatever the pages declare."""
    app_src = (V7 / "server" / "app.py").read_text()
    assert '@app.get("/favicon.ico")' in app_src
    assert 'FileResponse(VIEWER_DIR / "favicon.svg"' in app_src
