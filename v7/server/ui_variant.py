"""Two viewer front-ends, one server.

The site ships two complete, independent viewer trees:

    viewer/            "classic"  — black/green, thumbnail-card overview,
                                    ui-monospace. The long-standing look.
    viewer_vivarium/   "vivarium" — the 2026-08 "Tobacco & Ochre" restyle:
                                    petri-dish tray, ivory nematodes,
                                    Fragment Mono / Instrument Serif.

Which one a request gets:

    ?ui=vivarium        explicit per-request override (wins)
    WORMLET_UI=...      the process default (systemd)
    "classic"           the fallback

WHY TWO TREES INSTEAD OF A THEME. The two front-ends differ structurally,
not just chromatically — the overview is a grid of thumbnail cards in one
and a tray of petri dishes in the other, and the focus page grows a
specimen card. That is not expressible as a stylesheet swap, and a single
tree carrying both shapes behind `if (VARIANT === ...)` branches would make
every future edit a two-branch edit in one file. Separate trees keep each
front-end readable on its own terms, and let `viewer_vivarium/` stay
byte-identical to upstream so future merges from origin land clean.

The cost is duplication: 19 of the 26 files genuinely differ between the
trees, but 6 (focus/dock.js, focus/state.js, focus/magnifier.js,
focus/panel-chrome.js, focus/responsive.js, poems.js) are identical today
and must be edited in both places. If that set grows, promote it to a
shared `viewer_common/` mounted at /static/common/ rather than letting the
trees drift.

HOW /static RESOLVES. Every `/static/...` reference in either tree lives
ONLY in the five HTML files — there are none in any .js or .css (enforced
by tests/test_ui_variant.py). So instead of editing asset paths in either
tree, `page()` rewrites `/static/` to `/static/<variant>/` as it serves the
HTML, and both trees are mounted side by side. Neither tree contains a
single variant-aware line; the whole switch is this file.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

V7_ROOT = Path(__file__).resolve().parent.parent

CLASSIC = "classic"
VIVARIUM = "vivarium"

TREES: dict[str, Path] = {
    CLASSIC: V7_ROOT / "viewer",
    VIVARIUM: V7_ROOT / "viewer_vivarium",
}

#: Process default. An unknown value falls back to classic rather than
#: failing to boot — a typo in a systemd drop-in should not take the site
#: down, and the boot log line below makes it visible.
DEFAULT = os.environ.get("WORMLET_UI", CLASSIC).strip().lower()
if DEFAULT not in TREES:
    DEFAULT = CLASSIC

# Rewritten HTML, keyed (variant, filename). The trees are read-only at
# runtime, so one read + one replace per page per process is enough.
_cache: dict[tuple[str, str], str] = {}


def resolve(request: Request) -> str:
    """Variant for this request: ?ui= wins, else the process default."""
    requested = request.query_params.get("ui")
    if requested:
        requested = requested.strip().lower()
        if requested in TREES:
            return requested
    return DEFAULT


def dir_for(variant: str) -> Path:
    return TREES.get(variant, TREES[CLASSIC])


def page(request: Request, filename: str) -> HTMLResponse:
    """Serve `filename` from the request's variant tree.

    Asset URLs are rewritten to that variant's static prefix so a page
    served under ?ui=vivarium also pulls vivarium JS/CSS — the query param
    does not survive onto the asset requests the browser makes next.
    """
    variant = resolve(request)
    key = (variant, filename)
    html = _cache.get(key)
    if html is None:
        html = dir_for(variant).joinpath(filename).read_text(encoding="utf-8")
        html = html.replace("/static/", f"/static/{variant}/")
        _cache[key] = html
    # The variant is chosen per request, so a shared cache (Cloudflare) must
    # not hand one visitor's variant to the next.
    return HTMLResponse(html, headers={"Vary": "Accept-Encoding",
                                       "Cache-Control": "no-store"})


def mount_all(app) -> None:
    """Mount every tree at /static/<variant>/.

    Also mounts the process default at bare /static/ so any URL that
    predates the split (bookmarks, the board firmware, a hardcoded path in
    something outside this repo) keeps resolving. The per-variant mounts go
    first: Starlette matches in order, and a bare /static mount would
    otherwise swallow /static/classic/... as a missing subdirectory.

    Freshness is already handled: app.disable_cdn_cache sets
    no-cache/no-store on every /static/ response, so a changed asset can
    never be served stale despite focus.html versioning only its entry
    module. Do not add a second caching layer here.
    """
    from fastapi.staticfiles import StaticFiles

    for variant, path in TREES.items():
        app.mount(f"/static/{variant}", StaticFiles(directory=str(path)),
                  name=f"viewer_{variant}")
    app.mount("/static", StaticFiles(directory=str(dir_for(DEFAULT))),
              name="viewer")
