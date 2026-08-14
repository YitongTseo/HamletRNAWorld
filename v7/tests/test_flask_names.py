"""Tests for the WORMLET_FLASK_WORM_NAMES override. The name list is read at
module import, so these tests reload the orchestrator — and reload it AGAIN
after restoring the env, so later test modules in the same run_all.py process
see the default lineup."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server.orchestrator as orch


def _reload_with(names: str | None):
    if names is None:
        os.environ.pop("WORMLET_FLASK_WORM_NAMES", None)
    else:
        os.environ["WORMLET_FLASK_WORM_NAMES"] = names
    importlib.reload(orch)


def test_env_override_replaces_lineup():
    try:
        _reload_with(" swan, benc ,mikey,dowager cixi ")
        assert orch.FLASK_WORM_NAMES == ["swan", "benc", "mikey", "dowager cixi"]
    finally:
        _reload_with(None)
    assert orch.FLASK_WORM_NAMES[0] == "Alice"  # default restored for later tests


def test_unset_env_keeps_default_twenty():
    _reload_with(None)
    assert len(orch.FLASK_WORM_NAMES) == 20
    assert orch.FLASK_WORM_NAMES[0] == "Alice"


def test_short_list_still_raises_in_load_flasks():
    """The collision guard must apply to overridden lineups too."""
    try:
        _reload_with("only,three,names")
        try:
            orch.load_flasks(n_flasks=1, n_worms_per_flask=10)
            assert False, "expected ValueError for 3 names < 10 worms"
        except ValueError as e:
            assert "available worm names" in str(e)
    finally:
        _reload_with(None)
