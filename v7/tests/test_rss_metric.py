"""RSS reporting on /healthz — the measurement the leak hunt has been missing.

Context: on 2026-09-07 the live process sat at 1.36 GB against a measured
~190 MB baseline, while /debug/objects reported 30/30 live Worlds and only
~200k gc-tracked objects. So ~1.1 GB was real, invisible to the object census,
and — critically — of unknown SLOPE, because nothing had ever recorded RSS
over time. wormlet-metrics.sh scrapes /healthz every minute, so putting the
number there is the whole fix.

These pin the wiring, not the values: that the keys exist, that the cache
doesn't fork per probe, and that a `ps` failure degrades to None instead of
500ing the health endpoint the watchdog depends on.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

V7 = Path(__file__).resolve().parent.parent

from server import app as app_mod   # noqa: E402


def test_rss_kb_reads_a_plausible_size():
    app_mod._RSS_CACHE = (0.0, None)
    val = app_mod._rss_kb()
    assert val is not None, "ps -o rss= gave nothing on this host"
    # A CPython process that has imported nltk and numpy is never under 10 MB,
    # and 100 GB would mean we parsed the wrong column.
    assert 10_000 < val < 100_000_000, val


def test_rss_is_cached_not_forked_per_probe():
    """/healthz is hit every minute by cron and again by the watchdog. The
    number moves on a scale of minutes; the fork should not be per-probe."""
    app_mod._RSS_CACHE = (0.0, None)
    first = app_mod._rss_kb()
    app_mod._RSS_CACHE = (time.monotonic(), 4242)
    assert app_mod._rss_kb() == 4242, "second call ignored the cache"
    assert first != 4242


def test_stale_cache_is_refetched():
    app_mod._RSS_CACHE = (time.monotonic() - app_mod._RSS_TTL_S - 1, 4242)
    assert app_mod._rss_kb() != 4242, "expired entry was served anyway"


def test_ps_failure_degrades_to_none(monkeypatch=None):
    """Health must not 500 because a subprocess died. The watchdog restarts
    the process on a bad /healthz, so an exception here is a restart loop."""
    real = app_mod.subprocess.run

    def boom(*a, **kw):
        raise OSError("no ps on this host")

    app_mod.subprocess.run = boom
    try:
        app_mod._RSS_CACHE = (0.0, None)
        assert app_mod._rss_kb() is None
        # The failure is cached too, or a broken ps costs a fork per probe.
        at, val = app_mod._RSS_CACHE
        assert at != 0.0 and val is None
    finally:
        app_mod.subprocess.run = real
        app_mod._RSS_CACHE = (0.0, None)


def test_healthz_emits_both_memory_keys():
    src = (V7 / "server" / "app.py").read_text()
    assert '"rss_kb": rss_kb,' in src
    assert '"rss_peak_kb": resource.getrusage' in src
    # Forked off the event loop: /healthz is what gets probed when the box is
    # already struggling, and the tick watchdog kills on a 20 s stall.
    assert "await asyncio.to_thread(_rss_kb)" in src


def test_metrics_script_pushes_the_series():
    """The cron scraper is the only thing that turns /healthz into history."""
    script = Path("/data/homelab/jails/wormlet/wormlet-metrics.sh")
    if not script.exists():
        return          # not on droog; the server-side keys are tested above
    body = script.read_text()
    assert "wormlet_rss_bytes" in body
    assert "wormlet_rss_peak_bytes" in body
