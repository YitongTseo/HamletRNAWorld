"""Multi-worm smoke test: spin up 6 worms via the orchestrator, tick a few
seconds, assert each worm produced eaten words and persisted them to disk.

This test mutates v6/data/ — uses a temp DATA_DIR via monkeypatching."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import server.orchestrator as orch


def test_six_worms_produce_poems():
    with tempfile.TemporaryDirectory() as tmp:
        # Redirect orchestrator paths to a fresh temp dir.
        orch.DATA_DIR = Path(tmp) / "worms"
        orch.WORMS_CONFIG = Path(tmp) / "worms.json"
        worms = orch.load_worms(n_worms=6)
        assert len(worms) == 6
        names = [w.name for w in worms]
        assert names == ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
        # Tick for 90 simulated seconds. Words spawn at the top of the
        # canvas and take ~30s to drift down to the worm; allow extra so
        # all seeds reliably hit at least one.
        N_TICKS = 60 * 90
        for _ in range(N_TICKS):
            for w in worms:
                w.world.tick()
                for _word in orch.drain_and_persist(w):
                    pass
        for w in worms:
            w.close()
        # Every worm should have eaten at least one word.
        for w in worms:
            content = w.poem_path.read_text().strip()
            words = [l for l in content.splitlines() if l]
            assert len(words) > 0, f"{w.name} ate nothing in {N_TICKS} ticks"
            assert w.word_count == len(words)


if __name__ == "__main__":
    test_six_worms_produce_poems()
    print("PASS: six worms each produced a non-empty poem")
