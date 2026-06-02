"""Per-experiment definitions for the 4 sanity-check runs + the prod
coherent-poetry run. Each experiment is a self-contained record:

    mode             — slug used in URLs, data paths, env vars
    label            — human-readable display name (dropdown + banner)
    scorer           — fitness function over the worm's eaten-word list, or
                       None for the Claude-judge path (prod)
    briefing_file    — file under v6/docs/briefings/<mode>.md for the gardener
    port             — local port for the uvicorn process
    passage          — corpus passage (act1 for sim, full for prod)
    gardener_every   — how often the gardener writes (1 = every gen,
                       10 = every 10 gens; sim experiments run at 10)
    store_full_poems — whether per-worm poem_raw.txt files are kept (prod
                       commits these to git; sim experiments skip them to
                       keep the data dir small)
    public_url       — full https URL the dropdown jumps to. For sim
                       experiments this is a per-experiment cloudflared
                       subdomain on this host; for prod it's the existing
                       www.wordswordsworms.org served from the other server.
    blurb            — one-sentence summary for the banner

The viewer dropdown enumerates EXPERIMENTS in display order. Each process
knows its own mode via the WORMLET_EXPERIMENT_MODE env var (defaults to
None = legacy / prod path).

The DOMAIN env var (or default "wordswordsworms.org") plus the per-mode
subdomain compose the public URL — set DOMAIN=localhost.dev for local
debugging so the dropdown links don't leave the box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from server import pos_scorers


V6_ROOT = Path(__file__).resolve().parent.parent
BRIEFINGS_DIR = V6_ROOT / "docs" / "briefings"
DOMAIN = os.environ.get("WORMLET_DOMAIN", "wordswordsworms.org")


Scorer = Callable[[list[str]], float]


@dataclass(frozen=True)
class Experiment:
    mode: str
    label: str
    scorer: Scorer | None
    briefing_file: str
    port: int
    passage: str
    gardener_every: int
    store_full_poems: bool
    subdomain: str        # left empty = root (no subdomain) = the prod URL
    blurb: str

    @property
    def public_url(self) -> str:
        if not self.subdomain:
            # Prod stays at www.<domain>; nothing to prefix.
            return f"https://www.{DOMAIN}/"
        return f"https://{self.subdomain}.{DOMAIN}/"

    def briefing_text(self) -> str:
        p = BRIEFINGS_DIR / self.briefing_file
        if not p.exists():
            return "(briefing missing)"
        return p.read_text()


EXPERIMENTS: list[Experiment] = [
    Experiment(
        mode="words",
        label="Eat words",
        scorer=pos_scorers.score_words,
        briefing_file="words.md",
        port=8001,
        passage="act1",
        gardener_every=10,
        store_full_poems=False,
        subdomain="words",
        blurb="Sanity check 1 — maximize raw word count. Can the worms find food at all?",
    ),
    Experiment(
        mode="nouns",
        label="Eat nouns",
        scorer=pos_scorers.score_nouns,
        briefing_file="nouns.md",
        port=8002,
        passage="act1",
        gardener_every=10,
        store_full_poems=False,
        subdomain="nouns",
        blurb="Sanity check 2 — maximize nouns eaten. Can selection steer the chemosensation toward a POS class?",
    ),
    Experiment(
        mode="adj_noun",
        label="Adj→Noun pairs",
        scorer=pos_scorers.score_adj_noun,
        briefing_file="adj_noun.md",
        port=8003,
        passage="act1",
        gardener_every=10,
        store_full_poems=False,
        subdomain="adj-noun",
        blurb="Sanity check 3 — minimal memory test. Score = count of adjacent (ADJ, NOUN) pairs in eaten order.",
    ),
    Experiment(
        mode="pos_chain",
        label="POS chains",
        scorer=pos_scorers.score_pos_chain,
        briefing_file="pos_chain.md",
        port=8004,
        passage="act1",
        gardener_every=10,
        store_full_poems=False,
        subdomain="pos-chain",
        blurb="Sanity check 4 — chain memory. Score = consecutive eaten-word POS bigrams in a curated valid-English set.",
    ),
    Experiment(
        mode="poetry",
        label="Coherent poetry (prod)",
        scorer=None,
        briefing_file="../specs/2026-05-26-gardener-briefing.md",
        port=8000,
        passage="full",
        gardener_every=1,
        store_full_poems=True,
        subdomain="",
        blurb="The real experiment — Claude Haiku judges poetic + coherent windows. Full Hamlet, slower iteration.",
    ),
]


_BY_MODE = {e.mode: e for e in EXPERIMENTS}


def current() -> Experiment | None:
    """Return the experiment definition for THIS process, based on
    WORMLET_EXPERIMENT_MODE. None = legacy / prod path (no experiment
    framework in play). Prod runs without this env var set."""
    mode = os.environ.get("WORMLET_EXPERIMENT_MODE")
    if not mode:
        return None
    return _BY_MODE.get(mode)


def by_mode(mode: str) -> Experiment | None:
    return _BY_MODE.get(mode)


def all_for_dropdown() -> list[dict]:
    """Display-order list for the frontend dropdown. Each entry is a dict
    safe to JSON-serialize into the page."""
    return [
        {
            "mode": e.mode,
            "label": e.label,
            "public_url": e.public_url,
            "blurb": e.blurb,
        }
        for e in EXPERIMENTS
    ]
