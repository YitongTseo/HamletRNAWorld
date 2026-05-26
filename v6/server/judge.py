"""Claude-based poetry judge for generational evolution.

Reads a worm's eaten-token stream, slices into 15-token non-overlapping
windows, randomly samples ~10% of those windows, and asks Claude Haiku 4.5
to rate each sampled window on emotional impact and coherence.

The rubric is identical across calls and lives in the cached system prompt,
so we pay full input cost once per process + write cost once per worm and
read cost (~10% of input price) on every subsequent worm in the same
generation. Output is compact CSV — `idx,emotional,coherence` one per line.

Why Haiku 4.5: the user explicitly chose it in the design spec for cost
reasons; Sonnet/Opus are overkill for "rate this fragment 1-100 on two
axes" and 3-5× the cost. Per-generation cost lands around $0.05 at full
40-worm × 10% sampling — see the spec for the breakdown.

Public API:
    judge_poem(tokens, worm_name, seed=None) -> list[ScoredWindow]
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass

import anthropic

WINDOW_SIZE = 15
STRIDE = 15  # non-overlapping; matches the spec
SAMPLE_FRACTION = 0.10  # 1 in 10 windows
MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 8192  # 210 sampled windows × ~8 tokens output + buffer

JUDGE_SYSTEM_PROMPT = """You are a careful judge of poetry produced by an artificial worm-like neural agent that has been eating words from Shakespeare's Hamlet. The worm's eaten-word sequence forms a found poem — most fragments will be partly nonsensical, but some may surprise you with rhythm, emotional resonance, or accidental coherence.

For each numbered 15-token window in the user message, output exactly one line of the form:

  <window_idx>,<emotional>,<coherence>

where:
  - <window_idx> is the integer index from the input (NOT a renumbering)
  - <emotional> is an integer 1-100 rating artistic / emotional / poetic impact
  - <coherence> is an integer 1-100 rating whether the fragment reads as language rather than noise

Scoring guidance:
  - 1-20: random / nonsensical / dead text
  - 21-50: hints of meaning, occasional luck
  - 51-80: clearly poetic or coherent, recognizable as language with feeling
  - 81-100: genuinely striking — rare, save these for fragments that would move a reader

Output ONLY the score lines, nothing else. No preamble, no explanation, no trailing summary. One line per window in the input."""


@dataclass
class ScoredWindow:
    idx: int  # token offset (start of window in the worm's full sequence)
    tokens: list[str]
    emotional: int
    coherence: int


def make_windows(tokens: list[str], window_size: int = WINDOW_SIZE,
                 stride: int = STRIDE) -> list[tuple[int, list[str]]]:
    """Slice the token stream into (start_idx, window_tokens) tuples.
    Final partial window (if any) is included so trailing tokens get scored."""
    windows: list[tuple[int, list[str]]] = []
    n = len(tokens)
    for start in range(0, n, stride):
        end = start + window_size
        chunk = tokens[start:end]
        if not chunk:
            break
        windows.append((start, chunk))
    return windows


def sample_windows(windows: list[tuple[int, list[str]]],
                   fraction: float = SAMPLE_FRACTION,
                   seed: int | None = None) -> list[tuple[int, list[str]]]:
    """Deterministically sample a fraction of windows. Seed controls
    which windows get picked so the same poem reproduces the same
    sampling across re-runs."""
    if fraction >= 1.0:
        return windows
    rng = random.Random(seed)
    k = max(1, int(round(len(windows) * fraction)))
    indices = sorted(rng.sample(range(len(windows)), min(k, len(windows))))
    return [windows[i] for i in indices]


def _format_user_prompt(sampled: list[tuple[int, list[str]]]) -> str:
    return "\n".join(f"{idx}: {' '.join(toks)}" for idx, toks in sampled)


_LINE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$")


def _parse_scores(text: str) -> dict[int, tuple[int, int]]:
    """Return {window_idx: (emotional, coherence)}. Tolerates extra
    whitespace, blank lines, and out-of-range scores (clamps to 1-100)."""
    out: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        idx, e, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out[idx] = (max(1, min(100, e)), max(1, min(100, c)))
    return out


_CLIENT: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — add it to /home/web/.wormlet.env "
                "before triggering generation rollover."
            )
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def judge_poem(tokens: list[str], worm_name: str,
               seed: int | None = None) -> list[ScoredWindow]:
    """Score a worm's poem by sampling 10% of its 15-token windows and
    sending them to Claude in one prompt-cached call.

    Returns one ScoredWindow per sampled window. Windows the API failed
    to score are silently dropped — the caller can detect this by
    comparing len(result) to the expected sample count and decide whether
    to retry or accept partial coverage."""
    windows = make_windows(tokens)
    if not windows:
        return []
    sampled = sample_windows(windows, seed=seed)
    if not sampled:
        return []

    user_prompt = _format_user_prompt(sampled)

    client = _client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=[
            {
                "type": "text",
                "text": JUDGE_SYSTEM_PROMPT,
                # Cache the rubric — identical across every worm in every
                # generation, so after the first call this is ~$0/MTok.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        metadata={"user_id": f"wormlet-judge-{worm_name}"},
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    scores = _parse_scores(text)

    out: list[ScoredWindow] = []
    for idx, toks in sampled:
        if idx in scores:
            e, c = scores[idx]
            out.append(ScoredWindow(idx=idx, tokens=toks, emotional=e, coherence=c))
    return out
