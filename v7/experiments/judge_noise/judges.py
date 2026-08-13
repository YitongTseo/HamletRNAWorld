"""Experiment 1 — the two scoring methods under test.

- absolute_batch(): faithful to production (server.judge) — one batched Haiku
  call scores a list of windows, but with an explicit `temperature`.
- pairwise(): shows two worms' representative windows and asks which worm is the
  stronger writer; run in both orders to measure position bias.
- bradley_terry(): turn a set of pairwise outcomes into a per-worm rating.
"""
from __future__ import annotations

import re

import numpy as np
import anthropic

from server.judge import JUDGE_SYSTEM_PROMPT, _parse_scores, MODEL, MAX_OUTPUT_TOKENS
from pool_builder import Window

# One module-level client; the Anthropic SDK client is thread-safe. Extra
# retries + a generous timeout so a rate-limit blip during the ~25k-call run
# self-heals instead of dropping a data point.
_CLIENT = anthropic.Anthropic(max_retries=6, timeout=120.0)


# --- Absolute (production-faithful, temperature-parameterised) ---------------

def absolute_batch(windows: list[Window], temperature: float) -> dict[int, tuple[int, int]]:
    """Score every window in one call, exactly like production's judge_poem but
    with an explicit temperature. Returns {idx: (emotional, coherence)}."""
    user_prompt = "\n".join(f"{w.idx}: {' '.join(w.tokens)}" for w in windows)
    resp = _CLIENT.messages.create(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        system=[{"type": "text", "text": JUDGE_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _parse_scores(text)


# --- Pairwise comparison ------------------------------------------------------

PAIRWISE_SYSTEM = """You are judging poetry produced by two different writers, A and B. Each writer's work is shown as a set of short 15-word windows sampled from a longer piece. Judge on the same two qualities the ecosystem rewards: emotional impact and coherence.

Decide which writer's work is stronger overall. Respond with EXACTLY one character on a single line: "A" if writer A is stronger, or "B" if writer B is stronger. No ties, no preamble, no explanation."""


def _fmt_side(label: str, windows: list[Window]) -> str:
    lines = [f"# Writer {label}"]
    for w in windows:
        lines.append(" ".join(w.tokens))
    return "\n".join(lines)


def pairwise(a_windows: list[Window], b_windows: list[Window], temperature: float) -> str:
    """Return 'A' or 'B' (the stronger writer). 'A' is always shown first; the
    caller controls which worm is A vs B to measure position bias."""
    user = f"{_fmt_side('A', a_windows)}\n\n{_fmt_side('B', b_windows)}"
    resp = _CLIENT.messages.create(
        model=MODEL,
        max_tokens=8,
        temperature=temperature,
        system=[{"type": "text", "text": PAIRWISE_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    m = re.search(r"[AB]", text.upper())
    return m.group(0) if m else "A"  # default A on unparseable (rare)


# --- Bradley-Terry rating from pairwise wins ---------------------------------

def bradley_terry(names: list[str], wins: dict[tuple[str, str], int],
                  n_iter: int = 200) -> dict[str, float]:
    """MLE Bradley-Terry strengths (log scale, mean-centred) from a win matrix.
    `wins[(i, j)]` = number of times i beat j. Uses the standard MM iteration."""
    idx = {n: k for k, n in enumerate(names)}
    N = len(names)
    W = np.zeros((N, N))
    for (i, j), c in wins.items():
        W[idx[i], idx[j]] += c
    total_wins = W.sum(axis=1)  # games i won
    p = np.ones(N)
    games = W + W.T  # symmetric games-played count
    for _ in range(n_iter):
        denom = np.zeros(N)
        for i in range(N):
            s = 0.0
            for j in range(N):
                if i == j:
                    continue
                nij = games[i, j]
                if nij:
                    s += nij / (p[i] + p[j])
            denom[i] = s
        new_p = np.where(denom > 0, total_wins / denom, p)
        new_p = np.where(new_p <= 0, 1e-9, new_p)
        new_p *= N / new_p.sum()  # normalise to keep it identifiable
        if np.allclose(new_p, p, rtol=1e-6):
            p = new_p
            break
        p = new_p
    strengths = np.log(p)
    strengths -= strengths.mean()
    return {n: float(strengths[idx[n]]) for n in names}
