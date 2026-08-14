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

The judge is pluggable: WORMLET_JUDGE_BACKEND=anthropic (default, the path
described above) or openai (any OpenAI-compatible /v1/chat/completions —
Ollama, LM Studio, llama.cpp server, vLLM). See the backend block below.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass

import anthropic

WINDOW_SIZE = 15
STRIDE = 15  # non-overlapping; matches the spec
# Judge-noise experiment (2026-07-17, docs/superpowers/specs/2026-07-17-judge-
# noise-and-sigma-control-experiments.md). TWO findings, one solid, one corrected:
#   * SOLID: temperature was unset (API default 1.0) — the single biggest noise
#     source. Pinning JUDGE_TEMPERATURE=0 roughly doubles rank reproducibility
#     (mean Kendall τ 0.31 → 0.60 at 25% windows, p=0.004). KEEP.
#   * CORRECTED: an earlier cut suggested "5 windows beats 25%", but that was a
#     fixed-window artifact — the reproducibility check reused the SAME 5 windows
#     every repeat, hiding the variance of WHICH windows get drawn. Resampling
#     windows each rep (production-faithful) flips it: more windows = MORE
#     reproducible (m=5 τ 0.37 vs 25% τ 0.60). So we KEEP the 25% sample.
SAMPLE_FRACTION = 0.25  # judge ~25% of windows (~40 for a full poem)
SAMPLE_N = None         # (was 5, reverted) fixed-count path unused; fraction wins
JUDGE_TEMPERATURE = 0.0
MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 8192  # 210 sampled windows × ~8 tokens output + buffer

# --- pluggable backend ------------------------------------------------------
# WORMLET_JUDGE_BACKEND selects who scores the windows:
#   * "anthropic" (default) — Claude via the anthropic SDK, prompt-cached;
#     exactly the production path.
#   * "openai" — any OpenAI-compatible /v1/chat/completions endpoint (Ollama,
#     LM Studio, llama.cpp server, vLLM). Built for judging with a LOCAL model:
#     no API cost, no key, works with all external egress blocked. The rubric,
#     sampling, temperature-0 pin, CSV protocol and parsing are identical —
#     only the transport differs. Calibration is NOT: a small local model is a
#     different critic, so fitness histories judged by different models must
#     never be compared as if they were one judge.
# Env is read at call time, not import, so tests and multi-process configs can
# switch backends without reimporting the module.
#   WORMLET_JUDGE_MODEL     required for openai (e.g. "gemma3:4b"); optional
#                           override of MODEL for anthropic
#   WORMLET_JUDGE_URL       base URL, default http://127.0.0.1:11434/v1 (Ollama)
#   WORMLET_JUDGE_API_KEY   optional bearer token, openai backend only
#   WORMLET_JUDGE_TIMEOUT_S HTTP timeout, default 300 — a 4B model scoring ~40
#                           windows on consumer hardware can take minutes


def _judge_backend() -> str:
    return os.environ.get("WORMLET_JUDGE_BACKEND", "anthropic")


def _judge_model() -> str:
    model = os.environ.get("WORMLET_JUDGE_MODEL")
    if model:
        return model
    if _judge_backend() == "openai":
        raise RuntimeError(
            "WORMLET_JUDGE_MODEL is required with WORMLET_JUDGE_BACKEND=openai "
            "— set it to a model your server lists under /v1/models."
        )
    return MODEL


def _openai_payload(user_prompt: str) -> dict:
    """Request body for /chat/completions. No cache_control — OpenAI-compatible
    servers don't speak it, and local models don't bill input anyway."""
    return {
        "model": _judge_model(),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
        "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def _http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """Tiny stdlib POST — deliberately no new dependency for a backend most
    installs never enable. Tests stub this symbol to stay off the network."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _complete_openai(user_prompt: str) -> str:
    base = os.environ.get("WORMLET_JUDGE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    headers: dict[str, str] = {}
    key = os.environ.get("WORMLET_JUDGE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timeout = float(os.environ.get("WORMLET_JUDGE_TIMEOUT_S", "300"))
    # A local judge (Ollama on someone's gaming PC, LM Studio on a laptop) is
    # far less available than a hosted API — it may be asleep or rebooting at
    # rollover time. Retry transient transport errors briefly (0s/5s/15s);
    # a genuinely dead endpoint falls through to the caller, where the
    # all-zero guard in generations.py aborts the rollover rather than
    # letting the lineage evolve on noise. URLError/timeout/reset are all
    # OSError subclasses, so one except covers the transport failure modes.
    last_err: Exception | None = None
    payload = None
    for delay in (0, 5, 15):
        if delay:
            time.sleep(delay)
        try:
            payload = _http_post_json(
                base + "/chat/completions", _openai_payload(user_prompt),
                headers, timeout,
            )
            break
        except OSError as e:
            last_err = e
    if payload is None:
        raise RuntimeError(
            f"judge endpoint {base} unreachable after 3 attempts: {last_err!r}"
        ) from last_err
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"unexpected /chat/completions response shape from {base}: {e!r}"
        ) from e

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
                   seed: int | None = None,
                   n: int | None = SAMPLE_N) -> list[tuple[int, list[str]]]:
    """Deterministically sample windows. If `n` is set, pick exactly that many
    (Exp-1 result: 5 random windows rank worms more reproducibly than 25%);
    otherwise fall back to `fraction`. Seed controls which windows get picked so
    the same poem reproduces the same sampling across re-runs."""
    rng = random.Random(seed)
    if n is not None:
        k = min(n, len(windows))
    else:
        if fraction >= 1.0:
            return windows
        k = max(1, int(round(len(windows) * fraction)))
    if k >= len(windows):
        return windows
    indices = sorted(rng.sample(range(len(windows)), k))
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


def _complete_anthropic(user_prompt: str, worm_name: str) -> str:
    client = _client()
    response = client.messages.create(
        model=_judge_model(),
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=JUDGE_TEMPERATURE,
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
    return next((b.text for b in response.content if b.type == "text"), "")


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

    backend = _judge_backend()
    if backend == "anthropic":
        text = _complete_anthropic(user_prompt, worm_name)
    elif backend == "openai":
        text = _complete_openai(user_prompt)
    else:
        raise RuntimeError(
            f"unknown WORMLET_JUDGE_BACKEND={backend!r} — use 'anthropic' or 'openai'"
        )
    scores = _parse_scores(text)

    out: list[ScoredWindow] = []
    for idx, toks in sampled:
        if idx in scores:
            e, c = scores[idx]
            out.append(ScoredWindow(idx=idx, tokens=toks, emotional=e, coherence=c))
    return out
