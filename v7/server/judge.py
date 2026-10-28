"""Claude-based poetry judge for generational evolution.

Reads a worm's eaten-token stream, slices into 15-token non-overlapping
windows, randomly samples ~25% of those windows, and asks Claude Haiku 4.5
to rate each sampled window on emotional impact and coherence.

The rubric is identical across calls and lives in the cached system prompt,
so we pay full input cost once per process + write cost once per worm and
read cost (~10% of input price) on every subsequent worm in the same
generation. Output is compact CSV — `idx,emotional,coherence` one per line.

Why Haiku 4.5: the user explicitly chose it in the design spec for cost
reasons; Sonnet/Opus are overkill for "rate this fragment 1-100 on two
axes" and 3-5× the cost. Per-generation cost lands around $0.05 at full
40-worm × 25% sampling — see the spec for the breakdown.

Public API:
    judge_poem(tokens, worm_name, seed=None) -> list[ScoredWindow]

The judge is pluggable: WORMLET_JUDGE_BACKEND=anthropic (default, the path
described above), openai (any OpenAI-compatible /v1/chat/completions —
Ollama, LM Studio, llama.cpp server, vLLM), or human (a person ranks the
flask's worms through the bench UI; no LLM, no API spend). See the backend
block below.
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
#   * Made tunable without a deploy 2026-09-01. The trend above has no reason to
#     stop at 25% — 1.0 scores EVERY window and removes which-windows-drawn
#     variance entirely. It is NOT the default because it roughly quadruples
#     judge tokens (~NZD 60/month at 30 worms and 10 generations a day, against
#     ~NZD 15 at 0.25), and because the common-random-numbers fix landing the
#     same day (server/app.py COMMON_SEED) removes a larger noise source for
#     free. Measure whether CRN alone restores an elite advantage BEFORE paying
#     for this — changing both at once tells you nothing about either.
SAMPLE_FRACTION = float(os.environ.get("WORMLET_JUDGE_SAMPLE_FRACTION", "0.25"))
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
#   * "human" — a PERSON is the critic. No network call at all: the backend
#     drops a ballot on disk, and reads back a verdict the bench UI wrote.
#     Missing verdict raises JudgeAwaitingHuman, which the rollover's
#     all-zero guard turns into "the lineage waits". See the human block
#     further down for the full protocol and the scoring map.
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


def _judge_fallback() -> str | None:
    """WORMLET_JUDGE_FALLBACK names a second backend to try when the primary
    fails (e.g. openai primary on a LAN box that might be powered off,
    anthropic fallback). Empty/unset = no fallback: the rollover's all-zero
    guard aborts and retries later. NOTE the splice caveat: fallback-judged
    generations are scored by a DIFFERENT critic (measured tau ~0.47 vs
    Haiku) — fitness across the splice is not one lineage's history."""
    fb = os.environ.get("WORMLET_JUDGE_FALLBACK", "").strip()
    return fb or None


def _judge_model(backend: str | None = None) -> str:
    backend = backend or _judge_backend()
    if backend == "human":
        # Provenance needs to name WHO, not what: two people are two critics
        # exactly the way two models are. Default is deliberately not a name.
        return os.environ.get("WORMLET_JUDGE_HUMAN", "unnamed")
    if backend == "anthropic":
        # WORMLET_JUDGE_MODEL doubles as the anthropic override, but ONLY if
        # it names a claude model — when anthropic runs as the FALLBACK the
        # env var holds the local model's tag (e.g. "qwen2.5:14b"), which
        # must not be sent to the Anthropic API.
        model = os.environ.get("WORMLET_JUDGE_MODEL", "")
        return model if model.startswith("claude") else MODEL
    model = os.environ.get("WORMLET_JUDGE_MODEL")
    if model:
        return model
    raise RuntimeError(
        "WORMLET_JUDGE_MODEL is required with WORMLET_JUDGE_BACKEND=openai "
        "— set it to a model your server lists under /v1/models."
    )


def _openai_payload(user_prompt: str, corpus: str = "hamlet",
                    model: str | None = None) -> dict:
    """Request body for /chat/completions. No cache_control — OpenAI-compatible
    servers don't speak it, and local models don't bill input anyway.

    `model` overrides WORMLET_JUDGE_MODEL for callers that know which model
    they want. The bench does: it would otherwise have to set the env var
    around the call, and process-global state is not a safe way to pass an
    argument in a server handling concurrent requests."""
    return {
        "model": model or _judge_model("openai"),
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
        "stream": False,
        "messages": [
            {"role": "system", "content": judge_system_prompt(corpus)},
            {"role": "user", "content": user_prompt},
        ],
    }


def _http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """Tiny stdlib POST — deliberately no new dependency for a backend most
    installs never enable. Tests stub this symbol to stay off the network
    (which also skips the probe below).

    Fast-fail probe: a powered-OFF LAN host silently swallows SYNs, so a
    plain urlopen burned the full read timeout per attempt (observed: 600s
    x 3 retries x 10 worms ≈ a 40-minute failing rollover). A 4-second TCP
    probe detects a dead judge box in seconds instead."""
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    sock = socket.create_connection((parts.hostname, port), timeout=4)
    sock.close()
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _complete_openai(user_prompt: str, corpus: str = "hamlet",
                     model: str | None = None) -> str:
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
                base + "/chat/completions",
                _openai_payload(user_prompt, corpus, model),
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


# --- the human backend ------------------------------------------------------
# WORMLET_JUDGE_BACKEND=human makes a PERSON the critic. There is no network
# call, no model, and no API spend.
#
# WHAT THE PERSON ACTUALLY HAS TO DO, AND WHY IT IS SO LITTLE
# ----------------------------------------------------------
# Selection is rank-based. evolution.nes_update weights children by
# evolution.rank_weights, and for a 10-worm flask that vector is
#
#     [1.331  0.638  0.233  -0.055  -0.278  -0.374 x5]
#
# Three facts follow, and they are the whole reason a human judge is viable:
#   1. Absolute fitness VALUES are never read by the NES step — only the
#      order. Any monotone rescaling of the numbers below changes nothing.
#   2. The top three carry 100% of the positive gradient weight.
#   3. Ranks 6-10 are mathematically IDENTICAL. There is no information in
#      distinguishing the losers, so the UI must not ask anyone to try.
# So one verdict = an ordered list of the best few worms. Not 10 rankings,
# and emphatically not ~14 window scores x 10 worms the way an LLM judge is
# asked for them.
#
# THE PROTOCOL — two directories under WORMLET_DATA_DIR
# ----------------------------------------------------
#   ballots/<flask>/gen-NNNN/<worm>.json    written by THIS backend
#   verdicts/<flask>/gen-NNNN.json          written by the bench UI
#
# One ballot file PER WORM, never one merged file per generation: the per-worm
# file is written with a single atomic rename and no read-modify-write, so an
# aborted or re-fired rollover can never interleave two worms into a torn
# ballot. The bench reads the directory.
#
# A ballot carries every sampled window, not the subset the bench chooses to
# show. The record of what the worm produced must not depend on what the
# reader was shown.
#
# Verdict shape:
#   {"flask": "flask_1", "generation": 256, "judge": "joe",
#    "decided_at": 1760000000.0,
#    "top": ["<worm_a>", "<worm_b>", "<worm_c>"],      # ordered, 1-5 entries
#    "starred": {"<worm_a>": [45, 120]},               # optional, keepsake
#    "coherence": {"<worm_a>": {"45": 72}}}            # optional, see below
#
# STARS DELIBERATELY DO NOT ENTER FITNESS. They are a commonplace book of
# lines worth keeping, and they annotate the generation artifacts. Letting
# them move fitness would mean a heavily-starred third-place worm could
# overtake first place — i.e. the arithmetic silently contradicting the
# person's own stated order. `top` is the only selection signal, so the
# fitness ordering provably equals the human's ordering.
#
# THE TWO AXES. The person scores ONE thing: is this fragment any good. That
# lands on `emotional`. `coherence` ("reads as language rather than noise")
# is mechanical, so the bench may fill it from a local model and put it in
# the verdict; when it does, the two axes keep meaning what they say. When
# the verdict has no coherence block — local model offline, or never
# configured — coherence MIRRORS emotional, and the column then carries no
# independent information. Check metadata.json's judge_model before reading
# anything into a human lineage's coherence numbers.
_HUMAN_TIER_SCORES = (92, 78, 64, 54, 48)  # the person's 1st..5th
_HUMAN_REST_SCORE = 30                     # everyone else; NES ties them anyway


class JudgeAwaitingHuman(RuntimeError):
    """No verdict on disk yet for this flask-generation.

    Raised per worm, so the rollover's judge-outage guard in
    generations.py sees an all-unscored field and aborts WITHOUT touching
    state: the flask never respawns, its corpus stays exhausted, and the
    rollover re-fires. That is the entire scheduler for a human-paced
    lineage — it waits, and it never spends a generation nobody judged.

    Carried as its own type so app.py can log it at INFO on a cooldown
    instead of dumping a traceback every few seconds while someone takes
    their time (see _ROLLOVER_COOLDOWN_S there).
    """

    def __init__(self, flask: str, generation: int, verdict_path: Path):
        self.flask = flask
        self.generation = generation
        self.verdict_path = verdict_path
        super().__init__(
            f"awaiting human verdict for {flask} gen-{generation:04d} "
            f"({verdict_path})"
        )


def _human_root(kind: str) -> Path:
    """`ballots` or `verdicts` under the lineage's own data dir. Resolved at
    call time, like every other env read in this module, so a test or an
    experiment process can point it elsewhere without a reimport."""
    base = os.environ.get("WORMLET_DATA_DIR")
    root = Path(base) if base else Path(__file__).resolve().parents[2] / "v6" / "data"
    return root / kind


def _ballot_path(flask: str, generation: int, worm_name: str) -> Path:
    return (_human_root("ballots") / flask / f"gen-{generation:04d}"
            / f"{worm_name}.json")


def _verdict_path(flask: str, generation: int) -> Path:
    return _human_root("verdicts") / flask / f"gen-{generation:04d}.json"


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file in the SAME directory then rename, so the bench
    never reads a half-written ballot. Same-directory is the point: rename is
    only atomic within a filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _complete_human(sampled: list[tuple[int, list[str]]], worm_name: str,
                    flask: str, generation: int, corpus: str) -> str:
    """Drop this worm's ballot, then read the verdict and emit the same CSV
    an LLM backend would return, so judge_poem's parsing, renumber guard and
    ScoredWindow construction are shared verbatim.

    Window labels are PROMPT INDICES (0..n-1), matching _format_user_prompt,
    because that is what _parse_scores expects and what judge_poem maps back
    to true token offsets. The ballot records the true token offsets as well,
    since the verdict's `starred` list refers to them."""
    ballot = _ballot_path(flask, generation, worm_name)
    # Idempotent: the rollover re-fires every cooldown until a verdict lands,
    # and rewriting would churn the file the bench is reading.
    #
    # An existing ballot is still the RIGHT one, which matters because the
    # verdict's `starred` entries are token offsets into this exact stream.
    # Two things hold it still: the flask does not respawn until the rollover
    # completes, and TextScroller.corpus_exhausted — the condition that
    # triggered this rollover — is only true once every sentence has spawned
    # and no line is left on screen. There is physically nothing left to eat
    # while we wait, so the stream cannot grow under the ballot.
    if not ballot.exists():
        _write_atomic(ballot, json.dumps({
            "flask": flask,
            "generation": generation,
            "corpus": corpus,
            "worm": worm_name,
            "created_at": time.time(),
            # prompt index -> (true token offset, tokens). The bench shows
            # `text`; the verdict refers to windows by `idx`.
            "windows": [{"i": i, "idx": idx, "text": " ".join(toks)}
                        for i, (idx, toks) in enumerate(sampled)],
        }, indent=1))

    verdict_file = _verdict_path(flask, generation)
    try:
        verdict = json.loads(verdict_file.read_text())
    except FileNotFoundError:
        raise JudgeAwaitingHuman(flask, generation, verdict_file) from None
    except (OSError, ValueError) as e:
        # A corrupt verdict must NOT be silently treated as "not yet judged",
        # or the lineage waits forever on a file that will never parse.
        raise RuntimeError(f"unreadable verdict {verdict_file}: {e!r}") from e

    top = [str(n) for n in (verdict.get("top") or [])]
    if worm_name in top:
        rank = top.index(worm_name)
        emotional = (_HUMAN_TIER_SCORES[rank] if rank < len(_HUMAN_TIER_SCORES)
                     else _HUMAN_REST_SCORE)
    else:
        emotional = _HUMAN_REST_SCORE

    coh_map = (verdict.get("coherence") or {}).get(worm_name) or {}
    lines = []
    for i, (idx, _toks) in enumerate(sampled):
        # Keys may arrive as ints (python-written) or strings (JSON round
        # trip), and the value is clamped by _parse_scores anyway.
        coh = coh_map.get(str(idx), coh_map.get(idx))
        coherence = int(coh) if isinstance(coh, (int, float)) else emotional
        lines.append(f"{i},{emotional},{coherence}")
    return "\n".join(lines)

# {corpus_title} is filled per flask (corpus/library.TITLES). Three variants
# means three prompt-cache entries — ~350 tokens each, negligible; within one
# flask's generation the prompt stays byte-identical so caching still works.
JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are a careful judge of poetry produced by an artificial worm-like neural agent that has been eating words from {corpus_title}. The worm's eaten-word sequence forms a found poem — most fragments will be partly nonsensical, but some may surprise you with rhythm, emotional resonance, or accidental coherence.

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
    (unused: the Exp-1 "5 fixed windows" result was retracted — see the
    SAMPLE_FRACTION comment above); otherwise fall back to `fraction`. Seed
    controls which windows get picked so the same poem reproduces the same
    sampling across re-runs."""
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
    """Windows are numbered 0..n-1 IN PROMPT ORDER, not by token offset, and
    the prompt states the expected line count explicitly. Both are model-
    compliance armour: sparse offset labels (0, 15, 30…) and especially
    single-window prompts derailed a small local judge — qwen2.5:14b answered
    a lone '0: <15 tokens>' with fifteen invented lines numbered 1-15, so
    every score missed the join and the whole flask scored zero (2026-08-14).
    judge_poem maps prompt indices back to true token offsets."""
    n = len(sampled)
    header = (f"{n} window{'s' if n != 1 else ''} follow{'s' if n == 1 else ''}. "
              f"Output exactly {n} line{'s' if n != 1 else ''}, "
              f"one per window, using only the window indices shown.")
    lines = [f"{i}: {' '.join(toks)}" for i, (_idx, toks) in enumerate(sampled)]
    return header + "\n" + "\n".join(lines)


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


def judge_description() -> str:
    """Provenance string for generation artifacts: the PRIMARY judge's
    backend and model (fallback scorings are additionally logged per worm at
    judge time). Never raises — provenance must not break a rollover."""
    be = _judge_backend()
    try:
        model = _judge_model(be)
    except RuntimeError:
        model = "unconfigured"
    return f"{be}:{model}"


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


def _complete_anthropic(user_prompt: str, worm_name: str,
                        corpus: str = "hamlet") -> str:
    client = _client()
    response = client.messages.create(
        model=_judge_model("anthropic"),
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        system=[
            {
                "type": "text",
                "text": judge_system_prompt(corpus),
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
               seed: int | None = None,
               corpus: str = "hamlet",
               flask: str | None = None,
               generation: int | None = None) -> list[ScoredWindow]:
    """Score a worm's poem by sampling ~25% of its 15-token windows and
    sending them to Claude in one prompt-cached call.

    Returns one ScoredWindow per sampled window. Windows the API failed
    to score are silently dropped — the caller can detect this by
    comparing len(result) to the expected sample count and decide whether
    to retry or accept partial coverage.

    `flask` and `generation` are only consulted by the human backend, which
    files its ballots per flask-generation. They are keyword-optional so
    every existing caller and test double keeps working unchanged; the human
    backend refuses to run without them rather than guessing a path."""
    windows = make_windows(tokens)
    if not windows:
        return []
    sampled = sample_windows(windows, seed=seed)
    if not sampled:
        return []

    user_prompt = _format_user_prompt(sampled)

    # Backend chain: primary, then the optional fallback. Each backend gets
    # its own full retry cycle; a fallback success is logged loudly because
    # those generations are judged by a DIFFERENT critic (splice caveat in
    # _judge_fallback's docstring).
    backends = [_judge_backend()]
    fb = _judge_fallback()
    if fb and fb not in backends:
        backends.append(fb)
    text = None
    last_err: Exception | None = None
    for be in backends:
        try:
            if be == "anthropic":
                text = _complete_anthropic(user_prompt, worm_name, corpus)
            elif be == "openai":
                text = _complete_openai(user_prompt, corpus)
            elif be == "human":
                if flask is None or generation is None:
                    raise RuntimeError(
                        "WORMLET_JUDGE_BACKEND=human needs flask= and "
                        "generation= passed to judge_poem — a ballot has "
                        "nowhere to go without them"
                    )
                text = _complete_human(sampled, worm_name, flask, generation,
                                       corpus)
            else:
                raise RuntimeError(
                    f"unknown judge backend {be!r} — use 'anthropic', "
                    f"'openai' or 'human'"
                )
            if be != backends[0]:
                print(f"[JUDGE] primary backend failed; scored {worm_name} via "
                      f"FALLBACK {be} ({_judge_model(be)}) — splice caveat applies",
                      flush=True)
            break
        except JudgeAwaitingHuman:
            # NOT a failure — "not yet". Must never fall through to a paid
            # backend: with human primary and anthropic fallback, every
            # generation the person had not got round to judging would be
            # silently bought from the API instead, which is the exact spend
            # a human judge exists to end. The lineage waits.
            raise
        except Exception as e:
            last_err = e
            if be != backends[-1]:
                print(f"[JUDGE] backend {be} failed for {worm_name}: {e}; "
                      f"trying fallback", flush=True)
    if text is None:
        raise last_err if last_err else RuntimeError("no judge backend configured")
    scores = _parse_scores(text)

    # Renumber fallback: some models renumber anyway. Two safe shapes:
    #   * exact shift-by-one (model wrote 1..n for our 0..n-1) — for n >= 2
    #     this OVERLAPS the expected set, so an overlap test alone never
    #     fires and, worse, window i would silently receive the score meant
    #     for window i-1. Detect the exact shift and shift back.
    #   * fully disjoint labels with a matching count — zip positionally.
    # Anything else (partial overlap that isn't a clean shift, count
    # mismatch) is left alone: unmatched windows score nothing rather than
    # guessing, and the rollover's all-zero guard handles the fallout.
    expected = set(range(len(sampled)))
    if scores and len(scores) == len(sampled) and set(scores) != expected:
        keys = sorted(scores)
        if keys == list(range(1, len(sampled) + 1)):
            scores = {k - 1: scores[k] for k in keys}
        elif not (set(scores) & expected):
            scores = dict(enumerate(scores[k] for k in keys))

    out: list[ScoredWindow] = []
    for i, (idx, toks) in enumerate(sampled):
        if i in scores:
            e, c = scores[i]
            out.append(ScoredWindow(idx=idx, tokens=toks, emotional=e, coherence=c))
    return out


def score_windows_openai(sampled: list[tuple[int, list[str]]],
                         corpus: str = "hamlet",
                         model: str | None = None) -> dict[int, tuple[int, int]]:
    """Score windows on a local OpenAI-compatible endpoint and return
    {true_token_offset: (emotional, coherence)}.

    Exists for the bench UI, which uses a local model for the MECHANICAL
    axis only — "does this read as language" — and leaves the aesthetic call
    to the person. It deliberately goes through the same rubric,
    _format_user_prompt and _parse_scores as the judging path rather than
    growing a second prompt: every compliance fix in those functions was
    paid for by a real failure (notably qwen2.5:14b answering a single
    window with fifteen invented lines, 2026-08-14), and a parallel prompt
    would re-learn all of them.

    Raises on an unreachable endpoint. Callers are expected to degrade —
    the bench shows a banner and shortlists without a model. This must
    never be wired into judge_poem's human path: a lineage that stalls
    because a gaming PC is asleep is not a human-paced lineage.
    """
    if not sampled:
        return {}
    text = _complete_openai(_format_user_prompt(sampled), corpus, model)
    by_prompt_idx = _parse_scores(text)
    return {idx: by_prompt_idx[i] for i, (idx, _t) in enumerate(sampled)
            if i in by_prompt_idx}


def judge_system_prompt(corpus: str = "hamlet") -> str:
    """The rubric with the flask's text named. Corpus-neutral otherwise —
    same axes, banding, and CSV protocol for every flask."""
    from corpus import library
    return JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
        corpus_title=library.TITLES.get(corpus or "hamlet",
                                        library.TITLES["hamlet"]))
