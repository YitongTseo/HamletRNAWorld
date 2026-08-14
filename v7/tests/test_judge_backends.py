"""Tests for the pluggable judge backend — env dispatch, payload shape,
response handling. No network: the HTTP POST is stubbed at
judge._http_post_json, and the anthropic path is never invoked."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server.judge as judge


def _swap_env(**kv):
    """Set/unset env vars, returning the previous values for restore.
    Value None means 'ensure unset'."""
    old = {}
    for k, v in kv.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_default_backend_is_anthropic():
    old = _swap_env(WORMLET_JUDGE_BACKEND=None, WORMLET_JUDGE_MODEL=None)
    try:
        assert judge._judge_backend() == "anthropic"
        # unset model falls back to the module constant, no error
        assert judge._judge_model() == judge.MODEL
    finally:
        _restore_env(old)


def test_openai_backend_requires_model():
    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL=None)
    try:
        try:
            judge._judge_model()
            assert False, "expected RuntimeError for missing WORMLET_JUDGE_MODEL"
        except RuntimeError as e:
            assert "WORMLET_JUDGE_MODEL" in str(e)
    finally:
        _restore_env(old)


def test_openai_payload_shape():
    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL="gemma3:4b")
    try:
        p = judge._openai_payload("0: to be or not")
        assert p["model"] == "gemma3:4b"
        assert p["temperature"] == 0.0  # the judge-noise fix must survive transport
        assert p["max_tokens"] == judge.MAX_OUTPUT_TOKENS
        assert p["stream"] is False
        assert p["messages"][0]["role"] == "system"
        assert p["messages"][0]["content"] == judge.JUDGE_SYSTEM_PROMPT
        assert p["messages"][1] == {"role": "user", "content": "0: to be or not"}
    finally:
        _restore_env(old)


def test_unknown_backend_raises():
    old = _swap_env(WORMLET_JUDGE_BACKEND="palmreader")
    try:
        try:
            judge.judge_poem(["word"] * judge.WINDOW_SIZE, "wormy", seed=1)
            assert False, "expected RuntimeError for unknown backend"
        except RuntimeError as e:
            assert "palmreader" in str(e)
    finally:
        _restore_env(old)


def test_judge_poem_openai_end_to_end_stubbed():
    """Full judge_poem pass through the openai backend with the POST stubbed:
    the CSV comes back keyed by the sampled window idx and parses into
    ScoredWindows exactly like the anthropic path."""
    tokens = ["word"] * (judge.WINDOW_SIZE * 4)  # 4 full windows
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen["url"] = url
        seen["body"] = body
        seen["headers"] = headers
        seen["timeout"] = timeout
        # score every window index present in the user prompt (skip the
        # "N windows follow…" header line)
        lines = []
        for line in body["messages"][1]["content"].splitlines():
            head = line.split(":", 1)[0]
            if head.isdigit():
                lines.append(f"{head},70,55")
        return {"choices": [{"message": {"content": "\n".join(lines)}}]}

    old = _swap_env(
        WORMLET_JUDGE_BACKEND="openai",
        WORMLET_JUDGE_MODEL="gemma3:4b",
        WORMLET_JUDGE_URL="http://192.0.2.1:11434/v1/",  # trailing slash on purpose
        WORMLET_JUDGE_API_KEY="sekrit",
        WORMLET_JUDGE_TIMEOUT_S="7",
    )
    real_post = judge._http_post_json
    judge._http_post_json = fake_post
    try:
        out = judge.judge_poem(tokens, "wormy", seed=42)
        assert out, "expected at least one scored window (25% of 4 windows = 1)"
        assert all(isinstance(s, judge.ScoredWindow) for s in out)
        assert all(s.emotional == 70 and s.coherence == 55 for s in out)
        assert seen["url"] == "http://192.0.2.1:11434/v1/chat/completions"
        assert seen["headers"] == {"Authorization": "Bearer sekrit"}
        assert seen["timeout"] == 7.0
        # determinism: same seed samples the same windows either backend
        assert [s.idx for s in out] == [
            w[0] for w in judge.sample_windows(judge.make_windows(tokens), seed=42)
        ]
    finally:
        judge._http_post_json = real_post
        _restore_env(old)


def test_openai_retries_transient_errors():
    """Two connection failures then success: judge_poem should retry through
    them (sleeps stubbed out) and return scores on the third attempt."""
    calls = {"n": 0}

    def flaky(url, body, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection refused")
        return {"choices": [{"message": {"content": "0,50,50"}}]}

    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL="gemma3:4b")
    real_post, real_sleep = judge._http_post_json, judge.time.sleep
    judge._http_post_json = flaky
    judge.time.sleep = lambda s: None
    try:
        out = judge.judge_poem(["word"] * judge.WINDOW_SIZE, "wormy", seed=1)
        assert calls["n"] == 3
        assert out and out[0].emotional == 50
    finally:
        judge._http_post_json = real_post
        judge.time.sleep = real_sleep
        _restore_env(old)


def test_openai_gives_up_after_three_attempts():
    calls = {"n": 0}

    def dead(url, body, headers, timeout):
        calls["n"] += 1
        raise OSError("no route to host")

    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL="gemma3:4b")
    real_post, real_sleep = judge._http_post_json, judge.time.sleep
    judge._http_post_json = dead
    judge.time.sleep = lambda s: None
    try:
        try:
            judge.judge_poem(["word"] * judge.WINDOW_SIZE, "wormy", seed=1)
            assert False, "expected RuntimeError after retries exhausted"
        except RuntimeError as e:
            assert "unreachable after 3 attempts" in str(e)
        assert calls["n"] == 3
    finally:
        judge._http_post_json = real_post
        judge.time.sleep = real_sleep
        _restore_env(old)


def test_openai_renumbered_response_maps_positionally():
    """qwen2.5:14b answered prompt indices 0..n-1 with 1..n. Same line count,
    zero index overlap → positional fallback must recover the scores."""

    def fake(url, body, headers, timeout):
        idxs = [l.split(":", 1)[0] for l in body["messages"][1]["content"].splitlines()]
        idxs = [i for i in idxs if i.isdigit()]
        return {"choices": [{"message": {
            "content": "\n".join(f"{int(i) + 1},40,30" for i in idxs)}}]}

    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL="gemma3:4b")
    real_post = judge._http_post_json
    judge._http_post_json = fake
    try:
        out = judge.judge_poem(["word"] * judge.WINDOW_SIZE, "wormy", seed=1)
        assert len(out) == 1 and out[0].emotional == 40 and out[0].coherence == 30
        assert out[0].idx == 0  # true token offset restored, not the model's label
    finally:
        judge._http_post_json = real_post
        _restore_env(old)


def test_openai_hallucinated_extra_windows_score_nothing():
    """The 2026-08-14 outage shape: one window sent, fifteen invented lines
    back. Count mismatch → no positional guess, zero windows — the rollover
    guard aborts instead of laundering noise into fitness."""

    def fake(url, body, headers, timeout):
        return {"choices": [{"message": {
            "content": "\n".join(f"{i},{50 + i},{40 + i}" for i in range(1, 16))}}]}

    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL="gemma3:4b")
    real_post = judge._http_post_json
    judge._http_post_json = fake
    try:
        out = judge.judge_poem(["word"] * judge.WINDOW_SIZE, "wormy", seed=1)
        assert out == []
    finally:
        judge._http_post_json = real_post
        _restore_env(old)


def test_user_prompt_has_count_header_and_sequential_indices():
    sampled = [(0, ["a"] * 15), (45, ["b"] * 15)]
    p = judge._format_user_prompt(sampled)
    first, *rest = p.splitlines()
    assert first.startswith("2 windows follow")
    assert rest[0].startswith("0: ") and rest[1].startswith("1: ")  # not 45


def test_openai_bad_response_shape_raises():
    old = _swap_env(WORMLET_JUDGE_BACKEND="openai", WORMLET_JUDGE_MODEL="gemma3:4b")
    real_post = judge._http_post_json
    judge._http_post_json = lambda url, body, headers, timeout: {"error": "boom"}
    try:
        try:
            judge.judge_poem(["word"] * judge.WINDOW_SIZE, "wormy", seed=1)
            assert False, "expected RuntimeError on malformed response"
        except RuntimeError as e:
            assert "chat/completions" in str(e)
    finally:
        judge._http_post_json = real_post
        _restore_env(old)
