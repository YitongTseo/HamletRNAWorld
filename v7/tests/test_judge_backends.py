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
        # score every window idx present in the user prompt
        lines = []
        for line in body["messages"][1]["content"].splitlines():
            idx = int(line.split(":", 1)[0])
            lines.append(f"{idx},70,55")
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
