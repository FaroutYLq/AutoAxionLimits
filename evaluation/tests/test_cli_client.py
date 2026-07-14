"""Unit tests for the headless-CLI transport shim (pipeline/cli_client.py).

No subprocess: ``subprocess.run`` is monkeypatched with a recorder returning a
canned ``CompletedProcess``. Pins the contract: kwargs translate to argv +
stdin VERBATIM (transport-only guarantee — no methodology change), the
untrusted-paper lockdown flags are always present, the subprocess env is
scrubbed of billing/session leakage, and error mapping preserves the #648
semantics (auth/billing/usage-limit -> FatalAPIError; rate-limit/overload ->
genuine anthropic errors that _call_with_retry backs off on).

Run:
    pytest evaluation/tests/test_cli_client.py -v
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

anthropic = pytest.importorskip("anthropic")

from pipeline import cli_client
from pipeline.cli_client import ClaudeCLIClient
from pipeline.extractor import FatalAPIError

# A minimal valid 1x1 PNG.
_PNG_B64 = base64.standard_b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8cfc0f01f0005000180ff9b8b8e0000000049454e44ae426082"
)).decode()


class _Recorder:
    """Records the last subprocess.run call and returns a scripted result."""

    def __init__(self, stdout="", stderr="", returncode=0, raises=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, argv, input=None, text=None, capture_output=None,
                 cwd=None, env=None, timeout=None):
        rec = {
            "argv": argv, "input": input, "cwd": cwd, "env": env,
            "timeout": timeout,
            "cwd_files": sorted(os.listdir(cwd)) if cwd else [],
        }
        self.calls.append(rec)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode,
                                           stdout=self.stdout, stderr=self.stderr)


def _success_json(result="{\"ok\": true}", model="claude-opus-4-8"):
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": result, "session_id": "sess",
        "usage": {"input_tokens": 10, "output_tokens": 3},
        "modelUsage": {model: {"inputTokens": 10}},
    })


def _install(monkeypatch, recorder):
    monkeypatch.setattr(cli_client.subprocess, "run", recorder)


def _text_call(client, **over):
    kwargs = dict(model="claude-opus-4-8", max_tokens=128, system="SYS",
                  messages=[{"role": "user", "content": "hello"}])
    kwargs.update(over)
    return client.messages.create(**kwargs)


def _vision_call(client, **over):
    kwargs = dict(
        model="claude-opus-4-8", max_tokens=512, temperature=0.0,
        system=[{"type": "text", "text": "SYS", "cache_control": {"ttl": "1h"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "look:"},
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": _PNG_B64}},
        ]}],
    )
    kwargs.update(over)
    return client.messages.create(**kwargs)


# --------------------------------------------------------------------------- #
# Happy path + response shape
# --------------------------------------------------------------------------- #

def test_text_response_text_and_usage(monkeypatch):
    rec = _Recorder(stdout=_success_json(result="hi there"))
    _install(monkeypatch, rec)
    resp = _text_call(ClaudeCLIClient())
    assert resp.content[0].text == "hi there"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 3


def test_prompt_delivered_via_stdin_not_argv(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _text_call(ClaudeCLIClient(), messages=[{"role": "user", "content": "PAYLOAD"}])
    call = rec.calls[-1]
    assert call["input"] == "PAYLOAD"
    assert "PAYLOAD" not in call["argv"]


# --------------------------------------------------------------------------- #
# kwargs translation
# --------------------------------------------------------------------------- #

def test_unsupported_kwargs_dropped_from_argv(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _text_call(ClaudeCLIClient(), temperature=0.0, max_tokens=128)
    argv = rec.calls[-1]["argv"]
    assert "temperature" not in argv
    assert "0.0" not in argv
    assert "--max-tokens" not in argv
    assert "128" not in argv


def test_system_string_lands_in_flag(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _text_call(ClaudeCLIClient(), system="PLAIN")
    argv = rec.calls[-1]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "PLAIN"


def test_system_block_list_joined(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _vision_call(ClaudeCLIClient())
    argv = rec.calls[-1]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "SYS"


def test_image_blocks_written_and_referenced_in_order(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _vision_call(ClaudeCLIClient())
    call = rec.calls[-1]
    assert any(f.endswith(".png") for f in call["cwd_files"])
    # marker text references the figure and comes after the "look:" text
    prompt = call["input"]
    assert "look:" in prompt
    assert "Read tool" in prompt
    assert prompt.index("look:") < prompt.index("Read tool")


def test_multiple_user_messages_rejected(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    with pytest.raises(RuntimeError):
        ClaudeCLIClient().messages.create(
            model="claude-opus-4-8", max_tokens=8,
            messages=[{"role": "user", "content": "a"},
                      {"role": "user", "content": "b"}])


# --------------------------------------------------------------------------- #
# Lockdown (prompt-injection defence) — the security pin
# --------------------------------------------------------------------------- #

def test_text_call_disables_all_tools(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _text_call(ClaudeCLIClient())
    argv = rec.calls[-1]["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    disallowed = argv[argv.index("--disallowedTools") + 1]
    for tool in ("Bash", "Edit", "Write", "WebFetch", "WebSearch"):
        assert tool in disallowed
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in argv


def test_vision_call_allows_only_read(monkeypatch):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _vision_call(ClaudeCLIClient())
    argv = rec.calls[-1]["argv"]
    assert argv[argv.index("--tools") + 1] == "Read"


def test_cwd_is_isolated_and_env_scrubbed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-scrubbed")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _text_call(ClaudeCLIClient())
    call = rec.calls[-1]
    assert call["cwd"] != os.getcwd()
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "CLAUDECODE" not in call["env"]
    assert not any(k.startswith("CLAUDE_CODE_") for k in call["env"])


# --------------------------------------------------------------------------- #
# Error mapping — the #648 contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("result_text", [
    "Your credit balance is too low",
    "Not logged in · Please run /login",
    "authentication failed",
    "usage limit reached, resets at 5pm",
])
def test_fatal_errors(monkeypatch, result_text):
    rec = _Recorder(stdout=json.dumps(
        {"type": "result", "is_error": True, "result": result_text}))
    _install(monkeypatch, rec)
    with pytest.raises(FatalAPIError):
        _text_call(ClaudeCLIClient())


def test_rate_limit_maps_to_ratelimiterror(monkeypatch):
    rec = _Recorder(stdout=json.dumps(
        {"type": "result", "is_error": True, "result": "rate limit exceeded"}))
    _install(monkeypatch, rec)
    with pytest.raises(anthropic.RateLimitError):
        _text_call(ClaudeCLIClient())


def test_overload_maps_to_529(monkeypatch):
    rec = _Recorder(stdout=json.dumps(
        {"type": "result", "is_error": True, "result": "server overloaded"}))
    _install(monkeypatch, rec)
    with pytest.raises(anthropic.APIStatusError) as ei:
        _text_call(ClaudeCLIClient())
    assert ei.value.status_code == 529


def test_unparseable_output_is_paper_scoped_runtimeerror(monkeypatch):
    rec = _Recorder(stdout="not json at all", stderr="???", returncode=1)
    _install(monkeypatch, rec)
    with pytest.raises(RuntimeError) as ei:
        _text_call(ClaudeCLIClient())
    assert not isinstance(ei.value, FatalAPIError)


def test_empty_result_is_runtimeerror(monkeypatch):
    rec = _Recorder(stdout=json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": "  "}))
    _install(monkeypatch, rec)
    with pytest.raises(RuntimeError):
        _text_call(ClaudeCLIClient())


def test_missing_binary_is_fatal(monkeypatch):
    _install(monkeypatch, _Recorder(raises=FileNotFoundError("no claude")))
    with pytest.raises(FatalAPIError):
        _text_call(ClaudeCLIClient())


def test_timeout_maps_to_529(monkeypatch):
    _install(monkeypatch, _Recorder(
        raises=subprocess.TimeoutExpired(cmd="claude", timeout=1)))
    with pytest.raises(anthropic.APIStatusError) as ei:
        _text_call(ClaudeCLIClient())
    assert ei.value.status_code == 529


def test_silent_model_substitution_is_fatal(monkeypatch):
    rec = _Recorder(stdout=_success_json(model="claude-sonnet-5"))
    _install(monkeypatch, rec)
    with pytest.raises(FatalAPIError):
        _text_call(ClaudeCLIClient())  # requested opus, CLI billed sonnet


# --------------------------------------------------------------------------- #
# _call_with_retry integration — proves backoff wiring end-to-end
# --------------------------------------------------------------------------- #

def test_retry_backoff_then_success(monkeypatch):
    import time as _t

    from pipeline.extractor import _call_with_retry

    monkeypatch.setattr(_t, "sleep", lambda *_: None)

    state = {"n": 0}

    def flaky(argv, input=None, text=None, capture_output=None, cwd=None,
              env=None, timeout=None):
        state["n"] += 1
        if state["n"] <= 2:
            return subprocess.CompletedProcess(
                argv, 1, stdout=json.dumps(
                    {"type": "result", "is_error": True,
                     "result": "rate limit"}), stderr="")
        return subprocess.CompletedProcess(
            argv, 0, stdout=_success_json(result="finally"), stderr="")

    _install(monkeypatch, flaky)
    client = ClaudeCLIClient()
    resp = _call_with_retry(lambda: _text_call(client))
    assert resp.content[0].text == "finally"
    assert state["n"] == 3


# --------------------------------------------------------------------------- #
# Temp-dir lifecycle
# --------------------------------------------------------------------------- #

def test_tempdir_cleaned_on_success(monkeypatch, tmp_path):
    rec = _Recorder(stdout=_success_json())
    _install(monkeypatch, rec)
    _text_call(ClaudeCLIClient(scratch_root=str(tmp_path)))
    assert not list(tmp_path.glob("aal-cli-*"))


def test_tempdir_cleaned_on_error(monkeypatch, tmp_path):
    rec = _Recorder(stdout=json.dumps(
        {"type": "result", "is_error": True, "result": "billing problem"}))
    _install(monkeypatch, rec)
    with pytest.raises(FatalAPIError):
        _text_call(ClaudeCLIClient(scratch_root=str(tmp_path)))
    assert not list(tmp_path.glob("aal-cli-*"))
