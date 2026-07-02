"""Tests for the #648 billing/auth fail-fast path.

Since 2026-06-15 every Claude call failed with a billing 400, yet 18 daily
runs stayed green: the per-stage ``except Exception`` handlers failed closed
to ``is_new_limit=False`` and 85 papers were falsely marked processed. These
tests pin the fix: availability errors become :class:`FatalAPIError`, stage
handlers re-raise it instead of swallowing it, and the orchestrator aborts
without marking the paper.

No network: anthropic error objects are constructed directly over synthetic
httpx responses.

Run:
    pytest evaluation/tests/test_fatal_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

anthropic = pytest.importorskip("anthropic")
import httpx  # httpx is an anthropic dependency, present whenever it is

from pipeline.extractor import (
    FatalAPIError,
    _call_with_retry,
    _fatal_api_reason,
    _run_stage1,
)


def _api_error(cls, status: int, message: str):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req)
    return cls(message, response=resp, body={"error": {"message": message}})


def _billing_400():
    return _api_error(
        anthropic.BadRequestError, 400,
        "Your credit balance is too low to access the Anthropic API.")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_billing_400_is_fatal():
    assert _fatal_api_reason(_billing_400()) is not None


def test_auth_401_and_permission_403_are_fatal():
    assert _fatal_api_reason(
        _api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"))
    assert _fatal_api_reason(
        _api_error(anthropic.PermissionDeniedError, 403, "forbidden"))


def test_ordinary_400_is_not_fatal():
    e = _api_error(anthropic.BadRequestError, 400,
                   "max_tokens must be a positive integer")
    assert _fatal_api_reason(e) is None


# ---------------------------------------------------------------------------
# _call_with_retry conversion
# ---------------------------------------------------------------------------

def test_retry_converts_billing_400_to_fatal():
    def _fn():
        raise _billing_400()
    with pytest.raises(FatalAPIError):
        _call_with_retry(_fn)


def test_retry_raises_plain_error_for_ordinary_400():
    def _fn():
        raise _api_error(anthropic.BadRequestError, 400, "bad param")
    with pytest.raises(anthropic.BadRequestError):
        _call_with_retry(_fn)


def test_retry_still_retries_overload_then_succeeds():
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _api_error(anthropic.InternalServerError, 529, "overloaded")
        return "ok"

    assert _call_with_retry(_fn, base_delay=0.0) == "ok"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Stage handlers must NOT fail closed on availability errors
# ---------------------------------------------------------------------------

class _BillingClient:
    """messages.create always raises the billing 400."""
    class _Messages:
        def create(self, **kwargs):
            raise _billing_400()
    def __init__(self):
        self.messages = self._Messages()


class _Paper:
    title = "A fake axion paper"
    summary = "abstract text"
    categories = ["hep-ex"]


def test_stage1_propagates_fatal_instead_of_not_new_limit():
    # Before #648 this returned {"is_new_limit": False, ...} — the exact
    # failure that burned 85 papers.
    with pytest.raises(FatalAPIError):
        _run_stage1(_Paper(), "some pdf text", _BillingClient())


# ---------------------------------------------------------------------------
# Orchestrator: abort without marking the paper
# ---------------------------------------------------------------------------

def test_orchestrator_preflight_aborts_on_billing(monkeypatch):
    from pipeline import orchestrator
    with pytest.raises(FatalAPIError):
        orchestrator.preflight_api_check(_BillingClient())


def test_orchestrator_preflight_tolerates_transient(monkeypatch):
    class _FlakyClient:
        class _Messages:
            def create(self, **kwargs):
                raise ConnectionError("transient network blip")
        def __init__(self):
            self.messages = self._Messages()
    from pipeline import orchestrator
    orchestrator.preflight_api_check(_FlakyClient())  # must not raise


def test_orchestrator_loop_aborts_without_marking(monkeypatch, tmp_path):
    from pipeline import orchestrator

    marks = {"processed": [], "failed": []}
    monkeypatch.setattr(orchestrator, "preflight_api_check", lambda c: None)
    monkeypatch.setattr(orchestrator, "load_state", lambda: {"processed_ids": []})
    monkeypatch.setattr(orchestrator, "save_state", lambda s: None)
    monkeypatch.setattr(orchestrator, "mark_processed",
                        lambda s, pid, reason=None: marks["processed"].append(pid))
    monkeypatch.setattr(orchestrator, "mark_failed",
                        lambda s, pid, msg: marks["failed"].append(pid))
    monkeypatch.setattr(orchestrator, "fetch_paper_by_id", lambda aid: _Paper())
    monkeypatch.setattr(
        orchestrator, "_process_paper",
        lambda paper, pid, client, state, dry: (_ for _ in ()).throw(
            FatalAPIError("credit balance exhausted")))
    monkeypatch.setattr(orchestrator.anthropic, "Anthropic",
                        lambda api_key=None: _BillingClient())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    import pipeline.monitor as monitor
    monkeypatch.setattr(monitor, "_arxiv_id", lambda p: "2604.00001", raising=False)

    with pytest.raises(SystemExit) as exc:
        orchestrator.main(arxiv_id="2604.00001")
    assert exc.value.code == orchestrator.EXIT_FATAL_API
    # The decisive assertion: the outage paper was never marked.
    assert marks["processed"] == []
    assert marks["failed"] == []
