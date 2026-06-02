"""Tests for the bounded arXiv metadata fetch (issue #560).

The arxiv library calls ``requests.Session().get()`` with no timeout, so a
slow/throttling ``export.arxiv.org`` endpoint can hang ``_fetch_paper_metadata``
(and therefore the whole evaluation run) indefinitely. The fetch is now wrapped
in a hard wall-clock deadline with bounded retries and a graceful fallback to
the ground-truth title path.

These tests make **no** network calls — they monkeypatch the arxiv client to
hang or to raise, and assert the function returns the fallback within a bounded
wall-clock time instead of blocking forever.

Run:
    pytest evaluation/tests/test_arxiv_timeout.py -v
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

from evaluation import evaluate


def _install_fake_arxiv(monkeypatch, results_fn):
    """Install a fake ``arxiv`` module whose ``Client().results`` is results_fn.

    ``_fetch_paper_metadata`` does ``import arxiv as _arxiv`` internally, so we
    inject a stub into ``sys.modules`` to intercept it (no network).
    """
    fake = types.ModuleType("arxiv")

    class _FakeSearch:
        def __init__(self, *a, **k):
            pass

    class _FakeClient:
        def results(self, search):
            return results_fn()

    fake.Search = _FakeSearch
    fake.Client = _FakeClient
    monkeypatch.setitem(sys.modules, "arxiv", fake)


def test_arxiv_timeout_falls_back_within_bound(monkeypatch, tmp_path):
    """A hanging fetch returns the ("", "") fallback within the bounded time."""
    # Shrink the deadline/retries so the test is fast and deterministic.
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_TIMEOUT_S", 1)
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_RETRIES", 2)
    # Avoid the real (exponential) backoff sleeps slowing the test.
    monkeypatch.setattr(evaluate.time, "sleep", lambda *_a, **_k: None)

    def _hang():
        # Simulate a stalled CDN: block far longer than the deadline.
        time.sleep(60)
        return iter([])

    _install_fake_arxiv(monkeypatch, _hang)

    cache = tmp_path / "metadata_cache.json"
    t0 = time.time()
    title, abstract = evaluate._fetch_paper_metadata("1234.5678", cache)
    elapsed = time.time() - t0

    assert (title, abstract) == ("", "")  # graceful fallback, no crash
    # 2 attempts * 1s deadline (backoff is patched out) -> well under 10s.
    assert elapsed < 10, f"fetch did not bound the wait (took {elapsed:.1f}s)"
    # A timed-out fetch must NOT poison the cache.
    assert not cache.exists()


def test_arxiv_error_falls_back(monkeypatch, tmp_path):
    """A raising fetch (e.g. HTTP 429) degrades to the fallback, not a crash."""
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_TIMEOUT_S", 1)
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_RETRIES", 3)
    monkeypatch.setattr(evaluate.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise RuntimeError("HTTP 429 Too Many Requests")

    _install_fake_arxiv(monkeypatch, _boom)

    cache = tmp_path / "metadata_cache.json"
    title, abstract = evaluate._fetch_paper_metadata("1234.5678", cache)

    assert (title, abstract) == ("", "")
    assert calls["n"] == 3  # retried exactly ARXIV_FETCH_RETRIES times
    assert not cache.exists()


def test_arxiv_success_caches(monkeypatch, tmp_path):
    """The happy path still returns real metadata and writes the cache."""
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_TIMEOUT_S", 5)
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_RETRIES", 3)

    class _Result:
        title = "A Real Title"
        summary = "A real abstract."

    def _ok():
        return iter([_Result()])

    _install_fake_arxiv(monkeypatch, _ok)

    cache = tmp_path / "metadata_cache.json"
    title, abstract = evaluate._fetch_paper_metadata("1234.5678", cache)

    assert title == "A Real Title"
    assert abstract == "A real abstract."
    assert cache.exists()  # successful fetch is cached


def test_arxiv_cache_hit_skips_fetch(monkeypatch, tmp_path):
    """A cached id returns immediately without touching the arxiv client."""
    import json

    cache = tmp_path / "metadata_cache.json"
    cache.write_text(json.dumps({"1234.5678": {"title": "Cached", "abstract": "Cached abs"}}))

    def _should_not_run():
        raise AssertionError("arxiv client must not be called on a cache hit")

    _install_fake_arxiv(monkeypatch, _should_not_run)

    title, abstract = evaluate._fetch_paper_metadata("1234.5678", cache)
    assert (title, abstract) == ("Cached", "Cached abs")


# ---------------------------------------------------------------------------
# Hardening: the worker must actually DIE (socket timeout) and must never be
# able to block process exit (daemon thread). See issue #560 / PR #564.
# ---------------------------------------------------------------------------


def test_deadline_runs_on_daemon_thread():
    """The fetch runs on a daemon thread, so a hung worker can't block exit.

    ThreadPoolExecutor workers are non-daemon and ``_python_exit`` joins them
    unconditionally at interpreter shutdown — a still-hung fetch would stall the
    whole process. A daemon thread is never joined, closing that exit-hang.
    """
    import threading

    captured = {}

    def fn():
        captured["daemon"] = threading.current_thread().daemon
        return "ok"

    assert evaluate._call_with_deadline(fn, 5, label="x") == "ok"
    assert captured["daemon"] is True


def test_call_with_deadline_normalizes_requests_timeout():
    """A requests socket timeout surfaces as a builtin TimeoutError."""
    import requests

    def fn():
        raise requests.exceptions.ConnectTimeout("connect timed out")

    with pytest.raises(TimeoutError):
        evaluate._call_with_deadline(fn, 5, label="x")


def test_call_with_deadline_propagates_other_errors():
    """Non-timeout errors propagate unchanged (so callers can react to them)."""

    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        evaluate._call_with_deadline(fn, 5, label="x")


def test_install_session_timeout_injects_default_and_is_idempotent():
    """The injected timeout defaults a missing one but preserves an explicit one."""

    class _Session:
        def __init__(self):
            self.timeouts = []

        def request(self, method, url, **kwargs):
            self.timeouts.append(kwargs.get("timeout", "MISSING"))
            return "resp"

        def get(self, url, **kwargs):  # mirrors requests.Session.get -> request
            kwargs.setdefault("allow_redirects", True)
            return self.request("GET", url, **kwargs)

    class _Client:
        def __init__(self):
            self._session = _Session()

    client = _Client()
    evaluate._install_session_timeout(client, timeout_s=7)

    # The arxiv library calls .get(url, headers=...) with NO timeout; the
    # wrapper must inject the default so the underlying socket op can't hang.
    client._session.get("http://example/x")
    assert client._session.timeouts[-1] == 7

    # An explicit caller-supplied timeout is preserved (setdefault, not force).
    client._session.request("GET", "http://example/x", timeout=99)
    assert client._session.timeouts[-1] == 99

    # Idempotent: a second install does not re-wrap the already-wrapped session.
    wrapped = client._session.request
    evaluate._install_session_timeout(client, timeout_s=3)
    assert client._session.request is wrapped


def test_install_session_timeout_handles_missing_session():
    """A client without a ``_session`` (e.g. a test stub) is a harmless no-op."""

    class _NoSession:
        pass

    # Must not raise.
    evaluate._install_session_timeout(_NoSession(), timeout_s=5)


def test_requests_timeout_falls_back(monkeypatch, tmp_path):
    """A requests ReadTimeout degrades to the fallback, not a crash or hang."""
    import requests

    monkeypatch.setattr(evaluate, "ARXIV_FETCH_TIMEOUT_S", 5)
    monkeypatch.setattr(evaluate, "ARXIV_FETCH_RETRIES", 2)
    monkeypatch.setattr(evaluate.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def _timeout():
        calls["n"] += 1
        raise requests.exceptions.ReadTimeout("read timed out")

    _install_fake_arxiv(monkeypatch, _timeout)

    cache = tmp_path / "metadata_cache.json"
    title, abstract = evaluate._fetch_paper_metadata("1234.5678", cache)

    assert (title, abstract) == ("", "")
    assert calls["n"] == 2  # retried exactly ARXIV_FETCH_RETRIES times
    assert not cache.exists()
