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
