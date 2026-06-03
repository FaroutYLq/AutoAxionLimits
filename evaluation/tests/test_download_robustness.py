"""Tests for download_pdf robustness: cross-run cache + 429/timeout retry.

No network: httpx.Client is monkeypatched and time.sleep is stubbed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_HAVE_STACK = True
try:
    import httpx
    from pipeline import extractor
except Exception:
    _HAVE_STACK = False

requires_stack = pytest.mark.skipif(
    not _HAVE_STACK, reason="pipeline.extractor / httpx unavailable")


class _Resp:
    def __init__(self, status=200, content=b"%PDF-1.4 data"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self
            )


class _FakeClient:
    """Context-manager httpx.Client stand-in driven by a scripted response list."""

    script = []      # list of _Resp or Exception, consumed per .get()
    calls = 0

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        item = type(self).script[type(self).calls]
        type(self).calls += 1
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    if _HAVE_STACK:
        monkeypatch.setattr(extractor.time, "sleep", lambda *_: None)


def _install(monkeypatch, script, cache_dir):
    _FakeClient.script = script
    _FakeClient.calls = 0
    monkeypatch.setattr(extractor.httpx, "Client", _FakeClient)
    monkeypatch.setenv("AAL_PDF_CACHE", str(cache_dir))


@requires_stack
def test_retry_then_success(monkeypatch, tmp_path):
    _install(monkeypatch, [httpx.ConnectTimeout("t"), _Resp(429), _Resp(200)],
             tmp_path / "cache")
    out = extractor.download_pdf("1234.5678", tmp_path, base_delay=0.0)
    assert out.exists() and out.read_bytes().startswith(b"%PDF")
    assert _FakeClient.calls == 3                      # 2 failures + 1 success
    # cache populated
    assert (tmp_path / "cache" / "1234.5678.pdf").exists()


@requires_stack
def test_cache_hit_skips_network(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "1234.5678.pdf").write_bytes(b"%PDF cached")
    # any network use would IndexError on empty script
    _install(monkeypatch, [], cache)
    out = extractor.download_pdf("1234.5678", tmp_path)
    assert out.read_bytes() == b"%PDF cached"
    assert _FakeClient.calls == 0


@requires_stack
def test_404_not_retried(monkeypatch, tmp_path):
    _install(monkeypatch, [_Resp(404), _Resp(200)], tmp_path / "cache")
    with pytest.raises(httpx.HTTPStatusError):
        extractor.download_pdf("1234.5678", tmp_path, base_delay=0.0)
    assert _FakeClient.calls == 1                       # not retried


@requires_stack
def test_exhausts_retries_then_raises(monkeypatch, tmp_path):
    _install(monkeypatch, [httpx.ConnectTimeout("t")] * 5, tmp_path / "cache")
    with pytest.raises(httpx.TimeoutException):
        extractor.download_pdf("1234.5678", tmp_path, max_retries=3, base_delay=0.0)
    assert _FakeClient.calls == 3


@requires_stack
def test_cache_disabled_via_env(monkeypatch):
    monkeypatch.setenv("AAL_PDF_CACHE", "")
    assert extractor._pdf_cache_dir() is None


@requires_stack
def test_cache_default_path(monkeypatch):
    monkeypatch.delenv("AAL_PDF_CACHE", raising=False)
    d = extractor._pdf_cache_dir()
    assert d is not None and d.name == "aal_pdf_cache"
