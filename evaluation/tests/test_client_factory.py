"""Unit tests for backend selection (pipeline/client_factory.py).

Pins: default backend is the API client (CI stays bit-identical); AAL_BACKEND
=claude-cli returns the CLI shim and skips the ANTHROPIC_API_KEY check;
required=False tolerates a missing key on the API backend (backfill
--discover-only); an unknown backend or a missing CLI binary exits non-zero.

Run:
    pytest evaluation/tests/test_client_factory.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

anthropic = pytest.importorskip("anthropic")

from pipeline import client_factory
from pipeline.cli_client import ClaudeCLIClient


def test_default_backend_is_api_client(monkeypatch):
    monkeypatch.delenv("AAL_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = client_factory.make_client()
    assert isinstance(client, anthropic.Anthropic)


def test_api_backend_missing_key_exits(monkeypatch):
    monkeypatch.delenv("AAL_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        client_factory.make_client()


def test_api_backend_missing_key_optional_returns_none(monkeypatch):
    monkeypatch.delenv("AAL_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client_factory.make_client(required=False) is None


def test_cli_backend_returns_cli_client(monkeypatch):
    monkeypatch.setenv("AAL_BACKEND", "claude-cli")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # not needed for CLI
    monkeypatch.setattr(client_factory.shutil, "which", lambda _: "/usr/bin/claude")
    client = client_factory.make_client()
    assert isinstance(client, ClaudeCLIClient)


def test_cli_backend_missing_binary_exits(monkeypatch):
    monkeypatch.setenv("AAL_BACKEND", "claude-cli")
    monkeypatch.setattr(client_factory.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit):
        client_factory.make_client()


def test_cli_backend_preflight_fatal_propagates(monkeypatch):
    from pipeline.extractor import FatalAPIError

    monkeypatch.setenv("AAL_BACKEND", "claude-cli")
    monkeypatch.setattr(client_factory.shutil, "which", lambda _: "/usr/bin/claude")

    def boom(client):
        raise FatalAPIError("not logged in")

    monkeypatch.setattr(client_factory, "_preflight", boom)
    with pytest.raises(FatalAPIError):
        client_factory.make_client(preflight=True)


def test_unknown_backend_exits(monkeypatch):
    monkeypatch.setenv("AAL_BACKEND", "bedrock-ish")
    with pytest.raises(SystemExit):
        client_factory.make_client()


def test_resolve_backend_normalises(monkeypatch):
    monkeypatch.setenv("AAL_BACKEND", "  Claude-CLI ")
    assert client_factory.resolve_backend() == "claude-cli"
    monkeypatch.delenv("AAL_BACKEND", raising=False)
    assert client_factory.resolve_backend() == "api"
