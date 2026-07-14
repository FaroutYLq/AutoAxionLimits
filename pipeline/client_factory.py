"""Backend selection for the model client.

One env var, ``AAL_BACKEND``, chooses the transport every pipeline entrypoint
uses:

* ``api`` (default) — the historical behaviour: an ``anthropic.Anthropic``
  client billed against ``ANTHROPIC_API_KEY``. GitHub Actions leaves the var
  unset, so CI is bit-identical.
* ``claude-cli`` — a :class:`pipeline.cli_client.ClaudeCLIClient` that shells
  out to headless ``claude -p``, billed against the user's Claude Code
  subscription (keychain OAuth). No API key required.

This lives in its own module (not ``extractor.py``) so the methodology files
show zero diff: all four construction sites already build the client
themselves and only depend on ``client.messages.create``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

_BACKEND_ENV = "AAL_BACKEND"
_CLI_BACKENDS = ("claude-cli", "cli")


def resolve_backend() -> str:
    """The selected backend id, lower-cased; ``api`` when unset."""
    return os.environ.get(_BACKEND_ENV, "api").strip().lower() or "api"


def make_client(required: bool = True, *, preflight: bool = False):
    """Construct the model client for the resolved backend.

    ``required``: when False and the backend is ``api``, a missing
    ``ANTHROPIC_API_KEY`` returns ``None`` instead of exiting (preserves
    ``backfill --discover-only``, which needs no LLM). ``preflight``: for the
    ``claude-cli`` backend, issue a 1-token ping so an unauthenticated CLI or a
    billing/usage outage aborts the run at the start (#648) rather than after
    doing work; entrypoints that already run their own preflight leave this
    False. A fatal (:class:`FatalAPIError`) ping propagates; any other ping
    failure is tolerated (never a new way for a healthy run to die).
    """
    backend = resolve_backend()

    if backend in _CLI_BACKENDS:
        binary = os.environ.get("AAL_CLI_BINARY", "claude")
        if shutil.which(binary) is None:
            logger.error(
                "AAL_BACKEND=%s but %r is not on PATH. Install Claude Code or "
                "set AAL_BACKEND=api.", backend, binary,
            )
            sys.exit(1)
        from pipeline.cli_client import ClaudeCLIClient

        client = ClaudeCLIClient(binary=binary)
        logger.info("model backend: claude-cli (%s), subscription billing", binary)
        if preflight:
            _preflight(client)
        return client

    if backend != "api":
        logger.error(
            "unknown AAL_BACKEND=%r (expected 'api' or 'claude-cli')", backend
        )
        sys.exit(1)

    # Default: API-key transport (historical behaviour, verbatim).
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        if not required:
            return None
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def _preflight(client) -> None:
    """1-token ping; :class:`FatalAPIError` aborts, anything else is tolerated.

    Mirrors ``orchestrator.preflight_api_check`` but backend-agnostic, so the
    weekly/backfill entrypoints (which have no preflight of their own) fail
    fast on a broken CLI login before touching any state.
    """
    from pipeline.extractor import CLAUDE_MODEL, FatalAPIError, _call_with_retry

    try:
        _call_with_retry(lambda: client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        ))
        logger.info("backend preflight OK")
    except FatalAPIError:
        raise
    except Exception as e:  # noqa: BLE001 — preflight must not add failure modes
        logger.warning("backend preflight inconclusive (continuing): %s", e)
