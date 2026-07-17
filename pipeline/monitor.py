"""
arXiv monitoring: fetch recent papers, classify coupling type, manage processed state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import arxiv

from .config import ARXIV_CATEGORIES, ARXIV_KEYWORDS

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).parent / "state" / "processed.json"

# arXiv fetch robustness (issue #560). The arxiv library calls
# requests.Session().get() with NO timeout internally, so a slow/throttling
# export.arxiv.org endpoint (429/503 or a stalled CDN) can hang the daily
# pipeline indefinitely. We defend on two independent axes so the wait is
# bounded *and* a hung fetch can never block process shutdown:
#
#   1. Socket timeout — inject a default per-request ``timeout`` into the
#      client's ``requests.Session`` (see ``_install_session_timeout``). The
#      blocking ``get()`` then *raises* ``requests.Timeout`` instead of hanging,
#      so the worker thread actually terminates rather than merely leaking.
#   2. Wall-clock deadline — run the (blocking) iteration on a *daemon* thread
#      joined with a timeout (see ``_call_with_deadline``). Daemon threads are
#      never joined by the interpreter at exit (unlike ThreadPoolExecutor
#      workers, which ``_python_exit`` joins unconditionally with no timeout),
#      so even a still-hung fetch can never stall process shutdown.
ARXIV_FETCH_TIMEOUT_S = 60  # per-query hard wall-clock deadline (seconds)
# Per-request socket timeout injected into the arxiv client's session. Kept
# below the wall-clock deadline so the socket-level timeout fires first and the
# worker thread dies cleanly (the wall-clock join is the backstop for the
# pathological slow-drip case the socket read-timeout does not cover).
ARXIV_SOCKET_TIMEOUT_S = 30


def _install_session_timeout(client: arxiv.Client, timeout_s: float = ARXIV_SOCKET_TIMEOUT_S) -> None:
    """Force a default per-request timeout onto the arxiv client's session.

    The arxiv library calls ``self._session.get(url, headers=...)`` with no
    ``timeout``, so ``requests`` passes ``timeout=None`` and a stalled socket
    blocks forever. We wrap ``Session.request`` to inject a default ``timeout``
    when the caller (the arxiv library) supplies none, so the underlying call
    raises ``requests.Timeout`` and the worker thread terminates instead of
    being abandoned. Idempotent and a no-op if the client exposes no
    ``_session`` (e.g. a test stub).
    """
    session = getattr(client, "_session", None)
    if session is None or getattr(session, "_axionlimits_timeout_installed", False):
        return
    _orig_request = session.request

    def _request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_s)
        return _orig_request(method, url, **kwargs)

    session.request = _request_with_timeout
    session._axionlimits_timeout_installed = True


def _call_with_deadline(fn, timeout_s: float, label: str = "arXiv query"):
    """Run blocking ``fn()`` under a hard wall-clock deadline on a daemon thread.

    Daemon threads are never joined by the interpreter at exit, so a fetch that
    is still hung when the deadline fires cannot block process shutdown (the
    failure mode a ``ThreadPoolExecutor`` worker would cause, since
    ``_python_exit`` ``join()``s non-daemon workers unconditionally). A
    ``requests`` socket timeout is normalized to ``TimeoutError`` so the
    socket-level and wall-clock deadlines surface identically to callers.
    """
    import threading

    box: dict = {}

    def _runner() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    worker = threading.Thread(target=_runner, name="arxiv-fetch", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"{label} exceeded {timeout_s}s wall-clock deadline")
    err = box.get("error")
    if err is not None:
        try:
            import requests

            socket_timeout = requests.exceptions.Timeout
        except ImportError:
            socket_timeout = ()
        if isinstance(err, socket_timeout):
            raise TimeoutError(f"{label} hit socket timeout: {err}") from err
        raise err
    return box.get("result")


def _results_with_deadline(
    client: arxiv.Client, search: arxiv.Search, timeout_s: int = ARXIV_FETCH_TIMEOUT_S
) -> list[arxiv.Result]:
    """Materialize ``client.results(search)`` under a hard wall-clock deadline.

    Combines both robustness axes (see the module-level notes): a socket
    timeout is injected into the client session so the blocking ``requests.get``
    raises rather than hanging, and the iteration runs on a daemon thread joined
    with ``timeout_s`` so a still-hung fetch can never block process exit. On
    timeout we raise ``TimeoutError`` so callers never hang.
    """
    _install_session_timeout(client)
    return _call_with_deadline(
        lambda: list(client.results(search)), timeout_s, label="arXiv query"
    )


# ---------------------------------------------------------------------------
# arXiv fetching
# ---------------------------------------------------------------------------

def _build_queries() -> list[str]:
    """Build per-coupling-group queries to keep each URL short.

    arXiv rate-limits aggressively on very long query strings.  Splitting
    into smaller batches avoids HTTP 429 / 503 failures.
    """
    cat_query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)

    # Group coupling types into batches so each query has a manageable
    # number of keywords (roughly ≤20 phrases per request).
    MAX_KEYWORDS_PER_BATCH = 12
    all_keywords_ordered: list[str] = []
    seen: set[str] = set()
    for kws in ARXIV_KEYWORDS.values():
        for kw in kws:
            if kw not in seen:
                seen.add(kw)
                all_keywords_ordered.append(kw)

    queries: list[str] = []
    for i in range(0, len(all_keywords_ordered), MAX_KEYWORDS_PER_BATCH):
        batch = all_keywords_ordered[i : i + MAX_KEYWORDS_PER_BATCH]
        keyword_query = " OR ".join(f'abs:"{kw}"' for kw in batch)
        queries.append(f"({keyword_query}) AND ({cat_query})")
    return queries


def _iter_results_with_backoff(
    client: arxiv.Client,
    search: arxiv.Search,
    max_attempts: int = 4,
) -> list[arxiv.Result]:
    """Iterate over arxiv search results with exponential backoff on HTTP 429.

    The arxiv library has built-in retries, but they may not wait long
    enough under heavy rate-limiting.  This wrapper catches the final
    HTTPError and retries the whole query with increasing delays.
    """
    for attempt in range(max_attempts):
        try:
            return _results_with_deadline(client, search)
        except (arxiv.HTTPError, TimeoutError) as exc:
            is_429 = isinstance(exc, arxiv.HTTPError) and "429" in str(exc)
            is_timeout = isinstance(exc, TimeoutError)
            if (not is_429 and not is_timeout) or attempt == max_attempts - 1:
                raise
            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
            reason = "timed out" if is_timeout else "rate-limited (HTTP 429)"
            logger.warning(
                "arXiv query %s, retrying in %ds (attempt %d/%d)",
                reason, wait, attempt + 1, max_attempts,
            )
            time.sleep(wait)
    return []  # unreachable, but satisfies type checker


def fetch_recent_papers(days_back: int = 3, max_results: int = 100) -> list[arxiv.Result]:
    """Return recent arXiv papers matching dark matter / axion / dark photon keywords.

    Splits the keyword list into smaller batches and queries arXiv
    separately for each batch to avoid HTTP 429 rate-limit errors on
    very long query strings.  Results are deduplicated by arXiv ID.
    """
    queries = _build_queries()
    client = arxiv.Client(delay_seconds=10, num_retries=8)

    seen_ids: set[str] = set()
    results: list[arxiv.Result] = []

    for idx, query in enumerate(queries):
        if idx > 0:
            # Polite pause between batch requests to avoid rate-limiting.
            time.sleep(10)
        logger.info("arXiv query batch %d/%d", idx + 1, len(queries))
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        for paper in _iter_results_with_backoff(client, search):
            pid = _arxiv_id(paper)
            if pid not in seen_ids:
                seen_ids.add(pid)
                results.append(paper)

    logger.info("Fetched %d unique papers from arXiv (%d batches)", len(results), len(queries))
    return results


def fetch_paper_by_id(arxiv_id: str) -> arxiv.Result:
    """Fetch a single paper by arXiv ID."""
    client = arxiv.Client(delay_seconds=5, num_retries=5)
    search = arxiv.Search(id_list=[arxiv_id])
    # Bounded by a hard wall-clock deadline so a stalled endpoint cannot hang.
    results = _results_with_deadline(client, search)
    if not results:
        raise ValueError(f"arXiv paper {arxiv_id} not found")
    return results[0]


# ---------------------------------------------------------------------------
# Coupling type classification (cheap local pre-filter)
# ---------------------------------------------------------------------------

def classify_coupling_type(paper: arxiv.Result) -> Optional[str]:
    """
    Return the best-matching coupling type based on keyword overlap,
    or None if no match is found.  Uses title + abstract text.
    """
    text = (paper.title + " " + paper.summary).lower()
    scores: dict[str, int] = {}
    for coupling, keywords in ARXIV_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score:
            scores[coupling] = score
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state(path: Path = STATE_PATH) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "schema_version": 1,
        "last_run": None,
        "processed_ids": [],
        "failed_ids": {},
    }


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    """Atomic write via .tmp rename."""
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


def filter_new_papers(results: list[arxiv.Result], state: dict) -> list[arxiv.Result]:
    """Return only papers not already in processed_ids or failed_ids."""
    seen = set(state.get("processed_ids", [])) | set(state.get("failed_ids", {}).keys())
    new = [r for r in results if _arxiv_id(r) not in seen]
    logger.info("%d new (unprocessed) papers after filtering", len(new))
    return new


def mark_processed(state: dict, arxiv_id: str, reason: str = "success") -> None:
    state.setdefault("processed_ids", [])
    if arxiv_id not in state["processed_ids"]:
        state["processed_ids"].append(arxiv_id)
    state.setdefault("failed_ids", {})
    state["failed_ids"].pop(arxiv_id, None)
    logger.info("Marked %s as processed (%s)", arxiv_id, reason)


def mark_failed(state: dict, arxiv_id: str, error: str) -> None:
    state.setdefault("failed_ids", {})
    state["failed_ids"][arxiv_id] = error
    logger.warning("Marked %s as failed: %s", arxiv_id, error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arxiv_id(paper: arxiv.Result) -> str:
    """Return the canonical arXiv ID (e.g. '2412.12345', 'hep-ph/0307284')."""
    # get_short_id() preserves the category prefix on old-style ids; a naive
    # entry_id.split('/')[-1] would drop 'hep-ph/' and desync from the pool.
    return re.sub(r"v\d+$", "", paper.get_short_id())
