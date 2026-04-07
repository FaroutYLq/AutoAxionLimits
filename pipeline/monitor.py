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
            return list(client.results(search))
        except arxiv.HTTPError as exc:
            if "429" not in str(exc) or attempt == max_attempts - 1:
                raise
            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
            logger.warning(
                "arXiv rate-limited (HTTP 429), retrying in %ds (attempt %d/%d)",
                wait, attempt + 1, max_attempts,
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
    results = list(client.results(search))
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
    """Return the bare arXiv ID (e.g. '2412.12345')."""
    # paper.entry_id is like 'http://arxiv.org/abs/2412.12345v2'
    return re.sub(r"v\d+$", "", paper.entry_id.split("/")[-1])
