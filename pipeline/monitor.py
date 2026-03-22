"""
arXiv monitoring: fetch recent papers, classify coupling type, manage processed state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
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

def _build_query(days_back: int = 3) -> str:
    """Build a broad OR query across all tracked keywords."""
    all_keywords: set[str] = set()
    for kws in ARXIV_KEYWORDS.values():
        all_keywords.update(kws)

    # arXiv search uses ti: (title) or abs: (abstract) prefixes.
    # We search abs: to capture results tables / intro mentions too.
    keyword_query = " OR ".join(f'abs:"{kw}"' for kw in sorted(all_keywords))
    cat_query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    return f"({keyword_query}) AND ({cat_query})"


def fetch_recent_papers(days_back: int = 3, max_results: int = 200) -> list[arxiv.Result]:
    """Return recent arXiv papers matching dark matter / axion / dark photon keywords."""
    query = _build_query(days_back)
    client = arxiv.Client(delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = list(client.results(search))
    logger.info("Fetched %d papers from arXiv", len(results))
    return results


def fetch_paper_by_id(arxiv_id: str) -> arxiv.Result:
    """Fetch a single paper by arXiv ID."""
    client = arxiv.Client(delay_seconds=3, num_retries=3)
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
