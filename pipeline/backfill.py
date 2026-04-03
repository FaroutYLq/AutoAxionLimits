"""
Historical backfill: search INSPIRE-HEP for older papers not yet in the repo.

Usage:
  python -m pipeline.backfill --date-from 2020-01-01 --date-to 2024-12-31 [--dry-run]
  python -m pipeline.backfill --resume [--max-papers 10]
  python -m pipeline.backfill --discover-only --date-from 2020-01-01 --date-to 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import httpx

from .config import (
    ARXIV_KEYWORDS,
    BACKFILL_DEFAULT_MIN_CITATIONS,
    BACKFILL_MAX_PAPERS_PER_RUN,
    INSPIRE_SEARCH_QUERIES,
)
from .extractor import download_pdf, run_extraction_agent
from .monitor import (
    classify_coupling_type,
    fetch_paper_by_id,
    load_state as load_processed_state,
    mark_processed as mark_processed_global,
    save_state as save_processed_state,
    STATE_PATH as PROCESSED_STATE_PATH,
)
from .plot_regen import execute_notebook, execute_notebook_highlighted, get_notebook_plot_names
from .pr_creator import (
    checkout_branch,
    create_feature_branch,
    create_pull_request_preprint,
    stage_and_commit_files,
)
from .preprint_checker import scan_data_files_for_arxiv_ids
from .reviewer import ReviewResult, run_reviewer_agent, write_repo_files

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
BACKFILL_STATE_PATH = Path(__file__).parent / "state" / "backfill_state.json"

# INSPIRE-HEP API: free, no auth, designed for HEP literature.
INSPIRE_BASE_URL = "https://inspirehep.net/api/literature"
INSPIRE_DELAY = 1.0  # polite delay between requests
INSPIRE_MAX_RETRIES = 5
INSPIRE_RETRY_BASE_DELAY = 5.0
INSPIRE_PAGE_SIZE = 100  # results per page (max 250)
INSPIRE_FIELDS = "arxiv_eprints,titles,abstracts,citation_count,publication_info"


# ---------------------------------------------------------------------------
# Backfill state management
# ---------------------------------------------------------------------------

def load_backfill_state() -> dict:
    if BACKFILL_STATE_PATH.exists():
        with open(BACKFILL_STATE_PATH) as f:
            return json.load(f)
    return {
        "schema_version": 1,
        "config": None,
        "queue": [],
        "processed_ids": [],
        "skipped_ids": {},
        "runs": [],
    }


def save_backfill_state(state: dict) -> None:
    tmp = BACKFILL_STATE_PATH.with_suffix(".tmp")
    BACKFILL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(BACKFILL_STATE_PATH)


# ---------------------------------------------------------------------------
# Build set of already-known arXiv IDs (deduplication)
# ---------------------------------------------------------------------------

def build_known_ids() -> set[str]:
    """Union of all arXiv IDs already in the repo or processed by any pipeline."""
    known: set[str] = set()

    # 1. Daily pipeline processed.json
    proc = load_processed_state()
    known.update(proc.get("processed_ids", []))
    known.update(proc.get("failed_ids", {}).keys())

    # 2. Backfill state
    bf = load_backfill_state()
    known.update(bf.get("processed_ids", []))
    known.update(bf.get("skipped_ids", {}).keys())

    # 3. arXiv IDs embedded in existing data file headers
    file_map = scan_data_files_for_arxiv_ids(REPO_ROOT)
    known.update(file_map.values())

    logger.info("Known arXiv IDs (dedup set): %d", len(known))
    return known


# ---------------------------------------------------------------------------
# INSPIRE-HEP search
# ---------------------------------------------------------------------------

def _inspire_request(url: str, params: dict) -> httpx.Response:
    """Make an INSPIRE-HEP API request with retries and exponential backoff."""
    for attempt in range(INSPIRE_MAX_RETRIES):
        if attempt > 0:
            delay = INSPIRE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info("INSPIRE retry %d/%d in %.0fs", attempt + 1, INSPIRE_MAX_RETRIES, delay)
            time.sleep(delay)
        try:
            resp = httpx.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                logger.warning("INSPIRE rate limit (429); backing off")
                continue
            if resp.status_code == 503:
                logger.warning("INSPIRE unavailable (503); backing off")
                continue
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError:
            if attempt == INSPIRE_MAX_RETRIES - 1:
                raise
        except httpx.RequestError as e:
            if attempt == INSPIRE_MAX_RETRIES - 1:
                raise
            logger.warning("INSPIRE request error: %s", e)
    raise RuntimeError("INSPIRE request failed after retries")


def _build_inspire_query(query_phrase: str, date_from: str, date_to: str, min_citations: int) -> str:
    """Build INSPIRE SPIRES-style query string.

    Searches both title and abstract for the query phrase.

    Example output:
      find (t "dark photon" or abs "dark photon") and de >= 2023-06-01 and de <= 2023-06-30 and topcite 20+
    """
    # Split multi-term query phrases into individual search terms
    # e.g. "dark photon exclusion limit" -> search for "dark photon" in title OR abstract
    # Use the full phrase for relevance
    parts = [f'find (t "{query_phrase}" or abs "{query_phrase}")']
    parts.append(f"and de >= {date_from}")
    parts.append(f"and de <= {date_to}")
    if min_citations > 0:
        parts.append(f"and topcite {min_citations}+")
    return " ".join(parts)


def _search_inspire(query: str, date_from: str, date_to: str, min_citations: int) -> list[dict]:
    """Paginated INSPIRE-HEP search. Returns list of hit metadata dicts."""
    inspire_q = _build_inspire_query(query, date_from, date_to, min_citations)
    papers: list[dict] = []
    page = 1

    while True:
        time.sleep(INSPIRE_DELAY)
        resp = _inspire_request(
            INSPIRE_BASE_URL,
            params={
                "q": inspire_q,
                "sort": "mostcited",
                "size": INSPIRE_PAGE_SIZE,
                "page": page,
                "fields": INSPIRE_FIELDS,
            },
        )
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        papers.extend(hits)
        total = data.get("hits", {}).get("total", 0)
        logger.info("INSPIRE search '%s': fetched %d/%d", query[:50], len(papers), total)

        if len(papers) >= total or len(papers) >= 1000:
            break
        page += 1

    return papers


def discover_candidates(
    date_from: str,
    date_to: str,
    min_citations: int,
    coupling_types: list[str] | None = None,
) -> list[dict]:
    """
    Search INSPIRE-HEP for relevant papers, deduplicate, return candidates.

    Each candidate dict has: arxiv_id, title, abstract, citations, coupling_guess,
    inspire_id.
    """
    target_types = coupling_types or list(INSPIRE_SEARCH_QUERIES.keys())

    seen_arxiv_ids: set[str] = set()
    candidates: list[dict] = []

    for ct in target_types:
        queries = INSPIRE_SEARCH_QUERIES.get(ct, [])
        if not queries:
            logger.warning("No search queries for coupling type %s; skipping", ct)
            continue

        for query in queries:
            logger.info(
                "Searching INSPIRE: coupling=%s query='%s' dates=%s..%s min_cite=%d",
                ct, query, date_from, date_to, min_citations,
            )
            try:
                hits = _search_inspire(query, date_from, date_to, min_citations)
            except Exception as e:
                logger.warning("INSPIRE search failed for query '%s': %s; skipping", query, e)
                continue

            for hit in hits:
                meta = hit.get("metadata", {})

                # Extract arXiv ID
                eprints = meta.get("arxiv_eprints", [])
                if not eprints:
                    continue
                arxiv_id = eprints[0].get("value", "")
                if not arxiv_id or arxiv_id in seen_arxiv_ids:
                    continue
                seen_arxiv_ids.add(arxiv_id)

                # Extract title
                titles = meta.get("titles", [])
                title = titles[0].get("title", "") if titles else ""

                # Extract abstract
                abstracts = meta.get("abstracts", [])
                abstract = abstracts[0].get("value", "") if abstracts else ""

                citations = meta.get("citation_count", 0)
                inspire_id = str(meta.get("control_number", ""))

                candidates.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "citations": citations,
                    "coupling_guess": None,  # filled by filter step
                    "inspire_id": inspire_id,
                    "publication_types": [],  # INSPIRE doesn't separate pub types the same way
                })

    logger.info("INSPIRE discovery: %d unique candidates with arXiv IDs", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Pre-filtering
# ---------------------------------------------------------------------------

def filter_candidates(candidates: list[dict], known_ids: set[str]) -> list[dict]:
    """Apply local filters: dedup and keyword classification."""
    filtered = []

    for c in candidates:
        arxiv_id = c["arxiv_id"]

        # Filter 1: duplicate detection
        if arxiv_id in known_ids:
            logger.debug("Skip %s: already known", arxiv_id)
            continue

        # Filter 2: keyword classification on title + abstract
        text = f"{c.get('title', '')} {c.get('abstract', '')}"
        coupling_guess = _classify_text(text)
        if coupling_guess is None:
            logger.debug("Skip %s: no keyword match", arxiv_id)
            continue

        c["coupling_guess"] = coupling_guess
        filtered.append(c)

    logger.info("After local filters: %d candidates (from %d)", len(filtered), len(candidates))
    return filtered


def _classify_text(text: str) -> Optional[str]:
    """Keyword classification on raw text (same logic as monitor.classify_coupling_type)."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for coupling, keywords in ARXIV_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score:
            scores[coupling] = score
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


def batch_relevance_check(
    candidates: list[dict],
    client: anthropic.Anthropic,
    batch_size: int = 20,
) -> list[dict]:
    """
    Use Claude Haiku to batch-filter candidates by relevance.

    Sends title+abstract batches and asks which papers present NEW experimental
    exclusion limits (not reviews, not theory-only, not phenomenology).
    """
    if not candidates:
        return []

    from .extractor import _call_with_retry, CLAUDE_MODEL

    kept: list[dict] = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        papers_text = ""
        for idx, c in enumerate(batch):
            papers_text += (
                f"[{idx}] {c['title']}\n"
                f"    Abstract: {(c.get('abstract') or 'N/A')[:500]}\n\n"
            )

        prompt = (
            "You are a physics literature classifier. Below is a list of papers. "
            "For each paper, determine whether it presents NEW experimental exclusion "
            "limits or constraints on axions, dark photons, ALPs, scalar dark matter, "
            "or similar ultralight bosons.\n\n"
            "INCLUDE papers that:\n"
            "- Present new experimental or observational upper/lower limits\n"
            "- Report new constraints from data analysis\n"
            "- Present projected sensitivities from planned experiments\n\n"
            "EXCLUDE papers that:\n"
            "- Are purely theoretical (no new data or limits)\n"
            "- Are review articles summarising existing limits\n"
            "- Are phenomenology studies without new experimental constraints\n"
            "- Only discuss detection methods without presenting limits\n\n"
            f"Papers:\n{papers_text}\n"
            "Return ONLY a JSON array of the integer indices of papers to INCLUDE. "
            "Example: [0, 3, 7]\n"
            "If none qualify, return: []"
        )

        def _call():
            return client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

        try:
            response = _call_with_retry(_call)
            text = response.content[0].text.strip()
            # Parse the JSON array from the response
            # Handle cases where Claude wraps in markdown code blocks or adds text
            import re as _re
            match = _re.search(r"\[[\d\s,]*\]", text)
            if match:
                indices = json.loads(match.group())
            else:
                indices = json.loads(text)
            if not isinstance(indices, list):
                indices = []
            indices = [idx for idx in indices if isinstance(idx, int) and 0 <= idx < len(batch)]
        except Exception as e:
            logger.warning("LLM relevance check failed for batch %d: %s; keeping all", i, e)
            indices = list(range(len(batch)))

        for idx in indices:
            kept.append(batch[idx])

        logger.info(
            "LLM relevance filter batch %d-%d: kept %d/%d",
            i, i + len(batch), len(indices), len(batch),
        )

    logger.info("After LLM relevance filter: %d candidates (from %d)", len(kept), len(candidates))
    return kept


# ---------------------------------------------------------------------------
# Process a single candidate through extraction → review → PR
# ---------------------------------------------------------------------------

def _process_candidate(
    candidate: dict,
    client: anthropic.Anthropic,
    backfill_state: dict,
    processed_state: dict,
    dry_run: bool,
) -> bool:
    """
    Run the full extraction pipeline on one candidate.
    Returns True if a PR was created.
    """
    arxiv_id = candidate["arxiv_id"]
    logger.info(
        "Processing %s (citations=%d, coupling_guess=%s)",
        arxiv_id, candidate.get("citations", 0), candidate.get("coupling_guess"),
    )

    try:
        paper = fetch_paper_by_id(arxiv_id)
    except Exception as e:
        logger.warning("Could not fetch %s from arXiv: %s", arxiv_id, e)
        backfill_state.setdefault("skipped_ids", {})[arxiv_id] = f"arxiv_fetch_error: {e}"
        return False

    # Re-check keyword classification with full arXiv metadata
    coupling_guess = classify_coupling_type(paper)
    if coupling_guess is None:
        logger.info("%s: no keyword match on full metadata; skipping", arxiv_id)
        backfill_state.setdefault("skipped_ids", {})[arxiv_id] = "no_keyword_match"
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pdf_path = download_pdf(arxiv_id, Path(tmpdir))
            extraction = run_extraction_agent(paper, pdf_path, client)
        except Exception as e:
            logger.warning("Extraction failed for %s: %s", arxiv_id, e)
            backfill_state.setdefault("skipped_ids", {})[arxiv_id] = f"extraction_error: {e}"
            return False

    if not extraction.is_new_limit:
        logger.info("%s: not a new limit", arxiv_id)
        backfill_state.setdefault("skipped_ids", {})[arxiv_id] = "not_new_limit"
        return False

    if not extraction.data_points:
        logger.info("%s: no data points extracted", arxiv_id)
        backfill_state.setdefault("skipped_ids", {})[arxiv_id] = "no_data_points"
        return False

    if extraction.coupling_type is None:
        logger.info("%s: coupling type undetermined", arxiv_id)
        backfill_state.setdefault("skipped_ids", {})[arxiv_id] = "no_coupling_type"
        return False

    logger.info(
        "%s: %s limit, %d points, conf=%.2f",
        arxiv_id, extraction.coupling_type,
        len(extraction.data_points), extraction.extraction_confidence,
    )

    review = run_reviewer_agent(extraction, client)

    if dry_run:
        logger.info(
            "[DRY RUN] Would create PR: %s %s (conf=%.2f, citations=%d)",
            review.experiment_name, extraction.coupling_type,
            review.extraction_confidence, candidate.get("citations", 0),
        )
        return False

    # Write repo files
    write_repo_files(review, REPO_ROOT)

    # Regenerate plot(s)
    nb_ok, nb_err = execute_notebook(review.notebook_path, REPO_ROOT)
    if not nb_ok:
        logger.warning("Notebook execution failed: %s", nb_err[-500:])

    hl_ok, hl_err, highlight_files = execute_notebook_highlighted(
        review.notebook_path, review.notebook_call, REPO_ROOT,
        data_file_path=review.data_file_path,
    )
    if not hl_ok:
        logger.warning("Highlighted plot generation failed: %s", hl_err[-500:])

    # Git branch, commit, PR
    branch = create_feature_branch(arxiv_id, review.experiment_name, REPO_ROOT)

    # Mark in both states before committing
    backfill_state.setdefault("processed_ids", []).append(arxiv_id)
    save_backfill_state(backfill_state)
    mark_processed_global(processed_state, arxiv_id)
    save_processed_state(processed_state)

    changed_files = [
        str(BACKFILL_STATE_PATH.relative_to(REPO_ROOT)),
        str(PROCESSED_STATE_PATH.relative_to(REPO_ROOT)),
        review.data_file_path,
        review.plotfuncs_file,
        review.notebook_path,
        review.docs_file,
    ]
    plot_names = get_notebook_plot_names(review.notebook_path, REPO_ROOT)
    for name in plot_names:
        for p in [f"plots/{name}.pdf", f"plots/plots_png/{name}.png"]:
            if (REPO_ROOT / p).exists():
                changed_files.append(p)
    changed_files.extend(highlight_files)

    commit_msg = (
        f"Add {review.experiment_name} {extraction.coupling_type} limit\n\n"
        f"Source: arXiv:{arxiv_id}\n"
        f"Citations: {candidate.get('citations', 'N/A')}\n"
        f"Extraction confidence: {extraction.extraction_confidence:.2f}\n"
        f"Data points: {len(extraction.data_points)}\n"
        f"Auto-generated by AutoAxionLimits backfill pipeline\n"
    )

    # Build PR body
    coupling = extraction.coupling_type or "Unknown"
    prefix = "[BACKFILL] "
    if review.low_confidence:
        prefix = "[BACKFILL] [LOW CONFIDENCE] "
    elif review.is_projection:
        prefix = "[BACKFILL] [PROJECTION] "

    title = f"{prefix}Add {review.experiment_name} {coupling} limit (arXiv:{arxiv_id})"

    if extraction.data_points:
        masses = [m for m, _ in extraction.data_points]
        couplings = [g for _, g in extraction.data_points]
        range_summary = (
            f"- Mass range: {min(masses):.2e} -- {max(masses):.2e} eV\n"
            f"- Coupling range: {min(couplings):.2e} -- {max(couplings):.2e}\n"
        )
    else:
        range_summary = "- No data points extracted\n"

    corrections_md = "\n".join(f"- {c}" for c in review.corrections_applied) or "- None"
    flagged_md = "\n".join(f"- {c}" for c in review.corrections_flagged) or "- None"

    # Plot URL
    plot_names_list = get_notebook_plot_names(review.notebook_path, REPO_ROOT)
    plot_stem = plot_names_list[0] if plot_names_list else coupling
    plot_png = (
        f"https://raw.githubusercontent.com/FaroutYLq/AutoAxionLimits/"
        f"{branch}/plots/plots_png/{plot_stem}.png"
    )

    highlight_png_files = [f for f in highlight_files if f.endswith(".png")]
    if highlight_png_files:
        hl_stem = Path(highlight_png_files[0]).name
        hl_png = (
            f"https://raw.githubusercontent.com/FaroutYLq/AutoAxionLimits/"
            f"{branch}/plots/plots_png/{hl_stem}"
        )
        plot_section = (
            f"## Highlighted Plot (new limit in colour)\n\n"
            f"![{review.experiment_name} highlighted]({hl_png})\n\n"
            f"<details><summary>Full plot with all colours</summary>\n\n"
            f"![{coupling} limits]({plot_png})\n\n"
            f"</details>\n\n"
        )
    else:
        plot_section = f"## Plot\n\n![{coupling} limits]({plot_png})\n\n"

    body = (
        f"## Historical Backfill: {review.experiment_name}\n\n"
        f"**Paper:** [{review.paper_title}]({review.arxiv_url})\n"
        f"**Citations:** {candidate.get('citations', 'N/A')}\n"
        f"**Data source:** {extraction.data_source}\n"
        f"**Extraction confidence:** {extraction.extraction_confidence:.0%}\n\n"
        f"## Data Summary\n\n{range_summary}\n"
        f"## Physical Corrections Applied\n\n{corrections_md}\n\n"
        f"## Corrections Flagged for Human Review\n\n{flagged_md}\n\n"
        f"## Files Changed\n\n"
        f"- `{review.data_file_path}`\n"
        f"- `{review.plotfuncs_file}` (new method `{review.plotfuncs_class}.{review.experiment_name}`)\n"
        f"- `{review.notebook_path}`\n"
        f"- `{review.docs_file}`\n\n"
        f"{plot_section}"
        f"---\n"
        f"> Discovered via historical backfill (INSPIRE-HEP search). "
        f"Please verify extraction accuracy before merging.\n\n"
        f"Generated by AutoAxionLimits backfill pipeline"
    )

    try:
        stage_and_commit_files(changed_files, commit_msg, REPO_ROOT)
        pr_url = create_pull_request_preprint(branch, title, body, REPO_ROOT)
        logger.info("PR created: %s", pr_url)
        return True
    except Exception as e:
        logger.error("PR creation failed for %s: %s", arxiv_id, e)
        return False
    finally:
        checkout_branch("master", REPO_ROOT)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(
    date_from: str | None = None,
    date_to: str | None = None,
    min_citations: int = BACKFILL_DEFAULT_MIN_CITATIONS,
    max_papers: int = BACKFILL_MAX_PAPERS_PER_RUN,
    coupling_types: list[str] | None = None,
    dry_run: bool = False,
    discover_only: bool = False,
    resume: bool = False,
) -> int:
    """
    Run the backfill pipeline. Returns the number of queue items remaining.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not discover_only:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key) if api_key else None
    state = load_backfill_state()

    # --- Discovery phase ---
    if not resume:
        if not date_from or not date_to:
            logger.error("--date-from and --date-to are required (unless --resume)")
            sys.exit(1)

        # Save config for resume runs
        state["config"] = {
            "date_from": date_from,
            "date_to": date_to,
            "min_citations": min_citations,
            "coupling_types": coupling_types,
        }

        logger.info(
            "Starting discovery: %s to %s, min_citations=%d",
            date_from, date_to, min_citations,
        )

        # Step 1: Search INSPIRE-HEP
        raw_candidates = discover_candidates(date_from, date_to, min_citations, coupling_types)

        # Step 2: Local pre-filtering
        known_ids = build_known_ids()
        filtered = filter_candidates(raw_candidates, known_ids)

        # Log candidates before LLM filter for transparency
        logger.info("Candidates before LLM filter (%d):", len(filtered))
        for i, c in enumerate(sorted(filtered, key=lambda x: x.get("citations", 0), reverse=True)):
            logger.info(
                "  [%d] %s (citations=%d, coupling=%s) %s",
                i, c["arxiv_id"], c.get("citations", 0),
                c.get("coupling_guess", "?"), c.get("title", "")[:80],
            )

        # Step 3: LLM relevance check
        if client and filtered:
            filtered = batch_relevance_check(filtered, client)

        # Sort by citations (most cited first — higher quality papers processed first)
        filtered.sort(key=lambda c: c.get("citations", 0), reverse=True)

        # Save queue
        state["queue"] = filtered
        save_backfill_state(state)

        logger.info(
            "Discovery complete: %d candidates queued (sorted by citation count)",
            len(filtered),
        )

        if discover_only:
            # Log the candidates for review
            for i, c in enumerate(filtered[:50]):
                logger.info(
                    "  [%d] %s (citations=%d, coupling=%s) %s",
                    i, c["arxiv_id"], c.get("citations", 0),
                    c.get("coupling_guess", "?"), c.get("title", "")[:80],
                )
            if len(filtered) > 50:
                logger.info("  ... and %d more", len(filtered) - 50)
            return len(filtered)

    # --- Processing phase ---
    queue = state.get("queue", [])
    if not queue:
        logger.info("Queue is empty; nothing to process")
        return 0

    processed_state = load_processed_state()
    prs_created = 0
    papers_processed = 0

    while queue and papers_processed < max_papers:
        candidate = queue.pop(0)
        arxiv_id = candidate["arxiv_id"]

        # Double-check not already processed (could happen between discover and resume)
        if arxiv_id in set(state.get("processed_ids", [])) | set(state.get("skipped_ids", {}).keys()):
            logger.info("Skip %s: already handled in backfill state", arxiv_id)
            continue

        created = _process_candidate(candidate, client, state, processed_state, dry_run)
        if created:
            prs_created += 1
        papers_processed += 1

        # Save state after each paper (in case of crash)
        state["queue"] = queue
        save_backfill_state(state)

    # Record this run
    state.setdefault("runs", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processed": papers_processed,
        "prs_created": prs_created,
        "remaining": len(queue),
    })
    state["queue"] = queue
    save_backfill_state(state)

    logger.info(
        "Backfill run complete: processed=%d, PRs=%d, remaining=%d",
        papers_processed, prs_created, len(queue),
    )
    return len(queue)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="AutoAxionLimits historical backfill")
    parser.add_argument("--date-from", help="Start date YYYY-MM-DD")
    parser.add_argument("--date-to", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--min-citations", type=int, default=BACKFILL_DEFAULT_MIN_CITATIONS,
        help=f"Minimum citation count (default {BACKFILL_DEFAULT_MIN_CITATIONS})",
    )
    parser.add_argument(
        "--max-papers", type=int, default=BACKFILL_MAX_PAPERS_PER_RUN,
        help=f"Max papers to process this run (default {BACKFILL_MAX_PAPERS_PER_RUN})",
    )
    parser.add_argument(
        "--coupling-types",
        help="Comma-separated coupling types to search (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    parser.add_argument("--discover-only", action="store_true", help="Search and filter only; save queue but don't extract")
    parser.add_argument("--resume", action="store_true", help="Skip discovery; process existing queue")
    args = parser.parse_args()

    ct = args.coupling_types.split(",") if args.coupling_types else None

    remaining = main(
        date_from=args.date_from,
        date_to=args.date_to,
        min_citations=args.min_citations,
        max_papers=args.max_papers,
        coupling_types=ct,
        dry_run=args.dry_run,
        discover_only=args.discover_only,
        resume=args.resume,
    )

    # Write remaining count to a file for GitHub Actions to read
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"queue_remaining={remaining}\n")


if __name__ == "__main__":
    _cli()
