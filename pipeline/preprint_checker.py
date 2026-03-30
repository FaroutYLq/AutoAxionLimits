"""
Weekly preprint update checker.

Scans all limit_data/**/*.txt files for arXiv IDs in header comments,
checks for newer arXiv versions with changed data, and opens PRs.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import arxiv

from .extractor import (
    ExtractionResult,
    download_pdf,
    run_extraction_agent,
)
from .pr_creator import create_feature_branch, stage_and_commit_files
from .reviewer import ReviewResult, apply_corrections, format_data_file

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
STATE_PATH = Path(__file__).parent / "state" / "preprint_versions.json"

# Regex patterns to match arXiv URLs in comment lines
_ARXIV_URL_PATTERNS = [
    re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?"),
    re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?"),
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_version_state(path: Path = STATE_PATH) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"schema_version": 1, "last_checked": None, "files": {}}


def save_version_state(state: dict, path: Path = STATE_PATH) -> None:
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Scanning data files
# ---------------------------------------------------------------------------

def scan_data_files_for_arxiv_ids(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """
    Walk limit_data/**/*.txt, read first 10 comment lines,
    extract arXiv IDs from URL patterns.
    Returns {relative_file_path: arxiv_id}.
    """
    results: dict[str, str] = {}
    data_root = repo_root / "limit_data"

    for txt_file in sorted(data_root.rglob("*.txt")):
        try:
            lines = txt_file.read_text(errors="replace").splitlines()
        except OSError:
            continue

        for line in lines[:10]:
            if not line.startswith("#"):
                break
            for pattern in _ARXIV_URL_PATTERNS:
                m = pattern.search(line)
                if m:
                    arxiv_id = m.group(1)
                    rel_path = str(txt_file.relative_to(repo_root))
                    results[rel_path] = arxiv_id
                    break
            else:
                continue
            break  # found an ID for this file

    logger.info(
        "Scanned %d data files; found arXiv IDs in %d", len(list(data_root.rglob("*.txt"))), len(results)
    )
    return results


# ---------------------------------------------------------------------------
# arXiv version checking
# ---------------------------------------------------------------------------

def get_latest_version(arxiv_id: str) -> tuple[int, bool, arxiv.Result]:
    """
    Query arXiv for the latest version number and whether the paper is published.
    Returns (latest_version_number, is_published, paper).
    """
    client_arxiv = arxiv.Client(delay_seconds=3, num_retries=3)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client_arxiv.results(search))
    if not results:
        raise ValueError(f"arXiv paper {arxiv_id} not found")

    paper = results[0]
    version = _parse_paper_version(paper)
    published = is_published(paper)
    return version, published, paper


def _parse_paper_version(paper: arxiv.Result) -> int:
    """Extract version number from an arxiv.Result."""
    version_match = re.search(r"v(\d+)$", paper.entry_id)
    return int(version_match.group(1)) if version_match else 1


def _is_published_fast(paper: arxiv.Result) -> bool:
    """Check published status using only arXiv metadata (no external API calls)."""
    if paper.journal_ref and paper.journal_ref.strip():
        return True
    if paper.doi and paper.doi.strip():
        return True
    if paper.comment and re.search(
        r"published|accepted|in press|to appear", paper.comment, re.IGNORECASE
    ):
        return True
    return False


def batch_get_latest_versions(
    arxiv_ids: list[str],
    batch_size: int = 20,
) -> dict[str, tuple[int, bool, arxiv.Result]]:
    """
    Fetch version info for many arXiv IDs using batched API calls.
    Returns {arxiv_id: (version, is_published, paper)} for each found paper.
    Papers that could not be fetched are omitted from the result.

    Uses httpx directly with exponential backoff to handle arXiv rate limits
    more gracefully than the arxiv library's built-in retry logic.
    """
    import time
    import xml.etree.ElementTree as ET

    import httpx

    results: dict[str, tuple[int, bool, arxiv.Result]] = {}
    unique_ids = list(dict.fromkeys(arxiv_ids))
    base_url = "https://export.arxiv.org/api/query"

    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(unique_ids) + batch_size - 1) // batch_size

        # Retry with exponential backoff
        fetched = False
        for attempt in range(6):
            if attempt > 0:
                wait = min(10 * (2 ** (attempt - 1)), 120)  # 10, 20, 40, 80, 120s
                logger.info(
                    "Batch %d/%d: retry %d, waiting %ds...",
                    batch_num, total_batches, attempt, wait,
                )
                time.sleep(wait)

            try:
                resp = httpx.get(
                    base_url,
                    params={
                        "id_list": ",".join(batch),
                        "max_results": str(len(batch)),
                    },
                    timeout=30,
                )
                if resp.status_code == 429:
                    logger.warning(
                        "Batch %d/%d: HTTP 429 (attempt %d/6)",
                        batch_num, total_batches, attempt + 1,
                    )
                    continue
                resp.raise_for_status()
            except Exception as e:
                logger.warning(
                    "Batch %d/%d: request failed (attempt %d/6): %s",
                    batch_num, total_batches, attempt + 1, e,
                )
                continue

            # Parse the Atom XML feed
            try:
                papers = _parse_arxiv_feed(resp.text)
            except Exception as e:
                logger.warning("Batch %d/%d: XML parse failed: %s", batch_num, total_batches, e)
                break

            for paper in papers:
                short_id = paper.get_short_id().split("v")[0]
                version = _parse_paper_version(paper)
                # Fast check using only arXiv metadata (no Semantic Scholar)
                published = _is_published_fast(paper)
                results[short_id] = (version, published, paper)

            logger.info(
                "Batch %d/%d: fetched %d papers", batch_num, total_batches, len(papers),
            )
            fetched = True
            break

        if not fetched:
            logger.warning(
                "Batch %d/%d: all retries exhausted for IDs %d–%d",
                batch_num, total_batches, i, i + len(batch),
            )

        # Polite delay between batches — arXiv needs time to reset rate limits
        if i + batch_size < len(unique_ids):
            time.sleep(30)

    logger.info(
        "Batch-fetched %d/%d arXiv papers in %d batches",
        len(results), len(unique_ids), total_batches,
    )
    return results


# Atom XML namespace used by arXiv API
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


def _parse_arxiv_feed(xml_text: str) -> list[arxiv.Result]:
    """Parse arXiv Atom XML feed into arxiv.Result objects."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    papers: list[arxiv.Result] = []

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        entry_id = entry.findtext(f"{{{_ATOM_NS}}}id", "")
        # Skip error entries (arXiv returns these for invalid IDs)
        if not entry_id or "arxiv.org" not in entry_id:
            continue

        title = entry.findtext(f"{{{_ATOM_NS}}}title", "").strip()
        summary = entry.findtext(f"{{{_ATOM_NS}}}summary", "").strip()
        comment = entry.findtext(f"{{{_ARXIV_NS}}}comment", "") or ""
        journal_ref = entry.findtext(f"{{{_ARXIV_NS}}}journal_ref", "") or ""
        doi_el = entry.findtext(f"{{{_ARXIV_NS}}}doi", "") or ""

        paper = arxiv.Result(
            entry_id=entry_id,
            title=title,
            summary=summary,
            comment=comment.strip(),
            journal_ref=journal_ref.strip(),
            doi=doi_el.strip(),
        )
        papers.append(paper)

    return papers


def is_published(paper: arxiv.Result) -> bool:
    """Return True if the paper has been published in a journal.

    Checks arXiv metadata first (journal_ref, DOI, comment keywords),
    then falls back to Semantic Scholar for papers where authors did not
    update the arXiv record after publication.
    """
    if paper.journal_ref and paper.journal_ref.strip():
        return True
    if paper.doi and paper.doi.strip():
        return True
    if paper.comment and re.search(
        r"published|accepted|in press|to appear", paper.comment, re.IGNORECASE
    ):
        return True

    # Fallback: query Semantic Scholar (free, no auth required)
    arxiv_id = paper.get_short_id().split("v")[0]
    return _check_semantic_scholar(arxiv_id)


def _check_semantic_scholar(arxiv_id: str) -> bool:
    """Query Semantic Scholar to check if a paper is a journal article."""
    import httpx

    try:
        resp = httpx.get(
            f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}",
            params={"fields": "publicationTypes"},
            timeout=10,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        pub_types = data.get("publicationTypes") or []
        return "JournalArticle" in pub_types
    except Exception:
        return False


def batch_check_published_semantic_scholar(
    arxiv_ids: list[str],
    batch_size: int = 500,
) -> dict[str, bool]:
    """
    Batch-check published status via Semantic Scholar.
    Returns {arxiv_id: is_published} for papers found.
    Uses POST /paper/batch endpoint (up to 500 IDs per request).
    """
    import time

    import httpx

    results: dict[str, bool] = {}
    unique_ids = list(dict.fromkeys(arxiv_ids))

    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i : i + batch_size]
        s2_ids = [f"ArXiv:{aid}" for aid in batch]

        for attempt in range(3):
            if attempt > 0:
                time.sleep(5 * attempt)
            try:
                resp = httpx.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "externalIds,publicationTypes"},
                    json={"ids": s2_ids},
                    timeout=30,
                )
                if resp.status_code == 429:
                    logger.warning("Semantic Scholar batch: HTTP 429 (attempt %d/3)", attempt + 1)
                    continue
                resp.raise_for_status()
            except Exception as e:
                logger.warning("Semantic Scholar batch failed (attempt %d/3): %s", attempt + 1, e)
                continue

            data = resp.json()
            for paper_data in data:
                if paper_data is None:
                    continue
                aid = (paper_data.get("externalIds") or {}).get("ArXiv")
                if not aid:
                    continue
                pub_types = paper_data.get("publicationTypes") or []
                results[aid] = "JournalArticle" in pub_types
            logger.info(
                "Semantic Scholar batch: checked %d/%d papers",
                len(results), len(unique_ids),
            )
            break

        if i + batch_size < len(unique_ids):
            time.sleep(3)

    return results


# ---------------------------------------------------------------------------
# Data comparison
# ---------------------------------------------------------------------------

def data_has_changed(
    old_file_path: Path,
    new_data_points: list[tuple[float, float]],
    tolerance: float = 1e-6,
) -> bool:
    """
    Compare new_data_points to existing 2-column data file.
    Returns True if the number of points differs or any value differs beyond tolerance.
    """
    try:
        import numpy as np
        old_data = np.loadtxt(str(old_file_path))
    except Exception:
        # If we can't read the old file, assume changed
        return True

    if old_data.ndim == 1:
        old_data = old_data.reshape(1, -1)

    # Sort both arrays by mass (column 0) before comparing row-by-row
    old_data = old_data[old_data[:, 0].argsort()]
    new_array = sorted(new_data_points, key=lambda x: x[0])

    if len(old_data) != len(new_array):
        return True

    for i, (m, g) in enumerate(new_array):
        om, og = old_data[i, 0], old_data[i, 1]
        if abs(m - om) / max(abs(om), 1e-30) > tolerance:
            return True
        if abs(g - og) / max(abs(og), 1e-30) > tolerance:
            return True

    return False


# ---------------------------------------------------------------------------
# What changed — Claude summary
# ---------------------------------------------------------------------------

def summarise_changes(
    arxiv_id: str,
    old_version: int,
    new_version: int,
    new_paper: arxiv.Result,
    client: anthropic.Anthropic,
) -> str:
    """Ask Claude to summarise what changed between versions."""
    prompt = (
        f"arXiv paper {arxiv_id} has been updated from v{old_version} to v{new_version}.\n"
        f"New title: {new_paper.title}\n"
        f"New abstract: {new_paper.summary[:1500]}\n\n"
        "In 2-3 sentences, summarise what likely changed between these versions "
        "(corrections, updated data, new analysis, etc.)."
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"(Could not generate summary: {e})"


# ---------------------------------------------------------------------------
# Main weekly check
# ---------------------------------------------------------------------------

def run_weekly_check(
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
    init_only: bool = False,
) -> None:
    """
    1. Scan data files for arXiv IDs.
    2. Load version state.
    3. For each file:
       a. Check latest version on arXiv.
       b. If newer version: extract, compare, open PR if changed.
       c. If published + no data: flag for review.
    4. Save state.
    """
    api_key = __import__("os").environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    state = load_version_state()
    file_arxiv_map = scan_data_files_for_arxiv_ids(repo_root)

    now_iso = datetime.now(timezone.utc).isoformat()

    # Filter to papers that need checking
    ids_to_check: list[str] = []
    for file_path, arxiv_id in file_arxiv_map.items():
        file_state = state["files"].get(file_path, {})
        if file_state.get("published") and not init_only:
            logger.debug("Skipping %s (%s): already published", file_path, arxiv_id)
            continue
        ids_to_check.append(arxiv_id)

    # Step 1: Batch-check published status via Semantic Scholar (fast, 1-2 requests)
    s2_published = batch_check_published_semantic_scholar(ids_to_check)
    n_published = sum(1 for v in s2_published.values() if v)
    logger.info(
        "Semantic Scholar: %d/%d papers are published journal articles",
        n_published, len(s2_published),
    )

    # Step 2: Papers already tracked AND published can skip arXiv (they won't
    # change). All other papers — including NEW files whose paper is published —
    # need arXiv so we can extract and verify the data is still valid.
    ids_need_arxiv: list[str] = []
    for file_path, arxiv_id in file_arxiv_map.items():
        file_state = state["files"].get(file_path, {})
        if file_state.get("published") and not init_only:
            continue
        known_version = file_state.get("known_version")
        s2_is_published = s2_published.get(arxiv_id, False)

        # Already tracked + published via S2 → mark and skip arXiv
        if known_version is not None and s2_is_published:
            state["files"][file_path]["published"] = True
            state["files"][file_path]["last_checked"] = now_iso
            continue

        if arxiv_id not in ids_need_arxiv:
            ids_need_arxiv.append(arxiv_id)

    logger.info(
        "Need arXiv version check for %d papers (skipped %d published)",
        len(ids_need_arxiv), len(ids_to_check) - len(ids_need_arxiv),
    )

    # Step 3: Batch-fetch from arXiv (only unpublished papers)
    arxiv_cache = batch_get_latest_versions(ids_need_arxiv)

    for file_path, arxiv_id in file_arxiv_map.items():
        file_state = state["files"].get(file_path, {})
        known_version = file_state.get("known_version")

        # Skip papers already known to be published — they won't be updated.
        if file_state.get("published") and not init_only:
            continue

        cached = arxiv_cache.get(arxiv_id)
        if cached is None:
            logger.warning("Could not check %s (%s): not in batch results", file_path, arxiv_id)
            continue
        latest_version, published_fast, new_paper = cached
        # Merge: if Semantic Scholar says published, trust it
        published = published_fast or s2_published.get(arxiv_id, False)

        if init_only:
            # Just record the current state, no PRs
            state["files"][file_path] = {
                "arxiv_id": arxiv_id,
                "known_version": latest_version,
                "last_checked": now_iso,
                "published": published,
            }
            continue

        # First time seeing this file
        if known_version is None:
            if not published:
                # Unpublished preprint — set baseline, no PR
                state["files"][file_path] = {
                    "arxiv_id": arxiv_id,
                    "known_version": latest_version,
                    "last_checked": now_iso,
                    "published": False,
                }
                continue
            # Published paper with existing data file — extract to verify
            # the published version still provides this limit. Only flag
            # if extraction yields NO data (possible limit removal).
            logger.info(
                "%s: new file %s is already published — verifying data",
                arxiv_id, file_path,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    pdf_path = download_pdf(arxiv_id, Path(tmpdir))
                    verification = run_extraction_agent(new_paper, pdf_path, client)
                except Exception as e:
                    logger.warning("Verification failed for %s: %s — baselining", arxiv_id, e)
                    verification = None

            if verification is not None and not verification.data_points:
                # Published version has no data — flag for review
                logger.warning(
                    "Published paper %s yielded no data — possible limit removal for %s",
                    arxiv_id, file_path,
                )
                state["files"][file_path] = {
                    "arxiv_id": arxiv_id,
                    "known_version": latest_version,
                    "last_checked": now_iso,
                    "published": True,
                }
                state["last_checked"] = now_iso
                save_version_state(state)
                if not dry_run:
                    _create_removal_flag_pr(
                        repo_root=repo_root,
                        file_path=file_path,
                        arxiv_id=arxiv_id,
                        old_version=0,
                        new_version=latest_version,
                        new_paper=new_paper,
                    )
                else:
                    logger.info("[DRY RUN] Would create removal flag PR for %s", arxiv_id)
                continue

            # Extraction found data or failed — baseline as published, trust existing file
            state["files"][file_path] = {
                "arxiv_id": arxiv_id,
                "known_version": latest_version,
                "last_checked": now_iso,
                "published": True,
            }
            continue

        if latest_version <= known_version:
            state["files"][file_path]["last_checked"] = now_iso
            if published:
                state["files"][file_path]["published"] = True
            continue

        # Version change or new published file — do full published check
        if not published:
            published = is_published(new_paper)
        if known_version is not None:
            logger.info(
                "%s: new version v%d (was v%s)%s", arxiv_id, latest_version, known_version or "new",
                " [published]" if published else "",
            )

        # Extract data from new version
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                pdf_path = download_pdf(arxiv_id, Path(tmpdir))
                new_extraction = run_extraction_agent(new_paper, pdf_path, client)
            except Exception as e:
                logger.warning("Extraction failed for %s v%d: %s", arxiv_id, latest_version, e)
                continue

        if not new_extraction.data_points:
            if published:
                logger.warning(
                    "Published paper %s v%d yielded no data — possible limit removal for %s",
                    arxiv_id, latest_version, file_path,
                )
                state["files"][file_path] = {
                    "arxiv_id": arxiv_id,
                    "known_version": latest_version,
                    "last_checked": now_iso,
                    "published": True,
                }
                state["last_checked"] = now_iso
                save_version_state(state)

                if not dry_run:
                    _create_removal_flag_pr(
                        repo_root=repo_root,
                        file_path=file_path,
                        arxiv_id=arxiv_id,
                        old_version=known_version or 0,
                        new_version=latest_version,
                        new_paper=new_paper,
                    )
                else:
                    logger.info(
                        "[DRY RUN] Would create removal flag PR for %s v%d",
                        arxiv_id, latest_version,
                    )
            else:
                logger.info("No data extracted from %s v%d", arxiv_id, latest_version)
                state["files"][file_path] = {
                    "arxiv_id": arxiv_id,
                    "known_version": latest_version,
                    "last_checked": now_iso,
                    "published": False,
                }
            continue

        # Apply corrections
        corrected_data, applied, flagged = apply_corrections(new_extraction)

        # Compare with old data
        old_path = repo_root / file_path
        if not data_has_changed(old_path, corrected_data):
            logger.info("Data unchanged for %s v%d", arxiv_id, latest_version)
            state["files"][file_path]["known_version"] = latest_version
            state["files"][file_path]["last_checked"] = now_iso
            if published:
                state["files"][file_path]["published"] = True
            continue

        # Data changed — create PR
        old_ver = known_version or 0
        change_summary = summarise_changes(
            arxiv_id, old_ver, latest_version, new_paper, client
        )

        # Update and save state before creating the PR so it can be included in
        # the PR commit (avoids pushing directly to the protected master branch).
        state["files"][file_path] = {
            "arxiv_id": arxiv_id,
            "known_version": latest_version,
            "last_checked": now_iso,
            "published": published,
        }
        state["last_checked"] = now_iso
        save_version_state(state)

        if not dry_run:
            _create_update_pr(
                repo_root=repo_root,
                file_path=file_path,
                arxiv_id=arxiv_id,
                old_version=old_ver,
                new_version=latest_version,
                new_extraction=new_extraction,
                corrected_data=corrected_data,
                applied=applied,
                flagged=flagged,
                change_summary=change_summary,
                new_paper=new_paper,
                published=published,
            )
        else:
            logger.info("[DRY RUN] Would create PR for %s v%s→v%d", arxiv_id, known_version or "new", latest_version)

    state["last_checked"] = now_iso
    save_version_state(state)
    logger.info("Preprint check complete.")


def _create_removal_flag_pr(
    repo_root: Path,
    file_path: str,
    arxiv_id: str,
    old_version: int,
    new_version: int,
    new_paper: arxiv.Result,
) -> None:
    """Create a flag PR when a published paper yields no extractable data."""
    experiment_name = Path(file_path).stem
    branch = f"pipeline/review-{arxiv_id.replace('.', '-')}-v{new_version}"

    from .pr_creator import _run_git, _run_gh

    _run_git(["checkout", "-B", branch], repo_root)

    # Commit only the updated state file (no data file changes)
    state_file_rel = str(STATE_PATH.relative_to(repo_root))
    commit_msg = (
        f"Flag {experiment_name} for review: arXiv:{arxiv_id} v{new_version} (published)\n\n"
        f"Published version yielded no extractable data.\n"
        f"Auto-generated by preprint_checker\n"
    )
    try:
        _run_git(["add", state_file_rel], repo_root)
        _run_git(["commit", "-m", commit_msg], repo_root)
        _run_git(["push", "-u", "origin", branch], repo_root)
    except Exception:
        _run_git(["checkout", "master"], repo_root)
        raise

    old_url = f"https://arxiv.org/abs/{arxiv_id}v{old_version}"
    new_url = f"https://arxiv.org/abs/{arxiv_id}v{new_version}"

    title = f"[NEEDS REVIEW] {experiment_name}: published version may have removed limit"
    body = (
        f"## Possible Limit Removal: {experiment_name}\n\n"
        f"**Paper:** [{new_paper.title}]({new_url})\n\n"
        f"**arXiv ID:** [{arxiv_id}]({new_url})\n"
        f"- Old version: [v{old_version}]({old_url})\n"
        f"- Published version: [v{new_version}]({new_url})\n\n"
        f"> ⚠️ The published version of this paper yielded no extractable data points.\n"
        f"> This may indicate that the limit has been removed or substantially changed\n"
        f"> in the peer-reviewed version.\n\n"
        f"## Action Required\n\n"
        f"Please verify whether the limit in `{file_path}` is still valid by checking\n"
        f"the published version of the paper.\n\n"
        f"**No data files have been modified by this PR.**\n\n"
        f"🤖 Generated by AutoAxionLimits preprint checker"
    )

    try:
        _run_gh(
            ["pr", "create", "--title", title, "--body", body, "--base", "master",
             "--repo", "FaroutYLq/AutoAxionLimits"],
            repo_root,
        )
        logger.info("Created removal flag PR for %s v%d", arxiv_id, new_version)
    finally:
        _run_git(["checkout", "master"], repo_root)


def _create_update_pr(
    repo_root: Path,
    file_path: str,
    arxiv_id: str,
    old_version: int,
    new_version: int,
    new_extraction: ExtractionResult,
    corrected_data: list[tuple[float, float]],
    applied: list[str],
    flagged: list[str],
    change_summary: str,
    new_paper: arxiv.Result,
    published: bool = False,
) -> None:
    """Write updated data file and open a PR for a preprint update."""
    # Derive experiment name from file path
    experiment_name = Path(file_path).stem
    coupling_type = new_extraction.coupling_type or "Unknown"

    # Write updated data file
    new_content = format_data_file(corrected_data, new_extraction, applied)
    data_path = repo_root / file_path
    data_path.write_text(new_content)

    # Git branch + commit + PR
    branch = f"pipeline/preprint-{arxiv_id.replace('.', '-')}-v{old_version}-to-v{new_version}"

    from .pr_creator import _run_git, _run_gh

    _run_git(["checkout", "-B", branch], repo_root)

    state_file_rel = str(STATE_PATH.relative_to(repo_root))
    commit_msg = (
        f"Update {experiment_name}: arXiv:{arxiv_id} v{old_version}→v{new_version}\n\n"
        f"Auto-generated by preprint_checker\n"
    )
    try:
        _run_git(["add", file_path, state_file_rel], repo_root)
        _run_git(["commit", "-m", commit_msg], repo_root)
        _run_git(["push", "-u", "origin", branch], repo_root)
    except Exception:
        _run_git(["checkout", "master"], repo_root)
        raise

    title = f"Update {experiment_name} {coupling_type}: arXiv:{arxiv_id} v{old_version}→v{new_version}"

    old_url = f"https://arxiv.org/abs/{arxiv_id}v{old_version}"
    new_url = f"https://arxiv.org/abs/{arxiv_id}v{new_version}"

    corrections_md = "\n".join(f"- {c}" for c in applied) or "- None"
    flagged_md = "\n".join(f"- {c}" for c in flagged) or "- None"

    body = (
        f"## Preprint Update: {experiment_name}\n\n"
        f"**Paper:** [{new_paper.title}]({new_url})\n\n"
        f"**arXiv ID:** [{arxiv_id}]({new_url})\n"
        f"- Old version: [v{old_version}]({old_url})\n"
        f"- New version: [v{new_version}]({new_url})\n\n"
        f"> Note: {'Paper is now published in a journal. This is likely the final update.' if published else 'Paper is still a preprint (not yet peer-reviewed)'}\n\n"
        f"## What Changed\n\n{change_summary}\n\n"
        f"## Corrections Applied\n\n{corrections_md}\n\n"
        f"## Corrections Flagged for Human Review\n\n{flagged_md}\n\n"
        f"## Files Changed\n\n"
        f"- `{file_path}`\n\n"
        f"🤖 Generated by AutoAxionLimits preprint checker"
    )

    try:
        _run_gh(
            ["pr", "create", "--title", title, "--body", body, "--base", "master",
             "--repo", "FaroutYLq/AutoAxionLimits"],
            repo_root,
        )
        logger.info("Created PR for preprint update %s v%d→v%d", arxiv_id, old_version, new_version)
    finally:
        # Always return to master so subsequent iterations branch from the right base
        _run_git(["checkout", "master"], repo_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="AutoAxionLimits weekly preprint checker")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Populate preprint_versions.json with current versions; no PRs",
    )
    args = parser.parse_args()
    run_weekly_check(dry_run=args.dry_run, init_only=args.init_only)


if __name__ == "__main__":
    _cli()
