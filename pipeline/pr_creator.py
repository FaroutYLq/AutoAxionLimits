"""
Git branch creation, commit, and GitHub PR creation via gh CLI.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from .extractor import ExtractionResult
from .reviewer import ReviewResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Git / gh helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    cmd = ["git"] + args
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result.stdout.strip()


def _run_gh(args: list[str], cwd: Path = REPO_ROOT) -> str:
    cmd = ["gh"] + args
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args)} failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Branch management
# ---------------------------------------------------------------------------

def create_feature_branch(
    arxiv_id: str, experiment_name: str, repo_root: Path = REPO_ROOT
) -> str:
    """Create and checkout branch pipeline/arxiv-{id}-{name}."""
    safe_id = arxiv_id.replace(".", "-")
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "-", experiment_name)[:40]
    branch = f"pipeline/arxiv-{safe_id}-{safe_name}"

    # Try to create; if it already exists, check it out
    try:
        _run_git(["checkout", "-b", branch], repo_root)
    except RuntimeError:
        _run_git(["checkout", branch], repo_root)

    logger.info("On branch %s", branch)
    return branch


# ---------------------------------------------------------------------------
# Staging and committing
# ---------------------------------------------------------------------------

def stage_and_commit_files(
    files: list[str], commit_message: str, repo_root: Path = REPO_ROOT
) -> None:
    """Stage specific named files and commit."""
    for f in files:
        _run_git(["add", f], repo_root)
    _run_git(["commit", "-m", commit_message], repo_root)
    logger.info("Committed: %s", commit_message[:60])


# ---------------------------------------------------------------------------
# PR creation — daily digest
# ---------------------------------------------------------------------------

def create_pull_request(
    branch_name: str,
    review: ReviewResult,
    extraction: ExtractionResult,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Push branch and open a GitHub PR. Returns the PR URL."""
    _run_git(["push", "-u", "origin", branch_name], repo_root)

    # PR title
    prefix = ""
    if review.low_confidence:
        prefix = "[LOW CONFIDENCE] "
    elif review.is_projection:
        prefix = "[PROJECTION] "

    coupling = extraction.coupling_type or "Unknown"
    title = f"{prefix}Add {review.experiment_name} {coupling} limit (arXiv:{review.arxiv_id})"

    # Mass/coupling range summary
    if extraction.data_points:
        masses = [m for m, _ in extraction.data_points]
        couplings = [g for _, g in extraction.data_points]
        range_summary = (
            f"- Mass range: {min(masses):.2e} – {max(masses):.2e} eV\n"
            f"- Coupling range: {min(couplings):.2e} – {max(couplings):.2e}\n"
        )
    else:
        range_summary = "- No data points extracted\n"

    corrections_md = "\n".join(f"- {c}" for c in review.corrections_applied) or "- None"
    flagged_md = "\n".join(f"- {c}" for c in review.corrections_flagged) or "- None"
    confidence_note = (
        f"\n> ⚠️ **Low confidence extraction ({extraction.extraction_confidence:.0%})** — "
        "please verify data carefully before merging."
        if review.low_confidence
        else ""
    )

    plot_coupling = coupling.lower().replace("axion", "Axion")
    plot_png = f"plots/plots_png/{coupling}.png"

    body = (
        f"## New Limit: {review.experiment_name}\n\n"
        f"**Paper:** [{review.paper_title}]({review.arxiv_url})\n"
        f"**Data source:** {extraction.data_source}\n"
        f"**Confidence level:** {extraction.confidence_level:.0%}\n"
        f"**Extraction confidence:** {extraction.extraction_confidence:.0%}\n"
        f"{confidence_note}\n\n"
        f"## Data Summary\n\n{range_summary}\n"
        f"## Physical Corrections Applied\n\n{corrections_md}\n\n"
        f"## Corrections Flagged for Human Review\n\n{flagged_md}\n\n"
        f"## Files Changed\n\n"
        f"- `{review.data_file_path}`\n"
        f"- `{review.plotfuncs_file}` (new method `{review.plotfuncs_class}.{review.experiment_name}`)\n"
        f"- `{review.notebook_path}`\n"
        f"- `{review.docs_file}`\n\n"
        f"## Plot\n\n"
        f"![{coupling} limits]({plot_png})\n\n"
        f"---\n"
        f"> All updates are PRs — nothing merges automatically. "
        f"Please verify extraction accuracy before merging.\n\n"
        f"🤖 Generated by AutoAxionLimits daily pipeline"
    )

    pr_url = _run_gh(
        ["pr", "create", "--title", title, "--body", body, "--base", "master"],
        repo_root,
    )
    logger.info("Created PR: %s", pr_url)
    return pr_url


# ---------------------------------------------------------------------------
# PR creation — preprint updates (used by preprint_checker)
# ---------------------------------------------------------------------------

def create_pull_request_preprint(
    branch_name: str,
    title: str,
    body: str,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Push branch and open a preprint-update PR."""
    _run_git(["push", "-u", "origin", branch_name], repo_root)
    pr_url = _run_gh(
        ["pr", "create", "--title", title, "--body", body, "--base", "master"],
        repo_root,
    )
    logger.info("Created preprint PR: %s", pr_url)
    return pr_url
