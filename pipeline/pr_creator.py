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
from .plot_regen import get_notebook_plot_names
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

def _local_branch_exists(branch: str, repo_root: Path = REPO_ROOT) -> bool:
    """True if a local branch named *branch* already exists."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    return result.returncode == 0


def _remote_branch_exists(branch: str, repo_root: Path = REPO_ROOT) -> bool:
    """True if origin already has a branch named *branch*.

    A stale remote branch (e.g. from a long-closed PR) is exactly what makes a
    plain ``git push`` fail non-fast-forward. If ``ls-remote`` itself fails
    (offline / no remote configured), report False so branch selection degrades
    to the local-only check rather than aborting the whole run.
    """
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def _pick_unique_branch(base: str, exists) -> str:
    """Return *base*, or ``base-2``, ``base-3``, … — the first name *exists()* rejects.

    Pure and injectable: *exists* is a predicate ``str -> bool``, so the numbering
    logic is unit-testable without git.
    """
    if not exists(base):
        return base
    n = 2
    while exists(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def create_feature_branch(
    arxiv_id: str, experiment_name: str, repo_root: Path = REPO_ROOT
) -> str:
    """Create and checkout a unique branch ``pipeline/arxiv-{id}-{name}``.

    The name is uniquified against BOTH local and remote branches: re-processing a
    paper whose old PR branch still lives on origin used to reuse that name and then
    fail the push non-fast-forward (marking a successfully-extracted paper failed).
    A ``-2``/``-3``/… suffix is appended until the name is free on both sides.
    """
    safe_id = arxiv_id.replace(".", "-")
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "-", experiment_name)[:40]
    base = f"pipeline/arxiv-{safe_id}-{safe_name}"

    branch = _pick_unique_branch(
        base,
        lambda b: _local_branch_exists(b, repo_root) or _remote_branch_exists(b, repo_root),
    )

    # Fresh unique name → -b creates it. Guard the rare race where it appeared
    # between the check and now by falling back to a plain checkout.
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
    highlight_files: list[str] | None = None,
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

    # Use the first plot name produced by the selected notebook; fall back to coupling name
    # GitHub PR descriptions don't resolve relative image paths — use absolute raw URL
    plot_names = get_notebook_plot_names(review.notebook_path, repo_root)
    plot_stem = plot_names[0] if plot_names else coupling
    plot_png = f"https://raw.githubusercontent.com/FaroutYLq/AutoAxionLimits/{branch_name}/plots/plots_png/{plot_stem}.png"

    # Highlighted plot: new limit in colour, everything else grey
    highlight_png_files = [f for f in (highlight_files or []) if f.endswith(".png")]
    if highlight_png_files:
        hl_stem = Path(highlight_png_files[0]).name
        hl_png = f"https://raw.githubusercontent.com/FaroutYLq/AutoAxionLimits/{branch_name}/plots/plots_png/{hl_stem}"
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
        f"{plot_section}"
        f"---\n"
        f"> All updates are PRs — nothing merges automatically. "
        f"Please verify extraction accuracy before merging.\n\n"
        f"🤖 Generated by AutoAxionLimits daily pipeline"
    )

    pr_url = _run_gh(
        ["pr", "create", "--title", title, "--body", body, "--base", "master",
         "--repo", "FaroutYLq/AutoAxionLimits"],
        repo_root,
    )
    logger.info("Created PR: %s", pr_url)
    return pr_url


# ---------------------------------------------------------------------------
# PR creation — preprint updates (used by preprint_checker)
# ---------------------------------------------------------------------------

def checkout_branch(branch: str, repo_root: Path = REPO_ROOT) -> None:
    """Check out an existing branch."""
    _run_git(["checkout", branch], repo_root)


def create_pull_request_preprint(
    branch_name: str,
    title: str,
    body: str,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Push branch and open a preprint-update PR."""
    _run_git(["push", "-u", "origin", branch_name], repo_root)
    pr_url = _run_gh(
        ["pr", "create", "--title", title, "--body", body, "--base", "master",
         "--repo", "FaroutYLq/AutoAxionLimits"],
        repo_root,
    )
    logger.info("Created preprint PR: %s", pr_url)
    return pr_url
