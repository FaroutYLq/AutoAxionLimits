"""
Daily arXiv digest entrypoint.

Usage:
  python -m pipeline.orchestrator [--arxiv-id 2412.12345] [--dry-run] [--days-back 3]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from .config import MAX_PAPERS_PER_RUN
from .extractor import download_pdf, run_extraction_agent
from .monitor import (
    fetch_paper_by_id,
    fetch_recent_papers,
    filter_new_papers,
    classify_coupling_type,
    load_state,
    mark_failed,
    mark_processed,
    save_state,
    STATE_PATH,
)
from .plot_regen import execute_notebook, execute_notebook_highlighted, get_notebook_plot_names
from .pr_creator import create_feature_branch, stage_and_commit_files, create_pull_request, checkout_branch
from .reviewer import ReviewResult, run_reviewer_agent, write_repo_files

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def main(
    days_back: int = 3,
    dry_run: bool = False,
    arxiv_id: str | None = None,
    max_papers: int = MAX_PAPERS_PER_RUN,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    state = load_state()

    # Determine papers to process
    if arxiv_id:
        logger.info("Force-processing single paper: %s", arxiv_id)
        papers = [fetch_paper_by_id(arxiv_id)]
    else:
        all_papers = fetch_recent_papers(days_back=days_back)
        papers = filter_new_papers(all_papers, state)[:max_papers]
        logger.info("Processing %d new papers", len(papers))

    processed_count = 0

    for paper in papers:
        from .monitor import _arxiv_id as get_id
        paper_id = get_id(paper)

        try:
            _process_paper(paper, paper_id, client, state, dry_run)
            mark_processed(state, paper_id)
            processed_count += 1
        except Exception as e:
            logger.exception("Failed to process %s: %s", paper_id, e)
            mark_failed(state, paper_id, str(e))

    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    logger.info("Done. Processed %d papers.", processed_count)


def _process_paper(paper, paper_id: str, client: anthropic.Anthropic, state: dict, dry_run: bool) -> None:
    """Process one paper: extract → review → write files → regen plot → create PR."""

    # Quick local pre-filter
    coupling_guess = classify_coupling_type(paper)
    if coupling_guess is None:
        logger.info("No keyword match for %s; skipping", paper_id)
        mark_processed(state, paper_id, reason="no_keyword_match")
        return

    logger.info("Processing %s (guessed coupling: %s)", paper_id, coupling_guess)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = download_pdf(paper_id, Path(tmpdir))
        extraction = run_extraction_agent(paper, pdf_path, client)

    if not extraction.is_new_limit:
        logger.info("%s: not a new limit (is_new_limit=False)", paper_id)
        mark_processed(state, paper_id, reason="not_new_limit")
        return

    if not extraction.data_points:
        logger.info("%s: no data points extracted", paper_id)
        mark_processed(state, paper_id, reason="no_data_points")
        return

    if extraction.coupling_type is None:
        logger.info("%s: coupling type undetermined", paper_id)
        mark_processed(state, paper_id, reason="no_coupling_type")
        return

    logger.info(
        "%s: %s limit, %d points, conf=%.2f",
        paper_id,
        extraction.coupling_type,
        len(extraction.data_points),
        extraction.extraction_confidence,
    )

    review = run_reviewer_agent(extraction, client)

    if dry_run:
        logger.info(
            "[DRY RUN] Would create PR: %s %s (conf=%.2f)",
            review.experiment_name,
            extraction.coupling_type,
            review.extraction_confidence,
        )
        logger.info("[DRY RUN] Data file: %s", review.data_file_path)
        logger.info("[DRY RUN] Corrections applied: %s", review.corrections_applied)
        return

    # Write all repo files
    write_repo_files(review, REPO_ROOT)

    # Regenerate plot(s)
    nb_ok, nb_err = execute_notebook(review.notebook_path, REPO_ROOT)
    if not nb_ok:
        logger.warning("Notebook execution failed for %s: %s", review.notebook_path, nb_err[-500:])
        # Continue — PR still valuable even without regenerated plot

    # Generate highlighted plot (new limit in colour, everything else grey)
    hl_ok, hl_err, highlight_files = execute_notebook_highlighted(
        review.notebook_path, review.notebook_call, REPO_ROOT,
    )
    if not hl_ok:
        logger.warning("Highlighted plot generation failed: %s", hl_err[-500:])

    # Git branch, commit, PR
    branch = create_feature_branch(paper_id, review.experiment_name, REPO_ROOT)

    # Mark processed and save state so it's included in the PR commit.
    # This updates master's state when the PR merges, avoiding a direct push to the
    # protected master branch.
    mark_processed(state, paper_id)
    save_state(state)

    changed_files = [
        str(STATE_PATH.relative_to(REPO_ROOT)),
        review.data_file_path,
        review.plotfuncs_file,
        review.notebook_path,
        review.docs_file,
    ]
    # Include plot files actually produced by the notebook
    plot_names = get_notebook_plot_names(review.notebook_path, REPO_ROOT)
    for name in plot_names:
        for p in [f"plots/{name}.pdf", f"plots/plots_png/{name}.png"]:
            if (REPO_ROOT / p).exists():
                changed_files.append(p)
    # Include highlighted plot files
    changed_files.extend(highlight_files)

    commit_msg = (
        f"Add {review.experiment_name} {extraction.coupling_type} limit\n\n"
        f"Source: arXiv:{paper_id}\n"
        f"Extraction confidence: {extraction.extraction_confidence:.2f}\n"
        f"Data points: {len(extraction.data_points)}\n"
        f"Auto-generated by AutoAxionLimits daily pipeline\n"
    )
    try:
        stage_and_commit_files(changed_files, commit_msg, REPO_ROOT)
        pr_url = create_pull_request(branch, review, extraction, REPO_ROOT,
                                     highlight_files=highlight_files)
        logger.info("PR created: %s", pr_url)
    finally:
        # Always return to master so subsequent papers branch from the right base
        checkout_branch("master", REPO_ROOT)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="AutoAxionLimits daily arXiv digest")
    parser.add_argument("--arxiv-id", help="Force-process a specific arXiv ID")
    parser.add_argument("--days-back", type=int, default=3, help="Days of arXiv history to scan")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    parser.add_argument("--max-papers", type=int, default=MAX_PAPERS_PER_RUN)
    args = parser.parse_args()
    main(
        days_back=args.days_back,
        dry_run=args.dry_run,
        arxiv_id=args.arxiv_id,
        max_papers=args.max_papers,
    )


if __name__ == "__main__":
    _cli()
