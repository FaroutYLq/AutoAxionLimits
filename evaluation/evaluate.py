"""Main evaluation script for the extraction pipeline.

Usage:
    # Populate ground-truth data files from repo (one-time setup)
    python -m evaluation.evaluate --populate

    # Run extraction on all ground-truth papers (calls Claude API)
    python -m evaluation.evaluate --extract

    # Run extraction on a single paper
    python -m evaluation.evaluate --extract --arxiv-id 2208.03183

    # Compute metrics from cached results (no API calls)
    python -m evaluation.evaluate --metrics

    # Full pipeline: extract + metrics + report
    python -m evaluation.evaluate --extract --metrics --report

    # Generate report from cached results only
    python -m evaluation.evaluate --metrics --report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ground_truth import (
    GroundTruthEntry,
    load_ground_truth,
    populate_data_from_repo,
)
from evaluation.metrics import (
    ClassificationMetrics,
    CurveMetrics,
    InterpolationMetrics,
    compute_confidence_calibration,
    compute_curve_metrics,
    compute_interpolation_metrics,
)
from evaluation.report import generate_report

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def _fetch_paper_metadata(arxiv_id: str, cache_path: Path) -> tuple[str, str]:
    """Fetch real title and abstract from arXiv API. Cache results."""
    import json as _json
    if cache_path.exists():
        with open(cache_path) as f:
            cache = _json.load(f)
    else:
        cache = {}
    if arxiv_id in cache:
        return cache[arxiv_id]["title"], cache[arxiv_id]["abstract"]
    # Fetch from arXiv
    import arxiv as _arxiv
    search = _arxiv.Search(id_list=[arxiv_id])
    result = next(_arxiv.Client().results(search), None)
    if result:
        cache[arxiv_id] = {"title": result.title, "abstract": result.summary}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            _json.dump(cache, f, indent=2)
        return result.title, result.summary
    return "", ""


def _load_cached_result(arxiv_id: str) -> dict | None:
    """Load a cached extraction result, if it exists."""
    path = RESULTS_DIR / f"{arxiv_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_result(arxiv_id: str, result: dict):
    """Cache an extraction result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{arxiv_id}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved result for %s", arxiv_id)


def run_extraction(entry: GroundTruthEntry) -> dict:
    """Run the pipeline extraction on a single paper.

    Returns a dict with ExtractionResult fields + timing info.
    """
    import anthropic

    from pipeline.extractor import ExtractionResult, download_pdf, run_extraction_agent

    client = anthropic.Anthropic()

    # Create a minimal paper-like object for the extractor
    class _PaperStub:
        def __init__(self, arxiv_id: str, title: str, summary: str = "", categories: list = None):
            self.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
            self.title = title
            self.summary = summary
            self.categories = categories or []

        def get_short_id(self):
            return self.arxiv_id

    real_title, abstract = _fetch_paper_metadata(
        entry.arxiv_id, RESULTS_DIR / "metadata_cache.json"
    )
    title = real_title or entry.paper_title
    paper_stub = _PaperStub(entry.arxiv_id, title, summary=abstract)
    paper_stub.arxiv_id = entry.arxiv_id

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()

        try:
            pdf_path = download_pdf(entry.arxiv_id, Path(tmpdir))
        except Exception as e:
            logger.error("PDF download failed for %s: %s", entry.arxiv_id, e)
            return {
                "arxiv_id": entry.arxiv_id,
                "error": f"PDF download failed: {e}",
                "elapsed_s": time.time() - t0,
            }

        try:
            result: ExtractionResult = run_extraction_agent(
                paper_stub, pdf_path, client
            )
        except Exception as e:
            logger.error("Extraction failed for %s: %s", entry.arxiv_id, e)
            return {
                "arxiv_id": entry.arxiv_id,
                "error": f"Extraction failed: {e}",
                "elapsed_s": time.time() - t0,
            }

        elapsed = time.time() - t0

    return {
        "arxiv_id": result.arxiv_id,
        "paper_title": result.paper_title,
        "coupling_type": result.coupling_type,
        "is_new_limit": result.is_new_limit,
        "is_projection": result.is_projection,
        "data_points": result.data_points,
        "data_source": result.data_source,
        "dm_density_assumed": result.dm_density_assumed,
        "confidence_level": result.confidence_level,
        "extraction_confidence": result.extraction_confidence,
        "suggested_experiment_name": result.suggested_experiment_name,
        "notes": result.notes,
        "num_points": len(result.data_points),
        "elapsed_s": elapsed,
    }


def _normalize_predicted_coupling(raw_ct):
    """Normalize a predicted coupling type: handle lists, apply alias normalization."""
    from pipeline.reviewer import _normalize_coupling_type

    if raw_ct is None:
        return None
    # Handle list returns — take first element
    if isinstance(raw_ct, list):
        raw_ct = raw_ct[0] if raw_ct else None
    if raw_ct is None:
        return None
    # Try normalization via reviewer aliases
    try:
        return _normalize_coupling_type(raw_ct)
    except KeyError:
        return raw_ct  # keep raw if normalization fails


def compute_all_metrics(
    entries: list[GroundTruthEntry],
    results: list[dict],
) -> dict:
    """Compute all evaluation metrics.

    Returns a dict with classification metrics, curve metrics, and calibration data.
    """
    coupling_clf = ClassificationMetrics()
    is_limit_clf = ClassificationMetrics()
    is_projection_clf = ClassificationMetrics()
    data_source_clf = ClassificationMetrics()

    curve_metrics_list: list[CurveMetrics] = []
    interp_metrics_list: list[InterpolationMetrics] = []
    confidences: list[float] = []
    curve_arxiv_ids: list[str] = []

    per_paper: list[dict] = []

    # Build multi-coupling map: for papers with multiple GT entries,
    # the single extraction should match ANY expected coupling type
    expected_couplings_by_id = defaultdict(set)
    for entry in entries:
        expected_couplings_by_id[entry.arxiv_id].add(entry.coupling_type)

    # Track which arxiv_ids we've already recorded for coupling classification
    # to avoid counting the same prediction multiple times for multi-coupling papers
    seen_coupling_ids: set[str] = set()

    for entry, result in zip(entries, results):
        paper_report: dict = {"arxiv_id": entry.arxiv_id, "difficulty": entry.difficulty}
        is_multi_coupling = len(expected_couplings_by_id[entry.arxiv_id]) > 1
        paper_report["multi_coupling"] = is_multi_coupling

        if "error" in result:
            paper_report["status"] = "extraction_failed"
            paper_report["error"] = result["error"]
            per_paper.append(paper_report)
            continue

        paper_report["status"] = "extracted"
        paper_report["extraction_confidence"] = result.get("extraction_confidence", 0.0)
        paper_report["data_source"] = result.get("data_source")
        paper_report["num_points_extracted"] = result.get("num_points", 0)
        paper_report["elapsed_s"] = result.get("elapsed_s", 0.0)

        # Normalize predicted coupling type (handles lists + aliases)
        predicted_ct = _normalize_predicted_coupling(result.get("coupling_type"))

        # For multi-coupling papers, check against ALL expected types
        all_expected = expected_couplings_by_id[entry.arxiv_id]
        ct_correct = predicted_ct in all_expected if predicted_ct else False

        # Only record one classification per unique arxiv_id to avoid
        # inflating counts for multi-coupling papers
        if entry.arxiv_id not in seen_coupling_ids:
            coupling_clf.total += 1
            if ct_correct:
                coupling_clf.correct += 1
            else:
                coupling_clf.errors.append({
                    "arxiv_id": entry.arxiv_id,
                    "predicted": str(predicted_ct),
                    "expected": str(sorted(all_expected)) if is_multi_coupling else str(entry.coupling_type),
                })
            is_limit_clf.record(entry.arxiv_id, result.get("is_new_limit"), entry.is_new_limit)
            is_projection_clf.record(entry.arxiv_id, result.get("is_projection"), entry.is_projection)
            data_source_clf.record(entry.arxiv_id, result.get("data_source"), entry.data_source_expected)
            seen_coupling_ids.add(entry.arxiv_id)

        paper_report["coupling_type_correct"] = ct_correct
        paper_report["coupling_type_predicted"] = predicted_ct
        paper_report["coupling_type_expected"] = entry.coupling_type
        if is_multi_coupling:
            paper_report["all_expected_couplings"] = sorted(all_expected)

        # Curve comparison (if ground-truth data is available)
        gt_data = entry.load_data()
        if gt_data is None:
            gt_data = entry.load_reference_data(PROJECT_ROOT)

        extracted_points = result.get("data_points", [])
        if gt_data is not None and len(extracted_points) > 0:
            ext_array = np.array(extracted_points, dtype=float, ndmin=2)

            # Primary: interpolation-based metric
            im = compute_interpolation_metrics(entry.arxiv_id, ext_array, gt_data)
            interp_metrics_list.append(im)
            confidences.append(result.get("extraction_confidence", 0.0))
            curve_arxiv_ids.append(entry.arxiv_id)

            paper_report["interp_metrics"] = {
                "num_extracted": im.num_extracted,
                "num_ground_truth": im.num_ground_truth,
                "num_interpolatable": im.num_interpolatable,
                "interpolation_coverage": im.interpolation_coverage,
                "median_residual_dex": im.median_residual_dex,
                "mean_residual_dex": im.mean_residual_dex,
                "p90_residual_dex": im.p90_residual_dex,
                "max_residual_dex": im.max_residual_dex,
                "frac_within_0_1dex": im.frac_within_0_1dex,
                "frac_within_0_3dex": im.frac_within_0_3dex,
                "frac_within_0_5dex": im.frac_within_0_5dex,
                "frac_within_1_0dex": im.frac_within_1_0dex,
            }

            # Secondary: legacy point-matching metric
            cm = compute_curve_metrics(entry.arxiv_id, ext_array, gt_data)
            curve_metrics_list.append(cm)

            paper_report["curve_metrics"] = {
                "hausdorff_log": cm.hausdorff_log,
                "coverage_at_0_5dex": cm.coverage_at_0_5dex,
                "coverage_at_1_0dex": cm.coverage_at_1_0dex,
                "mass_range_overlap": cm.mass_range_overlap,
                "median_coupling_log_error": cm.median_coupling_log_error,
                "p90_coupling_log_error": cm.p90_coupling_log_error,
                "num_extracted": cm.num_extracted,
                "num_ground_truth": cm.num_ground_truth,
            }
        else:
            paper_report["interp_metrics"] = None
            paper_report["curve_metrics"] = None

        per_paper.append(paper_report)

    # Confidence calibration (uses interpolation metrics)
    calibration = compute_confidence_calibration(
        confidences, interp_metrics_list, curve_arxiv_ids
    )

    # Aggregate interpolation statistics (primary)
    if interp_metrics_list:
        valid = [m for m in interp_metrics_list if m.median_residual_dex < float("inf")]
        aggregate_interp = {
            "n_papers": len(interp_metrics_list),
            "mean_interpolation_coverage": float(np.mean([m.interpolation_coverage for m in interp_metrics_list])),
            "mean_median_residual_dex": float(np.mean([m.median_residual_dex for m in valid])) if valid else None,
            "mean_p90_residual_dex": float(np.mean([m.p90_residual_dex for m in valid])) if valid else None,
            "mean_frac_within_0_3dex": float(np.mean([m.frac_within_0_3dex for m in valid])) if valid else None,
            "mean_frac_within_0_5dex": float(np.mean([m.frac_within_0_5dex for m in valid])) if valid else None,
        }
    else:
        aggregate_interp = {"n_papers": 0}

    # Aggregate legacy curve statistics (secondary)
    if curve_metrics_list:
        coverages_05 = [m.coverage_at_0_5dex for m in curve_metrics_list]
        coverages_10 = [m.coverage_at_1_0dex for m in curve_metrics_list]
        med_errs = [m.median_coupling_log_error for m in curve_metrics_list
                    if m.median_coupling_log_error < float("inf")]
        mass_overlaps = [m.mass_range_overlap for m in curve_metrics_list]

        aggregate_curve = {
            "n_papers_with_curves": len(curve_metrics_list),
            "mean_coverage_0_5dex": float(np.mean(coverages_05)),
            "mean_coverage_1_0dex": float(np.mean(coverages_10)),
            "mean_median_coupling_log_error": float(np.mean(med_errs)) if med_errs else None,
            "mean_mass_range_overlap": float(np.mean(mass_overlaps)),
        }
    else:
        aggregate_curve = {"n_papers_with_curves": 0}

    # Per-difficulty breakdown
    difficulty_breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [p for p in per_paper if p.get("difficulty") == diff]
        if not subset:
            continue
        extracted = [p for p in subset if p.get("status") == "extracted"]
        with_interp = [p for p in extracted if p.get("interp_metrics") is not None]
        valid_interp = [p for p in with_interp
                        if p["interp_metrics"]["median_residual_dex"] < float("inf")]
        difficulty_breakdown[diff] = {
            "total": len(subset),
            "extracted": len(extracted),
            "coupling_type_accuracy": (
                sum(1 for p in extracted if p.get("coupling_type_correct")) / len(extracted)
                if extracted else 0.0
            ),
            "mean_median_residual_dex": (
                float(np.mean([p["interp_metrics"]["median_residual_dex"] for p in valid_interp]))
                if valid_interp else None
            ),
            "mean_frac_within_0_3dex": (
                float(np.mean([p["interp_metrics"]["frac_within_0_3dex"] for p in valid_interp]))
                if valid_interp else None
            ),
        }

    # Per-data-source breakdown
    source_breakdown = {}
    for source in ["table", "figure_vision", "text"]:
        subset = [p for p in per_paper if p.get("data_source") == source]
        if not subset:
            continue
        with_interp = [p for p in subset if p.get("interp_metrics") is not None]
        valid_interp = [p for p in with_interp
                        if p["interp_metrics"]["median_residual_dex"] < float("inf")]
        source_breakdown[source] = {
            "total": len(subset),
            "mean_median_residual_dex": (
                float(np.mean([p["interp_metrics"]["median_residual_dex"] for p in valid_interp]))
                if valid_interp else None
            ),
            "mean_frac_within_0_3dex": (
                float(np.mean([p["interp_metrics"]["frac_within_0_3dex"] for p in valid_interp]))
                if valid_interp else None
            ),
        }

    return {
        "classification": {
            "coupling_type": {"accuracy": coupling_clf.accuracy, "total": coupling_clf.total, "errors": coupling_clf.errors},
            "is_new_limit": {"accuracy": is_limit_clf.accuracy, "total": is_limit_clf.total, "errors": is_limit_clf.errors},
            "is_projection": {"accuracy": is_projection_clf.accuracy, "total": is_projection_clf.total, "errors": is_projection_clf.errors},
            "data_source": {"accuracy": data_source_clf.accuracy, "total": data_source_clf.total, "errors": data_source_clf.errors},
        },
        "interpolation_aggregate": aggregate_interp,
        "curve_aggregate": aggregate_curve,
        "confidence_calibration": [asdict(b) for b in calibration],
        "difficulty_breakdown": difficulty_breakdown,
        "source_breakdown": source_breakdown,
        "per_paper": per_paper,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate the AutoAxionLimits extraction pipeline")
    parser.add_argument("--populate", action="store_true",
                        help="Populate ground-truth data/ from repo reference files")
    parser.add_argument("--extract", action="store_true",
                        help="Run extraction on ground-truth papers (calls Claude API)")
    parser.add_argument("--metrics", action="store_true",
                        help="Compute metrics from cached extraction results")
    parser.add_argument("--report", action="store_true",
                        help="Generate evaluation report (markdown + plots)")
    parser.add_argument("--arxiv-id", type=str, default=None,
                        help="Only process this arXiv ID (with --extract)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cached result exists")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for report (default: evaluation/report.md)")

    args = parser.parse_args()

    if not any([args.populate, args.extract, args.metrics, args.report]):
        parser.print_help()
        return

    entries = load_ground_truth()
    logger.info("Loaded %d ground-truth papers", len(entries))

    if args.populate:
        n = populate_data_from_repo(PROJECT_ROOT)
        logger.info("Populated %d data files from repo", n)

    if args.extract:
        target_entries = entries
        if args.arxiv_id:
            target_entries = [e for e in entries if e.arxiv_id == args.arxiv_id]
            if not target_entries:
                logger.error("arXiv ID %s not found in ground truth", args.arxiv_id)
                return

        for entry in target_entries:
            cached = _load_cached_result(entry.arxiv_id)
            if cached and not args.force:
                logger.info("Using cached result for %s", entry.arxiv_id)
                continue

            logger.info("Extracting %s: %s", entry.arxiv_id, entry.paper_title)
            result = run_extraction(entry)
            _save_result(entry.arxiv_id, result)

            # Be nice to the API
            time.sleep(2)

    if args.metrics or args.report:
        # Load all cached results
        results = []
        valid_entries = []
        for entry in entries:
            cached = _load_cached_result(entry.arxiv_id)
            if cached is None:
                logger.warning("No cached result for %s, skipping", entry.arxiv_id)
                continue
            results.append(cached)
            valid_entries.append(entry)

        if not results:
            logger.error("No cached results found. Run --extract first.")
            return

        all_metrics = compute_all_metrics(valid_entries, results)

        # Save metrics
        metrics_path = RESULTS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info("Metrics saved to %s", metrics_path)

        if args.report:
            report_path = args.output or str(Path(__file__).parent / "report.md")
            generate_report(all_metrics, report_path)
            logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
