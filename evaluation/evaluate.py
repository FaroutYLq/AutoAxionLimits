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
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.conventions import canonical_convention
from evaluation.ground_truth import (
    GroundTruthEntry,
    load_ground_truth,
    populate_data_from_repo,
)
from evaluation.metrics import (
    ClassificationMetrics,
    CurveMetrics,
    InterpolationMetrics,
    SymmetricCurveMetrics,
    compute_confidence_calibration,
    compute_curve_metrics,
    compute_interpolation_metrics,
    compute_symmetric_curve_metrics,
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
    # Fetch from arXiv. The metadata API (export.arxiv.org) is aggressively
    # rate-limited; a failure here is non-fatal — we fall back to the
    # ground-truth title and an empty abstract so extraction can proceed.
    import arxiv as _arxiv
    result = None
    for attempt in range(4):
        try:
            search = _arxiv.Search(id_list=[arxiv_id])
            result = next(_arxiv.Client().results(search), None)
            break
        except Exception as e:  # HTTP 429, parse errors, transient network
            wait = 5 * (2 ** attempt)
            logger.warning("arXiv metadata fetch failed for %s (%s); retry in %ds",
                           arxiv_id, e, wait)
            time.sleep(wait)
    if result:
        cache[arxiv_id] = {"title": result.title, "abstract": result.summary}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            _json.dump(cache, f, indent=2)
        return result.title, result.summary
    logger.warning("No arXiv metadata for %s; using ground-truth title fallback", arxiv_id)
    return "", ""


def _safe_id(arxiv_id: str) -> str:
    """Filesystem-safe key for an arXiv id (old-style ids contain '/')."""
    return arxiv_id.replace("/", "_")


def _load_cached_result(arxiv_id: str) -> dict | None:
    """Load a cached extraction result, if it exists."""
    path = RESULTS_DIR / f"{_safe_id(arxiv_id)}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_result(arxiv_id: str, result: dict):
    """Cache an extraction result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_safe_id(arxiv_id)}.json"
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


# Map a limit_data/<dir>/ basename to its canonical coupling type. The data
# file's directory is the authoritative physical coupling of a GT curve —
# more reliable than GroundTruthEntry.coupling_type, which is a placeholder for
# auto-expanded entries and is occasionally wrong for multi-coupling papers
# (e.g. a DarkPhoton-labelled entry pointing at an AxionElectron data file).
try:
    from pipeline.config import COUPLING_TYPES as _COUPLING_TYPES_REG
    _DIR_TO_COUPLING = {
        Path(meta["data_dir"]).name: key for key, meta in _COUPLING_TYPES_REG.items()
    }
except Exception:  # pragma: no cover - config import is best-effort
    _DIR_TO_COUPLING = {}
_DIR_TO_COUPLING.setdefault("VectorB-L", "VectorBL")
_DIR_TO_COUPLING["fa"] = "AxionMass"  # m_a vs f_a plane is classified AxionMass


def _authoritative_coupling(entry: GroundTruthEntry) -> str:
    """Physical coupling of an entry's GT *data file*, from its repo path."""
    ref = entry.reference_repo_file
    if ref:
        parts = Path(ref).parts
        if len(parts) >= 2 and parts[0] == "limit_data":
            return _DIR_TO_COUPLING.get(parts[1], entry.coupling_type)
    return entry.coupling_type


def _usable_gt_stats(gt_data, coupling_type: str) -> tuple[int, int]:
    """(n_points, n_unique_masses) of GT points that survive boundary-closure
    filtering. Interpolation needs >= 2 distinct masses; a curve with only one
    distinct mass (a single prediction or a single-mass projection) is a point
    reference, not a comparable curve."""
    from evaluation.metrics import _COUPLING_CEILINGS, _filter_boundary
    ceil = _COUPLING_CEILINGS.get(coupling_type, 1e-2)
    f = _filter_boundary(gt_data, ceil)
    if len(f) == 0:
        return 0, 0
    return len(f), int(np.unique(f[:, 0]).size)


def _is_placeholder_entry(entry: GroundTruthEntry) -> bool:
    """True if the entry's scalar labels (is_new_limit, is_projection,
    data_source_expected, difficulty) are auto-generated placeholders rather
    than human-verified values, and therefore cannot be scored against."""
    return ("auto_expanded" in (entry.tags or [])) or entry.verified_by == "repo_upstream"


def compute_all_metrics(
    entries: list[GroundTruthEntry],
    results: list[dict],
) -> dict:
    """Compute all evaluation metrics.

    Returns a dict with classification metrics, curve metrics, and calibration data.
    """
    coupling_clf = ClassificationMetrics()
    # Scalar-label metrics are scored ONLY against human-verified entries.
    # Auto-expanded / repo_upstream entries carry placeholder labels
    # (is_new_limit=True, is_projection=False, data_source="table"), so scoring
    # against them measures the placeholder, not the pipeline.
    is_limit_clf = ClassificationMetrics()
    is_projection_clf = ClassificationMetrics()
    data_source_clf = ClassificationMetrics()

    curve_metrics_list: list[CurveMetrics] = []
    interp_metrics_list: list[InterpolationMetrics] = []
    symmetric_metrics_list: list[SymmetricCurveMetrics] = []
    confidences: list[float] = []
    curve_arxiv_ids: list[str] = []

    per_paper: list[dict] = []

    # Why a paper did / didn't get a curve comparison. Honest aggregates require
    # knowing this: a paper whose extracted coupling has no matching GT curve is
    # NOT an extraction-quality failure — it is simply not comparable.
    comparison_status_counts: Counter = Counter()

    # Group all GT entries by paper; one extraction result per paper. A single
    # paper often yields several repo files (one per coupling); we must NOT
    # score one extraction against curves for couplings it never targeted.
    by_id: "OrderedDict[str, dict]" = OrderedDict()
    for entry, result in zip(entries, results):
        slot = by_id.setdefault(entry.arxiv_id, {"entries": [], "result": result})
        slot["entries"].append(entry)
        # Results are identical across a paper's entries; prefer a non-error one.
        if "error" in slot["result"] and "error" not in result:
            slot["result"] = result

    for arxiv_id, slot in by_id.items():
        paper_entries: list[GroundTruthEntry] = slot["entries"]
        result = slot["result"]
        rep = paper_entries[0]  # representative entry for paper-level fields

        # Authoritative couplings = the couplings of the actual GT data files.
        true_couplings = {_authoritative_coupling(e) for e in paper_entries}

        paper_report: dict = {
            "arxiv_id": arxiv_id,
            "difficulty": rep.difficulty,
            "num_gt_entries": len(paper_entries),
            "true_couplings": sorted(true_couplings),
        }

        if "error" in result:
            paper_report["status"] = "extraction_failed"
            paper_report["error"] = result["error"]
            comparison_status_counts["extraction_failed"] += 1
            per_paper.append(paper_report)
            continue

        paper_report["status"] = "extracted"
        paper_report["extraction_confidence"] = result.get("extraction_confidence", 0.0)
        paper_report["data_source"] = result.get("data_source")
        paper_report["num_points_extracted"] = result.get("num_points", 0)
        paper_report["elapsed_s"] = result.get("elapsed_s", 0.0)

        # --- Coupling-type classification (against authoritative couplings) ---
        predicted_ct = _normalize_predicted_coupling(result.get("coupling_type"))
        ct_correct = predicted_ct in true_couplings if predicted_ct else False

        coupling_clf.total += 1
        if ct_correct:
            coupling_clf.correct += 1
        else:
            coupling_clf.errors.append({
                "arxiv_id": arxiv_id,
                "predicted": str(predicted_ct),
                "expected": str(sorted(true_couplings)),
            })

        # --- Scalar-label classification (human-verified entries only) ---
        verified = next((e for e in paper_entries if not _is_placeholder_entry(e)), None)
        if verified is not None:
            is_limit_clf.record(arxiv_id, result.get("is_new_limit"), verified.is_new_limit)
            is_projection_clf.record(arxiv_id, result.get("is_projection"), verified.is_projection)
            data_source_clf.record(arxiv_id, result.get("data_source"), verified.data_source_expected)

        paper_report["coupling_type_correct"] = ct_correct
        paper_report["coupling_type_predicted"] = predicted_ct
        paper_report["coupling_type_expected"] = sorted(true_couplings)

        # --- Curve comparison: ONLY against a GT curve of the same coupling ---
        extracted_points = result.get("data_points", [])
        ext_array = (np.array(extracted_points, dtype=float, ndmin=2)
                     if extracted_points else None)

        if predicted_ct is None:
            comparison_status = "no_prediction"
        elif predicted_ct not in true_couplings:
            # The extraction targeted a coupling for which we hold no GT curve.
            comparison_status = "no_comparable_gt"
        elif ext_array is None:
            comparison_status = "no_extracted_points"
        else:
            # The extraction has no convention field; its expected convention is
            # the canonical convention for its predicted coupling type. A GT
            # curve in a DIFFERENT convention (e.g. f_a [GeV] vs normalized, or
            # a large-valued scalar variable vs d_e) is NOT comparable: the
            # residual would be a units gap, not extraction error.
            expected_conv, _ = canonical_convention(predicted_ct)

            # Candidate GT entries: same authoritative coupling AND usable data.
            candidates = []
            has_point_ref = False  # matched GT exists but is a single-mass point
            has_convention_mismatch = False  # same coupling, different convention
            for e in paper_entries:
                if _authoritative_coupling(e) != predicted_ct:
                    continue
                # Skip (and flag) GT curves whose convention differs from the
                # extraction's expected one. None on either side = unknown, so
                # we do not treat it as a mismatch.
                if (expected_conv is not None
                        and e.coupling_convention is not None
                        and e.coupling_convention != expected_conv):
                    has_convention_mismatch = True
                    continue
                gt = e.load_data()
                if gt is None:
                    gt = e.load_reference_data(PROJECT_ROOT)
                if gt is None:
                    continue
                n_pts, n_mass = _usable_gt_stats(gt, predicted_ct)
                if n_mass >= 2:
                    candidates.append((n_mass, e, gt))
                elif n_pts >= 1:
                    has_point_ref = True
            if candidates:
                comparison_status = "compared"
                candidates.sort(key=lambda t: -t[0])  # richest GT curve wins
                _, chosen, gt_data = candidates[0]
            elif has_point_ref:
                # GT is a single-mass prediction/projection — not a curve.
                comparison_status = "gt_point_reference"
            elif has_convention_mismatch:
                # The only same-coupling GT curve(s) use a different convention.
                # Excluded from residuals — this is a units gap, not error.
                comparison_status = "convention_mismatch"
            else:
                comparison_status = "gt_unusable"

        paper_report["comparison_status"] = comparison_status
        comparison_status_counts[comparison_status] += 1

        if comparison_status == "compared":
            paper_report["gt_file"] = chosen.reference_repo_file

            im = compute_interpolation_metrics(
                arxiv_id, ext_array, gt_data, coupling_type=predicted_ct,
            )
            interp_metrics_list.append(im)
            confidences.append(result.get("extraction_confidence", 0.0))
            curve_arxiv_ids.append(arxiv_id)

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
                # Reverse pass (GT interpolated onto extracted masses).
                "num_interpolatable_reverse": im.num_interpolatable_reverse,
                "interpolation_coverage_reverse": im.interpolation_coverage_reverse,
                "median_residual_dex_reverse": im.median_residual_dex_reverse,
                "mean_residual_dex_reverse": im.mean_residual_dex_reverse,
                "p90_residual_dex_reverse": im.p90_residual_dex_reverse,
                "max_residual_dex_reverse": im.max_residual_dex_reverse,
            }

            # Symmetric / 2-D shape metrics: area-between-curves + mass Jaccard.
            sm = compute_symmetric_curve_metrics(
                arxiv_id, ext_array, gt_data, coupling_type=predicted_ct,
            )
            symmetric_metrics_list.append(sm)
            paper_report["symmetric_metrics"] = {
                "area_between_log": sm.area_between_log,
                "overlap_log_mass_width": sm.overlap_log_mass_width,
                "mass_jaccard": sm.mass_jaccard,
                "ext_log_mass_lo": sm.ext_log_mass_lo,
                "ext_log_mass_hi": sm.ext_log_mass_hi,
                "gt_log_mass_lo": sm.gt_log_mass_lo,
                "gt_log_mass_hi": sm.gt_log_mass_hi,
            }

            cm = compute_curve_metrics(arxiv_id, ext_array, gt_data)
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
            paper_report["symmetric_metrics"] = None

        per_paper.append(paper_report)

    # Confidence calibration (uses interpolation metrics)
    calibration = compute_confidence_calibration(
        confidences, interp_metrics_list, curve_arxiv_ids
    )

    # Aggregate interpolation statistics (primary).
    #
    # Two distinct failure modes are kept separate:
    #   (1) zero mass-range overlap -> median residual is inf. This is a
    #       MASS-RANGE failure (extraction spans the wrong masses, often only
    #       1-2 points), NOT a coupling-value error. Folding inf into a mean
    #       would be meaningless, so these are counted, not averaged.
    #   (2) finite residual -> a genuine coupling-value comparison. Summarised
    #       with the MEDIAN across papers (robust); the mean is outlier-driven
    #       and reported only as a secondary number.
    if interp_metrics_list:
        valid = [m for m in interp_metrics_list if m.median_residual_dex < float("inf")]
        n_zero_overlap = len(interp_metrics_list) - len(valid)
        med_resids = [m.median_residual_dex for m in valid]
        aggregate_interp = {
            "n_papers": len(interp_metrics_list),
            "n_zero_overlap": n_zero_overlap,
            "n_finite": len(valid),
            "mean_interpolation_coverage": float(np.mean([m.interpolation_coverage for m in interp_metrics_list])),
            # Robust headline: median across papers of each paper's median residual.
            "median_median_residual_dex": float(np.median(med_resids)) if valid else None,
            "p25_median_residual_dex": float(np.percentile(med_resids, 25)) if valid else None,
            "p75_median_residual_dex": float(np.percentile(med_resids, 75)) if valid else None,
            # Outlier-sensitive; kept for continuity with prior reports.
            "mean_median_residual_dex": float(np.mean(med_resids)) if valid else None,
            "mean_p90_residual_dex": float(np.mean([m.p90_residual_dex for m in valid])) if valid else None,
            "mean_frac_within_0_3dex": float(np.mean([m.frac_within_0_3dex for m in valid])) if valid else None,
            "mean_frac_within_0_5dex": float(np.mean([m.frac_within_0_5dex for m in valid])) if valid else None,
        }
        # Reverse pass aggregate (GT interpolated onto extracted masses).
        # A large gap between forward and reverse residual flags extent/shape
        # mismatch even when the forward residual alone looks good.
        valid_rev = [m for m in interp_metrics_list
                     if m.median_residual_dex_reverse < float("inf")]
        rev_resids = [m.median_residual_dex_reverse for m in valid_rev]
        aggregate_interp["n_finite_reverse"] = len(valid_rev)
        aggregate_interp["mean_interpolation_coverage_reverse"] = float(
            np.mean([m.interpolation_coverage_reverse for m in interp_metrics_list])
        )
        aggregate_interp["median_median_residual_dex_reverse"] = (
            float(np.median(rev_resids)) if valid_rev else None
        )
        aggregate_interp["mean_median_residual_dex_reverse"] = (
            float(np.mean(rev_resids)) if valid_rev else None
        )
    else:
        aggregate_interp = {"n_papers": 0}

    # Aggregate symmetric / 2-D shape metrics (area-between-curves, Jaccard).
    if symmetric_metrics_list:
        areas = [m.area_between_log for m in symmetric_metrics_list
                 if m.area_between_log < float("inf")]
        jaccards = [m.mass_jaccard for m in symmetric_metrics_list]
        aggregate_symmetric = {
            "n_papers": len(symmetric_metrics_list),
            "n_finite_area": len(areas),
            "median_area_between_log": float(np.median(areas)) if areas else None,
            "mean_area_between_log": float(np.mean(areas)) if areas else None,
            "median_mass_jaccard": float(np.median(jaccards)),
            "mean_mass_jaccard": float(np.mean(jaccards)),
        }
    else:
        aggregate_symmetric = {"n_papers": 0}

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

    # Per-difficulty breakdown. NOTE: difficulty is a placeholder label for the
    # repo-sourced pool (almost all "medium"), so this is informational only.
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
            "median_residual_dex": (
                float(np.median([p["interp_metrics"]["median_residual_dex"] for p in valid_interp]))
                if valid_interp else None
            ),
            "mean_frac_within_0_3dex": (
                float(np.mean([p["interp_metrics"]["frac_within_0_3dex"] for p in valid_interp]))
                if valid_interp else None
            ),
        }

    # Per-data-source breakdown (grouped by the pipeline's reported source).
    # Reported with the robust median to match the headline; the zero-overlap
    # count is kept separate so the vision/text signal is not muddied by
    # mass-range failures.
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
            "n_compared": len(with_interp),
            "n_zero_overlap": len(with_interp) - len(valid_interp),
            "median_residual_dex": (
                float(np.median([p["interp_metrics"]["median_residual_dex"] for p in valid_interp]))
                if valid_interp else None
            ),
            "mean_frac_within_0_3dex": (
                float(np.mean([p["interp_metrics"]["frac_within_0_3dex"] for p in valid_interp]))
                if valid_interp else None
            ),
        }

    n_papers = len(by_id)
    return {
        "n_papers": n_papers,
        "classification": {
            "coupling_type": {"accuracy": coupling_clf.accuracy, "total": coupling_clf.total, "errors": coupling_clf.errors},
            "is_new_limit": {"accuracy": is_limit_clf.accuracy, "total": is_limit_clf.total, "errors": is_limit_clf.errors},
            "is_projection": {"accuracy": is_projection_clf.accuracy, "total": is_projection_clf.total, "errors": is_projection_clf.errors},
            "data_source": {"accuracy": data_source_clf.accuracy, "total": data_source_clf.total, "errors": data_source_clf.errors},
        },
        "comparison_coverage": {
            "n_papers": n_papers,
            "n_compared": comparison_status_counts.get("compared", 0),
            "status_counts": dict(comparison_status_counts),
        },
        "interpolation_aggregate": aggregate_interp,
        "symmetric_aggregate": aggregate_symmetric,
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
