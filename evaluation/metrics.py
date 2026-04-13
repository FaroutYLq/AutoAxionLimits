"""Metric computation for extraction evaluation.

All metrics operate on log10 space since limit data spans many orders of magnitude.

Primary metric: interpolation-based comparison.
  1. Build a log-log interpolation function from extracted data points.
  2. Evaluate it at the ground-truth mass values.
  3. Report residuals (log10 coupling error) at each GT point.

This directly answers: "if a physicist uses the extracted curve, how wrong
is it at the masses where we know the true answer?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class ClassificationMetrics:
    """Metrics for categorical fields (coupling_type, is_new_limit, etc.)."""
    total: int = 0
    correct: int = 0
    errors: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def record(self, arxiv_id: str, predicted: object, expected: object):
        self.total += 1
        if predicted == expected:
            self.correct += 1
        else:
            self.errors.append({
                "arxiv_id": arxiv_id,
                "predicted": str(predicted),
                "expected": str(expected),
            })


@dataclass
class CurveMetrics:
    """Metrics comparing extracted vs ground-truth exclusion curves."""
    arxiv_id: str
    num_extracted: int
    num_ground_truth: int
    # Log-space Hausdorff-like distance (max of directed distances)
    hausdorff_log: float
    # Mean directed distance: extracted → ground truth
    mean_dist_ext_to_gt: float
    # Mean directed distance: ground truth → extracted
    mean_dist_gt_to_ext: float
    # Coverage: fraction of GT points within tolerance of an extracted point
    coverage_at_0_5dex: float  # within 0.5 dex (factor ~3)
    coverage_at_1_0dex: float  # within 1.0 dex (factor 10)
    # Mass range overlap (fraction of GT mass range covered)
    mass_range_overlap: float
    # Relative error statistics (on coupling, for mass-matched points)
    median_coupling_log_error: float
    p90_coupling_log_error: float


# ---------------------------------------------------------------------------
# Primary metric: interpolation-based curve comparison
# ---------------------------------------------------------------------------

@dataclass
class InterpolationMetrics:
    """Compare extracted vs ground-truth curves via interpolation.

    Build interp1d from extracted points in log-log space, evaluate at
    ground-truth mass values, report coupling residuals.
    """
    arxiv_id: str
    num_extracted: int
    num_ground_truth: int
    # How many GT points fall inside the extracted mass range (interpolatable)
    num_interpolatable: int
    # Fraction of GT points that are interpolatable
    interpolation_coverage: float
    # Residuals: |log10(g_interp) - log10(g_gt)| at interpolatable GT masses
    residuals_dex: np.ndarray  # raw array, not serialized — use summary stats
    median_residual_dex: float
    mean_residual_dex: float
    p90_residual_dex: float
    max_residual_dex: float
    # Fraction of interpolatable GT points within tolerance
    frac_within_0_1dex: float  # ~25% error
    frac_within_0_3dex: float  # factor-of-2 error
    frac_within_0_5dex: float  # factor-of-3 error
    frac_within_1_0dex: float  # order-of-magnitude error


def _filter_boundary(data: np.ndarray, coupling_ceil: float = 1e-2) -> np.ndarray:
    """Remove boundary-closure sentinel points (coupling >= ceil) and
    non-positive values.  Returns data sorted by mass."""
    mask = (data[:, 0] > 0) & (data[:, 1] > 0) & (data[:, 1] < coupling_ceil)
    filtered = data[mask]
    return filtered[np.argsort(filtered[:, 0])]


def _deduplicate_mass(log_mass: np.ndarray, log_coupling: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """When multiple extracted points share the same log-mass, keep the
    strongest constraint (min coupling in log space)."""
    unique_m, inverse = np.unique(log_mass, return_inverse=True)
    min_c = np.full(len(unique_m), np.inf)
    for i, idx in enumerate(inverse):
        if log_coupling[i] < min_c[idx]:
            min_c[idx] = log_coupling[i]
    return unique_m, min_c


def compute_interpolation_metrics(
    arxiv_id: str,
    extracted: np.ndarray,
    ground_truth: np.ndarray,
    coupling_ceil: float = 1e-2,
) -> InterpolationMetrics:
    """Primary evaluation metric.

    1. Filter boundary-closure points from both arrays.
    2. Build log-log interpolation from extracted data.
    3. Evaluate at GT mass values within the extracted mass range.
    4. Report residual statistics.

    Args:
        extracted: Nx2 (mass_eV, coupling) from pipeline.
        ground_truth: Mx2 (mass_eV, coupling) manually verified.
        coupling_ceil: Points with coupling >= this are treated as boundary
            closure sentinels and filtered out.
    """
    ext = _filter_boundary(extracted, coupling_ceil)
    gt = _filter_boundary(ground_truth, coupling_ceil)

    n_ext = len(ext)
    n_gt = len(gt)

    # Degenerate cases
    _empty = InterpolationMetrics(
        arxiv_id=arxiv_id, num_extracted=n_ext, num_ground_truth=n_gt,
        num_interpolatable=0, interpolation_coverage=0.0,
        residuals_dex=np.array([]),
        median_residual_dex=float("inf"), mean_residual_dex=float("inf"),
        p90_residual_dex=float("inf"), max_residual_dex=float("inf"),
        frac_within_0_1dex=0.0, frac_within_0_3dex=0.0,
        frac_within_0_5dex=0.0, frac_within_1_0dex=0.0,
    )
    if n_ext < 2 or n_gt == 0:
        return _empty

    # Log-space
    log_ext_m = np.log10(ext[:, 0])
    log_ext_c = np.log10(ext[:, 1])
    log_gt_m = np.log10(gt[:, 0])
    log_gt_c = np.log10(gt[:, 1])

    # Deduplicate extracted masses (keep strongest constraint)
    log_ext_m, log_ext_c = _deduplicate_mass(log_ext_m, log_ext_c)

    if len(log_ext_m) < 2:
        return _empty

    # Build interpolation (linear in log-log = power-law in linear)
    interp_fn = interp1d(
        log_ext_m, log_ext_c,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Evaluate at GT masses
    interp_c = interp_fn(log_gt_m)

    # Only keep points inside the extracted mass range (not extrapolated)
    valid = ~np.isnan(interp_c)
    n_interpolatable = int(np.sum(valid))

    if n_interpolatable == 0:
        return _empty

    residuals = np.abs(interp_c[valid] - log_gt_c[valid])

    return InterpolationMetrics(
        arxiv_id=arxiv_id,
        num_extracted=n_ext,
        num_ground_truth=n_gt,
        num_interpolatable=n_interpolatable,
        interpolation_coverage=n_interpolatable / n_gt,
        residuals_dex=residuals,
        median_residual_dex=float(np.median(residuals)),
        mean_residual_dex=float(np.mean(residuals)),
        p90_residual_dex=float(np.percentile(residuals, 90)),
        max_residual_dex=float(np.max(residuals)),
        frac_within_0_1dex=float(np.mean(residuals <= 0.1)),
        frac_within_0_3dex=float(np.mean(residuals <= 0.3)),
        frac_within_0_5dex=float(np.mean(residuals <= 0.5)),
        frac_within_1_0dex=float(np.mean(residuals <= 1.0)),
    )


# ---------------------------------------------------------------------------
# Legacy point-matching metrics (kept as secondary diagnostics)
# ---------------------------------------------------------------------------

def _log_points(data: np.ndarray) -> np.ndarray:
    """Convert Nx2 data to log10 space, filtering out non-positive values."""
    mask = (data[:, 0] > 0) & (data[:, 1] > 0)
    return np.log10(data[mask])


def _directed_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each point in source, find minimum Euclidean distance to any point in target.
    Both arrays are Nx2 in log10 space."""
    if len(source) == 0 or len(target) == 0:
        return np.array([float("inf")])

    # Normalize mass and coupling axes to comparable scales.
    # Mass range is typically wider (20+ decades) vs coupling (10-15 decades).
    # Use the GT spread on each axis for normalization.
    all_pts = np.vstack([source, target])
    spread = np.ptp(all_pts, axis=0)
    spread[spread == 0] = 1.0  # avoid division by zero

    src_norm = source / spread
    tgt_norm = target / spread

    dists = np.empty(len(src_norm))
    for i, pt in enumerate(src_norm):
        d = np.sqrt(np.sum((tgt_norm - pt) ** 2, axis=1))
        dists[i] = np.min(d)

    # Return in original log10 scale (undo normalization approximately)
    return dists * np.mean(spread)


def _mass_matched_coupling_errors(
    extracted: np.ndarray, ground_truth: np.ndarray, mass_tolerance_dex: float = 0.1
) -> np.ndarray:
    """For each GT point, find the closest extracted point in mass, and compute
    the absolute log10 coupling error. Only include points where mass match
    is within tolerance.

    Returns array of |log10(coupling_ext) - log10(coupling_gt)| values.
    """
    if len(extracted) == 0 or len(ground_truth) == 0:
        return np.array([])

    log_ext = _log_points(extracted)
    log_gt = _log_points(ground_truth)

    if len(log_ext) == 0 or len(log_gt) == 0:
        return np.array([])

    errors = []
    for gt_pt in log_gt:
        mass_diffs = np.abs(log_ext[:, 0] - gt_pt[0])
        nearest_idx = np.argmin(mass_diffs)
        if mass_diffs[nearest_idx] <= mass_tolerance_dex:
            err = abs(log_ext[nearest_idx, 1] - gt_pt[1])
            errors.append(err)

    return np.array(errors) if errors else np.array([])


def compute_curve_metrics(
    arxiv_id: str,
    extracted: np.ndarray,
    ground_truth: np.ndarray,
) -> CurveMetrics:
    """Compare extracted data against ground truth.

    Both inputs are Nx2 arrays of (mass_eV, coupling).
    Boundary closure points (coupling=1e0 or similar sentinel values) are
    filtered before comparison.
    """
    # Filter boundary closure points (coupling >= 1e-2 is likely a closure sentinel)
    gt_mask = ground_truth[:, 1] < 1e-2
    ext_mask = extracted[:, 1] < 1e-2
    gt_filtered = ground_truth[gt_mask] if gt_mask.any() else ground_truth
    ext_filtered = extracted[ext_mask] if ext_mask.any() else extracted

    log_ext = _log_points(ext_filtered)
    log_gt = _log_points(gt_filtered)

    n_ext = len(log_ext)
    n_gt = len(log_gt)

    if n_ext == 0 or n_gt == 0:
        return CurveMetrics(
            arxiv_id=arxiv_id,
            num_extracted=n_ext,
            num_ground_truth=n_gt,
            hausdorff_log=float("inf"),
            mean_dist_ext_to_gt=float("inf"),
            mean_dist_gt_to_ext=float("inf"),
            coverage_at_0_5dex=0.0,
            coverage_at_1_0dex=0.0,
            mass_range_overlap=0.0,
            median_coupling_log_error=float("inf"),
            p90_coupling_log_error=float("inf"),
        )

    # Directed distances
    d_ext_to_gt = _directed_distances(log_ext, log_gt)
    d_gt_to_ext = _directed_distances(log_gt, log_ext)

    hausdorff = max(np.max(d_ext_to_gt), np.max(d_gt_to_ext))

    # Coverage: fraction of GT points with a nearby extracted point
    coverage_05 = np.mean(d_gt_to_ext <= 0.5)
    coverage_10 = np.mean(d_gt_to_ext <= 1.0)

    # Mass range overlap
    gt_mass_range = (log_gt[:, 0].min(), log_gt[:, 0].max())
    ext_mass_range = (log_ext[:, 0].min(), log_ext[:, 0].max())
    overlap_lo = max(gt_mass_range[0], ext_mass_range[0])
    overlap_hi = min(gt_mass_range[1], ext_mass_range[1])
    gt_span = gt_mass_range[1] - gt_mass_range[0]
    mass_overlap = max(0, overlap_hi - overlap_lo) / gt_span if gt_span > 0 else 1.0

    # Mass-matched coupling errors
    coupling_errors = _mass_matched_coupling_errors(ext_filtered, gt_filtered)
    if len(coupling_errors) > 0:
        median_err = float(np.median(coupling_errors))
        p90_err = float(np.percentile(coupling_errors, 90))
    else:
        median_err = float("inf")
        p90_err = float("inf")

    return CurveMetrics(
        arxiv_id=arxiv_id,
        num_extracted=n_ext,
        num_ground_truth=n_gt,
        hausdorff_log=float(hausdorff),
        mean_dist_ext_to_gt=float(np.mean(d_ext_to_gt)),
        mean_dist_gt_to_ext=float(np.mean(d_gt_to_ext)),
        coverage_at_0_5dex=float(coverage_05),
        coverage_at_1_0dex=float(coverage_10),
        mass_range_overlap=float(mass_overlap),
        median_coupling_log_error=median_err,
        p90_coupling_log_error=p90_err,
    )


@dataclass
class ConfidenceBin:
    """One bin in the confidence calibration curve."""
    bin_lo: float
    bin_hi: float
    n_papers: int
    mean_confidence: float
    # Fraction of papers in this bin with "acceptable" curve quality
    # (coverage_at_1_0dex > 0.8 AND median_coupling_log_error < 0.5)
    actual_accuracy: float
    paper_ids: list[str]


def compute_confidence_calibration(
    confidences: list[float],
    interp_metrics: list[InterpolationMetrics],
    arxiv_ids: list[str],
    n_bins: int = 5,
    accuracy_threshold_residual: float = 0.3,
    accuracy_threshold_coverage: float = 0.5,
) -> list[ConfidenceBin]:
    """Bin papers by extraction_confidence and compute actual accuracy per bin.

    A paper is "accurate" if:
      - median interpolation residual < threshold (default 0.3 dex ≈ factor 2)
      - interpolation coverage > threshold (default 50% of GT points)
    """
    if not confidences:
        return []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        indices = [
            j for j, c in enumerate(confidences)
            if lo <= c < hi or (i == n_bins - 1 and c == hi)
        ]

        if not indices:
            bins.append(ConfidenceBin(
                bin_lo=lo, bin_hi=hi, n_papers=0,
                mean_confidence=0.0, actual_accuracy=0.0, paper_ids=[],
            ))
            continue

        bin_confs = [confidences[j] for j in indices]
        bin_ids = [arxiv_ids[j] for j in indices]
        bin_metrics = [interp_metrics[j] for j in indices]

        n_accurate = sum(
            1 for m in bin_metrics
            if m.median_residual_dex < accuracy_threshold_residual
            and m.interpolation_coverage >= accuracy_threshold_coverage
        )

        bins.append(ConfidenceBin(
            bin_lo=lo,
            bin_hi=hi,
            n_papers=len(indices),
            mean_confidence=float(np.mean(bin_confs)),
            actual_accuracy=n_accurate / len(indices),
            paper_ids=bin_ids,
        ))

    return bins
