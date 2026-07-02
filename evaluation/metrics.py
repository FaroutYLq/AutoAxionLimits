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

# NumPy 2.0 renamed `np.trapz` -> `np.trapezoid` and removed the old name.
# Resolve once so the metric works on both NumPy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


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
# Symmetric / 2-D curve-distance + mass-range-agreement metrics
# ---------------------------------------------------------------------------

@dataclass
class SymmetricCurveMetrics:
    """Symmetric, shape-aware comparison of two curves in log-log space.

    These are complementary to the (asymmetric, 1-D, vertical-only)
    interpolation residual:

    - ``area_between_log``: area between the two curves over their OVERLAPPING
      log-mass range, normalised by the overlap width (so it is a single
      shape+offset number in dex). Both a vertical offset and a horizontal
      (mass) shift inflate it, so it penalises a pure mass shift that the
      vertical residual can miss.
    - ``mass_jaccard``: Jaccard index of the two log-mass intervals
      (extracted vs GT). 1.0 = identical extent; small = an extraction that
      runs well past (or well short of) the GT range. Reported SEPARATELY
      from interpolation density/coverage.
    """
    arxiv_id: str
    num_extracted: int
    num_ground_truth: int
    # Normalised area between curves in log-coupling-dex over the overlap range.
    area_between_log: float
    # Width (in dex of log10 mass) of the overlapping mass range.
    overlap_log_mass_width: float
    # Jaccard index of the two log-mass intervals.
    mass_jaccard: float
    # Extracted / GT log-mass interval endpoints (for diagnostics).
    ext_log_mass_lo: float
    ext_log_mass_hi: float
    gt_log_mass_lo: float
    gt_log_mass_hi: float


def _log_mass_interval(log_m: np.ndarray) -> tuple[float, float]:
    return float(np.min(log_m)), float(np.max(log_m))


def _interval_jaccard(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Jaccard index of two closed intervals [a0,a1], [b0,b1]."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0:
        # Both intervals are degenerate (single point). Identical -> 1, else 0.
        return 1.0 if a[0] == b[0] else 0.0
    return inter / union


def compute_symmetric_curve_metrics(
    arxiv_id: str,
    extracted: np.ndarray,
    ground_truth: np.ndarray,
    coupling_ceil: float = 1e-2,
    coupling_type: str | None = None,
    n_samples: int = 200,
) -> SymmetricCurveMetrics:
    """Area-between-curves (log-log) + mass-range Jaccard.

    Mirrors the boundary filtering / log-log interpolation of
    ``compute_interpolation_metrics`` so the two metrics see the same points.

    The area is computed by sampling both curves on a common log-mass grid
    spanning their overlap, taking |Δ log10 coupling| and integrating
    (trapezoid), then dividing by the overlap width so the result is in dex,
    independent of how wide the overlap happens to be.
    """
    if coupling_type and coupling_type in _COUPLING_CEILINGS:
        coupling_ceil = _COUPLING_CEILINGS[coupling_type]

    ext = _filter_boundary(extracted, coupling_ceil)
    gt = _filter_boundary(ground_truth, coupling_ceil)
    n_ext = len(ext)
    n_gt = len(gt)

    _empty = SymmetricCurveMetrics(
        arxiv_id=arxiv_id, num_extracted=n_ext, num_ground_truth=n_gt,
        area_between_log=float("inf"), overlap_log_mass_width=0.0,
        mass_jaccard=0.0,
        ext_log_mass_lo=float("nan"), ext_log_mass_hi=float("nan"),
        gt_log_mass_lo=float("nan"), gt_log_mass_hi=float("nan"),
    )
    if n_ext < 2 or n_gt < 2:
        return _empty

    log_ext_m, log_ext_c = _deduplicate_mass(np.log10(ext[:, 0]), np.log10(ext[:, 1]))
    log_gt_m, log_gt_c = _deduplicate_mass(np.log10(gt[:, 0]), np.log10(gt[:, 1]))
    if len(log_ext_m) < 2 or len(log_gt_m) < 2:
        return _empty

    ext_iv = _log_mass_interval(log_ext_m)
    gt_iv = _log_mass_interval(log_gt_m)
    jaccard = _interval_jaccard(ext_iv, gt_iv)

    overlap_lo = max(ext_iv[0], gt_iv[0])
    overlap_hi = min(ext_iv[1], gt_iv[1])
    overlap_w = overlap_hi - overlap_lo
    if overlap_w <= 0:
        # No mass overlap: area is undefined (no common support).
        return SymmetricCurveMetrics(
            arxiv_id=arxiv_id, num_extracted=n_ext, num_ground_truth=n_gt,
            area_between_log=float("inf"), overlap_log_mass_width=0.0,
            mass_jaccard=jaccard,
            ext_log_mass_lo=ext_iv[0], ext_log_mass_hi=ext_iv[1],
            gt_log_mass_lo=gt_iv[0], gt_log_mass_hi=gt_iv[1],
        )

    ext_fn = interp1d(log_ext_m, log_ext_c, kind="linear",
                      bounds_error=False, fill_value=np.nan)
    gt_fn = interp1d(log_gt_m, log_gt_c, kind="linear",
                     bounds_error=False, fill_value=np.nan)

    grid = np.linspace(overlap_lo, overlap_hi, n_samples)
    diff = np.abs(ext_fn(grid) - gt_fn(grid))
    valid = ~np.isnan(diff)
    if not np.any(valid):
        area_norm = float("inf")
    else:
        g = grid[valid]
        d = diff[valid]
        # Normalised area = (∫|Δ| d(log m)) / width, i.e. mean |Δ| in dex.
        area = float(_trapezoid(d, g))
        area_norm = area / overlap_w

    return SymmetricCurveMetrics(
        arxiv_id=arxiv_id, num_extracted=n_ext, num_ground_truth=n_gt,
        area_between_log=area_norm, overlap_log_mass_width=float(overlap_w),
        mass_jaccard=float(jaccard),
        ext_log_mass_lo=ext_iv[0], ext_log_mass_hi=ext_iv[1],
        gt_log_mass_lo=gt_iv[0], gt_log_mass_hi=gt_iv[1],
    )


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
    # --- Reverse pass: interpolate the GT curve onto the EXTRACTED masses. ---
    # This is the mirror of the forward pass. The forward pass alone cannot see
    # an extraction that runs PAST the GT mass range (over-claiming): those
    # extracted masses simply have no GT point to be scored against. Evaluating
    # the GT interpolation at the extracted masses surfaces that error as a
    # large reverse residual / coverage gap. A large forward-vs-reverse
    # asymmetry flags an extent/shape mismatch.
    # How many extracted points fall inside the GT mass range (interpolatable)
    num_interpolatable_reverse: int = 0
    # Fraction of extracted points that are interpolatable against GT
    interpolation_coverage_reverse: float = 0.0
    median_residual_dex_reverse: float = float("inf")
    mean_residual_dex_reverse: float = float("inf")
    p90_residual_dex_reverse: float = float("inf")
    max_residual_dex_reverse: float = float("inf")
    # Raw reverse residual array (not serialized) — lets the caller promote the
    # reverse pass to the paper's effective score when the forward pass has no
    # interpolatable GT vertex (vertex-sparse GT / single-point extractions).
    residuals_dex_reverse: np.ndarray = field(default_factory=lambda: np.array([]))


# Coupling ceilings per type: points with coupling >= ceiling are treated as
# boundary-closure sentinels (vertices added only to close a filled polygon for
# plotting) and dropped. The ceiling must sit ABOVE the real data and BELOW the
# closure value, so it is calibrated to each type's actual y-axis scale:
#   - Standard couplings (g_aγγ, g_ae, ε, …): real data << 1e-2, closure at 1e0
#     or 1e99 -> default 1e-2 / explicit 1e0.
#   - AxionMass (m_a vs f_a plane files): the y-axis is a small normalised
#     coupling (~1e-24..4e-3), the closure sentinel is exactly 1.0. The previous
#     1e6 ceiling KEPT those 1.0 sentinels and corrupted the interpolation.
#   - Scalar (dilaton-like) couplings: real curve data spans ~1e0..1e17 with
#     closure sentinels at 1e20/1e30, so the ceiling is raised to 1e19. A 1e0
#     ceiling discarded the entire real curve.
_COUPLING_CEILINGS = {
    "AxionMass": 1e0,
    "DarkPhoton": 1e0,
    "AxionCPV": 1e0,
    "MonopoleDipole": 1e0,
    "ScalarPhoton": 1e19,
    "ScalarElectron": 1e19,
    "ScalarBaryon": 1e19,
    "ScalarNucleon": 1e19,
    "VectorBL": 1e0,
}
_DEFAULT_COUPLING_CEIL = 1e-2


def _ceil_for(coupling_type: str | None) -> float:
    """Boundary-closure ceiling for a coupling type (shared lookup)."""
    return _COUPLING_CEILINGS.get(coupling_type, _DEFAULT_COUPLING_CEIL)


def _filter_boundary(data: np.ndarray, coupling_ceil: float = 1e-2) -> np.ndarray:
    """Remove boundary-closure sentinel points (coupling >= ceil) and
    non-positive values.  Returns data sorted by mass."""
    mask = (data[:, 0] > 0) & (data[:, 1] > 0) & (data[:, 1] < coupling_ceil)
    filtered = data[mask]
    return filtered[np.argsort(filtered[:, 0])]


def _filter_boundary_keep_order(data: np.ndarray, coupling_ceil: float) -> np.ndarray:
    """Like ``_filter_boundary`` but preserves file order — needed to detect
    closed-contour (multi-branch) curves, which sorting would scramble."""
    mask = (data[:, 0] > 0) & (data[:, 1] > 0) & (data[:, 1] < coupling_ceil)
    return data[mask]


# Canonical wide mass window for mass-independent (flat) bounds (Lever E, #587).
# A bound the extractor reports as mass-independent records no usable mass
# (every row has mass <= 0, its sentinel). Encoded that way it survives no
# boundary filter (``_filter_boundary`` drops mass <= 0), so it can never
# overlap a GT curve and is spuriously scored zero_overlap even when its
# coupling matches GT (e.g. 2007.03694 RedGiants: g_ae ~ 1.3e-13, matches GT
# to ~0.04 dex but mass=0). Moved here from subset_compare.py (post-full346
# Phase 1c) so the full-pool scorer applies it too.
_FLAT_BOUND_MASS_LO = 1e-30
_FLAT_BOUND_MASS_HI = 1e4


def _expand_mass_independent(arr: np.ndarray) -> np.ndarray:
    """Expand a mass-independent extraction to a horizontal segment.

    If every row has mass <= 0 (the extractor's mass-independent sentinel) but
    at least one coupling is positive, return a 2-row segment at the median
    positive coupling spanning ``[_FLAT_BOUND_MASS_LO, _FLAT_BOUND_MASS_HI]``
    so the comparator can score the coupling against the GT at the GT's own
    masses. Otherwise return ``arr`` unchanged.

    A flat bound is physically valid at every mass, so this is faithful, not a
    GT edit: only the *extraction* is reshaped, the GT curve is untouched.
    """
    if arr is None or len(arr) == 0:
        return arr
    masses = arr[:, 0]
    if np.any(masses > 0):
        return arr  # usable mass info present; not a flat-bound sentinel
    valid = arr[:, 1][arr[:, 1] > 0]
    if valid.size == 0:
        return arr
    g = float(np.median(valid))
    return np.array([[_FLAT_BOUND_MASS_LO, g], [_FLAT_BOUND_MASS_HI, g]], dtype=float)


def _lower_envelope_if_contour(data: np.ndarray) -> np.ndarray:
    """Reduce a closed-contour / multivalued curve to its lower envelope.

    Some GT curves are closed exclusion BANDS (e.g. AxionProton/SN1987A.txt:
    a free-streaming lower edge AND a trapping upper edge traced as one
    polygon). Interpolating such a polygon as a single-valued curve mixes the
    branches and produces multi-dex phantom residuals (2306.01048: 3.51 dex
    raw vs 0.02 dex against the lower branch). Detection: after dropping
    zero-length steps, the log-mass sequence reverses direction (a monotonic
    curve, however jittery in y, never does). The input must be in FILE ORDER
    (use ``_filter_boundary_keep_order``); the output is sorted by mass with
    one (strongest) coupling per distinct mass.

    For monotonic input this is equivalent to a plain sort, so applying it
    unconditionally to already-single-valued curves is a no-op.
    """
    if data is None or len(data) < 3:
        return data[np.argsort(data[:, 0])] if data is not None and len(data) else data
    lm = np.log10(data[:, 0])
    lc = np.log10(data[:, 1])
    # Split into monotonic-in-mass branches at direction reversals.
    cuts = [0]
    cur = 0
    for i in range(1, len(lm)):
        dd = lm[i] - lm[i - 1]
        s = 1 if dd > 1e-12 else (-1 if dd < -1e-12 else 0)
        if s == 0:
            continue
        if cur == 0:
            cur = s
        elif s != cur:
            cuts.append(i - 1)  # the pivot vertex belongs to both branches
            cur = s
    cuts.append(len(lm) - 1)
    if len(cuts) <= 2:
        return data[np.argsort(data[:, 0])]  # monotonic: not a contour

    grid = np.unique(lm)
    env = np.full(grid.shape, np.inf)
    for k in range(len(cuts) - 1):
        a, b = cuts[k], cuts[k + 1] + 1
        bm, bc = _deduplicate_mass(lm[a:b], lc[a:b])
        if len(bm) >= 2:
            fn = interp1d(bm, bc, kind="linear", bounds_error=False,
                          fill_value=np.nan)
            vals = fn(grid)
        else:
            vals = np.full(grid.shape, np.nan)
            j = int(np.argmin(np.abs(grid - bm[0])))
            vals[j] = bc[0]
        env = np.fmin(env, vals)  # fmin ignores NaN
    ok = np.isfinite(env)
    return np.column_stack([10.0 ** grid[ok], 10.0 ** env[ok]])


# ---------------------------------------------------------------------------
# Single-point comparison (#612, moved from subset_compare.py in Phase 1c).
# Many results are a SINGLE limit value at one operating mass (e.g. 1806.00310
# QUAX 2018 notch; the DM-Radio Pathfinder point). Curve interpolation cannot
# run on them, so without this they are discarded as zero_overlap even when
# the extracted coupling is correct.
# ---------------------------------------------------------------------------
_SINGLE_POINT_MASS_TOL_DEX = 0.3   # ~factor 2 in mass = the same experimental point
_SINGLE_POINT_MAX_EXT_MASSES = 3   # fallback only for sparse single-value extractions


def _residuals_at(curve: np.ndarray, reference: np.ndarray,
                  tol_dex: float) -> np.ndarray | None:
    """|Δ log10 coupling| of ``curve`` evaluated at each ``reference`` point.

    Both Nx2 (mass, coupling), already boundary-filtered & positive. The curve
    is interpolated in log-log when it has >= 2 distinct masses; for reference
    masses outside the curve's range (or a single-point curve) the nearest
    curve point is used, but only if it is within ``tol_dex`` of the reference
    mass (the same operating mass). Returns the residuals at the matched
    reference points, or ``None`` if none match.
    """
    cm, cc = _deduplicate_mass(np.log10(curve[:, 0]), np.log10(curve[:, 1]))
    # Dedup the reference to one (strongest) point per operating mass:
    # duplicate rows at the same mass are the same point, not extra coverage.
    rm, rc = _deduplicate_mass(np.log10(reference[:, 0]), np.log10(reference[:, 1]))
    interp = None
    if len(cm) >= 2:
        interp = interp1d(cm, cc, kind="linear", bounds_error=False,
                          fill_value=np.nan)
    out = []
    for k in range(len(rm)):
        val = float(interp(rm[k])) if interp is not None else float("nan")
        if not np.isfinite(val):
            j = int(np.argmin(np.abs(cm - rm[k])))
            if abs(cm[j] - rm[k]) <= tol_dex:
                val = cc[j]
        if np.isfinite(val):
            out.append(abs(val - rc[k]))
    return np.array(out) if out else None


def single_point_compare(curve: np.ndarray, reference: np.ndarray,
                         coupling_type: str | None = None,
                         require_sparse_ref: bool = False):
    """Single-point residual: evaluate ``curve`` at the ``reference`` masses.

    ``reference`` is the side with the trustworthy operating mass(es) — the
    single GT point (single-mass GT), or the sparse single-value extraction
    (the fallback when a GT curve cannot be interpolated against a 1-point
    read). Returns ``(median_resid, n_matched, coverage, residuals)`` over the
    reference points, or ``None`` if nothing matches.

    ``require_sparse_ref`` refuses to score a rich multi-point reference this
    way — used in the curve fallback so a genuine wrong-window curve failure
    is not masked by one lucky near-mass extracted point.
    """
    ceil = _ceil_for(coupling_type)
    c = _filter_boundary(curve, ceil)
    r = _filter_boundary(reference, ceil)
    if len(c) == 0 or len(r) == 0:
        return None
    n_ref_masses = int(np.unique(r[:, 0]).size)
    if require_sparse_ref and n_ref_masses > _SINGLE_POINT_MAX_EXT_MASSES:
        return None
    res = _residuals_at(c, r, _SINGLE_POINT_MASS_TOL_DEX)
    if res is None:
        return None
    return float(np.median(res)), len(res), len(res) / n_ref_masses, res


def _deduplicate_mass(log_mass: np.ndarray, log_coupling: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """When multiple extracted points share the same log-mass, keep the
    strongest constraint (min coupling in log space)."""
    unique_m, inverse = np.unique(log_mass, return_inverse=True)
    min_c = np.full(len(unique_m), np.inf)
    for i, idx in enumerate(inverse):
        if log_coupling[i] < min_c[idx]:
            min_c[idx] = log_coupling[i]
    return unique_m, min_c


def _interp_residuals(
    src_m: np.ndarray, src_c: np.ndarray, tgt_m: np.ndarray, tgt_c: np.ndarray
) -> np.ndarray | None:
    """Build a log-log interp1d from (src_m, src_c) and evaluate it at tgt_m,
    returning |residual| against tgt_c for the in-range target masses only.

    All inputs are already in log10 space. Returns None if the source has <2
    distinct masses or no target mass falls inside the source range.
    """
    src_m, src_c = _deduplicate_mass(src_m, src_c)
    if len(src_m) < 2:
        return None
    interp_fn = interp1d(
        src_m, src_c, kind="linear", bounds_error=False, fill_value=np.nan,
    )
    interp_c = interp_fn(tgt_m)
    valid = ~np.isnan(interp_c)
    if not np.any(valid):
        return None
    return np.abs(interp_c[valid] - tgt_c[valid])


def compute_interpolation_metrics(
    arxiv_id: str,
    extracted: np.ndarray,
    ground_truth: np.ndarray,
    coupling_ceil: float = 1e-2,
    coupling_type: str | None = None,
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
        coupling_type: If provided, override coupling_ceil with a
            type-specific ceiling from _COUPLING_CEILINGS.
    """
    # Use coupling-type-specific ceiling if available
    if coupling_type and coupling_type in _COUPLING_CEILINGS:
        coupling_ceil = _COUPLING_CEILINGS[coupling_type]

    # Flat mass-independent bounds are expanded to a horizontal segment before
    # filtering (their mass<=0 sentinel rows would otherwise all be dropped).
    extracted = _expand_mass_independent(extracted)

    # Filter sentinels in FILE ORDER, then reduce closed-contour (multivalued)
    # curves to their lower envelope; monotonic curves come back sorted,
    # unchanged. See _lower_envelope_if_contour (2306.01048).
    ext = _lower_envelope_if_contour(
        _filter_boundary_keep_order(extracted, coupling_ceil))
    gt = _lower_envelope_if_contour(
        _filter_boundary_keep_order(ground_truth, coupling_ceil))

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
    if n_ext == 0 or n_gt == 0:
        return _empty

    # Log-space
    log_ext_m = np.log10(ext[:, 0])
    log_ext_c = np.log10(ext[:, 1])
    log_gt_m = np.log10(gt[:, 0])
    log_gt_c = np.log10(gt[:, 1])

    # Deduplicate extracted masses (keep strongest constraint)
    log_ext_m, log_ext_c = _deduplicate_mass(log_ext_m, log_ext_c)

    # --- Forward pass: interp from extraction, evaluate at GT masses. ---
    # Needs >= 2 distinct extracted masses; a single-point extraction is
    # scored by the reverse pass below instead of being auto-failed (#612 /
    # post-full346 Phase 1c — n_ext==1 used to early-return _empty here).
    residuals = None
    if len(log_ext_m) >= 2:
        interp_fn = interp1d(
            log_ext_m, log_ext_c,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_c = interp_fn(log_gt_m)
        # Only keep points inside the extracted mass range (no extrapolation)
        valid = ~np.isnan(interp_c)
        if np.any(valid):
            residuals = np.abs(interp_c[valid] - log_gt_c[valid])
    n_interpolatable = int(len(residuals)) if residuals is not None else 0

    # --- Reverse pass: build interp from GT, evaluate at extracted masses. ---
    # Computed even when the forward pass is empty (vertex-sparse GT, e.g. a
    # 2-vertex flat segment, or a 1-point extraction inside the GT range) so
    # the caller can promote it to the effective score instead of reporting a
    # spurious zero_overlap (1406.6053, 0910.5914, 1906.08814, 2102.08764).
    residuals_rev = _interp_residuals(log_gt_m, log_gt_c, log_ext_m, log_ext_c)
    if residuals_rev is None or len(residuals_rev) == 0:
        residuals_rev = np.array([])
        n_interp_rev = 0
        cov_rev = 0.0
        med_rev = mean_rev = p90_rev = max_rev = float("inf")
    else:
        n_interp_rev = int(len(residuals_rev))
        cov_rev = n_interp_rev / len(log_ext_m)
        med_rev = float(np.median(residuals_rev))
        mean_rev = float(np.mean(residuals_rev))
        p90_rev = float(np.percentile(residuals_rev, 90))
        max_rev = float(np.max(residuals_rev))

    if n_interpolatable == 0:
        return InterpolationMetrics(
            arxiv_id=arxiv_id, num_extracted=n_ext, num_ground_truth=n_gt,
            num_interpolatable=0, interpolation_coverage=0.0,
            residuals_dex=np.array([]),
            median_residual_dex=float("inf"), mean_residual_dex=float("inf"),
            p90_residual_dex=float("inf"), max_residual_dex=float("inf"),
            frac_within_0_1dex=0.0, frac_within_0_3dex=0.0,
            frac_within_0_5dex=0.0, frac_within_1_0dex=0.0,
            num_interpolatable_reverse=n_interp_rev,
            interpolation_coverage_reverse=cov_rev,
            median_residual_dex_reverse=med_rev,
            mean_residual_dex_reverse=mean_rev,
            p90_residual_dex_reverse=p90_rev,
            max_residual_dex_reverse=max_rev,
            residuals_dex_reverse=residuals_rev,
        )

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
        num_interpolatable_reverse=n_interp_rev,
        interpolation_coverage_reverse=cov_rev,
        median_residual_dex_reverse=med_rev,
        mean_residual_dex_reverse=mean_rev,
        p90_residual_dex_reverse=p90_rev,
        max_residual_dex_reverse=max_rev,
        residuals_dex_reverse=residuals_rev,
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


# ---------------------------------------------------------------------------
# Confidence-calibration accuracy threshold (issue #542).
#
# A paper counts as "accurate" iff its median interpolation residual is below
# this threshold (in dex). It is pinned to the *extraction noise floor*, NOT
# relaxed to the upstream-digitization floor:
#
#   * Extraction noise floor (BINDING): ~0.32 dex. This is the 90th-percentile
#     of the per-paper median-residual standard deviation across repeated LLM
#     extraction runs of the same paper (PR #545). Run-to-run LLM
#     non-determinism alone moves a paper's median residual by up to ~0.32 dex,
#     so demanding accuracy tighter than that would penalise extractions for
#     irreducible sampling noise. THIS sets the threshold.
#
#   * Digitization floor (NOT binding): ~0.034 dex for TABLE/text-sourced
#     papers (PR #558, truly-independent gold-vs-repo, N=10). The repo ground
#     truth is faithful to papers that publish numbers — it is NOT the ~0.5 dex
#     floor the original #542 premise assumed. That premise ("0.3 dex is below
#     the digitization floor, so the yardstick is too strict") is therefore
#     FALSIFIED: the old 0.3 dex was roughly right, but for the wrong reason.
#
# Consequence: any residual overconfidence gap measured against this threshold
# is REAL extractor overconfidence, not a yardstick artifact.
#
# CAVEAT: the 0.034 dex digitization floor is measured only for table/text
# sources. For FIGURE-ONLY papers the upstream figure-digitization error is
# UNMEASURED (the gold_vision tier bounds the *combined* error at ~1.13 dex but
# does not isolate cajohare's own figure-digitization component), so this is
# not a universal floor.
# ---------------------------------------------------------------------------
NOISE_FLOOR_RESIDUAL_DEX: float = 0.32

# Tau thresholds (dex) for the continuous proper-scoring view: empirical
# P(median residual < tau) per confidence bin. 0.1 = factor ~1.3 (stringent),
# 0.32 = the noise floor, 0.5 = factor ~3, 1.0 = order of magnitude.
CONTINUOUS_TAUS_DEX: tuple[float, ...] = (0.1, 0.32, 0.5, 1.0)


@dataclass
class ConfidenceBin:
    """One bin in the confidence calibration curve.

    ``actual_accuracy`` is the fraction of papers in this bin whose median
    interpolation residual is below ``NOISE_FLOOR_RESIDUAL_DEX`` (the binding
    run-to-run LLM noise floor) AND whose interpolation coverage is >= 50%.

    The continuous fields give the *distribution* of residuals in the bin (not
    just a pass rate), and the empirical P(residual < tau) for several tau:
      - ``median_residual_dex``: median over finite-residual papers in the bin.
      - ``p25_residual_dex`` / ``p75_residual_dex``: IQR over the same.
      - ``frac_within_tau``: {tau: fraction of papers with median residual < tau}.
    """
    bin_lo: float
    bin_hi: float
    n_papers: int
    mean_confidence: float
    actual_accuracy: float
    paper_ids: list[str]
    # Continuous calibration view (issue #542).
    median_residual_dex: Optional[float] = None
    p25_residual_dex: Optional[float] = None
    p75_residual_dex: Optional[float] = None
    n_finite: int = 0
    frac_within_tau: dict[float, float] = field(default_factory=dict)


def compute_confidence_calibration(
    confidences: list[float],
    interp_metrics: list[InterpolationMetrics],
    arxiv_ids: list[str],
    n_bins: int = 5,
    accuracy_threshold_residual: float = NOISE_FLOOR_RESIDUAL_DEX,
    accuracy_threshold_coverage: float = 0.5,
    taus: tuple[float, ...] = CONTINUOUS_TAUS_DEX,
) -> list[ConfidenceBin]:
    """Bin papers by extraction_confidence and compute calibration per bin.

    A paper is "accurate" if:
      - median interpolation residual < ``accuracy_threshold_residual``
        (default ``NOISE_FLOOR_RESIDUAL_DEX`` = 0.32 dex, the run-to-run LLM
        extraction noise floor from #545 — NOT the 0.034 dex digitization floor
        from #558; see the module-level note for why the threshold tracks
        noise, not digitization).
      - interpolation coverage >= ``accuracy_threshold_coverage``
        (default 50% of GT points). Coverage is retained because a paper that
        nails the coupling at only a sliver of the mass range is not a usable
        extraction — a low residual on 1-2 in-range points should not count as
        "accurate". It is a curve-quality gate, orthogonal to the residual
        floor, so both conditions are kept.

    In addition to the pass/fail ``actual_accuracy``, each bin reports the
    *continuous* residual distribution (median + IQR) and the empirical
    P(residual < tau) for each tau in ``taus``, so a bin shows the real residual
    distribution rather than only a thresholded rate.
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
                frac_within_tau={t: 0.0 for t in taus},
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

        # Continuous view: residual distribution over finite-residual papers
        # (zero-overlap papers have an infinite median residual and are
        # excluded from the distribution summary but still counted in n_papers).
        finite_resids = [
            m.median_residual_dex for m in bin_metrics
            if math.isfinite(m.median_residual_dex)
        ]
        if finite_resids:
            med_r = float(np.median(finite_resids))
            p25_r = float(np.percentile(finite_resids, 25))
            p75_r = float(np.percentile(finite_resids, 75))
        else:
            med_r = p25_r = p75_r = None

        # Empirical P(median residual < tau) over ALL papers in the bin
        # (an infinite residual correctly fails every finite tau).
        frac_within_tau = {
            t: float(np.mean([m.median_residual_dex < t for m in bin_metrics]))
            for t in taus
        }

        bins.append(ConfidenceBin(
            bin_lo=lo,
            bin_hi=hi,
            n_papers=len(indices),
            mean_confidence=float(np.mean(bin_confs)),
            actual_accuracy=n_accurate / len(indices),
            paper_ids=bin_ids,
            median_residual_dex=med_r,
            p25_residual_dex=p25_r,
            p75_residual_dex=p75_r,
            n_finite=len(finite_resids),
            frac_within_tau=frac_within_tau,
        ))

    return bins


# ---------------------------------------------------------------------------
# Synthetic self-check (issue #541). Run: python -m evaluation.metrics
# Not a pytest suite (that is issue #544); just guards the new metric maths.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A simple log-log power-law curve over a decade of mass.
    def _curve(m_lo_dex, m_hi_dex, c0_dex, slope, n=25):
        m = np.logspace(m_lo_dex, m_hi_dex, n)
        c = 10.0 ** (c0_dex + slope * (np.log10(m) - m_lo_dex))
        return np.column_stack([m, c])

    gt = _curve(-6, -5, -10, -0.5)

    # 1. Identical curves -> reverse residual 0, area 0, Jaccard 1.
    im = compute_interpolation_metrics("id", gt.copy(), gt.copy(),
                                       coupling_type="AxionPhoton")
    sm = compute_symmetric_curve_metrics("id", gt.copy(), gt.copy(),
                                         coupling_type="AxionPhoton")
    assert im.median_residual_dex < 1e-9, im.median_residual_dex
    assert im.median_residual_dex_reverse < 1e-9, im.median_residual_dex_reverse
    assert sm.area_between_log < 1e-9, sm.area_between_log
    assert abs(sm.mass_jaccard - 1.0) < 1e-9, sm.mass_jaccard
    print(f"[identical]   fwd={im.median_residual_dex:.3g} "
          f"rev={im.median_residual_dex_reverse:.3g} "
          f"area={sm.area_between_log:.3g} jaccard={sm.mass_jaccard:.3f}")

    # 2. Extraction extends WELL past the GT range (over-claiming):
    #    GT spans [-6,-5]; extraction spans [-6,-2] (same shape on the overlap).
    over = _curve(-6, -2, -10, -0.5, n=60)
    im2 = compute_interpolation_metrics("over", over, gt.copy(),
                                        coupling_type="AxionPhoton")
    sm2 = compute_symmetric_curve_metrics("over", over, gt.copy(),
                                          coupling_type="AxionPhoton")
    # Forward (GT masses onto extraction) is fine; reverse (extracted masses
    # onto GT) has many points outside the GT range -> low reverse coverage,
    # i.e. strong forward/reverse coverage asymmetry. Jaccard is low (1 of 4
    # decades overlap -> ~0.25).
    assert sm2.mass_jaccard < 0.4, sm2.mass_jaccard
    assert im2.interpolation_coverage_reverse < im2.interpolation_coverage, (
        im2.interpolation_coverage_reverse, im2.interpolation_coverage)
    print(f"[over-claim]  fwd_cov={im2.interpolation_coverage:.3f} "
          f"rev_cov={im2.interpolation_coverage_reverse:.3f} "
          f"area={sm2.area_between_log:.3g} jaccard={sm2.mass_jaccard:.3f}")

    # 3. Pure horizontal (mass) shift: same coupling values, masses shifted by
    #    half a decade. Vertical residual at matched masses looks ok-ish, but
    #    the area-between-curves and Jaccard both penalise it.
    shifted = gt.copy()
    shifted[:, 0] *= 10 ** 0.5  # shift mass by +0.5 dex
    sm3 = compute_symmetric_curve_metrics("shift", shifted, gt.copy(),
                                          coupling_type="AxionPhoton")
    assert sm3.area_between_log > 0.05, sm3.area_between_log
    assert sm3.mass_jaccard < 1.0, sm3.mass_jaccard
    print(f"[mass-shift]  area={sm3.area_between_log:.3g} "
          f"jaccard={sm3.mass_jaccard:.3f}")

    print("All synthetic checks passed.")
