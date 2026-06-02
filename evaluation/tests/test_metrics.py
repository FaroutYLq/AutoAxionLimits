"""Unit tests for ``evaluation.metrics`` (issue #544).

The boundary-ceiling bugs fixed in PR #535/#536 would have been caught
instantly by a metric unit test. These tests pin the metric surface — including
the #541 symmetric/reverse-pass metrics and the #542 noise-floor calibration —
against synthetic curves whose answers are known by construction.

Run:
    pytest evaluation/tests/
    python -m pytest evaluation/tests/test_metrics.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.metrics import (
    CONTINUOUS_TAUS_DEX,
    NOISE_FLOOR_RESIDUAL_DEX,
    _COUPLING_CEILINGS,
    _DEFAULT_COUPLING_CEIL,
    _filter_boundary,
    _interval_jaccard,
    compute_confidence_calibration,
    compute_interpolation_metrics,
    compute_symmetric_curve_metrics,
)


# ---------------------------------------------------------------------------
# Synthetic curve helpers
# ---------------------------------------------------------------------------

def power_law_curve(m_lo_dex, m_hi_dex, c0_dex, slope=-0.5, n=25):
    """A log-log power-law curve over [10**m_lo_dex, 10**m_hi_dex] eV.

    coupling = 10**(c0_dex + slope * (log10(m) - m_lo_dex)).
    """
    m = np.logspace(m_lo_dex, m_hi_dex, n)
    c = 10.0 ** (c0_dex + slope * (np.log10(m) - m_lo_dex))
    return np.column_stack([m, c])


# ---------------------------------------------------------------------------
# Forward interpolation residual: identical / shifted / disjoint
# ---------------------------------------------------------------------------

def test_identical_curves_zero_residual_full_coverage():
    gt = power_law_curve(-6, -5, -10)
    im = compute_interpolation_metrics("id", gt.copy(), gt.copy(),
                                       coupling_type="AxionPhoton")
    assert im.median_residual_dex == pytest.approx(0.0, abs=1e-9)
    assert im.mean_residual_dex == pytest.approx(0.0, abs=1e-9)
    assert im.max_residual_dex == pytest.approx(0.0, abs=1e-9)
    # Every GT point lies inside the (identical) extracted mass range.
    assert im.interpolation_coverage == pytest.approx(1.0)
    assert im.frac_within_0_1dex == pytest.approx(1.0)


def test_vertical_shift_by_known_factor():
    """A curve scaled by factor k in coupling has median residual log10(k)."""
    gt = power_law_curve(-6, -5, -10)
    for k in (2.0, 10.0, 100.0):
        ext = gt.copy()
        ext[:, 1] *= k
        im = compute_interpolation_metrics("shift", ext, gt.copy(),
                                           coupling_type="AxionPhoton")
        assert im.median_residual_dex == pytest.approx(math.log10(k), abs=1e-9)
        assert im.mean_residual_dex == pytest.approx(math.log10(k), abs=1e-9)


def test_non_overlapping_mass_ranges_infinite_residual_zero_coverage():
    gt = power_law_curve(-6, -5, -10)
    ext = power_law_curve(-2, -1, -10)  # disjoint mass range
    im = compute_interpolation_metrics("disjoint", ext, gt.copy(),
                                       coupling_type="AxionPhoton")
    assert math.isinf(im.median_residual_dex)
    assert im.interpolation_coverage == pytest.approx(0.0)
    assert im.num_interpolatable == 0


# ---------------------------------------------------------------------------
# Boundary-closure sentinel filtering (the PR #535 / #536 bug surface)
# ---------------------------------------------------------------------------

def test_coupling_ceilings_pin_the_known_values():
    """Lock the exact ceilings the PR #535/#536 bugs touched."""
    assert _COUPLING_CEILINGS["AxionMass"] == 1.0
    assert _COUPLING_CEILINGS["DarkPhoton"] == 1.0
    assert _COUPLING_CEILINGS["VectorBL"] == 1.0
    assert _COUPLING_CEILINGS["AxionCPV"] == 1.0
    assert _COUPLING_CEILINGS["MonopoleDipole"] == 1.0
    for scalar in ("ScalarPhoton", "ScalarElectron",
                   "ScalarBaryon", "ScalarNucleon"):
        assert _COUPLING_CEILINGS[scalar] == 1e19
    assert _DEFAULT_COUPLING_CEIL == 1e-2


def test_axionmass_strips_unity_sentinel_keeps_real_data():
    """AxionMass closes at 1.0: real ~1e-10 data kept, 1.0 sentinel stripped."""
    data = np.array([
        [1e-6, 1e-10],
        [1e-5, 1e-11],
        [1e-4, 1.0],     # boundary-closure sentinel
    ])
    f = _filter_boundary(data, _COUPLING_CEILINGS["AxionMass"])
    assert len(f) == 2
    assert np.all(f[:, 1] < 1.0)


def test_scalar_keeps_large_real_data_strips_extreme_sentinels():
    """Scalar* keeps real data up to ~1e17 but strips 1e20/1e30 sentinels."""
    data = np.array([
        [1e-3, 1e0],
        [1e-2, 1e10],
        [1e-1, 1e17],    # real data, large coupling
        [1e0, 1e20],     # closure sentinel
        [1e1, 1e30],     # closure sentinel
    ])
    f = _filter_boundary(data, _COUPLING_CEILINGS["ScalarPhoton"])
    assert len(f) == 3
    assert f[:, 1].max() == pytest.approx(1e17)


def test_darkphoton_unity_ceiling_strips_closure():
    data = np.array([
        [1e-6, 1e-12],
        [1e-5, 1e-13],
        [1e-4, 1.0],     # closure sentinel
    ])
    f = _filter_boundary(data, _COUPLING_CEILINGS["DarkPhoton"])
    assert len(f) == 2


def test_default_ceiling_strips_at_1e_minus_2():
    data = np.array([
        [1e-6, 1e-12],
        [1e-5, 1e-3],
        [1e-4, 1e-2],    # at the default ceiling -> stripped (strict <)
        [1e-3, 1e0],     # closure sentinel
    ])
    f = _filter_boundary(data, _DEFAULT_COUPLING_CEIL)
    assert len(f) == 2
    assert f[:, 1].max() < 1e-2


# --- Regression tests for the OLD buggy ceilings (must FAIL on old values) ---

def test_regression_axionmass_old_1e6_ceiling_kept_sentinels():
    """PR #535 bug: AxionMass ceiling of 1e6 KEPT the 1.0 closure sentinels,
    corrupting the interpolation. With the current 1.0 ceiling the sentinel is
    stripped; the old 1e6 ceiling would have kept it. This test encodes the
    contrast so a regression to 1e6 is caught."""
    data = np.array([
        [1e-6, 1e-10],
        [1e-5, 1e-11],
        [1e-4, 1.0],     # boundary-closure sentinel
    ])
    # Current (fixed) ceiling: sentinel stripped.
    fixed = _filter_boundary(data, _COUPLING_CEILINGS["AxionMass"])
    assert len(fixed) == 2
    assert 1.0 not in fixed[:, 1]
    # Old buggy ceiling: the 1.0 sentinel survives and pollutes the data.
    buggy = _filter_boundary(data, 1e6)
    assert len(buggy) == 3
    assert 1.0 in buggy[:, 1]


def test_regression_scalar_old_1e0_ceiling_discarded_whole_curve():
    """PR #536 bug: a 1e0 ceiling for Scalar* discarded the ENTIRE real curve
    (real data spans 1e0..1e17). The current 1e19 ceiling keeps it."""
    data = np.array([
        [1e-3, 1e2],
        [1e-2, 1e10],
        [1e-1, 1e17],
    ])
    fixed = _filter_boundary(data, _COUPLING_CEILINGS["ScalarPhoton"])
    assert len(fixed) == 3  # whole curve kept
    buggy = _filter_boundary(data, 1e0)
    assert len(buggy) == 0  # whole curve discarded


def test_regression_metric_end_to_end_axionmass_sentinel():
    """End-to-end: with the buggy 1e6 ceiling the retained 1.0 sentinel wrecks
    the residual; the fixed 1.0 ceiling yields a clean ~0 residual."""
    gt = np.array([
        [1e-6, 1e-10],
        [1e-5, 10 ** -10.5],
        [1e-4, 1e-11],
    ])
    # The extraction has the same real points PLUS a plotting closure sentinel
    # interleaved at an intermediate mass (1e-5.5). With the fixed 1.0 ceiling
    # the sentinel is stripped and the interpolation runs cleanly through the
    # real points. With the buggy 1e6 ceiling the 1.0 sentinel survives and
    # sits between GT masses, so interpolating GT masses across it distorts the
    # residual badly.
    ext = np.vstack([
        [1e-6, 1e-10],
        [10 ** -5.5, 1.0],   # boundary-closure sentinel between real points
        [1e-4, 1e-11],
    ])
    im_fixed = compute_interpolation_metrics(
        "fixed", ext, gt, coupling_ceil=_COUPLING_CEILINGS["AxionMass"])
    im_buggy = compute_interpolation_metrics(
        "buggy", ext, gt, coupling_ceil=1e6)
    assert im_fixed.median_residual_dex == pytest.approx(0.0, abs=1e-9)
    assert im_fixed.max_residual_dex == pytest.approx(0.0, abs=1e-9)
    # The buggy ceiling keeps the 1.0 sentinel between real points, so any GT
    # mass that interpolates across it is badly distorted: the worst-case
    # residual blows up by many dex (the fixed version is clean everywhere).
    assert im_buggy.max_residual_dex > 0.5


# ---------------------------------------------------------------------------
# Single-mass GT routing (gt_point_reference) — tested via the metric guard
# ---------------------------------------------------------------------------

def test_single_mass_gt_routes_to_point_reference():
    """A GT curve with a single distinct mass has < 2 usable masses, which is
    what ``evaluate.py`` uses to route to ``gt_point_reference`` instead of a
    curve comparison. The metric guard returns the empty/degenerate result."""
    from evaluation.evaluate import _usable_gt_stats

    single = np.array([[1e-6, 1e-10]])
    n_pts, n_mass = _usable_gt_stats(single, "AxionPhoton")
    assert n_pts == 1
    assert n_mass == 1  # < 2 distinct masses -> point reference, not a curve

    # And the metric itself does not produce a spurious residual: with < 2 GT
    # masses there is nothing to interpolate against.
    ext = power_law_curve(-6, -5, -10)
    im = compute_interpolation_metrics("pt", ext, single,
                                       coupling_type="AxionPhoton")
    # One GT point inside the extracted range is interpolatable, but a single
    # point can never form a comparable *curve* — the routing happens upstream.
    assert im.num_ground_truth == 1


def test_two_distinct_masses_is_a_curve():
    from evaluation.evaluate import _usable_gt_stats

    two = np.array([[1e-6, 1e-10], [1e-5, 1e-11]])
    n_pts, n_mass = _usable_gt_stats(two, "AxionPhoton")
    assert n_pts == 2
    assert n_mass == 2  # >= 2 distinct masses -> compared as a curve


# ---------------------------------------------------------------------------
# #541: reverse pass, area-between-curves, mass-range Jaccard
# ---------------------------------------------------------------------------

def test_reverse_pass_identical_curves_zero():
    gt = power_law_curve(-6, -5, -10)
    im = compute_interpolation_metrics("id", gt.copy(), gt.copy(),
                                       coupling_type="AxionPhoton")
    assert im.median_residual_dex_reverse == pytest.approx(0.0, abs=1e-9)
    assert im.interpolation_coverage_reverse == pytest.approx(1.0)


def test_reverse_pass_overclaim_lowers_reverse_coverage():
    """Extraction extends well past the GT range (over-claiming): forward
    coverage is fine but reverse coverage drops (extracted masses outside GT)."""
    gt = power_law_curve(-6, -5, -10)
    over = power_law_curve(-6, -2, -10, n=60)  # same shape, 4x wider
    im = compute_interpolation_metrics("over", over, gt.copy(),
                                       coupling_type="AxionPhoton")
    assert im.interpolation_coverage_reverse < im.interpolation_coverage


def test_area_between_identical_curves_zero_jaccard_one():
    gt = power_law_curve(-6, -5, -10)
    sm = compute_symmetric_curve_metrics("id", gt.copy(), gt.copy(),
                                         coupling_type="AxionPhoton")
    assert sm.area_between_log == pytest.approx(0.0, abs=1e-9)
    assert sm.mass_jaccard == pytest.approx(1.0, abs=1e-9)


def test_area_between_known_vertical_shift():
    """A pure vertical shift of k dex gives a normalised area equal to k dex
    (the integrand |Δlog10 c| is constant = k over the whole overlap)."""
    gt = power_law_curve(-6, -5, -10)
    for shift_dex in (0.3, 1.0):
        ext = gt.copy()
        ext[:, 1] *= 10 ** shift_dex
        sm = compute_symmetric_curve_metrics("v", ext, gt.copy(),
                                             coupling_type="AxionPhoton")
        assert sm.area_between_log == pytest.approx(shift_dex, abs=1e-6)
        assert sm.mass_jaccard == pytest.approx(1.0, abs=1e-9)


def test_mass_jaccard_disjoint_zero():
    gt = power_law_curve(-6, -5, -10)
    ext = power_law_curve(-2, -1, -10)
    sm = compute_symmetric_curve_metrics("disj", ext, gt.copy(),
                                         coupling_type="AxionPhoton")
    assert sm.mass_jaccard == pytest.approx(0.0)
    assert math.isinf(sm.area_between_log)


def test_mass_jaccard_partial_known_value():
    """Extraction spans [-6,-2] (4 dex), GT spans [-6,-5] (1 dex). Intersection
    is 1 dex, union is 4 dex -> Jaccard = 1/4 = 0.25."""
    gt = power_law_curve(-6, -5, -10)
    over = power_law_curve(-6, -2, -10, n=60)
    sm = compute_symmetric_curve_metrics("part", over, gt.copy(),
                                         coupling_type="AxionPhoton")
    assert sm.mass_jaccard == pytest.approx(0.25, abs=1e-6)


def test_interval_jaccard_unit():
    assert _interval_jaccard((0.0, 1.0), (0.0, 1.0)) == pytest.approx(1.0)
    assert _interval_jaccard((0.0, 1.0), (2.0, 3.0)) == pytest.approx(0.0)
    # [0,4] vs [0,1]: inter=1, union=4 -> 0.25
    assert _interval_jaccard((0.0, 4.0), (0.0, 1.0)) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# #542: noise-floor calibration + continuous P(residual < tau)
# ---------------------------------------------------------------------------

def test_accuracy_threshold_is_noise_floor():
    assert NOISE_FLOOR_RESIDUAL_DEX == 0.32
    # The default accuracy threshold of the calibration tracks the noise floor.
    import inspect

    sig = inspect.signature(compute_confidence_calibration)
    assert sig.parameters["accuracy_threshold_residual"].default == NOISE_FLOOR_RESIDUAL_DEX


def _make_interp_metrics(arxiv_id, median_residual, coverage=1.0):
    """Build a minimal InterpolationMetrics with a known median residual."""
    from evaluation.metrics import InterpolationMetrics

    return InterpolationMetrics(
        arxiv_id=arxiv_id,
        num_extracted=10,
        num_ground_truth=10,
        num_interpolatable=10,
        interpolation_coverage=coverage,
        residuals_dex=np.array([median_residual]),
        median_residual_dex=median_residual,
        mean_residual_dex=median_residual,
        p90_residual_dex=median_residual,
        max_residual_dex=median_residual,
        frac_within_0_1dex=0.0,
        frac_within_0_3dex=0.0,
        frac_within_0_5dex=0.0,
        frac_within_1_0dex=0.0,
    )


def test_confident_but_wrong_yields_overconfidence_gap():
    """High confidence (0.9) but residuals well above the noise floor => the
    top bin's accuracy is 0 while mean confidence is high: positive gap."""
    confidences = [0.9, 0.95, 0.92, 0.88]
    # All residuals are 1.0 dex (an order of magnitude) — way above 0.32.
    interp = [_make_interp_metrics(f"p{i}", 1.0) for i in range(4)]
    ids = [f"p{i}" for i in range(4)]
    bins = compute_confidence_calibration(confidences, interp, ids, n_bins=5)
    top = [b for b in bins if b.n_papers > 0][-1]
    assert top.actual_accuracy == pytest.approx(0.0)
    gap = top.mean_confidence - top.actual_accuracy
    assert gap > 0.0


def test_confident_and_correct_is_calibrated():
    """High confidence + residuals below the noise floor => accuracy 1.0."""
    confidences = [0.9, 0.95, 0.92, 0.88]
    interp = [_make_interp_metrics(f"p{i}", 0.05) for i in range(4)]  # < 0.32
    ids = [f"p{i}" for i in range(4)]
    bins = compute_confidence_calibration(confidences, interp, ids, n_bins=5)
    top = [b for b in bins if b.n_papers > 0][-1]
    assert top.actual_accuracy == pytest.approx(1.0)


def test_continuous_p_residual_below_tau_fields():
    """Smoke-test the continuous P(residual < tau) fields."""
    confidences = [0.85, 0.9, 0.95]
    # residuals: 0.05 (< all taus), 0.4 (< 0.5,1.0), 2.0 (< none)
    interp = [
        _make_interp_metrics("a", 0.05),
        _make_interp_metrics("b", 0.4),
        _make_interp_metrics("c", 2.0),
    ]
    ids = ["a", "b", "c"]
    bins = compute_confidence_calibration(confidences, interp, ids, n_bins=1)
    b = bins[0]
    assert b.n_papers == 3
    # frac_within_tau keys are the configured taus.
    assert set(b.frac_within_tau.keys()) == set(CONTINUOUS_TAUS_DEX)
    # tau=0.1: only the 0.05 paper -> 1/3
    assert b.frac_within_tau[0.1] == pytest.approx(1 / 3)
    # tau=0.5: the 0.05 and 0.4 papers -> 2/3
    assert b.frac_within_tau[0.5] == pytest.approx(2 / 3)
    # tau=1.0: same two (2.0 fails) -> 2/3
    assert b.frac_within_tau[1.0] == pytest.approx(2 / 3)
    # Finite-residual distribution summary (all 3 are finite).
    assert b.n_finite == 3
    assert b.median_residual_dex == pytest.approx(0.4)


def test_empty_calibration_returns_empty_list():
    assert compute_confidence_calibration([], [], []) == []


# ---------------------------------------------------------------------------
# Summary builder (the diffable metrics_summary.json) — issue #544
# ---------------------------------------------------------------------------

def test_build_metrics_summary_shape():
    from evaluation.evaluate import build_metrics_summary

    fake = {
        "n_papers": 3,
        "comparison_coverage": {"status_counts": {"compared": 2, "gt_unusable": 1}},
        "classification": {
            "coupling_type": {"accuracy": 1.0},
            "is_new_limit": {"accuracy": 0.5},
            "is_projection": {"accuracy": 0.9},
            "data_source": {"accuracy": 0.8},
        },
        "interpolation_aggregate": {
            "n_finite": 2, "n_zero_overlap": 0,
            "median_median_residual_dex": 0.12,
            "mean_frac_within_0_3dex": 0.7,
            "median_median_residual_dex_reverse": 0.15,
        },
        "symmetric_aggregate": {
            "median_area_between_log": 0.2, "median_mass_jaccard": 0.9,
        },
        "per_type_aggregate": {
            "n_types": 1, "n_papers_compared": 2,
            "micro_median_residual_dex": 0.12,
            "macro_median_residual_dex": 0.12,
            "macro_minus_micro_dex": 0.0,
            "per_type": {"AxionPhoton": {"n": 2, "median_residual_dex": 0.12}},
        },
        "confidence_calibration": [
            {"bin_lo": 0.0, "bin_hi": 0.8, "n_papers": 0,
             "mean_confidence": 0.0, "actual_accuracy": 0.0},
            {"bin_lo": 0.8, "bin_hi": 1.0, "n_papers": 2,
             "mean_confidence": 0.9, "actual_accuracy": 0.5},
        ],
    }
    summary = build_metrics_summary(fake)
    assert summary["n_papers"] == 3
    assert summary["status_counts"] == {"compared": 2, "gt_unusable": 1}
    assert summary["per_type_aggregate"]["per_type_n"] == {"AxionPhoton": 2}
    assert summary["calibration"]["noise_floor_residual_dex"] == NOISE_FLOOR_RESIDUAL_DEX
    # Top non-empty bin: confidence 0.9, accuracy 0.5 -> gap 0.4.
    assert summary["calibration"]["overconfidence_gap"] == pytest.approx(0.4)
