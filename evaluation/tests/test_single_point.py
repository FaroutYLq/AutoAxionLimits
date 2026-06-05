"""Tests for single-point-GT comparison in the subset comparator (#612).

No API/network. Many results are a SINGLE limit value at one operating mass
(e.g. 1706.00209 ORGAN g_ag<2.02e-12 @ 110 ueV; 2208.06519 QuantumCyclotron;
1806.00310 QUAX @ 58 ueV). The O'Hare file then has one distinct mass (a point
reference, not a curve), so curve interpolation cannot run and the paper is
discarded even when the extracted coupling matches. The comparator scores it as
a single point: evaluate the curve side at the reference operating mass(es)
(interpolating a real curve, falling back to the nearest point within
tolerance) and take |Δ log10 coupling|.

Run:
    pytest evaluation/tests/test_single_point.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.subset_compare import (  # noqa: E402
    _SINGLE_POINT_MASS_TOL_DEX,
    _residuals_at,
    _single_point_compare,
)


# --- _residuals_at: evaluate a curve at reference points ---------------------

def test_nearest_point_when_curve_is_single():
    """Single-point curve vs single reference at the same mass -> coupling gap."""
    curve = np.array([[6.12e-4, 3.2e-11]])
    ref = np.array([[6.12e-4, 4.0e-11]])
    res = _residuals_at(curve, ref, _SINGLE_POINT_MASS_TOL_DEX)
    assert res is not None and len(res) == 1
    assert res[0] == pytest.approx(abs(np.log10(3.2e-11) - np.log10(4.0e-11)), abs=1e-9)


def test_exact_mass_match_zero_residual():
    curve = np.array([[1.1e-4, 2.02e-12]])
    ref = np.array([[1.1e-4, 2.02e-12]])
    res = _residuals_at(curve, ref, _SINGLE_POINT_MASS_TOL_DEX)
    assert res is not None and res[0] == pytest.approx(0.0, abs=1e-9)


def test_interpolates_dense_curve_at_reference_point():
    """A real multi-point curve passing through the reference mass is evaluated
    there by interpolation (2110.14406: the traced curve hits the GT point)."""
    curve = np.array([[2.0e-5, 1.5e-13], [2.637e-5, 2.2e-13], [3.0e-5, 5.0e-14]])
    ref = np.array([[2.637e-5, 2.2e-13]])
    res = _residuals_at(curve, ref, _SINGLE_POINT_MASS_TOL_DEX)
    assert res is not None and res[0] == pytest.approx(0.0, abs=1e-6)


def test_reference_outside_curve_range_uses_nearest_within_tol():
    """Reference just outside the interp range still matches the nearest curve
    point when within tolerance (2102.08764: rounded masses bracket the box)."""
    curve = np.array([[3.3117e-5, 2.6e-6], [3.313e-5, 2.6e-6]])
    ref = np.array([[3.31157e-5, 2.6e-6], [3.313027e-5, 2.6e-6]])  # endpoints just outside
    res = _residuals_at(curve, ref, _SINGLE_POINT_MASS_TOL_DEX)
    assert res is not None and len(res) == 2
    assert np.allclose(res, 0.0, atol=1e-6)


def test_no_match_outside_tolerance_returns_none():
    """A reference mass far from the curve (beyond tolerance) does not match."""
    curve = np.array([[1.0e-6, 1e-13]])
    ref = np.array([[1.0e-3, 1e-13]])  # 3 dex away in mass
    assert _residuals_at(curve, ref, _SINGLE_POINT_MASS_TOL_DEX) is None


# --- _single_point_compare: filtering, dedup, coverage -----------------------

def test_compare_dedups_reference_rows_coverage_le_one():
    """Duplicate-mass reference rows are one operating point -> coverage <= 1."""
    curve = np.array([[1.1e-4, 2.02e-12]])
    # GT as stored: two identical rows + a 1e0 ceiling sentinel.
    ref = np.array([[1.1e-4, 2.02e-12], [1.1e-4, 2.02e-12], [1.1e-4, 1e0]])
    out = _single_point_compare(curve, ref, "AxionPhoton")
    assert out is not None
    resid, n_match, cov = out
    assert resid == pytest.approx(0.0, abs=1e-9)
    assert cov == pytest.approx(1.0)


def test_require_sparse_ref_rejects_rich_reference():
    """The curve-fallback refuses to score a rich multi-point reference this weak
    way (protects a genuine wrong-window curve failure from a lucky near point)."""
    curve = np.array([[1e-6, 1e-12], [1e-3, 2e-12]])
    rich_ref = np.array([[1e-6, 1e-12], [2e-6, 1e-12], [4e-6, 1e-12],
                         [8e-6, 1e-12], [1.6e-5, 1e-12]])  # 5 distinct masses
    assert _single_point_compare(curve, rich_ref, "AxionPhoton",
                                 require_sparse_ref=True) is None
    # Without the guard, a sparse reference is still scored.
    assert _single_point_compare(curve, rich_ref, "AxionPhoton") is not None


def test_boundary_sentinels_filtered_before_matching():
    """A reference that is only ceiling sentinels has no usable point."""
    curve = np.array([[1.1e-4, 2.02e-12]])
    ref = np.array([[1.1e-4, 1e0]])  # AxionPhoton ceiling (default 1e-2) drops it
    assert _single_point_compare(curve, ref, "AxionPhoton") is None
