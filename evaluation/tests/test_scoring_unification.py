"""Tests for post-full346 Phase 1c — scoring unification.

The full-pool scorer (evaluate.py --metrics) now applies the exact logic the
subset comparator has used since #587/#612:

* both-sides convention canonicalization (2103.03783: declared GeV^-1 scalar
  convention converted x sqrt2*M_Pl before residuals),
* UNCONVERTIBLE declarations -> convention_mismatch, not a 15-dex "residual"
  (2208.07293 e*cm EDM amplitude),
* mass-independent flat bounds expanded before filtering (2007.03694),
* reverse-pass promotion when the forward pass has no interpolatable GT vertex
  (1406.6053 / 0910.5914 vertex-sparse GT; 1906.08814 n_ext==1),
* sparse single-point tolerance fallback and single-mass-GT scoring (#612),
* closed-contour GT reduced to its lower envelope (2306.01048),
* wrong-mass-window extractions are NOT rescued by any of the above.

Run:
    pytest evaluation/tests/test_scoring_unification.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import compute_all_metrics
from evaluation.ground_truth import GroundTruthEntry
from evaluation.metrics import (
    _lower_envelope_if_contour,
    compute_interpolation_metrics,
)


def _entry(**kw) -> GroundTruthEntry:
    base = dict(
        arxiv_id="0000.00000", paper_title="t", coupling_type="AxionPhoton",
        coupling_convention="g_GeV^-1", coupling_units="g [GeV^-1]",
        is_new_limit=True, is_projection=False, data_source_expected="table",
        confidence_level=0.9, dm_density_assumed=None, difficulty="medium",
        tags=[], notes="", ground_truth_data_file=None,
        reference_repo_file="limit_data/AxionPhoton/Fake.txt",
        ground_truth_mass_range_eV=None, ground_truth_coupling_range=None,
        ground_truth_num_points=None, verified_by="test",
        verification_date="2026-07-02",
    )
    base.update(kw)
    return GroundTruthEntry(**base)


def _result(**kw) -> dict:
    base = {"arxiv_id": "0000.00000", "coupling_type": "AxionPhoton",
            "data_points": [], "extraction_confidence": 0.9, "num_points": 0}
    base.update(kw)
    return base


def _run_one(entry, result):
    m = compute_all_metrics([entry], [result])
    return m["per_paper"][0]


# ---------------------------------------------------------------------------
# Reverse-pass promotion (vertex-sparse GT / single-point extraction)
# ---------------------------------------------------------------------------

def test_vertex_sparse_flat_gt_scored_via_reverse():
    # 1406.6053 shape: GT flat segment encoded by only 2 far-apart vertices;
    # extraction's 3 masses all lie strictly between them -> forward pass has
    # 0 GT vertices in range, but the reverse pass scores them.
    gt = np.array([[8.68e-13, 6.55e-11], [1.0e3, 6.55e-11]])
    e = _entry()
    e.load_data = lambda: gt
    r = _result(data_points=[[1e-10, 6.6e-11], [1.0, 6.6e-11], [1e3 * 0.9, 6.6e-11]],
                num_points=3)
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["scored_via"] == "reverse"
    assert p["interp_metrics"]["median_residual_dex"] == pytest.approx(
        abs(np.log10(6.6e-11) - np.log10(6.55e-11)), abs=1e-6)


def test_single_extracted_point_scored_via_reverse():
    # 1906.08814 shape: perfect single-point extraction inside the GT span.
    gt = np.array([[2.031e-9, 1.5e-9], [2.039e-9, 1.5e-9]])
    e = _entry(coupling_type="DarkPhoton",
               coupling_convention="epsilon",
               reference_repo_file="limit_data/DarkPhoton/Fake.txt")
    e.load_data = lambda: gt
    r = _result(coupling_type="DarkPhoton",
                data_points=[[2.035e-9, 1.5e-9]], num_points=1)
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["scored_via"] == "reverse"
    assert p["interp_metrics"]["median_residual_dex"] == pytest.approx(0.0, abs=1e-9)


def test_wrong_window_not_rescued():
    # A rich multi-point extraction entirely outside the GT mass range must
    # stay a genuine zero_overlap (inf residual) — no reverse or single-point
    # rescue may mask a wrong-mass-window failure.
    gt = np.column_stack([np.logspace(-6, -5, 20), np.full(20, 1e-12)])
    e = _entry()
    e.load_data = lambda: gt
    ext = [[m, 1e-12] for m in np.logspace(2, 4, 30)]  # 100 eV..1e4 eV: way off
    r = _result(data_points=ext, num_points=30)
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["scored_via"] == "forward"
    assert p["interp_metrics"]["median_residual_dex"] == float("inf")


# ---------------------------------------------------------------------------
# Convention canonicalization + UNCONVERTIBLE guard
# ---------------------------------------------------------------------------

def test_scalar_gev_inv_declaration_canonicalized():
    # 2103.03783 shape: extraction declares Lambda^-1 [GeV^-1]; GT stores
    # dimensionless d_e. to_canonical multiplies by sqrt2*M_Pl = 3.394e18.
    f = np.sqrt(2.0) * 2.4e18
    gt = np.column_stack([np.logspace(-13, -11, 10), np.full(10, 0.2)])
    e = _entry(coupling_type="ScalarPhoton", coupling_convention="d_e",
               reference_repo_file="limit_data/ScalarPhoton/Fake.txt")
    e.load_data = lambda: gt
    ext = [[m, 0.2 / f] for m in np.logspace(-12.8, -11.2, 8)]
    r = _result(coupling_type="ScalarPhoton", data_points=ext, num_points=8,
                coupling_convention="Lambda_gamma^-1 in GeV^-1")
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["interp_metrics"]["median_residual_dex"] == pytest.approx(0.0, abs=1e-6)


def test_unconvertible_declaration_becomes_convention_mismatch():
    # A declaration recognized as a DIFFERENT quantity (foreign / no vetted
    # conversion) -> convention_mismatch, never a residual. (e*cm is now the
    # d_n_ecm converter, so use a bespoke novel plane that stays unconvertible.)
    gt = np.array([[4.37e-10, 1.47e-7], [5.72e-10, 1.5e-7]])
    e = _entry(coupling_type="AxionEDM", coupling_convention="g_angamma",
               reference_repo_file="limit_data/AxionEDM/Fake.txt")
    e.load_data = lambda: gt
    r = _result(coupling_type="AxionEDM",
                data_points=[[4.95e-10, 1e-22], [5.02e-10, 1e-22]], num_points=2,
                coupling_convention="oscillating EDM amplitude in GeV^-1")
    p = _run_one(e, r)
    assert p["comparison_status"] == "convention_mismatch"
    assert p["interp_metrics"] is None


# ---------------------------------------------------------------------------
# Flat-bound expansion in the full-pool path
# ---------------------------------------------------------------------------

def test_mass_independent_bound_scored():
    # 2007.03694 shape: mass<=0 sentinel rows with a correct coupling.
    gt = np.array([[1e-30, 1.3e-13], [1e4, 1.3e-13]])
    e = _entry(coupling_type="AxionElectron", coupling_convention="g_ae",
               reference_repo_file="limit_data/AxionElectron/Fake.txt")
    e.load_data = lambda: gt
    r = _result(coupling_type="AxionElectron",
                data_points=[[0.0, 1.29e-13], [0.0, 1.58e-13]], num_points=2)
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["interp_metrics"]["median_residual_dex"] < 0.1


# ---------------------------------------------------------------------------
# Single-mass GT (point reference) scoring (#612)
# ---------------------------------------------------------------------------

def test_single_mass_gt_scored_at_operating_mass():
    gt = np.array([[1.1e-4, 2.02e-12]])
    e = _entry()
    e.load_data = lambda: gt
    r = _result(data_points=[[1.1e-4, 2.5e-12]], num_points=1)
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["scored_via"] == "single_point_gt"
    assert p["interp_metrics"]["median_residual_dex"] == pytest.approx(
        abs(np.log10(2.5e-12) - np.log10(2.02e-12)), abs=1e-9)
    # No degenerate shape metrics for a point comparison.
    assert p["symmetric_metrics"] is None


def test_single_mass_gt_far_extraction_stays_point_reference():
    gt = np.array([[1.1e-4, 2.02e-12]])
    e = _entry()
    e.load_data = lambda: gt
    r = _result(data_points=[[5.0, 2.0e-12]], num_points=1)  # 4.7 dex away in mass
    p = _run_one(e, r)
    assert p["comparison_status"] == "gt_point_reference"


# ---------------------------------------------------------------------------
# Closed-contour lower envelope
# ---------------------------------------------------------------------------

def test_lower_envelope_reduces_closed_contour():
    # A rectangular exclusion band traced as a closed polygon: lower edge at
    # 1e-9, upper edge at 1e-6. The envelope must keep only the lower edge.
    contour = np.array([
        [1.0, 1e-9], [1e6, 1e-9],       # lower edge, mass rising
        [1e6, 1e-6], [1.0, 1e-6],       # upper edge, mass falling back
    ])
    env = _lower_envelope_if_contour(contour)
    assert np.allclose(np.unique(env[:, 1]), 1e-9)


def test_contour_gt_scored_against_lower_envelope():
    # 2306.01048 shape: GT is a closed band; a correct extraction of the lower
    # (free-streaming) edge must score ~0, not the band height.
    lower = np.column_stack([np.logspace(0, 6, 15), np.full(15, 6.2e-10)])
    upper = np.column_stack([np.logspace(6, 0, 15), np.full(15, 2.5e-6)])
    gt = np.vstack([lower, upper])
    e = _entry(coupling_type="AxionProton", coupling_convention="g_ap",
               reference_repo_file="limit_data/AxionProton/Fake.txt")
    e.load_data = lambda: gt
    ext = [[m, 6.2e-10] for m in np.logspace(0.5, 5.5, 10)]
    r = _result(coupling_type="AxionProton", data_points=ext, num_points=10)
    p = _run_one(e, r)
    assert p["comparison_status"] == "compared"
    assert p["interp_metrics"]["median_residual_dex"] == pytest.approx(0.0, abs=1e-6)


def test_monotonic_curve_unchanged_by_envelope():
    curve = np.column_stack([np.logspace(-6, -4, 12),
                             np.logspace(-12, -11, 12)])
    out = _lower_envelope_if_contour(curve[::-1])  # reversed order is still monotonic
    assert np.allclose(out, curve)


def test_interpolation_metrics_n_ext_one_reverse_only():
    gt = np.column_stack([np.logspace(-6, -4, 10), np.full(10, 1e-12)])
    ext = np.array([[1e-5, 1.1e-12]])
    im = compute_interpolation_metrics("x", ext, gt, coupling_type="AxionPhoton")
    assert im.num_interpolatable == 0
    assert im.median_residual_dex == float("inf")
    assert im.num_interpolatable_reverse == 1
    assert im.median_residual_dex_reverse == pytest.approx(
        abs(np.log10(1.1e-12) - np.log10(1e-12)), abs=1e-9)
