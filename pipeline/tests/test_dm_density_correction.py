"""Regression tests pinning the DM-density rescale DIRECTION.

A DM-search coupling limit scales as ``g_limit ∝ 1/sqrt(rho_DM)`` because the
excluded signal power ∝ ``rho·g²`` at a fixed detection threshold. Re-expressing
a limit quoted at ``rho_paper`` in the repository convention ``rho_repo`` must
therefore give ``g_repo = g_paper·sqrt(rho_paper/rho_repo)``: a *higher* assumed
density (more dark matter) yields a *stronger* (lower) coupling limit.

An earlier version applied the inverse ``sqrt(rho_repo/rho_paper)``, which
weakened every density-corrected limit and disagreed with the repository's own
``PlotFuncs.py`` DM-search methods (which multiply stored paper-native data by
``sqrt(rho_paper/rho_repo)`` at plot time, e.g. ``sqrt(0.3/0.45)``). These tests
pin the correct direction so it cannot silently invert again.

Run:
    python -m pytest pipeline/tests/test_dm_density_correction.py -v
"""

from __future__ import annotations

import math

from pipeline.extractor import ExtractionResult
from pipeline.reviewer import (
    _inject_density_rescale,
    apply_corrections,
    apply_dm_density_correction,
    dm_density_params,
)


def _dm_result(coupling="DarkPhoton", rho=0.3, points=None):
    return ExtractionResult(
        arxiv_id="0000.00000", paper_title="t", arxiv_url="u",
        coupling_type=coupling, is_new_limit=True, is_projection=False,
        data_points=points if points is not None else [(2.0e-5, 1.0e-12), (3.0e-5, 1.0e-12)],
        data_source="text", dm_density_assumed=rho, polarization_assumption=None,
        confidence_level=0.95, suggested_experiment_name="EXP", extraction_confidence=0.7,
    )


def test_higher_repo_density_strengthens_limit():
    # paper rho=0.3, repo rho=0.45: repo assumes MORE dark matter, so the
    # limit must get STRONGER (lower coupling), factor sqrt(0.3/0.45) < 1.
    pts = [(1.0e-5, 1.0e-12), (3.0e-5, 1.0e-12)]
    corrected, note = apply_dm_density_correction(pts, rho_paper=0.3, rho_repo=0.45)
    factor = corrected[0][1] / pts[0][1]
    assert factor < 1.0, "higher assumed density must strengthen (lower) the limit"
    assert math.isclose(factor, math.sqrt(0.3 / 0.45), rel_tol=1e-9)
    assert math.isclose(factor, 0.81649658, rel_tol=1e-6)


def test_matches_plotfuncs_convention():
    # PlotFuncs DM-search methods use sqrt(rho_paper/rho_repo) = sqrt(0.3/0.45).
    pts = [(1.0e-5, 2.5e-13)]
    corrected, _ = apply_dm_density_correction(pts, rho_paper=0.3, rho_repo=0.45)
    assert math.isclose(corrected[0][1], 2.5e-13 * math.sqrt(0.3 / 0.45), rel_tol=1e-9)


def test_lower_repo_density_weakens_limit():
    # If the repo convention assumed LESS dark matter than the paper, the limit
    # would weaken (factor > 1). Guards the direction symmetrically.
    pts = [(1.0e-5, 1.0e-12)]
    corrected, _ = apply_dm_density_correction(pts, rho_paper=0.45, rho_repo=0.30)
    factor = corrected[0][1] / pts[0][1]
    assert factor > 1.0
    assert math.isclose(factor, math.sqrt(0.45 / 0.30), rel_tol=1e-9)


def test_equal_densities_is_identity():
    pts = [(1.0e-5, 1.0e-12), (2.0e-5, 5.0e-13)]
    corrected, _ = apply_dm_density_correction(pts, rho_paper=0.45, rho_repo=0.45)
    for (m0, g0), (m1, g1) in zip(pts, corrected):
        assert math.isclose(g0, g1, rel_tol=1e-12)


def test_note_reports_the_applied_direction():
    _, note = apply_dm_density_correction([(1e-5, 1e-12)], rho_paper=0.3, rho_repo=0.45)
    assert "sqrt(0.3/0.45)" in note


# ---------------------------------------------------------------------------
# Single-owner convention (2026-07-05): density is owned by the plotting
# method, NOT baked into the stored data file.
# ---------------------------------------------------------------------------

def test_apply_corrections_keeps_data_paper_native():
    # apply_corrections must NOT rescale the stored data — it stays paper-native
    # so the plotting method is the single owner of the sqrt(rho) conversion
    # (baking it in AND letting the method rescale would double-count).
    pts = [(2.0e-5, 1.0e-12), (3.0e-5, 1.0e-12)]
    data, applied, flagged = apply_corrections(_dm_result(rho=0.3, points=list(pts)))
    assert data == pts, "stored data must be paper-native (unrescaled)"
    assert any("paper-native" in a and "sqrt(0.3/0.45)" in a for a in applied)


def test_dm_density_params_selects_haloscopes_only():
    assert dm_density_params(_dm_result(coupling="DarkPhoton", rho=0.3)) == (0.3, 0.45)
    # a coupling with no dm_density entry (e.g. a stellar bound) -> None
    assert dm_density_params(_dm_result(coupling="ScalarPhoton", rho=0.3)) is None
    # density equal to repo convention -> no rescale needed
    assert dm_density_params(_dm_result(coupling="DarkPhoton", rho=0.45)) is None


def test_inject_density_rescale_after_loadtxt():
    code = (
        "    @staticmethod\n"
        "    def EXP(ax,col='r'):\n"
        "        dat = loadtxt('limit_data/DarkPhoton/EXP.txt',ndmin=2)\n"
        "        ax.fill_between(dat[:,0],dat[:,1],y2=1e0)\n"
    )
    out = _inject_density_rescale(code, 0.3, 0.45)
    lines = out.splitlines()
    load_i = next(i for i, l in enumerate(lines) if "loadtxt" in l)
    assert "dat[:,1] = dat[:,1]*sqrt(0.3/0.45)" in lines[load_i + 1]
    assert lines[load_i + 1].startswith("        ")  # preserves indentation
    # idempotent
    assert _inject_density_rescale(out, 0.3, 0.45).count("sqrt(0.3/0.45)") == 1


def test_inject_respects_variable_name():
    code = "    def EXP(ax):\n        d = loadtxt('x.txt',ndmin=2)\n        pass\n"
    out = _inject_density_rescale(code, 0.4, 0.45)
    assert "d[:,1] = d[:,1]*sqrt(0.4/0.45)" in out
