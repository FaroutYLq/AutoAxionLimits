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

from pipeline.reviewer import apply_dm_density_correction


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
