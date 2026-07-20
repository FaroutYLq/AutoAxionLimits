"""Gauge-group type correction — U(1)_B / U(1)_{B-L} vector DM misclassified
as DarkPhoton.

A gauged baryonic vector boson (LIGO/LISA-Pathfinder/PPTA class) plots its
coupling as a bare dimensionless epsilon, so the figure axis cannot distinguish
it from kinetic-mixing chi and the classifier emits DarkPhoton. The disambiguator
is the gauge group, which the model names in its own convention declaration. The
deterministic guard re-labels DarkPhoton -> VectorBL from that declaration; the
strings below are the ACTUAL declared conventions of the three failing papers in
the final347_remediated run (2105.13085, 2301.08736, 2112.07687).

Run:  pytest pipeline/tests/test_gauge_group_correction.py -v
"""

import pytest

from pipeline.transform_guard import gauge_group_type_correction
from pipeline.extractor import apply_axis_crosscheck  # noqa: F401 (import parity)


# Real declared_convention strings from the definitive-run snapshots.
DECL_2105 = ("dimensionless epsilon (U(1)_B dark-photon/baryon coupling); "
             "values are sqrt of the reported epsilon^2 (not kinetic mixing chi)")
DECL_2112 = ("dimensionless epsilon (U(1)_{B-L} coupling normalized to "
             "electromagnetic coupling e); NOT converted to g_BL (would need "
             "factor e~0.303). Values are sqrt of the plotted epsilon^2.")
DECL_2301 = ("epsilon^2 (squared coupling strength) at 95% CL, dimensionless "
             "— plotted as epsilon^2_{95%}")
# 2301.08736 as re-emitted under the strengthened prompt (real subscription/Opus
# read on this branch): the declaration now names the baryon coupling even though
# the model still filled coupling_type=DarkPhoton — the guard must catch it.
DECL_2301_REEXTRACTED = ("epsilon^2_{95%}, squared coupling of the dark photon "
                         "to baryons, dimensionless (plotted as ε² directly, NOT ε)")


@pytest.mark.parametrize("decl", [DECL_2105, DECL_2112, DECL_2301_REEXTRACTED])
def test_declared_bl_gauge_group_remaps_darkphoton(decl):
    # Every paper whose declaration names the gauge group OR a baryon coupling is
    # corrected deterministically.
    assert gauge_group_type_correction("DarkPhoton", decl) == "VectorBL"


def test_bare_epsilon_squared_declaration_is_noop():
    # The ORIGINAL 2301.08736 convention named only epsilon^2 — no gauge/baryon
    # signal at all, so the guard correctly declines (it was the prompt that
    # enriched the declaration on re-extraction; see DECL_2301_REEXTRACTED).
    assert gauge_group_type_correction("DarkPhoton", DECL_2301) is None


@pytest.mark.parametrize("decl", [
    "U(1)_B gauged baryon number",
    "g_BL dimensionless B-L gauge coupling",
    "baryon minus lepton vector coupling",
    "epsilon for a vector coupled to baryon number",
])
def test_various_gauge_group_phrasings(decl):
    assert gauge_group_type_correction("DarkPhoton", decl) == "VectorBL"


def test_only_darkphoton_is_corrected():
    # No-op for every other classified type, so an already-correct VectorBL (or
    # any other type) is never touched — the guard cannot regress a correct pick.
    assert gauge_group_type_correction("VectorBL", DECL_2112) is None
    assert gauge_group_type_correction("AxionPhoton", DECL_2112) is None
    assert gauge_group_type_correction(None, DECL_2112) is None


@pytest.mark.parametrize("decl", [
    "dimensionless kinetic mixing chi",
    "kinetic mixing parameter epsilon between the photon and a hidden photon",
    "hidden-photon / paraphoton mixing chi",
    "",
    None,
])
def test_genuine_dark_photon_not_demoted(decl):
    # A real kinetic-mixing dark photon declares chi / kinetic mixing, never a
    # B/B-L gauge coupling, so it is left as DarkPhoton.
    assert gauge_group_type_correction("DarkPhoton", decl) is None


def test_pure_and_total():
    # Never raises on odd input; deterministic.
    assert gauge_group_type_correction("DarkPhoton", 12345) is None
    assert gauge_group_type_correction("DarkPhoton", DECL_2105) == "VectorBL"
    assert gauge_group_type_correction("DarkPhoton", DECL_2105) == "VectorBL"
