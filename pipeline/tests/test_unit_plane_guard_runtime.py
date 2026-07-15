"""Phase 1 (#625) — fail-closed unit-plane guards, runtime mirror.

Pins pipeline.transform_guard against the eval side:
  * 1a — convention_review_needed flags a canonical-symbol declaration whose
    explicit unit power contradicts the canonical plane (with no converter);
  * 1b — convertible_out_of_profile flags a registry-convertible AxionEDM
    inv_fa declaration whose emitted magnitude cannot be the declared quantity;
  * both mirrors agree verdict-for-verdict on the shared corpus.

Run:  pytest pipeline/tests/test_unit_plane_guard_runtime.py -v
"""

import pytest

from pipeline.transform_guard import (
    convention_review_needed,
    convertible_out_of_profile,
    _unit_plane_contradicts,
    _explicit_unit_plane,
)
from evaluation.conventions import (
    _unit_plane_contradicts as _eval_contradicts,
    _explicit_unit_plane as _eval_unit,
)


# ============================ 1a — review flag ================================

@pytest.mark.parametrize("ct,decl,expected", [
    ("DarkPhoton",   "epsilon in GeV^-1",                  True),   # symbol ok, unit wrong
    ("AxionPhoton",  "g_agamma in GeV^-2",                 True),   # wrong power
    ("DarkPhoton",   "kinetic mixing chi (dimensionless)", False),  # canonical
    ("DarkPhoton",   "epsilon, kinetic mixing",            False),  # no explicit unit
    ("AxionPhoton",  "g_agamma [GeV^-1]",                  False),  # canonical
    ("AxionEDM",     "g_d in GeV^-2",                      False),  # canonical
    ("AxionEDM",     "C_G/f_a in GeV^-1",                  False),  # convertible (1b owns it)
    ("AxionNeutron", "g_ann in GeV^-1",                    False),  # convertible
    ("AxionElectron","g_ae (dimensionless)",               False),  # canonical
])
def test_convention_review_needed_unit_plane(ct, decl, expected):
    assert convention_review_needed(ct, decl) is expected


# ============================ 1b — out-of-profile =============================

@pytest.mark.parametrize("median,expected", [
    (5.4e-28, True),   # e*cm amplitude mislabeled -> flag
    (1.5e-28, True),
    (2e-13,   False),  # genuine 1/f_a -> no flag
    (1e-9,    False),  # genuine (f_a=1e9 GeV)
])
def test_convertible_out_of_profile_invfa(median, expected):
    pts = [(1e-15, median), (1e-14, median)]
    assert convertible_out_of_profile("AxionEDM", "C_G/f_a in GeV^-1", pts) is expected


def test_out_of_profile_scoped_to_invfa_edm_only():
    # Non-inv_fa AxionEDM (canonical g_d) never fires here.
    assert convertible_out_of_profile(
        "AxionEDM", "g_d in GeV^-2", [(1e-15, 6e-21)]) is False
    # Non-AxionEDM never fires.
    assert convertible_out_of_profile(
        "DarkPhoton", "epsilon", [(1e-6, 1e-6)]) is False
    # No data -> no fire.
    assert convertible_out_of_profile("AxionEDM", "C_G/f_a in GeV^-1", []) is False


# ============================ mirror-sync ====================================

_CORPUS = [
    ("DarkPhoton",   "epsilon in GeV^-1"),
    ("AxionPhoton",  "g_agamma in GeV^-2"),
    ("AxionPhoton",  "g_agamma [GeV^-1]"),
    ("AxionEDM",     "g_d in GeV^-2"),
    ("AxionEDM",     "C_G/f_a in GeV^-1"),
    ("DarkPhoton",   "kinetic mixing chi (dimensionless)"),
    ("AxionElectron","g_ae (dimensionless)"),
    ("ScalarPhoton", "d_e in GeV^-1"),
    ("VectorBL",     "g_bl in GeV"),
]


@pytest.mark.parametrize("ct,decl", _CORPUS)
def test_unit_plane_mirrors_agree(ct, decl):
    d = decl.lower()
    assert _unit_plane_contradicts(ct, d) == _eval_contradicts(ct, d)
    assert _explicit_unit_plane(d) == _eval_unit(d)
