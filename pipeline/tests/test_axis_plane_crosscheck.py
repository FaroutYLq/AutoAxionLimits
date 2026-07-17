"""Phase 3 (#625) — axis-plane consistency cross-check.

Pure deterministic functions (no model calls): the axis→coupling-type map, the
confusable-family override guardrail, fail-open on unreadable axes, and the
in-place re-label helper used by the extractor. Model-call validation of the
end-to-end effect is deferred to the final benchmark (budget rule).

Run:  pytest pipeline/tests/test_axis_plane_crosscheck.py -v
"""

import pytest

from pipeline.transform_guard import (
    _CONFUSABLE_FAMILIES,
    axis_implies_coupling_type,
    axis_plane_crosscheck,
    in_confusable_family,
    same_confusable_family,
)
from pipeline.extractor import apply_axis_crosscheck


# ============================ axis -> type map ================================

@pytest.mark.parametrize("label,expected", [
    ("g_{B-L}", "VectorBL"),
    ("g_BL [dimensionless]", "VectorBL"),
    ("kinetic mixing ε", "DarkPhoton"),
    ("χ", "DarkPhoton"),
    ("epsilon (kinetic mixing)", "DarkPhoton"),
    ("|g_{aγ}| [GeV^-1]", "AxionPhoton"),
    ("g_agamma", "AxionPhoton"),
    ("g_ae", "AxionElectron"),
    ("g_ap", "AxionProton"),
    ("g_an", "AxionNeutron"),
    ("d_n [e cm]", "AxionEDM"),
    ("d_AC [e*cm]", "AxionEDM"),
    ("d_me", "ScalarElectron"),
    ("d_g", "ScalarNucleon"),
    ("d_e", "ScalarPhoton"),
    ("f_a [GeV]", "AxionMass"),
])
def test_axis_map(label, expected):
    assert axis_implies_coupling_type(label) == expected


@pytest.mark.parametrize("label", [
    "", "coupling strength", "y-axis", "some garbled text", None,
])
def test_axis_map_fail_open(label):
    assert axis_implies_coupling_type(label) is None


# --- final347 misfire classes (2026-07-16): both must now fail open/correct ---

@pytest.mark.parametrize("label", [
    # multi-symbol combination planes -> None (measured: 1708.02111, 1604.08514)
    "GeV^-1 (this is sqrt(g_ae*g_aγ), the square-root of the product)",
    "log10 of dimensionless coupling combination d_e + 0.043(d_m̂ − d_g)",
    # bare epsilon: B−L papers use ε for their gauge coupling (2112.07687
    # wrote 'epsilon^2 (squared dimensionless coupling constant)' for ε_B-L²)
    "epsilon^2 (squared dimensionless coupling constant)",
    "dimensionless epsilon",
])
def test_axis_map_ambiguous_fails_open(label):
    assert axis_implies_coupling_type(label) is None


def test_axis_map_epsilon_context_rules():
    # ε with a B−L qualifier is the B−L gauge coupling (2403.03004).
    assert axis_implies_coupling_type(
        "epsilon_{B-L} dimensionless (gauge coupling normalized by EM coupling)"
    ) == "VectorBL"
    # ε with kinetic-mixing context is DarkPhoton (2012.05427 — the good
    # override must keep firing).
    assert axis_implies_coupling_type(
        "dimensionless kinetic mixing epsilon") == "DarkPhoton"
    # χ is unambiguous kinetic mixing on its own.
    assert axis_implies_coupling_type("χ") == "DarkPhoton"


def test_specific_symbol_beats_generic():
    # g_agamma must not be read as g_ae; d_me must not be read as d_e.
    assert axis_implies_coupling_type("g_agamma [GeV^-1]") == "AxionPhoton"
    assert axis_implies_coupling_type("d_me (dimensionless)") == "ScalarElectron"


def test_mapping_covers_every_confusable_type():
    # Completeness: each type in a confusable family must be reachable from at
    # least one axis label (else the override can never restore it).
    reachable = {axis_implies_coupling_type(l) for l in (
        "g_{B-L}", "χ", "g_agamma", "g_ae", "g_ap", "g_an", "d_n [e cm]",
        "d_me", "d_g", "d_e", "f_a [GeV]")}
    for fam in _CONFUSABLE_FAMILIES:
        # AxionCPV has no distinct axis symbol (it rides the AxionEDM/mass planes)
        for ct in fam - {"AxionCPV", "ScalarBaryon"}:
            assert ct in reachable, f"{ct} unreachable from any axis label"


# ============================ family guardrail ===============================

def test_same_confusable_family():
    assert same_confusable_family("DarkPhoton", "VectorBL")
    assert same_confusable_family("ScalarPhoton", "ScalarNucleon")
    assert same_confusable_family("AxionElectron", "AxionPhoton")
    assert not same_confusable_family("DarkPhoton", "AxionPhoton")
    assert not same_confusable_family("ScalarPhoton", "AxionEDM")
    assert same_confusable_family("DarkPhoton", "DarkPhoton")


def test_in_confusable_family():
    assert in_confusable_family("VectorBL")
    assert in_confusable_family("ScalarNucleon")
    assert not in_confusable_family("MonopoleDipole")
    assert not in_confusable_family(None)


# ============================ crosscheck action ==============================

@pytest.mark.parametrize("pred,label,action,resolved", [
    ("DarkPhoton", "g_{B-L}", "override", "VectorBL"),
    ("VectorBL", "kinetic mixing ε", "override", "DarkPhoton"),
    ("AxionPhoton", "g_ae", "override", "AxionElectron"),
    ("ScalarPhoton", "d_g", "override", "ScalarNucleon"),
    ("AxionProton", "g_an", "override", "AxionNeutron"),
    # cross-family contradiction -> review, NOT overridden
    ("AxionPhoton", "d_e", "review", "AxionPhoton"),
    ("DarkPhoton", "g_ae", "review", "DarkPhoton"),
    # consistent / unreadable -> noop
    ("AxionPhoton", "g_agamma [GeV^-1]", "noop", "AxionPhoton"),
    ("DarkPhoton", "garbled", "noop", "DarkPhoton"),
])
def test_crosscheck_actions(pred, label, action, resolved):
    ct, act, _note = axis_plane_crosscheck(pred, label)
    assert act == action
    assert ct == resolved


# ============================ extractor wiring ===============================

def test_wiring_override_relabels_in_place():
    s = {"coupling_type": "DarkPhoton", "extraction_confidence": 0.9, "notes": ""}
    action = apply_axis_crosscheck(s, "DarkPhoton", "g_{B-L}", "test")
    assert action == "override"
    assert s["coupling_type"] == "VectorBL"
    assert "Phase 3" in s["notes"]
    assert s["extraction_confidence"] == 0.9   # override does not cap confidence


def test_wiring_review_caps_confidence_keeps_type():
    s = {"coupling_type": "AxionPhoton", "extraction_confidence": 0.9, "notes": ""}
    action = apply_axis_crosscheck(s, "AxionPhoton", "d_e", "test")
    assert action == "review"
    assert s["coupling_type"] == "AxionPhoton"        # NOT overridden cross-family
    assert s["extraction_confidence"] == 0.5          # capped
    assert "[COUPLING REVIEW]" in s["notes"]


def test_wiring_noop_leaves_everything():
    s = {"coupling_type": "AxionPhoton", "extraction_confidence": 0.9, "notes": "x"}
    assert apply_axis_crosscheck(s, "AxionPhoton", "g_agamma", "t") == "noop"
    assert s == {"coupling_type": "AxionPhoton", "extraction_confidence": 0.9, "notes": "x"}
    # no axis / no type -> noop
    assert apply_axis_crosscheck(s, "AxionPhoton", "", "t") == "noop"
    assert apply_axis_crosscheck(s, None, "g_ae", "t") == "noop"
