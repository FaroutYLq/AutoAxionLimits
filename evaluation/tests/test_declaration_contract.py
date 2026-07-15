"""Tests for the #594 truthful-declaration contract (post-full346 follow-up).

The eval registry converts based on what the extractor DECLARES its emitted
values to be; a false declaration is unfixable at scoring time. This layer
makes false declarations structurally harder:

* stage-1 prompt carries the DECLARATION CONTRACT;
* on a vision win, the stage-2a axis read-back (a measurement) overrides a
  canonical-claiming declaration (2301.06560: Gamma values declared
  'GeV^-1 g_agamma');
* candidates whose effective declared convention failed convention review are
  demoted in the selector (1708.06367: the flagged e*cm text read beat the
  vision trace of the figure already in g_d);
* transform_guard consistency: e*cm is no longer "canonical" for AxionEDM
  (#604), and the _declared_convertible mirror knows the round-1+2 registry.

No API. Run:
    pytest evaluation/tests/test_declaration_contract.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

anthropic = pytest.importorskip("anthropic")

from pipeline.extractor import (
    _STAGE1_SYSTEM,
    _axis_conv_hint,
    _make_candidate,
    _reconcile_declared_convention,
)
from pipeline.transform_guard import convention_review_needed, quality


# ---------------------------------------------------------------------------
# Axis hints
# ---------------------------------------------------------------------------

def test_axis_hint_decay_rate_and_lifetime():
    decl, tok = _axis_conv_hint("AxionPhoton", "Gamma_{a->gamma gamma} [s^-1]")
    assert "s^-1" in decl and tok == "s^-1"
    decl, tok = _axis_conv_hint("AxionPhoton", "lifetime tau_95 [s]")
    assert "lifetime" in decl


def test_axis_hint_canonical_axis_is_none():
    assert _axis_conv_hint("AxionPhoton", "g_agamma [GeV^-1]") is None
    assert _axis_conv_hint("AxionPhoton", "10^{-10} GeV^-1") is None
    assert _axis_conv_hint("AxionEDM", "g_d [GeV^-2]") is None
    assert _axis_conv_hint("AxionPhoton", "") is None
    assert _axis_conv_hint(None, "s^-1") is None


def test_axis_hint_squared_and_ecm_and_fa():
    decl, _ = _axis_conv_hint("AxionNeutron", "(g_p^n)^2/(4pi)")
    assert "4pi" in decl
    decl, _ = _axis_conv_hint("AxionElectron", "(g_p^e)^2/hbar c")
    assert "squared" in decl
    decl, _ = _axis_conv_hint("AxionEDM", "d_n [e cm]")
    assert "e*cm" in decl
    decl, _ = _axis_conv_hint("AxionMass", "f_a [GeV]")
    assert "f_a in GeV" in decl
    assert _axis_conv_hint("AxionMass", "1/f_a [GeV^-1]") is None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _vision_result(declared, axis_unit, ct="AxionPhoton"):
    return {"data_source": "figure_vision", "coupling_type": ct,
            "coupling_convention": declared, "_axis_y_unit": axis_unit,
            "notes": ""}


def test_reconciliation_overrides_false_canonical_claim():
    # 2301.06560 shape: Gamma values declared as canonical g_agamma.
    r = _vision_result("GeV^-1 g_agamma", "Gamma [s^-1]")
    _reconcile_declared_convention(r)
    assert "s^-1" in r["coupling_convention"]
    assert "reconciled" in r["notes"]


def test_reconciliation_overrides_empty_declaration():
    r = _vision_result("", "Gamma [s^-1]")
    _reconcile_declared_convention(r)
    assert "s^-1" in r["coupling_convention"]


def test_reconciliation_keeps_truthful_declaration():
    r = _vision_result("decay rate Gamma in s^-1, not coupling", "Gamma [s^-1]")
    _reconcile_declared_convention(r)
    assert r["coupling_convention"] == "decay rate Gamma in s^-1, not coupling"
    assert r["notes"] == ""


def test_reconciliation_noop_for_text_source_and_canonical_axis():
    r = _vision_result("GeV^-1 g_agamma", "Gamma [s^-1]")
    r["data_source"] = "text"
    _reconcile_declared_convention(r)
    assert r["coupling_convention"] == "GeV^-1 g_agamma"  # not a vision win
    r2 = _vision_result("GeV^-1 g_agamma", "g_agamma [GeV^-1]")
    _reconcile_declared_convention(r2)
    assert r2["notes"] == ""  # canonical axis: nothing to reconcile


# ---------------------------------------------------------------------------
# Selector demotion (1708.06367 shape)
# ---------------------------------------------------------------------------

def test_flagged_convention_candidate_loses_to_known_convention():
    pts_text = [(10 ** (-23 + 0.2 * i), 10 ** (-25.5 - 0.02 * i)) for i in range(6)]
    pts_vis = [(10 ** (-23 + 0.2 * i), 10 ** (-20 - 0.1 * i)) for i in range(20)]
    # Text read: a genuinely unknown/unconvertible convention (e*cm is now the
    # d_n_ecm converter, so use a bespoke novel plane that still flags).
    text_c = _make_candidate("text", pts_text, "AxionEDM", 0.9,
                             convention_flagged=convention_review_needed(
                                 "AxionEDM", "bespoke novel amplitude, non-standard plane"))
    # Vision read of the reinterpretation figure already in g_d [GeV^-2].
    vis_c = _make_candidate("figure_vision", pts_vis, "AxionEDM", 0.5,
                            convention_flagged=convention_review_needed(
                                "AxionEDM", None))
    assert text_c.convention_flagged is True
    assert vis_c.convention_flagged is False
    assert quality(vis_c) > quality(text_c)


def test_unflagged_text_still_beats_vision():
    pts_text = [(10 ** (-6 + 0.1 * i), 10 ** (-12 - 0.05 * i)) for i in range(20)]
    pts_vis = [(10 ** (-6 + 0.1 * i), 10 ** (-11.9 - 0.05 * i)) for i in range(20)]
    text_c = _make_candidate("text", pts_text, "AxionPhoton", 0.9)
    vis_c = _make_candidate("figure_vision", pts_vis, "AxionPhoton", 0.5)
    assert quality(text_c) > quality(vis_c)


# ---------------------------------------------------------------------------
# transform_guard consistency
# ---------------------------------------------------------------------------

def test_ecm_now_convertible_for_axionedm():
    # Phase 2 (#625): e*cm is the mass-dependent d_n_ecm converter -> not flagged.
    assert convention_review_needed("AxionEDM", "d_n in e cm") is False
    assert convention_review_needed("AxionEDM", "g_d [GeV^-2]") is False


def test_mirror_knows_round2_registry():
    # Convertible alternates must NOT be flagged (they'd over-queue #656).
    for ct, decl in [
        ("AxionPhoton", "decay rate Gamma in s^-1"),
        ("AxionPhoton", "lifetime tau in s"),
        ("AxionMass", "f_a in GeV vs m_a in eV"),
        ("AxionElectron", "(g_p^e)^2/(hbar c), dimensionless coupling-squared"),
        ("ScalarPhoton", "Lambda_gamma^-1 in GeV^-1"),
    ]:
        assert convention_review_needed(ct, decl) is False, (ct, decl)


def test_prompt_contains_declaration_contract():
    assert "DECLARATION CONTRACT" in _STAGE1_SYSTEM
    assert "NEVER write \"converted from X\"" in _STAGE1_SYSTEM
