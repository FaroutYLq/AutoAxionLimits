"""Catastrophic-tail convention guards (2026-07-15 audit, post-#625 phases).

Three fail-closed levers, each anchored to a measured multi-dex "compared"
paper in full346_postfix_opus:

  A1 — f_a_gev AXIS-READ-BACK contradiction (1708.08464 @10.5, 2105.13963 @13.0
       dex): the stage-2a read-back MEASURED an f_a [GeV] axis, so canonical-
       scale values contradict the measurement — the registry's mislabel escape
       (med < 1e-3 -> compare raw) must not apply. A MODEL-declared f_a keeps
       the escape (1708.07521 compares at 0.13 dex).
  A2 — VectorBL fifth-force |alpha-tilde| (1207.2442 @13.4 dex): a Yukawa
       strength relative to gravity, not the gauge coupling g_BL;
       "dimensionless" matched canonical vocabulary so it compared raw.
  A3 — AxionMass epsilon-bias plane (2410.21590 @8.35 dex):
       epsilon=(m_a/m_aQCD)^2 slips every screen ("m_a" is an expected stem).

Both mirrors (eval registry + runtime review flag) are pinned, plus the
canonical/convertible cases that must stay untouched.

Run:  pytest evaluation/tests/test_tail_guards.py -v
"""

from __future__ import annotations

import pytest

from evaluation.conventions import (
    GUARD_REFUSED,
    UNCONVERTIBLE,
    classify_reported_convention,
    to_canonical,
)
from pipeline.transform_guard import (
    convention_review_needed,
    convertible_out_of_profile,
)


# ============================ A1 — f_a_gev axis read-back =====================

_AXIS_DECL = "f_a in GeV (axis read-back)"


def test_a1_axis_readback_routes_to_axis_token():
    assert classify_reported_convention("AxionMass", _AXIS_DECL) == "f_a_gev_axis"
    # model-declared f_a keeps the plain token
    assert classify_reported_convention(
        "AxionMass", "mass_ev = m_a in ev, coupling = f_a in gev") == "f_a_gev"


def test_a1_axis_token_refuses_canonical_scale_values():
    # 1708.08464 / 2105.13963: median 1e-5 on a measured f_a[GeV] axis.
    _o, note = to_canonical("AxionMass", [(1e-20, 1e-5), (1e-19, 1e-5)],
                            "f_a_gev_axis")
    assert note.startswith(GUARD_REFUSED)


def test_a1_axis_token_converts_genuine_decay_constant():
    out, note = to_canonical("AxionMass", [(1e-20, 1e12)], "f_a_gev_axis")
    assert not note.startswith(GUARD_REFUSED)
    assert out[0][1] == pytest.approx(1e-12)


def test_a1_model_declared_mislabel_escape_intact():
    # 1708.07521 (0.13 dex): model-declared f_a with canonical-scale values
    # keeps the compare-raw escape.
    _o, note = to_canonical("AxionMass", [(1e-20, 2.2e-5)], "f_a_gev")
    assert not note.startswith(GUARD_REFUSED)
    assert "mislabeled declaration" in note


def test_a1_runtime_mirror():
    assert convertible_out_of_profile("AxionMass", _AXIS_DECL,
                                      [(1e-20, 1e-5)]) is True
    assert convertible_out_of_profile("AxionMass", _AXIS_DECL,
                                      [(1e-20, 1e12)]) is False
    # model-declared: never fires (registry escape owns it)
    assert convertible_out_of_profile("AxionMass", "coupling = f_a in gev",
                                      [(1e-20, 2.2e-5)]) is False


# ============================ A2 — VectorBL alpha-tilde =======================

_D1207 = "|alpha-tilde|, dimensionless strength of the B-L vector Yukawa interaction"


def test_a2_alpha_tilde_fails_closed_both_mirrors():
    assert classify_reported_convention("VectorBL", _D1207) == UNCONVERTIBLE
    assert convention_review_needed("VectorBL", _D1207) is True


@pytest.mark.parametrize("decl", [
    "dimensionless g_BL (B-L gauge coupling)",
    "g_{B-L}, dimensionless gauge coupling as plotted",
    "dimensionless g_BL = gB-L(hbar c)^-1/2",
    "epsilon_{B-L}^{95%} = g_BL, dimensionless B-L gauge coupling",
])
def test_a2_canonical_vectorbl_untouched(decl):
    assert classify_reported_convention("VectorBL", decl) is None
    assert convention_review_needed("VectorBL", decl) is False


def test_a2_scoped_to_vectorbl_scalar_alpha_intact():
    # The per-coupling "alpha" foreign token is scoped to VectorBL: a Scalar*
    # alpha declaration must NOT become UNCONVERTIBLE (classify keeps its
    # pre-existing None), and the GT-side alpha_fifthforce converter still
    # converts.
    # (avoid "force strength" — a pre-existing GLOBAL foreign-class token).
    # The invariant this PR must keep: the per-CT "alpha" token does NOT make a
    # Scalar* alpha declaration UNCONVERTIBLE on the eval side (classify keeps
    # its pre-existing None). The runtime already review-flagged ScalarNucleon
    # alpha declarations before this PR (not canonical, not convertible) —
    # unchanged, not asserted here.
    decl = "Yukawa alpha relative to gravity"
    assert classify_reported_convention("ScalarNucleon", decl) is None
    out, note = to_canonical("ScalarNucleon", [(1e-10, 1e-6)], "alpha_fifthforce")
    assert note.startswith("convention:") and out[0][1] == pytest.approx(500 * 1e-3)


# ============================ A3 — AxionMass epsilon-bias =====================

_D2410 = ("epsilon = (m_a/m_a^QCD)^2 parameter (dimensionless), NOT f_a in GeV; "
          "values as plotted")


def test_a3_epsilon_bias_fails_closed_both_mirrors():
    assert classify_reported_convention("AxionMass", _D2410) == UNCONVERTIBLE
    assert convention_review_needed("AxionMass", _D2410) is True


def test_a3_bias_parameter_spelling():
    d = "bias parameter xi (dimensionless), plotted vs m_a"
    assert classify_reported_convention("AxionMass", d) == UNCONVERTIBLE
    assert convention_review_needed("AxionMass", d) is True


@pytest.mark.parametrize("decl,expect_tok", [
    ("1/f_a in gev^-1 (inverse axion decay constant)", None),
    ("f_a in gev (axion decay constant); excluded region", "f_a_gev"),
    ("dimensionless normalized coupling", None),
])
def test_a3_canonical_axionmass_untouched(decl, expect_tok):
    assert classify_reported_convention("AxionMass", decl) == expect_tok
    assert convention_review_needed("AxionMass", decl) is False
