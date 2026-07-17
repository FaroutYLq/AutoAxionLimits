"""Double-conversion guards (2026-07-16 audit) — the two unguarded conversion
paths where a model that converted DURING the read (violating the #684
plotted-values contract while truthfully declaring canonical) was re-converted:

  * eval registry: an axis-reconciled squared declaration ("(axis read-back)")
    sqrt'd already-linear emissions (2207.11968: raw 0.06 dex from GT -> 6.1
    dex). Fix: provenance-split tokens (g_squared_axis) + linear-profile
    arbitration — median inside the coupling's linear band = mislabeled
    rewrite, compare raw; outside = genuinely squared, convert.
  * runtime normalize_convention: the notes regex fired x2m_e on an
    already-canonical g_ae because the notes DISCUSSED the paper's C/F_a
    convention (1902.04246: 2.55e-10 -> 2.6e-4). Fix: convert only when the
    median is below the C/F_a input ceiling (genuine inputs ~1e-16).

Fixtures are verbatim declarations/medians from final347_remediated.

Run:  pytest evaluation/tests/test_double_conversion_guard.py -v
"""

from __future__ import annotations

import math

import pytest

from evaluation.conventions import classify_reported_convention as cls, to_canonical
from pipeline.transform_guard import normalize_convention


# ================= eval: axis-reconciled squared arbitration =================

def test_reconciled_squared_token_split():
    # "(axis read-back)" provenance -> _axis variant; model's own algebra -> base.
    assert cls("AxionElectron", "coupling squared, g^2 (axis read-back)") == "g_squared_axis"
    assert cls("AxionElectron", "(g_p^e)^2/(hbar c), dimensionless squared") == "g_squared"
    assert cls("AxionNeutron", "(g_p^n)^2/(4pi), dimensionless") == "g_squared_over_4pi"
    assert cls("AxionNeutron",
               "coupling squared over 4pi, (g)^2/(4pi) (axis read-back)") == "g_squared_over_4pi_axis"


def test_reconciled_linear_emission_compared_raw():
    # 2207.11968: emitted linear g_ae (median 8.5e-13, 0.06 dex from GT); the
    # reconciled g^2 rewrite must NOT sqrt it.
    out, note = to_canonical("AxionElectron", [(1e2, 8.5e-13)], "g_squared_axis")
    assert out[0][1] == pytest.approx(8.5e-13)
    assert "compared raw" in note
    # 2306.01048: AxionProton linear-scale 4e-6 likewise stays raw.
    out, note = to_canonical("AxionProton", [(1e0, 4e-6)], "g_squared_axis")
    assert out[0][1] == pytest.approx(4e-6)


def test_reconciled_genuine_squared_still_converts():
    # A reconciled trace whose median is BELOW the linear-profile floor is
    # genuinely squared (1508.02463-scale g^2 = 5e-16) -> sqrt.
    out, _ = to_canonical("AxionElectron", [(1e2, 5e-16)], "g_squared_axis")
    assert out[0][1] == pytest.approx(math.sqrt(5e-16))


def test_model_declared_squared_unchanged():
    # The model's OWN squared declarations keep converting regardless of scale
    # (truthful-declaration contract governs).
    out, _ = to_canonical("AxionElectron", [(1e2, 5e-16)], "g_squared")
    assert out[0][1] == pytest.approx(math.sqrt(5e-16))
    out, _ = to_canonical("AxionNeutron", [(1e-9, 7.5e-9)], "g_squared_over_4pi")
    assert out[0][1] == pytest.approx(math.sqrt(4 * math.pi * 7.5e-9))


# ================= runtime: normalize_convention magnitude guard ==============

def test_notes_mention_does_not_reconvert_canonical():
    # 1902.04246: already-canonical g_ae ~2.55e-10 + notes discussing the
    # paper's Ce/Fa convention -> conversion must NOT fire.
    pts = [(1e-23, 2.55e-10), (1e-19, 2.6e-10)]
    out, note = normalize_convention(
        "AxionElectron", pts, "",
        "abstract limit; paper reports Ce/Fa; c_e/f_a in eV^-1 convention")
    assert out[0][1] == pytest.approx(2.55e-10)
    assert note == ""


def test_genuine_cfa_input_still_converts():
    # The founding #572 case: raw C_e/F_a = 5.00e-16 -> x2m_e -> 5.11e-10.
    out, note = normalize_convention(
        "AxionElectron", [(1e-20, 5.0e-16)], "", "C_e/F_a in eV^-1")
    assert out[0][1] == pytest.approx(5.11e-10, rel=1e-3)
    assert "2 m" in note or "x1.022e+06" in note or "convention" in note


def test_nucleon_guard_symmetric():
    # Already-dimensionless g_aN scale must not get x2m_N.
    out, note = normalize_convention(
        "AxionNeutron", [(1e-10, 3e-10)], "", "C_n/F_a in eV^-1 noted")
    assert out[0][1] == pytest.approx(3e-10)
    # genuine tiny C_N/F_a converts.
    out, _ = normalize_convention(
        "AxionNeutron", [(1e-10, 1e-18)], "", "C_n/F_a in eV^-1")
    assert out[0][1] == pytest.approx(1e-18 * 2 * 9.395654205e8)
