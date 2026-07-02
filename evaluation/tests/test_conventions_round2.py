"""Tests for the round-2 convention converters (post-full346 Phase 1d).

Source of truth: GPD/explanations/coupling-convention-conversions-round2-EXPLAIN.md
(citation-audited; every factor numerically spot-checked against its paper/GT
pair). Mirrors the round-1 suite in test_convention_guard.py.

Run:
    pytest evaluation/tests/test_conventions_round2.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.conventions import (
    GUARD_REFUSED,
    UNCONVERTIBLE,
    classify_reported_convention,
    file_source_convention,
    to_canonical,
)


def _y(points):
    return [g for _m, g in points]


# ---------------------------------------------------------------------------
# Family 1 — f_a [GeV] -> 1/f_a (AxionMass)
# ---------------------------------------------------------------------------

def test_fa_gev_classify_and_invert():
    # 2105.13963's observed declaration.
    tok = classify_reported_convention(
        "AxionMass", "f_a in GeV vs m_a in eV (excluded band 1.6e16 < f_a < 1e18 GeV)")
    assert tok == "f_a_gev"
    out, note = to_canonical("AxionMass", [(1e-12, 1.6e16)], tok)
    assert out[0][1] == pytest.approx(6.25e-17)
    assert "reciprocal" in note


def test_fa_gev_guard_refuses_snapped_values():
    # The full346 snapshots carry anchor-snapped garbage (1.6e-4..1e-2); the
    # magnitude guard must refuse rather than "convert" corrupted values.
    out, note = to_canonical("AxionMass", [(1e-12, 1.6e-4), (1e-13, 1e-2)], "f_a_gev")
    assert note.startswith(GUARD_REFUSED)
    assert _y(out) == [1.6e-4, 1e-2]  # unchanged


def test_fa_gev_mislabel_with_canonical_values_compared_raw():
    # Observed repeatedly in full346 (2410.19902, 1708.07521, ...): the
    # declaration says f_a in GeV but the emitted values are already on the
    # canonical 1/f_a scale (<1e-3, provably not a decay constant). The
    # registry treats that as a mislabeled declaration and leaves the values
    # untouched instead of excluding a good comparison.
    out, note = to_canonical("AxionMass", [(1e-12, 9e-9)], "f_a_gev")
    assert _y(out) == [9e-9]
    assert "mislabeled" in note and not note.startswith(GUARD_REFUSED)


def test_inverse_fa_declaration_stays_canonical():
    assert classify_reported_convention("AxionMass", "1/f_a [GeV^-1]") is None


# ---------------------------------------------------------------------------
# Family 2 — decay rate / lifetime plane (AxionPhoton)
# ---------------------------------------------------------------------------

def test_decay_rate_conversion_matches_paper_spot_check():
    # 2301.06560: Gamma = 2.2e-24 s^-1 at m = 10 eV -> g = 1.706e-11 GeV^-1.
    tok = classify_reported_convention("AxionPhoton", "decay rate Gamma in s^-1, not coupling")
    assert tok == "decay_rate_s_inv"
    out, _ = to_canonical("AxionPhoton", [(10.0, 2.2e-24)], tok)
    assert out[0][1] == pytest.approx(1.706e-11, rel=1e-3)


def test_lifetime_conversion_is_reciprocal_of_decay_rate():
    tok = classify_reported_convention("AxionPhoton", "s")
    assert tok == "lifetime_s"
    tau = 1.0 / 2.2e-24
    out, _ = to_canonical("AxionPhoton", [(10.0, tau)], tok)
    assert out[0][1] == pytest.approx(1.706e-11, rel=1e-3)


def test_decay_rate_guard_refuses_non_rate_values():
    out, note = to_canonical("AxionPhoton", [(10.0, 1e-3)], "decay_rate_s_inv")
    assert note.startswith(GUARD_REFUSED)


def test_lifetime_guard_refuses_small_values():
    out, note = to_canonical("AxionPhoton", [(10.0, 42.0)], "lifetime_s")
    assert note.startswith(GUARD_REFUSED)


def test_canonical_g_agamma_declaration_not_reclassified():
    assert classify_reported_convention("AxionPhoton", "g_agamma [GeV^-1]") is None


# ---------------------------------------------------------------------------
# Family 3 — squared axes: sqrt(4pi*y) vs sqrt(y), 0.55 dex apart
# ---------------------------------------------------------------------------

def test_squared_over_4pi_token_and_value():
    tok = classify_reported_convention("AxionNeutron", "(g_p^n)^2/(4pi), dimensionless")
    assert tok == "g_squared_over_4pi"
    out, _ = to_canonical("AxionNeutron", [(1e-10, 5.5e-10)], tok)
    assert out[0][1] == pytest.approx(math.sqrt(4 * math.pi * 5.5e-10))
    assert out[0][1] == pytest.approx(8.31e-5, rel=1e-2)  # doc spot-check


def test_plain_squared_token_and_value():
    # 1508.02463's observed declaration (no 4pi).
    tok = classify_reported_convention(
        "AxionElectron", "(g_p^e)^2/(hbar c), dimensionless coupling-squared")
    assert tok == "g_squared"
    out, _ = to_canonical("AxionElectron", [(1e-10, 5.5e-17)], tok)
    assert out[0][1] == pytest.approx(7.42e-9, rel=1e-2)  # doc spot-check


def test_two_squared_tokens_differ_by_sqrt_4pi():
    a, _ = to_canonical("AxionElectron", [(1.0, 1e-10)], "g_squared")
    b, _ = to_canonical("AxionElectron", [(1.0, 1e-10)], "g_squared_over_4pi")
    assert b[0][1] / a[0][1] == pytest.approx(math.sqrt(4 * math.pi))


def test_converted_from_declaration_never_reconverted():
    # 0809.4700's actual string: claims converted while the data were raw —
    # the token must describe the EMITTED values; "converted from" strings are
    # canonical-claimed (mislabels are the #594 extractor contract's problem).
    tok = classify_reported_convention(
        "AxionNeutron", "dimensionless g_an, converted from (g_p^n)^2/(4pi)<5.8e-10")
    assert tok is None


# ---------------------------------------------------------------------------
# Family 4 — thermal-axion xi (model-locked)
# ---------------------------------------------------------------------------

def test_xi_thermal_classify_and_convert():
    # astro-ph/0611502's observed declaration; note it also mentions
    # "NOT g_agamma in GeV^-1", which must not shadow the xi token.
    tok = classify_reported_convention(
        "AxionPhoton", "dimensionless xi (4/3*(E/N-1.92)), NOT g_agamma in GeV^-1")
    assert tok == "xi_thermal"
    out, note = to_canonical("AxionPhoton", [(4.5, 7.17e-3)], tok)
    assert out[0][1] == pytest.approx(4.50e-12, rel=1e-2)  # doc spot-check
    assert "model-locked" in note


def test_xi_word_boundary_does_not_fire_on_axion():
    assert classify_reported_convention(
        "AxionPhoton", "axion photon coupling") is None


def test_xi_guard_refuses_outside_optical_window():
    out, note = to_canonical("AxionPhoton", [(1e-6, 5e-3)], "xi_thermal")
    assert note.startswith(GUARD_REFUSED)


# ---------------------------------------------------------------------------
# Family 5 — AxionEDM ordering + UNCONVERTIBLE sentinel preserved
# ---------------------------------------------------------------------------

def test_edm_gev2_checked_before_ecm():
    # 1401.6460 declares 'gd in GeV^-2 ... converted' — must stay canonical
    # (scoreable arithmetic error), not excluded.
    assert classify_reported_convention(
        "AxionEDM", "gd in GeV^-2 (converted from d_n in e*cm)") is None


def test_edm_oscillating_amplitude_still_unconvertible():
    assert classify_reported_convention(
        "AxionEDM", "oscillating neutron EDM amplitude d_n in e*cm") == UNCONVERTIBLE


# ---------------------------------------------------------------------------
# Family 6 — scalar Lambda scale, magnitude-directed
# ---------------------------------------------------------------------------

def test_scalar_inverse_values_multiplied():
    f = math.sqrt(2) * 2.4e18
    out, _ = to_canonical("ScalarPhoton", [(1e-22, 9.1e-31)], "gev_inv_scalar")
    assert out[0][1] == pytest.approx(9.1e-31 * f)


def test_scalar_lambda_in_gev_reciprocated():
    # QSNET back-check: Lambda_gamma = 1.10e30 GeV -> d_e ~ 3.1e-12.
    f = math.sqrt(2) * 2.4e18
    lam = f / 3.121e-12
    out, note = to_canonical("ScalarPhoton", [(1e-22, lam)], "gev_inv_scalar")
    assert out[0][1] == pytest.approx(3.121e-12, rel=1e-6)
    assert "Lambda" in note


def test_scalar_ambiguous_magnitude_refused():
    out, note = to_canonical("ScalarPhoton", [(1e-22, 5.0)], "gev_inv_scalar")
    assert note.startswith(GUARD_REFUSED)


# ---------------------------------------------------------------------------
# Round-1 registry bug fix — per-file dimensionless overrides
# ---------------------------------------------------------------------------

def test_dimensionless_nucleon_files_not_inflated():
    # These files store dimensionless g_aN (no in-code multiplier); the family
    # default x2 m_N inflated their GT side by +0.27 dex.
    for f, ct in [
        ("limit_data/AxionNeutron/K-3He_Comagnetometer.txt", "AxionNeutron"),
        ("limit_data/AxionNeutron/TorsionBalance.txt", "AxionNeutron"),
        ("limit_data/AxionNeutron/129Xe.txt", "AxionNeutron"),
        ("limit_data/AxionNeutron/Casimir.txt", "AxionNeutron"),
        ("limit_data/AxionNeutron/SN1987A.txt", "AxionNeutron"),
        ("limit_data/AxionNeutron/NeutronStars.txt", "AxionNeutron"),
        ("limit_data/AxionProton/TorsionBalance.txt", "AxionProton"),
        ("limit_data/AxionProton/Casimir.txt", "AxionProton"),
        ("limit_data/AxionProton/SN1987A.txt", "AxionProton"),
        ("limit_data/AxionProton/NeutronStars.txt", "AxionProton"),
        ("limit_data/AxionProton/Projections/MnCO3.txt", "AxionProton"),
    ]:
        assert file_source_convention(f, ct) is None, f
        assert (PROJECT_ROOT / f).exists(), f


def test_family_default_and_sno_override_unchanged():
    assert file_source_convention(
        "limit_data/AxionNeutron/nEDM.txt", "AxionNeutron") == "g_aNN_inv_gev"
    assert file_source_convention(
        "limit_data/AxionNeutron/SNO.txt", "AxionNeutron") == "g_aN_over_mN_inv_gev"
