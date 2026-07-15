"""Phase 2 (#625) — mass-dependent d_n [e*cm] -> g_{anγ} [GeV^-2] converter.

The first mass-dependent converter in the registry. Physics + derivation:
GPD/explanations/convention-dn-ecm-g_angamma.md
  g_{anγ}[GeV^-2] = C * d_n[e*cm] * m_a[eV],  C = 6.19e24
via d_n = g_{anγ} * a_0, a_0 = sqrt(2 rho_DM)/m_a.

Also pins 2a's contract: existing constant converters are UNCHANGED.

Run:  pytest evaluation/tests/test_dn_ecm_converter.py -v
"""

from __future__ import annotations

import math

import pytest

from evaluation.conventions import (
    GUARD_REFUSED,
    UNCONVERTIBLE,
    _K_DN_ECM,
    classify_reported_convention,
    to_canonical,
)


# ============================ routing =========================================

@pytest.mark.parametrize("decl", [
    "d_n in e*cm",
    "oscillating neutron EDM amplitude d_n in e*cm (read from Fig. 2)",
    "d_AC (deuteron oscillating EDM) in e*cm",
    "d_n oscillation amplitude in e cm",
])
def test_ecm_declarations_route_to_dn_ecm(decl):
    assert classify_reported_convention("AxionEDM", decl) == "d_n_ecm"


def test_bare_dn_without_ecm_unit_stays_unconvertible():
    # The calibration is e*cm-specific; a d_n symbol with no e*cm unit is not
    # routed to the converter.
    assert classify_reported_convention("AxionEDM", "d_n (bare)") == UNCONVERTIBLE


# ============================ conversion physics ==============================

def test_constant_value():
    # C = e*(cm in GeV^-1)*1e-9 / sqrt(2 rho_DM), rho_DM = 0.4 GeV/cm^3.
    assert _K_DN_ECM == pytest.approx(6.19e24, rel=0.02)


def test_linear_in_mass_and_dn():
    # g_angamma = C * d_n * m_a — doubling either input doubles the output.
    out1, _ = to_canonical("AxionEDM", [(1e-20, 1e-24)], "d_n_ecm")
    out2, _ = to_canonical("AxionEDM", [(2e-20, 1e-24)], "d_n_ecm")
    out3, _ = to_canonical("AxionEDM", [(1e-20, 2e-24)], "d_n_ecm")
    assert out2[0][1] == pytest.approx(2 * out1[0][1])
    assert out3[0][1] == pytest.approx(2 * out1[0][1])
    # exact formula
    assert out1[0][1] == pytest.approx(_K_DN_ECM * 1e-24 * 1e-20)


def test_spot_check_against_nEDM_gt():
    # 1708.06367 (nEDM): extracted d_n points; converted g_angamma must land on
    # the paper's own g_angamma GT curve. Anchor at a representative point:
    # m=4.1e-22 eV, d_n=5e-26 e*cm -> g ~ 1.3e-22; GT ~ 2.0e-22 (0.2 dex).
    out, note = to_canonical("AxionEDM", [(4.1e-22, 5e-26)], "d_n_ecm")
    g = out[0][1]
    assert note.startswith("convention: d_n [e*cm]")
    assert abs(math.log10(g / 1.99e-22)) < 0.3


def test_spot_check_against_JEDI_gt():
    # 2208.07293 (JEDI deuteron): m=4.95e-10 eV, d_AC=6.4e-23 e*cm -> g ~ 2.0e-7;
    # GT ~ 1.47e-7 (0.13 dex).
    out, _ = to_canonical("AxionEDM", [(4.95e-10, 6.4e-23)], "d_n_ecm")
    assert abs(math.log10(out[0][1] / 1.47e-7)) < 0.3


# ============================ magnitude guard =================================

@pytest.mark.parametrize("dn,refused", [
    (3.5e-26, False),   # genuine nEDM amplitude
    (6.4e-23, False),   # genuine JEDI amplitude
    (1.0,     True),    # absurd — not an e*cm EDM amplitude
    (1e-40,   True),    # below the plausible floor
])
def test_magnitude_guard(dn, refused):
    _out, note = to_canonical("AxionEDM", [(1e-15, dn), (1e-14, dn)], "d_n_ecm")
    assert note.startswith(GUARD_REFUSED) is refused


# ============================ 2a: existing converters unchanged ===============

def test_existing_constant_converters_unchanged():
    # inv_fa (in-band), squared, eps^2, f_a reciprocal — pin exact behaviour so
    # the mass-dependent extension does not perturb the constant converters.
    o, _ = to_canonical("AxionEDM", [(1e-15, 1e-12)], "1/f_a")
    assert o[0][1] == pytest.approx(1e-12 * 3.7e-3)
    o, _ = to_canonical("AxionNeutron", [(1e-15, 4.0)], "g_squared")
    assert o[0][1] == pytest.approx(2.0)
    o, _ = to_canonical("DarkPhoton", [(1e-9, 4e-20)], "epsilon_squared")
    assert o[0][1] == pytest.approx(2e-10)
    o, _ = to_canonical("AxionMass", [(1e-9, 1e12)], "f_a_gev")
    assert o[0][1] == pytest.approx(1e-12)
