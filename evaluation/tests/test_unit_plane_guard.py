"""Phase 1 (#625) — fail-closed unit-plane guards, eval side.

Two mechanisms, both pure functions over the model's own declared output:

  * 1a — unit-PLANE screen: a declaration whose EXPLICIT unit power contradicts
    the coupling's canonical plane, and that no vetted converter serves, is
    UNCONVERTIBLE even when it carries a canonical SYMBOL ("epsilon in GeV^-1").
    Token matching sees the plane vocabulary but not the unit power; this closes
    that blind spot. A declaration with NO explicit unit token is untouched.

  * 1b — inv_fa magnitude guard: the ONE vetted converter shipped without the
    round-2 plausible-range guard. A mislabeled oscillating-EDM d_n [e*cm]
    amplitude (~1e-28) declared "C_G/f_a in GeV^-1" was x3.7e-3'd into a
    "compared" 16-dex garbage curve (2204.01454, 2410.02218). The guard refuses
    input outside the plausible 1/f_a band, so it becomes convention_mismatch.

Fixtures are the VERBATIM declarations + medians from the full346_postfix_opus /
final2_opus_n1 snapshots.

Run:  pytest evaluation/tests/test_unit_plane_guard.py -v
"""

from __future__ import annotations

import pytest

from evaluation.conventions import (
    GUARD_REFUSED,
    UNCONVERTIBLE,
    classify_reported_convention,
    to_canonical,
)


# --- Verbatim declarations from the benchmark snapshots -----------------------
_EDM_GARBAGE_DECL = "C_G/f_a in GeV^-1 (ALP-gluon coupling over decay constant)"
_EDM_GENUINE_DECL = "C_G/f_a in GeV^-1 as plotted on y-axis"


# ============================ 1b — inv_fa guard ==============================

@pytest.mark.parametrize("median", [5.4e-28, 1.5e-28, 2.0e-28, 1e-26])
def test_invfa_refuses_ecm_amplitude_mislabeled_as_gev_inv(median):
    """A ~1e-28 median is the e*cm amplitude, not a 1/f_a in GeV^-1 (which needs
    f_a in [1e7,1e20] GeV, i.e. 1/f_a in [1e-20,1e-7]). Refuse -> convention gap."""
    tok = classify_reported_convention("AxionEDM", _EDM_GARBAGE_DECL)
    assert tok == "inv_fa"          # the declaration routes to the converter...
    pts = [(1e-15, median), (1e-14, median * 1.2)]
    _out, note = to_canonical("AxionEDM", pts, tok)
    assert note.startswith(GUARD_REFUSED)   # ...but the magnitude guard refuses it


def test_invfa_converts_genuine_decay_constant():
    """2204.01454's legitimate final2 read: median 2e-13 = 1/(5e12 GeV), inside
    the band -> convert x3.7e-3, stays compared (2.79 dex)."""
    tok = classify_reported_convention("AxionEDM", _EDM_GENUINE_DECL)
    assert tok == "inv_fa"
    pts = [(1e-15, 2e-13), (1e-14, 2e-13)]
    out, note = to_canonical("AxionEDM", pts, tok)
    assert not note.startswith(GUARD_REFUSED)
    assert out[0][1] == pytest.approx(2e-13 * 3.7e-3)


def test_invfa_band_edges():
    for med, refused in [(1e-7, False), (1e-8, False), (1e-20, False),
                         (1e-6, True), (1e-21, True)]:
        pts = [(1e-15, med)]
        _o, note = to_canonical("AxionEDM", pts, "1/f_a")
        assert note.startswith(GUARD_REFUSED) is refused, (med, note)


# ============================ 1a — unit-plane screen ==========================

def test_1a_flags_canonical_symbol_with_wrong_unit():
    # DarkPhoton canonical plane is dimensionless; "epsilon in GeV^-1" carries
    # the canonical symbol but an inverse-energy unit -> UNCONVERTIBLE.
    assert classify_reported_convention("DarkPhoton", "epsilon in GeV^-1") == UNCONVERTIBLE
    # AxionPhoton canonical is GeV^-1; a GeV^-2 declaration is the wrong power.
    assert classify_reported_convention("AxionPhoton", "g_agamma in GeV^-2") == UNCONVERTIBLE


def test_1a_passes_canonical_and_unitless():
    # Canonical declarations (unit == canonical plane) are untouched.
    assert classify_reported_convention("AxionPhoton", "g_agamma [GeV^-1]") is None
    assert classify_reported_convention("AxionEDM", "g_d in GeV^-2") is None
    assert classify_reported_convention("DarkPhoton", "kinetic mixing chi (dimensionless)") is None
    # No explicit unit token -> untouched (no new false flags).
    assert classify_reported_convention("DarkPhoton", "epsilon, kinetic mixing") is None
    assert classify_reported_convention("AxionElectron", "g_ae (dimensionless)") is None


def test_1a_does_not_flag_convertible_contradiction():
    # AxionEDM "C_G/f_a in GeV^-1" contradicts the GeV^-2 canonical plane, but a
    # vetted converter (inv_fa) serves it -> 1a defers to the converter (1b
    # guards the magnitude). It must NOT be pre-empted to UNCONVERTIBLE by 1a.
    assert classify_reported_convention("AxionEDM", _EDM_GENUINE_DECL) == "inv_fa"
    # Nucleon GeV^-1 derivative coupling is a vetted alternate, not a flag.
    assert classify_reported_convention("AxionNeutron", "g_ann in GeV^-1") == "g_aNN_inv_gev"


def test_1a_axionmass_multiplane_not_flagged():
    # AxionMass is legitimately multi-plane (dimensionless f_a_norm, 1/f_a
    # [GeV^-1], f_a [GeV]) — a dimensionless declaration must NOT be flagged
    # (1606.07494 theta_0 stays whatever the core decides, not a false 1a flag).
    assert classify_reported_convention(
        "AxionMass", "dimensionless normalized coupling") is None


def test_1a_decay_rate_plane_excluded():
    # 2301.06560: "decay rate Gamma in s^-1 (NOT g_agamma)". The core short-
    # circuits to None on the g_agamma substring; baseline scored it as raw
    # 12-dex garbage. 1a sees s^-1 != GeV^-1 -> UNCONVERTIBLE (convention gap).
    decl = ("decay rate Gamma_{a->gamma gamma} in s^-1 (NOT g_agamma; Gamma "
            "proportional to g_agamma^2 * m_a^3). Values emitted exactly as "
            "plotted on the y-axis.")
    assert classify_reported_convention("AxionPhoton", decl) == UNCONVERTIBLE
