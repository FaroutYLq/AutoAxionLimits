"""Convention vocabulary hardening (2026-07-14): the final2_opus_n1 flag audit
found 7 of 20 [CONVENTION REVIEW] firings were canonical physics phrased with
family notation outside the vetted stems (ALP field named phi/chi, Z' for the
B-L vector, a d_me definition gloss, an explicit equivalence assertion).

These tests pin BOTH mirrors (pipeline.transform_guard and
evaluation.conventions) on the exact declarations from the benchmark run:
the 5 vocabulary fixes must un-flag, the genuinely-unknown declarations must
STAY flagged (#683: vocabulary must never suppress review flags), and the two
mirrors must agree verdict-for-verdict on the whole corpus.
"""

import pytest

from pipeline.transform_guard import (
    convention_review_needed,
    _foreign_quantity_declared as tg_foreign,
)
from evaluation.conventions import (
    UNCONVERTIBLE,
    classify_reported_convention,
    _foreign_quantity_declared as ev_foreign,
)


# ---------------------------------------------------------------------------
# The real declarations from final2_opus_n1 (verbatim, lowercased by callee)
# ---------------------------------------------------------------------------

# Canonical physics, family notation — must NOT flag after the vocab fix.
UNFLAGGED = [
    ("AxionPhoton",
     "c_gamma/Lambda in GeV^-1 (equivalent to g_agamma in GeV^-1)"),      # 1903.03586
    ("AxionPhoton",
     "g_agamma (g_chi-gamma-gamma) in GeV^-1, as plotted"),               # 2205.01079
    ("AxionPhoton",
     "g_phi_gamma_gamma in GeV^-1, ALP-photon coupling as plotted"),      # 2503.04726
    ("AxionPhoton",
     "g_agamma in GeV^-1 (approximate, derived from 0.93*g_KSVZ)"),       # 2312.11003
    ("VectorBL",
     "g_Z' (g_BL), dimensionless gauge coupling as plotted"),             # 2304.12907
    ("ScalarElectron",
     "dimensionless d_me = g_e * M_Pl / m_e (approximate axis-read values)"),  # 2201.02042
]

# Genuinely unknown/foreign — must STAY flagged (and stay UNCONVERTIBLE or
# review-needed on the eval side where applicable).
# NOTE: 1604.08514 (clock combo) and 2303.00778 (Higgs-portal sin theta) sat
# in this list until the 2026-07-14 queue drain promoted them (GPD-derived,
# citation-audited); their coverage moved to the promotion tests below.
# NOTE: the AxionEDM e*cm entries were removed by the Phase 2 (#625) queue drain
# — the mass-dependent d_n_ecm converter now handles them (see
# ECM_NOW_CONVERTIBLE below). The ScalarNucleon d_g entries stay flagged: d_g is
# a distinct coupling from canonical d_e with no universal conversion.
STILL_FLAGGED = [
    ("ScalarNucleon",
     "|d_mhat^(1) - d_g^(1)|, dimensionless (plotted as log10 on y-axis)"),  # 1807.04512
    # ScalarNucleon d_g stems deliberately not added (would suppress the
    # combined-coupling flag above) — these two stay flagged as designed.
    ("ScalarNucleon", "coupling d_g, dimensionless"),                     # 2301.10784
    ("ScalarNucleon", "d_g dimensionless, plotted as on y-axis of Fig 3a"),  # 2404.00616
]

# Promoted by the Phase 2 drain — the oscillating-EDM amplitude in e*cm is now
# the mass-dependent d_n_ecm converter (convertible, not review-flagged).
ECM_NOW_CONVERTIBLE = [
    ("AxionEDM", "d_n in e*cm"),                                          # 2101.01241
    ("AxionEDM",
     "d_n oscillation amplitude in e*cm (limit on dn-(mu_n/mu_Hg)dHg)"),  # 1708.06367
    ("AxionEDM",
     "d_n in e*cm (oscillating deuteron EDM amplitude d_AC)"),            # 2208.07293
]


@pytest.mark.parametrize("ct,decl", UNFLAGGED)
def test_family_notation_not_flagged(ct, decl):
    assert not convention_review_needed(ct, decl), (
        f"{ct}: canonical family notation must not flag: {decl!r}")


@pytest.mark.parametrize("ct,decl", UNFLAGGED)
def test_family_notation_not_unconvertible_on_eval_side(ct, decl):
    assert classify_reported_convention(ct, decl) != UNCONVERTIBLE, (
        f"{ct}: eval registry must not exclude canonical family notation: {decl!r}")


@pytest.mark.parametrize("ct,decl", STILL_FLAGGED)
def test_unknown_conventions_still_flagged(ct, decl):
    assert convention_review_needed(ct, decl), (
        f"{ct}: genuinely unknown convention must keep its review flag: {decl!r}")


@pytest.mark.parametrize("ct,decl", ECM_NOW_CONVERTIBLE)
def test_ecm_now_convertible_both_mirrors(ct, decl):
    # Phase 2: e*cm routes to the mass-dependent d_n_ecm converter on the eval
    # side and is no longer review-flagged at runtime.
    assert classify_reported_convention(ct, decl) == "d_n_ecm"
    assert not convention_review_needed(ct, decl)


# ---------------------------------------------------------------------------
# Equivalence-assertion exemption: scoped, never a blanket pass
# ---------------------------------------------------------------------------

def test_equivalence_to_foreign_symbol_exempts_nothing():
    # Asserting equivalence to a symbol OUTSIDE the family must not bypass
    # the screen.
    assert convention_review_needed(
        "AxionElectron", "V_dd potential (equivalent to c_gamma something)")


def test_equivalence_clause_requires_expected_stem():
    # Same declaration shape as 1903.03586 but asserting a non-family symbol.
    assert convention_review_needed(
        "AxionPhoton", "c_gamma/Lambda in GeV^-1 (equivalent to d_me)")


def test_bare_foreign_symbol_without_equivalence_still_flags():
    # A bare c_gamma/Lambda WITHOUT the equivalence assertion stays flagged:
    # the c/Lambda normalization is convention-dependent (can differ from
    # g_agamma by alpha/2pi-class factors), so only the model's explicit
    # canonical-equivalence assertion clears it.
    assert convention_review_needed("AxionPhoton", "c_gamma/Lambda in GeV^-1")


# ---------------------------------------------------------------------------
# Historical regressions must not resurface
# ---------------------------------------------------------------------------

def test_1508_02463_dipole_dipole_still_foreign():
    # The incident that motivated the foreign screen: a spin-spin potential
    # strength masquerading via "squared"+"dimensionless" tokens.
    decl = "(g_p^e)^2/(hbar c), dimensionless dipole-dipole potential strength V_dd"
    assert convention_review_needed("AxionElectron", decl)
    assert classify_reported_convention("AxionElectron", decl) == UNCONVERTIBLE


def test_negation_clause_still_dropped():
    # Negations describe what the values are NOT; still exempt.
    assert not convention_review_needed(
        "AxionElectron", "g_ae dimensionless, NOT the potential V_dd")


# ---------------------------------------------------------------------------
# Mirror synchrony: both implementations agree on every corpus declaration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ct,decl", UNFLAGGED + STILL_FLAGGED)
def test_mirrors_agree(ct, decl):
    assert tg_foreign(ct, decl.lower()) == ev_foreign(ct, decl.lower()), (
        f"transform_guard and evaluation.conventions disagree on {ct}: {decl!r}")


# ---------------------------------------------------------------------------
# Mirror-drift fix: runtime inv_fa tokens must match the eval registry
# ---------------------------------------------------------------------------

def test_cg_over_fa_not_flagged_at_runtime():
    # 2204.01454: eval converts this via inv_fa; the runtime mirror was missing
    # the spelled-out token and flagged a convertible declaration.
    decl = "C_G/f_a in GeV^-1 as plotted on y-axis (NOT d_n in e cm)"
    assert not convention_review_needed("AxionEDM", decl)
    assert classify_reported_convention("AxionEDM", decl) == "inv_fa"


def test_braced_inverse_gev_canonical_for_axionmass():
    # 2302.00685: LaTeX-braced spelling of the canonical inverse plane must not
    # flag at runtime (eval's inv_gev already accepts the braced variant).
    decl = "f_a^{-1} in GeV^{-1}, plotted as y-axis value directly"
    assert not convention_review_needed("AxionMass", decl)
    assert classify_reported_convention("AxionMass", decl) is None  # canonical


# ---------------------------------------------------------------------------
# Drained-token promotions (2026-07-14): clock combo + Higgs-portal sin theta
# ---------------------------------------------------------------------------

from evaluation.conventions import to_canonical  # noqa: E402

_CLOCK_DECL = "log10 of d_e + 0.043(d_m̂ - d_g); emitted as absolute coupling value (10^y)"
_SINTH_DECL = "sin theta (Higgs-portal scalar-Higgs mixing angle)"


def test_clock_combo_promoted_canonical():
    # Token 1604.08514 (note convention-1604.08514-clock-combo-d_e.md,
    # citation-audited PASS): d_e-leading sub-unity combination == d_e under
    # one-coupling dominance. No flag, no conversion.
    assert not convention_review_needed("ScalarPhoton", _CLOCK_DECL)
    assert classify_reported_convention("ScalarPhoton", _CLOCK_DECL) is None


def test_clock_combo_scoped_to_d_e_leading():
    # A combination WITHOUT the leading d_e term is a different plane: the
    # 1807.04512 declaration must keep failing closed (also in STILL_FLAGGED).
    assert convention_review_needed(
        "ScalarNucleon", "|d_mhat^(1) - d_g^(1)|, dimensionless")
    # A coefficient >= 1 is not the vetted sensitivity-combination shape.
    assert convention_review_needed(
        "ScalarPhoton", "d_e + 1.5(d_m̂ - d_g), dimensionless")
    # Scoped to ScalarPhoton — the same string on another type still flags.
    assert convention_review_needed("ScalarNucleon", _CLOCK_DECL)


def test_sintheta_promoted_convertible():
    # Token 2303.00778 (note convention-2303.00778-sintheta-d_me.md,
    # citation-audited PASS): recognized at runtime (no flag), converted by
    # the eval registry.
    assert not convention_review_needed("ScalarElectron", _SINTH_DECL)
    assert classify_reported_convention("ScalarElectron", _SINTH_DECL) == "sin_theta_higgs"


def test_sintheta_numeric_spot_check():
    # d_me = sqrt2 M_Pl/v * sin theta; the derivation's spot-check value:
    # 1.9e-10 -> 2.619e6 vs repo WhiteDwarfs.txt 2.5793e6 (+0.0067 dex).
    pts = [(1e-3, 1.9e-10), (1e3, 1.9e-10)]
    out, note = to_canonical("ScalarElectron", pts, "sin_theta_higgs")
    assert "sin theta -> d_me" in note
    import math
    assert abs(math.log10(out[0][1] / 2.5793e6)) < 0.05


def test_sintheta_magnitude_guard_refuses_non_angle():
    # d_me-scale values under a sin-theta declaration: mislabel, refuse.
    pts = [(1e-3, 2.6e6)]
    out, note = to_canonical("ScalarElectron", pts, "sin_theta_higgs")
    assert out == pts and "refused" in note.lower()


def test_sintheta_scoped_to_scalar_electron():
    # ScalarPhoton sin-theta would carry a different constant: keep flagging.
    assert convention_review_needed("ScalarPhoton", _SINTH_DECL)


# ---------------------------------------------------------------------------
# Fable-probe screen interactions (2026-07-17, #733): verbose definitional
# glosses must not trip the screens. Both fixes are model-agnostic.
# ---------------------------------------------------------------------------

_FABLE_GLOSS_UNFLAGGED = [
    # C_N in a g_aNN definitional gloss is family notation (precedent: c_g,
    # g_e); 2111.09892's 0.00-dex extraction was excluded by the old screen.
    ("AxionNeutron",
     "dimensionless g_ann (axion-neutron coupling g_aNN = C_N m_N / f_a)"),
    # clock-combo spelled without the underscore ("de + 0.043(...)").
    ("ScalarPhoton",
     "dimensionless d_e (paper's limit is on the combination de + 0.043(d_mhat - d_g))"),
]


@pytest.mark.parametrize("ct,decl", _FABLE_GLOSS_UNFLAGGED)
def test_fable_gloss_not_flagged_both_mirrors(ct, decl):
    assert not convention_review_needed(ct, decl)
    assert classify_reported_convention(ct, decl) != UNCONVERTIBLE


def test_no_de_leading_combo_still_flagged():
    # #683 scope guard: a combination WITHOUT the leading d_e term stays closed
    # even under the relaxed spelling.
    decl = "|d_mhat^(1) - d_g^(1)|, dimensionless (plotted as log10 on y-axis)"
    assert convention_review_needed("ScalarNucleon", decl)
