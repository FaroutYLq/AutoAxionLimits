"""Coupling convention/units inference for ground-truth entries.

A "coupling type" (e.g. ``AxionMass``, ``DarkPhoton``) does NOT pin down the
*variable* or *units* on the y-axis. The same physics is published in several
conventions, and the repo data files are not internally consistent. Comparing
two different conventions produces multi-dex residuals that are NOT extraction
error. This module derives, per data file, a canonical ``coupling_convention``
string and a human-readable ``coupling_units`` string so the evaluation can
refuse to compare across conventions.

Derivation rules:
  * The canonical convention/units come from ``pipeline.config.COUPLING_TYPES``
    ``axes["y"]`` for the data file's coupling type (keyed by its
    ``limit_data/<dir>/`` directory, the same dir→coupling logic used in
    ``evaluate.py``).
  * For couplings with a known multi-convention trap (``AxionMass``/``fa``
    plane, ``DarkPhoton`` epsilon vs epsilon^2, ``ScalarPhoton``/
    ``ScalarElectron`` d_e vs a large-valued variable), the actual convention
    is inferred from the data file's coupling value range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Reuse the same dir→coupling map the evaluator uses so a data file's
# directory is the authoritative physical coupling.
try:
    from pipeline.config import COUPLING_TYPES as _COUPLING_TYPES_REG
    _DIR_TO_COUPLING = {
        Path(meta["data_dir"]).name: key for key, meta in _COUPLING_TYPES_REG.items()
    }
except Exception:  # pragma: no cover - config import is best-effort
    _COUPLING_TYPES_REG = {}
    _DIR_TO_COUPLING = {}
_DIR_TO_COUPLING.setdefault("VectorB-L", "VectorBL")
_DIR_TO_COUPLING["fa"] = "AxionMass"  # m_a vs f_a plane is classified AxionMass


# Canonical (convention, units) per coupling type. The units string mirrors the
# config ``axes["y"]`` label; the convention is a short machine token used for
# equality checks. These are the defaults used when the value range is
# unambiguous.
_CANONICAL: dict[str, tuple[str, str]] = {
    "DarkPhoton":     ("epsilon",     "kinetic mixing chi (dimensionless)"),
    "AxionPhoton":    ("g_GeV^-1",    "g_agamma [GeV^-1]"),
    "AxionElectron":  ("g_ae",        "g_ae (dimensionless)"),
    "AxionNeutron":   ("g_an",        "g_an (dimensionless)"),
    "AxionProton":    ("g_ap",        "g_ap (dimensionless)"),
    "AxionEDM":       ("g_angamma",   "g_angamma [GeV^-2]"),
    "AxionCPV":       ("coupling",    "coupling (dimensionless)"),
    "AxionMass":      ("f_a_norm",    "normalized axion coupling (dimensionless)"),
    "MonopoleDipole": ("coupling",    "coupling (dimensionless)"),
    "ScalarPhoton":   ("d_e",         "d_e (dimensionless)"),
    "ScalarElectron": ("d_e",         "d_e (dimensionless)"),
    "ScalarBaryon":   ("coupling",    "coupling (dimensionless)"),
    "ScalarNucleon":  ("coupling",    "coupling (dimensionless)"),
    "VectorBL":       ("g_BL",        "g_BL (dimensionless)"),
}


def canonical_convention(coupling_type: str) -> tuple[Optional[str], Optional[str]]:
    """(convention, units) for the canonical form of a coupling type."""
    return _CANONICAL.get(coupling_type, (None, None))


# Polygon-closing `fill_between` vertices (`1e20`, `1e30`, `1e99`, and other
# huge "fill walls") are sentinels, NOT coupling values. They must be stripped
# BEFORE range-based convention discrimination, else a genuine-`d_e` file (e.g.
# ScalarElectron/HSi.txt interior 1e-5..1e-2 with a trailing `1e30` row) is
# misread as the large-valued non-`d_e` convention. The vetted rule
# (GPD/explanations/coupling-convention-conversions-EXPLAIN.md, "Sentinel rule"):
# discard rows with y >= 1e19 before classifying. No genuine coupling — including
# the largest decay scale f_a ~ 1e18 GeV — reaches 1e19.
_SENTINEL_FLOOR = 1e19


def _coupling_value_range(data_file: Path) -> Optional[tuple[float, float]]:
    """(min, max) of strictly-positive, non-sentinel y-values in a 2-column file.

    Tolerant of comment lines and comma-separated rows. Strips `fill_between`
    sentinel rows (y >= ``_SENTINEL_FLOOR``) so the range reflects real interior
    couplings. Returns None if the file is missing, single-column, or has no
    positive non-sentinel y-values.
    """
    if not data_file.exists():
        return None
    arr = None
    for delim in (None, ","):
        try:
            arr = np.loadtxt(str(data_file), ndmin=2, delimiter=delim)
            break
        except Exception:
            arr = None
    if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
        return None
    y = arr[:, 1]
    y = y[np.isfinite(y) & (y > 0) & (y < _SENTINEL_FLOOR)]
    if y.size == 0:
        return None
    return float(np.min(y)), float(np.max(y))


def infer_convention(
    coupling_type: str,
    data_file: Optional[Path] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Infer (coupling_convention, coupling_units) for a GT data file.

    Starts from the canonical convention for the coupling type, then, for
    couplings with a known multi-convention trap, overrides it by inspecting
    the data file's value range.
    """
    conv, units = canonical_convention(coupling_type)

    if data_file is None:
        return conv, units
    rng = _coupling_value_range(data_file)
    if rng is None:
        return conv, units
    ymin, ymax = rng

    if coupling_type == "AxionMass":
        # f_a [GeV] (~1e9-1e18) vs a small normalized coupling (~1e-24-1e-3).
        # The decay constant f_a is a macroscopic energy scale; anything above
        # ~1e6 cannot be the normalized (sub-unity) coupling.
        if ymax > 1e6:
            return "f_a_GeV", "f_a [GeV]"
        return "f_a_norm", "normalized axion coupling (dimensionless)"

    if coupling_type in ("ScalarPhoton", "ScalarElectron"):
        # d_e is a small dimensionless coupling (<= O(1)); the alternative
        # convention reaches very large values (up to ~1e30+).
        if ymax > 1e3:
            return "d_e_large", "large-valued scalar coupling variable (non-d_e convention)"
        return "d_e", "d_e (dimensionless)"

    if coupling_type in ("ScalarNucleon", "ScalarBaryon"):
        # Same trap as ScalarPhoton/Electron: the canonical dimensionless
        # coupling is <= O(1), but several repo files store a large-valued
        # variable (e.g. ScalarNucleon/IUPUI.txt ~7.5e3..2.2e17, the Yukawa
        # |alpha|-relative-to-gravity or 1/Lambda convention — #536 / #587 P-B,
        # 1410.7267). Mark those so the comparator refuses the cross-convention
        # residual instead of scoring a ~9-16 dex units gap as extraction error.
        if ymax > 1e3:
            return "coupling_large", "large-valued scalar coupling variable (non-dimensionless convention)"
        return "coupling", "coupling (dimensionless)"

    if coupling_type == "DarkPhoton":
        # epsilon (kinetic mixing) vs epsilon^2. Both span wide ranges and the
        # repo canonical form is epsilon, so we keep the canonical default
        # unless a clearer signal exists. Values cannot exceed O(1) for either
        # convention in this pool, so range alone is not separable here; default
        # to the canonical epsilon.
        return "epsilon", "kinetic mixing chi (dimensionless)"

    return conv, units


def infer_convention_for_repo_file(
    reference_repo_file: Optional[str],
    coupling_type_fallback: str,
    data_file: Optional[Path] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve coupling type from a ``limit_data/<dir>/`` path (the same logic
    as ``evaluate._authoritative_coupling``) and infer its convention/units.
    """
    coupling = coupling_type_fallback
    if reference_repo_file:
        parts = Path(reference_repo_file).parts
        if len(parts) >= 2 and parts[0] == "limit_data":
            coupling = _DIR_TO_COUPLING.get(parts[1], coupling_type_fallback)
    return infer_convention(coupling, data_file)


# ---------------------------------------------------------------------------
# Per-convention CANONICALIZATION (#536 / #587) — vetted closed-form conversions
# ---------------------------------------------------------------------------
# Source: GPD/explanations/coupling-convention-conversions-EXPLAIN.md, every
# factor below verified directly against PlotFuncs.py / PlotFuncs_ScalarVector.py
# / the notebooks (not guessed). The canonical variable per coupling type is the
# DIMENSIONLESS quantity the repo PLOTS. `to_canonical` maps a curve in a named
# source convention TO canonical so the comparator can score like-for-like.
#
# NOTE (wiring): this is the deterministic core. Converting only ONE side breaks
# pairs that already agree in a shared non-canonical convention, so the caller
# MUST canonicalize BOTH the GT curve (its file convention, resolvable here) AND
# the extraction (its reported convention) before computing residuals. Applying
# the same factor to both sides preserves an existing match and fixes a mismatch.

import math as _math

# Nucleon masses [GeV] — exactly the values PlotFuncs.py uses in-code.
_M_NUCLEON_GEV = {"AxionNeutron": 0.93957, "AxionProton": 0.93828}

# Reduced Planck mass [GeV] — the value the Scalars notebook uses for the
# d_i <-> g_{phi} [GeV^-1] AlternativeCouplingAxis (`M_pl = 2.4e18`). Use the
# notebook value (not 2.418e18) so a converted extraction lands on the SAME scale
# as the d-axis GT the notebook produced.
_M_PL_GEV = 2.4e18

# Per-FILE source-convention overrides (verified from PlotFuncs.py per-file
# multipliers). Default for AxionNeutron/Proton files is g_aNN = C_N/(2 f_a)
# [GeV^-1] (-> x2 m_N); SNO stores g_aN/m_N [GeV^-1] (-> x m_N only).
#
# ROUND-2 BUG FIX (coupling-convention-conversions-round2-EXPLAIN.md, "registry
# bug"): the family default is WRONG for the spin-force/astrophysics subset —
# these files already store the DIMENSIONLESS g_aN (headers `g_an`/`g_ap`,
# plotted RAW in PlotFuncs.py with no multiplier), so the blanket x2 m_N
# default inflated their GT side by +0.27 dex. `None` = already canonical.
_FILE_CONVENTION: dict[str, Optional[str]] = {
    "limit_data/AxionNeutron/SNO.txt": "g_aN_over_mN_inv_gev",
    "limit_data/AxionProton/SNO.txt":  "g_aN_over_mN_inv_gev",
    # Already-dimensionless files (round-2 audit, verified vs PlotFuncs.py):
    "limit_data/AxionNeutron/K-3He_Comagnetometer.txt": None,
    "limit_data/AxionNeutron/TorsionBalance.txt": None,
    "limit_data/AxionNeutron/129Xe.txt": None,
    "limit_data/AxionNeutron/Casimir.txt": None,
    "limit_data/AxionNeutron/SN1987A.txt": None,
    "limit_data/AxionNeutron/NeutronStars.txt": None,
    "limit_data/AxionProton/TorsionBalance.txt": None,
    "limit_data/AxionProton/Casimir.txt": None,
    "limit_data/AxionProton/SN1987A.txt": None,
    "limit_data/AxionProton/NeutronStars.txt": None,
    "limit_data/AxionProton/Projections/MnCO3.txt": None,
}

# Sentinel returned by `classify_reported_convention` for an extraction whose
# DECLARED convention is recognized but has NO vetted closed-form conversion to
# canonical (e.g. an AxionEDM oscillating-EDM amplitude in e*cm, which maps to
# g_angamma only through the per-point field amplitude a_0 = sqrt(2 rho)/m_a).
# The comparator treats this as a convention gap (exclude), NOT extraction error.
UNCONVERTIBLE = "__unconvertible__"

# Note prefix returned by `to_canonical` when a recognized token REFUSES to
# convert because the input values violate the token's magnitude guard (round-2
# rule: every token carries a plausible-range guard; converting values that
# cannot be the declared quantity — e.g. anchor-snap-corrupted f_a values, or a
# Lambda-in-GeV curve fed to the multiply branch — is worse than excluding).
# Callers treat a note starting with this prefix like UNCONVERTIBLE.
GUARD_REFUSED = "__convention_guard_refused__"

# --- Round-2 constants (coupling-convention-conversions-round2-EXPLAIN.md,
# citation-audited; every factor numerically spot-checked against its paper/GT
# pair) --------------------------------------------------------------------
_HBAR_GEV_S = 6.582119569e-25   # hbar [GeV s]
_64PI = 64.0 * _math.pi         # two-photon decay prefactor (Gamma = g^2 m^3/64pi)
_SQRT_4PI = _math.sqrt(4.0 * _math.pi)
_K_XI = 1.395e-10               # g[GeV^-1] = K_XI * xi * m[eV] (thermal QCD axion)


def _median_positive(vals) -> Optional[float]:
    v = sorted(x for x in vals if x and x > 0)
    return v[len(v) // 2] if v else None

# Recognized source-convention tokens per coupling family (canonical first).
_CANONICAL_TOKEN = {
    "AxionNeutron": "g_aN", "AxionProton": "g_aN",
    "ScalarPhoton": "d_e",  "ScalarElectron": "d_me",
    "ScalarNucleon": "d_e", "ScalarBaryon": "d_e",
    "DarkPhoton": "chi", "AxionMass": "inv_fa", "AxionEDM": "g_angamma",
}


def file_source_convention(reference_repo_file: Optional[str],
                           coupling_type: Optional[str]) -> Optional[str]:
    """Source convention token for a GT data file (per-file override, else the
    family default for files known to store a non-canonical variable). Returns
    None when the file is taken to already be canonical / is unknown."""
    if reference_repo_file and reference_repo_file in _FILE_CONVENTION:
        return _FILE_CONVENTION[reference_repo_file]
    # Family defaults for the stored variable (verified): nucleon files store the
    # GeV^-1 derivative coupling; the repo multiplies by 2 m_N to plot.
    if coupling_type in ("AxionNeutron", "AxionProton"):
        return "g_aNN_inv_gev"
    return None


def to_canonical(coupling_type: Optional[str], data_points, convention: Optional[str]):
    """Convert ``data_points`` from ``convention`` to the canonical variable for
    ``coupling_type``. Pure, closed-form. Returns ``(points, note)``; points are
    returned UNCHANGED with note="" when the convention is already canonical,
    empty, or not a recognized/convertible alternate (the caller then decides to
    compare, flag, or exclude). Never raises.
    """
    if not data_points or not coupling_type or not convention:
        return data_points, ""
    conv = convention.strip().lower()
    canon = _CANONICAL_TOKEN.get(coupling_type)
    if canon and conv == canon.lower():
        return data_points, ""
    med_y = _median_positive(g for _m, g in data_points)
    try:
        # ------------------- Round-2 vetted families -------------------
        # (coupling-convention-conversions-round2-EXPLAIN.md; each branch
        # carries the doc's magnitude guard and refuses out-of-range input.)

        # Family 1: f_a [GeV] -> 1/f_a [GeV^-1] (AxionMass, pure reciprocal).
        # Three magnitude regimes (they never overlap physically): a genuine
        # decay constant is >~1e5 GeV -> invert; canonical 1/f_a-normalized
        # values are <~4e-4 -> the declaration is a provable mislabel and the
        # emitted values are ALREADY canonical (observed repeatedly in the
        # full346 snapshots: vision traces the 1/f_a axis but declares the
        # paper's f_a convention) -> compare raw; anything between (e.g.
        # anchor-snap-corrupted f_a, 2105.13963's x1e-20 residue at ~1e-2) is
        # neither -> refuse.
        if coupling_type == "AxionMass" and conv == "f_a_gev":
            if med_y is None:
                return data_points, ""
            if med_y > 1e3:
                return [(m, 1.0 / g) for m, g in data_points if g > 0], (
                    "convention: f_a [GeV] -> 1/f_a [GeV^-1] (reciprocal)")
            if med_y < 1e-3:
                return data_points, (
                    "convention: declared f_a [GeV] but values are already "
                    f"canonical-1/f_a scale (median {med_y:g}) — mislabeled "
                    "declaration, compared raw (#594)")
            return data_points, (
                f"{GUARD_REFUSED}: f_a_gev values neither decay-constant "
                f"(>1e3 GeV) nor canonical 1/f_a (<1e-3) scale (median "
                f"{med_y:g}) — likely snapped/corrupted")

        # Family 2: decay-rate / lifetime plane -> g_agamma [GeV^-1].
        # Gamma = g^2 m^3 / 64pi  =>  g = sqrt(64pi * hbar * Gamma / m^3),
        # m in GeV (= 1e-9 * mass[eV]); lifetime: Gamma = 1/tau.
        if coupling_type == "AxionPhoton" and conv == "decay_rate_s_inv":
            if med_y is None or med_y > 1e-10:
                return data_points, (
                    f"{GUARD_REFUSED}: decay_rate_s_inv expects Gamma << "
                    f"1e-10 s^-1 (median {med_y!r})")
            out = [(m, _math.sqrt(_64PI * _HBAR_GEV_S * g / (1e-9 * m) ** 3))
                   for m, g in data_points if g > 0 and m > 0]
            return out, ("convention: Gamma [s^-1] -> g_agamma [GeV^-1] "
                         "(sqrt(64pi*hbar*Gamma/m^3), per point)")
        if coupling_type == "AxionPhoton" and conv == "lifetime_s":
            if med_y is None or med_y < 1e10:
                return data_points, (
                    f"{GUARD_REFUSED}: lifetime_s expects tau >> 1e10 s "
                    f"(median {med_y!r})")
            out = [(m, _math.sqrt(_64PI * _HBAR_GEV_S / (g * (1e-9 * m) ** 3)))
                   for m, g in data_points if g > 0 and m > 0]
            return out, ("convention: tau [s] -> g_agamma [GeV^-1] "
                         "(sqrt(64pi*hbar/(tau*m^3)), per point)")

        # Family 3: squared-coupling axes. TWO distinct tokens 0.55 dex apart:
        # g^2/(4pi) (alpha-like, 0809.4700) vs plain g^2 (/hbar c, 1508.02463).
        if coupling_type in ("AxionNeutron", "AxionProton", "AxionElectron"):
            if conv == "g_squared_over_4pi":
                return [(m, _math.sqrt(4.0 * _math.pi * g))
                        for m, g in data_points if g > 0], (
                    "convention: g^2/(4pi) -> g (sqrt(4pi*y))")
            if conv == "g_squared":
                return [(m, _math.sqrt(g)) for m, g in data_points if g > 0], (
                    "convention: g^2 -> g (sqrt)")

        # Family 4: thermal-axion xi -> g_agamma (model-locked: thermal QCD
        # axion, Grin et al. tau = 6.8e24 xi^-2 m^-5 s + Gamma = g^2 m^3/64pi).
        # Medium confidence: documented +0.12-0.25 dex conversion floor.
        if coupling_type == "AxionPhoton" and conv == "xi_thermal":
            masses = [m for m, _g in data_points if m and m > 0]
            m_lo, m_hi = (min(masses), max(masses)) if masses else (0, 0)
            if (med_y is None or not (1e-4 <= med_y <= 1.0)
                    or not (0.5 <= m_lo and m_hi <= 30.0)):
                return data_points, (
                    f"{GUARD_REFUSED}: xi_thermal expects xi in [1e-4,1] and "
                    f"optical-window masses ~1-30 eV (median {med_y!r}, "
                    f"masses [{m_lo:g},{m_hi:g}])")
            return [(m, _K_XI * g * m) for m, g in data_points if g > 0], (
                "convention: thermal-axion xi -> g_agamma "
                f"({_K_XI:g}*xi*m_eV; model-locked to thermal QCD axion)")
        # --- Axion-nucleon: GeV^-1 derivative coupling -> dimensionless g_aN ---
        if coupling_type in ("AxionNeutron", "AxionProton"):
            m_n = _M_NUCLEON_GEV[coupling_type]
            if conv in ("g_ann_inv_gev", "g_ann", "gev^-1", "inv_gev"):
                f = 2.0 * m_n           # g_aN = 2 m_N * (C_N/2 f_a)
                return [(m, g * f) for m, g in data_points], (
                    f"convention: {coupling_type} g_aNN [GeV^-1] -> dimensionless "
                    f"(x2 m_N = {f:.4g})")
            if conv in ("g_an_over_mn_inv_gev", "g_an/m_n"):
                f = m_n                 # SNO file: g_aN = m_N * (g_aN/m_N)
                return [(m, g * f) for m, g in data_points], (
                    f"convention: {coupling_type} g_aN/m_N [GeV^-1] -> dimensionless "
                    f"(x m_N = {f:.4g})")

        # --- Scalar / dilaton: fifth-force alpha -> dimensionless d_e / d_me ---
        if coupling_type in ("ScalarPhoton", "ScalarElectron", "ScalarNucleon", "ScalarBaryon"):
            if conv in ("alpha_fifthforce", "alpha", "yukawa_alpha"):
                pref = 4000.0 if coupling_type == "ScalarElectron" else 500.0
                out = [(m, pref * _math.sqrt(g)) for m, g in data_points if g > 0]
                return out, f"convention: Scalar Yukawa alpha -> d ({pref:g}*sqrt(alpha))"
            # GeV^-1 Compton-like coupling g_{phi i} -> dimensionless d_i. The
            # notebook's AlternativeCouplingAxis is g_{phi gamma} = d_e/(sqrt2 M_Pl)
            # (and identically d_{m_e} = g_{phi e} sqrt2 M_Pl by the universal
            # phi_hat = phi/M_Pl normalization). Inverse map: d = g[GeV^-1]*sqrt2*M_Pl.
            # M_Pl matches the repo notebook value (2.4e18) so converted extractions
            # land on the same scale as the d-axis GT. Vetted: EXPLAIN doc Sec.1.
            #
            # Round-2 (Family 6): the SAME sqrt2*M_Pl factor also covers a
            # published new-physics SCALE Lambda [GeV] (QSNET Eq. 15,
            # 1/Lambda = kappa*d_e with kappa = sqrt(4 pi G_N)), but in the
            # RECIPROCAL direction: d = sqrt2*M_Pl / Lambda. Direction is
            # decided by magnitude — the two regimes are cleanly separated
            # (1/Lambda values << 1; a valid clock-network scale Lambda >> 1e3
            # GeV) — and in-between values are refused rather than guessed.
            # This guard also stops the `lambda_gamma` substring from
            # multiplying a Lambda-in-GeV curve into d ~ 1e48.
            if conv in ("gev_inv_scalar", "g_phi_gev_inv", "lambda_inv_gev",
                        "lambda_scale_gev"):
                f = _math.sqrt(2.0) * _M_PL_GEV
                med = _median_positive(g for _m, g in data_points)
                if med is not None and med < 1.0:
                    return [(m, g * f) for m, g in data_points], (
                        f"convention: Scalar g_phi [GeV^-1] -> d (x sqrt2 M_Pl = {f:.4g})")
                if med is not None and med > 1e3:
                    return [(m, f / g) for m, g in data_points if g > 0], (
                        f"convention: Scalar Lambda [GeV] -> d (sqrt2 M_Pl / y, "
                        f"sqrt2 M_Pl = {f:.4g})")
                return data_points, (
                    f"{GUARD_REFUSED}: scalar GeV^-1/Lambda values must be "
                    f"<1 (1/Lambda) or >1e3 (Lambda in GeV); median {med!r}")

        # --- DarkPhoton: epsilon^2 -> kinetic mixing chi ---
        if coupling_type == "DarkPhoton" and conv in ("epsilon_squared", "eps^2", "chi^2"):
            return [(m, _math.sqrt(g)) for m, g in data_points if g > 0], (
                "convention: DarkPhoton eps^2 -> chi (sqrt)")

        # --- AxionMass / AxionEDM linear links ---
        if coupling_type == "AxionEDM" and conv in ("inv_fa", "1/f_a"):
            f = 3.7e-3              # g_angamma [GeV^-2] = 3.7e-3 * (1/f_a) [GeV^-1]
            return [(m, g * f) for m, g in data_points], (
                "convention: 1/f_a [GeV^-1] -> g_angamma [GeV^-2] (x3.7e-3)")
    except Exception:
        return data_points, ""   # never break the comparator on a bad point
    return data_points, ""


# Map a model-DECLARED output-convention/units string (the convention the
# extractor says its emitted data_points are in) to a `to_canonical` token.
# Conservative: returns a non-canonical alternate ONLY on a clear unit signal,
# else None (treated as canonical / no conversion). Used to canonicalize the
# EXTRACTION side (the GT side uses `file_source_convention`).
def classify_reported_convention(coupling_type: Optional[str],
                                 units_label: Optional[str]) -> Optional[str]:
    if not coupling_type or not units_label:
        return None
    raw = units_label.lower()
    u = raw.replace(" ", "")
    # Declaration contract (round-2, #594): a string claiming the values were
    # ALREADY converted ("converted from ...") must be treated as canonical-
    # claimed — the token describes the EMITTED values, and re-converting a
    # genuinely converted output would corrupt it (0809.4700 declared
    # converted-but-emitted-raw; that mislabel is the extractor contract's
    # problem, not a registry guess).
    already_converted = "converted" in u
    # Inverse-GeV (prefix-aware: must be 'gev', not bare 'ev' which is eV^-1).
    inv_gev = any(t in u for t in ("gev^-1", "gev-1", "gev^{-1}", "gev$^{-1}$", "1/gev"))
    # Squared-coupling axes (round-2 Family 3): two DISTINCT tokens 0.55 dex
    # apart — g^2/(4pi) vs plain g^2 (e.g. "/hbar c"). Never fired on
    # "converted from" declarations.
    squared = ("^2" in u or "squared" in u) and not already_converted
    if coupling_type in ("AxionNeutron", "AxionProton", "AxionElectron"):
        if squared:
            return ("g_squared_over_4pi"
                    if ("4pi" in u or "4π" in u) else "g_squared")
    if coupling_type in ("AxionNeutron", "AxionProton"):
        return "g_aNN_inv_gev" if inv_gev else None
    if coupling_type == "AxionMass":
        # Canonical is 1/f_a [GeV^-1]; a declared plain f_a [GeV] needs the
        # per-point reciprocal (round-2 Family 1). Inverse markers win first.
        if inv_gev or "1/f" in u:
            return None  # already the inverse-scale convention
        if any(t in u for t in ("f_aingev", "faingev", "f_a[gev]", "fa[gev]",
                                "f[gev]", "fingev")):
            return "f_a_gev"
        return None
    if coupling_type == "AxionPhoton":
        # Round-2 Families 2 & 4. Word-boundary xi check runs on the RAW
        # label ("axion" contains "xi" after space-stripping); it precedes the
        # canonical check because the observed declaration is
        # 'dimensionless xi (...), NOT g_agamma in GeV^-1'.
        import re as _re
        if _re.search(r"\bxi\b", raw):
            return "xi_thermal"
        if "g_agamma" in u or inv_gev:
            return None  # already canonical g_agamma [GeV^-1]
        if "s^-1" in u or "s-1" in u or "decayrate" in u or "1/s" in u:
            return "decay_rate_s_inv"
        if u in ("s", "sec", "seconds") or "lifetime" in u or "tau" in u:
            return "lifetime_s"
        return None
    if coupling_type == "DarkPhoton":
        if "eps^2" in u or "epsilon^2" in u or "chi^2" in u or "squared" in u:
            return "epsilon_squared"
        return None
    if coupling_type == "AxionEDM":
        # Canonical AxionEDM = g_angamma [GeV^-2] (#604: every repo AxionEDM file
        # is g_{a gamma n}/g_d/g_EDM [GeV^-2], NOT d_n [e cm]). Two cases:
        #  (a) the gluon coupling C_G/f_a (== 1/f_a with C_G~1) [GeV^-1] -> convert
        #      x3.7e-3 to canonical (1708.06367 declares 'CG/fa in GeV^-1').
        #  (b) the oscillating EDM AMPLITUDE d_n/d_d [e cm], or a bare 'GeV^-1'
        #      (the magnitude is the e*cm amplitude, e.g. 2204.01454/2101.01241/
        #      2208.07293): NOT convertible to g_angamma without the mass-dependent
        #      field amplitude a_0 = sqrt(2 rho)/m_a (a per-point response factor,
        #      not a constant — see EXPLAIN Sec.3b). Flag UNCONVERTIBLE so the
        #      comparator EXCLUDES it (convention gap) instead of scoring a ~14-dex
        #      raw units mismatch as extraction error.
        if "gev^-2" in u or "gev-2" in u or "g_angamma" in u or "g_{a" in u:
            return None  # already canonical g_angamma [GeV^-2]
        if any(t in u for t in ("1/f_a", "invfa", "1/fa", "cg/fa", "c_g/f_a",
                                "c_g/(f_a", "gluon")):
            return "inv_fa"
        if any(t in u for t in ("e*cm", "ecm", "e cm", "e·cm",
                                "d_n", "d_d", "edm")) or inv_gev:
            return UNCONVERTIBLE
        return None
    if coupling_type in ("ScalarPhoton", "ScalarElectron"):
        # The ONLY scalar conversion enabled (#600, vetted EXPLAIN Sec.1): the
        # GeV^-1 Compton-like coupling g_{phi} -> dimensionless d (x sqrt2 M_Pl).
        # Fires only when the extractor DECLARES a GeV^-1 / inverse-Lambda form
        # (the model-declared-convention contract, #594) — a model that declares
        # plain `d_e`/`d_me` is left canonical. The native large-valued scalar
        # files stay governed by the #591 d_e_large exclusion guard (their
        # per-file storage is unverified — do NOT auto-convert those here).
        lam_inv = ("lambda^-1" in u or "lambda_gamma^-1" in u
                   or "lambda_gamma" in u or "1/lambda" in u)
        if inv_gev or lam_inv:
            return "gev_inv_scalar"
        return None
    # Other scalars (Nucleon/Baryon) NOT auto-converted (native-file mapping
    # unverified — #536); they keep the #591 exclusion guard.
    return None
