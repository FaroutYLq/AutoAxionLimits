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
    "AxionEDM":       ("d_n",         "d_n [e cm]"),
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


def _coupling_value_range(data_file: Path) -> Optional[tuple[float, float]]:
    """(min, max) of strictly-positive y-values in a 2-column data file.

    Tolerant of comment lines and comma-separated rows. Returns None if the
    file is missing, single-column, or has no positive y-values.
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
    y = y[np.isfinite(y) & (y > 0)]
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

# Per-FILE source-convention overrides (verified from PlotFuncs.py per-file
# multipliers). Default for AxionNeutron/Proton files is g_aNN = C_N/(2 f_a)
# [GeV^-1] (-> x2 m_N); SNO stores g_aN/m_N [GeV^-1] (-> x m_N only).
_FILE_CONVENTION: dict[str, str] = {
    "limit_data/AxionNeutron/SNO.txt": "g_aN_over_mN_inv_gev",
    "limit_data/AxionProton/SNO.txt":  "g_aN_over_mN_inv_gev",
}

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
    try:
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
    u = units_label.lower().replace(" ", "")
    # Inverse-GeV (prefix-aware: must be 'gev', not bare 'ev' which is eV^-1).
    inv_gev = any(t in u for t in ("gev^-1", "gev-1", "gev^{-1}", "gev$^{-1}$", "1/gev"))
    if coupling_type in ("AxionNeutron", "AxionProton"):
        return "g_aNN_inv_gev" if inv_gev else None
    if coupling_type == "DarkPhoton":
        if "eps^2" in u or "epsilon^2" in u or "chi^2" in u or "squared" in u:
            return "epsilon_squared"
        return None
    if coupling_type == "AxionEDM":
        if "1/f_a" in u or "invfa" in u or "1/fa" in u:
            return "inv_fa"
        return None
    # Scalars deliberately NOT auto-converted here (native-file mapping partly
    # unverified — #536); they keep the #591 exclusion guard.
    return None
