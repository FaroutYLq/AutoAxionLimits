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
