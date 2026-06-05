"""Tests for the GT-convention guard in the subset comparator (issue #536).

No API/network. Two surfaces:
  * `conventions.infer_convention` correctly tags large-valued scalar files
    (the d_e_large / coupling_large trap) so the comparator can refuse them;
  * `subset_compare._paper_record` excludes a GT curve whose coupling_convention
    differs from the extraction's expected (canonical) convention, returning
    `convention_mismatch` instead of scoring a multi-dex units gap as error
    (2401.18076 @ 18.9 dex, 2006.07055 @ 18.6 dex).

Run:
    pytest evaluation/tests/test_convention_guard.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import subset_compare as sc
from evaluation.conventions import infer_convention


# ---------------------------------------------------------------------------
# conventions.infer_convention — large-valued scalar trap
# ---------------------------------------------------------------------------

def _write(tmp_path, name, ymax):
    f = tmp_path / name
    f.write_text(f"1e-5 1e-10\n1e-4 {ymax/10:.3e}\n1e-3 {ymax:.3e}\n")
    return f


def test_scalar_photon_large_valued_tagged(tmp_path):
    f = _write(tmp_path, "Holometer.txt", 1e13)
    conv, _ = infer_convention("ScalarPhoton", f)
    assert conv == "d_e_large"

def test_scalar_photon_small_is_canonical_de(tmp_path):
    f = _write(tmp_path, "SrSi.txt", 4.6e-3)
    conv, _ = infer_convention("ScalarPhoton", f)
    assert conv == "d_e"

def test_scalar_nucleon_large_valued_tagged(tmp_path):
    # ScalarNucleon/IUPUI.txt ~7.5e3..2.2e17 — previously mislabeled "coupling"
    # (no large-valued override), so the guard never fired (1410.7267, #536).
    f = _write(tmp_path, "IUPUI.txt", 2.2e17)
    conv, _ = infer_convention("ScalarNucleon", f)
    assert conv == "coupling_large"

def test_scalar_baryon_small_is_canonical(tmp_path):
    f = _write(tmp_path, "small.txt", 1e-3)
    conv, _ = infer_convention("ScalarBaryon", f)
    assert conv == "coupling"


# ---------------------------------------------------------------------------
# subset_compare convention guard
# ---------------------------------------------------------------------------

class _StubGT:
    """Minimal GroundTruthEntry stand-in for _paper_record."""
    def __init__(self, coupling_type, convention, repo_file, data):
        self.coupling_type = coupling_type
        self.coupling_convention = convention
        self.reference_repo_file = repo_file
        self._data = np.array(data, dtype=float)

    def load_data(self):
        return self._data

    def load_reference_data(self, root):
        return self._data


_GT_CURVE = [[1e-5, 1e-10], [1e-4, 1e-10], [1e-3, 1e-10]]
_RESULT = {"coupling_type": "ScalarElectron",
           "data_points": [[1e-5, 1.2e-10], [1e-3, 1.1e-10]],
           "data_source": "figure_vision"}


def test_mismatched_convention_excluded():
    # GT is the large-valued convention; extraction's expected canonical is d_e.
    gt = _StubGT("ScalarElectron", "d_e_large",
                 "limit_data/ScalarElectron/DAMNED.txt", _GT_CURVE)
    rec = sc._paper_record("2006.07055", _RESULT, [gt])
    assert rec["status"] == "convention_mismatch"
    assert rec["median_resid"] is None  # excluded from residuals


def test_matching_convention_still_compared():
    # GT in the canonical d_e convention is comparable -> scored normally.
    gt = _StubGT("ScalarElectron", "d_e",
                 "limit_data/ScalarElectron/SrSi.txt", _GT_CURVE)
    rec = sc._paper_record("good", _RESULT, [gt])
    assert rec["status"] == "compared"


def test_unknown_convention_not_treated_as_mismatch():
    # None convention on the GT side = unknown, so we still compare (no false
    # exclusion) — matches evaluate.py's guard semantics.
    gt = _StubGT("ScalarElectron", None,
                 "limit_data/ScalarElectron/X.txt", _GT_CURVE)
    rec = sc._paper_record("unknown", _RESULT, [gt])
    assert rec["status"] == "compared"
