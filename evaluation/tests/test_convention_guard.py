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
import pytest

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


def test_sentinel_row_does_not_inflate_genuine_de(tmp_path):
    # ScalarElectron/HSi.txt shape: genuine d_e interior (1e-5..1e-2) with a
    # trailing 1e30 fill-wall sentinel. Sentinel-aware range discrimination
    # (#596) must classify it as d_e, NOT d_e_large (the sentinel-unaware
    # classifier mislabeled it, which would wrongly exclude the paper).
    f = tmp_path / "HSi.txt"
    f.write_text("1e-5 1.2e-5\n1e-4 1e-3\n1e-3 1e-2\n1e-2 1e30\n")
    conv, _ = infer_convention("ScalarElectron", f)
    assert conv == "d_e"


def test_sentinel_does_not_hide_large_interior(tmp_path):
    # A genuinely large-valued (fifth-force) file with a sentinel row must still
    # be tagged large — stripping the sentinel leaves the >1e3 interior.
    f = tmp_path / "IUPUI.txt"
    f.write_text("1e-5 7.5e3\n1e-4 1e10\n1e-3 2.2e17\n1e-2 1e30\n")
    conv, _ = infer_convention("ScalarNucleon", f)
    assert conv == "coupling_large"


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


# ---------------------------------------------------------------------------
# Per-file canonicalization registry (#536/#587) — vetted conversions
# ---------------------------------------------------------------------------

from evaluation.conventions import to_canonical, file_source_convention


def test_axion_nucleon_gev_inv_to_dimensionless():
    # g_aNN = C_N/(2 f_a) [GeV^-1] -> dimensionless g_aN = 2 m_N * value
    out, note = to_canonical("AxionProton", [(1e-15, 1e-9)], "g_aNN_inv_gev")
    assert out[0][1] == pytest.approx(1e-9 * 2 * 0.93828)
    assert "x2 m_N" in note
    out_n, _ = to_canonical("AxionNeutron", [(1e-15, 1e-9)], "g_aNN_inv_gev")
    assert out_n[0][1] == pytest.approx(1e-9 * 2 * 0.93957)


def test_sno_is_per_file_x_mN_exception():
    assert file_source_convention("limit_data/AxionNeutron/SNO.txt", "AxionNeutron") == "g_aN_over_mN_inv_gev"
    out, note = to_canonical("AxionNeutron", [(1e-15, 1e-9)], "g_aN_over_mN_inv_gev")
    assert out[0][1] == pytest.approx(1e-9 * 0.93957)  # x m_N only, NOT 2 m_N
    assert "x m_N" in note


def test_default_nucleon_file_is_2mN():
    # a non-SNO nucleon file defaults to the GeV^-1 derivative coupling
    assert file_source_convention("limit_data/AxionNeutron/NASDUCK-SERF.txt", "AxionNeutron") == "g_aNN_inv_gev"


def test_scalar_alpha_to_d_e_and_d_me():
    # d_e = 500*sqrt(alpha); d_me = 4000*sqrt(alpha)
    assert to_canonical("ScalarPhoton", [(1e-3, 1e-4)], "alpha")[0][0][1] == pytest.approx(5.0)
    assert to_canonical("ScalarElectron", [(1e-3, 1e-4)], "alpha")[0][0][1] == pytest.approx(40.0)


def test_dark_photon_eps_squared_to_chi():
    out, _ = to_canonical("DarkPhoton", [(1e-3, 1e-14)], "eps^2")
    assert out[0][1] == pytest.approx(1e-7)


def test_axion_edm_invfa_to_g_angamma():
    out, _ = to_canonical("AxionEDM", [(1e-15, 1.0)], "1/f_a")
    assert out[0][1] == pytest.approx(3.7e-3)


def test_canonical_or_unknown_is_unchanged():
    pts = [(1e-3, 1e-12)]
    assert to_canonical("AxionPhoton", pts, "g_GeV^-1") == (pts, "")
    assert to_canonical("AxionProton", pts, "g_aN") == (pts, "")  # already canonical
    assert to_canonical("AxionNeutron", [], "g_aNN_inv_gev") == ([], "")
    assert to_canonical("AxionNeutron", pts, None) == (pts, "")


# ---------------------------------------------------------------------------
# Comparator both-sides canonicalization (#536/#587 wiring)
# ---------------------------------------------------------------------------

def _gt_entry(coupling_type, convention, repo_file, data):
    return _StubGT(coupling_type, convention, repo_file, data)


def test_comparator_canonicalizes_both_axion_nucleon():
    # GT file is raw GeV^-1 g_aNN; extraction declares GeV^-1. Both get x2 m_N,
    # so a shared-GeV^-1 match is PRESERVED (equal factor both sides).
    gt = _gt_entry("AxionProton", "g_ap", "limit_data/AxionProton/NASDUCK-SERF.txt",
                   [[1e-13, 1e-9], [1e-12, 2e-9], [1e-11, 3e-9]])
    res = {"coupling_type": "AxionProton",
           "data_points": [[1e-13, 1.05e-9], [1e-11, 2.9e-9]],
           "data_source": "text", "coupling_convention": "GeV^-1"}
    rec = sc._paper_record("2209.13588-like", res, [gt])
    assert rec["status"] == "compared"
    assert rec["median_resid"] < 0.5   # near-match preserved, not a ~0.3 dex 2m_N gap


def test_comparator_no_field_is_backcompat_noop():
    # Field-less snapshot (old): no canonicalization -> raw comparison (unchanged).
    gt = _gt_entry("AxionProton", "g_ap", "limit_data/AxionProton/NASDUCK-SERF.txt",
                   [[1e-13, 1e-9], [1e-11, 3e-9]])
    res = {"coupling_type": "AxionProton",
           "data_points": [[1e-13, 1e-9], [1e-11, 3e-9]], "data_source": "text"}
    rec = sc._paper_record("nofield", res, [gt])
    assert rec["status"] == "compared"  # still compared, no crash, raw


def test_comparator_canonicalize_helper_noop_on_unknown():
    import numpy as np
    arr = np.array([[1e-5, 1e-12]])
    out = sc._canonicalize_curve("AxionPhoton", arr, None)
    assert np.array_equal(out, arr)
