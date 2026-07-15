"""Phase 4 (#625) — header-canonical d_e override for false d_e_large exclusions.

ScalarPhoton/HQuartzSapphire.txt and DyQuartz.txt declare `d_e` in their header
but their fill_between top wall trips the ymax>1e3 -> d_e_large range heuristic,
falsely excluding a canonical-d_e vs canonical-d_e pair. The override trusts the
header; a genuinely large-valued curve (WhiteDwarfs, uniformly ~1e6) is NOT
overridden. Evidence + audit: evaluation/ground_truth/PHASE4_convention_audit.md

Run:  pytest evaluation/tests/test_header_canonical_de.py -v
"""

from __future__ import annotations

from pathlib import Path

from evaluation.conventions import infer_convention


def test_header_canonical_files_are_de():
    for f in ("limit_data/ScalarPhoton/HQuartzSapphire.txt",
              "limit_data/ScalarPhoton/DyQuartz.txt"):
        conv, units = infer_convention("ScalarPhoton", Path(f))
        assert conv == "d_e", f"{f} should be canonical d_e, got {conv}"


def test_genuinely_large_curve_still_d_e_large():
    # WhiteDwarfs is uniformly ~2.6e6 (a real large-valued curve, extraction 22
    # dex off) — must STAY d_e_large so it keeps being excluded.
    conv, _ = infer_convention(
        "ScalarElectron", Path("limit_data/ScalarElectron/WhiteDwarfs.txt"))
    assert conv == "d_e_large"


def test_override_scoped_to_listed_files_only():
    # A non-listed ScalarPhoton file keeps the range-based inference.
    from evaluation.conventions import _HEADER_CANONICAL_DE_FILES
    assert "limit_data/ScalarPhoton/HQuartzSapphire.txt" in _HEADER_CANONICAL_DE_FILES
    assert "limit_data/ScalarElectron/WhiteDwarfs.txt" not in _HEADER_CANONICAL_DE_FILES
