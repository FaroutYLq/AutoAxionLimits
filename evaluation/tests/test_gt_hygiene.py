"""Tests for post-full346 Phase 1a+1b GT hygiene.

Covers:
* the exclusion mechanism: excluded entries are skipped by compute_all_metrics
  (scoring AND classification) and surfaced in ``gt_exclusions`` /
  ``status_counts["excluded_gt"]``;
* ingestion-time unit conversions in populate_data_from_repo: lambda[m] → eV
  x-columns and per-file y-scale factors (COBEFIRAS_Cyr epsilon → g_agamma);
* per-entry data-file naming for multi-entry arXiv ids;
* papers.json invariants: every excluded entry documents its reason, and
  multi-entry ids no longer share one data file across different repo files.

Run:
    pytest evaluation/tests/test_gt_hygiene.py -v
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.build_ground_truth import data_file_name
from evaluation.evaluate import compute_all_metrics
from evaluation.ground_truth import (
    _HBARC_EV_M,
    GroundTruthEntry,
    _ingest_reference_file,
    load_ground_truth,
)


def _entry(**kw) -> GroundTruthEntry:
    base = dict(
        arxiv_id="0000.00000", paper_title="t", coupling_type="AxionPhoton",
        coupling_convention="g_GeV^-1", coupling_units="g [GeV^-1]",
        is_new_limit=True, is_projection=False, data_source_expected="table",
        confidence_level=0.9, dm_density_assumed=None, difficulty="medium",
        tags=[], notes="", ground_truth_data_file=None,
        reference_repo_file="limit_data/AxionPhoton/Fake.txt",
        ground_truth_mass_range_eV=None, ground_truth_coupling_range=None,
        ground_truth_num_points=None, verified_by="test",
        verification_date="2026-07-02",
    )
    base.update(kw)
    return GroundTruthEntry(**base)


# ---------------------------------------------------------------------------
# Exclusion mechanism
# ---------------------------------------------------------------------------

def test_fully_excluded_paper_not_scored():
    e = _entry(excluded=True, exclusion_reason="band file",
               exclusion_evidence="doc § 0000.00000")
    result = {"arxiv_id": "0000.00000", "coupling_type": "AxionPhoton",
              "data_points": [[1e-6, 1e-12], [1e-5, 1e-12]],
              "extraction_confidence": 0.9, "num_points": 2}
    m = compute_all_metrics([e], [result])
    assert m["comparison_coverage"]["status_counts"] == {"excluded_gt": 1}
    assert m["gt_exclusions"]["n_entries"] == 1
    assert m["gt_exclusions"]["n_papers_fully_excluded"] == 1
    assert m["gt_exclusions"]["entries"][0]["exclusion_reason"] == "band file"
    # No classification is scored against an excluded GT.
    assert m["classification"]["coupling_type"]["total"] == 0
    assert m["per_paper"][0]["status"] == "excluded_gt"


def test_partially_excluded_paper_scores_remaining_entries(tmp_path):
    gt = np.column_stack([np.logspace(-6, -5, 10), np.full(10, 1e-12)])
    f = tmp_path / "gt.txt"
    np.savetxt(f, gt)
    good = _entry(ground_truth_data_file=None)
    good.load_data = lambda: gt  # bypass file IO
    bad = _entry(coupling_type="AxionMass",
                 reference_repo_file="limit_data/AxionMass/Band.txt",
                 excluded=True, exclusion_reason="band file")
    result = {"arxiv_id": "0000.00000", "coupling_type": "AxionPhoton",
              "data_points": [[1.1e-6, 1e-12], [9e-6, 1e-12]],
              "extraction_confidence": 0.9, "num_points": 2}
    m = compute_all_metrics([good, bad], [result, result])
    assert m["comparison_coverage"]["status_counts"].get("compared") == 1
    assert m["gt_exclusions"]["n_entries"] == 1
    assert m["gt_exclusions"]["n_papers_fully_excluded"] == 0
    # Classification runs against the surviving entry only.
    assert m["classification"]["coupling_type"]["total"] == 1
    assert m["per_paper"][0]["coupling_type_correct"] is True


# ---------------------------------------------------------------------------
# Ingestion conversions
# ---------------------------------------------------------------------------

def test_lambda_header_converted_to_ev(tmp_path):
    src = tmp_path / "Moon.txt"
    src.write_text("# Moon\n# lambda [m]  g_s * g_p\n1.9732698e7 1e-26\n")
    lines = _ingest_reference_file(src, "limit_data/MonopoleDipole/NucleonNucleon/Moon.txt")
    x, y = map(float, lines[0].split())
    assert x == pytest.approx(_HBARC_EV_M / 1.9732698e7)
    assert x == pytest.approx(1e-14)
    assert y == pytest.approx(1e-26)


def test_cobefiras_y_scale(tmp_path):
    src = tmp_path / "COBEFIRAS_Cyr.txt"
    src.write_text("# epsilon defined as g_agamma*B0_T/(1e-11 GeV^-1 * 1 nG)\n"
                   "5.4e-14 1.0\n")
    lines = _ingest_reference_file(src, "limit_data/AxionPhoton/COBEFIRAS_Cyr.txt")
    x, y = map(float, lines[0].split())
    assert x == pytest.approx(5.4e-14)
    assert y == pytest.approx(1e-11)


def test_plain_file_untouched(tmp_path):
    src = tmp_path / "Plain.txt"
    src.write_text("# mass [eV] coupling\n1e-6 1e-12\n")
    lines = _ingest_reference_file(src, "limit_data/AxionPhoton/Plain.txt")
    assert lines == ["1e-6 1e-12"]


# ---------------------------------------------------------------------------
# Per-entry data files + papers.json invariants
# ---------------------------------------------------------------------------

def test_data_file_name_unique_per_repo_file():
    a = data_file_name("2112.12116", "limit_data/AxionElectron/XENON1T_DM_SE.txt")
    b = data_file_name("2112.12116", "limit_data/DarkPhoton/XENON1T_SE.txt")
    assert a != b
    assert a == "2112.12116__AxionElectron_XENON1T_DM_SE.txt"
    # Old-style ids stay filesystem-safe.
    assert "/" not in data_file_name("astro-ph/0611502", "limit_data/AxionPhoton/X.txt")


def test_papers_json_exclusions_documented():
    entries = load_ground_truth()
    excluded = [e for e in entries if e.excluded]
    assert len(excluded) >= 16  # 13 band + 3 unfixable + tombstones
    for e in excluded:
        assert e.exclusion_reason, e.arxiv_id
        assert e.exclusion_evidence, e.arxiv_id


def test_papers_json_no_cross_entry_data_file_collisions():
    doc = json.loads((PROJECT_ROOT / "evaluation/ground_truth/papers.json").read_text())
    by_file: dict[str, set] = defaultdict(set)
    for p in doc["papers"]:
        f = p.get("ground_truth_data_file")
        if f:
            by_file[f].add((p["arxiv_id"], p.get("reference_repo_file")))
    # A data file may be shared only by literal duplicate (id, repo_file) pairs,
    # never by entries for DIFFERENT repo files.
    for f, users in by_file.items():
        assert len(users) == 1, (f, users)


def test_band_papers_fully_excluded():
    entries = load_ground_truth()
    band_ids = {"1202.5851", "1505.07455", "1509.00026", "1606.03145",
                "1608.05414", "1705.00676", "1906.00967", "2007.04990",
                "2108.05368", "2206.11598", "2412.08699",
                "2005.14694", "2011.08693", "2112.03439", "1003.0964"}
    for aid in band_ids:
        es = [e for e in entries if e.arxiv_id == aid]
        assert es and all(e.excluded for e in es), aid
