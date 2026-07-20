"""Tests for the also_published_types classification-grading mechanism.

Multi-plane papers (EP tests, stellar-cooling papers, reviews) publish limits
in more coupling planes than the benchmark pool ingested curves for. A
prediction of any demonstrably-published plane is a correct identification,
so ``papers.json`` entries may carry ``also_published_types`` (plus mandatory
``also_published_evidence``). The field affects ONLY classification grading —
curve comparison still runs exclusively against actual GT curves.

Run:
    pytest evaluation/tests/test_also_published_types.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import compute_all_metrics
from evaluation.ground_truth import GroundTruthEntry, load_ground_truth


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
        verification_date="2026-07-20",
    )
    base.update(kw)
    return GroundTruthEntry(**base)


def _result(ct):
    return {"arxiv_id": "0000.00000", "coupling_type": ct,
            "data_points": [[1e-6, 1e-12], [1e-5, 1e-12]],
            "extraction_confidence": 0.9, "num_points": 2}


def test_also_published_pick_graded_correct_without_curve_compare():
    e = _entry(also_published_types=["AxionElectron"],
               also_published_evidence="abstract quotes a g_ae limit")
    m = compute_all_metrics([e], [_result("AxionElectron")])
    p = m["per_paper"][0]
    assert p["coupling_type_correct"] is True
    assert p["coupling_type_expected"] == ["AxionElectron", "AxionPhoton"]
    # Authoritative (curve-bearing) couplings are reported unaugmented.
    assert p["true_couplings"] == ["AxionPhoton"]
    # No GT curve exists for the picked plane: no curve comparison happens.
    assert p["comparison_status"] == "no_comparable_gt"
    assert m["classification"]["coupling_type"]["accuracy"] == 1.0


def test_pick_outside_published_planes_still_wrong():
    e = _entry(also_published_types=["AxionElectron"],
               also_published_evidence="abstract quotes a g_ae limit")
    m = compute_all_metrics([e], [_result("DarkPhoton")])
    p = m["per_paper"][0]
    assert p["coupling_type_correct"] is False
    err = m["classification"]["coupling_type"]["errors"][0]
    assert "AxionElectron" in err["expected"]  # augmented set is reported


def test_default_entries_unaffected():
    e = _entry()
    assert e.also_published_types == []
    m = compute_all_metrics([e], [_result("AxionElectron")])
    assert m["per_paper"][0]["coupling_type_correct"] is False


def test_papers_json_also_published_have_evidence():
    entries = load_ground_truth()
    with_also = [e for e in entries if e.also_published_types]
    assert with_also, "expected at least one also_published_types entry"
    for e in with_also:
        assert e.also_published_evidence, e.arxiv_id
        # Never a lone echo of the authoritative type.
        assert all(t != e.coupling_type for t in e.also_published_types), e.arxiv_id
