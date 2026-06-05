"""Unit + integration tests for the extraction-regression gate (P4 / #569).

No API calls. Two layers:

* **Boundary unit tests** on the pure :func:`evaluation.gate.apply_rules`
  (synthetic `_summarize` dicts) — pin each G-rule at exactly its threshold and
  one tick past it (the #544-style guarantee that a metric/threshold bug is
  caught instantly).
* **Integration tests** on committed snapshot fixtures — the gate must reproduce
  the known #550+#561 FAIL (`evaluation/tests/fixtures/gate_regressed` vs the
  committed `evaluation/eval_runs/baseline`), the no-op PASS (baseline vs
  itself), and no-flap across two master repeats
  (`evaluation/tests/fixtures/gate_repeats` r0 vs r1).

Run:
    pytest evaluation/tests/test_gate.py -v
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import gate
from evaluation.gate import (
    COMPARED_SLACK,
    FLOOR_SLACK,
    FRAC_0_3_SLACK,
    NOISE_FLOOR_DEX,
    apply_rules,
    evaluate_gate,
    run_gate,
    _hard_floor_violations,
    _symmetric_ids,
)
from evaluation.subset_compare import _load_result_dir

BASELINE = PROJECT_ROOT / "evaluation" / "eval_runs" / "baseline"
REGRESSED = PROJECT_ROOT / "evaluation" / "tests" / "fixtures" / "gate_regressed"
REPEATS = PROJECT_ROOT / "evaluation" / "tests" / "fixtures" / "gate_repeats"
SUBSET = json.loads((PROJECT_ROOT / "evaluation" / "subset" / "subset.json").read_text())

# The end-to-end integration tests call evaluate_gate -> subset_compare ->
# evaluate._normalize_predicted_coupling, which lazily imports pipeline.reviewer
# (which imports anthropic). The boundary unit tests and the HARD-floor scan need
# none of that. Skip the full-stack tests when anthropic is absent (the minimal
# no-API eval_tests CI job) so the threshold logic still runs there; they run
# locally and once the eval_tests job installs the import-only deps.
try:
    import anthropic as _anthropic  # noqa: F401
    _HAVE_FULL_STACK = True
except Exception:
    _HAVE_FULL_STACK = False

requires_full_stack = pytest.mark.skipif(
    not _HAVE_FULL_STACK,
    reason="evaluate_gate pairs via pipeline.reviewer (imports anthropic); "
           "unavailable in the minimal no-API job",
)


# ---------------------------------------------------------------------------
# Synthetic summary builder + helpers for the pure boundary tests
# ---------------------------------------------------------------------------

def _summary(*, overall=0.5, n_compared=40, n_zero_overlap=10, unit_offset=0,
             fig_median=0.6, fig_frac=0.4):
    return {
        "n_total": 82,
        "n_compared": n_compared,
        "n_zero_overlap": n_zero_overlap,
        "zo_causes": {"unit_offset": unit_offset} if unit_offset else {},
        "overall_median_resid": overall,
        "mean_coverage": 0.6,
        "sources": {
            "figure_vision": {
                "papers": 50, "compared": n_compared, "zero_overlap": n_zero_overlap,
                "median_resid": fig_median, "frac_0_3": fig_frac,
            }
        },
    }


def _rows(b, a, *, b_logic=0, a_logic=0, b_floor=0, a_floor=0):
    return apply_rules(b, a, b_logic_n=b_logic, a_logic_n=a_logic,
                       b_floor_n=b_floor, a_floor_n=a_floor)


def _row(rows, rid):
    return next(r for r in rows if r.rule == rid)


# ---------------------------------------------------------------------------
# G1 — overall median residual, noise-floor tolerance
# ---------------------------------------------------------------------------

def test_g1_passes_at_exactly_noise_floor():
    rows = _rows(_summary(overall=0.5), _summary(overall=0.5 + NOISE_FLOOR_DEX))
    assert _row(rows, "G1").passed is True

def test_g1_fails_just_above_noise_floor():
    rows = _rows(_summary(overall=0.5), _summary(overall=0.5 + NOISE_FLOOR_DEX + 0.01))
    assert _row(rows, "G1").passed is False

def test_g1_fails_when_after_loses_all_compared():
    rows = _rows(_summary(overall=0.5), _summary(overall=None))
    assert _row(rows, "G1").passed is False


# ---------------------------------------------------------------------------
# G2 / G3 — counts with a small flap slack (N=3 voting + #585)
# ---------------------------------------------------------------------------

def test_g2_passes_within_flap_slack():
    from evaluation.gate import ZO_SLACK
    rows = _rows(_summary(n_zero_overlap=14), _summary(n_zero_overlap=14 + ZO_SLACK))
    assert _row(rows, "G2").passed is True

def test_g2_fails_beyond_flap_slack():
    from evaluation.gate import ZO_SLACK
    rows = _rows(_summary(n_zero_overlap=14), _summary(n_zero_overlap=14 + ZO_SLACK + 1))
    assert _row(rows, "G2").passed is False

def test_g3_passes_within_flap_slack():
    from evaluation.gate import UNIT_OFFSET_SLACK
    rows = _rows(_summary(unit_offset=0), _summary(unit_offset=UNIT_OFFSET_SLACK))
    assert _row(rows, "G3").passed is True

def test_g3_fails_beyond_flap_slack():
    from evaluation.gate import UNIT_OFFSET_SLACK
    rows = _rows(_summary(unit_offset=0), _summary(unit_offset=UNIT_OFFSET_SLACK + 1))
    assert _row(rows, "G3").passed is False


# ---------------------------------------------------------------------------
# G4 / G5 — figure_vision row
# ---------------------------------------------------------------------------

def test_g4_passes_at_exactly_noise_floor():
    rows = _rows(_summary(fig_median=0.642), _summary(fig_median=0.642 + NOISE_FLOOR_DEX))
    assert _row(rows, "G4").passed is True

def test_g4_fails_just_above_noise_floor():
    rows = _rows(_summary(fig_median=0.642), _summary(fig_median=0.642 + NOISE_FLOOR_DEX + 0.01))
    assert _row(rows, "G4").passed is False

def test_g5_passes_at_exactly_slack():
    rows = _rows(_summary(fig_frac=0.40), _summary(fig_frac=0.40 - FRAC_0_3_SLACK))
    assert _row(rows, "G5").passed is True

def test_g5_fails_just_below_slack():
    rows = _rows(_summary(fig_frac=0.40), _summary(fig_frac=0.40 - FRAC_0_3_SLACK - 0.01))
    assert _row(rows, "G5").passed is False


# ---------------------------------------------------------------------------
# G6 — logic errors (strict) + HARD-floor violations (slacked)
# ---------------------------------------------------------------------------

def test_g6_fails_on_one_new_logic_error():
    rows = _rows(_summary(), _summary(), b_logic=0, a_logic=1)
    assert _row(rows, "G6").passed is False

def test_g6_logic_errors_are_strict_no_slack():
    rows = _rows(_summary(), _summary(), b_logic=2, a_logic=3)
    assert _row(rows, "G6").passed is False

def test_g6_passes_floor_within_slack():
    rows = _rows(_summary(), _summary(), b_floor=6, a_floor=6 + FLOOR_SLACK)
    assert _row(rows, "G6").passed is True

def test_g6_fails_floor_beyond_slack():
    rows = _rows(_summary(), _summary(), b_floor=6, a_floor=6 + FLOOR_SLACK + 1)
    assert _row(rows, "G6").passed is False


# ---------------------------------------------------------------------------
# G7 — papers-compared collapse
# ---------------------------------------------------------------------------

def test_g7_passes_at_exactly_slack():
    rows = _rows(_summary(n_compared=45), _summary(n_compared=45 - COMPARED_SLACK))
    assert _row(rows, "G7").passed is True

def test_g7_fails_beyond_slack():
    rows = _rows(_summary(n_compared=45), _summary(n_compared=45 - COMPARED_SLACK - 1))
    assert _row(rows, "G7").passed is False


# ---------------------------------------------------------------------------
# Integration — committed snapshot fixtures
# ---------------------------------------------------------------------------

@requires_full_stack
def test_known_regression_is_blocked():
    # baseline (N=3 voted) vs the #550+#561 regressed snapshot: the gate must
    # BLOCK. The regression adds +18 zero-overlap and +12 unit_offset, so the
    # count rules catch it robustly regardless of the (noisier post-#580)
    # residual floors; G2/G3 are the stable signature of a gross regression.
    rows, _ = evaluate_gate(BASELINE, REGRESSED, SUBSET["union"])
    assert _row(rows, "G2").passed is False
    assert _row(rows, "G3").passed is False
    assert run_gate(BASELINE, REGRESSED, SUBSET["union"]) == 1

@requires_full_stack
def test_noop_baseline_vs_itself_passes():
    rows, _ = evaluate_gate(BASELINE, BASELINE, SUBSET["union"])
    assert all(r.passed for r in rows)
    assert run_gate(BASELINE, BASELINE, SUBSET["union"]) == 0

@requires_full_stack
def test_soft_mode_never_fails():
    assert run_gate(BASELINE, REGRESSED, SUBSET["union"], soft=True) == 0

def test_hard_floor_scan_flags_regressed_snapshot():
    # Acceptance #5: >=1 HARD-floor violation on the pre-P0 regressed snapshot.
    viol = _hard_floor_violations(_load_result_dir(REGRESSED), SUBSET["union"])
    assert len(viol) >= 1

@requires_full_stack
def test_no_flap_across_master_repeats(tmp_path):
    # Two master repeats (same code, different LLM samples) must not trip the
    # gate — validates the noise-floor tolerances against real repeat spread.
    r0, r1 = tmp_path / "r0", tmp_path / "r1"
    r0.mkdir(); r1.mkdir()
    for f in REPEATS.glob("*_r0.json"):
        shutil.copy(f, r0 / f.name.replace("_r0", ""))
    for f in REPEATS.glob("*_r1.json"):
        shutil.copy(f, r1 / f.name.replace("_r1", ""))
    assert run_gate(r0, r1, SUBSET["unit_offset"]) == 0


def test_fixtures_present():
    # Guard against an incomplete checkout silently skipping the integration body.
    # Baseline re-pinned to the post-roadmap union(80) (#598 pruned 2 invalid GTs
    # from the 82-paper pin). The REGRESSED fixture is a frozen 82-paper artifact
    # (the 2 pruned papers are simply not in SUBSET["union"], so never scored).
    assert len([f for f in BASELINE.glob("*.json") if f.stem != "META"]) == 80
    assert (BASELINE / "META.json").exists()
    assert len(list(REGRESSED.glob("*.json"))) == 82
    assert len(list(REPEATS.glob("*.json"))) == 33


# ---------------------------------------------------------------------------
# Symmetric-set counting (infra drops cancel; logic errors kept)
# ---------------------------------------------------------------------------

def test_symmetric_ids_drops_infra_error_either_side():
    b = {
        "p1": {"data_points": [[1, 1]]},                       # clean both sides
        "p2": {"error": "PDF download failed: 429"},           # infra on before
        "p3": {"data_points": [[1, 1]]},
        "p4": {"data_points": [[1, 1]]},
    }
    a = {
        "p1": {"data_points": [[1, 1]]},
        "p2": {"data_points": [[1, 1]]},
        "p3": {"error": "read operation timed out"},           # infra on after
        "p4": {"error": "could not parse coupling"},           # LOGIC error -> kept
    }
    kept, dropped = _symmetric_ids(b, a, ["p1", "p2", "p3", "p4"])
    assert set(dropped) == {"p2", "p3"}        # infra on one side each
    assert kept == ["p1", "p4"]                # logic error stays in the set


def test_symmetric_ids_all_clean():
    b = {"p1": {"data_points": [[1, 1]]}}
    a = {"p1": {"data_points": [[1, 1]]}}
    kept, dropped = _symmetric_ids(b, a, ["p1"])
    assert kept == ["p1"] and dropped == []


@requires_full_stack
def test_gate_infra_drop_does_not_false_fail(tmp_path):
    # A paper that downloaded cleanly in the baseline but 429s in the PR run must
    # be dropped symmetrically, NOT counted as a new zero-overlap / lost-compared
    # (which would false-fail G2/G7). Build an after = baseline-copy with one
    # paper replaced by an infra error, and assert the gate still PASSes.
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir(); after.mkdir()
    src = [f for f in BASELINE.glob("*.json") if f.stem != "META"]
    for f in src:
        shutil.copy(f, before / f.name)
        shutil.copy(f, after / f.name)
    victim = src[0]
    (after / victim.name).write_text(json.dumps(
        {"arxiv_id": victim.stem, "error": "PDF download failed: 429 Unknown Error"}))
    rows, detail = evaluate_gate(before, after, SUBSET["union"])
    assert victim.stem in detail["dropped_infra"]
    assert all(r.passed for r in rows), [r.fmt() for r in rows if not r.passed]
