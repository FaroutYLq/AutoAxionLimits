"""Unit tests for the best-extraction selector (P2, issue #571).

No API calls. The pure-selector tests build synthetic `Candidate`s and assert the
winner for each of the five discriminating routing cases from the per-paper
fan-out (`evaluation/eval_runs/per_paper_findings.md`) — every one flips the
*opposite* direction from point count, which is exactly why a count-based gate
could never get them all right. The extractor-helper tests (which need
`pipeline.extractor` -> anthropic) self-skip in the minimal no-API job.

Run:
    pytest evaluation/tests/test_selector.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.transform_guard import (
    Candidate,
    ConsistencyScore,
    quality,
    select_best,
    should_consider_vision,
)


def _cand(source, *, in_range=True, conf=0.5, n=10, y_const=False, span=4.0,
          recoverable=True, bench=None, spot=None):
    return Candidate(
        source=source,
        data_points=tuple((1e-5, 1e-12) for _ in range(max(n, 1))),
        coupling_type="AxionPhoton",
        extraction_confidence=conf,
        score=ConsistencyScore(in_valid_ranges=in_range, n_points=n, y_const=y_const,
                               span_dex=span, benchmark_ratio=bench, spotcheck_ratio=spot),
        recoverable=recoverable,
    )


# ---------------------------------------------------------------------------
# The five discriminating routing cases (§1.2 of the P2 design)
# ---------------------------------------------------------------------------

def test_2204_01454_vision_beats_sparse_out_of_range_text():
    # 4-pt text conf 0.42, coupling 8.6e13 out-of-range/unrecoverable vs 29-pt
    # in-range vision. T0 (validity) settles it for vision.
    text = _cand("text", in_range=False, conf=0.42, n=4, recoverable=False)
    vision = _cand("figure_vision", in_range=True, conf=0.5, n=29)
    winner, _ = select_best([text, vision])
    assert winner.source == "figure_vision"


def test_1808_02340_text_beats_wrong_panel_flat_vision():
    # 5-pt in-range text vs a wrong-panel flat (y_const, span<1) vision read.
    # vision is degenerate (T1=0) and out-of-range (T0=0); text wins.
    text = _cand("text", in_range=True, conf=0.7, n=5)
    vision = _cand("figure_vision", in_range=False, conf=0.6, n=20,
                   y_const=True, span=0.5, recoverable=False)
    winner, _ = select_best([text, vision])
    assert winner.source == "text"


def test_2102_08764_text_beats_268pt_cv_benchmark_trace():
    # 2-pt exact text limit vs a 268-pt cv_trace of the DFSZ benchmark line; both
    # in-range, so T3 source tier (text > cv_trace) decides — not the 268 points.
    text = _cand("text", in_range=True, conf=0.6, n=2)
    cv = _cand("cv_trace", in_range=True, conf=0.5, n=268)
    winner, _ = select_best([text, cv])
    assert winner.source == "text"


def test_1905_13650_boundary_beats_293pt_degenerate_trace():
    # 27-pt LLM boundary vs a 293-pt cv_trace pinned to a flat line (y_const,
    # span 0.997). The trace is degenerate (T1=0); the boundary wins.
    boundary = _cand("figure_vision", in_range=True, conf=0.5, n=27)
    cv = _cand("cv_trace", in_range=True, conf=0.5, n=293, y_const=True, span=0.997)
    winner, _ = select_best([boundary, cv])
    assert winner.source == "figure_vision"


def test_2007_04899_vision_beats_3pt_text():
    # 3-pt text that lands out-of-range after a snap vs a 40-pt in-range vision.
    text = _cand("text", in_range=False, conf=0.55, n=3, recoverable=False)
    vision = _cand("figure_vision", in_range=True, conf=0.5, n=40)
    winner, _ = select_best([text, vision])
    assert winner.source == "figure_vision"


# ---------------------------------------------------------------------------
# quality() tier semantics
# ---------------------------------------------------------------------------

def test_validity_is_the_top_tier():
    good = _cand("cv_trace", in_range=True, conf=0.1, n=5)      # lowest tier...
    bad = _cand("table", in_range=False, conf=0.99, n=500, recoverable=False)  # highest tier
    assert quality(good) > quality(bad)  # T0 validity beats source tier

def test_point_limit_sources_exempt_from_degeneracy_penalty():
    # A 2-point text limit is sparse but must NOT be penalised as degenerate.
    text = _cand("text", in_range=True, conf=0.6, n=2, span=0.0)
    assert quality(text)[1] == 1  # T1 non-degenerate

def test_traced_curve_penalised_for_degeneracy():
    flat = _cand("figure_vision", in_range=True, n=200, y_const=True, span=0.5)
    assert quality(flat)[1] == 0

def test_n_points_only_breaks_full_ties():
    a = _cand("figure_vision", in_range=True, conf=0.5, n=50)
    b = _cand("figure_vision", in_range=True, conf=0.5, n=10)
    assert quality(a) > quality(b)
    # but a higher-tier candidate with fewer points still wins
    text = _cand("text", in_range=True, conf=0.5, n=2)
    assert quality(text) > quality(a)

def test_corroboration_breaks_ties_between_two_figure_reads():
    plain = _cand("figure_vision", in_range=True, conf=0.5, n=20)
    corrob = _cand("figure_vision", in_range=True, conf=0.5, n=20, bench=1.0)
    assert quality(corrob) > quality(plain)


# ---------------------------------------------------------------------------
# select_best edge cases + determinism
# ---------------------------------------------------------------------------

def test_select_best_empty():
    assert select_best([]) == (None, "no candidates")

def test_select_best_single():
    c = _cand("text")
    winner, reason = select_best([c])
    assert winner is c and "sole candidate" in reason

def test_select_best_is_deterministic_under_confidence_jitter():
    # T5 rounds confidence to 2 dp, so sub-0.005 jitter cannot reorder.
    a = _cand("figure_vision", in_range=True, conf=0.501, n=20)
    b = _cand("figure_vision", in_range=True, conf=0.503, n=20)
    assert quality(a) == quality(b)  # jitter collapses


# ---------------------------------------------------------------------------
# should_consider_vision gate
# ---------------------------------------------------------------------------

def test_strong_text_skips_vision():
    strong = _cand("text", in_range=True, conf=0.7, n=5)
    assert should_consider_vision(strong) is False

@pytest.mark.parametrize("text", [
    _cand("text", in_range=True, conf=0.42, n=4),     # low conf
    _cand("text", in_range=True, conf=0.7, n=4),      # sparse
    _cand("text", in_range=False, conf=0.7, n=5, recoverable=False),  # out of range
    _cand("text", in_range=True, conf=0.7, n=10, y_const=True),  # degenerate
])
def test_weak_text_runs_vision(text):
    assert should_consider_vision(text) is True

def test_no_text_runs_vision():
    assert should_consider_vision(None) is True


# ---------------------------------------------------------------------------
# Extractor candidate-builder helpers (need anthropic transitively)
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic  # noqa: F401
    _HAVE_EXTRACTOR = True
except Exception:
    _HAVE_EXTRACTOR = False

requires_extractor = pytest.mark.skipif(
    not _HAVE_EXTRACTOR, reason="pipeline.extractor needs anthropic (minimal no-API job)")


@requires_extractor
def test_make_candidate_scores_in_range():
    from pipeline.extractor import _make_candidate
    c = _make_candidate("text", [(1e-5, 1e-12), (1e-4, 5e-13)], "AxionPhoton", 0.7)
    assert c.score.in_valid_ranges is True and c.recoverable is True
    assert c.source == "text" and c.score.n_points == 2

@requires_extractor
def test_make_candidate_flags_unrecoverable_coupling():
    from pipeline.extractor import _make_candidate
    c = _make_candidate("text", [(1e-5, 8.6e13), (1e-4, 8.6e13)], "AxionEDM", 0.42)
    assert c.score.in_valid_ranges is False
    assert c.recoverable is False

@requires_extractor
def test_coupling_recoverable_by_decade_factor():
    from pipeline.extractor import _coupling_recoverable
    # AxionPhoton coupling window (1e-25, 1e-3); 1e-1 is one decade out but a
    # decade factor lands it in range -> recoverable.
    assert _coupling_recoverable([(1e-5, 1e-1)], "AxionPhoton") is True
    # 8.6e13 has no decade factor into (1e-40, 1e-15) -> not recoverable.
    assert _coupling_recoverable([(1e-5, 8.6e13)], "AxionEDM") is False
