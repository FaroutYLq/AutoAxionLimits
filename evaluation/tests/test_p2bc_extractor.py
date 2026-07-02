"""Tests for post-full346 Phase 2b+2c extractor fixes (no API).

* 2b (Lever 4): AxionMass VALID_RANGES floor widened to the fa-plane domain;
  explicit mass anchor; snaps revert unless they land the median inside the
  STRICT window; snaps suppressed when the declared convention is
  non-canonical (Lever 3 guard deferred from Phase 1d).
* 2c (Levers 6+7): wider result-excerpt context; Hz-only scope on the
  4.136e-15 prompt factor; Stage-1 flat-bound instruction present;
  R4 span floor scales to the figure's own axis extent; text candidates whose
  notes admit analytic reconstruction are demoted below figure_vision.

Run:
    pytest evaluation/tests/test_p2bc_extractor.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

anthropic = pytest.importorskip("anthropic")

from pipeline.config import VALID_RANGES
from pipeline.extractor import (
    _EXPECTED_MASS_ANCHOR_EV,
    _axis_extent_dex,
    _make_candidate,
    _notes_admit_reconstruction,
    _result_excerpts,
    _validate_extracted_range,
)
from pipeline.transform_guard import (
    ConsistencyScore,
    R4_MIN_SPAN_DEX,
    _r4_span_floor,
    passes_contract,
    quality,
)
import pipeline.extractor as extractor_mod


# ---------------------------------------------------------------------------
# 2b — AxionMass window + anchors + strict revert + suppression
# ---------------------------------------------------------------------------

def test_axionmass_window_covers_ultralight():
    lo, hi = VALID_RANGES["AxionMass"]["mass"]
    assert lo <= 1e-22  # superradiance / fuzzy-DM fa bounds
    assert hi == 1e18


def test_ultralight_axionmass_window_not_snapped():
    # 5 superradiance papers (masses 1e-21..1e-11) were snapped by the old
    # 1e-12 floor; with the widened window the data must pass through intact.
    pts = [(1e-21, 1e-10), (1e-15, 1e-9), (1e-11, 1e-8)]
    out, note = _validate_extracted_range(list(pts), "AxionMass")
    assert out == pts
    assert "Auto-correct" not in note


def test_axionmass_has_explicit_anchor():
    assert _EXPECTED_MASS_ANCHOR_EV["AxionMass"] == pytest.approx(1e-11)


def test_snap_suppressed_on_non_canonical_declaration():
    # 2105.13963 shape: declared f_a-in-GeV values 1.6e16..1e18 are outside
    # the AxionMass coupling window; the old soft snap fired x1e-20 and
    # corrupted the curve. With suppress_snaps the data pass through with a
    # SUPPRESSED note.
    pts = [(1e-12, 1.6e16), (5e-13, 1e18)]
    out, note = _validate_extracted_range(list(pts), "AxionMass", suppress_snaps=True)
    assert out == pts
    assert "SUPPRESSED" in note


def test_snap_reverted_if_median_lands_outside_strict_window():
    # A factor that moves toward the anchor but cannot reach the strict
    # window must be reverted, not committed as reduced corruption.
    # AxionPhoton coupling window (1e-25, 1e-3): median 1e30 — the largest
    # discrete factor cannot land it inside the strict window.
    pts = [(1e-6, 1e30), (1e-5, 1e30)]
    out, note = _validate_extracted_range(list(pts), "AxionPhoton")
    assert [tuple(p) for p in out] == pts  # unchanged
    assert "Auto-corrected couplings" not in note


# ---------------------------------------------------------------------------
# 2c — excerpts, prompts, R4 scaling, reconstruction demotion
# ---------------------------------------------------------------------------

def test_result_excerpt_window_includes_following_lines():
    # 2402.00741 shape: the value line trails the keyword line by 4 lines.
    text = "\n".join([
        "intro", "we set a new upper limit", "table follows", "col a", "col b",
        "d_e < 3.2e-5 at 95% CL", "conclusion", "refs",
    ])
    out = _result_excerpts(text, budget=10_000)
    assert "d_e < 3.2e-5" in out


def test_hz_factor_prompt_scoped_to_frequencies():
    src = Path(PROJECT_ROOT / "pipeline/extractor.py").read_text()
    assert src.count("applies ONLY to values the paper gives as frequencies") == 2


def test_stage1_prompt_has_flat_bound_instruction():
    from pipeline.extractor import _STAGE1_SYSTEM
    assert "MASS-INDEPENDENT (flat) bounds" in _STAGE1_SYSTEM
    assert "TWO-POINT horizontal line" in _STAGE1_SYSTEM


def test_r4_floor_scales_to_narrow_figure():
    # 2110.10497 shape: the whole figure spans 0.4 dex; a faithful full-width
    # trace (0.35 dex) must pass R4.
    s = ConsistencyScore(in_valid_ranges=True, n_points=12,
                         span_dex=0.35, y_const=False, axis_extent_dex=0.4)
    assert _r4_span_floor(s) == pytest.approx(0.2)
    ok, reason = passes_contract(s)
    assert ok, reason


def test_r4_floor_flat_without_axis_info():
    s = ConsistencyScore(in_valid_ranges=True, n_points=12,
                         span_dex=0.35, y_const=False)
    assert _r4_span_floor(s) == R4_MIN_SPAN_DEX
    ok, reason = passes_contract(s)
    assert not ok and "R4" in reason


def test_axis_extent_dex_parsing():
    assert _axis_extent_dex({"x_axis_min": 1e-6, "x_axis_max": 1e-4}) == pytest.approx(2.0)
    assert _axis_extent_dex({"x_axis_min": "?", "x_axis_max": 1e-4}) is None
    assert _axis_extent_dex(None) is None


def test_reconstruction_notes_detected():
    assert _notes_admit_reconstruction("curve analytically reconstructed from Eq. 12")
    assert _notes_admit_reconstruction("approximate read of Fig. 3")
    assert not _notes_admit_reconstruction("digitized from Table 2")


def test_reconstruction_text_loses_to_valid_vision():
    # Lever 6 (1401.6460 / 2407.10618): an analytically-reconstructed text
    # curve must rank below a valid vision trace of the real curve.
    pts_text = [(10 ** (-6 + 0.1 * i), 10 ** (-12 - 0.05 * i)) for i in range(20)]
    pts_vis = [(10 ** (-6 + 0.1 * i), 10 ** (-11.9 - 0.05 * i)) for i in range(20)]
    text_c = _make_candidate("text", pts_text, "AxionPhoton", 0.9,
                             demote_to_reconstruction=True)
    vis_c = _make_candidate("figure_vision", pts_vis, "AxionPhoton", 0.5)
    assert quality(vis_c) > quality(text_c)
    # And a clean (non-reconstructed) text read keeps beating vision.
    text_clean = _make_candidate("text", pts_text, "AxionPhoton", 0.9)
    assert quality(text_clean) > quality(vis_c)
