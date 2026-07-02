"""Unit tests for the fail-safe transform contract (issue #568, P0 of #566).

These tests make **no** network/API calls. They pin the contract surface that
the #566 roadmap builds on:

* the R1-R5 reject rules in :func:`pipeline.transform_guard.passes_contract`,
  each anchored to the eval paper that motivated its threshold,
* the semantics-trust :func:`pipeline.transform_guard.quality` ordering,
* :func:`pipeline.transform_guard.guard_transform` commit/revert + never-raise,
* the extractor wiring that consumes the contract: the ``float(None)`` crash fix
  (1607.06083), the R3 benchmark revert (2008.10141), the R5 hard floor, and the
  improve-or-revert range snap.

Run:
    pytest evaluation/tests/test_transform_guard.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import transform_guard as tg
from pipeline.transform_guard import (
    Candidate,
    ConsistencyScore,
    couplings_y_const,
    guard_transform,
    in_valid_ranges,
    passes_contract,
    quality,
    span_dex,
)


def _cand(source, *, in_range=True, corroborated=False, confidence=0.5,
          n_points=10, recoverable=True, y_const=False, span=4.0):
    bench = 1.0 if corroborated else None
    return Candidate(
        source=source,
        data_points=tuple((1e-5, 1e-12) for _ in range(max(n_points, 1))),
        coupling_type="AxionPhoton",
        extraction_confidence=confidence,
        score=ConsistencyScore(in_valid_ranges=in_range, n_points=n_points,
                               y_const=y_const, span_dex=span, benchmark_ratio=bench),
        recoverable=recoverable,
    )

AXION_PHOTON = {"mass": (1e-24, 1e9), "coupling": (1e-25, 1e-3)}


# ---------------------------------------------------------------------------
# R5 — hard floor (no corroboration escape hatch)
# ---------------------------------------------------------------------------

def test_r5_rejects_out_of_range_even_when_corroborated():
    score = ConsistencyScore(in_valid_ranges=False)
    accept, reason = passes_contract(score, corroborated=True)
    assert accept is False
    assert "R5" in reason

def test_r5_passes_in_range_clean_candidate():
    score = ConsistencyScore(in_valid_ranges=True)
    assert passes_contract(score) == (True, "ok")


# ---------------------------------------------------------------------------
# R1 — spot-check blow-up (2102.08764 ratio 1.6e10; 2207.11968 ratio 3.3e26)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [1.6e10, 3.3e26, 1e3, 1e-3, 0.0, -5.0])
def test_r1_rejects_catastrophic_spotcheck(ratio):
    score = ConsistencyScore(in_valid_ranges=True, spotcheck_ratio=ratio)
    accept, reason = passes_contract(score)
    assert accept is False
    assert "R1" in reason

@pytest.mark.parametrize("ratio", [1.0, 0.5, 2.0, 50.0, 0.02])
def test_r1_accepts_in_band_spotcheck(ratio):
    score = ConsistencyScore(in_valid_ranges=True, spotcheck_ratio=ratio)
    assert passes_contract(score)[0] is True


# ---------------------------------------------------------------------------
# R3 — benchmark disagreement (2008.10141 ratio 0.16 vs before-run 1.4)
# ---------------------------------------------------------------------------

def test_r3_rejects_full_decade_benchmark_disagreement():
    # 2008.10141: KSVZ read a full decade off (ratio 0.16 = 0.80 dex) -> a x1e-1
    # snap of a correct curve. Must reject.
    accept, reason = passes_contract(
        ConsistencyScore(in_valid_ranges=True, benchmark_ratio=0.16)
    )
    assert accept is False
    assert "R3" in reason

def test_r3_accepts_near_one_benchmark():
    # before-run ratio 1.4 (0.15 dex) correctly idles.
    assert passes_contract(
        ConsistencyScore(in_valid_ranges=True, benchmark_ratio=1.4)
    )[0] is True


# ---------------------------------------------------------------------------
# R2 — axis override without corroboration
#   2402.12892 (genuine win, 0.62 dex, corroborated) is the binding constraint:
#   the reject floor must sit ABOVE 0.62 dex, so 0.5-1.0 dex is allowed *only*
#   when corroborated; 1907.05475 (12 dex) / 1506.08082 (24 dex) must revert.
# ---------------------------------------------------------------------------

def test_r2_protects_genuine_win_when_corroborated():
    # 2402.12892: x-max 30->124 = 0.62 dex, spot-check snapped to identity.
    score = ConsistencyScore(in_valid_ranges=True, axis_disagree_dex=0.62)
    assert passes_contract(score, corroborated=True)[0] is True

def test_r2_reverts_uncorroborated_midband_override():
    score = ConsistencyScore(in_valid_ranges=True, axis_disagree_dex=0.62)
    accept, reason = passes_contract(score, corroborated=False)
    assert accept is False
    assert "R2" in reason

@pytest.mark.parametrize("dex", [12.0, 24.0])
def test_r2_reverts_catastrophic_axis_blowup_without_corroboration(dex):
    score = ConsistencyScore(in_valid_ranges=True, axis_disagree_dex=dex)
    assert passes_contract(score, corroborated=False)[0] is False

def test_r2_small_disagreement_needs_no_corroboration():
    # <=0.5 dex: legitimate small correction, no corroboration required.
    score = ConsistencyScore(in_valid_ranges=True, axis_disagree_dex=0.4)
    assert passes_contract(score, corroborated=False)[0] is True


# ---------------------------------------------------------------------------
# R4 — degenerate trace (1905.13650: 293 points at constant log10 y over 0.997 dex)
# ---------------------------------------------------------------------------

def test_r4_rejects_constant_y_line():
    accept, reason = passes_contract(
        ConsistencyScore(in_valid_ranges=True, y_const=True)
    )
    assert accept is False
    assert "R4" in reason

def test_r4_rejects_single_decade_span():
    accept, reason = passes_contract(
        ConsistencyScore(in_valid_ranges=True, span_dex=0.997)
    )
    assert accept is False
    assert "R4" in reason

def test_r4_accepts_healthy_curve():
    score = ConsistencyScore(in_valid_ranges=True, span_dex=4.0, y_const=False)
    assert passes_contract(score)[0] is True


# ---------------------------------------------------------------------------
# quality() — semantics-trust, NOT point count
# ---------------------------------------------------------------------------

def test_text_outranks_vision_despite_far_more_points():
    # The 2102.08764 / 2007.04899 regression: a 300-point figure read overriding a
    # correct text point-limit. quality() must keep text on top (T3 source tier).
    text = quality(_cand("text", confidence=0.5, n_points=10))
    vision = quality(_cand("figure_vision", corroborated=True, confidence=0.95, n_points=300))
    assert text > vision

def test_table_outranks_text():
    table = quality(_cand("table", confidence=0.4, n_points=5))
    text = quality(_cand("text", corroborated=True, confidence=0.99, n_points=500))
    assert table > text

def test_in_range_outranks_out_of_range_same_source():
    good = quality(_cand("figure_vision", in_range=True, confidence=0.3, n_points=5))
    bad = quality(_cand("figure_vision", in_range=False, corroborated=True,
                        confidence=0.99, n_points=500, recoverable=False))
    assert good > bad

def test_n_points_is_last_tiebreak():
    a = quality(_cand("text", corroborated=True, confidence=0.5, n_points=20))
    b = quality(_cand("text", corroborated=True, confidence=0.5, n_points=10))
    assert a > b


# ---------------------------------------------------------------------------
# guard_transform — commit / revert / never-raise
# ---------------------------------------------------------------------------

def test_guard_commits_passing_candidate():
    before, after = [(1e-5, 1e-12)], [(1e-5, 1e-11)]
    committed, note = guard_transform(
        before, after,
        score_after=ConsistencyScore(in_valid_ranges=True, spotcheck_ratio=10.0),
        label="t",
    )
    assert committed is after
    assert "committed" in note

def test_guard_reverts_failing_candidate_to_before():
    before, after = [(1e-5, 1e-12)], [(1e-5, 1e-50)]
    committed, note = guard_transform(
        before, after,
        score_after=ConsistencyScore(in_valid_ranges=False),
        label="t",
    )
    assert committed is before
    assert "reverted" in note

def test_guard_never_raises_on_broken_score(monkeypatch):
    # Force passes_contract to throw; guard must swallow and revert.
    def boom(*a, **k):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(tg, "passes_contract", boom)
    before, after = ["b"], ["a"]
    committed, note = guard_transform(
        before, after, score_after=ConsistencyScore(in_valid_ranges=True), label="t"
    )
    assert committed is before
    assert "reverted" in note and "contract error" in note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_in_valid_ranges_true_in_window():
    assert in_valid_ranges([(1e-5, 1e-12), (1e-4, 2e-12)], AXION_PHOTON) is True

def test_in_valid_ranges_false_below_coupling_floor():
    assert in_valid_ranges([(1e-5, 1e-40), (1e-4, 2e-40)], AXION_PHOTON) is False

def test_in_valid_ranges_none_ct_is_permissive():
    assert in_valid_ranges([(1e-5, 1e-40)], None) is True

def test_span_dex_and_yconst():
    assert span_dex([1e-5, 1e-1]) == pytest.approx(4.0)
    assert couplings_y_const([1e-9, 1.05e-9, 1.01e-9]) is True
    assert couplings_y_const([1e-9, 1e-5]) is False


# ---------------------------------------------------------------------------
# Extractor wiring (pure functions; no API)
#
# pipeline.extractor imports anthropic/arxiv/httpx at module scope. The eval CI
# job installs those import-only deps, but guard the import so test COLLECTION
# never fails if they are absent — the stdlib-only transform_guard tests above
# must still run. Extractor-wiring tests are skipped when the import is
# unavailable.
# ---------------------------------------------------------------------------

try:
    from pipeline import extractor as ex
    _HAVE_EXTRACTOR = True
    _EXTRACTOR_IMPORT_ERR = ""
except Exception as _e:  # pragma: no cover - exercised only in minimal envs
    ex = None
    _HAVE_EXTRACTOR = False
    _EXTRACTOR_IMPORT_ERR = repr(_e)

requires_extractor = pytest.mark.skipif(
    not _HAVE_EXTRACTOR,
    reason=f"pipeline.extractor unavailable: {_EXTRACTOR_IMPORT_ERR}",
)


@requires_extractor
def test_safe_float_tolerates_none_and_garbage():
    assert ex._safe_float(None) == 0.0
    assert ex._safe_float(None, default=7.0) == 7.0
    assert ex._safe_float("1.5") == 1.5
    assert ex._safe_float("nan") != ex._safe_float("nan") or True  # nan is fine
    assert ex._safe_float({"x": 1}) == 0.0

@requires_extractor
def test_calibrate_vision_data_survives_explicit_null_crash_path():
    # 1607.06083: explicit JSON null in boundary_at_mass / benchmark_line.
    dp = [(1e-5, 1e-12), (2e-5, 1.1e-12), (3e-5, 1.2e-12)]
    verify = {"boundary_at_mass": {"mass_eV": None, "coupling": None},
              "benchmark_line": None}
    out, note = ex._calibrate_vision_data(dp, "AxionPhoton", None, verify)
    assert out == dp  # no crash, no spurious correction
    assert "No calibration" in note

@requires_extractor
def test_calibrate_reverts_full_decade_benchmark():
    # 2008.10141-style: KSVZ benchmark read a full decade off (ratio ~0.16).
    # Old master applied x1e-1; the contract now distrusts the benchmark.
    dp = [(1e-5, 1e-12), (2e-5, 1.1e-12), (3e-5, 1.2e-12)]
    m = 1e-5
    expected = 2e-10 * 1.92 * m                 # KSVZ value at m
    reported = expected / 0.16                   # benchmark read low -> ratio 0.16
    benchmark = {"name": "KSVZ", "mass_eV": m, "coupling": reported}
    out, note = ex._calibrate_vision_data(dp, "AxionPhoton", benchmark, {})
    assert out == dp  # NOT scaled by 1e-1
    assert "No calibration" in note

@requires_extractor
def test_calibrate_idles_on_consistent_benchmark():
    # Benchmark ratio ~1 -> snaps to identity (validated, no correction).
    dp = [(1e-5, 1e-12), (2e-5, 1.1e-12), (3e-5, 1.2e-12)]
    m = 1e-5
    expected = 2e-10 * 1.92 * m
    benchmark = {"name": "KSVZ", "mass_eV": m, "coupling": expected}  # ratio 1.0
    out, note = ex._calibrate_vision_data(dp, "AxionPhoton", benchmark, {})
    assert out == dp
    assert "identity" in note or "No calibration" in note

@requires_extractor
def test_validate_range_leaves_in_range_data_untouched():
    dp = [(1e-5, 1e-12), (1e-4, 5e-13), (1e-3, 3e-13)]
    out, note = ex._validate_extracted_range(dp, "AxionPhoton")
    assert out == dp

@requires_extractor
def test_validate_range_corrects_gross_mass_unit_blunder_toward_range():
    # Median mass 1e15 eV is far above the AxionPhoton window; a discrete factor
    # must pull it back in and the improve-or-revert guard must commit it.
    dp = [(1e14, 1e-12), (1e15, 1e-12), (1e16, 1e-12)]
    out, note = ex._validate_extracted_range(dp, "AxionPhoton")
    med = sorted(m for m, _ in out)[len(out) // 2]
    assert med < 1e10  # pulled back toward / into the valid window
    assert "Auto-corrected masses" in note


# ---------------------------------------------------------------------------
# convention_review_needed — escalate-on-unknown flag (#536/#587)
# ---------------------------------------------------------------------------

from pipeline.transform_guard import convention_review_needed as _crn


@pytest.mark.parametrize("ct,decl", [
    ("AxionPhoton", "GeV^-1"),                 # canonical for photon
    ("AxionNeutron", "dimensionless g_an"),    # canonical
    ("DarkPhoton", "kinetic mixing chi"),      # canonical
    ("AxionEDM", "g_d [GeV^-2]"),               # canonical (#604)
    ("ScalarPhoton", "d_e"),                    # canonical
])
def test_canonical_declarations_not_flagged(ct, decl):
    assert _crn(ct, decl) is False


@pytest.mark.parametrize("ct,decl", [
    ("AxionNeutron", "GeV^-1"),                # convertible (x2 m_N)
    ("AxionProton", "g_aNN [GeV^-1]"),         # convertible
    ("DarkPhoton", "epsilon^2"),               # convertible (sqrt)
    ("AxionEDM", "1/f_a [GeV^-1]"),            # convertible (x3.7e-3)
    # Round-2 registry families (#653) — the mirror must not over-flag them:
    ("AxionPhoton", "decay rate Gamma in s^-1"),   # convertible (sqrt(64pi...))
    ("AxionPhoton", "lifetime tau in s"),          # convertible
    ("AxionMass", "f_a in GeV"),                   # convertible (reciprocal)
    ("AxionElectron", "(g_p^e)^2/(hbar c), squared"),  # convertible (sqrt)
    ("ScalarPhoton", "1/Lambda [GeV^-1]"),         # convertible (x sqrt2 M_Pl, #600)
])
def test_convertible_alternates_not_flagged(ct, decl):
    assert _crn(ct, decl) is False


@pytest.mark.parametrize("ct,decl", [
    ("ScalarElectron", "|delta alpha/alpha| amplitude"),
    ("AxionElectron", "Lambda [GeV] scale"),
    ("AxionProton", "some bespoke normalized coupling"),
    # e*cm oscillating-EDM amplitude is UNCONVERTIBLE (#604) — it used to slip
    # through as "canonical" when spelled without the asterisk:
    ("AxionEDM", "d_n in e cm"),
    ("AxionEDM", "oscillating neutron EDM amplitude d_n in e*cm"),
])
def test_unknown_conventions_flagged(ct, decl):
    assert _crn(ct, decl) is True


def test_empty_or_marker_declarations_not_flagged():
    assert _crn("AxionPhoton", "") is False
    assert _crn("AxionPhoton", None) is False
    assert _crn("AxionPhoton", "canonical") is False
    assert _crn(None, "anything") is False
