"""Unit tests for the deterministic scale correction in ``pipeline.extractor``
(issue #561).

The order-of-magnitude / unit auto-correction used to land on a *different*
factor across repeated runs of the same paper, producing 2-17 dex run-to-run
swings in the final coupling scale. The root causes were (a) continuous,
LLM-read multiplicative factors and (b) first-match / point-ordering
heuristics. The fix makes the correction a pure deterministic function of the
extracted points (+ any readings), chosen from a fixed discrete candidate set
with deterministic tie-breaking.

These tests pin that contract. They do NOT touch the network/API — only the
pure helpers ``_validate_extracted_range``, ``_calibrate_vision_data`` and
``_choose_discrete_factor``.

Run:
    python -m pytest pipeline/tests/test_extractor_correction.py -v
"""

from __future__ import annotations

import math
import random

import pytest

from pipeline.extractor import (
    _COUPLING_FACTOR_CANDIDATES,
    _MASS_FACTOR_CANDIDATES,
    _calibrate_vision_data,
    _choose_discrete_factor,
    _validate_extracted_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_points(masses, coupling=1e-12):
    """A synthetic 2-column point list with the given masses (arbitrary coupling)."""
    return [(float(m), float(coupling)) for m in masses]


def _median_mass(points):
    ms = sorted(p[0] for p in points if p[0] > 0)
    return ms[len(ms) // 2]


def _median_coupling(points):
    cs = sorted(p[1] for p in points if p[1] > 0)
    return cs[len(cs) // 2]


def _shuffled(points, seed):
    out = list(points)
    random.Random(seed).shuffle(out)
    return out


# ---------------------------------------------------------------------------
# _choose_discrete_factor: the core deterministic primitive
# ---------------------------------------------------------------------------

class TestChooseDiscreteFactor:
    def test_identity_preferred_on_tie(self):
        # value already at target → identity factor wins.
        f, _ = _choose_discrete_factor(1e-3, 1e-3, _COUPLING_FACTOR_CANDIDATES)
        assert f == 1.0

    def test_snaps_jitter_to_same_decade(self):
        # Two raw ratios that round to the same nearest decade must snap equal.
        f1, _ = _choose_discrete_factor(1.0, 9.0, _COUPLING_FACTOR_CANDIDATES)
        f2, _ = _choose_discrete_factor(1.0, 11.0, _COUPLING_FACTOR_CANDIDATES)
        assert f1 == f2 == 10.0

    def test_pure_function_repeatable(self):
        results = {
            _choose_discrete_factor(1e9, 1e-3, _MASS_FACTOR_CANDIDATES)
            for _ in range(50)
        }
        assert len(results) == 1

    def test_in_range_filter_falls_back_to_identity(self):
        # If no candidate lands in range, identity is returned.
        f, label = _choose_discrete_factor(
            1.0, 1.0, _MASS_FACTOR_CANDIDATES, in_range=lambda c: False
        )
        assert (f, label) == (1.0, "none")

    def test_tie_break_prefers_gentle_factor(self):
        # Equidistant-in-log candidates: gentler |log10(factor)| wins after identity.
        # target chosen so x10 and x100 are NOT equidistant; use a constructed case.
        cands = [(1.0, "none"), (10.0, "x10"), (1e6, "x1e6")]
        f, _ = _choose_discrete_factor(1.0, 1e3, cands)  # closest is x1e6? log dist: x10->2, x1e6->3
        assert f == 10.0


# ---------------------------------------------------------------------------
# Determinism under shuffling / repetition (PRIMARY pass/fail bar)
# ---------------------------------------------------------------------------

# (coupling_type, masses) covering in-range, out-of-range, and cluster inputs.
_DETERMINISM_CASES = [
    ("AxionPhoton", [0.165, 0.5, 1.2, 2.84]),          # #2211 cluster
    ("AxionPhoton", [1e-4, 1e-3, 0.5, 2.0]),           # #1709 cluster
    ("ScalarElectron", [1e1, 1e3, 1e6, 1e8]),          # #2111 cluster
    ("AxionPhoton", [1.3e9, 1.3e9, 1.3e9, 1.29e9]),    # #2112 cluster
    ("AxionPhoton", [1.8, 5.0, 10.0, 18.0]),           # #2401 cluster
    ("AxionElectron", [1e7, 1e8, 1e9, 3e9]),           # #2207 cluster
    ("ScalarElectron", [1e11, 1e13, 1e15]),            # hard out-of-range
    ("DarkPhoton", [1e-6, 1e-5, 1e-4]),                # already valid, untouched
]


@pytest.mark.parametrize("coupling_type,masses", _DETERMINISM_CASES)
def test_validate_deterministic_under_shuffle(coupling_type, masses):
    base = _make_points(masses)
    ref_pts, ref_note = _validate_extracted_range(list(base), coupling_type)
    ref_median = _median_mass(ref_pts)
    for seed in range(8):
        pts, note = _validate_extracted_range(_shuffled(base, seed), coupling_type)
        assert math.isclose(_median_mass(pts), ref_median, rel_tol=1e-12), (
            f"shuffle seed={seed} changed median for {coupling_type} {masses}"
        )
        assert note == ref_note


@pytest.mark.parametrize("coupling_type,masses", _DETERMINISM_CASES)
def test_validate_repeatable(coupling_type, masses):
    base = _make_points(masses)
    medians = set()
    for _ in range(20):
        pts, _note = _validate_extracted_range(list(base), coupling_type)
        medians.add(round(math.log10(_median_mass(pts)), 9))
    assert len(medians) == 1


# ---------------------------------------------------------------------------
# In-range "unit_offset" clusters must be LEFT UNTOUCHED (issue #561 fix)
# ---------------------------------------------------------------------------

# (arxiv_id, coupling_type, masses) — these all sit INSIDE the wide VALID_RANGES
# window. The original #561 SOFT anchor-distance trigger rescaled them toward a
# fixed per-coupling anchor (e.g. 1e-5 eV), which the #550/#561 before/after eval
# showed destroyed ~20 already-correct extractions (the unit_offset zero-overlap
# cluster). The corrected behaviour is: in-range data is never rescaled. We can
# only correct GROSS, out-of-range unit blunders (the hard trigger below).
_IN_RANGE_CASES = [
    ("2211.12699", "AxionPhoton", [0.165, 0.5, 1.2, 2.84]),     # eV-scale ALP
    ("1709.00009", "AxionPhoton", [1e-4, 1e-3, 0.5, 2.0]),
    ("2111.06883", "ScalarElectron", [1e1, 1e3, 1e6, 1e8]),
    ("2112.03439", "AxionPhoton", [1.3e9, 1.3e9, 1.29e9]),      # GeV-scale collider
    ("2401.16747", "AxionPhoton", [1.8e3, 5e3, 1e4, 1.8e4]),    # keV-scale X-ray (real eROSITA window)
]


@pytest.mark.parametrize("arxiv_id,coupling_type,masses", _IN_RANGE_CASES)
def test_in_range_clusters_left_untouched(arxiv_id, coupling_type, masses):
    """In-range mass clusters must NOT be rescaled, regardless of how far they
    sit from any nominal "expected" mass scale. This is the #561 regression fix:
    the soft anchor-distance trigger that used to snap these toward 1e-5 eV is
    gone. We still require determinism (identical result under shuffling)."""
    base = _make_points(masses)
    pts, note = _validate_extracted_range(list(base), coupling_type)

    # No mass correction applied; data is preserved exactly.
    assert "Auto-corrected masses" not in note, (
        f"{arxiv_id}: in-range data was spuriously rescaled: {note!r}"
    )
    assert _median_mass(pts) == _median_mass(base)

    # Still deterministic under shuffling (no-op is trivially deterministic).
    for seed in range(5):
        spts, snote = _validate_extracted_range(_shuffled(base, seed), coupling_type)
        assert _median_mass(spts) == _median_mass(base)
        assert snote == note


def test_hard_out_of_range_recovered():
    """A median far above mass_hi must be pulled back into range deterministically."""
    base = _make_points([1e11, 1e13, 1e15])
    pts, note = _validate_extracted_range(list(base), "ScalarElectron")
    med = _median_mass(pts)
    # ScalarElectron mass range is (1e-24, 1e9); corrected median must be in-window.
    assert 1e-25 <= med <= 1e10
    assert "Auto-corrected masses" in note


def test_in_range_data_untouched():
    """Data already at a physical scale near the anchor must not be rescaled."""
    base = _make_points([1e-6, 1e-5, 1e-4], coupling=1e-13)
    pts, note = _validate_extracted_range(list(base), "DarkPhoton")
    assert _median_mass(pts) == _median_mass(base)
    assert "Auto-corrected masses" not in note


# ---------------------------------------------------------------------------
# Idempotence: applying the full correction twice == once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coupling_type,masses", _DETERMINISM_CASES)
def test_validate_idempotent(coupling_type, masses):
    base = _make_points(masses)
    once, note1 = _validate_extracted_range(list(base), coupling_type)
    twice, _note2 = _validate_extracted_range(list(once), coupling_type)
    assert math.isclose(_median_mass(once), _median_mass(twice), rel_tol=1e-12)
    assert math.isclose(_median_coupling(once), _median_coupling(twice), rel_tol=1e-12)


def test_coupling_correction_idempotent():
    # Coupling many decades too large for AxionPhoton (coupling range 1e-25..1e-3).
    base = [(1e-5, 1e5), (1e-4, 1e6)]
    once, _ = _validate_extracted_range(list(base), "AxionPhoton")
    twice, _ = _validate_extracted_range(list(once), "AxionPhoton")
    assert math.isclose(_median_coupling(once), _median_coupling(twice), rel_tol=1e-12)
    # And the first pass actually brought it into range.
    assert _median_coupling(once) <= 1e-3 * 10


# ---------------------------------------------------------------------------
# _calibrate_vision_data: snaps LLM-read jitter to the same discrete factor
# ---------------------------------------------------------------------------

class TestCalibrateVisionData:
    def _points(self):
        return [(1e-5, 1e-13), (1e-4, 1e-12), (1e-3, 1e-11)]

    def test_spot_check_jitter_snaps_identically(self):
        # Stage-2 coupling at mass 1e-4 is 1e-12. Spot-check readings that all
        # imply a ~x10 correction (within one snapping bin: 10**0.5..10**1.5)
        # must collapse to the SAME applied factor despite LLM jitter.
        pts = self._points()
        outs = []
        for spot_c in (8e-12, 1e-11, 1.2e-11):  # ratios 8, 10, 12 -> all snap to x10
            vr = {"boundary_at_mass": {"mass_eV": 1e-4, "coupling": spot_c}}
            out, _note = _calibrate_vision_data(list(pts), "AxionPhoton", None, vr)
            outs.append(_median_coupling(out))
        assert len({round(math.log10(o), 9) for o in outs}) == 1

    def test_no_calibration_when_ratio_near_one(self):
        pts = self._points()
        vr = {"boundary_at_mass": {"mass_eV": 1e-4, "coupling": 1.2e-12}}  # ratio 1.2 -> identity
        out, note = _calibrate_vision_data(list(pts), "AxionPhoton", None, vr)
        assert _median_coupling(out) == _median_coupling(pts)
        assert "No calibration needed" in note

    def test_calibration_deterministic_under_shuffle(self):
        pts = self._points()
        vr = {"boundary_at_mass": {"mass_eV": 1e-4, "coupling": 1e-11}}
        ref, _ = _calibrate_vision_data(list(pts), "AxionPhoton", None, vr)
        ref_med = _median_coupling(ref)
        for seed in range(6):
            out, _n = _calibrate_vision_data(_shuffled(pts, seed), "AxionPhoton", None, vr)
            assert math.isclose(_median_coupling(out), ref_med, rel_tol=1e-12)

    def test_calibration_idempotent_after_snap(self):
        # Applying calibration, then re-running with a now-consistent spot value,
        # must not move the data again.
        pts = self._points()
        vr = {"boundary_at_mass": {"mass_eV": 1e-4, "coupling": 1e-11}}  # ratio 10 -> x10
        once, _ = _calibrate_vision_data(list(pts), "AxionPhoton", None, vr)
        # After x10, stage2 at 1e-4 is 1e-11; a consistent reading now gives ratio 1.
        vr2 = {"boundary_at_mass": {"mass_eV": 1e-4, "coupling": 1e-11}}
        twice, _ = _calibrate_vision_data(list(once), "AxionPhoton", None, vr2)
        assert math.isclose(_median_coupling(once), _median_coupling(twice), rel_tol=1e-12)

    def test_empty_input(self):
        out, note = _calibrate_vision_data([], "AxionPhoton", None, {})
        assert out == []
        assert note == ""
