"""Unit tests for N-sample read voting (Follow-up B of #566).

Pure / no API: curve distance + medoid consensus on synthetic point sets.

Run:
    pytest evaluation/tests/test_read_vote.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")
from pipeline.read_vote import curve_distance, select_consensus  # noqa: E402

# A clean 3-decade curve and a copy shifted +0.5 dex in coupling.
_CURVE = [(1e-6, 1e-12), (1e-4, 1e-11), (1e-2, 1e-10), (1.0, 1e-9)]
_CURVE_SHIFTED = [(m, g * 10 ** 0.5) for m, g in _CURVE]
_CURVE_OUTLIER = [(m, g * 10 ** 2.0) for m, g in _CURVE]   # +2 dex (drifted run)
_CURVE_DISJOINT = [(1e3, 1e-9), (1e6, 1e-8)]               # no mass overlap


# ---------------------------------------------------------------------------
# curve_distance
# ---------------------------------------------------------------------------

def test_curve_distance_identical_is_zero():
    assert curve_distance(_CURVE, _CURVE) == pytest.approx(0.0, abs=1e-6)


def test_curve_distance_constant_shift():
    assert curve_distance(_CURVE, _CURVE_SHIFTED) == pytest.approx(0.5, abs=1e-6)


def test_curve_distance_no_overlap_is_inf():
    assert curve_distance(_CURVE, _CURVE_DISJOINT) == float("inf")


def test_curve_distance_too_few_points_is_inf():
    assert curve_distance([(1.0, 1.0)], _CURVE) == float("inf")
    assert curve_distance([], _CURVE) == float("inf")


# ---------------------------------------------------------------------------
# select_consensus — medoid rejects the outlier sample
# ---------------------------------------------------------------------------

def test_medoid_rejects_outlier_curve():
    samples = [
        ("AxionPhoton", _CURVE),
        ("AxionPhoton", _CURVE_SHIFTED),   # 0.5 dex from CURVE
        ("AxionPhoton", _CURVE_OUTLIER),   # 2 dex from both -> outlier
    ]
    idx, note = select_consensus(samples)
    assert idx in (0, 1)                   # never the 2-dex outlier
    assert "medoid" in note and "AxionPhoton (3/3)" in note


def test_majority_coupling_vote():
    samples = [
        ("AxionPhoton", _CURVE),
        ("AxionPhoton", _CURVE_SHIFTED),
        ("DarkPhoton", _CURVE_OUTLIER),    # minority coupling -> excluded
    ]
    idx, note = select_consensus(samples)
    assert idx in (0, 1)
    assert "AxionPhoton (2/3)" in note


def test_single_sample():
    idx, note = select_consensus([("AxionPhoton", _CURVE)])
    assert idx == 0


def test_no_consensus_curve_falls_back_to_most_points():
    # all samples too sparse to form a curve -> take the modal-ct one with most pts
    samples = [
        ("AxionPhoton", [(1.0, 1e-10)]),
        ("AxionPhoton", [(1.0, 1e-10), (2.0, 1e-10)]),   # 2 pts -> this is a curve
        ("AxionPhoton", [(1.0, 1e-10)]),
    ]
    idx, note = select_consensus(samples)
    assert idx == 1


def test_all_no_data_returns_valid_index():
    samples = [("AxionPhoton", []), (None, []), ("AxionPhoton", [])]
    idx, note = select_consensus(samples)
    assert 0 <= idx < 3


def test_empty_samples():
    idx, note = select_consensus([])
    assert idx == 0
