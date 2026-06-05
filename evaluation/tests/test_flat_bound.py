"""Tests for the mass-independent (flat) bound expansion in the comparator
(Lever E, #587).

No API/network. The extractor records a mass-independent bound with no usable
mass (every row mass <= 0). `_filter_boundary` drops mass <= 0, so such a bound
can never overlap a GT curve and is spuriously scored zero_overlap even when its
coupling matches GT (2007.03694 RedGiants: g_ae ~ 1.3e-13, matches GT to
~0.04 dex). `_expand_mass_independent` reshapes only the *extraction* into a
horizontal segment over [1e-30, 1e4] at the median positive coupling so the
coupling can be scored at the GT's masses.

Run:
    pytest evaluation/tests/test_flat_bound.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.subset_compare import (  # noqa: E402
    _FLAT_BOUND_MASS_HI,
    _FLAT_BOUND_MASS_LO,
    _expand_mass_independent,
)


def test_all_zero_mass_expands_to_segment():
    """All masses <= 0 with a positive coupling -> 2-row horizontal segment."""
    arr = np.array([[0.0, 1.58e-13], [0.0, 1.29e-13]])
    out = _expand_mass_independent(arr)
    assert out.shape == (2, 2)
    assert out[0, 0] == _FLAT_BOUND_MASS_LO
    assert out[1, 0] == _FLAT_BOUND_MASS_HI
    # Constant coupling = median of the positive couplings, flat across the segment.
    assert out[0, 1] == out[1, 1]
    assert out[0, 1] == pytest.approx(np.median([1.58e-13, 1.29e-13]))


def test_single_zero_mass_point_expands():
    """A single mass=0 text value also expands (1808.02340 shape)."""
    arr = np.array([[0.0, 1.1e-11]])
    out = _expand_mass_independent(arr)
    assert out.shape == (2, 2)
    assert out[0, 1] == pytest.approx(1.1e-11)
    assert out[0, 0] == _FLAT_BOUND_MASS_LO and out[1, 0] == _FLAT_BOUND_MASS_HI


def test_negative_masses_treated_as_sentinel():
    """Negative masses are non-positive sentinels too -> expand."""
    arr = np.array([[-1.0, 2e-13], [0.0, 4e-13]])
    out = _expand_mass_independent(arr)
    assert out.shape == (2, 2)
    assert out[0, 1] == pytest.approx(np.median([2e-13, 4e-13]))


def test_any_positive_mass_is_left_unchanged():
    """A real mass-dependent curve (any positive mass) is NOT touched."""
    arr = np.array([[1e-6, 1e-12], [1e-3, 2e-12]])
    out = _expand_mass_independent(arr)
    np.testing.assert_array_equal(out, arr)


def test_single_distinct_positive_mass_untouched():
    """A vertical-line vision artifact (single positive distinct mass, many rows,
    e.g. 2410.19902) is NOT a flat-bound sentinel and stays unchanged."""
    arr = np.array([[1.1e8, 0.053], [1.1e8, 0.01], [1.1e8, 1e-4]])
    out = _expand_mass_independent(arr)
    np.testing.assert_array_equal(out, arr)


def test_no_positive_coupling_left_unchanged():
    """All masses <= 0 but no positive coupling -> nothing to anchor, unchanged."""
    arr = np.array([[0.0, 0.0], [0.0, -1.0]])
    out = _expand_mass_independent(arr)
    np.testing.assert_array_equal(out, arr)


def test_empty_unchanged():
    assert _expand_mass_independent(np.empty((0, 2))).size == 0
