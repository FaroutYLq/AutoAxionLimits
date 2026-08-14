"""Pins for the equal-count (tie-aware quantile) confidence calibration bins.

The manuscript's calibration figure uses equal-occupancy bins because
extraction models emit only a handful of round confidence values; fixed-width
bins leave the extremes nearly empty. Ties are atomic: every paper sharing a
confidence value must land in the same bin.
"""
from evaluation.metrics import (
    _equal_count_confidence_groups,
    compute_confidence_calibration,
)


def test_ties_are_never_split():
    confs = [0.5] * 10 + [0.85] * 30 + [0.9] * 5
    groups = _equal_count_confidence_groups(confs, n_bins=3)
    flat = [v for g in groups for v in g]
    assert flat == sorted(set(confs))          # each value in exactly one group
    for g in groups:
        assert g == sorted(g)


def test_groups_cover_all_values_in_order():
    confs = [0.3, 0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.85, 0.9, 0.95]
    groups = _equal_count_confidence_groups(confs, n_bins=5)
    assert [v for g in groups for v in g] == sorted(set(confs))
    assert 1 <= len(groups) <= 5


def test_cannot_make_more_bins_than_distinct_values():
    groups = _equal_count_confidence_groups([0.5] * 100 + [0.9] * 3, n_bins=5)
    assert len(groups) == 2


def test_single_value_degenerates_to_one_group():
    assert _equal_count_confidence_groups([0.85] * 7, n_bins=5) == [[0.85]]


def test_occupancy_roughly_balanced():
    # 13 distinct round values with one dominant tie, like the real corpus
    confs = ([0.5] * 55 + [0.6] * 40 + [0.65] * 30 + [0.7] * 45 +
             [0.75] * 35 + [0.8] * 30 + [0.85] * 20 + [0.9] * 16)
    groups = _equal_count_confidence_groups(confs, n_bins=5)
    counts = {v: confs.count(v) for v in set(confs)}
    sizes = [sum(counts[v] for v in g) for g in groups]
    assert len(sizes) == 5
    # no bin may be starved relative to the equal-occupancy target
    target = len(confs) / 5
    assert min(sizes) >= 0.25 * target


def test_equal_width_mode_unchanged_and_unknown_mode_raises():
    import numpy as np
    import pytest
    from evaluation.metrics import InterpolationMetrics
    confs = [0.1, 0.5, 0.9]

    def im(aid):
        r = np.array([0.1, 0.2])
        return InterpolationMetrics(
            arxiv_id=aid, num_extracted=2, num_ground_truth=2,
            num_interpolatable=2, interpolation_coverage=1.0,
            residuals_dex=r, median_residual_dex=0.15,
            mean_residual_dex=0.15, p90_residual_dex=0.2,
            max_residual_dex=0.2, frac_within_0_1dex=0.5,
            frac_within_0_3dex=1.0, frac_within_0_5dex=1.0,
            frac_within_1_0dex=1.0)

    ims = [im(a) for a in ("a", "b", "c")]
    ids = ["a", "b", "c"]
    default = compute_confidence_calibration(confs, ims, ids)
    explicit = compute_confidence_calibration(confs, ims, ids, binning="equal_width")
    assert [b.bin_lo for b in default] == [b.bin_lo for b in explicit]
    with pytest.raises(ValueError):
        compute_confidence_calibration(confs, ims, ids, binning="nope")
