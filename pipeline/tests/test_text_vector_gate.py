"""Text-vector corroboration gate (#683 extension, catastrophic-tail audit).

The vector-trace channel's geometry is exact but its curve/panel identity is
one cheap model call; a wrong pick ships a dense conf-0.85 tier-3.5 garbage
curve that outranks the sparse correct text read (measured: 2102.06722 4.9 dex,
2110.01582 3.5 dex in full346_postfix_opus). `text_anchor_rejects` applies the
SAME 2-dex shared-support gate the vision channel has had since #683.

Pure function tests — no API calls.

Run:  pytest pipeline/tests/test_text_vector_gate.py -v
"""

import pytest

from pipeline.extractor import (
    TEXT_VISION_DISAGREE_DEX,
    _make_candidate,
    text_anchor_rejects,
)


def _text(points, ct="AxionPhoton", conf=0.7):
    return _make_candidate("text", points, ct, conf)


def _vector(points, ct="AxionPhoton"):
    return _make_candidate("vector_trace", points, ct, 0.85)


# Shared-support fixtures: anchor at g ~ 1e-10 over m in 1e-9..1e-8.
_ANCHOR = [(1e-9 * 10 ** (i / 4), 1e-10) for i in range(5)]


def test_rejects_gross_disagreement():
    # Vector curve 5 dex above the anchor over the same masses (2102.06722-like).
    bad = [(1e-9 * 10 ** (i / 10), 1e-5) for i in range(11)]
    note = text_anchor_rejects(_text(_ANCHOR), _vector(bad), "vector")
    assert note is not None and "TEXT-VECTOR DISAGREEMENT" in note


def test_keeps_agreeing_trace():
    good = [(1e-9 * 10 ** (i / 10), 1.5e-10) for i in range(11)]
    assert text_anchor_rejects(_text(_ANCHOR), _vector(good), "vector") is None


def test_threshold_is_the_shared_constant():
    # Just under the gate threshold -> keep; just over -> reject.
    near = [(m, g * 10 ** (TEXT_VISION_DISAGREE_DEX - 0.2)) for m, g in _ANCHOR]
    far = [(m, g * 10 ** (TEXT_VISION_DISAGREE_DEX + 0.5)) for m, g in _ANCHOR]
    assert text_anchor_rejects(_text(_ANCHOR), _vector(near), "vector") is None
    assert text_anchor_rejects(_text(_ANCHOR), _vector(far), "vector") is not None


def test_fails_open_without_shared_support():
    # 2006.09721-like: trace masses ~17 dex away from the anchor — no shared
    # support, nothing to corroborate -> fail open (the gate is not a blind
    # window check; Gate C owns wrong-regime rejection).
    elsewhere = [(1e-15 * 10 ** (i / 10), 1e-18) for i in range(11)]
    assert text_anchor_rejects(_text(_ANCHOR), _vector(elsewhere), "vector") is None


def test_fails_open_on_out_of_range_anchor():
    # A suspect (out-of-window) text anchor must not veto the trace.
    bad_anchor = [(1e-9, 1e+5), (1e-8, 1e+5)]   # coupling above AxionPhoton band
    bad = [(1e-9 * 10 ** (i / 10), 1e-5) for i in range(11)]
    t = _text(bad_anchor)
    assert not t.score.in_valid_ranges
    assert text_anchor_rejects(t, _vector(bad), "vector") is None


def test_fails_open_on_missing_candidates():
    assert text_anchor_rejects(None, _vector(_ANCHOR), "vector") is None
    assert text_anchor_rejects(_text(_ANCHOR), None, "vector") is None
