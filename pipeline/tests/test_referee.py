"""Unit tests for the confusable-pair classification referee (PR A).

The referee is a focused, conservative second opinion on the coupling type: it
re-picks only WITHIN the predicted type's confusable neighbour set, spends an
API call on the high-volume types only when a deterministic signal warrants it,
and overrides only on a confident in-candidate pick. These tests pin that
contract with a mocked model — no network/API.

Run:
    python -m pytest pipeline/tests/test_referee.py -v
"""

from __future__ import annotations

import json
import types

import pytest

import pipeline.extractor as ex
from pipeline.extractor import (
    _CONFUSABLE_NEIGHBORS,
    _REFEREE_DOMINANT,
    _coupling_magnitude_summary,
    _coupling_range_suspicious,
    _referee_coupling_type,
)


def _paper(title="T", summary="S"):
    return types.SimpleNamespace(title=title, summary=summary)


def _mock_model(monkeypatch, *, pick, conf, calls):
    """Patch extractor._create to return a fixed JSON response and count calls."""
    def fake_create(client, **kw):
        calls.append(kw)
        text = json.dumps({"coupling_type": pick, "confidence": conf})
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])
    monkeypatch.setattr(ex, "_create", fake_create)


# ---------------------------------------------------------------------------
# Routing / cost gate — when does the referee spend a call?
# ---------------------------------------------------------------------------

def test_non_confusable_type_no_call(monkeypatch):
    calls = []
    _mock_model(monkeypatch, pick="MonopoleDipole", conf=0.99, calls=calls)
    ct, note = _referee_coupling_type("MonopoleDipole", _paper(), [], None)
    assert ct == "MonopoleDipole" and note == "" and calls == []


def test_dominant_type_no_signal_no_call(monkeypatch):
    calls = []
    _mock_model(monkeypatch, pick="DarkPhoton", conf=0.99, calls=calls)
    # AxionPhoton is dominant; pre_ct agrees; coupling in-band -> no call.
    pts = [(1e-6, 1e-12), (1e-5, 1e-11)]  # g_agamma-scale, in AxionPhoton band
    ct, note = _referee_coupling_type("AxionPhoton", _paper(), pts, None,
                                      pre_ct="AxionPhoton")
    assert ct == "AxionPhoton" and note == "" and calls == []


def test_dominant_type_disagreement_triggers_call(monkeypatch):
    calls = []
    _mock_model(monkeypatch, pick="AxionPhoton", conf=0.5, calls=calls)  # low conf -> keep
    pts = [(1e-6, 1e-12)]
    ct, note = _referee_coupling_type("AxionPhoton", _paper(), pts, None,
                                      pre_ct="AxionElectron")  # disagreement
    assert len(calls) == 1  # a call was spent because of the disagreement signal


def test_dominant_type_suspicious_range_triggers_call(monkeypatch):
    calls = []
    _mock_model(monkeypatch, pick="AxionMass", conf=0.9, calls=calls)
    # AxionPhoton band is (1e-25, 1e-3); values ~1e9 (f_a-like) are out of band.
    pts = [(1e-6, 1e9), (1e-5, 1e10)]
    ct, note = _referee_coupling_type("AxionPhoton", _paper(), pts, None,
                                      pre_ct="AxionPhoton")
    assert len(calls) == 1
    assert ct == "AxionMass"  # AxionMass is in AxionPhoton's neighbour set


def test_rare_confusable_type_always_refereed(monkeypatch):
    calls = []
    _mock_model(monkeypatch, pick="AxionProton", conf=0.9, calls=calls)
    ct, note = _referee_coupling_type("AxionNeutron", _paper(), [], None,
                                      pre_ct="AxionNeutron")  # agree, no range
    assert len(calls) == 1  # rare type refereed regardless of signal


# ---------------------------------------------------------------------------
# Override discipline — within-cluster only, confidence-gated
# ---------------------------------------------------------------------------

def test_override_within_cluster_confident(monkeypatch):
    _mock_model(monkeypatch, pick="AxionProton", conf=0.9, calls=[])
    ct, note = _referee_coupling_type("AxionNeutron", _paper(), [], None)
    assert ct == "AxionProton" and note.startswith("[REFEREE]")


def test_reject_pick_outside_cluster(monkeypatch):
    # Model returns a type not in AxionNeutron's neighbour set -> keep original.
    _mock_model(monkeypatch, pick="DarkPhoton", conf=0.99, calls=[])
    ct, note = _referee_coupling_type("AxionNeutron", _paper(), [], None)
    assert ct == "AxionNeutron" and note == ""


def test_low_confidence_keeps_original(monkeypatch):
    _mock_model(monkeypatch, pick="AxionProton", conf=0.5, calls=[])
    ct, note = _referee_coupling_type("AxionNeutron", _paper(), [], None)
    assert ct == "AxionNeutron" and note == ""


def test_same_pick_no_note(monkeypatch):
    _mock_model(monkeypatch, pick="AxionNeutron", conf=0.99, calls=[])
    ct, note = _referee_coupling_type("AxionNeutron", _paper(), [], None)
    assert ct == "AxionNeutron" and note == ""


def test_fatal_api_error_propagates(monkeypatch):
    def boom(client, **kw):
        raise ex.FatalAPIError("billing")
    monkeypatch.setattr(ex, "_create", boom)
    with pytest.raises(ex.FatalAPIError):
        _referee_coupling_type("AxionNeutron", _paper(), [], None)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_coupling_range_suspicious():
    # AxionPhoton band (1e-25, 1e-3). Whole set above hi -> suspicious.
    assert _coupling_range_suspicious([(1e-6, 1e9), (1e-5, 1e10)], "AxionPhoton") is True
    # In-band -> not suspicious.
    assert _coupling_range_suspicious([(1e-6, 1e-12)], "AxionPhoton") is False
    # Empty / unknown type -> not suspicious.
    assert _coupling_range_suspicious([], "AxionPhoton") is False
    assert _coupling_range_suspicious([(1e-6, 1e-12)], None) is False


def test_coupling_magnitude_summary():
    assert _coupling_magnitude_summary([]) is None
    s = _coupling_magnitude_summary([(1e-6, 3e-12), (1e-5, 5e-10)])
    assert "3.00e-12" in s and "5.00e-10" in s


# ---------------------------------------------------------------------------
# Regression: every dense benchmark confusion pair is reachable
# ---------------------------------------------------------------------------

# (ground_truth, predicted) pairs measured >=2x on final2_opus_n1 vs GT.
_DENSE_CONFUSIONS = [
    ("ScalarElectron", "ScalarPhoton"),
    ("AxionMass", "AxionEDM"),
    ("AxionProton", "AxionNeutron"),
    ("DarkPhoton", "AxionElectron"),
    ("DarkPhoton", "AxionPhoton"),
    ("AxionMass", "AxionPhoton"),
    ("AxionMass", "ScalarNucleon"),
    ("AxionNeutron", "AxionEDM"),
    ("AxionPhoton", "AxionElectron"),
    ("ScalarPhoton", "ScalarNucleon"),
    ("VectorBL", "DarkPhoton"),
    ("AxionMass", "AxionNeutron"),
    ("DarkPhoton", "VectorBL"),
]


@pytest.mark.parametrize("gt,pred", _DENSE_CONFUSIONS)
def test_dense_confusions_reachable(gt, pred):
    # For a GT->pred confusion, the referee (keyed on pred) must be able to
    # output GT, i.e. GT is in the predicted type's neighbour set.
    assert gt in _CONFUSABLE_NEIGHBORS.get(pred, ()), f"{gt} unreachable from {pred}"


def test_self_is_first_neighbor():
    for pred, neigh in _CONFUSABLE_NEIGHBORS.items():
        assert neigh[0] == pred, f"{pred} should be its own first candidate"


def test_dominant_types_are_confusable_keys():
    for t in _REFEREE_DOMINANT:
        assert t in _CONFUSABLE_NEIGHBORS
