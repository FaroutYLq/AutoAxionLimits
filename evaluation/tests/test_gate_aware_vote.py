"""Unit tests for the gate-aware read-vote consensus (#666).

No API calls: run_extraction_agent is monkeypatched to return scripted
samples. Pins the #666 contract:

* majority gate-rejected + a surviving fallback with points -> the vote runs
  over the gate-aware samples only (the unconfessed survivor cannot win on
  point count — the 1512.06165 resurrection);
* majority gate-rejected + no salvaged fallback -> zero points, confidence
  capped at 0.5, an explanatory note;
* minority rejection -> advisory only, the normal vote is unchanged;
* gate-D demotes (soft) never trigger the filter.

Run:
    pytest evaluation/tests/test_gate_aware_vote.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import extractor as ex

CURVE = [(10 ** (-6 + 0.1 * i), 10 ** (-11 - 0.05 * i)) for i in range(20)]
FALLBACK = [(2e-6, 3e-11), (5e-6, 6e-11), (2e-5, 2e-10), (8e-5, 9e-10)]


@dataclass
class _R:
    coupling_type: str = "AxionPhoton"
    data_points: list = field(default_factory=list)
    notes: str = ""
    extraction_confidence: float = 0.7
    arxiv_id: str = "0000.00000"


def _voted(monkeypatch, scripted):
    it = iter(scripted)
    monkeypatch.setattr(ex, "run_extraction_agent", lambda *a, **k: next(it))
    monkeypatch.setenv("AAL_READ_SAMPLES", str(len(scripted)))
    return ex.run_extraction_agent_voted(object(), Path("/nonexistent.pdf"), None)


REJECT_NOTE = (" | [VISION GATE A] is_projection=true but the vision notes say "
               "the traced curve is existing bounds — not the paper's own curve")


def test_majority_rejected_survivor_cannot_win(monkeypatch):
    # 1512.06165 pattern: 2 samples gate-rejected (fell back to a 4-pt text
    # read), the third emitted the same mistrace unconfessed with 20 points.
    r = _voted(monkeypatch, [
        _R(data_points=list(FALLBACK), notes="fallback" + REJECT_NOTE),
        _R(data_points=list(FALLBACK), notes="fallback" + REJECT_NOTE),
        _R(data_points=list(CURVE), notes="unconfessed mistrace, no gate note"),
    ])
    assert r.data_points == FALLBACK          # survivor's 20-pt curve excluded
    assert "gate-aware" in r.notes and "#666" in r.notes


def test_majority_rejected_no_fallback_emits_zero_points(monkeypatch):
    r = _voted(monkeypatch, [
        _R(data_points=[], notes="pool emptied" + REJECT_NOTE, extraction_confidence=0.7),
        _R(data_points=[], notes="pool emptied" + REJECT_NOTE, extraction_confidence=0.6),
        _R(data_points=list(CURVE), notes="unconfessed survivor"),
    ])
    assert r.data_points == []
    assert r.extraction_confidence <= 0.5
    assert "2/3" in r.notes and "#666" in r.notes


def test_minority_rejection_is_advisory(monkeypatch):
    # only 1/3 rejected: normal vote over all samples; the two agreeing full
    # curves outvote the fallback.
    r = _voted(monkeypatch, [
        _R(data_points=list(FALLBACK), notes="fallback" + REJECT_NOTE),
        _R(data_points=list(CURVE), notes="clean sample"),
        _R(data_points=list(CURVE), notes="clean sample"),
    ])
    assert r.data_points == CURVE
    assert "gate-aware" not in r.notes


def test_gate_d_demote_does_not_trigger_filter(monkeypatch):
    demote = " | [VISION GATE D] vision notes admit tracing a compilation envelope"
    r = _voted(monkeypatch, [
        _R(data_points=list(CURVE), notes="demoted" + demote),
        _R(data_points=list(CURVE), notes="demoted" + demote),
        _R(data_points=list(CURVE), notes="clean"),
    ])
    assert r.data_points == CURVE
    assert "gate-aware" not in r.notes


def test_no_gate_notes_unchanged(monkeypatch):
    r = _voted(monkeypatch, [
        _R(data_points=list(CURVE), notes="a"),
        _R(data_points=list(CURVE), notes="b"),
        _R(data_points=list(CURVE), notes="c"),
    ])
    assert r.data_points == CURVE
    assert "read-vote N=3" in r.notes
