"""Unit tests for the read-vote prompt-cache markers in pipeline/extractor.py.

No API calls: a stub client captures the request kwargs. Pins the gated caching
contract (the 1h cache only pays off when a prompt is re-read, i.e. votes 2..N
of a read-vote; on a single read it is a pure ~2x write penalty with zero reads,
so it is disabled — measured, see appendix on cost):

* on the SINGLE-READ path (default) NO block carries ``cache_control`` — the
  breakpoint is gated off so unique per-paper payloads are not written to cache
  and read back zero times;
* when caching IS enabled (the ``_cache_state.enabled`` flag the read-vote loop
  sets for N>1), every stage marks exactly one block with ``cache_control``
  (ttl 1h), the marked block and everything before it is VOTE-STABLE, and every
  per-vote-varying section (stage-2 axis_context, verify spot-check mass) sits
  strictly AFTER the marker (caching is a prefix match).

Run:
    pytest evaluation/tests/test_prompt_caching.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import extractor as ex


class _Resp:
    class _Block:
        text = '{"is_new_limit": false, "data_points": [], "found_exclusion_plot": false, "found_limit_plot": false, "extraction_confidence": 0.0}'
    content = [_Block()]


class _Paper:
    title = "A Test Paper"
    summary = "An abstract about axions." * 10


@pytest.fixture()
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_create(client, **kwargs):
        calls.append(kwargs)
        return _Resp()

    monkeypatch.setattr(ex, "_create", fake_create)
    monkeypatch.setattr(ex, "_call_with_retry", lambda f: f())
    return calls


@pytest.fixture()
def cache_on():
    """Enable the read-vote cache gate for the duration of a test."""
    prev = getattr(ex._cache_state, "enabled", False)
    ex._cache_state.enabled = True
    try:
        yield
    finally:
        ex._cache_state.enabled = prev


@pytest.fixture()
def figures(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"fig{i}.png"
        p.write_bytes(b"\x89PNG fake image bytes %d" % i)
        paths.append(p)
    return paths


def _marked_indices(content):
    return [i for i, b in enumerate(content) if "cache_control" in b]


# --- Single-read path: caching is OFF (the fix) -----------------------------

def test_single_read_has_no_cache_markers(captured, figures):
    """Default (N=1) path must not mark any block cacheable."""
    ex._run_stage1(_Paper(), "paper text " * 100, client=None)
    ex._run_stage2a_axes(_Paper(), figures, client=None)
    ex._run_stage2(_Paper(), figures, client=None,
                   coupling_hint="AxionPhoton",
                   axis_info={"found_exclusion_plot": True, "x_axis_min": 1e-6,
                              "x_axis_max": 1e-3, "y_axis_min": 1e-12, "y_axis_max": 1e-9})
    ex._run_vision_verify(_Paper(), figures, client=None,
                          stage2_data=[(1e-6, 1e-11)], coupling_type="AxionPhoton")
    assert getattr(ex._cache_state, "enabled", False) is False
    for kwargs in captured:
        content = kwargs["messages"][0]["content"]
        assert _marked_indices(content) == []


# --- Read-vote path (caching enabled): placement contract holds -------------

def test_stage1_caches_full_prompt(captured, cache_on):
    ex._run_stage1(_Paper(), "paper text " * 100, client=None)
    content = captured[0]["messages"][0]["content"]
    assert _marked_indices(content) == [len(content) - 1]
    assert content[-1]["cache_control"] == ex._CACHE_1H
    assert "paper text" in content[-1]["text"]


def test_stage2a_marks_last_image(captured, cache_on, figures):
    ex._run_stage2a_axes(_Paper(), figures, client=None)
    content = captured[0]["messages"][0]["content"]
    assert _marked_indices(content) == [len(content) - 1]
    assert content[-1]["type"] == "image"


def test_stage2_axis_context_after_marker(captured, cache_on, figures):
    axis_info = {"found_exclusion_plot": True, "x_axis_min": 1e-6, "x_axis_max": 1e-3,
                 "y_axis_min": 1e-12, "y_axis_max": 1e-9}
    ex._run_stage2(_Paper(), figures, client=None,
                   coupling_hint="AxionPhoton", axis_info=axis_info)
    content = captured[0]["messages"][0]["content"]
    (mark,) = _marked_indices(content)
    assert content[mark]["type"] == "image"
    # AXIS CALIBRATION (per-vote varying) must be strictly after the marker
    after = "".join(b.get("text", "") for b in content[mark + 1:])
    before = "".join(b.get("text", "") for b in content[:mark])
    assert "AXIS CALIBRATION" in after and "AXIS CALIBRATION" not in before
    # the vote-stable hint stays in the prefix
    assert "AxionPhoton" in before


def test_stage2_without_axis_info_has_no_trailing_text(captured, cache_on, figures):
    ex._run_stage2(_Paper(), figures, client=None)
    content = captured[0]["messages"][0]["content"]
    assert content[-1]["type"] == "image"
    assert _marked_indices(content) == [len(content) - 1]


def test_verify_spotcheck_mass_after_marker(captured, cache_on, figures):
    ex._run_vision_verify(_Paper(), figures, client=None,
                          stage2_data=[(1e-6, 3e-11), (2e-6, 4e-11), (5e-6, 9e-11)],
                          coupling_type="AxionPhoton")
    content = captured[0]["messages"][0]["content"]
    (mark,) = _marked_indices(content)
    assert content[mark]["type"] == "image"
    after = "".join(b.get("text", "") for b in content[mark + 1:])
    before = "".join(b.get("text", "") for b in content[:mark])
    assert "At mass" in after and "At mass" not in before
    assert "A Test Paper" in before  # stable title stays in the cached prefix


def test_exactly_one_marker_per_request(captured, cache_on, figures):
    ex._run_stage1(_Paper(), "text " * 50, client=None)
    ex._run_stage2a_axes(_Paper(), figures, client=None)
    ex._run_stage2(_Paper(), figures, client=None)
    ex._run_vision_verify(_Paper(), figures, client=None, stage2_data=[(1e-6, 1e-11)])
    for kwargs in captured:
        content = kwargs["messages"][0]["content"]
        assert len(_marked_indices(content)) == 1
