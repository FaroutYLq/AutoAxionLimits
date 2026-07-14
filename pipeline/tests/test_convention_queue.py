"""Unit tests for the convention-escalation queue (PR C).

Pins the production side of DESIGN_convention_escalation.md: a deterministic,
deduplicated JSON append per [CONVENTION REVIEW] firing, with an atomic write
and a lifecycle that never reopens an adjudicated token. No network/API.

Run:
    python -m pytest pipeline/tests/test_convention_queue.py -v
"""

from __future__ import annotations

import json

import pytest

from pipeline.convention_queue import (
    STATUS_NEEDS_HUMAN,
    STATUS_PROMOTED,
    STATUS_QUEUED,
    append_flag,
    cache_key,
    load_queue,
    normalize_declaration,
    record_convention_flag,
    save_queue,
)


@pytest.fixture
def qpath(tmp_path):
    return tmp_path / "convention_queue.json"


# ---------------------------------------------------------------------------
# Normalization / cache key
# ---------------------------------------------------------------------------

def test_normalize_declaration_stable():
    a = normalize_declaration("Decay rate  Gamma, in s^-1!")
    b = normalize_declaration("decay rate gamma in s 1")
    assert a == b == "decay rate gamma in s 1"


def test_normalize_empty():
    assert normalize_declaration(None) == ""
    assert normalize_declaration("   ") == ""


def test_cache_key_scoped_by_type():
    k1 = cache_key("AxionPhoton", "Gamma in s^-1")
    k2 = cache_key("AxionElectron", "Gamma in s^-1")
    assert k1 != k2
    assert k1 == cache_key("axionphoton", "gamma in s 1")  # case/punct-insensitive


# ---------------------------------------------------------------------------
# Append + dedup
# ---------------------------------------------------------------------------

def test_append_creates_entry(qpath):
    e = append_flag(coupling_type="AxionPhoton", declared_convention="Gamma s^-1",
                    arxiv_id="2311.05476", sample_points=[(4.1, 1.2e-27)],
                    path=qpath, now="2026-07-13T00:00:00Z")
    assert e["status"] == STATUS_QUEUED and e["count"] == 1
    q = load_queue(qpath)
    assert len(q["entries"]) == 1
    assert q["entries"][0]["sample_points"] == [[4.1, 1.2e-27]]


def test_dedup_bumps_counter_same_key(qpath):
    append_flag(coupling_type="AxionPhoton", declared_convention="Gamma in s^-1",
                arxiv_id="2311.05476", path=qpath, now="t1")
    # Same declaration, different case/whitespace/punctuation -> same token.
    append_flag(coupling_type="AxionPhoton", declared_convention="gamma  IN  s^-1 !!",
                arxiv_id="2401.00001", path=qpath, now="t2")
    q = load_queue(qpath)
    assert len(q["entries"]) == 1  # same normalized token -> one entry
    e = q["entries"][0]
    assert e["count"] == 2
    assert e["arxiv_ids"] == ["2311.05476", "2401.00001"]
    assert e["last_seen"] == "t2"


def test_distinct_tokens_distinct_entries(qpath):
    append_flag(coupling_type="AxionPhoton", declared_convention="Gamma s^-1",
                arxiv_id="a", path=qpath)
    append_flag(coupling_type="AxionEDM", declared_convention="C_G/(f_a m_a) GeV^-2",
                arxiv_id="b", path=qpath)
    assert len(load_queue(qpath)["entries"]) == 2


def test_same_paper_same_token_idempotent_ids(qpath):
    append_flag(coupling_type="AxionPhoton", declared_convention="Gamma s^-1",
                arxiv_id="dup", path=qpath)
    append_flag(coupling_type="AxionPhoton", declared_convention="Gamma s^-1",
                arxiv_id="dup", path=qpath)
    e = load_queue(qpath)["entries"][0]
    assert e["count"] == 2 and e["arxiv_ids"] == ["dup"]  # id not duplicated


def test_sample_points_capped_at_five(qpath):
    pts = [(float(i), 1e-20 * i) for i in range(1, 10)]
    e = append_flag(coupling_type="AxionPhoton", declared_convention="x",
                    arxiv_id="p", sample_points=pts, path=qpath)
    assert len(e["sample_points"]) == 5


# ---------------------------------------------------------------------------
# Lifecycle: adjudicated tokens are not reopened
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [STATUS_PROMOTED, STATUS_NEEDS_HUMAN])
def test_adjudicated_token_only_counter_bumps(qpath, status):
    append_flag(coupling_type="AxionPhoton", declared_convention="Gamma s^-1",
                arxiv_id="a", path=qpath)
    q = load_queue(qpath)
    q["entries"][0]["status"] = status
    save_queue(q, qpath)
    # A new firing of the same token must not flip the status back to queued.
    append_flag(coupling_type="AxionPhoton", declared_convention="Gamma s^-1",
                arxiv_id="b", path=qpath)
    e = load_queue(qpath)["entries"][0]
    assert e["status"] == status and e["count"] == 2


# ---------------------------------------------------------------------------
# IO robustness
# ---------------------------------------------------------------------------

def test_load_missing_returns_empty(qpath):
    q = load_queue(qpath)
    assert q == {"version": 1, "entries": []}


def test_load_corrupt_returns_empty(qpath):
    qpath.write_text("{ not json")
    assert load_queue(qpath)["entries"] == []


def test_atomic_write_leaves_no_tmp(qpath):
    append_flag(coupling_type="AxionPhoton", declared_convention="x",
                arxiv_id="a", path=qpath)
    assert not qpath.with_suffix(".tmp").exists()
    # File is valid JSON.
    json.loads(qpath.read_text())


def test_record_convention_flag_never_raises(qpath):
    # Bad sample points must be swallowed, not crash the extraction.
    record_convention_flag("AxionPhoton", "x", "a",
                           data_points=[("bad", None)], path=qpath)
    # Nothing appended because the append raised internally and was swallowed.
    assert load_queue(qpath)["entries"] == []


def test_record_convention_flag_happy_path(qpath):
    record_convention_flag("AxionPhoton", "Gamma s^-1", "2311.05476",
                           data_points=[(4.1, 1.2e-27), (12.0, 4.5e-26)], path=qpath)
    e = load_queue(qpath)["entries"][0]
    assert e["coupling_type"] == "AxionPhoton" and e["count"] == 1
    assert e["sample_points"] == [[4.1, 1.2e-27], [12.0, 4.5e-26]]


# ---------------------------------------------------------------------------
# Wiring: the extractor's [CONVENTION REVIEW] site records to the queue
# ---------------------------------------------------------------------------

def test_extractor_wires_record_convention_flag():
    # The extractor must call the queue recorder at its flag site. Guard against
    # an accidental import/call removal without running a full extraction.
    import pipeline.extractor as ex
    import pipeline.convention_queue as cq
    assert ex.record_convention_flag is cq.record_convention_flag
    src = __import__("inspect").getsource(ex.run_extraction_agent)
    assert "record_convention_flag(" in src, "flag site no longer records to queue"
