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


# ---------------------------------------------------------------------------
# Known-unconvertible pre-verdict (e*cm class) — 2026-07-14
# ---------------------------------------------------------------------------

from pipeline.convention_queue import (  # noqa: E402
    STATUS_UNCONVERTIBLE,
    UNDECLARED_TOKEN,
    known_unconvertible,
    undeclared_suspicious,
)


@pytest.mark.parametrize("decl", [
    "d_n in e*cm",
    "d_n oscillation amplitude in e*cm (limit on dn-(mu_n/mu_Hg)dHg)",
    "d_n in e*cm (oscillating deuteron EDM amplitude d_AC)",
    "oscillating neutron EDM amplitude d_n in e cm",
])
def test_ecm_class_is_known_unconvertible(decl):
    assert known_unconvertible("AxionEDM", decl)


def test_ecm_converted_declaration_exempt():
    # #594: "converted from ..." means the EMITTED values are canonical-claimed.
    assert not known_unconvertible(
        "AxionEDM", "g_d in GeV^-2, converted from d_n in e*cm")


def test_ecm_other_coupling_not_unconvertible():
    assert not known_unconvertible("AxionPhoton", "d_n in e*cm")


def test_ecm_entry_enters_as_unconvertible(qpath):
    e = append_flag(coupling_type="AxionEDM", declared_convention="d_n in e*cm",
                    arxiv_id="2101.01241", path=qpath)
    assert e["status"] == STATUS_UNCONVERTIBLE


def test_unknown_token_still_enters_queued(qpath):
    e = append_flag(coupling_type="AxionEDM",
                    declared_convention="C_G/f_a in GeV^-1 as plotted",
                    arxiv_id="2204.01454", path=qpath)
    assert e["status"] == STATUS_QUEUED


def test_unconvertible_entry_never_reopened(qpath):
    append_flag(coupling_type="AxionEDM", declared_convention="d_n in e*cm",
                arxiv_id="2101.01241", path=qpath)
    e = append_flag(coupling_type="AxionEDM", declared_convention="d_n in e*cm",
                    arxiv_id="2208.07293", path=qpath)
    assert e["status"] == STATUS_UNCONVERTIBLE
    assert e["count"] == 2 and set(e["arxiv_ids"]) == {"2101.01241", "2208.07293"}


# ---------------------------------------------------------------------------
# Blind-spot #1: undeclared + suspicious magnitude — 2026-07-14
# ---------------------------------------------------------------------------

_RANGES = {"ScalarNucleon": {"mass": (1e-24, 1e9), "coupling": (1e-20, 1e0)}}


def _pts(vals):
    return tuple((1e-10, v) for v in vals)


def test_undeclared_huge_values_suspicious():
    # d_e_large-class storage: ~1e30 vs ceiling 1e0 (+3 dex margin).
    assert undeclared_suspicious("ScalarNucleon", "", _pts([1e30, 1e31, 1e32]),
                                 _RANGES)


def test_undeclared_in_range_not_suspicious():
    assert not undeclared_suspicious("ScalarNucleon", "", _pts([1e-5, 1e-6]),
                                     _RANGES)


def test_margin_is_three_decades():
    # Just above ceiling but within margin: not suspicious (strong-limit noise,
    # unit sloppiness — the selector's range machinery owns that band).
    assert not undeclared_suspicious("ScalarNucleon", "", _pts([5e2]), _RANGES)
    assert undeclared_suspicious("ScalarNucleon", "", _pts([5e3]), _RANGES)


def test_real_declaration_bypasses_blind_spot_guard():
    # A populated declaration goes through convention_review_needed instead.
    assert not undeclared_suspicious("ScalarNucleon", "|d_mhat - d_g|",
                                     _pts([1e30]), _RANGES)


@pytest.mark.parametrize("decl", ["", None, "canonical", "standard", "n/a"])
def test_canonical_claim_variants_covered(decl):
    assert undeclared_suspicious("ScalarNucleon", decl, _pts([1e30]), _RANGES)


def test_small_values_never_suspicious():
    # High side only: a stronger-than-expected limit is not a convention signal.
    assert not undeclared_suspicious("ScalarNucleon", "", _pts([1e-40]), _RANGES)


def test_no_ranges_or_type_fail_open():
    assert not undeclared_suspicious(None, "", _pts([1e30]), _RANGES)
    assert not undeclared_suspicious("ScalarNucleon", "", _pts([1e30]), None)
    assert not undeclared_suspicious("Unknown", "", _pts([1e30]), _RANGES)


def test_undeclared_token_groups_per_type(qpath):
    e = append_flag(coupling_type="ScalarNucleon",
                    declared_convention=UNDECLARED_TOKEN,
                    arxiv_id="9999.00001", path=qpath)
    e2 = append_flag(coupling_type="ScalarNucleon",
                     declared_convention=UNDECLARED_TOKEN,
                     arxiv_id="9999.00002", path=qpath)
    assert e2["cache_key"] == e["cache_key"] and e2["count"] == 2
