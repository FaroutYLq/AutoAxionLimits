"""Pins for the two 2026-08 incident fixes.

1. Metadata-cache robustness (2026-08-05): a corrupt cache file must degrade
   to an empty cache (with the corrupt file preserved), never raise into
   every extraction; writes are atomic and survive concurrent read-modify-
   write from thread-pool workers.
2. Snapshot triage (2026-07-30 / 2026-08-05): husk vs error vs good
   classification, and environmental-error detection (#648: an availability
   error is a property of the run, not the paper).
"""
import json
import threading
from pathlib import Path

import pytest

from evaluation.benchmark.snapshot_triage import (
    classify_snapshot,
    is_environmental_error,
)
from evaluation.evaluate import _load_metadata_cache, _save_metadata_cache


# ---------------------------------------------------------------- triage

def test_good_snapshot():
    snap = {"arxiv_id": "1234.5678", "data_source": "text", "num_points": 2,
            "data_points": [[1e-6, 1e-10], [1e-5, 1e-10]]}
    assert classify_snapshot(snap) == "good"


def test_sparse_but_real_read_is_good():
    # a one-point stated bound is legitimately sparse, not a husk
    snap = {"arxiv_id": "1234.5678", "data_source": "text", "num_points": 1,
            "data_points": [[1e-6, 1e-10]]}
    assert classify_snapshot(snap) == "good"


def test_husk_is_empty_with_no_error():
    snap = {"arxiv_id": "1234.5678", "data_source": "none", "num_points": 0,
            "data_points": [], "is_new_limit": True}
    assert classify_snapshot(snap) == "husk"


def test_error_stub_is_error_not_husk():
    # the 2026-08-05 trap: an error stub is empty AND carries an error;
    # it must classify as error so triage loops don't purge-retry it blindly
    snap = {"arxiv_id": "1234.5678", "status": "error",
            "error": "Expecting ',' delimiter: line 388 column 18"}
    assert classify_snapshot(snap) == "error"


def test_populated_snapshot_with_error_field_is_error():
    snap = {"arxiv_id": "1234.5678", "data_source": "text", "num_points": 3,
            "data_points": [[1, 1]] * 3, "error": "partial failure"}
    assert classify_snapshot(snap) == "error"


@pytest.mark.parametrize("msg", [
    "claude subscription usage limit reached: You've hit your session limit",
    "rate limit exceeded, retry later",
    "Overloaded",
    "HTTP 529 from upstream",
    "Your credit balance is too low",
    "billing hard limit reached",
])
def test_environmental_errors_detected(msg):
    assert is_environmental_error(msg)


@pytest.mark.parametrize("msg", [
    None, "", "PDF download failed: 404",
    "Extraction failed: no figure found",
    "Expecting ',' delimiter: line 388 column 18",
])
def test_paper_level_errors_are_not_environmental(msg):
    assert not is_environmental_error(msg)


# ------------------------------------------------------------- cache guard

def test_corrupt_cache_degrades_to_empty_and_is_preserved(tmp_path):
    p = tmp_path / "metadata_cache.json"
    p.write_text('{"1234.5678": {"title": "T", "abstract":""broken"}}')
    assert _load_metadata_cache(p) == {}
    assert (tmp_path / "metadata_cache.json.corrupt").exists()


def test_non_dict_cache_degrades_to_empty(tmp_path):
    p = tmp_path / "metadata_cache.json"
    p.write_text('["not", "a", "dict"]')
    assert _load_metadata_cache(p) == {}


def test_missing_cache_is_empty(tmp_path):
    assert _load_metadata_cache(tmp_path / "nope.json") == {}


def test_save_is_atomic_and_merges(tmp_path):
    p = tmp_path / "metadata_cache.json"
    _save_metadata_cache(p, "1111.1111", "Title A", "Abs A")
    _save_metadata_cache(p, "2222.2222", "Title B", "Abs B")
    cache = json.loads(p.read_text())
    assert set(cache) == {"1111.1111", "2222.2222"}
    assert cache["1111.1111"]["title"] == "Title A"
    # no stray temp files left behind
    assert [f.name for f in tmp_path.iterdir()] == ["metadata_cache.json"]


def test_concurrent_saves_lose_no_entries(tmp_path):
    p = tmp_path / "metadata_cache.json"
    ids = [f"{i:04d}.{i:05d}" for i in range(32)]
    threads = [threading.Thread(
        target=_save_metadata_cache, args=(p, i, f"t{i}", f"a{i}"))
        for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cache = json.loads(p.read_text())
    assert set(cache) == set(ids)
