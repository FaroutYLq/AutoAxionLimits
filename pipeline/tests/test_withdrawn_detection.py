"""Regression tests for withdrawn-paper detection in the weekly checker.

Motivating incident (arXiv:2607.19319 / QUALIPHIDE_FIR, 2026-08-05): the paper
behind a freshly merged limit was withdrawn from arXiv as v2, and the weekly
checker did not notice it at ANY of three independent points:

  1. The data file was new, so ``known_version is None`` sent it down the
     "first sight → set baseline, no PR" branch. That branch banked
     ``known_version = 2`` (already the withdrawal version), after which every
     later run short-circuits on ``latest_version <= known_version`` and the
     withdrawal is invisible forever.
  2. With the file tracked at v1, the version bump WAS detected, but a
     withdrawn version has no PDF: ``download_pdf`` got HTTP 404 and the
     generic ``except Exception: continue`` swallowed it. No PR, and the state
     was not advanced either, so it re-failed silently every week.
  3. Had extraction returned no data, the removal-flag PR was gated behind
     ``if published:``. A withdrawn paper has no journal_ref, no DOI and is not
     a Semantic Scholar JournalArticle, so it fell to a log-only branch.

Withdrawal is also invisible in arXiv's Atom API: the feed still carries the
ORIGINAL title and abstract, and the <arxiv:comment> is free text that need not
contain the word "withdrawn". Detection therefore reads arXiv's abs-page banner.

These tests are pure — no network, no Anthropic API.

Run:
    python -m pytest pipeline/tests/test_withdrawn_detection.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import arxiv
import pytest

from pipeline import preprint_checker as pc


# ---------------------------------------------------------------------------
# Fixtures — abs-page HTML shapes
# ---------------------------------------------------------------------------

# Verbatim shape of the real arXiv:2607.19319 abs page (2026-08-05).
_WITHDRAWN_HTML = """
<html><body>
  <span class="error" style="border: 2px solid grey">This paper has been withdrawn by Lanqing Yuan</span>
  <div class="submission-history">
  [v1] Tue, 21 Jul 2026 17:39:52 UTC (4,102 KB)<br/>
  [v2] Thu, 23 Jul 2026 01:36:47 UTC (1 KB) <em>(withdrawn)</em><br/>
  </div>
  <ul><li>Withdrawn</li></ul>
</body></html>
"""

# Banner reworded/removed but arXiv's structural markers still present.
_WITHDRAWN_HTML_NO_BANNER = """
<html><body>
  <div class="submission-history">
  [v2] Thu, 23 Jul 2026 01:36:47 UTC (1 KB) <em>(withdrawn)</em><br/>
  </div>
  <ul><li>Withdrawn</li></ul>
</body></html>
"""

_LIVE_HTML = """
<html><body>
  <blockquote class="abstract">Here we report results from QUALIPHIDE ...</blockquote>
  <div class="submission-history">
  [v1] Tue, 21 Jul 2026 17:39:52 UTC (4,102 KB)<br/>
  </div>
</body></html>
"""

# A live paper that merely *discusses* withdrawal in its abstract must not trip
# the detector: one stray match is not enough to retract a curated limit.
_LIVE_HTML_MENTIONS_WITHDRAWN = """
<html><body>
  <blockquote class="abstract">We revisit a claim that was later withdrawn by other groups.</blockquote>
</body></html>
"""


class _FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


def _patch_abs_page(monkeypatch, *, status=200, html="", raises=None):
    import httpx

    def fake_get(url, **kwargs):
        if raises is not None:
            raise raises
        return _FakeResponse(status, html)

    monkeypatch.setattr(httpx, "get", fake_get)


# ---------------------------------------------------------------------------
# is_withdrawn
# ---------------------------------------------------------------------------

def test_detects_withdrawal_banner(monkeypatch):
    _patch_abs_page(monkeypatch, html=_WITHDRAWN_HTML)
    assert pc.is_withdrawn("2607.19319") is True


def test_detects_withdrawal_without_banner_via_two_markers(monkeypatch):
    _patch_abs_page(monkeypatch, html=_WITHDRAWN_HTML_NO_BANNER)
    assert pc.is_withdrawn("2607.19319") is True


def test_live_paper_is_not_withdrawn(monkeypatch):
    _patch_abs_page(monkeypatch, html=_LIVE_HTML)
    assert pc.is_withdrawn("2209.03419") is False


def test_single_stray_mention_is_not_a_withdrawal(monkeypatch):
    _patch_abs_page(monkeypatch, html=_LIVE_HTML_MENTIONS_WITHDRAWN)
    assert pc.is_withdrawn("1234.56789") is False


def test_network_failure_returns_unknown_not_withdrawn(monkeypatch):
    """A transient arXiv outage must never mass-flag the corpus for removal."""
    _patch_abs_page(monkeypatch, raises=RuntimeError("connection reset"))
    assert pc.is_withdrawn("2607.19319") is None


def test_non_200_returns_unknown(monkeypatch):
    _patch_abs_page(monkeypatch, status=503, html="upstream error")
    assert pc.is_withdrawn("2607.19319") is None


# ---------------------------------------------------------------------------
# run_weekly_check wiring
# ---------------------------------------------------------------------------

_FILE = "limit_data/DarkPhoton/QUALIPHIDE_FIR.txt"
_ARXIV_ID = "2607.19319"


def _fake_paper(version: int = 2) -> arxiv.Result:
    return arxiv.Result(
        entry_id=f"http://arxiv.org/abs/{_ARXIV_ID}v{version}",
        title="Dark matter searches with a 13 meV threshold superconducting sensor array",
        summary="Many well-motivated dark matter models predict meV-scale energy deposits ...",
        comment="The authors identified a numerical problem in efficiency estimation",
    )


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Drive run_weekly_check with every external dependency stubbed."""
    calls = {"flag_prs": [], "update_prs": [], "extractions": 0}
    state_file = tmp_path / "preprint_versions.json"

    def load_state(path=None):
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {"schema_version": 1, "last_checked": None, "files": {}}

    def save_state(state, path=None):
        state_file.write_text(json.dumps(state, indent=2))

    monkeypatch.setattr(pc, "load_version_state", load_state)
    monkeypatch.setattr(pc, "save_version_state", save_state)
    monkeypatch.setattr(pc, "make_client", lambda **kw: object())
    monkeypatch.setattr(pc, "scan_data_files_for_arxiv_ids", lambda root=None: {_FILE: _ARXIV_ID})
    monkeypatch.setattr(pc, "batch_check_published_semantic_scholar", lambda ids, **kw: {})
    monkeypatch.setattr(
        pc, "batch_get_latest_versions", lambda ids, **kw: {_ARXIV_ID: (2, False, _fake_paper(2))}
    )

    def flag_pr(**kwargs):
        calls["flag_prs"].append(kwargs)

    def update_pr(**kwargs):
        calls["update_prs"].append(kwargs)

    def never_extract(*a, **kw):
        calls["extractions"] += 1
        raise AssertionError("extraction must not be attempted on a withdrawn paper")

    monkeypatch.setattr(pc, "_create_removal_flag_pr", flag_pr)
    monkeypatch.setattr(pc, "_create_update_pr", update_pr)
    monkeypatch.setattr(pc, "download_pdf", never_extract)
    monkeypatch.setattr(pc, "run_extraction_agent", never_extract)

    calls["state_file"] = state_file
    calls["seed"] = lambda entry: state_file.write_text(
        json.dumps({"schema_version": 1, "last_checked": None, "files": {_FILE: entry}}, indent=2)
    )
    calls["read"] = lambda: json.loads(state_file.read_text())["files"][_FILE]
    return calls


def test_first_sight_withdrawn_is_flagged_not_silently_baselined(monkeypatch, harness):
    """Bug 1: a newly merged file whose paper is already withdrawn.

    The old code hit the ``known_version is None`` baseline branch and recorded
    the withdrawal version with no PR, burying it permanently.
    """
    monkeypatch.setattr(pc, "is_withdrawn", lambda aid, **kw: True)
    pc.run_weekly_check(repo_root=Path("."), dry_run=False)

    assert len(harness["flag_prs"]) == 1, "withdrawal on first sight must open a flag PR"
    pr = harness["flag_prs"][0]
    assert pr["withdrawn"] is True
    assert pr["arxiv_id"] == _ARXIV_ID
    assert harness["extractions"] == 0

    entry = harness["read"]()
    assert entry["withdrawn"] is True
    assert entry["known_version"] == 2


def test_tracked_paper_withdrawn_flags_without_extracting(monkeypatch, harness):
    """Bugs 2+3: version bump to a withdrawn version, paper NOT published.

    Old behaviour: download_pdf 404 → generic except → no PR. And even the
    no-data branch would have been skipped because it was gated on ``published``.
    """
    harness["seed"]({
        "arxiv_id": _ARXIV_ID, "known_version": 1,
        "last_checked": "2026-07-22T00:00:00+00:00", "published": False,
    })
    monkeypatch.setattr(pc, "is_withdrawn", lambda aid, **kw: True)
    pc.run_weekly_check(repo_root=Path("."), dry_run=False)

    assert len(harness["flag_prs"]) == 1
    pr = harness["flag_prs"][0]
    assert pr["withdrawn"] is True
    assert pr["old_version"] == 1 and pr["new_version"] == 2
    # The flag must not depend on the paper being published.
    assert harness["extractions"] == 0


def test_dry_run_opens_no_pr(monkeypatch, harness):
    monkeypatch.setattr(pc, "is_withdrawn", lambda aid, **kw: True)
    pc.run_weekly_check(repo_root=Path("."), dry_run=True)
    assert harness["flag_prs"] == []
    assert harness["read"]()["withdrawn"] is True


def test_already_flagged_does_not_reopen_pr(monkeypatch, harness):
    """Second weekly run over an already-flagged file must stay quiet."""
    harness["seed"]({
        "arxiv_id": _ARXIV_ID, "known_version": 2,
        "last_checked": "2026-08-05T00:00:00+00:00", "published": False, "withdrawn": True,
    })
    called = {"n": 0}

    def counting_check(aid, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(pc, "is_withdrawn", counting_check)
    pc.run_weekly_check(repo_root=Path("."), dry_run=False)

    assert harness["flag_prs"] == []
    assert called["n"] == 0, "no version movement → don't even re-fetch the abs page"


def test_unknown_withdrawal_status_does_not_flag(monkeypatch, harness):
    """is_withdrawn() == None must fall through to the normal path, not flag."""
    monkeypatch.setattr(pc, "is_withdrawn", lambda aid, **kw: None)
    pc.run_weekly_check(repo_root=Path("."), dry_run=False)

    assert harness["flag_prs"] == []
    entry = harness["read"]()
    assert "withdrawn" not in entry
    assert entry["known_version"] == 2


def test_live_paper_first_sight_still_baselines_silently(monkeypatch, harness):
    """The pre-existing happy path must be unchanged."""
    monkeypatch.setattr(pc, "is_withdrawn", lambda aid, **kw: False)
    pc.run_weekly_check(repo_root=Path("."), dry_run=False)

    assert harness["flag_prs"] == [] and harness["update_prs"] == []
    entry = harness["read"]()
    assert entry["known_version"] == 2 and "withdrawn" not in entry
