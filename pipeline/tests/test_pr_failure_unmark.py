"""A paper marked processed for the PR commit must not stay marked when no PR exists.

Both the daily orchestrator and the backfill mark the paper processed *before*
committing (so the state file rides in the science PR). If the push/PR step
fails, or the run is interrupted in that window, the mark would otherwise
retire a paper nobody reviewed. Runner state publication must never carry it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import backfill, monitor, orchestrator


def _extraction():
    return SimpleNamespace(is_new_limit=True, is_projection=False, data_points=[(1.0, 2.0)],
                           coupling_type="DarkPhoton", extraction_confidence=0.9, data_source="text")


def _review():
    return SimpleNamespace(experiment_name="Fixture", extraction_confidence=0.9,
                           data_file_path="limit_data/DarkPhoton/Fixture.txt",
                           plotfuncs_file="PlotFuncs.py", plotfuncs_class="DarkPhoton",
                           notebook_path="DarkPhoton.ipynb", docs_file="docs/dp.md",
                           notebook_call="x", corrections_applied=[], corrections_flagged=[],
                           low_confidence=False, is_projection=False, paper_title="Fixture paper",
                           arxiv_url="https://arxiv.org/abs/2601.00002")


def _stub_extraction_chain(monkeypatch, module):
    monkeypatch.setattr(module, "classify_coupling_type", lambda paper: "DarkPhoton")
    monkeypatch.setattr(module, "download_pdf", lambda *args: Path("unused.pdf"))
    monkeypatch.setattr(module, "run_extraction_agent", lambda *args: _extraction())
    monkeypatch.setattr(module, "run_reviewer_agent", lambda *args: _review())
    monkeypatch.setattr(module, "write_repo_files", lambda *args: None)
    monkeypatch.setattr(module, "execute_notebook", lambda *args: (True, ""))
    monkeypatch.setattr(module, "execute_notebook_highlighted", lambda *args, **kw: (True, "", []))
    monkeypatch.setattr(module, "get_notebook_plot_names", lambda *args: [])
    monkeypatch.setattr(module, "create_feature_branch", lambda *args: "pipeline/arxiv-fixture")


@pytest.mark.parametrize("failure", [RuntimeError("push rejected"), KeyboardInterrupt()])
def test_daily_withdraws_processed_mark_when_no_pr_is_created(monkeypatch, tmp_path, failure):
    _stub_extraction_chain(monkeypatch, orchestrator)
    checkouts = []
    monkeypatch.setattr(orchestrator, "checkout_branch", lambda branch, root: checkouts.append(branch))
    monkeypatch.setattr(orchestrator, "stage_and_commit_files", lambda *args: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(orchestrator, "create_pull_request", lambda *a, **k: pytest.fail("unreachable"))
    path = tmp_path / "processed.json"
    monkeypatch.setattr(orchestrator, "save_state", lambda value: monitor.save_state(value, path))
    state = {"processed_ids": [], "failed_ids": {}}
    with pytest.raises(type(failure)):
        orchestrator._process_paper(SimpleNamespace(id="2601.00001"), "2601.00001", object(), state, False)
    assert checkouts == ["master"]
    assert state["processed_ids"] == []
    assert json.loads(path.read_text())["processed_ids"] == []


def test_daily_keeps_processed_mark_when_pr_is_created(monkeypatch, tmp_path):
    _stub_extraction_chain(monkeypatch, orchestrator)
    monkeypatch.setattr(orchestrator, "checkout_branch", lambda branch, root: None)
    monkeypatch.setattr(orchestrator, "stage_and_commit_files", lambda *args: None)
    monkeypatch.setattr(orchestrator, "create_pull_request",
                        lambda *a, **k: "https://github.com/example/fixture/pull/1")
    path = tmp_path / "processed.json"
    monkeypatch.setattr(orchestrator, "save_state", lambda value: monitor.save_state(value, path))
    state = {"processed_ids": [], "failed_ids": {}}
    orchestrator._process_paper(SimpleNamespace(id="2601.00001"), "2601.00001", object(), state, False)
    assert state["processed_ids"] == ["2601.00001"]
    assert json.loads(path.read_text())["processed_ids"] == ["2601.00001"]


@pytest.mark.parametrize("failure", [RuntimeError("push rejected"), KeyboardInterrupt()])
def test_backfill_withdraws_both_marks_when_no_pr_is_created(monkeypatch, tmp_path, failure):
    _stub_extraction_chain(monkeypatch, backfill)
    monkeypatch.setattr(backfill, "fetch_paper_by_id", lambda arxiv_id: SimpleNamespace(id=arxiv_id))
    monkeypatch.setattr(backfill, "checkout_branch", lambda branch, root: None)
    monkeypatch.setattr(backfill, "stage_and_commit_files", lambda *args: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(backfill, "create_pull_request_preprint", lambda *a, **k: pytest.fail("unreachable"))
    backfill_path = tmp_path / "backfill.json"
    processed_path = tmp_path / "processed.json"
    monkeypatch.setattr(backfill, "save_backfill_state", lambda value: monitor.save_state(value, backfill_path))
    monkeypatch.setattr(backfill, "save_processed_state", lambda value: monitor.save_state(value, processed_path))
    backfill_state = {"queue": [], "processed_ids": [], "skipped_ids": {}, "runs": []}
    processed_state = {"processed_ids": [], "failed_ids": {}}
    candidate = {"arxiv_id": "2601.00002", "citations": 5}
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            backfill._process_candidate(candidate, object(), backfill_state, processed_state, False)
        assert backfill_state["skipped_ids"] == {}
    else:
        assert backfill._process_candidate(candidate, object(), backfill_state, processed_state, False) is False
        assert backfill_state["skipped_ids"] == {"2601.00002": "pr_error: push rejected"}
    assert backfill_state["processed_ids"] == []
    assert processed_state["processed_ids"] == []
    assert json.loads(backfill_path.read_text())["processed_ids"] == []
    assert json.loads(processed_path.read_text())["processed_ids"] == []
