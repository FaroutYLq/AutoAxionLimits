"""A paper whose science PR was never created must stay eligible for retry.

Both the daily orchestrator and the backfill mark the paper processed *before*
committing (so the state file rides in the science PR). If the push/PR step
fails, or the run is interrupted in that window, the paper has no PR: it must
end up neither processed nor failed/skipped (daily: `filter_new_papers` still
returns it; backfill: the candidate is back at the head of the saved queue),
and a publication failure aborts the run like an availability failure (#648).
These tests drive the real entrypoints with extraction and git stubbed.
"""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import backfill, monitor, orchestrator
from pipeline.pr_creator import PublicationError


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


def _raise(error):
    def raiser(*args, **kwargs):
        raise error
    return raiser


def _stub_extraction_chain(monkeypatch, module, publish=None):
    monkeypatch.setattr(module, "classify_coupling_type", lambda paper: "DarkPhoton")
    monkeypatch.setattr(module, "download_pdf", lambda *args: Path("unused.pdf"))
    monkeypatch.setattr(module, "run_extraction_agent", lambda *args: _extraction())
    monkeypatch.setattr(module, "run_reviewer_agent", lambda *args: _review())
    monkeypatch.setattr(module, "write_repo_files", lambda *args: None)
    monkeypatch.setattr(module, "execute_notebook", lambda *args: (True, ""))
    monkeypatch.setattr(module, "execute_notebook_highlighted", lambda *args, **kw: (True, "", []))
    monkeypatch.setattr(module, "get_notebook_plot_names", lambda *args: [])
    monkeypatch.setattr(module, "create_feature_branch", lambda *args: "pipeline/arxiv-fixture")
    monkeypatch.setattr(module, "checkout_branch", lambda branch, root: None)
    monkeypatch.setattr(module, "stage_and_commit_files", publish or (lambda *args: None))


def wire_daily(monkeypatch, tmp_path, publish):
    _stub_extraction_chain(monkeypatch, orchestrator, publish)
    path = tmp_path / "processed.json"
    path.write_text(json.dumps({"processed_ids": [], "failed_ids": {}}))
    monkeypatch.setattr(orchestrator, "load_state", partial(monitor.load_state, path))
    monkeypatch.setattr(orchestrator, "save_state", lambda value: monitor.save_state(value, path))
    monkeypatch.setattr(orchestrator, "make_client", lambda: object())
    monkeypatch.setattr(orchestrator, "preflight_api_check", lambda client: None)
    monkeypatch.setattr(monitor, "_arxiv_id", lambda paper: paper.id)
    papers = [SimpleNamespace(id="2601.00001"), SimpleNamespace(id="2601.00002")]
    monkeypatch.setattr(orchestrator, "fetch_recent_papers", lambda **kwargs: papers)
    monkeypatch.setattr(orchestrator, "create_pull_request",
                        lambda *a, **k: "https://github.com/example/fixture/pull/1")
    return path, papers


def test_daily_push_failure_aborts_and_leaves_paper_retryable(monkeypatch, tmp_path):
    path, papers = wire_daily(monkeypatch, tmp_path, _raise(RuntimeError("push rejected")))
    with pytest.raises(SystemExit) as error:
        orchestrator.main()
    assert error.value.code == orchestrator.EXIT_PUBLICATION_FAILED
    state = json.loads(path.read_text())
    assert state["processed_ids"] == [] and state["failed_ids"] == {}
    assert [p.id for p in monitor.filter_new_papers(papers, state)] == ["2601.00001", "2601.00002"]


def test_daily_interrupt_leaves_paper_retryable(monkeypatch, tmp_path):
    path, papers = wire_daily(monkeypatch, tmp_path, _raise(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        orchestrator.main()
    state = json.loads(path.read_text())
    assert state["processed_ids"] == [] and state["failed_ids"] == {}
    assert len(monitor.filter_new_papers(papers, state)) == 2


def test_daily_success_marks_processed_and_continues(monkeypatch, tmp_path):
    path, papers = wire_daily(monkeypatch, tmp_path, None)
    orchestrator.main()
    state = json.loads(path.read_text())
    assert state["processed_ids"] == ["2601.00001", "2601.00002"]
    assert monitor.filter_new_papers(papers, state) == []


def test_daily_extraction_error_is_still_marked_failed(monkeypatch, tmp_path):
    # Only publication failures are run-level; a paper-specific crash keeps the
    # existing mark_failed behaviour so it is not re-extracted every day.
    path, papers = wire_daily(monkeypatch, tmp_path, None)
    monkeypatch.setattr(orchestrator, "run_extraction_agent", _raise(ValueError("bad pdf")))
    orchestrator.main()
    state = json.loads(path.read_text())
    assert set(state["failed_ids"]) == {"2601.00001", "2601.00002"}


def wire_backfill(monkeypatch, tmp_path, publish):
    _stub_extraction_chain(monkeypatch, backfill, publish)
    queue = [{"arxiv_id": "2601.00002", "citations": 5}, {"arxiv_id": "2601.00003", "citations": 4}]
    path = tmp_path / "backfill.json"
    path.write_text(json.dumps({"queue": queue, "processed_ids": [], "skipped_ids": {}, "runs": []}))
    processed = tmp_path / "processed.json"
    processed.write_text(json.dumps({"processed_ids": [], "failed_ids": {}}))
    monkeypatch.setattr(backfill, "load_backfill_state", lambda: json.loads(path.read_text()))
    monkeypatch.setattr(backfill, "save_backfill_state", lambda value: monitor.save_state(value, path))
    monkeypatch.setattr(backfill, "load_processed_state", lambda: json.loads(processed.read_text()))
    monkeypatch.setattr(backfill, "save_processed_state", lambda value: monitor.save_state(value, processed))
    monkeypatch.setattr(backfill, "make_client", lambda **kwargs: object())
    monkeypatch.setattr(backfill, "fetch_paper_by_id", lambda arxiv_id: SimpleNamespace(id=arxiv_id))
    monkeypatch.setattr(backfill, "create_pull_request_preprint",
                        lambda *a, **k: "https://github.com/example/fixture/pull/2")
    return path, processed, queue


def test_backfill_push_failure_requeues_candidate_and_aborts(monkeypatch, tmp_path):
    path, processed, queue = wire_backfill(monkeypatch, tmp_path, _raise(RuntimeError("push rejected")))
    with pytest.raises(SystemExit) as error:
        backfill.main(resume=True, max_papers=2)
    assert error.value.code == backfill.EXIT_PUBLICATION_FAILED
    state = json.loads(path.read_text())
    assert state["queue"] == queue  # head candidate restored, nothing consumed
    assert state["processed_ids"] == [] and state["skipped_ids"] == {}
    assert json.loads(processed.read_text())["processed_ids"] == []


def test_backfill_interrupt_keeps_candidate_in_saved_queue(monkeypatch, tmp_path):
    path, processed, queue = wire_backfill(monkeypatch, tmp_path, _raise(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        backfill.main(resume=True, max_papers=2)
    state = json.loads(path.read_text())
    assert state["queue"] == queue
    assert state["processed_ids"] == [] and state["skipped_ids"] == {}
    assert json.loads(processed.read_text())["processed_ids"] == []


def test_backfill_success_consumes_queue_and_marks_both_states(monkeypatch, tmp_path):
    path, processed, queue = wire_backfill(monkeypatch, tmp_path, None)
    assert backfill.main(resume=True, max_papers=1) == 1
    state = json.loads(path.read_text())
    assert state["queue"] == queue[1:]
    assert state["processed_ids"] == ["2601.00002"]
    assert json.loads(processed.read_text())["processed_ids"] == ["2601.00002"]


def test_worker_raises_publication_error_after_withdrawing_marks(monkeypatch, tmp_path):
    _stub_extraction_chain(monkeypatch, orchestrator, _raise(RuntimeError("push rejected")))
    monkeypatch.setattr(orchestrator, "save_state", lambda value: None)
    state = {"processed_ids": [], "failed_ids": {}}
    with pytest.raises(PublicationError, match="push rejected"):
        orchestrator._process_paper(SimpleNamespace(id="2601.00001"), "2601.00001", object(), state, False)
    assert state["processed_ids"] == []
