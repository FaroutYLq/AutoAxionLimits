"""Preview then run must see the same work; no API or git operations."""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import backfill, convention_derivation, convention_queue, monitor
from pipeline import orchestrator, preprint_checker
from pipeline.extractor import FatalAPIError
from pipeline.run_context import preview_run


def state_file(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def wire_daily(monkeypatch, tmp_path):
    path = state_file(tmp_path, "processed.json", {"processed_ids": [], "failed_ids": {}})
    monkeypatch.setattr(orchestrator, "load_state", partial(monitor.load_state, path))
    monkeypatch.setattr(orchestrator, "save_state", lambda value: monitor.save_state(value, path))
    monkeypatch.setattr(orchestrator, "make_client", lambda: object())
    monkeypatch.setattr(orchestrator, "preflight_api_check", lambda client: None)
    monkeypatch.setattr(monitor, "_arxiv_id", lambda paper: paper.id)
    papers = [SimpleNamespace(id="2601.00001"), SimpleNamespace(id="2601.00002")]
    monkeypatch.setattr(orchestrator, "fetch_recent_papers", lambda **kwargs: papers)
    return path


def test_daily_preview_then_real_run_preserves_candidates_and_nested_state(monkeypatch, tmp_path):
    path = wire_daily(monkeypatch, tmp_path)
    queue = state_file(tmp_path, "queue.json", {"entries": []})
    cache = tmp_path / "cache.json"  # Nonexistent state must not be created either.
    before = path.read_bytes(), queue.read_bytes()
    seen = []

    def process(paper, paper_id, client, state, dry_run):
        seen.append((paper_id, dry_run))
        convention_queue.save_queue({"entries": [{"id": paper_id}]}, queue)
        convention_derivation._save_cache({"entries": {paper_id: {}}}, cache)

    monkeypatch.setattr(orchestrator, "_process_paper", process)
    orchestrator.main(dry_run=True)
    assert (path.read_bytes(), queue.read_bytes()) == before
    assert not cache.exists()
    orchestrator.main()
    assert seen == [("2601.00001", True), ("2601.00002", True),
                    ("2601.00001", False), ("2601.00002", False)]
    assert len(json.loads(path.read_text())["processed_ids"]) == 2
    assert cache.exists()


@pytest.mark.parametrize("dry_run", [True, False])
def test_daily_availability_failure_keeps_prior_progress_only_in_real_run(monkeypatch, tmp_path, dry_run):
    path = wire_daily(monkeypatch, tmp_path)
    before = path.read_bytes()

    def process(paper, *args):
        if paper.id.endswith("2"):
            raise FatalAPIError("subscription unavailable")

    monkeypatch.setattr(orchestrator, "_process_paper", process)
    with pytest.raises(SystemExit) as error:
        orchestrator.main(dry_run=dry_run)
    assert error.value.code == 2
    if dry_run:
        assert path.read_bytes() == before
    else:
        state = json.loads(path.read_text())
        assert state["processed_ids"] == ["2601.00001"]
        assert state["failed_ids"] == {}
    # A caught exit must not leave state suppression enabled for later work.
    monitor.save_state({"after": True}, path)
    assert json.loads(path.read_text()) == {"after": True}


def wire_backfill(monkeypatch, tmp_path):
    queue = [{"arxiv_id": "2601.00001"}, {"arxiv_id": "2601.00002"}]
    path = state_file(tmp_path, "backfill.json", {"queue": queue, "processed_ids": [],
                                                  "skipped_ids": {}, "runs": []})
    monkeypatch.setattr(backfill, "BACKFILL_STATE_PATH", path)
    monkeypatch.setattr(backfill, "make_client", lambda **kwargs: None)
    monkeypatch.setattr(backfill, "load_processed_state", lambda: {"processed_ids": []})
    return path, queue


def test_backfill_preview_then_resume_does_not_consume_queue(monkeypatch, tmp_path):
    path, queue = wire_backfill(monkeypatch, tmp_path)
    before = path.read_bytes()
    seen = []

    def process(candidate, client, state, processed, dry_run):
        seen.append((candidate["arxiv_id"], dry_run))
        if not dry_run:
            state["processed_ids"].append(candidate["arxiv_id"])
        return not dry_run

    monkeypatch.setattr(backfill, "_process_candidate", process)
    backfill.main(resume=True, dry_run=True, max_papers=1)
    assert path.read_bytes() == before
    backfill.main(resume=True, max_papers=1)
    assert seen == [(queue[0]["arxiv_id"], True), (queue[0]["arxiv_id"], False)]
    assert json.loads(path.read_text())["queue"] == queue[1:]


@pytest.mark.parametrize("dry_run", [True, False])
def test_backfill_failure_requeues_current_candidate(monkeypatch, tmp_path, dry_run):
    path, queue = wire_backfill(monkeypatch, tmp_path)
    before = path.read_bytes()

    def process(candidate, client, state, processed, preview):
        if candidate["arxiv_id"].endswith("2"):
            raise FatalAPIError("unavailable")
        state["processed_ids"].append(candidate["arxiv_id"])
        return True

    monkeypatch.setattr(backfill, "_process_candidate", process)
    with pytest.raises(SystemExit) as error:
        backfill.main(resume=True, dry_run=dry_run)
    assert error.value.code == 2
    if dry_run:
        assert path.read_bytes() == before
    else:
        state = json.loads(path.read_text())
        assert state["queue"] == queue[1:]
        assert state["processed_ids"] == [queue[0]["arxiv_id"]]


def test_discovery_saves_queue_unless_explicitly_previewed(monkeypatch, tmp_path):
    path, queue = wire_backfill(monkeypatch, tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(backfill, "discover_candidates", lambda *args: queue[:1])
    monkeypatch.setattr(backfill, "build_known_ids", lambda: set())
    monkeypatch.setattr(backfill, "filter_candidates", lambda candidates, known: candidates)
    backfill.main("2023-01-01", "2023-12-31", discover_only=True, dry_run=True)
    assert path.read_bytes() == before
    backfill.main("2023-01-01", "2023-12-31", discover_only=True)
    assert json.loads(path.read_text())["queue"] == queue[:1]


@pytest.mark.parametrize("mode", ["baseline", "withdrawn", "published_no_data", "changed"])
def test_weekly_preview_does_not_hide_next_real_update(monkeypatch, tmp_path, mode):
    filename = "limit_data/DarkPhoton/Test.txt"
    entry = {} if mode == "baseline" else {filename: {"known_version": 1, "published": False}}
    path = state_file(tmp_path, "versions.json", {"files": entry})
    before = path.read_bytes()
    save = preprint_checker.save_version_state
    monkeypatch.setattr(preprint_checker, "load_version_state", lambda: json.loads(path.read_text()))
    monkeypatch.setattr(preprint_checker, "save_version_state", lambda value: save(value, path))
    monkeypatch.setattr(preprint_checker, "make_client", lambda **kwargs: object())
    monkeypatch.setattr(preprint_checker, "scan_data_files_for_arxiv_ids", lambda root: {filename: "2601.00001"})
    monkeypatch.setattr(preprint_checker, "batch_check_published_semantic_scholar", lambda ids: {})
    monkeypatch.setattr(preprint_checker, "batch_get_latest_versions",
                        lambda ids: {"2601.00001": (2, mode == "published_no_data", object())})
    monkeypatch.setattr(preprint_checker, "is_withdrawn", lambda aid: mode == "withdrawn")
    monkeypatch.setattr(preprint_checker, "is_published", lambda paper: False)
    monkeypatch.setattr(preprint_checker, "download_pdf", lambda *args: Path("unused"))
    monkeypatch.setattr(preprint_checker, "run_extraction_agent", lambda *args: SimpleNamespace(
        data_points=[] if mode == "published_no_data" else [(1, 2)],
        is_projection=False, extraction_confidence=0.9))
    monkeypatch.setattr(preprint_checker, "apply_corrections", lambda result: ([(1, 2)], [], []))
    monkeypatch.setattr(preprint_checker, "data_has_changed", lambda *args: True)
    monkeypatch.setattr(preprint_checker, "summarise_changes", lambda *args: "changed")
    import pipeline.removal
    monkeypatch.setattr(pipeline.removal, "describe_limit_artifacts", lambda *args: [])
    prs = []
    monkeypatch.setattr(preprint_checker, "_create_removal_flag_pr", lambda **kwargs: prs.append("removal"))
    monkeypatch.setattr(preprint_checker, "_create_update_pr", lambda **kwargs: prs.append("update"))

    preprint_checker.run_weekly_check(tmp_path, True)  # Positional preview flag.
    assert path.read_bytes() == before
    assert prs == []
    preprint_checker.run_weekly_check(tmp_path)
    assert json.loads(path.read_text())["files"][filename]["known_version"] == 2
    assert prs == ([] if mode == "baseline" else ["update" if mode == "changed" else "removal"])


def test_weekly_init_preview_does_not_write(monkeypatch, tmp_path):
    path = state_file(tmp_path, "versions.json", {"files": {}})
    before = path.read_bytes()
    save = preprint_checker.save_version_state
    monkeypatch.setattr(preprint_checker, "make_client", lambda **kwargs: None)
    monkeypatch.setattr(preprint_checker, "load_version_state", lambda: {"files": {}})
    monkeypatch.setattr(preprint_checker, "save_version_state", lambda value: save(value, path))
    monkeypatch.setattr(preprint_checker, "scan_data_files_for_arxiv_ids", lambda root: {})
    monkeypatch.setattr(preprint_checker, "batch_check_published_semantic_scholar", lambda ids: {})
    monkeypatch.setattr(preprint_checker, "batch_get_latest_versions", lambda ids: {})
    preprint_checker.run_weekly_check(tmp_path, dry_run=True, init_only=True)
    assert path.read_bytes() == before


def test_nested_preview_scope_cannot_reenable_state_writes(tmp_path):
    path = tmp_path / "state.json"

    @preview_run
    def inner(dry_run=False):
        monitor.save_state({"written": True}, path)

    @preview_run
    def outer(dry_run=False):
        inner(dry_run=False)

    outer(True)
    assert not path.exists()
    inner()
    assert path.exists()
