"""Exercise real clones, subprocesses and leased pushes using a local bare remote."""

import json
from pathlib import Path
import subprocess

import pytest

from pipeline import local_runner as runner


STUB = '''import json, os, sys
from pathlib import Path
# runpy's module specification identifies which entrypoint the runner selected.
name = {"orchestrator": "processed.json", "preprint_checker": "preprint_versions.json",
        "backfill": "backfill_state.json"}[__spec__.name.split(".")[-1]]
print("backend=" + os.environ["AAL_BACKEND"])
print("model=" + os.environ.get("EXTRACTOR_MODEL", "unset")
      + " samples=" + os.environ.get("AAL_READ_SAMPLES", "unset")
      + " timeout=" + os.environ.get("AAL_CLI_TIMEOUT", "unset"))
if "--queue" in sys.argv:
    Path(os.environ["AAL_CONVENTION_QUEUE"]).write_text('{"entries": ["local"]}')
if "--dry-run" not in sys.argv:
    Path("pipeline/state", name).write_text(json.dumps({"completed": ["paper-1"]}))
    print("PR created: https://github.com/example/fixture/pull/17")
    print("Reusing existing state PR https://github.com/example/fixture/pull/99")
if "--fail" in sys.argv:
    sys.exit(2)
'''
DAILY = runner.PIPELINE_STATE


def git(repo, *args):
    return runner.git(repo, *args).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "master")
    git(source, "config", "user.name", "Fixture Author")
    git(source, "config", "user.email", "fixture@example.invalid")
    pipeline = source / "pipeline"
    (pipeline / "state").mkdir(parents=True)
    (pipeline / "__init__.py").touch()
    for module, owned in runner.PIPELINES.values():
        (pipeline / (module.split(".")[-1] + ".py")).write_text(STUB)
        for entry in owned:
            (pipeline / "state" / entry.file).write_text('{"completed": []}')
    (pipeline / "state" / "convention_queue.json").write_text('{"entries": []}')
    (source / "science.txt").write_text("original science\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "fixture baseline")
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))
    git(source, "remote", "add", "origin", str(remote))
    git(source, "push", "origin", "master")
    # This is deliberately dirty and must remain byte-for-byte unchanged.
    (source / "science.txt").write_text("user's uncommitted science\n")
    return source, remote


def prepare(tmp_path, repository, pipeline="daily", ref=None):
    source, _ = repository
    directory = tmp_path / "run"
    directory.mkdir()
    manifest = runner.prepare(directory, pipeline, source, ref, "claude-cli")
    return directory, manifest


def push_state(source, branch, contents, message="remote state"):
    """Commit *contents* ({filename: text}) on *branch* in the source and push it."""
    git(source, "checkout", "-B", branch)
    for filename, text in contents.items():
        (source / "pipeline/state" / filename).write_text(text)
    git(source, "add", "pipeline/state")
    git(source, "commit", "--allow-empty", "-m", message)
    oid = git(source, "rev-parse", "HEAD")
    git(source, "push", "-f", "origin", branch)
    git(source, "checkout", "master")
    return oid


@pytest.mark.parametrize("pipeline", runner.PIPELINES)
def test_isolation_preview_real_run_and_reuse(tmp_path, repository, pipeline):
    directory, manifest = prepare(tmp_path, repository, pipeline)
    source, _ = repository
    before = runner.state_snapshot(source)
    assert manifest["ref_published"] is True
    assert git(directory / "checkout", "branch", "--show-current") == "master"
    assert (directory / "checkout/science.txt").read_text() == "original science\n"
    assert runner.execute(directory, manifest, ["--dry-run"]) == 0
    assert manifest["attempts"][-1]["changed_state"] == []
    assert manifest["attempts"][-1]["pr_urls"] == []
    reused = runner.prepare(directory, pipeline, source, None, "claude-cli")
    assert runner.execute(directory, reused, []) == 0
    filename = runner.owned_files(pipeline)[0].file
    assert reused["attempts"][-1]["changed_state"] == [filename]
    # Only PRs this attempt created, not every URL the log happens to mention.
    assert reused["attempts"][-1]["pr_urls"] == ["https://github.com/example/fixture/pull/17"]
    assert "backend=claude-cli" in (directory / "attempt-2.log").read_text()
    assert runner.state_snapshot(source) == before
    assert (source / "science.txt").read_text() == "user's uncommitted science\n"
    assert git(source, "branch", "--show-current") == "master"


def test_nonzero_exit_retains_progress_and_can_resume(tmp_path, repository):
    directory, manifest = prepare(tmp_path, repository)
    assert runner.execute(directory, manifest, ["--fail"]) == 2
    saved = json.loads((directory / "run.json").read_text())
    assert saved["attempts"][-1]["exit_code"] == 2
    assert saved["attempts"][-1]["status"] == "finished"
    after = json.loads((directory / "attempt-1-after.json").read_text())
    assert json.loads(after["processed.json"])["completed"] == ["paper-1"]
    runner.prepare(directory, "daily", repository[0], None, "claude-cli")
    assert (directory / "checkout").exists()


def test_inherited_queue_override_cannot_write_outside_clone(tmp_path, repository, monkeypatch):
    outside = tmp_path / "external-queue.json"
    outside.write_text("preserve external state")
    monkeypatch.setenv("AAL_CONVENTION_QUEUE", str(outside))
    directory, manifest = prepare(tmp_path, repository)
    assert runner.execute(directory, manifest, ["--queue"]) == 0
    assert outside.read_text() == "preserve external state"
    assert json.loads((directory / "checkout/pipeline/state/convention_queue.json").read_text())["entries"] == ["local"]


def test_explicit_env_credentials_are_redacted_in_manifest_and_output(tmp_path, repository, capsys):
    directory, manifest = prepare(tmp_path, repository)
    assert runner.execute(directory, manifest, ["--dry-run"],
                          env_overrides=["MY_API_KEY=hunter2", "EXTRACTOR_MODEL=m"]) == 0
    assert manifest["attempts"][-1]["env_overrides"] == {"MY_API_KEY": "<redacted>", "EXTRACTOR_MODEL": "m"}
    assert "hunter2" not in capsys.readouterr().out
    assert "hunter2" not in (directory / "run.json").read_text()


def test_benchmark_environment_is_scrubbed_and_explicit_env_recorded(tmp_path, repository, monkeypatch):
    # A benchmark shell leaves model/extraction overrides exported; a real run
    # must not inherit them silently, while operational knobs pass through.
    monkeypatch.setenv("EXTRACTOR_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("AAL_READ_SAMPLES", "3")
    monkeypatch.setenv("AAL_CLI_TIMEOUT", "5")
    directory, manifest = prepare(tmp_path, repository)
    assert runner.execute(directory, manifest, ["--dry-run"]) == 0
    log = (directory / "attempt-1.log").read_text()
    assert "model=unset samples=unset timeout=5" in log
    attempt = manifest["attempts"][-1]
    assert attempt["env_scrubbed"] == ["AAL_READ_SAMPLES", "EXTRACTOR_MODEL"]
    assert attempt["env_overrides"] == {}
    assert runner.execute(directory, manifest, ["--dry-run"],
                          env_overrides=["EXTRACTOR_MODEL=explicit-model"]) == 0
    assert "model=explicit-model samples=unset" in (directory / "attempt-2.log").read_text()
    assert manifest["attempts"][-1]["env_overrides"] == {"EXTRACTOR_MODEL": "explicit-model"}
    with pytest.raises(ValueError, match="NAME=VALUE"):
        runner.child_environment({}, directory / "checkout", "api", ["not-an-assignment"])


def test_restore_only_owned_file_from_remote_state_branch(tmp_path, repository):
    source, _ = repository
    # Create a remote state snapshot with an unrelated file that must not leak.
    oid = push_state(source, DAILY, {
        "processed.json": '{"completed": ["remote-paper"]}',
        "preprint_versions.json": '{"unrelated": true}',
        "convention_queue.json": '{"entries": ["stale-remote-flag"]}',
    })
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    assert manifest["leases"][DAILY] == oid
    assert json.loads((checkout / "pipeline/state/processed.json").read_text())["completed"] == ["remote-paper"]
    assert (checkout / "pipeline/state/preprint_versions.json").read_text() == '{"completed": []}'
    # The convention queue is never restored: a lagging branch copy must not
    # regress master's triage results when the state PR merges.
    assert (checkout / "pipeline/state/convention_queue.json").read_text() == '{"entries": []}'
    # Baseline is committed locally, so a science branch round trip retains it.
    git(checkout, "checkout", "-b", "fixture/science")
    git(checkout, "checkout", "master")
    assert git(checkout, "status", "--porcelain") == ""
    assert json.loads((checkout / "pipeline/state/processed.json").read_text())["completed"] == ["remote-paper"]


def test_branch_without_owned_file_falls_back_to_master_baseline(tmp_path, repository):
    source, _ = repository
    git(source, "checkout", "-B", DAILY)
    git(source, "rm", "-q", "pipeline/state/processed.json")
    git(source, "commit", "-m", "branch lacks the owned file")
    git(source, "push", "origin", DAILY)
    git(source, "checkout", "master")
    directory, manifest = prepare(tmp_path, repository)
    assert (directory / "checkout/pipeline/state/processed.json").read_text() == '{"completed": []}'


def test_backfill_restores_daily_processed_file_and_publishes_both_branches(tmp_path, repository, monkeypatch):
    source, remote = repository
    push_state(source, DAILY, {"processed.json": '{"completed": ["daily-paper"]}'})
    directory, manifest = prepare(tmp_path, repository, "backfill")
    checkout = directory / "checkout"
    # build_known_ids() reads the daily file: without this restore a paper the
    # daily digest already proposed would be re-discovered and re-proposed.
    assert json.loads((checkout / "pipeline/state/processed.json").read_text())["completed"] == ["daily-paper"]
    (checkout / "pipeline/state/backfill_state.json").write_text('{"completed": ["bf-paper"]}')
    (checkout / "pipeline/state/processed.json").write_text('{"completed": ["daily-paper", "bf-paper"]}')
    calls = fake_gh(monkeypatch)
    runner.publish_state(directory)
    assert json.loads(git(remote, "show", f"{runner.BACKFILL_STATE}:pipeline/state/backfill_state.json"))["completed"] == ["bf-paper"]
    assert json.loads(git(remote, "show", f"{DAILY}:pipeline/state/processed.json"))["completed"] == ["daily-paper", "bf-paper"]
    assert git(remote, "diff", "--name-only", "master", runner.BACKFILL_STATE) == "pipeline/state/backfill_state.json"
    assert sum(call[1:3] == ["pr", "create"] for call in calls) == 2
    saved = json.loads((directory / "run.json").read_text())
    assert set(saved["state_prs"]) == {DAILY, runner.BACKFILL_STATE}


def fake_gh(monkeypatch, existing=False, fail_create=False):
    original = runner.command
    calls = []

    def command(argv, cwd, **kwargs):
        if argv[0] != "gh":
            return original(argv, cwd, **kwargs)
        calls.append(argv)
        if argv[1:3] == ["pr", "list"]:
            body = '[{"url":"https://github.com/example/fixture/pull/21"}]' if existing else "[]"
        else:
            if fail_create:
                raise RuntimeError("simulated PR outage after successful push")
            assert Path(argv[argv.index("--body-file") + 1]).read_text()
            body = "https://github.com/example/fixture/pull/21\n"
        return subprocess.CompletedProcess(argv, 0, body, "")

    monkeypatch.setattr(runner, "command", command)
    return calls


@pytest.mark.parametrize("existing", [False, True])
def test_state_publication_excludes_science_and_unowned_state(tmp_path, repository, monkeypatch, existing):
    source, _ = repository
    # Execution may use a development revision; it must not become a state PR.
    (source / "development.txt").write_text("unmerged implementation\n")
    git(source, "add", "development.txt")
    git(source, "commit", "-m", "development revision")
    directory, manifest = prepare(tmp_path, repository, ref="HEAD")
    assert manifest["ref_published"] is False
    checkout = directory / "checkout"
    (checkout / "pipeline/state/processed.json").write_text('{"completed": ["paper-1"]}')
    (checkout / "pipeline/state/convention_queue.json").write_text('{"entries": ["new-flag"]}')
    (checkout / "pipeline/state/preprint_versions.json").write_text('{"not": "owned by daily"}')
    (checkout / "science.txt").write_text("do not publish this\n")
    git(checkout, "add", ".")  # Even staged science must not enter the state PR.
    calls = fake_gh(monkeypatch, existing=existing)
    runner.publish_state(directory)
    remote = repository[1]
    assert git(remote, "diff", "--name-only", "master", DAILY).split() == [
        "pipeline/state/convention_queue.json", "pipeline/state/processed.json"]
    assert git(remote, "show", f"{DAILY}:science.txt") == "original science"
    assert (checkout / "science.txt").read_text() == "do not publish this\n"
    assert sum(call[1:3] == ["pr", "create"] for call in calls) == (0 if existing else 1)
    runner.publish_state(directory)  # No-op after successful publication.
    assert len(calls) == (1 if existing else 2)


def test_concurrent_remote_update_is_merged_not_rejected(tmp_path, repository, monkeypatch):
    """CI force-pushes the state branch daily; a local run set up before that
    must still publish, by merging onto the fresh tip rather than failing the
    lease forever (the #547 duplicate-PR mode)."""
    source, remote = repository
    push_state(source, DAILY, {"processed.json": '{"completed": ["old"], "failed": {}}'})
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    (checkout / "pipeline/state/processed.json").write_text(
        '{"completed": ["old", "local-paper"], "failed": {"x": "local error"}}')
    concurrent = push_state(source, DAILY, {
        "processed.json": '{"completed": ["old", "ci-paper"], "failed": {}, "last_run": "ci"}',
        "derived_conventions.json": '{"kept": "by ci"}',
    }, message="concurrent CI run")
    fake_gh(monkeypatch)
    runner.publish_state(directory)
    tip = git(remote, "rev-parse", DAILY)
    assert git(remote, "rev-parse", f"{tip}^") == concurrent  # linear on top of CI's commit
    published = json.loads(git(remote, "show", f"{tip}:pipeline/state/processed.json"))
    assert published == {"completed": ["old", "ci-paper", "local-paper"],
                         "failed": {"x": "local error"}, "last_run": "ci"}
    assert git(remote, "show", f"{tip}:pipeline/state/derived_conventions.json") == '{"kept": "by ci"}'
    # The merged result is now the local baseline, in the tree and committed.
    assert json.loads((checkout / "pipeline/state/processed.json").read_text()) == published
    assert git(checkout, "status", "--porcelain", "pipeline/state") == ""
    saved = json.loads((directory / "run.json").read_text())
    assert saved["leases"][DAILY] == tip
    assert json.loads(saved["baseline_state"]["processed.json"]) == published


def test_irreconcilable_remote_change_stops_publication_without_pushing(tmp_path, repository, monkeypatch):
    source, remote = repository
    push_state(source, runner.PREPRINT_STATE, {"preprint_versions.json": '{"files": {"f": {"note": "base"}}}'})
    directory, manifest = prepare(tmp_path, repository, "weekly")
    checkout = directory / "checkout"
    (checkout / "pipeline/state/preprint_versions.json").write_text('{"files": {"f": {"note": "ours"}}}')
    concurrent = push_state(source, runner.PREPRINT_STATE,
                            {"preprint_versions.json": '{"files": {"f": {"note": "theirs"}}}'})
    calls = fake_gh(monkeypatch)
    with pytest.raises(RuntimeError, match="cannot be reconciled.*files/f/note"):
        runner.publish_state(directory)
    assert git(remote, "rev-parse", runner.PREPRINT_STATE) == concurrent
    assert calls == []
    assert json.loads((checkout / "pipeline/state/preprint_versions.json").read_text()) == {"files": {"f": {"note": "ours"}}}


def test_lagging_branch_copy_never_regresses_convention_status(tmp_path, repository, monkeypatch):
    """The branch copy of the queue can lag master (triage merged in between);
    publishing must not roll a promoted convention back to queued."""
    source, remote = repository
    push_state(source, DAILY, {"convention_queue.json":
                               '{"entries": [{"cache_key": "k", "status": "queued", "count": 3}]}'})
    # Master (hence the clone) already carries the promotion.
    (source / "pipeline/state/convention_queue.json").write_text(
        '{"entries": [{"cache_key": "k", "status": "promoted", "count": 1}]}')
    git(source, "add", "pipeline/state/convention_queue.json")
    git(source, "commit", "-m", "triage promoted k")
    git(source, "push", "origin", "master")
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    (checkout / "pipeline/state/convention_queue.json").write_text(
        '{"entries": [{"cache_key": "k", "status": "promoted", "count": 1}, {"cache_key": "new", "status": "queued", "count": 1}]}')
    fake_gh(monkeypatch)
    runner.publish_state(directory)
    published = json.loads(git(remote, "show", f"{DAILY}:pipeline/state/convention_queue.json"))
    assert published["entries"] == [{"cache_key": "k", "status": "promoted", "count": 3},
                                    {"cache_key": "new", "status": "queued", "count": 1}]


def test_tip_moving_during_publication_is_rejected_and_local_state_kept(tmp_path, repository, monkeypatch):
    source, remote = repository
    stale = push_state(source, DAILY, {"processed.json": '{"completed": ["old"]}'})
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    content = '{"completed": ["old", "local-paper"]}'
    (checkout / "pipeline/state/processed.json").write_text(content)
    push_state(source, DAILY, {"processed.json": '{"completed": ["old", "racer"]}'})
    remote_before = git(remote, "rev-parse", DAILY)
    monkeypatch.setattr(runner, "remote_tips", lambda checkout, branches: {DAILY: stale})
    calls = fake_gh(monkeypatch)
    with pytest.raises(RuntimeError, match="moved"):
        runner.publish_state(directory)
    assert git(remote, "rev-parse", DAILY) == remote_before
    assert (checkout / "pipeline/state/processed.json").read_text() == content
    assert calls == []


def test_publish_refuses_checkout_left_on_science_branch(tmp_path, repository, monkeypatch):
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    git(checkout, "checkout", "-b", "pipeline/arxiv-interrupted")
    (checkout / "pipeline/state/processed.json").write_text('{"completed": ["no-pr-paper"]}')
    calls = fake_gh(monkeypatch)
    with pytest.raises(RuntimeError, match="not on master"):
        runner.publish_state(directory)
    assert calls == []
    assert not git(repository[1], "branch", "--list", DAILY)


def test_pr_outage_after_push_can_be_retried(tmp_path, repository, monkeypatch):
    directory, manifest = prepare(tmp_path, repository)
    (directory / "checkout/pipeline/state/processed.json").write_text('{"completed": ["paper-1"]}')
    fake_gh(monkeypatch, fail_create=True)
    with pytest.raises(RuntimeError, match="simulated PR outage"):
        runner.publish_state(directory)
    saved = json.loads((directory / "run.json").read_text())
    assert saved["leases"][DAILY] == git(repository[1], "rev-parse", DAILY)
    fake_gh(monkeypatch, existing=True)
    runner.publish_state(directory)
    assert json.loads((directory / "run.json").read_text())["state_prs"][DAILY].endswith("/21")


def test_unmerged_ref_requires_preview_or_explicit_opt_in(tmp_path, repository):
    source, _ = repository
    (source / "development.txt").write_text("unmerged implementation\n")
    git(source, "add", "development.txt")
    git(source, "commit", "-m", "development revision")
    directory, manifest = prepare(tmp_path, repository, ref="HEAD")
    assert manifest["ref_published"] is False
    with pytest.raises(RuntimeError, match="not on remote master"):
        runner.execute(directory, manifest, [])
    assert manifest["attempts"] == []
    assert runner.execute(directory, manifest, ["--dry-run"]) == 0
    assert runner.execute(directory, manifest, [], allow_unmerged_ref=True) == 0
    # An older production revision is still published: allowed.
    git(source, "reset", "-q", "--hard", "origin/master")
    older = tmp_path / "older"
    older.mkdir()
    assert runner.prepare(older, "daily", source, "origin/master~0", "claude-cli")["ref_published"] is True


def test_remote_failure_is_not_treated_as_missing_state(tmp_path, repository):
    source, _ = repository
    git(source, "remote", "set-url", "origin", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError, match="failed"):
        prepare(tmp_path, repository)


def test_reuse_rejects_wrong_pipeline_backend_or_feature_branch(tmp_path, repository):
    directory, manifest = prepare(tmp_path, repository)
    for pipeline, backend in [("weekly", "claude-cli"), ("daily", "api")]:
        with pytest.raises(RuntimeError, match="Existing run is daily/claude-cli"):
            runner.prepare(directory, pipeline, repository[0], None, backend)
    git(directory / "checkout", "checkout", "-b", "unfinished/science")
    with pytest.raises(RuntimeError, match="not on master"):
        runner.prepare(directory, "daily", repository[0], None, "claude-cli")


def test_every_extraction_pipeline_owns_the_convention_queue():
    for pipeline in runner.PIPELINES:
        assert runner.CONVENTION_QUEUE in runner.owned_files(pipeline), pipeline


def test_run_directory_lock_rejects_overlap(tmp_path):
    with runner.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another operation"):
            with runner.run_lock(tmp_path):
                pytest.fail("overlapping operation entered")


def test_cli_forwards_arguments_without_shell_expansion(tmp_path, repository):
    directory = tmp_path / "run with spaces"
    assert runner.main(["run", "daily", "--source", str(repository[0]), "--run-dir", str(directory),
                        "--", "--dry-run", "literal $(not-a-command)"]) == 0
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["attempts"][0]["arguments"][-1] == "literal $(not-a-command)"


def test_cli_resolves_relative_run_dir_against_caller_cwd(tmp_path, repository, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.main(["run", "daily", "--source", str(repository[0]), "--run-dir", "./here",
                        "--", "--dry-run"]) == 0
    assert (tmp_path / "here/run.json").exists()


def test_cli_normalises_backend_aliases_and_rejects_unknown(tmp_path, repository, monkeypatch, capsys):
    monkeypatch.setenv("AAL_BACKEND", "CLI")
    directory = tmp_path / "aliased"
    assert runner.main(["run", "daily", "--source", str(repository[0]), "--run-dir", str(directory),
                        "--", "--dry-run"]) == 0
    assert json.loads((directory / "run.json").read_text())["backend"] == "claude-cli"
    monkeypatch.setenv("AAL_BACKEND", "cluade-cli")
    with pytest.raises(SystemExit):
        runner.main(["run", "daily", "--source", str(repository[0]), "--run-dir", str(directory),
                     "--", "--dry-run"])
    assert "unknown backend" in capsys.readouterr().err


def test_state_diff_reports_pending_owned_changes(tmp_path, repository, capsys):
    directory, manifest = prepare(tmp_path, repository)
    assert runner.state_diff(directory) is False
    assert "No owned state changes." in capsys.readouterr().out
    (directory / "checkout/pipeline/state/processed.json").write_text('{"completed": ["paper-1"]}')
    assert runner.state_diff(directory) is True
    out = capsys.readouterr().out
    assert f"+++ {DAILY}/processed.json" in out and '+{"completed": ["paper-1"]}' in out


class TestMergeState:
    def test_union_keeps_both_sides_additions(self):
        base = {"processed_ids": ["a"], "failed_ids": {"a": "x"}, "last_run": "t0"}
        ours = {"processed_ids": ["a", "local"], "failed_ids": {}, "last_run": "t1"}
        theirs = {"processed_ids": ["a", "ci"], "failed_ids": {"a": "x", "ci2": "y"}, "last_run": "t2"}
        assert runner.merge_state(base, ours, theirs) == {
            "processed_ids": ["a", "ci", "local"],
            "failed_ids": {"a": "x", "ci2": "y"},  # union: no removals without allow_delete
            "last_run": "t2",  # monotone: the later timestamp wins regardless of side
        }

    def test_unchanged_side_yields_the_other(self):
        assert runner.merge_state({"a": 1}, {"a": 1}, {"a": 2}) == {"a": 2}
        assert runner.merge_state({"a": 1}, {"a": 3}, {"a": 1}) == {"a": 3}
        assert runner.merge_state(runner._ABSENT, {"a": 1}, {"b": 2}) == {"b": 2, "a": 1}

    def test_versions_timestamps_and_flags_are_monotone(self):
        base = {"files": {"f": {"known_version": 1, "last_checked": "t0", "published": False}}}
        ours = {"files": {"f": {"known_version": 2, "last_checked": "t2", "published": False}}}
        theirs = {"files": {"f": {"known_version": 3, "last_checked": "t1", "published": True, "withdrawn": True}}}
        assert runner.merge_state(base, ours, theirs) == {"files": {"f": {
            "known_version": 3, "last_checked": "t2", "published": True, "withdrawn": True}}}

    def test_both_sides_changing_a_free_scalar_is_a_conflict(self):
        with pytest.raises(runner.StateConflict, match="files/f/journal_ref"):
            runner.merge_state({"files": {"f": {"journal_ref": "c"}}},
                               {"files": {"f": {"journal_ref": "o"}}},
                               {"files": {"f": {"journal_ref": "t"}}})
        # Error/skip reasons are free text: this run's wording wins, no conflict.
        assert runner.merge_state({"failed_ids": {}}, {"failed_ids": {"x": "a"}},
                                  {"failed_ids": {"x": "b"}}) == {"failed_ids": {"x": "a"}}

    def test_convention_status_only_advances_and_ours_is_authoritative(self):
        base = {"entries": [{"cache_key": "k", "status": "promoted", "count": 1, "pr_url": "u1"}]}
        ours = {"entries": [{"cache_key": "k", "status": "promoted", "count": 1, "pr_url": "u1"},
                            {"cache_key": "new", "status": "queued"}]}
        lagging = {"entries": [{"cache_key": "k", "status": "queued", "count": 3, "pr_url": "u0"}]}
        merged = runner.merge_state(base, ours, lagging, prefer_ours=True)
        assert merged["entries"][0] == {"cache_key": "k", "status": "promoted", "count": 3, "pr_url": "u1"}
        assert merged["entries"][1] == {"cache_key": "new", "status": "queued"}
        # And a genuine remote advance is kept even when we are unchanged.
        advanced = {"entries": [{"cache_key": "k", "status": "promoted", "count": 1, "pr_url": "u1"},
                                {"cache_key": "ci", "status": "needs_human"}]}
        assert runner.merge_state(base, base, advanced, prefer_ours=True) == advanced

    def test_queue_consumption_survives_with_allow_delete(self):
        item = lambda i: {"arxiv_id": i, "title": f"t{i}"}
        base = {"queue": [item("1"), item("2"), item("3")], "processed_ids": [], "runs": []}
        ours = {"queue": [item("3")], "processed_ids": ["1", "2"], "runs": [{"processed": 2}]}
        theirs = {"queue": [item("2"), item("3"), item("4")], "processed_ids": ["1"], "runs": [{"processed": 1}]}
        merged = runner.merge_state(base, ours, theirs, allow_delete=True)
        assert [q["arxiv_id"] for q in merged["queue"]] == ["3", "4"]
        assert merged["processed_ids"] == ["1", "2"]
        assert merged["runs"] == [{"processed": 1}, {"processed": 2}]

    def test_keyed_entries_merge_recursively(self):
        base = {"entries": [{"cache_key": "k", "count": 1, "arxiv_ids": ["a"]}]}
        ours = {"entries": [{"cache_key": "k", "count": 2, "arxiv_ids": ["a", "b"]}]}
        theirs = {"entries": [{"cache_key": "k", "count": 3, "arxiv_ids": ["a", "c"]},
                              {"cache_key": "new", "count": 1}]}
        merged = runner.merge_state(base, ours, theirs)
        assert merged["entries"][0] == {"cache_key": "k", "count": 3, "arxiv_ids": ["a", "c", "b"]}
        assert merged["entries"][1] == {"cache_key": "new", "count": 1}

    def test_nested_file_map_merges_per_key(self):
        base = {"files": {"f1": {"known_version": 1}}}
        ours = {"files": {"f1": {"known_version": 2}, "f2": {"known_version": 1}}}
        theirs = {"files": {"f1": {"known_version": 1, "published": True}, "f3": {"known_version": 4}}}
        assert runner.merge_state(base, ours, theirs) == {"files": {
            "f1": {"known_version": 2, "published": True},
            "f3": {"known_version": 4},
            "f2": {"known_version": 1},
        }}
