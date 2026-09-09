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
if "--dry-run" not in sys.argv:
    Path("pipeline/state", name).write_text(json.dumps({"completed": ["paper-1"]}))
    print("PR created: https://github.com/example/fixture/pull/17")
if "--fail" in sys.argv:
    sys.exit(2)
'''


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
    for module, filename, branch in runner.PIPELINES.values():
        (pipeline / (module.split(".")[-1] + ".py")).write_text(STUB)
        (pipeline / "state" / filename).write_text('{"completed": []}')
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


def prepare(tmp_path, repository, pipeline="daily"):
    source, _ = repository
    directory = tmp_path / "run"
    directory.mkdir()
    manifest = runner.prepare(directory, pipeline, source, "HEAD", "claude-cli")
    return directory, manifest


@pytest.mark.parametrize("pipeline", runner.PIPELINES)
def test_isolation_preview_real_run_and_reuse(tmp_path, repository, pipeline):
    directory, manifest = prepare(tmp_path, repository, pipeline)
    source, _ = repository
    before = runner.state_snapshot(source)
    assert git(directory / "checkout", "branch", "--show-current") == "master"
    assert (directory / "checkout/science.txt").read_text() == "original science\n"
    assert runner.execute(directory, manifest, ["--dry-run"]) == 0
    assert manifest["attempts"][-1]["changed_state"] == []
    assert manifest["attempts"][-1]["pr_urls"] == []
    reused = runner.prepare(directory, pipeline, source, "HEAD", "claude-cli")
    assert runner.execute(directory, reused, []) == 0
    filename = runner.PIPELINES[pipeline][1]
    assert reused["attempts"][-1]["changed_state"] == [filename]
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
    after = json.loads((directory / "attempt-1-after.json").read_text())
    assert json.loads(after["processed.json"])["completed"] == ["paper-1"]
    runner.prepare(directory, "daily", repository[0], "HEAD", "claude-cli")
    assert (directory / "checkout").exists()


def test_restore_only_owned_file_from_remote_state_branch(tmp_path, repository):
    source, _ = repository
    branch = runner.PIPELINES["daily"][2]
    # Create a remote state snapshot with an unrelated file that must not leak.
    git(source, "checkout", "-b", branch)
    (source / "pipeline/state/processed.json").write_text('{"completed": ["remote-paper"]}')
    (source / "pipeline/state/preprint_versions.json").write_text('{"unrelated": true}')
    git(source, "add", "pipeline/state")
    git(source, "commit", "-m", "remote state")
    oid = git(source, "rev-parse", "HEAD")
    git(source, "push", "origin", branch)
    git(source, "checkout", "master")
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    assert manifest["state_remote_oid"] == oid
    assert json.loads((checkout / "pipeline/state/processed.json").read_text())["completed"] == ["remote-paper"]
    assert (checkout / "pipeline/state/preprint_versions.json").read_text() == '{"completed": []}'
    # Baseline is committed locally, so a science branch round trip retains it.
    git(checkout, "checkout", "-b", "fixture/science")
    git(checkout, "checkout", "master")
    assert git(checkout, "status", "--porcelain") == ""
    assert json.loads((checkout / "pipeline/state/processed.json").read_text())["completed"] == ["remote-paper"]


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
def test_state_publication_excludes_science_and_other_state(tmp_path, repository, monkeypatch, existing):
    source, _ = repository
    # Execution may use a development revision; it must not become a state PR.
    (source / "development.txt").write_text("unmerged implementation\n")
    git(source, "add", "development.txt")
    git(source, "commit", "-m", "development revision")
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    (checkout / "pipeline/state/processed.json").write_text('{"completed": ["paper-1"]}')
    (checkout / "pipeline/state/convention_queue.json").write_text('{"entries": ["keep locally"]}')
    (checkout / "science.txt").write_text("do not publish this\n")
    git(checkout, "add", ".")  # Even staged science must not enter the state PR.
    calls = fake_gh(monkeypatch, existing=existing)
    runner.publish_state(directory)
    branch = manifest["state_branch"]
    remote = repository[1]
    assert git(remote, "diff", "--name-only", "master", branch) == "pipeline/state/processed.json"
    assert git(remote, "show", f"{branch}:science.txt") == "original science"
    assert (checkout / "science.txt").read_text() == "do not publish this\n"
    assert sum(call[1:3] == ["pr", "create"] for call in calls) == (0 if existing else 1)
    runner.publish_state(directory)  # No-op after successful publication.
    assert len(calls) == (1 if existing else 2)


def test_stale_lease_rejects_remote_race_and_preserves_local_state(tmp_path, repository, monkeypatch):
    directory, manifest = prepare(tmp_path, repository)
    checkout = directory / "checkout"
    content = '{"completed": ["local-paper"]}'
    (checkout / "pipeline/state/processed.json").write_text(content)
    source, remote = repository
    branch = manifest["state_branch"]
    git(source, "push", "origin", f"HEAD:refs/heads/{branch}")
    remote_before = git(remote, "rev-parse", branch)
    calls = fake_gh(monkeypatch)
    with pytest.raises(RuntimeError, match="failed"):
        runner.publish_state(directory)
    assert git(remote, "rev-parse", branch) == remote_before
    assert (checkout / "pipeline/state/processed.json").read_text() == content
    assert calls == []


def test_pr_outage_after_push_can_be_retried(tmp_path, repository, monkeypatch):
    directory, manifest = prepare(tmp_path, repository)
    (directory / "checkout/pipeline/state/processed.json").write_text('{"completed": ["paper-1"]}')
    fake_gh(monkeypatch, fail_create=True)
    with pytest.raises(RuntimeError, match="simulated PR outage"):
        runner.publish_state(directory)
    saved = json.loads((directory / "run.json").read_text())
    assert saved["state_remote_oid"] == git(repository[1], "rev-parse", manifest["state_branch"])
    fake_gh(monkeypatch, existing=True)
    runner.publish_state(directory)
    assert json.loads((directory / "run.json").read_text())["state_pr_url"].endswith("/21")


def test_remote_failure_is_not_treated_as_missing_state(tmp_path, repository):
    source, _ = repository
    git(source, "remote", "set-url", "origin", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError, match="failed"):
        prepare(tmp_path, repository)


def test_reuse_rejects_wrong_pipeline_backend_or_feature_branch(tmp_path, repository):
    directory, manifest = prepare(tmp_path, repository)
    for pipeline, backend in [("weekly", "claude-cli"), ("daily", "api")]:
        with pytest.raises(RuntimeError, match="different"):
            runner.prepare(directory, pipeline, repository[0], "HEAD", backend)
    git(directory / "checkout", "checkout", "-b", "unfinished/science")
    with pytest.raises(RuntimeError, match="not on master"):
        runner.prepare(directory, "daily", repository[0], "HEAD", "claude-cli")


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
