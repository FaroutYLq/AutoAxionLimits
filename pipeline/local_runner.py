"""Run a pipeline in a recoverable, isolated clone; publish state explicitly.

Uses only the standard library. Model calls remain in the existing entrypoints.
The clone has its own master branch because those entrypoints return to master.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


PIPELINES = {
    "daily": ("pipeline.orchestrator", "processed.json", "chore/update-pipeline-state"),
    "weekly": ("pipeline.preprint_checker", "preprint_versions.json", "chore/update-preprint-state"),
    "backfill": ("pipeline.backfill", "backfill_state.json", "chore/update-backfill-state"),
}
SOURCE = Path(__file__).resolve().parents[1]


def command(argv, cwd, *, check=True, env=None, input=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True,
                            env=env, input=input, timeout=120)
    if check and result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return result


def git(repo, *args, **kwargs):
    # Use the authenticated gh helper without changing the user's git config.
    return command(["git", "-c", "credential.helper=", "-c",
                    "credential.helper=!gh auth git-credential", *args], repo, **kwargs)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def state_snapshot(checkout):
    return {p.name: p.read_text() for p in sorted((checkout / "pipeline/state").glob("*.json"))}


@contextmanager
def run_lock(directory):
    """Serialize local reuse and publication; remote publication also uses a lease."""
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another operation is using {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare(directory, pipeline, source, ref, backend):
    manifest_path = directory / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["pipeline"] != pipeline or manifest["backend"] != backend:
            raise RuntimeError("Existing run has a different pipeline/backend; use a new run directory")
        if git(directory / "checkout", "branch", "--show-current").stdout.strip() != "master":
            raise RuntimeError("Checkout is not on master after an interrupted run; inspect it before resuming")
        return manifest

    checkout = directory / "checkout"
    if checkout.exists():
        raise RuntimeError("Incomplete setup: checkout already exists; inspect it or choose a new run directory")
    base = git(source, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    remote = git(source, "remote", "get-url", "origin").stdout.strip()
    # Shared objects avoid copying the large plot history. This clone must stay
    # on this machine with its source repo; never delete its source object store.
    git(source, "clone", "--shared", "--no-checkout", str(source), str(checkout))
    git(checkout, "remote", "set-url", "origin", remote)
    git(checkout, "checkout", "-B", "master", base)
    git(checkout, "fetch", "--no-tags", "origin", "master")
    publication_base = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
    for field in ("user.name", "user.email"):
        value = git(source, "config", "--get", field, check=False)
        if value.returncode == 0:
            git(checkout, "config", field, value.stdout.strip())

    _, filename, branch = PIPELINES[pipeline]
    result = git(checkout, "ls-remote", "origin", f"refs/heads/{branch}")
    remote_oid = None
    if result.stdout.strip():
        git(checkout, "fetch", "origin", branch)
        remote_oid = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
        restored = git(checkout, "show", f"{remote_oid}:pipeline/state/{filename}", check=False)
        if restored.returncode:
            raise RuntimeError(f"State branch {branch} exists but its {filename} cannot be read")
        # Validate before allowing a run to overwrite or publish the baseline.
        json.loads(restored.stdout)
        (checkout / "pipeline/state" / filename).write_text(restored.stdout)

    # Commit ONLY restored state on the private local master. Otherwise branch
    # switches in the pipeline could discard that restored baseline.
    git(checkout, "add", "--", f"pipeline/state/{filename}")
    if git(checkout, "diff", "--cached", "--quiet", check=False).returncode:
        git(checkout, "commit", "-m", "Local pipeline state baseline")
    manifest = {
        "pipeline": pipeline, "backend": backend, "source": str(source),
        "base_commit": base, "publication_base": publication_base,
        "state_branch": branch, "state_remote_oid": remote_oid,
        "baseline_state": state_snapshot(checkout), "attempts": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def execute(directory, manifest, arguments):
    checkout = directory / "checkout"
    module, _, _ = PIPELINES[manifest["pipeline"]]
    attempt_number = len(manifest["attempts"]) + 1
    log_path = directory / f"attempt-{attempt_number}.log"
    before = state_snapshot(checkout)
    write_json(directory / f"attempt-{attempt_number}-before.json", before)
    attempt = {"arguments": arguments, "started": datetime.now(timezone.utc).isoformat(),
               "log": str(log_path), "exit_code": None, "status": "running"}
    manifest["attempts"].append(attempt)
    write_json(directory / "run.json", manifest)
    env = os.environ.copy()
    env["AAL_BACKEND"] = manifest["backend"]
    # A benchmark may leave this override exported. A local operational run
    # must not write its escalation queue into another checkout or evaluation.
    env["AAL_CONVENTION_QUEUE"] = str(checkout / "pipeline/state/convention_queue.json")
    env.pop("GITHUB_OUTPUT", None)
    argv = [sys.executable, "-u", "-m", module, *arguments]
    print(f"Run directory: {directory}\nBase commit: {manifest['base_commit']}\n"
          f"Backend: {manifest['backend']}\nLog: {log_path}", flush=True)
    code = 1
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(argv, cwd=checkout, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                process.stdout.close()
    except KeyboardInterrupt:
        code = 130
    finally:
        after = state_snapshot(checkout)
        write_json(directory / f"attempt-{attempt_number}-after.json", after)
        attempt.update(exit_code=code, status="finished", changed_state=[
            name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)
        ], pr_urls=sorted(set(re.findall(r"https://github\.com/[^\s/]+/[^\s/]+/pull/\d+",
                                       log_path.read_text() if log_path.exists() else ""))))
        write_json(directory / "run.json", manifest)
        print(f"Exit: {code}; changed state: {attempt['changed_state']}; PRs: {attempt['pr_urls']}")
        print(f"Checkout and recovery files retained at {directory}")
    return code


def publish_state(directory):
    """Publish only this pipeline's owned state file, never a science commit."""
    manifest = json.loads((directory / "run.json").read_text())
    checkout = directory / "checkout"
    _, filename, branch = PIPELINES[manifest["pipeline"]]
    relative = f"pipeline/state/{filename}"
    content = (checkout / relative).read_text()
    json.loads(content)
    if content == manifest["baseline_state"].get(filename):
        print("No owned state changes to publish.")
        return
    # A private index based on remote master excludes science and development
    # changes even when the selected execution revision is a feature branch.
    with tempfile.TemporaryDirectory(prefix="aal-state-index-") as temporary:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        git(checkout, "read-tree", manifest["publication_base"], env=env)
        git(checkout, "add", "--", relative, env=env)
        tree = git(checkout, "write-tree", env=env).stdout.strip()
        commit = git(checkout, "commit-tree", tree, "-p", manifest["publication_base"],
                     "-m", f"chore: update {manifest['pipeline']} pipeline state").stdout.strip()
    expected = manifest["state_remote_oid"] or ""
    git(checkout, "push", f"--force-with-lease=refs/heads/{branch}:{expected}",
        "origin", f"{commit}:refs/heads/{branch}")
    # Persist the successful push before gh, so a PR-creation error doesn't
    # lose knowledge of the new remote lease.
    manifest["state_remote_oid"] = commit
    write_json(directory / "run.json", manifest)
    prs = command(["gh", "pr", "list", "--head", branch, "--base", "master", "--state", "open",
                   "--json", "url"], checkout)
    existing = json.loads(prs.stdout)
    if existing:
        url = existing[0]["url"]
    else:
        body = directory / "state-pr-body.md"
        body.write_text(f"Updates `{relative}` from a local {manifest['pipeline']} run.\n\n"
                        "Contains only pipeline bookkeeping. Scientific proposals remain in separate PRs.\n")
        url = command(["gh", "pr", "create", "--base", "master", "--head", branch,
                       "--title", f"chore: update {manifest['pipeline']} pipeline state",
                       "--body-file", str(body)], checkout).stdout.strip()
    manifest["baseline_state"][filename] = content
    manifest["state_pr_url"] = url
    write_json(directory / "run.json", manifest)
    print(url)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run", help="Start or reuse an isolated run")
    run.add_argument("pipeline", choices=PIPELINES)
    run.add_argument("--run-dir", type=Path, help="Reuse this directory to continue a run")
    run.add_argument("--source", type=Path, default=SOURCE)
    run.add_argument("--ref", default="HEAD", help="Committed source revision for a NEW run")
    run.add_argument("--backend", choices=("api", "claude-cli"),
                     default=os.environ.get("AAL_BACKEND", "claude-cli"))
    publish = subparsers.add_parser("publish-state", help="Push owned state with a lease and open/reuse its PR")
    publish.add_argument("run_dir", type=Path)
    # Everything after -- belongs to the existing pipeline parser, unmodified.
    values = list(sys.argv[1:] if argv is None else argv)
    divider = values.index("--") if "--" in values else len(values)
    args = parser.parse_args(values[:divider])
    forwarded = values[divider + 1:]
    if args.operation != "run" and forwarded:
        parser.error("Only run accepts pipeline arguments after --")
    directory = args.run_dir
    if directory is None:
        runs = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "autoaxionlimits/runs"
        runs.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=runs))
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with run_lock(directory):
            if args.operation == "publish-state":
                publish_state(directory)
                return 0
            manifest = prepare(directory, args.pipeline, args.source.resolve(), args.ref, args.backend)
            return execute(directory, manifest, forwarded)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Local runner stopped: {error}\nRecovery directory: {directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
