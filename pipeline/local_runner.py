"""Run a pipeline in a recoverable, isolated clone; publish state explicitly.

Uses only the standard library. Model calls remain in the existing entrypoints.
The clone has its own master branch because those entrypoints return to master.

State publication is a three-way merge: each owned state file is merged onto
the current remote state-branch tip (base = the copy restored at setup, ours =
this run's copy, theirs = the tip's copy) and pushed with a lease on that tip,
so a GitHub Actions run that advanced the branch in the meantime is absorbed
rather than rejected forever.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
from typing import NamedTuple


class Owned(NamedTuple):
    """One state file a pipeline owns, and the reusable branch it publishes to."""
    file: str
    branch: str
    restore: bool = True        # restore the branch copy as this run's baseline
    allow_delete: bool = False  # removals (consumed queue items) survive a merge
    prefer_ours: bool = False   # this run's copy is authoritative on disagreement


PIPELINE_STATE = "chore/update-pipeline-state"
PREPRINT_STATE = "chore/update-preprint-state"
BACKFILL_STATE = "chore/update-backfill-state"
# The convention queue is append-only bookkeeping shared by every extraction
# path (daily, weekly and backfill all run the extraction agent). It is never
# restored and this run's copy (master's triage results plus the new flags) is
# authoritative on disagreement: the branch copy can lag master, and a lagging
# copy must never regress a promoted convention to queued.
CONVENTION_QUEUE = Owned("convention_queue.json", PIPELINE_STATE, restore=False, prefer_ours=True)
PIPELINES = {
    "daily": ("pipeline.orchestrator",
              [Owned("processed.json", PIPELINE_STATE), CONVENTION_QUEUE]),
    "weekly": ("pipeline.preprint_checker",
               [Owned("preprint_versions.json", PREPRINT_STATE), CONVENTION_QUEUE]),
    # Backfill also advances the daily processed-papers file (backfill.py
    # marks both); restoring it from the daily branch is what lets
    # build_known_ids() skip papers the daily digest already proposed.
    "backfill": ("pipeline.backfill",
                 [Owned("backfill_state.json", BACKFILL_STATE, allow_delete=True),
                  Owned("processed.json", PIPELINE_STATE), CONVENTION_QUEUE]),
}
SOURCE = Path(__file__).resolve().parents[1]
NETWORK_TIMEOUT = 900
# Environment that changes what the pipeline extracts or how it is billed.
# A benchmark shell may leave these exported; a real run must not inherit them
# silently (the model confound is invisible in the science PR). Operational
# knobs that do not alter extraction output pass through.
ENV_PASSTHROUGH = {"AAL_CLI_BINARY", "AAL_CLI_TIMEOUT", "AAL_CLI_VISION_TIMEOUT",
                   "AAL_PDF_CACHE", "AAL_SOURCE_CACHE"}
ENV_SCRUBBED_NAMES = {"EXTRACTOR_MODEL", "REVIEWER_MODEL"}
SENSITIVE_ENV = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL|AUTH", re.IGNORECASE)
_ABSENT = object()


def command(argv, cwd, *, check=True, env=None, input=None, timeout=120):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True,
                            env=env, input=input, timeout=timeout)
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


def owned_files(pipeline):
    return PIPELINES[pipeline][1]


def owned_branches(pipeline):
    return list(dict.fromkeys(owned.branch for owned in owned_files(pipeline)))


def remote_tips(checkout, branches):
    """One ls-remote for every state branch: {branch: oid} for those that exist."""
    result = git(checkout, "ls-remote", "origin", *[f"refs/heads/{b}" for b in branches],
                 timeout=NETWORK_TIMEOUT)
    tips = {}
    for line in result.stdout.splitlines():
        oid, _, ref = line.partition("\t")
        if ref.startswith("refs/heads/"):
            tips[ref[len("refs/heads/"):]] = oid
    return tips


def fetch_tip(checkout, branch, expected):
    """Fetch a state branch and confirm it is still the tip ls-remote reported."""
    git(checkout, "fetch", "--no-tags", "origin", branch, timeout=NETWORK_TIMEOUT)
    fetched = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
    if fetched != expected:
        raise RuntimeError(f"{branch} moved while it was being read; retry")
    return fetched


def read_state_file(checkout, commit, filename):
    """The JSON text of pipeline/state/<filename> at *commit*, or None if absent."""
    result = git(checkout, "show", f"{commit}:pipeline/state/{filename}", check=False)
    if result.returncode:
        return None
    json.loads(result.stdout)  # Validate before it becomes a baseline.
    return result.stdout


class StateConflict(RuntimeError):
    """Both sides changed the same value differently and no rule can combine them."""


# Field rules applied at any depth. Monotone fields take the larger (or
# smaller) value so a lagging copy on either side can never roll progress
# back; booleans OR (published/withdrawn are one-way); a convention status
# only ever advances along the triage lifecycle.
MAX_FIELDS = {"schema_version", "known_version", "last_checked", "last_run", "last_seen", "count"}
MIN_FIELDS = {"first_seen"}
STATUS_RANK = {"queued": 0, "needs_human": 1, "unconvertible": 2, "promoted": 3}
# Free-text bookkeeping (error/skip reasons) is never worth blocking a
# publication: this run's wording wins.
OURS_WIN_CONTAINERS = {"failed_ids", "skipped_ids"}


def _merge_key(item):
    if isinstance(item, dict):
        for key in ("cache_key", "arxiv_id"):
            if key in item:
                return (key, json.dumps(item[key], sort_keys=True))
        return ("dict", json.dumps(item, sort_keys=True))
    if isinstance(item, list):
        return ("list", json.dumps(item, sort_keys=True))
    return ("scalar", item)


def _same_kind(ours, theirs):
    return (isinstance(ours, bool) == isinstance(theirs, bool)
            and ((isinstance(ours, (int, float)) and isinstance(theirs, (int, float)))
                 or (isinstance(ours, str) and isinstance(theirs, str))))


def _resolve_scalar(path, base, ours, theirs, prefer_ours):
    key = path[-1] if path else None
    if isinstance(ours, bool) and isinstance(theirs, bool):
        return ours or theirs
    if key in MAX_FIELDS and _same_kind(ours, theirs):
        return max(ours, theirs)
    if key in MIN_FIELDS and _same_kind(ours, theirs):
        return min(ours, theirs)
    if key == "status" and ours in STATUS_RANK and theirs in STATUS_RANK:
        return max(ours, theirs, key=STATUS_RANK.get)
    if prefer_ours or any(part in OURS_WIN_CONTAINERS for part in path):
        return ours
    if theirs == base:
        return ours
    if ours == base:
        return theirs
    raise StateConflict(
        f"{'/'.join(map(str, path)) or '<root>'}: this run has {ours!r}, the state branch has "
        f"{theirs!r}, both changed from {'<absent>' if base is _ABSENT else repr(base)}")


def _merge_withdrawal(ours, theirs, path):
    """Withdrawal belongs to a paper version; a later version can reinstate it."""
    our_version = ours.get("known_version")
    their_version = theirs.get("known_version")
    if (type(our_version) is int and type(their_version) is int
            and our_version != their_version):
        newer = ours if our_version > their_version else theirs
        # Absence is intentional when the checker cleared the flag.
        return newer.get("withdrawn", _ABSENT)
    if ours.get("withdrawn", False) == theirs.get("withdrawn", False):
        return ours.get("withdrawn", _ABSENT)
    raise StateConflict(
        f"{'/'.join(map(str, path))}/withdrawn: withdrawal status differs without "
        f"a newer known_version (this run: {our_version!r}, state branch: {their_version!r})")


def merge_state(base, ours, theirs, *, allow_delete=False, prefer_ours=False, path=()):
    """Three-way merge of JSON state with per-field rules (see MAX_FIELDS etc.).

    Dict keys merge recursively; lists are keyed sets (cache_key / arxiv_id for
    dict items) merged the same way. Removals relative to *base* are honoured
    only with *allow_delete* (queue consumption); otherwise the result is a
    union. In preprint file records, withdrawal follows the newer version,
    including flag removal on reinstatement; ambiguous status is a conflict.
    A scalar both sides changed differently, with no monotone rule and
    no *prefer_ours*, raises StateConflict rather than guessing.
    """
    if ours == theirs:
        return ours
    options = dict(allow_delete=allow_delete, prefer_ours=prefer_ours)
    if isinstance(ours, dict) and isinstance(theirs, dict):
        base = base if isinstance(base, dict) else {}
        result = {}
        versioned_withdrawal = (len(path) == 2 and path[0] == "files"
                               and ("withdrawn" in ours or "withdrawn" in theirs))
        if versioned_withdrawal:
            withdrawn = _merge_withdrawal(ours, theirs, path)
        for key in dict.fromkeys([*theirs, *ours]):
            if key == "withdrawn" and versioned_withdrawal:
                if withdrawn is not _ABSENT:
                    result[key] = withdrawn
            elif key in ours and key in theirs:
                result[key] = merge_state(base.get(key, _ABSENT), ours[key], theirs[key],
                                          path=(*path, key), **options)
            elif key in ours:
                if not (allow_delete and key in base):  # else deleted by them
                    result[key] = ours[key]
            elif not (allow_delete and key in base):  # else deleted by us
                result[key] = theirs[key]
        return result
    if isinstance(ours, list) and isinstance(theirs, list):
        base_items = {_merge_key(x): x for x in (base if isinstance(base, list) else [])}
        our_items = {_merge_key(x): x for x in ours}
        their_items = {_merge_key(x): x for x in theirs}
        result = []
        for key in dict.fromkeys([*their_items, *our_items]):
            if key in our_items and key in their_items:
                result.append(merge_state(base_items.get(key, _ABSENT), our_items[key],
                                          their_items[key], path=(*path, key[1]), **options))
            elif key in our_items:
                if not (allow_delete and key in base_items):
                    result.append(our_items[key])
            elif not (allow_delete and key in base_items):
                result.append(their_items[key])
        return result
    return _resolve_scalar(path, base, ours, theirs, prefer_ours)


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


def current_branch(checkout):
    return git(checkout, "branch", "--show-current").stdout.strip()


def prepare(directory, pipeline, source, ref, backend):
    """Create (or validate for reuse) the isolated clone and restore its state baseline.

    *ref* None selects remote master, the production revision. Any other
    revision that is not already on remote master is recorded as unpublished:
    science PRs branch from the clone's master, so a real run from such a
    revision would carry its unmerged commits into every science PR.
    """
    manifest_path = directory / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if "leases" not in manifest or "ref_published" not in manifest:
            raise RuntimeError("This run directory predates the leased multi-file publication "
                               "format; publish or archive it with the runner that created it")
        if manifest["pipeline"] != pipeline or manifest["backend"] != backend:
            raise RuntimeError(
                f"Existing run is {manifest['pipeline']}/{manifest['backend']}, not "
                f"{pipeline}/{backend}; pass the same values or use a new run directory")
        if current_branch(directory / "checkout") != "master":
            raise RuntimeError("Checkout is not on master after an interrupted run; inspect it before resuming")
        return manifest

    checkout = directory / "checkout"
    if checkout.exists():
        raise RuntimeError("Incomplete setup: checkout already exists; inspect it or choose a new run directory")
    remote = git(source, "remote", "get-url", "origin").stdout.strip()
    # Shared objects avoid copying the large plot history. This clone must stay
    # on this machine with its source repo; never delete its source object store.
    git(source, "clone", "--shared", "--no-checkout", str(source), str(checkout))
    git(checkout, "remote", "set-url", "origin", remote)
    git(checkout, "fetch", "--no-tags", "origin", "master", timeout=NETWORK_TIMEOUT)
    publication_base = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
    if ref is None:
        base = publication_base
    else:
        base = git(source, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    git(checkout, "checkout", "-B", "master", base)
    ref_published = git(checkout, "merge-base", "--is-ancestor", base, publication_base,
                        check=False).returncode == 0
    for field in ("user.name", "user.email"):
        value = git(source, "config", "--get", field, check=False)
        if value.returncode == 0:
            git(checkout, "config", field, value.stdout.strip())

    tips = remote_tips(checkout, owned_branches(pipeline))
    for branch, tip in tips.items():
        fetch_tip(checkout, branch, tip)
    restored = []
    for owned in owned_files(pipeline):
        tip = tips.get(owned.branch)
        if not owned.restore or tip is None:
            continue
        text = read_state_file(checkout, tip, owned.file)
        if text is None:
            print(f"{owned.branch} has no {owned.file}; using the master baseline", flush=True)
            continue
        (checkout / "pipeline/state" / owned.file).write_text(text)
        restored.append(f"pipeline/state/{owned.file}")
    # Commit ONLY restored state on the private local master. Otherwise branch
    # switches in the pipeline could discard that restored baseline.
    if restored:
        git(checkout, "add", "--", *restored)
        if git(checkout, "diff", "--cached", "--quiet", check=False).returncode:
            git(checkout, "commit", "-m", "Local pipeline state baseline")
    manifest = {
        "pipeline": pipeline, "backend": backend, "source": str(source),
        "base_commit": base, "ref": ref, "ref_published": ref_published,
        "publication_base": publication_base,
        "leases": {branch: tips.get(branch) for branch in owned_branches(pipeline)},
        "state_prs": {},
        "baseline_state": state_snapshot(checkout), "attempts": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def child_environment(parent, checkout, backend, overrides=()):
    """The child's environment: benchmark overrides scrubbed, explicit ones applied."""
    env = {name: value for name, value in parent.items()
           if not ((name.startswith("AAL_") and name not in ENV_PASSTHROUGH)
                   or name in ENV_SCRUBBED_NAMES)}
    scrubbed = sorted(set(parent) - set(env))
    applied = {}
    for item in overrides:
        name, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"--env expects NAME=VALUE, got {item!r}")
        env[name] = value
        # Recorded for provenance, never as a credential store.
        applied[name] = "<redacted>" if SENSITIVE_ENV.search(name) else value
    env["AAL_BACKEND"] = backend
    # A benchmark may leave this override exported. A local operational run
    # must not write its escalation queue into another checkout or evaluation.
    env["AAL_CONVENTION_QUEUE"] = str(checkout / "pipeline/state/convention_queue.json")
    env.pop("GITHUB_OUTPUT", None)
    return env, scrubbed, applied


def _stream(process, log_path):
    with log_path.open("w") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)


def _terminate(process):
    """SIGINT the child's session; a second interrupt or a 30 s stall escalates
    to SIGKILL. Always returns with the child reaped."""
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=30)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, ProcessLookupError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return process.returncode


def execute(directory, manifest, arguments, *, env_overrides=(), allow_unmerged_ref=False):
    checkout = directory / "checkout"
    module, _ = PIPELINES[manifest["pipeline"]]
    if not manifest["ref_published"] and "--dry-run" not in arguments and not allow_unmerged_ref:
        raise RuntimeError(
            f"Revision {manifest['base_commit'][:12]} ({manifest['ref']}) is not on remote "
            "master, so science PRs from a real run would carry its unmerged commits. "
            "Preview with --dry-run, start a new run from remote master (omit --ref), "
            "or pass --allow-unmerged-ref deliberately.")
    env, scrubbed, applied = child_environment(os.environ, checkout, manifest["backend"], env_overrides)
    attempt_number = len(manifest["attempts"]) + 1
    log_path = directory / f"attempt-{attempt_number}.log"
    before = state_snapshot(checkout)
    write_json(directory / f"attempt-{attempt_number}-before.json", before)
    attempt = {"arguments": arguments, "started": datetime.now(timezone.utc).isoformat(),
               "log": str(log_path), "exit_code": None, "status": "running",
               "env_scrubbed": scrubbed, "env_overrides": applied}
    manifest["attempts"].append(attempt)
    write_json(directory / "run.json", manifest)
    argv = [sys.executable, "-u", "-m", module, *arguments]
    print(f"Run directory: {directory}\nBase commit: {manifest['base_commit']}"
          f"{'' if manifest['ref_published'] else ' (NOT on remote master)'}\n"
          f"Backend: {manifest['backend']}\nLog: {log_path}", flush=True)
    if scrubbed:
        print(f"Ignored inherited environment: {' '.join(scrubbed)}", flush=True)
    if applied:
        print(f"Explicit environment: {' '.join(f'{k}={v}' for k, v in applied.items())}", flush=True)
    code = None
    status = "crashed"
    process = None
    try:
        process = subprocess.Popen(argv, cwd=checkout, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   start_new_session=True)
        # A reader thread keeps draining the pipe even while the child unwinds
        # after a signal, so its cleanup can never block on a full pipe.
        reader = threading.Thread(target=_stream, args=(process, log_path), daemon=True)
        reader.start()
        try:
            code = process.wait()
            status = "finished"
        except BaseException as error:
            code = _terminate(process)
            status = "interrupted"
            if not isinstance(error, KeyboardInterrupt):
                raise
        finally:
            reader.join(timeout=10)
    finally:
        after = state_snapshot(checkout)
        write_json(directory / f"attempt-{attempt_number}-after.json", after)
        log_text = log_path.read_text() if log_path.exists() else ""
        attempt.update(exit_code=code, status=status, changed_state=[
            name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)
        ], pr_urls=sorted(set(re.findall(
            r"(?m)^.*\b(?:PR created|Created PR)\b.*?(https://github\.com/[^\s/]+/[^\s/]+/pull/\d+)",
            log_text))))
        write_json(directory / "run.json", manifest)
        print(f"Exit: {code} ({status}); changed state: {attempt['changed_state']}; PRs: {attempt['pr_urls']}")
        print(f"Checkout and recovery files retained at {directory}")
    if status == "interrupted":
        return 130
    return 1 if code is None else code


def state_diff(directory):
    """Print the owned-state changes this run would publish (for review first)."""
    manifest = json.loads((directory / "run.json").read_text())
    checkout = directory / "checkout"
    changed = False
    for owned in owned_files(manifest["pipeline"]):
        baseline = manifest["baseline_state"].get(owned.file, "")
        current = (checkout / "pipeline/state" / owned.file).read_text()
        if current == baseline:
            continue
        changed = True
        sys.stdout.writelines(difflib.unified_diff(
            baseline.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile=f"baseline/{owned.file}", tofile=f"{owned.branch}/{owned.file}"))
    if not changed:
        print("No owned state changes.")
    return changed


def publish_state(directory):
    """Publish only this pipeline's owned state files, never a science commit."""
    manifest = json.loads((directory / "run.json").read_text())
    checkout = directory / "checkout"
    pipeline = manifest["pipeline"]
    if current_branch(checkout) != "master":
        raise RuntimeError("Checkout is not on master: the run was interrupted mid-paper and its "
                           "state may record a paper that has no PR. Inspect it before publishing")
    current = {owned.file: (checkout / "pipeline/state" / owned.file).read_text()
               for owned in owned_files(pipeline)}
    for text in current.values():
        json.loads(text)
    pending = {}
    for owned in owned_files(pipeline):
        if current[owned.file] != manifest["baseline_state"].get(owned.file):
            pending.setdefault(owned.branch, []).append(owned)
    if not pending:
        print("No owned state changes to publish.")
        return
    tips = remote_tips(checkout, list(pending))
    for branch, files in pending.items():
        tip = tips.get(branch)
        parent = fetch_tip(checkout, branch, tip) if tip else manifest["publication_base"]
        merged = {}
        for owned in files:
            ours_text = current[owned.file]
            theirs_text = read_state_file(checkout, parent, owned.file) if tip else None
            if theirs_text is None:
                merged[owned.file] = ours_text
                continue
            baseline = manifest["baseline_state"].get(owned.file)
            try:
                value = merge_state(json.loads(baseline) if baseline is not None else _ABSENT,
                                    json.loads(ours_text), json.loads(theirs_text),
                                    allow_delete=owned.allow_delete, prefer_ours=owned.prefer_ours)
            except StateConflict as conflict:
                raise RuntimeError(
                    f"{owned.file} cannot be reconciled with {branch} automatically ({conflict}). "
                    f"Inspect both copies, edit the run's file, then retry; nothing was pushed") from conflict
            merged[owned.file] = ours_text if value == json.loads(ours_text) else json.dumps(value, indent=2) + "\n"
            if merged[owned.file] != ours_text:
                print(f"Merged {owned.file} with the concurrent update on {branch}", flush=True)
        # A previous attempt may have pushed and then failed at PR creation:
        # the tip already holds this content, so only the PR step remains.
        already_published = bool(tip) and all(
            read_state_file(checkout, parent, filename) == text for filename, text in merged.items())
        if already_published:
            print(f"{branch} already holds this state; ensuring its PR exists", flush=True)
        else:
            # A private index based on the branch tip (or remote master)
            # excludes science and development changes even when the selected
            # execution revision is a feature branch, and keeps whatever else
            # CI committed on the branch.
            with tempfile.TemporaryDirectory(prefix="aal-state-index-") as temporary:
                env = os.environ.copy()
                env["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
                git(checkout, "read-tree", parent, env=env)
                for filename, text in merged.items():
                    blob = git(checkout, "hash-object", "-w", "--stdin", input=text).stdout.strip()
                    git(checkout, "update-index", "--add", "--cacheinfo",
                        f"100644,{blob},pipeline/state/{filename}", env=env)
                tree = git(checkout, "write-tree", env=env).stdout.strip()
            commit = git(checkout, "commit-tree", tree, "-p", parent,
                         "-m", f"chore: update {pipeline} pipeline state").stdout.strip()
            git(checkout, "push", f"--force-with-lease=refs/heads/{branch}:{tip or ''}",
                "origin", f"{commit}:refs/heads/{branch}", timeout=NETWORK_TIMEOUT)
            # Persist the successful push before gh, so a PR-creation error
            # doesn't lose knowledge of the new remote lease. The baseline is
            # advanced only once the PR exists, so a retry still gets here.
            manifest["leases"][branch] = commit
            write_json(directory / "run.json", manifest)
        prs = command(["gh", "pr", "list", "--head", branch, "--base", "master", "--state", "open",
                       "--json", "url"], checkout, timeout=NETWORK_TIMEOUT)
        existing = json.loads(prs.stdout)
        if existing:
            url = existing[0]["url"]
        else:
            body = directory / f"state-pr-body-{branch.rsplit('/', 1)[-1]}.md"
            body.write_text(f"Updates {', '.join(f'`pipeline/state/{f}`' for f in merged)} from a "
                            f"local {pipeline} run.\n\nContains only pipeline bookkeeping. "
                            "Scientific proposals remain in separate PRs.\n")
            url = command(["gh", "pr", "create", "--base", "master", "--head", branch,
                           "--title", f"chore: update {pipeline} pipeline state",
                           "--body-file", str(body)], checkout, timeout=NETWORK_TIMEOUT).stdout.strip()
        manifest["state_prs"][branch] = url
        for filename, text in merged.items():
            manifest["baseline_state"][filename] = text
            if text != current[filename]:
                (checkout / "pipeline/state" / filename).write_text(text)
        write_json(directory / "run.json", manifest)
        # The merged result becomes the committed local baseline (only these
        # paths: staged science must never ride along).
        git(checkout, "commit", "-m", "Local pipeline state baseline (published)", "--",
            *[f"pipeline/state/{filename}" for filename in merged], check=False)
        print(url)


def normalise_backend(value):
    value = value.strip().lower()
    value = {"cli": "claude-cli"}.get(value, value)
    if value not in ("api", "claude-cli"):
        raise ValueError(f"unknown backend {value!r}; use api or claude-cli")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run", help="Start or reuse an isolated run")
    run.add_argument("pipeline", choices=PIPELINES)
    run.add_argument("--run-dir", type=Path, help="Reuse this directory to continue a run")
    run.add_argument("--source", type=Path, default=SOURCE)
    run.add_argument("--ref", default=None,
                     help="Committed source revision for a NEW run (default: remote master)")
    run.add_argument("--backend", default=os.environ.get("AAL_BACKEND", "claude-cli"))
    run.add_argument("--allow-unmerged-ref", action="store_true",
                     help="Allow a real run from a revision that is not on remote master")
    run.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                     help="Explicit child environment (inherited AAL_*/model overrides are dropped)")
    publish = subparsers.add_parser("publish-state", help="Merge owned state onto its branch, push with a lease, open/reuse its PR")
    publish.add_argument("run_dir", type=Path)
    diff = subparsers.add_parser("state-diff", help="Show owned state changes not yet published")
    diff.add_argument("run_dir", type=Path)
    # Everything after -- belongs to the existing pipeline parser, unmodified.
    values = list(sys.argv[1:] if argv is None else argv)
    divider = values.index("--") if "--" in values else len(values)
    args = parser.parse_args(values[:divider])
    forwarded = values[divider + 1:]
    if args.operation != "run" and forwarded:
        parser.error("Only run accepts pipeline arguments after --")
    if args.operation == "run":
        try:
            args.backend = normalise_backend(args.backend)
        except ValueError as error:
            parser.error(str(error))
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
            if args.operation == "state-diff":
                state_diff(directory)
                return 0
            manifest = prepare(directory, args.pipeline, args.source.resolve(), args.ref, args.backend)
            return execute(directory, manifest, forwarded, env_overrides=args.env,
                           allow_unmerged_ref=args.allow_unmerged_ref)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Local runner stopped: {error}\nRecovery directory: {directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
