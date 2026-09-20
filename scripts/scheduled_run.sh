#!/usr/bin/env bash
# Unattended AutoAxionLimits run on the Claude Code subscription (launchd).
#
#   scripts/scheduled_run.sh daily|weekly [pipeline args...]
#
# Runs remote master through the isolated local runner (scripts/run_pipeline.sh)
# with the claude-cli backend, then publishes the run's owned state files to
# their lease-protected chore branch and PR (publish-state). Science PRs are
# opened by the pipeline as usual and are never merged here. Nothing prompts:
# the runner is non-interactive and the subscription session needs no API key.
#
# Exit code is the pipeline's (2 = availability failure such as an exhausted
# subscription window: state of the papers finished before it is still
# published; 3 = publication failure inside the run: the affected paper is
# left eligible for retry). A --dry-run publishes nothing.
set -uo pipefail

# Everything runs inside main() so bash parses the whole file before executing
# it: the run fast-forwards this checkout (and therefore this very file) to
# origin/master, and bash otherwise reads a script incrementally while it runs.
main() {
    KIND=${1:?usage: scheduled_run.sh daily|weekly [pipeline args...]}
    shift
    # /Library/TeX/texbin first: the notebooks render text with usetex + Palatino,
    # and a launchd job has no shell PATH (first real run 2026-09-19: plot
    # regeneration failed with "pplr7t.tfm not found" and the science PR showed
    # the stale committed plot).
    export PATH="/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:/opt/anaconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    export CONDA_EXE="${CONDA_EXE:-/opt/anaconda3/bin/conda}"
    # Subscription billing only: a stray key would silently flip the session to
    # API billing; CLAUDECODE would make the CLI think it is nested.
    unset CLAUDECODE ANTHROPIC_API_KEY
    # pydantic otherwise scans every dist-info entry_points.txt in the conda
    # env at import time; the 2026-09-20 05:00 run died there with
    # "OSError: [Errno 11] Resource deadlock avoided" (transient filesystem
    # state at wake-up). No pydantic plugins are used, so skip the scan.
    export PYDANTIC_DISABLE_PLUGINS=__all__
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    LOGDIR="${AAL_SCHED_LOGS:-$HOME/.aal_bench/scheduler/logs}"
    mkdir -p "$LOGDIR"
    exec >>"$LOGDIR/$(date +%F)-$KIND.log" 2>&1

    echo "== $(date '+%F %T') $KIND start (launcher $ROOT)"
    # The runner executes remote master regardless of the source checkout; this
    # only keeps the launcher itself (this script, run_pipeline.sh) current.
    if git -C "$ROOT" fetch -q origin master && git -C "$ROOT" merge -q --ff-only origin/master; then
        echo "== launcher at $(git -C "$ROOT" rev-parse --short HEAD)"
    else
        echo "== warn: could not fast-forward $ROOT to origin/master; using its current files"
    fi

    # One retry, only for a run that failed before doing anything (exit 1 with
    # no state change and no PR): that is an environment hiccup (import error,
    # arXiv down at wake-up), not a paper. Exit 2/3 and partial runs are never
    # retried; their state is published below as usual.
    for attempt in 1 2; do
        OUT=$(mktemp)
        bash "$ROOT/scripts/run_pipeline.sh" run "$KIND" --backend claude-cli -- "$@" 2>&1 | tee "$OUT"
        rc=${PIPESTATUS[0]}
        RUNDIR=$(sed -n 's/^Run directory: //p' "$OUT" | head -1)
        empty_fail=0
        if [[ "$rc" == 1 ]] && grep -q "changed state: \[\]; PRs: \[\]" "$OUT"; then
            empty_fail=1
        fi
        rm -f "$OUT"
        echo "== run exit $rc, run dir ${RUNDIR:-<none>} (attempt $attempt)"
        if [[ "$empty_fail" == 1 && "$attempt" == 1 ]]; then
            echo "== run failed before changing anything; retrying in ${AAL_SCHED_RETRY_DELAY:-120}s"
            sleep "${AAL_SCHED_RETRY_DELAY:-120}"
            continue
        fi
        break
    done

    for a in "$@"; do
        if [[ "$a" == "--dry-run" ]]; then
            echo "== preview run: state not published"
            exit "$rc"
        fi
    done
    if [[ -z "$RUNDIR" ]]; then
        echo "== no run directory was created; nothing to publish"
        exit "$rc"
    fi
    case "$rc" in
        2) echo "== availability failure; publishing the state of papers finished before it" ;;
        3) echo "== publication failure inside the run; the affected paper stays eligible, publishing the rest" ;;
    esac
    echo "== owned-state diff:"
    bash "$ROOT/scripts/run_pipeline.sh" state-diff "$RUNDIR"
    bash "$ROOT/scripts/run_pipeline.sh" publish-state "$RUNDIR"
    prc=$?
    echo "== publish-state exit $prc"
    echo "== $(date '+%F %T') $KIND done"
    exit "$rc"
}

main "$@"
