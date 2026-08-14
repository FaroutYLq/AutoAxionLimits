#!/usr/bin/env bash
# Noise-floor repeat-2: fresh N=1 re-read of the frozen 100 ids, one arm per model.
# Transport MUST match that arm's repeat-1 (the benchmark run it is paired against).
# Model asserted in-driver (extract_driver.py silent-Opus guard, #677).
#
#   ./run_repeat2.sh opus     repeat-1 = final2_opus_n1     API transport
#   ./run_repeat2.sh haiku    repeat-1 = final2_haiku_n1    API transport
#   ./run_repeat2.sh fable    repeat-1 = final347_fable     claude-cli (subscription)
#
# The fable arm runs on the pinned benchmark worktree (~/.aal_bench/worktree at
# 92820cdc), which is the exact code final347_fable was extracted with, and on the
# subscription backend, which is the transport it was extracted through. Do not
# run it from the main repo checkout: master has moved past the pin (#742/#743).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
REPO="$PWD"

ARM="${1:?usage: run_repeat2.sh <opus|haiku|fable>}"
WORKTREE="$REPO"
WORKERS=8
BACKEND=api
case "$ARM" in
  opus)  MODEL=claude-opus-4-8 ;             EXPECT_SHA=a4a6216f ;;
  haiku) MODEL=claude-haiku-4-5-20251001 ;   EXPECT_SHA=a4a6216f ;;
  # repeat-1 for fable is final347_fable: pinned worktree, CLI backend, 2 workers.
  fable) MODEL=claude-fable-5 ;              EXPECT_SHA=92820cdc
         WORKTREE="$HOME/.aal_bench/worktree"; WORKERS=2; BACKEND=claude-cli ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

BASE=evaluation/eval_runs/noise_floor_100_reuse
OUT="$REPO/$BASE/${ARM}_repeat2"
mkdir -p "$OUT"

# transport-match repeat-1: N=1, never batch
unset AAL_BATCH || true
export AAL_READ_SAMPLES=1
export EXTRACTOR_MODEL="$MODEL"
export AAL_BACKEND="$BACKEND"
if [ "$BACKEND" = "claude-cli" ]; then
  # subscription path: no API key may leak into the run (it would silently bill
  # and change the transport out from under the pairing).
  unset ANTHROPIC_API_KEY || true
else
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(security find-generic-password -s anthropic_api_key -w)}"
fi

# code-match guard: the extractor path must sit where repeat-1 was extracted.
EXTR_SHA=$(git -C "$WORKTREE" log -1 --format=%h -- pipeline/extractor.py)
echo "[$ARM] extractor.py commit in $WORKTREE: $EXTR_SHA (expect $EXPECT_SHA)"
[ "$EXTR_SHA" = "$EXPECT_SHA" ] || { echo "WARN: extractor.py at $EXTR_SHA != $EXPECT_SHA — pairing may straddle a code change"; }

IDS=$(python3 -c "import json;print(','.join(json.load(open('$REPO/$BASE/frozen_ids.json'))['ids']))")

echo "[$ARM] re-reading 100 frozen ids, model=$MODEL, backend=$BACKEND, N=1, workers=$WORKERS -> $OUT"
python3 "$REPO/evaluation/benchmark/extract_driver.py" \
  --worktree "$WORKTREE" \
  --outdir "$OUT" \
  --ids "$IDS" \
  --workers "$WORKERS" \
  2>&1 | tee -a "$REPO/$BASE/${ARM}_repeat2.log"

# Score from the MAIN repo, not the pinned worktree: repeat-1's metrics_noproj.json
# was produced by the current scorer (#744 best-match multi-GT, #745), so repeat-2
# must be scored by the same one or the pairing compares scorers, not reads.
echo "[$ARM] scoring (noproj scope, current scorer, matches repeat-1) ..."
cd "$REPO"
AAL_RESULTS_DIR="$OUT" AAL_EXCLUDE_PROJECTIONS=1 python3 -m evaluation.evaluate --metrics \
  2>&1 | tee -a "$REPO/$BASE/${ARM}_repeat2.log"

echo "[$ARM] done. metrics at $OUT/metrics_noproj.json"
