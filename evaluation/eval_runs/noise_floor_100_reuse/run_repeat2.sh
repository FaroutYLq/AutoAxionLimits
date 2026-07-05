#!/usr/bin/env bash
# Noise-floor repeat-2: fresh N=1 re-read of the frozen 100 ids, one arm per model.
# Transport MUST match final2 (repeat-1) == DIRECT (no AAL_BATCH), sequential caller thread.
# Model asserted in-driver (extract_driver.py silent-Opus guard, #677).
#
# Usage:  ./run_repeat2.sh opus     |     ./run_repeat2.sh haiku
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ARM="${1:?usage: run_repeat2.sh <opus|haiku>}"
case "$ARM" in
  opus)  MODEL=claude-opus-4-8 ;;
  haiku) MODEL=claude-haiku-4-5-20251001 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

BASE=evaluation/eval_runs/noise_floor_100_reuse
OUT="$BASE/${ARM}_repeat2"
mkdir -p "$OUT"

# transport-match final2: DIRECT (unset AAL_BATCH), N=1
unset AAL_BATCH || true
export AAL_READ_SAMPLES=1
export EXTRACTOR_MODEL="$MODEL"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(security find-generic-password -s anthropic_api_key -w)}"

# code-match guard: extractor path must be at #684 (a4a6216f) == final2 code.
EXTR_SHA=$(git log -1 --format=%h -- pipeline/extractor.py)
echo "extractor.py HEAD commit: $EXTR_SHA (expect a4a6216f == final2 code)"
[ "$EXTR_SHA" = "a4a6216f" ] || { echo "WARN: extractor.py changed since final2 ($EXTR_SHA != a4a6216f) — pairing may straddle a code change"; }

IDS=$(python3 -c "import json;print(','.join(json.load(open('$BASE/frozen_ids.json'))['ids']))")

echo "[$ARM] re-reading 100 frozen ids, model=$MODEL, DIRECT transport, N=1 -> $OUT"
python3 evaluation/benchmark/extract_driver.py \
  --worktree "$PWD" \
  --outdir "$OUT" \
  --ids "$IDS" \
  --workers 8 \
  2>&1 | tee -a "$BASE/${ARM}_repeat2.log"

echo "[$ARM] scoring (noproj scope, matches numbers.json) ..."
AAL_RESULTS_DIR="$OUT" AAL_EXCLUDE_PROJECTIONS=1 python3 -m evaluation.evaluate --metrics \
  2>&1 | tee -a "$BASE/${ARM}_repeat2.log"

echo "[$ARM] done. metrics at $OUT/metrics_noproj.json"
