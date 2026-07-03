# Final-benchmark tooling (handover artifacts, 2026-07-03)

- `extract_driver.py` — parallel extraction into an isolated snapshot dir
  (skip-existing resume; hard-abort on billing errors; old-style-id-safe
  filenames). Run with `EXTRACTOR_MODEL=claude-haiku-4-5-20251001`,
  `AAL_READ_SAMPLES=3`, `AAL_BATCH=1` (Message-Batches transport, 50% price)
  and `--workers 100`.
- `score_snapshots.py` — scores a snapshot dir with the OFFICIAL
  evaluate.py metrics path by redirecting RESULTS_DIR (the old baseline in
  evaluation/results/ stays untouched); edit the FINAL path constant.
- `reuse_ids_provably_unchanged.txt` — the 65 papers whose old
  evaluation/results snapshots are provably drawn from the current
  pipeline's distribution (text/table winner, >3 points, strict
  VALID_RANGES, no source-data candidate, no vector candidate, no
  wrong-curve gate fires). Copy their old snapshots into the outdir before
  launching, so the driver skips them.

Plan + incident context: the project memory handover and
evaluation/eval_runs/final_benchmark_PARTIAL.md (retraction).
