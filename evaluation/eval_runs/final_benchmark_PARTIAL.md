# Final full-scale benchmark — PARTIAL (RETRACTED: model-confounded)

> **RETRACTION (2026-07-03, silent-Opus incident):** every number below
> compared old-code snapshots (Haiku) against new-code snapshots that
> SILENTLY ran claude-opus-4-8 — the EXTRACTOR_MODEL override did not exist
> on master (fixed in #677). The comparison is model-confounded and must not
> be cited. The Opus snapshots are preserved as a tier-calibration artifact
> in `evaluation/eval_runs/opus_newcode_sample/` (see its README). The
> protocol-clean Haiku benchmark plan: 65 provably-unchanged old snapshots
> reused (tightened criterion: non-sparse in-range text/table winners, no
> channel candidates, no gate fires) + 281 papers re-run at true Haiku N=3
> via the Batches transport (AAL_BATCH=1), est. \$12-18. The original
> partial-run text follows for the record.


Run 2026-07-03 on master @ 5b3a4fce (all extraction-channels work merged:
gates #663/#667/#669, source_data #665/#670, caching #668, vector_trace
#671/#672, GT fix #673). `EXTRACTOR_MODEL=claude-haiku-4-5-20251001`,
`AAL_READ_SAMPLES=3`, prompt caching active, isolated outdir
`evaluation/eval_runs/final_full346/`.

**The API credit balance ran out mid-run**: 58 papers extracted for real
before billing-400s began; the remaining 290 papers produced error stubs
(deleted — the outdir is resume-ready). The old full346 baseline in
`evaluation/results/` is untouched.

## What the partial slice shows (49 scoreable papers, paired vs old baseline)

| | old full346 (same 49 ids) | final partial |
|---|---|---|
| micro median residual | 0.210 dex | **0.206 dex** |
| scored | 37 | 36 |
| zero-overlap (partial set) | — | **0** |
| sources | 30 vision / 17 text / 1 table | 29 vision / 13 text / **4 vector_trace** |

Parity on aggregate (this slice is dominated by papers the pipeline already
handled well), with the new channels visibly active. The channel-level wins
were already demonstrated in the targeted validations (see
`gates_ab_report.md`, the #670 and #672 PR bodies): the rescues live in the
failure tail, which this alphabetical-prefix slice under-samples.

## Cost lesson (important for the re-run)

The observed burn implies ~$0.5–1.0 per paper at N=3 with the image-grounded
vector-select — several times my pre-run estimate (which under-counted the
8-figure vision payloads across 3 votes). For the complete re-run of the
remaining ~290 papers, either:

- **N=1**: ~$15–25, no vote-denoising (acceptable for a before/after paired
  read; the old baseline is N=3 though), or
- **N=3**: budget **$50–90**, or
- do the **stage-wise Batches refactor first** (50% off; medium effort).

The extraction driver now hard-aborts on the first billing/availability
error instead of burning the id list into stubs.

## Exact resume path (when credits are added)

```
export ANTHROPIC_API_KEY=... EXTRACTOR_MODEL=claude-haiku-4-5-20251001 AAL_READ_SAMPLES=3
python <scratchpad>/val_extract.py --worktree <repo> \
  --outdir evaluation/eval_runs/final_full346 --ids <all 346, comma-sep> --workers 6
# existing snapshots are skipped; then score with <scratchpad>/final_score.py
```
