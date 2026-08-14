# Noise-floor measurement (GitHub issue #701)

Directly measured, CI-bearing run-to-run noise floor for the extraction pipeline,
replacing the manuscript's prior n=2 / 30-paper-probe estimate ("up to ~0.07 dex").

**Status: FABLE (production) arm COMPLETE, n=96 paired, 2026-08-05.** Opus arm
remains Plan B (partial n=57: a planned 100-paper re-read was truncated at 58/100
by API credit exhaustion, clean #648 fail-fast, 0 error stubs). Haiku arm deferred
(user decision), still resumable with zero re-spend (`run_repeat2.sh haiku`).

The **production floor is now measured directly** rather than inherited from Opus:

| arm | n paired | aggregate floor 95% CI | half-width | per-paper std (median) | channel flips |
|---|---|---|---|---|---|
| **fable (production)** | **96** | **[0.161, 0.205]** | **±0.022 dex** | 0.017 dex | 12/96 = 12.5% |
| fable, restricted to Opus's 56 | 56 | -- | ±0.036 dex | -- | -- |
| opus 4.8 | 56 | [0.160, 0.261] | ±0.050 dex | 0.022 dex | 10/56 = 17.9% |

The Fable arm is more reproducible than Opus on **identical papers** (±0.036 vs
±0.050 on the same 56), so the improvement is not a sample-size artifact. The
Opus paired set is a strict subset of the Fable one, which makes the restriction
exact rather than approximate.

### CAVEAT: the bootstrap CI is NOT the whole-run shift
The read-selection bootstrap mixes reads per paper, but a real run is all-repeat-1
or all-repeat-2. The realized whole-run pool medians are **0.164 (r1) vs 0.199 (r2)
= 0.035 dex**, which is LARGER than the ±0.022 bootstrap half-width. The Opus arm
hid this because its two medians were identical (realized shift 0.000). At n=2
reads the whole-run median's sampling distribution is not estimable; quote the
bootstrap CI and the realized shift together, never the CI alone.

## Design (reuse: benchmark run == "repeat 1")

**Branch A** (reuse valid). The severe DM-density bug fix (#702/#704) lives in the
**reviewer/plot layer** (`reviewer.py`, `config.py`, `PlotFuncs.py`); the **extraction
path is untouched** — last change to `extractor.py`/`vision_gates.py`/`read_vote.py`/
`selector.py` is #684 (`a4a6216f`), predating `final2`. The eval scores **raw
extraction** (`run_extraction_agent_voted`; density is never applied on the eval path).
So `final2` is still code-matched to the current extractor and can serve as repeat-1.

- **repeat 1** = `final2_opus_n1/` — the definitive benchmark run itself (cached, N=1).
- **repeat 2** = `opus_repeat2/` — a fresh, matched-config N=1 re-read.
- Per-paper n=2 → the noise floor. Disclosed in the paper: repeat 1 *is* the benchmark
  run; repeat 2 is a fresh matched re-run (a standard repeatability design).

### Provenance match (all three re-verify checks passed)
| Axis | Match |
|---|---|
| Code | extractor path frozen at #684 (`a4a6216f`) == final2; bug fix was reviewer/plot-only; eval scores raw extraction |
| Transport | **DIRECT** (final2 log has 0 batch-dispatch lines, sequential per-paper timing); re-read run with `AAL_BATCH` unset |
| Model + N | `EXTRACTOR_MODEL=claude-opus-4-8`, `AAL_READ_SAMPLES=1`; `extract_driver.py` asserts the resolved model (silent-Opus guard #677) |

## Paper set (frozen)

- `frozen_ids.json`: **seed 701**, 100 ids drawn from the **measured-limits comparable
  intersection** (papers with a finite `interp_metrics.median_residual_dex` in BOTH
  `final2` arms → 212-paper pool; projections already excluded by `metrics_noproj`).
- Random selection is deliberate — it removes the vision-stratification upper-bound bias
  of the original `nprobe` estimate.
- **Plan-B subset**: the 58 processed are the first-processed (lowest-arXiv-id) prefix of
  the frozen 100, i.e. skewed toward older papers. The **absolute** median on the subset
  (0.218 dex) is therefore below the frozen-100 repeat-1 median (0.259 dex) and the full
  headline (0.251 dex). This does **not** bias the *floor*: the reported quantity is how
  much the median *moves*, and the subset preserves the full pool's ~39% figure-vision
  fraction (22/58 = 38%), so the routing-flip noise source is fully sampled.

## Results (Opus, N=1, DIRECT, n=57 paired)

Per-paper metric = `metrics_noproj.json → per_paper[].interp_metrics.median_residual_dex`
(the same quantity behind `paper/numbers.json`). Analysis: `analyze.py`
(deterministic, bootstrap seed 701, 20000 iters). Raw output: `noise_floor_results.json`.

**Deliverable 1 — per-paper noise (how much one paper wobbles).**
std at n=2 = `|Δ|/√2`. Report the **median** (robust); the mean (0.32 dex) is dominated
by the routing-flip tail and should not be cited.
- **median per-paper std = 0.024 dex** (median |Δ| = 0.034 dex)
- histogram is **bimodal** — a stable core + a routing-flip tail:

  | per-paper std (dex) | <0.02 | 0.02–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.5 | 0.5–1 | 1–2 | ≥2 |
  |---|---|---|---|---|---|---|---|---|
  | papers | 27 | 5 | 3 | 5 | 7 | 6 | 2 | 2 |

**Deliverable 2 — aggregate-median floor (headline guard; how much the reported median
moves).** Read-selection bootstrap: resample which of the 2 reads wins per paper,
recompute the pool median, take percentiles. **This is the ONLY floor placed next to the
reported medians / the paired delta.**
- pool median (repeat 1) = pool median (repeat 2) = **0.218 dex** (realized shift 0.000 dex)
- bootstrap median = 0.218 dex, **95% CI [0.159, 0.276]**, **half-width ±0.059 dex**

**Deliverable 3 — paired-delta noise (haiku − opus):** NOT AVAILABLE in Plan B (Haiku
arm deferred). Requires the Haiku re-read.

### Mechanism (directly observed)
- **11 of 57 paired papers (19%) flip channel** (text↔vision) between runs.
- These flips populate the tail: e.g. 1903.12190 (text→vision, 7.82→1.66 dex), 2106.00022
  (text→vision, 5.23→0.62), 1503.06886 (vision→text, 1.95→0.24). A few same-channel
  numerical wobbles also appear (2111.08025 text→text, 2.15→0.16).

## Interpretation for the manuscript
- **Aggregate floor ≈ ±0.06 dex (95% CI half-width).** Single-run aggregate deltas below
  this are not signal. Consistent with the old "up to ~0.07 dex" estimate, but now
  CI-bearing and measured on a random (representative) set.
- Keep the two floors **distinctly labeled**: the per-paper 0.02 dex ("wobble") must NOT
  be placed next to the ~0.2–0.3-class headline as "the floor" — that revives the
  "headline sits at the noise floor" misreading.

## Fable arm provenance (2026-08-05)
- repeat-1 = `final347_fable` (the definitive production benchmark arm itself), so
  the pairing is code-matched (both at pinned worktree `92820cdc`) and
  transport-matched (both `AAL_BACKEND=claude-cli`, N=1, 2 workers) **by
  construction** — a stronger provenance claim than the Opus arm, which had to
  argue a code match across `final2` and a separate driver run.
- Scored from the MAIN repo (current scorer, #744/#745), matching how repeat-1's
  `metrics_noproj.json` was produced. Scoring with the pinned worktree's older
  scorer would compare scorers, not reads.
- **Collected in two batches.** The first pass banked 72/100 and then failed on
  every remaining paper in 0-2s. Cause was NOT model flakiness and NOT arXiv rate
  limiting (both were hypothesised and both were wrong): `evaluation/results/
  metadata_cache.json` had one malformed entry (`2504.07559`, unescaped quote in
  the abstract), and `_fetch_paper_metadata` loads it with an unguarded
  `json.load` that runs before every extraction — so one bad entry killed every
  subsequent paper with a misleading JSON error. Cache repaired by dropping that
  entry (backup: `metadata_cache.json.corrupt.bak`); the remaining 28 then
  extracted clean (0 errors, 0 husks). Extraction code stayed pinned throughout;
  only a data file changed, so the pairing is still code-matched.
- Two follow-ups this exposed, both open: (1) guard the cache load and write it
  atomically/under a lock — concurrent workers do an unsynchronised
  read-modify-write on it; (2) husk predicates MUST exclude records carrying
  `status == "error"`, or genuine errors get silently delete-and-retried forever
  (this cost ~3h of no-op attempts here).

## Files
- `frozen_ids.json` — seed 701, the 100 ids (58 processed in Plan B)
- `opus_repeat2/` — 58 fresh N=1 Opus reads + `metrics_noproj.json`
- `run_repeat2.sh <opus|haiku>` — resumable runner (direct transport, model assert, auto-score)
- `analyze.py` — three deliverables + bootstrap CI
- `noise_floor_results.json` — machine-readable results

## To finish (once credits topped up)
```bash
./evaluation/eval_runs/noise_floor_100_reuse/run_repeat2.sh opus    # 42 remaining, skips 58
./evaluation/eval_runs/noise_floor_100_reuse/run_repeat2.sh haiku   # 100 fresh (deliverable 3)
python3 evaluation/eval_runs/noise_floor_100_reuse/analyze.py
```

## Rescoring note (2026-08-14)
All metrics in this directory are scored with the repository's CURRENT scorer
(post #737/#738/#741/#743/#744/#745), so a fresh clone regenerates them
bit-identically. Under that scorer the Plan-B Opus arm pairs 56 papers (one
paper of the original 57 left the comparable pool) and its floor is ±0.050 dex
with a 10/56 = 17.9% flip rate; the values previously quoted from the July
scorer were 57 paired / ±0.059 / 19.3%. No production (fable) number changed.
