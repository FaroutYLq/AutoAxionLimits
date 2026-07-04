# Final full-pool Haiku benchmark (346 papers) — complete, model-confounded headline

**TL;DR:** The protocol-clean full-pool benchmark completed cleanly (346/346 papers, 0 errors, ~$57). The headline delta vs the 0.245 dex baseline is **model-confounded** — the baseline is Opus 4.8 + old code, this run is Haiku + new code — and a controlled model-isolation experiment on identical new code shows Haiku alone costs **+0.95 dex** vs Opus, dwarfing any plausible code effect. The new extraction channels fire correctly in vivo (19 vector_trace, 2 source_data). A free **fixed-model (Opus) code A/B** closes the loop: at constant model the extraction-channels code does **not** regress (−0.016 dex) and recovers +4 papers of coverage — so the regression is the model, not the code. A companion causal digest (`haiku_regression_causal_digest.md`) stress-tests this and lands at high confidence on direction, moderate on the full-pool magnitude.

Run 2026-07-03/04 on master (all extraction-channels work merged, PRs
#663–#680: wrong-curve gates, source-data channel tier 5, vector-trace
channel tier 3.5, prompt caching, `EXTRACTOR_MODEL` env override #677,
all-Haiku workflows, Message-Batches transport shim `AAL_BATCH=1`).

## Run provenance & configuration

| | |
|---|---|
| Code | master (post #663–#680) |
| Model | `EXTRACTOR_MODEL=claude-haiku-4-5-20251001` |
| Consensus | `AAL_READ_SAMPLES=3` (3-vote) |
| Transport | `AAL_BATCH=1` (Message-Batches shim), 100 workers |
| Outdir | `evaluation/eval_runs/final_full346_haiku/` (isolated) |
| Design | 65 "provably-unchanged" old snapshots reused (copied in, skipped by driver) + 281 papers re-run on true Haiku |
| Pool | 346 papers / 433 ground-truth channel-entries |

The reuse criterion is the tightened one from `final_benchmark_PARTIAL.md`:
non-sparse in-range text/table winners with no channel candidates and no
gate fires — snapshots whose output the new code provably could not change.

## Central finding: the headline comparison is model-confounded

The intended comparison was FINAL (Haiku, new code) vs the standing
baseline in `evaluation/results/metrics.json` (micro-median 0.245 dex).
But that baseline is **Opus 4.8 + old code**: at the baseline commit
b66f2cfe (2026-06-08), `CLAUDE_MODEL = "claude-opus-4-8"` was hardcoded,
and the `EXTRACTOR_MODEL` override only landed in #677 (2026-07-03).

Comparing 0.245 → FINAL therefore changes **two variables at once**:
old→new code AND Opus→Haiku model. This is the same class of confound
that retracted the partial benchmark (`final_benchmark_PARTIAL.md`).
The headline delta **must not be reported as "the new code regressed"** —
the model-isolation experiment below attributes essentially all of it to
the model swap.

## Headline metrics

| metric | baseline (Opus 4.8, old code) | FINAL (Haiku + 65 Opus-reused, new code) |
|---|---|---|
| micro-median residual | 0.245 dex | 0.604 dex |
| macro-median residual | 1.58 dex | 3.74 dex |
| n_papers_compared | 267 | 243 |
| zero-overlap | 10 | 23 |
| coupling_type accuracy | 0.909 | 0.888 |
| is_new_limit accuracy | 0.886 | 0.914 |
| is_projection accuracy | 0.971 | 1.000 |
| data_source accuracy | 0.629 | 0.371 |

Both columns are mixtures of confounded variables; see the decomposition
and isolation experiment below before reading anything into the deltas.
(The `data_source` drop is expected in part by construction: the new code
routes to channels that did not exist in the old GT labeling.)

## Sub-slice decomposition of the FINAL run

The mixed 0.604 decomposes cleanly along the reuse boundary:

| slice | micro-median | n compared | interpretation |
|---|---|---|---|
| 65 reused (Opus text snapshots) | 0.259 dex | 56 | matches the ~0.245 baseline — reuse mechanism sanity check ✓ |
| 281 re-run (pure Haiku, new code) | **0.975 dex** | 175 | the true pure-Haiku-new-code operating point, ~4× the Opus baseline |
| full 346 (mixed) | 0.604 dex | 243 | headline; diluted downward by the Opus-quality reused slice |

The reused slice landing at 0.259 dex confirms the copy-and-skip
mechanism worked and that those snapshots are near-baseline quality;
they pull the mixed average down. **0.975 dex is the honest pure-Haiku
number.**

## Model-isolation experiment (the decisive test)

The quarantined `evaluation/eval_runs/opus_newcode_sample/` (verified per
its README: the **same new code**, master @ 5b3a4fce, silently run on
claude-opus-4-8 at N=3 — the silent-Opus incident) provides a same-code,
different-model pair. Scoring both runs on their **77 shared papers**
(identical code, identical papers, only the model differs):

| model (identical new code, 77 shared papers) | micro-median | n compared |
|---|---|---|
| Opus 4.8 | 0.353 dex | 51 |
| Haiku 4.5 | 1.307 dex | 42 |
| **model effect (Haiku − Opus)** | **+0.954 dex** | Haiku also yields fewer comparable extractions (42 vs 51) |

Caveat: the 77-paper shared set over-samples the hard failure tail, so
the *absolute* numbers are higher than the full pool; the **+0.95 dex
delta** is the clean signal. It dwarfs any plausible code effect and
accounts for essentially the entire headline gap.

## Opus-only code A/B (added) — the complementary control that DOES grade the code

The model-isolation test above varies the model at fixed (new) code. The
inverse control varies the **code at fixed model** (Opus 4.8): the same
`opus_newcode_sample` (Opus, NEW code) vs the standing baseline (Opus, OLD
code), on their 77 paired papers. This is free — both snapshot sets already
exist — and it isolates the extraction-channels **code** effect.

| Opus 4.8, 77 paired papers | micro-median | n compared |
|---|---|---|
| OLD code (baseline) | 0.369 dex | 47 |
| NEW code (channels) | 0.353 dex | 51 |
| **code effect (new − old)** | **−0.016 dex** (slight improvement) | **+4 coverage** |

Paired per-paper: 21 improved, 12 regressed, 13 ~same; coverage recovered on
5 papers (4 via text, 1 via figure_vision/41 pts), lost on 1 (1512.06165 — a
projection/proposal paper the new code correctly declined with 0 points). So
at fixed model the new code does **not** regress — it slightly helps and nets
+4 papers of coverage. **The regression appears only when the model changes,
never when the code changes.** Combined with the model-isolation table, this
is a clean 2-factor decomposition: code ≈ −0.02 dex, model ≈ +0.95 dex.

A dedicated causal digest stress-tests this attribution against seven
alternative explanations (code drift, parsing bugs, scoring asymmetry, reuse
contamination — all ruled out; failure-tail sampling, N=3 noise, transport —
limits on magnitude only): `haiku_regression_causal_digest.md`. Verdict: high
confidence that the model swap caused the regression and that the code did not;
moderate confidence on the exact +0.95 dex magnitude generalizing to the full
pool (the shared-77 set over-samples the hard tail).

## Mechanism: Haiku's vision-JSON failures

The run log contains **208 stage-2 "No JSON found in response" fallback
failures** — Haiku frequently returned malformed, fence-wrapped, or
truncated JSON on vision (plot-reading) calls, forcing fallbacks to
lower-tier channels. This is a concrete mechanism behind the ~1 dex
vision penalty. The run also produced 15 zero-point extractions (no data
recovered).

## Channel activity: the new channels fire in vivo

`data_source` mix across all 346 final snapshots:

| source | count |
|---|---|
| text | 172 |
| figure_vision | 135 |
| vector_trace (tier 3.5, new) | **19** |
| none | 10 |
| table | 6 |
| figure | 2 |
| source_data (tier 5, new) | **2** |

The channel routing works mechanically — 19 papers won via vector-trace
and 2 via source-data in a real full-pool run. What this run cannot do is
quantify the channels' *accuracy* benefit, because of the model confound.

## Run health & cost

- **346/346 papers completed; 0 error snapshots, 0 timeouts, 0
  tracebacks** — pristine.
- Wall-clock ~6.0 h (relaunch → ALL DONE), plus ~1 h on a first attempt
  that stalled (see transport note below).
- Cost: **~$56.71 total** — $12.59 batch-metered text stages (50% batch
  discount) + ~$44.12 estimated for 1103 direct vision calls at standard
  price. Far above the handover's $12–18 estimate: the estimate assumed
  vision would ride the batch discount, but the vision calls had to be
  moved off the batch transport.

## Transport stall & fix (methods note)

The first attempt stalled at batch 3: a wave of vision calls (base64 page
images, up to 40 MB per item) serialized through the batch shim's single
dispatcher connection at ~100 KB/s and died on the request timeout,
looping on retries and failing every in-flight paper.

Fix (commit 2b23357d, branch `fix/batch-vision-bypass`, transport-only):
items whose serialized params exceed 3 MB (i.e. vision calls) bypass the
batch and go as plain `messages.create` on the caller thread — standard
price, parallel upload, the proven pre-batch transport. Batches are
capped at 25 MB. Consequence: vision calls no longer get the 50% batch
discount, which is why the run cost ~$57 rather than ~$18. All 7
batch-shim unit tests pass, including 2 new ones. Params are forwarded
verbatim, so the extraction distribution is unchanged by the fix.

## What can and cannot be concluded

**CAN conclude:**

1. On identical new code, Haiku is ~0.95 dex worse than Opus at
   extraction and yields fewer comparable extractions (42 vs 51 on the
   shared set). Haiku is not viable at the quality bar the Opus baseline
   set, primarily due to vision/plot-reading (208 JSON failures).
2. The new channels route correctly in vivo (19 vector_trace +
   2 source_data winners across 346 real papers).
3. The pipeline runs clean end-to-end at scale (346/346, 0 errors) on
   the batch+direct-bypass transport.

**CAN NOW conclude (via the free Opus-only code A/B above):** at fixed
Opus model, the extraction-channels code does **not** regress accuracy
(−0.016 dex) and **recovers +4 papers of coverage** on the 77-paper
sample. The 0.245→0.604 headline is therefore a model effect, not a code
regression. The `opus_newcode_sample` (Opus, OLD-vs-NEW code exists as
baseline-vs-sample) supplied the same-model pair after all.

**Still CANNOT conclude:** the *full-pool* magnitude of either effect. The
code A/B and the model isolation both rest on the 77-paper sample, which
over-samples the failure tail. Pinning the full-346 numbers needs a
full-pool Opus (new code) run — the one definitive test in the digest.

## Recommended next steps (options, not decisions)

1. **To grade the code cleanly:** run a same-model A/B — either
   re-baseline the old code on Haiku (cheap-ish), or run the new code on
   Opus for the full pool (the opus_newcode_sample covers only 77 shared
   papers).
2. **If Haiku is the production target for cost:** accept that extraction
   quality drops ~1 dex vs Opus and treat **0.975 dex** (pure Haiku, new
   code) as the Haiku-tier operating point. The channels still help, but
   the model floor dominates.
3. The 0.245 dex Opus baseline remains the valid Opus-tier reference —
   do not overwrite it.

## Performance impact

- **Before (baseline):** 0.245 dex micro-median — Opus 4.8, **old** code
  (`evaluation/results/metrics.json`, commit b66f2cfe).
- **After (this run):** 0.604 dex mixed / **0.975 dex pure-Haiku** —
  Haiku 4.5, **new** code.
- **Attribution:** the delta is **model-confounded**. The controlled
  same-code comparison measures the Opus→Haiku swap at **≈ +0.95 dex**
  on identical code and papers, accounting for essentially the entire
  gap. **This is NOT a code regression** — the extraction-channels code
  must not be reported as having hurt accuracy. Isolating the code
  effect requires a same-model A/B (option 1 above).
