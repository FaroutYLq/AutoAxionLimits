# Causal-analysis digest: did the Opus→Haiku swap cause the extraction regression?

**TL;DR verdict: Yes, with high confidence on direction and mechanism.** The
model swap is isolated by a genuine same-code, same-papers, same-scorer
comparison (+0.954 dex on 77 shared papers), the complementary fixed-model
code A/B shows the new code does *not* regress (−0.016 dex on Opus), and a
concrete Haiku-specific failure mode (208 vision-JSON parse failures) supplies
the mechanism. What remains open is the exact **magnitude** on the full pool —
the isolation sample over-weights the failure tail, each arm is a single N=3
run, and the transport differed between arms — so ≈+0.95 dex is the
best point estimate, not a settled constant.

## The claim

The final full-pool Haiku benchmark (0.604 dex mixed / 0.975 dex pure-Haiku
micro-median) is far worse than the standing 0.245 dex baseline. The
0.245→0.604 headline confounds two changes — old→new extraction code AND
Opus 4.8→Haiku 4.5 — because the baseline predates the `EXTRACTOR_MODEL`
override (#677) and ran hardcoded Opus on old code (commit b66f2cfe). The
claim under scrutiny: **essentially all of the regression is the model swap,
not the extraction-channels code.** This digest tries to break that claim.

## Evidence for the model cause

Three independent lines, each attacking the confound from a different side:

**1. Same-code model isolation (the direct test).** The quarantined
`opus_newcode_sample/` (NEW code on Opus 4.8, N=3, master @ 5b3a4fce — the
silent-Opus incident, before #677 existed) and the final Haiku run (NEW code
on Haiku 4.5, N=3, master @ 55ff3d7f) share 77 papers. Scored identically:

| arm (identical extraction code, 77 shared papers) | micro-median | n compared |
|---|---|---|
| Opus 4.8 | 0.353 dex | 51 |
| Haiku 4.5 | 1.307 dex | 42 |
| **model effect** | **+0.954 dex** | Haiku also loses 9 comparable extractions |

Haiku is worse on *both* axes — accuracy on the papers it does extract, and
coverage (42 vs 51 comparable). A code-caused regression cannot produce this
table, because the code is the same in both rows.

**2. Fixed-model code A/B (the complementary control).** Holding the model at
Opus and varying only the code, on the same 77 paired papers: OLD code
0.369 dex / 47 compared vs NEW code 0.353 dex / 51 compared — a **−0.016 dex**
(slight improvement) code effect with **+4** coverage. Per-paper: 21 improved,
12 regressed, 13 ~same; coverage recovered on 5 papers (4 text, 1
figure_vision/41 pts), lost on 1 (1512.06165, a projection/proposal the new
code correctly declined). So the regression appears **only when the model
changes, never when the code changes**. The two controls together form a
clean 2-factor decomposition: code ≈ −0.02 dex, model ≈ +0.95 dex.

**3. Mechanism.** The Haiku run log contains **208 stage-2 "No JSON found in
response" failures** — malformed, fence-wrapped, or truncated JSON on vision
(plot-reading) calls, forcing fallbacks to lower-tier channels — plus 15
zero-point extractions. This is a concrete, model-shaped failure mode
(instruction-following on structured output under vision load), consistent
with the coverage loss and the ~1 dex vision penalty. Caveat below (#5).

## Alternative explanations, each examined

| # | Alternative | Verdict | Reasoning |
|---|---|---|---|
| 1 | **Code drift between the two isolation arms** (5b3a4fce vs 55ff3d7f) | **Ruled out** | The only pipeline/ commits between the two SHAs are #677 (`EXTRACTOR_MODEL` env plumbing — config only), #678 (`REVIEWER_MODEL` — reviewer agent, not in the extraction scoring path), and #679 (`batch_client.py` — transport shim). Prompts, JSON parsing, channel routing (source_data/vector_trace), and read-vote consensus are byte-identical. The 77-paper comparison is genuinely same-extraction-code. |
| 2 | **The 77-paper sample is the failure tail** (over-samples hard vision papers) | **Limits, does not refute** | The absolute numbers (0.353 / 1.307) are inflated relative to the full pool — that is honest and stated in the final report. But a same-paper paired delta cannot be *created* by sample selection: both arms saw the identical 77 papers, so selection inflates the difficulty level, not the sign of the within-paper contrast. What it can do is make +0.95 dex an overestimate (or, less likely, underestimate) of the *full-pool* model effect if the model gap is difficulty-dependent — which the vision-JSON mechanism suggests it is. Direction survives; the exact magnitude does not generalize automatically. |
| 3 | **N=3 vote stochasticity** (single run per arm) | **Limits, does not refute** | Noise is demonstrably present: the fixed-model Opus A/B — where the true effect is near zero — still scattered 21 improved / 12 regressed per paper, yet its *aggregate* landed at −0.016 dex. That is the empirical scale of run-to-run aggregate noise on this pool: a few hundredths of a dex. The model contrast is +0.954 dex — roughly 50× larger — accompanied by a systematic coverage loss (42 vs 51) and 208 mechanistic failures. Noise of that magnitude, all in one direction, on 77 paired papers, is not a plausible draw. Aggregate direction: safe. Per-paper attributions and the third significant figure: not. |
| 4 | **Transport differences** (Opus arm: plain direct calls; Haiku arm: batch shim + >3 MB direct-bypass for vision) | **Residual, low probability** | The shim forwards params verbatim and is transport-only by design (7/7 unit tests pass), and the vision calls — where the failures concentrate — actually ran on the *same* plain `messages.create` path in both arms after the bypass fix. But "params verbatim" is asserted from code review + unit tests, not proven end-to-end against results; a subtle transport→output interaction (e.g. batch-side truncation) has not been formally excluded. It would also have to explain why the reused 65 Opus snapshots inside the same batched Haiku run scored 0.259 dex (near-baseline) — hard to square with a transport-level corruption. Kept open as a small residual confound. |
| 5 | **The 208 JSON failures are prompt/parsing bugs, not the model** | **Ruled out as a code cause; rate comparison caveated** | The prompts and the JSON parser are identical code in both arms (see #1), and Opus on the same prompts and papers produced *more* comparable extractions (51 vs 42), not a matching collapse. If the parser or prompt were broken, it would be broken for Opus too. Honest caveat: the 208 count comes from the Haiku run log only — the Opus sample's raw logs were not retained, so there is no logged Opus failure *rate* to A/B against. "Haiku-specific" is inferred from the failure's nature (malformed/truncated JSON emission is a model behavior, not a parser behavior) plus the coverage asymmetry, not from a direct rate comparison. |
| 6 | **GT/scoring asymmetry** | **Ruled out** | Both arms scored with the same scorer, same ground truth, same 77 papers, same exclusion rules. Nothing in the scoring path knows which model produced a snapshot. |
| 7 | *(Added)* **Reuse-slice contamination of the headline** | **Ruled out for the isolation test; already handled for the headline** | The final run's 65 reused Opus snapshots dilute the mixed 0.604 headline (that is why 0.975 pure-Haiku is reported separately), but the 77-paper isolation comparison is scored on re-run papers in both arms — the reuse mechanism does not touch the causal test. The reused slice landing at 0.259 dex is itself a scorer sanity check: same scorer, Opus-quality inputs, baseline-quality score. |

No alternative survives as a competing *cause*. Two (failure-tail sampling,
single-run noise) and one design assertion (transport) survive as limits on
how precisely +0.95 dex transfers to the full pool.

## Confidence

- **Direction (Haiku materially worse than Opus on identical code): high
  confidence.** Both controlled comparisons point the same way, the effect is
  ~50× the empirically observed aggregate noise floor, coverage and accuracy
  degrade together, and a concrete mechanism is in the log.
- **Mechanism (vision-JSON emission failures): high confidence that it
  contributes, moderate on it being the whole story** — the Opus failure rate
  was never logged, so the 208 count has no direct A/B partner.
- **Magnitude (+0.95 dex generalizing to the full 346-paper pool): moderate
  confidence.** The shared-77 set over-samples the hard tail; if the model
  gap widens with difficulty (the mechanism says it should), the full-pool
  gap is plausibly somewhat smaller than 0.95 dex but still large. Treat
  "≈1 dex" as an operating estimate, not a measured full-pool constant.
- **"The code did not cause the regression": high confidence** — this is the
  one leg supported by its own dedicated control (the fixed-model A/B), and
  it is the leg that matters for the extraction-channels work.

## The one definitive test

**Full-pool Opus (new code) vs full-pool Haiku (new code):** same code, same
346 papers, same transport (batch + direct-bypass), same N=3, same scorer —
only the model varies. This closes both open limits simultaneously: it
replaces the tail-biased 77-paper sample with the full pool, and it puts both
arms on the identical transport, extinguishing the residual shim confound.
Cost: roughly 5× the Haiku run, order **$150–300** (Opus pricing, vision off
the batch discount). Everything short of this — including this digest —
leaves the failure-tail and single-run-noise caveats formally open; they are
argued down here, not measured away.
