# Definitive unconfounded benchmark: Opus vs Haiku on the fully-fixed pipeline (346 papers each)

**TL;DR:** The definitive two-arm benchmark — both models on the SAME fully-fixed
pipeline (master ad8ecc1a incl. #683/#684), same 346 fresh papers, same N=1, same
direct transport — closes the arc with three verdicts. **CODE:** the pipeline work
paid off — fixed-pipeline Opus at N=1 lands at 0.2646 dex micro-median, headline
parity with the old 0.245 N=3 baseline at one-third the votes, with the per-type
tail dramatically better (macro 1.156 vs 1.582) and the new defenses firing 19+20
times in vivo. **MODEL:** the true Opus↔Haiku gap is **+0.22 dex median** on 206
paired papers — not the earlier confounded ~1 dex estimate — but the tail gap is
real (41 vs 18 catastrophic; macro 4.95 vs 1.16). **OPERATIONS:** recommendation
(decision is the user's): flip the daily/weekly extraction back to Opus, keep
Haiku as the benchmark/eval default — at 1–3 papers/day the ~5× cost argument is
cents, while the tail risk lands on the human reviewer.

Run 2026-07-04 on master ad8ecc1a, which includes ALL fixes: #683 (text-vision
corroboration gate + foreign-quantity convention screen), #684 (plotted-values
contract: vision emits raw axis values + declares them), #681 (transport), plus
the earlier extraction-channels stack (#663–#680).

## Run provenance & configuration

| | Opus arm | Haiku arm |
|---|---|---|
| Code | master ad8ecc1a (post #683/#684) | identical |
| Model | claude-opus-4-8 | claude-haiku-4-5 |
| Consensus | N=1 (single read) | N=1 |
| Transport | direct (`messages.create`) | direct |
| Pool | 346 papers, all fresh — **no snapshot reuse** | identical |
| Completion | 346/346, zero error stubs | 346/346, zero error stubs |
| Outdir | `evaluation/eval_runs/final2_opus_n1/` | `evaluation/eval_runs/final2_haiku_n1/` |

Design notes:

- **N=1 justified by the probe:** on Opus, N=3 − N=1 measured at **+0.003 dex**
  (`nprobe_and_routing_memo.md`) — consensus voting buys essentially nothing at
  Opus quality, so both arms run single-read for cost and symmetry.
- **Direct transport** because the batch dispatcher wedged repeatedly; both arms
  use the identical plain path, so transport is not a confound this time.
- Two clean credit-exhaustion pauses + resumes on the Opus/Haiku boundary
  (skip-existing); no partial or corrupted snapshots resulted.
- Unlike the previous full-pool run, **nothing is reused** — every snapshot in
  both arms is a fresh extraction on the fixed code, so neither headline is a
  mixture.

## Results

### Arm headlines (official `evaluate.py` metrics)

| | Opus 4.8, N=1, fixed | Haiku 4.5, N=1, fixed |
|---|---|---|
| micro-median residual | **0.2646 dex** | **0.6605 dex** |
| macro-median (per-type tail) | **1.156 dex** | **4.947 dex** |
| papers compared | 266 | 231 |
| per-paper scored | 251 | 222 |
| catastrophic (>3 dex) | 18 | 41 |
| >1 dex | 47 | 89 |

Channel mix (winning `data_source` across all 346 snapshots per arm):

| channel | Opus | Haiku |
|---|---|---|
| text | 141 | 173 |
| figure_vision | 156 | 132 |
| vector_trace | 17 | 20 |
| figure | 0 | 4 |
| table | 5 | 3 |
| source_data | 3 | 2 |
| none | 24 | 12 |

Defense activity in the Opus arm: **19 corroboration-gate rejections**
(`[TEXT-VISION DISAGREEMENT]`) and **20 `[CONVENTION REVIEW]` flags** — the
#683/#684 defenses are load-bearing at scale, not theoretical.

### Paired model comparison (the clean #678 evidence)

206 papers scored in BOTH arms — identical code, identical papers, identical
scorer, only the model differs:

| paired statistic (206 shared papers) | value |
|---|---|
| median per-paper residual delta (Haiku − Opus) | **+0.220 dex** |
| Opus better (>0.05 dex) | 132 papers |
| Haiku better (>0.05 dex) | 36 papers |
| ~tied (≤0.05 dex) | 38 papers |

This is the comparison the causal digest called "the one definitive test":
full-pool, same code, same transport, same N — the confounds that limited the
77-paper isolation estimate are extinguished.

## Verdict 1 — CODE: the pipeline work paid off

Reference points:

- Old baseline: **0.245 dex** micro / **1.582** macro — Opus 4.8, OLD code, N=3
  (`evaluation/results/metrics.json`).
- This run: **0.2646 dex** micro / **1.156** macro — Opus 4.8, fixed code, **N=1**.

Reading:

1. **Headline parity at one-third the votes.** 0.2646 vs 0.245 is within the
   measured run-to-run noise floor (~0.02–0.07 dex aggregate), achieved with
   N=1 instead of N=3 — a 3× reduction in extraction calls per paper at
   constant headline accuracy.
2. **Parity is conservative, not flattering.** The hardened convention registry
   now honestly EXCLUDES convention-gap papers that the old baseline scored
   raw; the fixed pipeline is being graded on a stricter contract.
3. **The tail is where the code work shows.** Macro-median (per-coupling-type,
   dominated by the hard tail) improved 1.582 → **1.156 dex** — the
   catastrophic per-type failures the old code shipped are substantially
   reduced.
4. **The defenses work in vivo.** 19 text-vision corroboration rejections and
   20 convention-review flags fired in a real 346-paper run — the #683/#684
   machinery catches actual mis-extractions at scale rather than sitting idle.

This closes the question the confounded run left open: at fixed (Opus) model,
the full-pool code effect is parity-or-better on the headline and a clear win
on the tail, consistent with the earlier 77-paper code A/B (−0.016 dex).

## Verdict 2 — MODEL: the true gap is +0.22 dex median, not ~1 dex

The earlier estimate (`haiku_regression_causal_digest.md`) put the Opus→Haiku
swap at **+0.95 dex** on 77 shared papers and flagged its own two limits:
failure-tail sampling and the single-run design. The definitive paired
measurement lands at **+0.220 dex median** over 206 papers. The earlier number
was inflated by two now-understood factors:

1. **Failure-tail sampling.** The 77-paper shared set over-sampled hard vision
   papers (its Opus arm alone scored 0.353 vs 0.2646 here). The digest
   predicted the full-pool gap would be smaller if the model gap is
   difficulty-dependent; it is.
2. **The #684 mechanism (plane bookkeeping).** The pre-fix code
   double-converted or zero-converted values read off non-canonical axes.
   That bug punished Haiku disproportionately: Haiku's erratic
   instruction-following produced more non-canonical / inconsistently-declared
   vision output, which the broken bookkeeping then turned into decade-scale
   errors. The plotted-values contract (vision emits raw axis values and
   declares them; #684) removed a code amplifier of a model weakness. Haiku's
   own absolute number improved from **0.975 dex** (pre-fix, N=3!) to
   **0.6605 dex** (fixed, N=1).

**But the retraction cuts one way only — Haiku is still genuinely worse:**

- Tail: **41 vs 18** catastrophic (>3 dex) papers; **89 vs 47** above 1 dex;
  macro-median **4.947 vs 1.156** — the per-type tail is where Haiku falls
  apart.
- Coverage: 231 vs 266 papers compared — Haiku produces fewer comparable
  extractions, itself part of the gap (papers Haiku fails to extract don't
  appear in the residual median at all).
- Pairwise: Opus better on **132 of 206** paired papers vs 36 for Haiku.

Haiku is a genuinely worse extractor — just not 10×-in-dex worse. The correct
summary is: modest median gap (+0.22 dex), large tail gap.

## Verdict 3 — Cost & operations (recommendation; decision is the user's)

Price ratio: Haiku is ~5× cheaper than Opus per extraction. Whether that
matters depends entirely on volume:

- **Bulk benchmarks (hundreds of papers):** Haiku at ~5× cheaper with a
  +0.22 dex median penalty is a defensible trade when cost-bound. Keeping
  Haiku as the benchmark/eval default (per the standing cost rule) remains
  reasonable.
- **The daily pipeline (#678 made all workflows Haiku):** volume is ~1–3
  papers/day, so the absolute Opus−Haiku cost difference is
  **cents-to-a-dollar per day**. The cost argument that motivated the
  all-Haiku switch does not really apply at daily volume — while the tail
  risk does: 41/346 catastrophic Haiku extractions (~12%) vs 18/346 (~5%)
  for Opus means materially more decade-scale mis-extractions reaching the
  human reviewer.

**Recommendation:** flip the daily/weekly extraction back to Opus (the
`EXTRACTOR_MODEL` env var in the workflows, #677), keep Haiku as the
benchmark/eval default, and re-enable the daily schedule. Every science PR is
still human-reviewed, so either choice is safe-guarded — this is about not
wasting reviewer time and trust on tail failures, not about a safety gap.

## What changed since the confounded run

| step | artifact/PR | one-line finding |
|---|---|---|
| Confounded full-pool run | `final_full346_haiku_report.md` | 0.604 mixed / 0.975 pure-Haiku; headline model-confounded; code A/B on 77 papers cleared the code (−0.016 dex) |
| Causal digest | `haiku_regression_causal_digest.md` | model swap causal with high confidence on direction; +0.95 dex magnitude flagged as tail-biased, "one definitive test" specified |
| #683 | text-vision corroboration gate + foreign-quantity convention screen | independent text/vision cross-check rejects uncorroborated vision reads; screens out foreign-quantity axis conversions |
| #684 | plotted-values contract | vision emits raw axis values and declares them — kills the double/zero-conversion plane-bookkeeping bug that amplified Haiku's errors |
| N-probe | `nprobe_and_routing_memo.md` | N=3 − N=1 = +0.003 dex on Opus — consensus voting unnecessary at Opus quality; both definitive arms run N=1 |
| #681 | transport | direct transport; batch dispatcher (which wedged repeatedly) not used for this run — both arms identical path |
| This run | `final2_opus_n1/`, `final2_haiku_n1/` | the definitive unconfounded two-arm measurement |

## Caveats

- **Classification-accuracy fields were not populated** in this scoring config
  (n/a) — the headline is the interpolation-metric suite only; no
  coupling-type / is_new_limit accuracy comparison is available for this run.
- **Single run per arm.** The measured routing/noise floor (~0.02–0.07 dex
  aggregate) applies to all headline numbers. The paired median (+0.220 dex,
  206 papers, sign consistent at 132 vs 36) is robust to that; the second
  decimal is not.
- **Compared-counts differ between arms** (266 vs 231): Haiku produces fewer
  comparable extractions, so its residual median is computed over an
  easier-surviving subset — the arm-headline gap likely *understates* the
  true model gap. The paired comparison (206 papers scored in both) is the
  clean number.
- **Headline parity with the old baseline is not a point-identity claim.**
  The old 0.245 baseline scored some papers the hardened registry now
  excludes; the two numbers are computed over slightly different effective
  pools under different convention contracts.

## Performance impact (before/after)

- **Code (at fixed Opus model):** old baseline **0.245 dex** micro (OLD code,
  N=3) → **0.2646 dex** (fixed code, **N=1**) — headline parity within the
  noise floor at **one-third the votes per paper**, with macro-median
  improved **1.582 → 1.156 dex** and the #683/#684 defenses firing 19+20
  times in vivo. The pipeline work did not regress accuracy and materially
  improved the tail.
- **Haiku operating point:** pre-fix pure-Haiku **0.975 dex** (N=3, batch) →
  **0.6605 dex** (fixed code, N=1, direct) — the fixes recovered ~0.3 dex of
  what was previously mis-attributed entirely to the model.
- **Model gap, re-measured cleanly:** the confounded/tail-biased **+0.95 dex**
  estimate is superseded by the definitive paired measurement:
  **+0.220 dex median** (Haiku − Opus, 206 shared papers, identical code and
  transport), with the caveat that Haiku's tail (41 vs 18 catastrophic,
  macro 4.95 vs 1.16) and coverage (231 vs 266 compared) remain substantially
  worse.
