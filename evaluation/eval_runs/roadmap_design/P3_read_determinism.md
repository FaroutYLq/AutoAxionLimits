# P3 — Read-layer determinism (temperature-0 + multi-sample vote + convention normalizer)

> Roadmap phase **P3** of issue **#566** (governs) / **#550** Phase-1 family.
> Design/scoping only — no pipeline code in this doc.
> Acceptance harness: `evaluation/subset_eval.py` + `evaluation/subset_compare.py`
> (the `--repeats N` determinism path, `determinism_report`).
> Code designed against: `AAL-integration/pipeline/extractor.py`,
> `AAL-integration/pipeline/plot_calibration.py`, `AAL-integration/pipeline/config.py`.
> Builds on **P0** (`pipeline/transform_guard.py`, the HARD R5 floor and
> improve-or-revert) — every P3 read-aggregate and convention conversion is a
> *candidate* that P0's `guard_transform` commits-or-reverts.

P0/P1/P2 all assumed the **reads** are stable and the **convention** is fixed.
They are not. #561 made the *post-hoc correction* deterministic (median over
sorted values, discrete-factor snap — extractor.py:573-580), but the layer
upstream of the correction — the LLM reads themselves — still draws at
**temperature 1.0** (no `temperature=` argument appears on *any*
`client.messages.create` call: extractor.py:529, 743, 1153, 1231, 1485) and runs
**once**. P3 is the read-layer fix: deterministic sampling, multi-sample
aggregation for the noisy reads, and a per-coupling **convention normalizer** so
that a value read correctly but in the *wrong convention* cannot swing the result
by multiple decades.

---

## 1. Problem & evidence

P3 targets **failure class D** of #566 ("Upstream LLM read nondeterminism — the
layer the #561 work didn't reach", 3 papers) plus the *convention* sub-case of
class A/E. For each, the per-paper fan-out (`per_paper_findings.md`) proves the
regression is *not* a transform bug — #561 and #550 are **inert** — it is
run-to-run LLM stochasticity in the read or in the convention choice. P3 is the
only phase that addresses the read itself.

### 1.1 Coupling-value drift between identical-axis vision reads — **1403.1290**

(`per_paper_findings.md` L192-195.) Both runs are `figure_vision`, MonopoleDipole,
with a **byte-identical mass axis** ("Stage 2a axes: x=[0.003, 10.0] cm (log)"
in both). No `#561` line ("Auto-correcting"), no `#550` transform ("CV
calibration unavailable … used LLM axes"). The *only* thing that changed is the
coupling values, and the point-by-point after/before ratios are **non-uniform,
7×–25×**:

> "9e-4 eV → 13.3×, 6.6e-4 → 20×, 3.3e-4 → 25×, 1.97e-5 → 14×, 1.97e-6 → 7×"

so it is **not** a clean decade snap — it is the vision model re-reading the same
log-log curve ~1 decade higher on a single sample. resid **3.699 → 4.951**. The
findings' own fix-hint: *"Make the figure_vision extraction deterministic
(temperature 0 plus seed-stable single read, or average/median multiple
reads)."* This is exactly P3's multi-read median (§2.2) under P0's guard.

### 1.2 Convention flip C_e/F_a (eV⁻¹) ↔ dimensionless g_ae — **1902.04246**

(`per_paper_findings.md` L197-200.) Both runs are **pure text path**
(`src_b=src_a=text`). No `#561`, no `#550` line in the log; mass window
**identical and correct** (1.00e-23..1.00e-18 vs GT 1.08e-22..1.24e-18 eV). The
swing is **entirely a coupling-convention change made by the LLM**:

- *before*: reported the gold-matching **dimensionless g_ae**
  (coup_median 5.10e-10, "computed via g_ae = 2·m_e·C_e/F_a") → resid **0.234**;
- *after*: reported the **raw paper number C_e/F_a = 5.00e-16 eV⁻¹**
  (after-note: *"y-axis of Fig.5 is C_e/F_a in eV⁻¹, NOT the conventional
  dimensionless g_ae"*), **~5.6 dex** below the gold curve → resid **5.790**.

Critically, **#561's decade-snap cannot rescue it**: AxionElectron
`VALID_RANGES["coupling"] = (1e-20, 1e0)` (config.py:323) and 5e-16 lies *inside*
that window, so the snap trigger (`median_coup < coup_lo·1e-3 = 1e-23`) never
fires. P0's R5 HARD floor *also* won't catch it (5e-16 is in range). **Only a
convention normalizer** that recognizes the eV⁻¹ unit and converts
`g_ae = 2·m_e·C_e/F_a` can fix this (§2.3). The findings' fix-hint says exactly
this.

### 1.3 Pre-classifier flip AxionProton ↔ AxionNeutron — **2209.13588**

(`per_paper_findings.md` L202-205.) Both runs `text`; no vision/CV path; no
`#561`/`#550` line. The regression (resid **0.571 → 1.351**) is *"purely LLM
stochasticity in Stage 1: the after call returned 4 points @ coup_median 1e-4 vs
before's 8 points @ coup_median 3e-5, and the pre-classifier flipped from
AxionProton (conf 0.70, before) to AxionNeutron (conf 0.55, after)."* The
classifier (`_classify_coupling_type`, extractor.py:519-546) runs **once** on
title+abstract at temperature 1.0; a 0.70→0.55 confidence wobble between draws
flips the coupling type. P3 fixes this with (a) temperature-0 and (b) a
**majority vote** over a few classifier draws, breaking ties by confidence
(§2.4). The coupling-type flip also changes which `_BENCHMARK_LINES` entry
applies (KSVZ_neutron uses `|−0.02|` vs KSVZ proton coefficients), so stabilizing
it also stabilizes P0's benchmark corroboration.

### 1.4 Why P3 is necessary even after P0/P1/P2

P0 reverts a *bad transform* to the *pre-transform read*. But in all three class-D
papers there **is no transform** — the read itself is the variable. P0 cannot
choose between two equally-in-range reads (5.10e-10 vs 5.00e-16 for 1902.04246
are *both* in VALID_RANGES). P3 removes the variance at the source:
deterministic decoding kills most of it; multi-read aggregation absorbs the
residual; the convention normalizer collapses the two-valued convention ambiguity
to one canonical value **before** P0 scores it. The determinism harness measures
exactly this (median-log10-coupling std across repeats; §4).

The eval's own determinism table (`comparison.md`, "Determinism (#561)") shows the
reads are still noisy *after* #561: mean after-std **0.230 dex** with individual
papers at **0.471 dex** (2111.06883), **0.480 dex** (1709.00009), **0.793 dex**
(2410.10363 — *worse* after), **0.188 dex** (2209.13588). The noise floor the
harness assumes is **0.32 dex**. P3's target is to push the mean and the worst
offenders below that floor.

---

## 2. Design

Four components, in dependency order. All four emit *candidates* that P0's
`guard_transform` / R5 floor commit-or-revert; none bypasses the P0 contract.

### 2.1 Deterministic decoding (temperature-0 + seed) — the cheap 80%

**Where:** every `client.messages.create` call in `extractor.py` — the
pre-classifier (:529), Stage-1 text (:1153), Stage-2a axes (:1231), Stage-2
trace (:1485), and the vision-verify spot-check (:743). None currently pass
`temperature`; the Anthropic default is **1.0**.

**Change:** add `temperature=0.0` to all five calls. The Anthropic SDK does not
expose a user RNG seed, so temperature-0 (greedy/near-greedy decoding) is the
determinism lever available; it is what the 1403.1290 fix-hint names first
("temperature 0 plus seed-stable single read"). Implementation: a single module
constant `_READ_TEMPERATURE = 0.0` and a thin wrapper
`_create(client, **kw)` that injects it, so the setting is defined once and is
unit-testable (it appears in the kwargs of every call — the P4 CI gate can assert
no `messages.create` is invoked without it).

**Expected effect:** temperature-0 alone removes the bulk of the
single-sample-to-single-sample jitter that produced the 7-25× drift in 1403.1290
and the 0.70↔0.55 classifier wobble in 2209.13588. It does **not** make
log-axis vision reads perfectly reproducible (the model can still land on a
different greedy path for a different image crop), which is why §2.2 adds
multi-read aggregation on top, not instead.

**Cost:** zero new latency for the deterministic calls; the multi-read in §2.2 is
the only added cost and is scoped to the two genuinely noisy reads.

### 2.2 Multi-sample median/vote for the two noisy reads

Temperature-0 is necessary but not sufficient for the **axis read** and the
**curve trace**, which the #550 roadmap itself identifies as the metrology-weak
reads. P3 runs **N=3** independent samples of *only* these two reads and
aggregates:

| Read | Sampling | Aggregation | Rationale |
|---|---|---|---|
| **Stage-2a axes** (`_run_stage2a_axes`, extractor.py:1211-1254) | N=3 at a small temperature spread (one greedy at T=0 + two at T=0.4) | **per-endpoint median** of `x_min,x_max,y_min,y_max` in **log10 space**; scale/unit by **majority vote** | endpoints are independent scalars; log-median is robust to a single 1-decade misread (the 1403.1290 failure mode) |
| **Curve trace** (the CV `_attempt_cv_curve_trace` proposal, extractor.py:1398-1426, *and* the LLM Stage-2 `_run_stage2` points) | N=3 | **point-wise vertical median**: bin the union of x-samples into the common mass grid, take the median coupling per bin in log10 space; keep a bin only if ≥2 of 3 runs cover it | a single run reading the curve a decade high (1403.1290) is outvoted; sparse/degenerate runs (the 1905.13650 floor-pin) are minority-filtered |

**Aggregation in log10 space** matches the harness scoring exactly: the
determinism metric is `median(log10 coupling)` per run (`subset_compare.py:196`),
so a log-median of three reads is the literal minimizer of that metric's spread.

**Why N=3, not 5+:** the harness noise floor is **0.32 dex** and the mean
post-#561 std is **0.230 dex**. With 3 draws the standard error of the median of
roughly-log-normal reads is ≈ `1.25·σ/√3 ≈ 0.72·σ`, i.e. a ~30% spread
reduction, which moves the worst offenders (2111.06883 0.471, 1709.00009 0.480)
toward/under the 0.32 floor while only tripling the two noisy-read API calls.
N=5 buys ≈ another 10% for ~67% more cost — not worth it on an 82-paper subset.
N is a module constant (`_READ_SAMPLES = 3`) so P4 can sweep it.

**Determinism vs. temperature-0:** the two-temperature spread (T=0 + 2×T=0.4) is
*deliberate* — three identical greedy draws give zero new information. The
aggregation is what provides determinism: the **median of a fixed-size sample is
itself a deterministic estimator** once the sample is drawn, and the spread
across the harness's outer `--repeats` collapses because we report the
*aggregate*, not a single draw. (Concretely: the harness re-runs extraction 3×;
each extraction internally medians 3 reads; the variance the harness sees is the
variance of a median-of-3, ≈ half the single-read variance.)

**Composition with P0/P2:** the aggregated axis is a *single candidate* fed to
P0's `_calibrate_axis` guard (R2); the aggregated trace is a *single candidate*
fed to P0's CV-trace gate (R4 degenerate / quality tuple). P3 reduces the
**input** variance; P0 still decides commit-or-revert. The minority-coverage
filter (≥2/3) is a *pre*-filter that makes the 1905.13650-style floor-pin less
likely to even reach P0's R4.

### 2.3 Per-coupling **convention normalizer** (the 5.6-dex fix)

A new pure function `normalize_convention(coupling_type, data_points,
axis_unit_label, notes) -> (data_points', conversion_note)` in
`pipeline/transform_guard.py` (P0's module — it is the natural home; it already
owns the in-range / consistency logic and is unit-testable like the #544 metric
tests). It runs **after the read, before P0's R5 floor and before
`_validate_extracted_range`** (extractor.py:~1075 where `data_points` are first
materialized as floats).

It maps a value reported in a paper-native convention to the **repo canonical
convention** declared in `config.COUPLINGS[ct]["axes"]["y"]` (config.py:26,34,…).
The rule set is **per coupling type**, keyed on the read-back axis unit label and
on `notes` (the LLM already *states* the convention it used — 1902.04246's note
literally says "C_e/F_a in eV⁻¹, NOT … dimensionless g_ae"; we parse that):

| Coupling | Repo canonical (config) | Detected native convention | Conversion | Constant |
|---|---|---|---|---|
| **AxionElectron** | dimensionless `g_ae` | `C_e/F_a` in **eV⁻¹** (label contains `eV^-1`/`eV⁻¹`/`/eV`, or note says `C_e/F_a`) | `g_ae = 2·m_e·(C_e/F_a)` | `m_e = 5.11e5 eV` → factor **1.022e6** |
| **AxionProton/Neutron** | dimensionless `g_aN` | `C_N/F_a` in eV⁻¹ | `g_aN = 2·m_N·(C_N/F_a)` | `m_p=9.383e8`, `m_n=9.396e8 eV` |
| **AxionPhoton** | `g_aγγ` in **GeV⁻¹** | reported in **eV⁻¹** (label `eV^-1`) | `g[GeV⁻¹] = 1e-9·g[eV⁻¹]` | 1e-9 |
| **DarkPhoton** | kinetic mixing `ε` (dimensionless) | already dimensionless | identity (guard only) | — |
| **AxionEDM** | `d_n` in `e·cm` (or `g_d` GeV⁻²) | C_G/f_a vs d_n — flagged, **no auto-convert** (ambiguous; see Risks) | — | — |

The conversion uses the **same physics constant** the benchmark lines already
encode: `_BENCHMARK_LINES["AxionElectron"]` (extractor.py:558) is the DFSZ g_ae
line, and the g_ae↔C_e/F_a map is the standard `g_ae = 2 m_e C_e/F_a`
(this is precisely the relation 1902.04246's *before* run applied correctly and
its *after* run dropped). `m_e = 511000 eV` is a CODATA constant, no new dep.

**Detection precedence (deterministic, no LLM):**
1. axis unit label parsed by Stage-2a (`y_axis_unit`) — if it contains an
   `eV^-1`/`/eV` token *and* the canonical is dimensionless/GeV⁻¹ → convert;
2. else, regex on `notes` for the convention string the LLM emitted
   (`C_e/F_a`, `C_N/F_a`, `eV^-1`);
3. else, **range-based fallback**: if the median coupling for AxionElectron is in
   `[1e-18, 1e-12]` (the eV⁻¹ band, ~6 decades below the dimensionless band) and
   the dimensionless equivalent `2·m_e·median` lands inside the *expected*
   benchmark band (within 2 dex of `DFSZ_upper(median_mass)`), convert. This
   range fallback is what catches 1902.04246 even when the note is absent,
   because **5.00e-16 · 1.022e6 = 5.11e-10**, which lands on the gold curve
   (GT coup_median 5.10e-10 — an exact match to the *before*, correct value).

**This is a guarded transform:** the converted candidate is scored by P0
(benchmark-ratio R3 must *improve* — for 1902.04246 it goes from 5.6 dex off to
~0 dex off, a clear improvement; R5 floor must hold — 5.11e-10 is in
`(1e-20,1e0)`). If conversion does *not* improve the benchmark ratio, P0 reverts
it. So the normalizer can never *introduce* a 1902.04246-in-reverse error.

### 2.4 Stabilize the pre-classifier (coupling-type vote)

`_classify_coupling_type` (extractor.py:519-546) runs once. P3:
1. sets `temperature=0.0` (§2.1) — removes most of the 0.70↔0.55 wobble;
2. draws **N=3** classifier samples (cheap, 128 tokens each) and takes the
   **modal coupling_type**, tie-broken by **mean confidence**;
3. carries the *vote margin* (votes_for_winner / N) as a new confidence input so
   a 2-1 split is flagged lower-confidence than 3-0.

This directly fixes 2209.13588's AxionProton↔AxionNeutron flip: a 3-sample vote
at T=0 is overwhelmingly more likely to land on the same type than a single
T=1.0 draw, and the modal type is deterministic given the sample. Because the
coupling type selects the `_BENCHMARK_LINES` entry, stabilizing it also makes
P0's R3 benchmark corroboration reproducible.

### 2.5 Data flow (P3 layered onto P0)

```
_classify_coupling_type  →  [P3] N=3 vote @T=0, modal type + margin     ← 2209.13588 fix
_run_stage1 (text)       →  [P3] T=0 single read (deterministic)
_run_stage2a_axes        →  [P3] N=3 (T=0,0.4,0.4), per-endpoint log-median  ← 1403.1290 axis stability
        └─ P0 _calibrate_axis guard (R2)                                  (commit-or-revert)
_run_stage2 / _attempt_cv_curve_trace
                         →  [P3] N=3, point-wise log-median, ≥2/3 coverage filter  ← 1403.1290 / 1905.13650
        └─ P0 CV-trace gate (R4 degenerate / quality tuple)              (commit-or-revert)
materialize data_points (float)
   └─ [P3] normalize_convention(ct, points, unit_label, notes)           ← 1902.04246 fix (5.6 dex)
        └─ P0 guard: benchmark-ratio must IMPROVE + R5 floor             (commit-or-revert)
_run_vision_verify / _calibrate_vision_data  (P0 R1/R3 guards, T=0 spot-check)
_validate_extracted_range  (#561 snap, P0 improve-or-revert + R5)
final P0 HARD-floor assert
```

Every P3 component reduces **input variance**; every P0 component still gates
**commit-or-revert**. P3 never overrides P0.

---

## 3. Integration points (exact files / functions)

All edits in the `AAL-integration` worktree.

### 3.1 Deterministic decoding (`extractor.py`)
- Add `_READ_TEMPERATURE = 0.0` and `_READ_SAMPLES = 3` module constants near the
  `CLAUDE_MODEL` definitions.
- Add `temperature=_READ_TEMPERATURE` to `client.messages.create` at **:529**
  (pre-classifier), **:743** (vision-verify spot-check), **:1153** (Stage-1),
  **:1231** (Stage-2a axes), **:1485** (Stage-2 trace). Prefer a thin
  `_create(client, **kw)` wrapper so the kwarg is defined once and P4 can assert
  it. `_call_with_retry` (extractor.py:39) is unchanged — it already wraps each
  `create`.

### 3.2 Multi-read aggregation
- New helpers in `extractor.py` (or `plot_calibration.py` for the trace median,
  since it already owns `trace_curve`): `_median_axes(samples)` (per-endpoint
  log10 median + majority unit/scale vote) wrapping the `_run_stage2a_axes` call
  at **:1021/1247** (the axis-info producer), and `_median_trace(samples)`
  (point-wise log10 median + ≥2/3 coverage filter) wrapping
  `_attempt_cv_curve_trace` (**:1025**) and the `_run_stage2` points (**:1021**).
  Both emit a *single* candidate into the existing P0 guard call sites — no change
  to P0's gate signatures.

### 3.3 Convention normalizer
- New `normalize_convention(...)` in **`pipeline/transform_guard.py`** (P0's
  module). Called from `extractor.py` right after `data_points` are materialized
  as floats (**~:1075**), *before* `_validate_extracted_range` (:856) and before
  the P0 final floor. It reads `config.COUPLINGS[ct]["axes"]["y"]` for the
  canonical convention and the Stage-2a `y_axis_unit` + `stage1_result["notes"]`
  for the native one. Wrap the result in P0's `guard_transform` with the
  benchmark-ratio-must-improve criterion (reuses `_BENCHMARK_LINES`,
  extractor.py:556).
- Constants: a small `_CONVENTION_CONSTANTS` dict in `transform_guard.py`
  (`m_e=5.11e5`, `m_p=9.383e8`, `m_n=9.396e8` eV; `eV_to_GeV=1e-9`). No new dep.

### 3.4 Pre-classifier vote
- Modify `_classify_coupling_type` (**:519-546**) to loop `_READ_SAMPLES` times,
  collect `(ct, conf)`, return `(modal_ct, mean_conf_of_modal, margin)`. Update
  the single caller at **:970** (`pre_ct, pre_conf = ...`) to accept the margin
  and fold it into the LOW_CONFIDENCE decision (config.py:341).

### 3.5 Composition with the other phases
- **P0** is a hard prerequisite: every P3 candidate (median axis, median trace,
  converted couplings, voted type) is committed only through P0's
  `guard_transform`/R5 floor. P3 *shrinks the candidate's variance*; P0 *gates
  it*. The convention normalizer specifically relies on P0's R3
  benchmark-improve check so a mis-detected convention auto-reverts.
- **P1** (axis OCR metrology): P3's per-endpoint axis median runs *over the LLM
  axis reads*; when P1 lands, the OCR'd axis becomes one more sample in the median
  (or, better, the corroboration that lets P3 drop to N=1 when OCR is confident).
  P3 and P1 are complementary: P1 makes one read trustworthy, P3 makes the
  fallback reads stable.
- **P2** (selector): P3 reduces the within-candidate variance that would
  otherwise make P2's quality scores themselves nondeterministic; the voted
  coupling type is what P2's `source_tier`/benchmark scoring keys on.
- **P4** (CI gate): asserts `temperature=0.0` on every `messages.create`, and
  adds the determinism `--repeats 3` mean-std threshold (§4) as a required check.

---

## 4. Acceptance criteria (measured on `subset_eval.py`)

Run the determinism path (the harness's purpose-built mechanism — `--repeats N`
writing `<id>_r{k}.json`, scored by `determinism_report` =
`pstdev(median(log10 coupling) per run)`):

```
python -m evaluation.subset_eval extract --key union --repeats 3 \
    --outdir evaluation/eval_runs/after_p3_repeats
python -m evaluation.subset_eval extract --key union \
    --outdir evaluation/eval_runs/after_p3
python -m evaluation.subset_eval compare \
    --before evaluation/eval_runs/before \
    --after  evaluation/eval_runs/after_p3 \
    --before-repeats evaluation/eval_runs/before_repeats \
    --after-repeats  evaluation/eval_runs/after_p3_repeats \
    --out evaluation/eval_runs/comparison_p3.md
```

P3 is accepted iff (all on top of P0 already applied, vs the master `before`
baseline and the #561 `after` determinism column):

1. **Determinism — mean coupling-scale std** across the determinism table ≤
   **0.150 dex** (from the #561 after-mean **0.230**, well under the **0.32 dex**
   noise floor). The two worst offenders must clear the floor: **2111.06883
   ≤ 0.32** (from 0.471) and **1709.00009 ≤ 0.32** (from 0.480). No paper's
   after-std may *increase* vs its #561-after value (2410.10363 must not stay at
   0.793 — target ≤ 0.40).
2. **1403.1290** (vision drift): scale_range_dex across 3 repeats ≤ **0.5 dex**
   (the single-read drift was 7-25× ≈ 0.85-1.4 dex); single-run resid no worse
   than master's **3.699** (the 4.951 after-regression must be gone — the
   log-median outvotes the 1-decade-high read).
3. **1902.04246** (convention): single-run resid ≤ **0.5 dex** (recover the
   master **0.234**; the 5.790 convention-flip regression must be gone). Assert
   the normalizer fired: converted coupling = `5.00e-16 · 1.022e6 = 5.11e-10`,
   landing on GT 5.10e-10. Determinism: scale_std ≤ **0.1 dex** (convention is now
   single-valued).
4. **2209.13588** (classifier flip): coupling_type identical across all 3
   repeats (no AxionProton↔AxionNeutron flip); single-run resid ≤ **0.7 dex**
   (recover master **0.571**; the 1.351 regression gone); scale_std ≤ **0.15 dex**
   (from #561-after **0.188**).
5. **No net regression** on the aggregate (P0's invariants must hold): overall
   median residual ≤ **0.485 dex**; zero-overlap ≤ **14**; zero `status==error`.
   P3 must not *add* drift anywhere — every per-paper after-std ≤ its #561-after
   value within ±0.05 dex tolerance.
6. **Genuine wins preserved**: the 5 P0-protected wins (1207.3275, 1508.01798,
   1708.06367, 1804.05750, 2402.12892) keep their improved residuals; P3's
   convention normalizer must be a **no-op** on all of them (none is an eV⁻¹
   AxionElectron case) and the N=3 axis/trace median must not move their already-
   correct windows.

Stretch (informational): figure_vision `≤0.3 dex` fraction trends up as the
median-of-3 trace removes single-read outliers.

---

## 5. Risks & open questions

- **Cost of N=3 on the noisy reads.** Tripling Stage-2a + Stage-2 (vision,
  `CLAUDE_MODEL_VISION`) is the dominant added cost. Scoped to *only* the two
  metrology-weak reads (axis, trace), not Stage-1 text or the spot-check, keeps it
  bounded. Open: gate N down to 1 when P1's OCR corroboration is high-confidence
  (P3+P1 interaction) to claw the cost back on the easy cases.
- **Temperature-0 can lock in a *consistent* wrong read.** Greedy decoding makes a
  systematic misread reproducible rather than averaging it out — determinism is
  not accuracy. This is why P3 keeps the **two-temperature spread (T=0 + 2×T=0.4)**
  for the median: identical greedy draws give no new signal. And P0's guards
  still gate the aggregate. Open: confirm on 1403.1290 that T=0 alone doesn't lock
  in the 1-decade-high read (if it does, the median of the T=0.4 draws is what
  saves it).
- **Convention detection false positives.** The range-based fallback (§2.3)
  could mis-convert a *correct* dimensionless g_ae that happens to sit low. Two
  backstops: (a) it only fires when the converted value *improves* the benchmark
  ratio (P0 R3), so a correct value (already on the benchmark) cannot be
  "improved" by ×1e6; (b) it is gated to the eV⁻¹ band `[1e-18,1e-12]` which is
  ~6 decades below the dimensionless band. Open: AxionEDM (C_G/f_a vs d_n) is
  genuinely ambiguous (different physical observables, not a unit rescale) — P3
  **flags, does not auto-convert**, and hands it to a `[NEEDS REVIEW]` note;
  confirm this doesn't regress 1708.06367 (AxionEDM, a protected win) — it must
  not, because that paper's win came from vision routing, not a convention.
- **The SDK has no user seed.** Temperature-0 is the only determinism lever;
  full bit-reproducibility is not achievable through the public API. The harness
  measures *statistical* spread (std across repeats), which is the right target —
  the acceptance thresholds are stated as std/range, not exact equality.
- **N=3 median can still wobble** if all three draws are noisy (e.g. a paper whose
  figure is genuinely unreadable). Those should fall to `[LOW CONFIDENCE]` via the
  carried vote-margin (§2.4) and P0's confidence inputs, not be silently emitted.
- **Threshold N is small.** The 0.150-dex mean-std target is set from ~9 papers in
  the determinism table; P4's CI gate keeps it honest as the subset grows. Land
  P3 *with* the `--repeats 3` re-run, not on argument alone.

---

## 6. Draft sub-issue body

> **Title:** `[extractor][P3] Read-layer determinism: temperature-0 + multi-sample median/vote + coupling-convention normalizer`

**Parent:** #566 (governs). Builds on **P0** (`pipeline/transform_guard.py`,
HARD floor, improve-or-revert). Addresses #566 **failure class D** (upstream LLM
read nondeterminism — the layer #561 did not reach).

### Why
#561 made the *post-hoc correction* deterministic, but the **reads** still draw at
**temperature 1.0** (no `temperature=` on any `messages.create`: extractor.py:529,
743, 1153, 1231, 1485) and run **once**. Three subset regressions are pure
read/convention nondeterminism — #561 and #550 are *inert* in all three
(`per_paper_findings.md`):
- **1403.1290** — identical mass axis, coupling values drift **7-25×** between
  vision reads (resid 3.699→4.951). No transform fired; it is single-sample
  vision jitter.
- **1902.04246** — coupling-**convention** flip `C_e/F_a` (eV⁻¹) ↔ dimensionless
  `g_ae`, a **5.6-dex** swing (resid 0.234→5.790). #561's decade-snap can't catch
  it (5e-16 is *inside* `VALID_RANGES (1e-20,1e0)`); only `g_ae = 2 m_e C_e/F_a`
  conversion fixes it.
- **2209.13588** — pre-classifier flip **AxionProton↔AxionNeutron** (0.70→0.55
  confidence wobble) + Stage-1 point-count change (resid 0.571→1.351).

The determinism table (`comparison.md`) confirms reads are still noisy post-#561:
mean after-std **0.230 dex**, worst 0.471/0.480/0.793, against a **0.32 dex**
noise floor.

### What
1. **Deterministic decoding:** `temperature=0.0` on all five `messages.create`
   calls via a single `_create()` wrapper + `_READ_TEMPERATURE` constant
   (P4-assertable). The SDK exposes no user seed; T=0 is the determinism lever.
2. **Multi-sample aggregation (N=3)** for the two metrology-weak reads only:
   - **Stage-2a axes** — per-endpoint **log10 median** of x/y min/max,
     majority-vote unit/scale (fixes 1403.1290 1-decade axis misreads).
   - **Curve trace** (CV `_attempt_cv_curve_trace` + LLM Stage-2 points) —
     **point-wise log10 median** on the common mass grid, keep a bin only if
     ≥2/3 runs cover it (outvotes the 1403.1290 high read and the 1905.13650
     floor-pin). Aggregation in **log10 space** matches the harness metric
     `median(log10 coupling)` (subset_compare.py:196). N=3 chosen against the
     0.32-dex floor: median-of-3 ≈ 0.72σ, ~30% spread cut at 3× the two noisy
     calls; N is a swept constant.
3. **Per-coupling convention normalizer** `normalize_convention()` in
   `pipeline/transform_guard.py` (P0's module): map paper-native → repo-canonical
   (`config.COUPLINGS[ct]["axes"]["y"]`). Rules: AxionElectron eV⁻¹ `C_e/F_a` →
   `g_ae = 2·m_e·C_e/F_a` (m_e=5.11e5 eV ⇒ ×1.022e6); Axion{Proton,Neutron} →
   `2·m_N·C_N/F_a`; AxionPhoton eV⁻¹→GeV⁻¹ ×1e-9; AxionEDM **flagged, not
   auto-converted** (ambiguous observable). Detect via Stage-2a unit label →
   notes regex → range fallback (`[1e-18,1e-12]` eV⁻¹ band). For 1902.04246:
   `5.00e-16 · 1.022e6 = 5.11e-10`, landing on GT 5.10e-10. **Guarded by P0**:
   committed only if it *improves* the benchmark ratio (R3) and holds R5.
4. **Pre-classifier vote:** `_classify_coupling_type` (extractor.py:519) draws
   N=3 at T=0, returns modal type + margin (fixes 2209.13588 type flip);
   stabilizes the `_BENCHMARK_LINES` selection P0's R3 depends on.

Every P3 output is a *candidate* committed through P0's `guard_transform`/R5
floor — P3 reduces input variance, P0 gates commit-or-revert.

### Integration points (`AAL-integration`)
- `extractor.py`: `_READ_TEMPERATURE`/`_READ_SAMPLES` consts; `temperature` on
  :529/743/1153/1231/1485; `_median_axes`/`_median_trace` wrappers around
  :1021/1025/1247; classifier vote at :519-546 (+ caller :970); convention call
  at ~:1075 before `_validate_extracted_range`.
- `pipeline/transform_guard.py`: `normalize_convention()` + `_CONVENTION_CONSTANTS`.
- `plot_calibration.py`: `_median_trace` may live beside `trace_curve`.

### Acceptance (on `subset_eval.py --repeats 3`, via `determinism_report`)
1. Mean coupling-scale std ≤ **0.150 dex** (from 0.230; floor 0.32). 2111.06883
   ≤ 0.32 (from 0.471); 1709.00009 ≤ 0.32 (from 0.480); 2410.10363 ≤ 0.40
   (from 0.793). No paper's std increases.
2. **1403.1290**: range ≤ 0.5 dex across repeats; resid ≤ master 3.699 (4.951
   gone).
3. **1902.04246**: resid ≤ 0.5 dex (4.951→0.234 recovered); normalizer fires
   (5.11e-10); std ≤ 0.1 dex.
4. **2209.13588**: coupling_type stable across repeats; resid ≤ 0.7 dex; std ≤
   0.15 dex.
5. No aggregate regression (P0 invariants hold): overall median residual ≤ 0.485
   dex; zero-overlap ≤ 14; zero `status==error`.
6. The 5 P0-protected wins preserved; normalizer is a no-op on all of them.

### Non-goals
Axis OCR metrology (P1), the full multi-candidate selector (P2), and the CI gate
(P4) are separate issues. P3 depends on P0 landing first.

### Artifacts
`evaluation/eval_runs/comparison.md`, `evaluation/eval_runs/per_paper_findings.md`,
`evaluation/eval_runs/roadmap_design/P3_read_determinism.md`,
`evaluation/eval_runs/roadmap_design/P0_failsafe_contract.md`.
