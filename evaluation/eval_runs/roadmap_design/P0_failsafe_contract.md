# P0 — Fail-safe transform contract (validate-or-revert)

> Roadmap phase **P0** of issue **#566** (governs) / **#550** Phase 1 foundation.
> Design/scoping only — no pipeline code in this doc.
> Acceptance harness: `evaluation/subset_eval.py` + `evaluation/subset_compare.py`.
> Code designed against: `AAL-integration/pipeline/extractor.py`,
> `AAL-integration/pipeline/plot_calibration.py`, `AAL-integration/pipeline/config.py`.

P0 is the **foundation** that P1 (axis OCR metrology), P2 (best-extraction
selector), P3 (read determinism + convention normalizer) and P4 (CI gate) all
build on. P0 introduces *no new models and no new dependencies*. It is a thin
**contract layer**: every metrology/correction transform must declare a
candidate output, the contract scores it against paper-internal consistency
checks, and the transform is **committed only if it does not fail any check** —
otherwise the pre-transform state is restored byte-for-byte (revert).

---

## 1. Problem & evidence

### 1.1 The meta-pattern

The #550+#561 integration branch regressed the 82-paper subset hard
(`evaluation/eval_runs/comparison.md`):

| Metric | Before (master) | After (#550+#561) | Δ |
|---|---|---|---|
| Overall median residual | 0.485 dex | 0.842 dex | **+0.357** |
| Zero-overlap papers | 14 | 32 | **+18** |
| Mean interp. coverage | 63.9% | 35.4% | **−28.5 pp** |
| Papers compared | 45 | 24 | **−21** |

The 41-agent per-paper fan-out (`per_paper_findings.md`) shows **33/41 changed
papers moved *away* from ground truth**, and the new zero-overlap bucket is
dominated by `unit_offset` (0→12) and `wrong_window` (7→13). The root cause is
not any single layer; it is that **every transform overwrites the prior result
with no check that the overwrite is better, and no floor against physically
impossible output**. P0 fixes exactly that: it makes each transform
*validate-or-revert*.

The 5 genuine improvements in the eval (`1207.3275`, `1508.01798`, `1708.06367`,
`1804.05750`, `2402.12892`) are precisely the cases where the transform happened
to be validatable — benchmark line landed (`2008.10141` before-run KSVZ ratio
1.4 → no-op), or the spot-check ratio snapped to identity
(`1508.01798` raw_ratio 1.00; `2402.12892` raw_ratio 0.83;
`1207.3275` raw_ratio 0.76). **P0 makes that validation *mandatory* for all
transforms instead of accidental.**

### 1.2 Which eval failures P0 fixes (with citations)

P0 directly targets failure classes **A** (unsafe CV axis calibration, 7 papers),
**B** (unsafe CV curve-trace override, routing), and **the crash** (1 paper). It
also installs the *hard floor* that prevents the #561 soft-anchor snaps (class E,
20 papers) from emitting out-of-window data, even before P3 fixes the anchor
logic itself. Concretely:

**Class A — CV axis correction overrides good LLM axes by many decades:**
- **1506.08082**: `_attach_cv_calibration` wrote `y-max 2e-6 → 1e+22 GeV⁻¹` (24
  decades; an *unphysical* coupling axis). Downstream #561 then snapped ×1e-1.
  resid 0.043 → 1.03. (`per_paper_findings.md` L131-134.)
- **1907.05475**: `x-min 1e-21 → 2.10e-33`, `x-max 1e-08 → 8.09e-18` (~12
  decades); the spot-check then ran at 6.31e-26 eV instead of the correct
  3.00e-12 eV. resid 0.073 → 1.065. (L136-139.)
- **2207.11968**: `x-max 1000 → 320910`; CV-traced couplings ~1e-39; spot-check
  `verify=7e-13 / stage2=2.11e-39, ratio=3.3e26` — **yet not rejected**. resid
  2.115 → ∞ (zero-overlap). (L151-154.)
- **2212.01139**: a single mislocated bottom tick gave `y-min 1e-10 → 2.26e-11`
  (~4.4×), re-routing the traced curve ~1 dex off GT. resid 1.262 → 2.281.
  (L156-159.)
- **1907.11485**, **2008.10141** (`KSVZ` benchmark read a full decade off,
  ratio 0.16 → spurious ×1e-1), **1410.7267** (CV path *dropped* master's 0.02
  spot-check factor) round out class A.

**Class B — CV curve-trace override replaces a better extraction:**
- **1905.13650**: `trace_curve` latched onto the bottom frame — 293 points with
  identical `log10 y = −8.9258`, spanning a single mass decade — overriding a
  correct 27-point LLM boundary. Gate fired only because `293 ≥ max(8,…)`.
  (L168-171.)
- **2102.08764**: traced the **DFSZ benchmark line**, not the experimental
  limit; overrode a correct *text* point-limit (`268 ≥ 8`). Spot-check
  `ratio=1.6e10` — **not rejected**. resid → 10.455. (L173-176.)
- **1607.06083 (crash)**: CV-trace override → `_run_vision_verify` returned an
  explicit JSON `null` in `boundary_at_mass`; the unguarded
  `float(spot.get("mass_eV", 0))` in `_calibrate_vision_data` raised
  `float(None)`, killing the whole extraction. (L185-188.)

**Class C/D context (handed off to P2/P3 but the floor helps now):** the
soft-anchor cluster (e.g. **2007.13071** Hz→eV ×4.136e-15 dragging a correct
GeV-scale collider window 14 decades down; **2308.06339**, **1709.00009**,
**2401.16747**, **2403.03004 VectorBL**, **2410.19902 AxionMass**) are all the
**same disease**: a transform applied with no in-window / improve check. P0's
hard floor (§2.4) catches the subset of these whose corrected median lands
*outside* VALID_RANGES; the rest are fully resolved by P3's anchor fix, but P0's
contract is what makes that fix safe to land.

---

## 2. Design

### 2.1 Core idea: a `TransformGuard` contract around each layer

Today each transform mutates `data_points` / `axis_info` in place and trusts
itself. P0 wraps every transform in a uniform **propose → score → commit-or-revert**
envelope. Nothing about *what* a transform computes changes; only whether its
output is *accepted*.

New module **`pipeline/transform_guard.py`** (pure Python, stdlib + `math` only —
no new deps, Apache-2.0-compatible with the repo). It exposes:

```python
@dataclass(frozen=True)
class ConsistencyScore:
    in_valid_ranges: bool          # HARD floor (§2.4): median in VALID_RANGES
    benchmark_ratio: float | None  # expected/reported on a known model line
    spotcheck_ratio: float | None  # verify / stage2 at the spot-check mass
    axis_disagree_dex: float | None  # |log10(CV axis) - log10(LLM axis)|, max over endpoints
    n_points: int
    span_dex: float                # log10(mass_hi/mass_lo) of the candidate
    y_const: bool                  # all couplings within 0.05 dex (degenerate line)

def passes_contract(score, *, corroborated: bool) -> tuple[bool, str]:
    """Return (accept, reason). A candidate is accepted ONLY if it violates
    no reject rule in §2.3/§2.4."""

def guard_transform(before, after, *, score_after, corroborated, label):
    """Return (committed_value, note). Commits `after` iff passes_contract,
    else returns `before` with a 'reverted' note. Never raises."""
```

The two signals already exist in the code and are simply *read* rather than
*ignored*:
- **benchmark ratio**: `_calibrate_vision_data` already computes
  `ratio = expected / bm_coupling` (extractor.py:793) and logs it; today it only
  *gates application* of a multiplier (`0.01 < ratio < 100`). P0 reuses the same
  number as a *consistency score* on the whole candidate.
- **spot-check ratio**: `spot_ratio = spot_coupling / closest[1]`
  (extractor.py:817). Same reuse.
- **VALID_RANGES**: `config.VALID_RANGES` (config.py:320-334) — already imported
  by `_validate_extracted_range` (extractor.py:860).
- **axis disagreement**: `_calibrate_axis` already computes
  `disagree = abs(log10(meas_v) - log10(llm_v))` (extractor.py:1369) and writes
  the CV value back whenever `disagree > 0.5`. P0 turns that one-sided "always
  overwrite if > 0.5 dex" into a *gated* overwrite (§3.2).

### 2.2 The four objective, paper-internal consistency checks

These are "is this better?" signals that require **no ground truth** (so they are
usable in production, not just in eval):

1. **Benchmark-line agreement.** If a known model line is present
   (`_BENCHMARK_LINES`: KSVZ for AxionPhoton, DFSZ_upper for AxionElectron,
   SolarConstraint for DarkPhoton, KSVZ_neutron for AxionNeutron — extractor.py
   :556-561), the candidate's reading of that line must land near its known
   value: `benchmark_ratio = expected/reported ∈ [1/3, 3]` (≤ 0.48 dex). This is
   the strongest signal and is what made `2008.10141`'s *before* run correctly
   no-op (ratio 1.4 ✅) and its *after* run wrong (ratio 0.16 ❌, snapped ×1e-1).

2. **Spot-check verify/stage2 ratio → 1.** The Stage-3 re-read of the boundary
   at the mid-mass must agree with the traced/extracted curve there:
   `spotcheck_ratio ∈ [1e-2, 1e2]` to even be *considered* (the band already in
   extractor.py:822), and within `[1/3, 3]` to count as *corroborating*. Ratios
   like **1.6e10** (2102.08764) and **3.3e26** (2207.11968) are catastrophic
   failures and must HARD-reject the candidate (§2.3).

3. **OCR-tick-vs-LLM-label agreement** *(fully defined in P1; referenced here as
   the corroboration signal for axis overrides)*. When P1 lands, a CV axis
   override is only "corroborated" if the OCR'd tick *labels* match the LLM's
   reported tick *values* and the geometric fit agrees. In P0 (pre-P1) the
   available proxy is `build_log_transform`'s fit quality (`r2 ≥ 0.95`,
   plot_calibration.py:426-428) plus the spot-check; the contract is written to
   accept a richer corroboration object from P1 without changing its interface.

4. **In-VALID_RANGES membership** (the HARD floor, §2.4): the candidate's
   `_sorted_median` mass and coupling must lie in `VALID_RANGES[ct]`.

### 2.3 Concrete reject thresholds (calibrated from the evidence)

| Rule | Reject when | Evidence anchor |
|---|---|---|
| **R1 spot-check blow-up** | `spotcheck_ratio` outside `[1e-2, 1e2]` | 2207.11968 ratio 3.3e26; 2102.08764 ratio 1.6e10 — both must auto-reject (today both *applied*). The `[1e-2,1e2]` band already exists for *gating the multiplier* (extractor.py:822) — P0 promotes it to *gating the whole candidate*. |
| **R2 axis-override without corroboration** | a CV axis endpoint disagrees with the LLM axis by `> 1.0 dex` **and** the override is not corroborated (no OCR-tick match in P1; pre-P1: no passing spot-check) | 1907.05475 x 1e-21→2.1e-33 (12 dex); 1506.08082 y 2e-6→1e22 (24 dex). Threshold = **1.0 dex** for *rejecting*, distinct from the existing **0.5 dex** that merely *triggers* an override (extractor.py:1370). I.e. 0.5–1.0 dex overrides are allowed but must be corroborated; > 1.0 dex requires strong corroboration or is reverted. |
| **R3 benchmark disagreement** | `benchmark_ratio` outside `[1/3, 3]` (0.48 dex) | 2008.10141 ratio 0.16 (0.80 dex) snapped a correct curve ×1e-1; the *before* ratio 1.4 (0.15 dex) correctly idled. A full-decade benchmark disagreement means the curve itself is mis-scaled, not that a ×0.1 fix is warranted. |
| **R4 degenerate trace** | `y_const` (all couplings within 0.05 dex) **or** `span_dex < 1.0` (single-decade x) for a *curve* candidate | 1905.13650: 293 points all at `log10 y=−8.9258`, x-span 0.997 dex — a floor-pinned line. Today accepted on point-count alone. |
| **R5 HARD floor (no exceptions)** | candidate median mass **or** coupling outside `VALID_RANGES[ct]` (strict, not the ×0.1/×10 widened window) | catches the residue of the soft-anchor cluster whose snap lands outside range (e.g. 2207.11968 coupling 1.4e-19 → flagged; couplings 3.3e-47 in 1907.11485 already warned but still emitted). |

All thresholds are in **dex** (log10) and chosen to sit *between* the genuine
improvements and the regressions:
- Benchmark/spot-check **corroboration** band `[1/3, 3]` = 0.48 dex sits above
  the genuine no-ops (0.15 dex for ratio 1.4; 0.0 for raw_ratio 1.00) and below
  the regression (0.80 dex for ratio 0.16).
- Axis **reject** at 1.0 dex sits above the legitimate corrections (e.g.
  `2402.12892` x-max 30→124 = 0.62 dex, a *genuine improvement*, so we must
  *allow* 0.5–1.0 dex with corroboration) and below the catastrophic 12–24 dex
  blow-ups.

`2402.12892` (genuine improvement, x-max 30→124.29 = 0.62 dex) is the binding
constraint: it proves the axis-reject floor must be **above 0.62 dex**, hence
**1.0 dex with required corroboration in [0.5, 1.0]**, not a blanket 0.5 dex
revert. The spot-check there snapped to identity (raw_ratio 0.83) — i.e. the
override *was* corroborated — so it passes the contract.

### 2.4 The HARD floor (R5) — no layer may emit out-of-VALID_RANGES values

This is the one rule with **no corroboration escape hatch**. After every
transform and at the *final* stage, the committed `data_points` must satisfy:

```
mass_lo  ≤ _sorted_median(masses)    ≤ mass_hi     # strict VALID_RANGES[ct]["mass"]
coup_lo  ≤ _sorted_median(couplings) ≤ coup_hi     # strict VALID_RANGES[ct]["coupling"]
```

If a transform's output violates this and the *pre-transform* state satisfied it,
**revert**. If neither satisfies it (the raw read was already bad), keep the
pre-transform state and emit a `[LOW CONFIDENCE]`-flagging note rather than a
silently-corrupted value. This is intentionally stricter than today's
`[mass_lo*0.1, mass_hi*10]` widened window used by `_validate_extracted_range`'s
`in_window` (extractor.py:903): the widened window is for *choosing a correction
factor*; the floor is for *rejecting a result*.

### 2.5 "Never override a higher-confidence extraction" (the contract; full selector is P2)

P0 defines the **comparison rule**; P2 builds the full multi-candidate selector
on top of it. The rule:

A transform may **augment a lower-confidence extraction but never override a
higher-confidence one.** Quality is compared by an ordered tuple, *not* by raw
point count (the current `len(traced) >= max(8, stage1_points)` is exactly the
bug):

```
quality(candidate) = (
    source_tier,            # table=3 > text=2 > figure_vision=1  (semantics-trust)
    in_valid_ranges,        # True > False
    benchmark_or_spotcheck_corroborated,  # True > False
    extraction_confidence,  # the LLM's own 0..1 (extractor.py:1129)
    n_points,               # ONLY as the last tie-break
)
```

A CV-trace candidate (`source_tier=1`) therefore can **never** override a text
point-limit (`source_tier=2`) — which is exactly the `2102.08764` and
`2007.04899` regressions (text overridden by a worse vision/CV trace). The
existing `Vision returned fewer points (%d) than text` branch (extractor.py
:1045-1049) shows the codebase already *has* the notion of "keep the better
one"; P0 generalizes it from point-count to the quality tuple.

### 2.6 The `float(None)` crash fix (1607.06083)

In `_calibrate_vision_data`, the verify/benchmark readings come straight from an
LLM JSON response and can contain explicit `null`. `dict.get(key, default)`
returns `None` (not the default) for an explicit-`null` key, so
`float(spot.get("mass_eV", 0))` (extractor.py:805-806) and the benchmark variant
(extractor.py:789-790) raise `float(None)`. Fix with a None-tolerant coercion:

```python
def _safe_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default
```

Apply at extractor.py:789-790 (`bm_mass`, `bm_coupling`) and 805-806
(`spot_mass`, `spot_coupling`). This is *latent in master too* (the agent verified
`git show master:extractor.py` has the same unguarded lines), so the fix is
correct independent of #550; #550's CV-trace override merely makes the crashing
path reachable more often. The fix belongs in P0 because the contract layer calls
`_calibrate_vision_data` and must not be able to throw.

### 2.7 Data flow (after P0)

```
_run_stage2a_axes
   └─ _attach_cv_calibration                 # proposes CV axis override
        └─ guard_transform(before=LLM axes, after=CV axes,
                           score: axis_disagree_dex, spotcheck/r2 corroboration)
              → commit CV axes only if R2 passes, else KEEP LLM axes   ← class A fix
_run_stage2  (uses committed axes)
_attempt_cv_curve_trace                       # proposes CV-traced points
   └─ guard_transform(before=LLM boundary, after=CV trace,
                      score: R4 degenerate, R5 floor, quality tuple §2.5)
         → commit CV trace only if quality(trace) > quality(LLM)        ← class B fix
            AND R4/R5 pass
_run_vision_verify → _calibrate_vision_data   # _safe_float guards (§2.6)  ← crash fix
   └─ guard_transform(before=raw curve, after=calibrated curve,
                      score: R1 spot-check band, R3 benchmark band)
         → commit multiplier only if R1 & R3 pass                       ← 2008.10141 fix
_validate_extracted_range                     # #561 snaps
   └─ guard_transform(before, after=snapped,
                      score: R5 HARD floor; improve-vs-revert)
         → commit snap only if it moves median TOWARD anchor AND stays in floor
final HARD-floor assert (R5) on committed data_points                    ← E residue fix
```

Every box is `propose → guard → commit-or-revert`. No box can make the result
worse on its own consistency signal.

---

## 3. Integration points (exact files / functions)

All edits are in the `AAL-integration` worktree (the branch that carries both
#550 and #561), against the functions named in the brief.

### 3.1 `_calibrate_vision_data` (extractor.py:755-853)
- Add `_safe_float` helper (module scope, near `_sorted_median` at :635) and use
  it at :789-790 and :805-806 — **crash fix (§2.6)**.
- Wrap the multiplier application (:834-839) in `guard_transform`: the
  `raw_ratio` is already computed; **reject** (keep raw `data_points`) when R1
  (`spotcheck_ratio ∉ [1e-2,1e2]`) or R3 (`benchmark_ratio ∉ [1/3,3]`) fails,
  instead of the current single `0.01 < ratio < 100` *application* gate. Emit a
  `Calibration reverted (…)` note so it surfaces in the PR body (the note already
  flows to `stage1_result["notes"]` at :1094).

### 3.2 `_attach_cv_calibration` / `_calibrate_axis` (extractor.py:1256-1396)
- The inner `_calibrate_axis` currently writes the CV value back whenever
  `disagree > 0.5` (extractor.py:1370-1372). Change to: compute `disagree`; if
  `disagree ≤ 0.5` → no-op (as today); if `0.5 < disagree ≤ 1.0` → override only
  if **corroborated** (pre-P1: the axis's `build_log_transform` r2 ≥ 0.95
  already passed *and* — when available — the downstream spot-check agrees);
  if `disagree > 1.0` → **revert to LLM axis** unless P1's OCR-tick corroboration
  is present. This is **R2** and fixes 1506.08082 / 1907.05475 / 2212.01139.
- Record both the proposed and committed values in `cv_calibration_note` so the
  eval log (`per_paper_findings` reads these exact strings) shows
  `reverted`/`committed`.

### 3.3 The CV-trace override gate (extractor.py:1025-1038)
- Replace `if traced_points and len(traced_points) >= max(8, stage1_points)`
  (:1031) with `if guard_accepts_cv_trace(traced_points, stage1_result, score)`:
  - R4 degenerate-trace reject (`y_const` or `span_dex < 1.0`) → fixes 1905.13650.
  - quality-tuple comparison (§2.5): CV trace `source_tier=1` cannot override a
    `text` (`source_tier=2`) extraction → fixes 2102.08764 / 2007.04899.
  - R5 HARD floor on the traced median.
- `_attempt_cv_curve_trace` (extractor.py:1398-1426) is unchanged (it still
  *proposes*); only the *gate* at :1031 changes from accept-on-count to
  accept-on-contract.

### 3.4 `_validate_extracted_range` (extractor.py:856-957)
- Keep the existing soft/hard trigger machinery, but route the *committed* result
  through `guard_transform` with the **R5 HARD floor** and an explicit
  improve-or-revert: the snap is committed only if `dist_after < dist0` **and**
  the corrected median lies in strict `VALID_RANGES` (not the widened window).
  This is the generalization of the #561 soft-anchor fix (`fix/561-soft-anchor-regression`)
  to *all* snaps, and it is what makes P3's anchor changes safe.
- Add a **final floor assert** right after the call site (extractor.py:1098): if
  the committed `data_points` median is out of strict range, revert to the last
  in-range state and tag the result low-confidence.

### 3.5 Composition with later phases
- **P1** plugs its OCR-tick corroboration object into the `corroborated` argument
  of `guard_transform` for axis overrides (§2.2 check 3). P0 ships with the
  pre-P1 proxy (`r2 ≥ 0.95` + spot-check) so it is useful immediately and P1 only
  *strengthens* the corroboration.
- **P2** consumes the `quality()` tuple (§2.5) verbatim and extends it to a
  multi-candidate argmax across {text, vision-LLM, CV-trace}; P0's two-way
  "augment-not-override" rule is the degenerate 2-candidate case.
- **P3**'s anchor / convention-normalizer changes are *gated by P0's R5 floor and
  improve-or-revert*, so a wrong anchor can no longer emit out-of-range data.
- **P4** asserts the P0 invariants (no R5 violations; no `float(None)` crash) as
  required CI checks via `subset_eval.py`.

---

## 4. Acceptance criteria (measured on `subset_eval.py`)

Run, on the `AAL-integration` branch with P0 applied:

```
python -m evaluation.subset_eval extract --key union \
    --outdir evaluation/eval_runs/after_p0
python -m evaluation.subset_eval compare \
    --before evaluation/eval_runs/before \
    --after  evaluation/eval_runs/after_p0 \
    --out evaluation/eval_runs/comparison_p0.md
```

P0 is accepted iff, versus the **master baseline** (`before`, the current
`comparison.md` "Before" column):

1. **No net regression — overall median residual** ≤ **0.485 dex** (recover from
   the integration branch's 0.842). Target: ≤ 0.485.
2. **Zero-overlap papers** ≤ **14** (recover from 32). The 12 new `unit_offset`
   ZOs and the routing ZOs (1905.13650, 2102.08764, 2007.04899) must clear.
   Target `unit_offset` cause count: **0** (from 12); `wrong_window`: ≤ 7.
3. **The crash is gone**: `1607.06083` returns a non-`error` status (it was
   `error` in `after`; master gave 28 figure_vision points). Zero papers with
   `status == error`.
4. **figure_vision per-source row does not regress**: median residual ≤ master's
   **0.642 dex** (from the integration branch's 1.002) and `≤0.3 dex` fraction
   ≥ master's **33.3%** (from 25.2%).
5. **No HARD-floor violations**: every committed extraction has its
   `_sorted_median` mass and coupling inside strict `VALID_RANGES` (assert in a
   tiny check script over the `after_p0` JSONs; 0 violations).
6. **Genuine improvements preserved** (regression guard on the 5 wins): the
   contract must NOT revert `1207.3275` (axis 0.62 dex, spot-check 0.76→identity),
   `2402.12892` (x 30→124 = 0.62 dex, corroborated), `1508.01798`,
   `1708.06367`, `1804.05750`. Their `after_p0` residuals must stay ≤ their
   integration-branch `after` values.
7. **Determinism unchanged or better**: re-run with `--repeats 3` on the
   `unit_offset` key; mean coupling-scale std ≤ the integration branch's
   **0.230 dex** (P0 is deterministic — pure functions of the readings — so it
   must not add spread). Noise floor 0.32 dex per the harness.

Stretch (informational, mostly P1/P2 territory but P0 should not block):
figure_vision `≤0.3 dex` fraction trending toward the `text` row (45–48%).

---

## 5. Risks & open questions

- **R5 strictness vs. legitimately heavy/light masses.** Strict VALID_RANGES is
  wide (`AxionPhoton` mass 1e-24..1e9) precisely so collider ALPs and ultralight
  haloscopes both fit; so R5 alone won't catch the soft-anchor cluster
  (`2007.13071` snapped *within* range). That is intended — R5 is the floor; the
  *improve-or-revert* in §3.4 plus P3's anchor fix do the rest. Open question:
  should P0 also revert any snap that *crosses* the extracted axis-unit's implied
  scale (e.g. Stage-2a x in GeV ⇒ never multiply mass by Hz→eV)? That is a
  cheaper version of P3 and could be folded into P0's `_validate_extracted_range`
  guard; flagged for the P0/P3 boundary.
- **Corroboration before P1.** Pre-P1, the only axis corroboration is `r2 ≥ 0.95`
  + spot-check. `1506.08082`'s bad CV fit may still pass r2 if the mislocated
  tick is self-consistent; the > 1.0 dex revert (R2) is the backstop there, but
  we should confirm on the eval that R2 alone clears all 7 class-A papers without
  P1. (If `2212.01139`'s 0.64-dex y-min shift sneaks under 1.0 dex, it needs the
  spot-check corroboration check; verify empirically.)
- **Threshold calibration N is small.** The `[1/3,3]` and `1.0 dex` bands are set
  from ~10–15 papers. P4's CI gate is what keeps them honest as the subset grows;
  P0 should land *with* the eval re-run, not on argument alone.
- **Where does the contract live — guard module vs. inline?** Proposed a single
  `transform_guard.py` for testability (unit-testable pure functions, matching the
  existing `evaluation` metric unit tests from #544). Alternative: inline helpers
  in `extractor.py`. The module is preferred so P2's selector can import the same
  `quality()`.
- **Notes/PR-body verbosity.** Reverts add notes; ensure they don't bury the
  human reviewer. Suggest a single compact `contract: reverted axis(>1dex) |
  reverted snap(floor)` summary line.

---

## 6. Draft sub-issue body

> **Title:** `[extractor][P0] Fail-safe transform contract: validate-or-revert for every axis/curve/correction layer`

**Parent:** #566 (governs). Foundation for P1–P4. Refines #550 Phase 1.

### Why
The #550+#561 integration regressed the 82-paper subset (median residual
0.485→0.842 dex, zero-overlap 14→32, coverage 63.9%→35.4%;
`evaluation/eval_runs/comparison.md`). The 41-paper fan-out
(`per_paper_findings.md`) shows **33/41 changed papers moved away from truth**.
Root cause: **every metrology/correction layer overwrites the prior result with
no check that the overwrite is better and no floor against impossible output.**
The 5 genuine wins are exactly the cases where a transform *happened* to be
validatable. P0 makes that validation **mandatory**.

### What
Introduce a uniform **propose → score → commit-or-revert** contract
(`pipeline/transform_guard.py`, pure stdlib, no new deps) wrapping every
transform. A transform commits **only if** it passes objective, paper-internal
consistency checks; otherwise the pre-transform state is restored.

**Consistency signals (all already computed in code, just ignored):**
benchmark-line agreement (`expected/reported`, extractor.py:793), spot-check
verify/stage2 ratio (extractor.py:817), CV-axis-vs-LLM-axis disagreement in dex
(extractor.py:1369), and in-`VALID_RANGES` membership (config.py:320-334).
OCR-tick-vs-LLM-label agreement is defined in P1 and plugs into the
`corroborated` slot.

**Reject thresholds (calibrated from evidence):**
- **R1** spot-check ratio outside `[1e-2, 1e2]` → reject. Cites **2207.11968**
  (ratio 3.3e26) and **2102.08764** (ratio 1.6e10), both currently *applied*.
- **R2** CV axis endpoint disagreeing with LLM axis by **> 1.0 dex** without
  corroboration → revert to LLM axis. Cites **1907.05475** (x 1e-21→2.1e-33,
  12 dex) and **1506.08082** (y 2e-6→1e22, 24 dex). The genuine win
  **2402.12892** (x 30→124, 0.62 dex, corroborated) sets the floor *above*
  0.62 dex → 1.0 dex with corroboration in [0.5, 1.0].
- **R3** benchmark ratio outside `[1/3, 3]` → reject. Cites **2008.10141**
  (KSVZ ratio 0.16 snapped a correct curve ×1e-1; before-run ratio 1.4 correctly
  idled).
- **R4** degenerate trace (all couplings within 0.05 dex, or x-span < 1 decade)
  → reject. Cites **1905.13650** (293 points all at `log10 y=−8.9258`).
- **R5 HARD floor (no exceptions):** no layer may emit a median mass/coupling
  outside strict `VALID_RANGES`. Final assert after `_validate_extracted_range`.

**Never override a higher-confidence extraction:** replace the point-count gate
`len(traced) >= max(8, stage1_points)` (extractor.py:1031) with a quality tuple
`(source_tier, in_valid_ranges, corroborated, extraction_confidence, n_points)`;
`table > text > figure_vision`, so a CV trace can't override a text point-limit
(**2102.08764**, **2007.04899**).

**Make each existing transform revert on failure:**
- CV-trace override gate (extractor.py:1031) → R4 + R5 + quality tuple.
- `_attach_cv_calibration`/`_calibrate_axis` (extractor.py:1370) → R2.
- `_calibrate_vision_data` (extractor.py:834) → R1 + R3.
- `_validate_extracted_range` (extractor.py:902-957) → R5 + improve-or-revert.

**Crash fix:** add `_safe_float` and use it at extractor.py:789-790 and 805-806
so an explicit JSON `null` in `boundary_at_mass`/`benchmark_line` can't raise
`float(None)` (**1607.06083**). Latent in master too; #550's CV-trace path makes
it reachable.

### Acceptance (on `evaluation/subset_eval.py`, vs master baseline)
1. Overall median residual ≤ **0.485 dex** (from 0.842).
2. Zero-overlap papers ≤ **14** (from 32); `unit_offset` cause → **0** (from 12).
3. Zero papers with `status == error` (1607.06083 recovers).
4. figure_vision row: median resid ≤ **0.642 dex**, `≤0.3 dex` ≥ **33.3%**.
5. Zero HARD-floor (R5) violations across all `after_p0` JSONs.
6. The 5 genuine wins are NOT reverted (1207.3275, 2402.12892, 1508.01798,
   1708.06367, 1804.05750).
7. `--repeats 3` mean coupling-scale std ≤ **0.230 dex** (P0 is deterministic).

### Non-goals
No new models/deps. Axis OCR (P1), full multi-candidate selector (P2), read
determinism + convention normalizer (P3), and the CI gate (P4) are separate
issues that build on this contract.

### Artifacts
`evaluation/eval_runs/comparison.md`, `evaluation/eval_runs/per_paper_findings.md`,
`evaluation/eval_runs/roadmap_design/P0_failsafe_contract.md`.
