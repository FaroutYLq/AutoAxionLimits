# P2 — Principled best-extraction selector (replaces point-count routing)

> Roadmap phase **P2** of issue **#566** (governs) / **#550** Phase 2 follow-up.
> Design/scoping only — no pipeline code in this doc.
> Acceptance harness: `evaluation/subset_eval.py` + `evaluation/subset_compare.py`.
> Code designed against: `AAL-integration/pipeline/extractor.py`,
> `AAL-integration/pipeline/plot_calibration.py`, `AAL-integration/pipeline/config.py`.
> Builds **on top of P0** (`pipeline/transform_guard.py`): P2 reuses P0's
> `quality()` tuple and `ConsistencyScore` verbatim and generalizes P0's two-way
> "augment-not-override" rule into a multi-candidate argmax.

P2 replaces the **two** ad-hoc routing decisions in `run_extraction_agent` —
the Stage-1 short-circuit (`stage1_ok`, extractor.py:988-992 → :996) and the
CV-trace override gate (`len(traced) >= max(8, stage1_points)`,
extractor.py:1031) — with **one** selector that collects *all* candidate
extractions for a paper, scores each on the same objective signals, and emits
the argmax. The decision stops being "whichever read produced more points" and
becomes "whichever read is most valid, most internally consistent, and most
confident."

---

## 1. Problem & evidence

### 1.1 The two routing bugs P2 fixes

The integration branch regressed the 82-paper subset partly because **point
count is used as a proxy for quality** in two places, and point count is
uncorrelated with correctness. The 41-paper fan-out
(`per_paper_findings.md`) gives four clean, opposite-direction failures that a
single principled selector resolves:

**Bug 1 — sparse text short-circuits a correct vision read** (Stage-1 gate too
permissive, extractor.py:988-992):

- **1808.02340** (`per_paper_findings.md` L163-166): master used the **text**
  path (5 points, the correct bosonic-DM `g_ae` window 8.00e+02..5.00e+05 eV,
  resid 1.003). The after-run instead judged "Stage 1 insufficient", routed to
  vision, and the CV path traced the **wrong figure** (Fig. 3 solar flat line,
  not Fig. 5), emitting masses 9.06e-06..1.05e+00 eV — 8 decades low,
  zero-overlap. Here vision was *worse* than a perfectly good text read.
- **2204.01454** (L178-181): the inverse. Master's Stage-1 text returned
  reasoning-only prose (no JSON) → fell to **vision**, which got the correct
  29-point window 1.00e-19..3.00e-12 eV (resid 3.816, compared). The after-run's
  Stage-1 text "succeeded" with **4 crude points at conf=0.42**, short-circuited
  vision, and produced a zero-overlap window with an unrecoverable coupling
  (`8.6e+13`, no decade factor recovers it). A 4-point conf-0.42 text read should
  never have outranked the 29-point figure read.
- **1704.05189** (L28-31), **2007.04899** (L115-122): same pattern — Stage-1 text
  "succeeded" with 2-3 points at conf 0.55 and blocked the richer vision
  extraction master used. 2007.04899 then compounded with a #561 snap, but the
  *root* routing error is the thin text read winning.

The binding constant is `MIN_DATA_POINTS_TEXT = 3` (extractor.py:33) combined
with `extraction_confidence >= 0.4` (extractor.py:991): a 3-4-point conf-0.42
read passes `stage1_ok` and never lets vision run at all (vision is in the
`else` branch, extractor.py:1004). The gate has **no validity check** (is the
window in `VALID_RANGES`? is the coupling recoverable?) and **no comparison to
what vision would have produced**.

**Bug 2 — a CV trace overrides a *better* text/LLM extraction on point count
alone** (override gate too permissive, extractor.py:1031):

- **2102.08764** (L173-176): master extracted the EXACT ground-truth point limit
  from **text** (`m_a=3.31e-5 eV`, `g_aee<2.6e-6`, source=text) and logged
  "Vision returned fewer points (2) than text (2); keeping text". The after-run's
  CV trace returned **268 points** of the **DFSZ benchmark line** (not the
  experimental limit). At extractor.py:1031, `268 >= max(8, 2)` is True, so the
  trace **overrode the correct text limit**, flipping source to figure_vision
  with coup_median 8.94e-17 — **10.5 dex** wrong. The spot-check screamed
  `ratio=1.6e10` and was ignored. Point count (268) beat correctness.
- **1905.13650** (L168-171): `trace_curve` latched onto the bottom frame — **293
  points all at the identical `log10 y = −8.9258`**, spanning a single mass
  decade (`log10 x = −22.0..−21.003`). At extractor.py:1031, `293 >= max(8, 27)`
  is True, so this **degenerate floor-pinned line overrode a correct 27-point LLM
  boundary** (mass 1e-22..1e-13 eV, resid 0.342). A trace with one distinct mass
  column and zero coupling variance has no business outranking anything.

### 1.2 What the selector must get right (the discriminating cases)

| Paper | Correct candidate | Wrong candidate that won | Why point-count failed |
|---|---|---|---|
| 1808.02340 | text, 5 pts, in-range | vision, wrong-panel, 8 dex low | vision had ≥ points but wrong window |
| 2204.01454 | vision, 29 pts, in-range | text, 4 pts conf 0.42, coupling unrecoverable | text "succeeded" sparse + low-conf |
| 2102.08764 | text, 2 pts, exact GT | CV trace, 268 pts of benchmark line | CV `268 ≥ 8` |
| 1905.13650 | LLM boundary, 27 pts | CV trace, 293 pts, y-constant | CV `293 ≥ max(8,27)` |
| 2007.04899 | vision, 40 pts, in-range | text, 3 pts conf 0.55 | text short-circuit |

The selector's scoring **must rank all five correct candidates above their
respective wrong winners** with no ground truth available. The signals that do
this are exactly the ones P0 already computes (`ConsistencyScore`): in-range
validity, benchmark/spot-check corroboration, source-tier trust, confidence,
and non-degenerate shape. P2 is the place those signals route the **source
choice**, not just gate a single transform.

---

## 2. Design

### 2.1 Core idea: collect candidates, score each, argmax once

Today `run_extraction_agent` makes the source decision **imperatively and
early**: `stage1_ok` (extractor.py:988) decides text-vs-vision *before vision
runs*, and the CV override (extractor.py:1031) mutates `stage1_result` in place.
P2 inverts this into **collect → score → select**:

```
candidates: list[Candidate] = []

# always available if Stage 1 produced anything parseable
candidates.append(make_candidate("text", stage1_result, ...))

# run vision UNCONDITIONALLY when a Stage-1 text read is sparse/low-validity
# (see §2.4 — vision is no longer in an else-branch)
if should_consider_vision(stage1_result):
    axis_info     = _run_stage2a_axes(...)        # P0-guarded axes
    stage2_result = _run_stage2(...)
    candidates.append(make_candidate("figure_vision", stage2_result, axis_info))
    traced = _attempt_cv_curve_trace(axis_info)
    if traced is not None:
        candidates.append(make_candidate("cv_trace", {...traced...}, axis_info))

chosen = select_best(candidates)     # P2's selector — single argmax
```

`select_best` lives in the **new P0 module `pipeline/transform_guard.py`** (so
P2 adds *no* new file and *no* new dependency — pure stdlib + `math`), exposed
as:

```python
@dataclass(frozen=True)
class Candidate:
    source: str                  # "table" | "text" | "figure_vision" | "cv_trace"
    data_points: list[tuple[float, float]]
    coupling_type: str | None
    extraction_confidence: float
    score: ConsistencyScore      # the P0 dataclass, computed per-candidate

def quality(c: Candidate) -> tuple:      # the P0 tuple (§2.5 of P0), reused verbatim
    ...

def select_best(candidates: list[Candidate]) -> tuple[Candidate, str]:
    """Return (winner, reason). Pure function of the candidates; deterministic."""
```

`make_candidate` populates `ConsistencyScore` by reusing the *already-computed*
numbers P0 wired up:
- `in_valid_ranges` from `VALID_RANGES[ct]` on `_sorted_median(mass)` /
  `_sorted_median(coupling)` (config.py:320-334; helper at extractor.py:635).
- `benchmark_ratio` / `spotcheck_ratio` from `_calibrate_vision_data`
  (extractor.py:793, :817) for vision/CV candidates; `None` for text.
- `y_const` / `span_dex` / `n_points` computed directly from the candidate's
  points.
- `axis_disagree_dex` from `_calibrate_axis` (extractor.py:1369) for CV
  candidates that proposed an axis override.

### 2.2 The validity-first ordered score (tie-break ladder)

The selector is a **lexicographic argmax** over the `quality()` tuple — the same
ordered tuple P0 defines, **extended** with the degenerate-shape and
recoverability guards so the discriminating cases in §1.2 resolve correctly:

```python
quality(c) = (
    1 if c.score.in_valid_ranges else 0,          # T0  HARD validity — first
    1 if not c.score.y_const and
         c.score.span_dex >= 1.0 else 0,          # T1  non-degenerate shape
    1 if coupling_recoverable(c) else 0,          # T2  coupling in/near VALID_RANGES
    SOURCE_TIER[c.source],                        # T3  table=3 > text=2
                                                  #     > figure_vision=1 > cv_trace=0
    1 if c.score.corroborated else 0,             # T4  benchmark/spot-check in [1/3,3]
    round(c.extraction_confidence, 2),            # T5  LLM self-confidence
    c.score.n_points,                             # T6  point count — LAST resort
)
```

`select_best` returns `max(candidates, key=quality)` with deterministic
tie-breaking (ties broken by `source` lexical order, then arXiv-stable index)
so it is a **pure deterministic function** of the candidates — no run-to-run
spread introduced (acceptance §4.7).

**Tier semantics and why each tier sits where it does:**

- **T0 in_valid_ranges (HARD, first).** This is the P0 R5 floor promoted to the
  *top* discriminator. **2204.01454**'s 4-point text read has coupling `8.6e+13`,
  far outside `VALID_RANGES["AxionEDM"]` coupling `(1e-40, 1e-15)`, with "no
  decade factor recovers it" — `in_valid_ranges = False` → T0=0, so the 29-point
  in-range **vision** candidate (T0=1) outranks it regardless of point count.
  Fixes Bug 1 (2204.01454) at the first tier.
- **T1 non-degenerate shape.** `y_const` = all couplings within **0.05 dex**;
  `span_dex` = `log10(mass_hi/mass_lo) < 1.0`. **1905.13650**'s 293-point CV
  trace is `y_const=True` (every point `log10 y=−8.9258`) **and** `span_dex≈0.997
  < 1.0` → T1=0, so the 27-point LLM boundary (T1=1) wins even though the trace
  has 11× more points. Fixes Bug 2 (1905.13650). The 0.05-dex / 1.0-decade
  thresholds are the **P0 R4** constants, reused so P0 and P2 agree byte-for-byte.
- **T2 coupling_recoverable.** Separates "in-range" from "recoverable by a single
  decade factor" for couplings that sit *just* outside range (catches the residue
  of unrecoverable-coupling reads like 2204.01454's `8.6e+13` and 1907.11485's
  `3.3e-47` that the existing code already flags with "no decade factor recovers
  it", extractor.py:954-956). Implemented by re-using the same
  `_choose_discrete_factor` search the validator runs; T2=1 iff some factor lands
  the median inside strict `VALID_RANGES`.
- **T3 SOURCE_TIER (`table=3 > text=2 > figure_vision=1 > cv_trace=0`).** This is
  the **semantics-trust** ordering from P0 §2.5, *given equal validity and
  shape*. A clean text point-limit is more trustworthy than a hallucinated/traced
  pixel boundary. **2102.08764**: the text candidate (T3=2, in-range, exact GT)
  and the CV-trace candidate (T3=0, the DFSZ benchmark line) tie on T0/T1/T2, so
  T3 decides → text wins, the 268-point benchmark-line trace loses. Fixes Bug 2
  (2102.08764). `cv_trace` is split out *below* `figure_vision` because a CV
  pixel trace with no LLM semantic check (which-curve / which-panel) is the
  least-trusted of all when it ties on validity — it is exactly the source that
  produced the 2102.08764 and 1905.13650 disasters.
- **T4 corroborated.** `benchmark_ratio ∈ [1/3, 3]` OR `spotcheck_ratio ∈ [1/3,
  3]` (the P0 corroboration band, 0.48 dex). Breaks ties between two figure
  candidates: the one whose benchmark line / spot-check lands correctly wins.
- **T5 extraction_confidence**, rounded to 2 dp so float jitter can't reorder.
- **T6 n_points** — the *only* place point count appears, as the final
  tie-break. This is the single line that today (extractor.py:1031) is the
  *primary* gate; P2 demotes it to the bottom of the ladder.

### 2.3 Why this ranks all five discriminating cases correctly

| Paper | Winner under P2 | Decisive tier |
|---|---|---|
| 1808.02340 | text (5 pts, in-range) over wrong-panel vision | vision candidate: traced-flat-line is `y_const` → **T1=0** (and 8-dex-low window → **T0=0**); text T0=1,T1=1 wins |
| 2204.01454 | vision (29 pts, in-range) over 4-pt text | text coupling `8.6e+13` out of range → **T0=0**; vision T0=1 wins |
| 2102.08764 | text (2 pts, exact GT) over 268-pt CV benchmark trace | both T0/T1=1; **T3** text(2) > cv_trace(0) |
| 1905.13650 | LLM boundary (27 pts) over 293-pt CV trace | CV trace `y_const` & span<1 dec → **T1=0**; boundary T1=1 wins |
| 2007.04899 | vision (40 pts) over 3-pt text | text + #561 snap lands coupling/window out of range → **T0=0**; vision in-range T0=1 wins |

All five flip the **opposite direction from point count** — which is precisely
why a count-based gate could never get them all right.

### 2.4 Replacing the Stage-1 short-circuit: `should_consider_vision`

The Stage-1 gate (`stage1_ok`, extractor.py:988-992) currently does double duty:
it decides *both* "did Stage 1 work" *and* "should we skip vision". P2 splits
these. Stage 1 still runs and produces a **text candidate** whenever it returns
parseable data; but **vision is no longer gated behind `else`** — it runs (and
contributes a candidate) whenever the text candidate is *not clearly dominant*:

```python
def should_consider_vision(text_cand) -> bool:
    # run vision unless the text candidate is already strong:
    return not (
        text_cand.score.in_valid_ranges            # valid window AND
        and text_cand.score.n_points >= 5          # not sparse AND
        and text_cand.extraction_confidence >= 0.6 # confident
        and not text_cand.score.y_const            # non-degenerate
    )
```

The thresholds are calibrated from the evidence:
- **`n_points >= 5`** (not the current `MIN_DATA_POINTS_TEXT = 3`): 2204.01454's
  4-point and 2007.04899's/1704.05189's 2-3-point text reads must **not**
  suppress vision. 1808.02340's correct text read has 5 points and stays.
- **`extraction_confidence >= 0.6`** (not the current `0.4`): 2204.01454
  (conf 0.42), 1704.05189 / 2007.04899 (conf 0.55) all fall below 0.6, so vision
  runs and gets a fair shot in the selector. This matches
  `LOW_CONFIDENCE_THRESHOLD = 0.6` (config.py:341) — the project's existing line
  for "trustworthy".
- **`in_valid_ranges`**: a text read whose coupling can't fit `VALID_RANGES` (the
  `8.6e+13` case) never blocks vision.

Crucially, `should_consider_vision` only decides whether to *spend the API call*
to produce a vision candidate; it never decides the winner. If text is genuinely
strong (in-range, ≥5 pts, conf ≥0.6, non-degenerate), vision is skipped for cost
(preserving the text-first cost discipline in CLAUDE.md "Text-first extraction")
**and** text is the sole candidate, so it is selected. If text is weak, vision
runs and the selector picks on merit. This means 1808.02340 (strong text) costs
no extra API call, while 2204.01454 (weak text) does — exactly the right
trade-off.

### 2.5 Composition with the per-transform guards (P0)

P2 is the **source selector**; P0 is the **per-transform guard**. They compose:

1. Each candidate is produced by transforms that P0 already guards (axis override
   reverted if >1.0 dex uncorroborated; multiplier reverted if spot-check/
   benchmark out of band; snap reverted unless improve-and-in-range). So by the
   time a candidate reaches the selector, its **own** internal transforms are
   already validate-or-revert.
2. P2 then chooses **between** the guarded candidates. A CV-trace candidate whose
   P0 guards reverted everything (e.g. axis override rejected) still exists as a
   candidate but carries `in_valid_ranges`/`corroborated` flags reflecting its
   un-transformed state — so it competes honestly.
3. After selection, the chosen candidate flows through the **final R5 HARD-floor
   assert** P0 installs after extractor.py:1098. P2 does not duplicate that floor;
   it relies on T0 to *prefer* in-range candidates and on P0's assert to *forbid*
   committing an out-of-range one when no in-range candidate exists.

The clean division: **P0 guarantees no single transform makes a candidate worse;
P2 guarantees we pick the best candidate.** Neither subsumes the other —
2102.08764 needs P2 (the text candidate must beat the CV trace; P0's two-way rule
already covers this specific case, but P2 generalizes it so a *third* candidate,
the LLM-vision boundary, also competes), and 1506.08082 needs P0 (a single axis
transform reverted; no source choice involved).

### 2.6 Determinism

`select_best` and `quality` are pure functions of the candidate list. T5 rounds
confidence to 2 dp and T6 is an integer, so float jitter in the LLM reads cannot
reorder candidates unless they are genuinely near-tied. This keeps P2 inside the
harness noise floor (0.32 dex, `comparison.md` L32) and does not regress the
integration branch's `--repeats 3` mean coupling-scale std (0.230 dex). The
upstream *read* nondeterminism (1403.1290's 7-25× coupling drift, 1902.04246's
convention flip) is **out of P2's scope** — it is P3 — but P2's validity-first
ranking means a drifted read that lands out of `VALID_RANGES` is at least
deselected rather than committed.

---

## 3. Integration points (exact files / functions)

All edits are in the `AAL-integration` worktree, against `run_extraction_agent`
(extractor.py:960-1132) and the new selector functions in
`pipeline/transform_guard.py` (the P0 module).

### 3.1 `pipeline/transform_guard.py` (P0 module — P2 additions)
- Add `Candidate` dataclass, `SOURCE_TIER` dict (`{"table":3,"text":2,
  "figure_vision":1,"cv_trace":0}`), `coupling_recoverable()`, and `select_best()`.
- Extend the existing P0 `quality()` tuple from §2.5-P0 with the **T1
  non-degenerate** and **T2 recoverable** tiers (P0's tuple is the
  `(source_tier, in_valid_ranges, corroborated, confidence, n_points)` subset;
  P2's is the full 7-tuple above). P0's two-candidate `guard_transform` keeps
  using its 5-tuple; P2's `select_best` uses the 7-tuple. Both read the same
  `ConsistencyScore` fields, so there is one source of truth for the signals.
- Unit-testable like the #544 metric tests: feed synthetic candidate lists
  (the five §2.3 cases as fixtures) and assert the winner.

### 3.2 `run_extraction_agent` — the Stage-1 gate (extractor.py:986-1012)
- Replace the `stage1_ok` early decision (extractor.py:988-992) with: always
  build the **text candidate** from `stage1_result`; compute
  `should_consider_vision(text_cand)` (§2.4). Stage 1 still short-circuits the
  *PDF-too-short* path (extractor.py:976) unchanged.
- The `if stage1_ok:` / `else:` branch structure (extractor.py:996-1072)
  collapses: vision now runs inside an `if should_consider_vision(...)` block
  that **appends candidates** rather than overwriting `stage1_result`.

### 3.3 `run_extraction_agent` — the CV-trace override gate (extractor.py:1024-1049)
- Delete the point-count override at extractor.py:1031
  (`len(traced_points) >= max(8, stage1_points)`) and the
  `stage2_points > stage1_points` branch (extractor.py:1039) and the
  "Vision returned fewer points" branch (extractor.py:1045-1049). All three are
  subsumed by `select_best`.
- `_attempt_cv_curve_trace` (extractor.py:1398-1426) is **unchanged** — it still
  *proposes* a trace; P2 wraps its output in a `cv_trace` Candidate (T3=0) instead
  of overriding on count. The `>= 8` floor inside it (extractor.py:1422) stays as
  a cheap "is there anything to trace" guard.
- After `chosen, reason = select_best(candidates)`, copy
  `chosen.data_points`/`source`/`coupling_type`/`extraction_confidence` back into
  `stage1_result` (the existing downstream contract at extractor.py:1074-1132 is
  untouched), and append `reason` to `notes` (single compact line, e.g.
  `selector: figure_vision over text (text out of VALID_RANGES)`).

### 3.4 Composition with later phases
- **P0** is a hard prerequisite: `select_best` reads `ConsistencyScore`, which P0
  defines and populates. P2 cannot land before P0.
- **P1** (axis OCR) strengthens the `corroborated` (T4) and `in_valid_ranges`
  (T0) signals for figure/CV candidates — a CV candidate with OCR-corroborated
  axes scores higher, so P1 improves *which* figure candidate wins without
  changing the selector.
- **P3** (read determinism + convention normalizer) reduces the rate at which a
  drifted read lands out of range; P2's validity-first ranking is the safety net
  that deselects the residue P3 doesn't catch (e.g. 1902.04246's convention flip
  lands `5e-16` *inside* `VALID_RANGES`, so P2 alone cannot catch it — that is
  explicitly P3's job, noted as an open question §5).
- **P4** (CI gate) asserts the P2 invariant: no committed extraction is a
  `cv_trace` that lost to an in-range higher-tier candidate (a regression
  tripwire on the selector).

---

## 4. Acceptance criteria (measured on `subset_eval.py`)

Run, on `AAL-integration` with **P0+P2** applied (P2 requires P0):

```
python -m evaluation.subset_eval extract --key union \
    --outdir evaluation/eval_runs/after_p2
python -m evaluation.subset_eval compare \
    --before evaluation/eval_runs/before \
    --after  evaluation/eval_runs/after_p2 \
    --out evaluation/eval_runs/comparison_p2.md
```

P2 is accepted iff, versus the **master baseline** (`before`):

1. **The four routing failures are fixed** (the binding cases):
   - `2102.08764`: `data_source == "text"`, status `compared` (not figure_vision
     resid 10.455). 
   - `1905.13650`: `data_source == "figure_vision"` (LLM boundary, not the
     293-pt CV trace), status `compared`, resid ≤ master's **0.342**.
   - `1808.02340`: `data_source == "text"`, status `compared`, resid ≤ master's
     **1.003** (not zero-overlap).
   - `2204.01454`: `data_source == "figure_vision"`, status `compared`, resid
     ≤ master's **3.816** (not zero-overlap).
   - `2007.04899`: `data_source == "figure_vision"` (40-pt vision, not 3-pt
     text), not zero-overlap.
2. **Zero-overlap papers ≤ master's 14** (from the integration branch's 32); the
   four routing ZOs above clear. `too_few_points` cause count ≤ master's **7**
   (the sparse-text-short-circuits-vision cases no longer strand at ≤4 points).
3. **No `cv_trace` candidate is committed when a higher-tier in-range candidate
   exists** (selector invariant): a check script over `after_p2` JSONs asserts
   every figure-sourced commit either (a) won T0/T1 honestly or (b) had no
   in-range text/table alternative. **0 violations.**
4. **figure_vision per-source row does not regress**: median residual ≤ master's
   **0.642 dex**; `≤0.3 dex` fraction ≥ master's **33.3%**. (P2 routes *away*
   from figure_vision when text is stronger, so this row should *improve* — the
   1905.13650 / 2102.08764 garbage figure reads leave the bucket.)
5. **text per-source row does not regress**: median residual ≤ master's
   **0.375 dex**; `≤0.3 dex` fraction ≥ master's **47.8%**. (P2 routes *into*
   text only when it wins on validity, so the bucket should not dilute.)
6. **The 5 genuine wins are preserved** (not re-routed away): `1207.3275`,
   `2402.12892` (both figure_vision, must stay figure_vision and keep their
   improved resids 2.134 / 0.102), `1708.06367` (text→figure_vision win must
   survive — vision is the in-range candidate), `1508.01798`, `1804.05750`.
7. **Determinism unchanged**: `--repeats 3` on the routing-sensitive key; mean
   coupling-scale std ≤ the integration branch's **0.230 dex** (selector is a
   pure deterministic function; T5 rounding prevents jitter-reorders).

Stretch (informational): overall median residual ≤ **0.485 dex** (the headline
P0 target; P2 contributes the routing share of it).

---

## 5. Risks & open questions

- **Cost of running vision more often.** Loosening the Stage-1 gate
  (`should_consider_vision`) means vision runs on more papers (every text read
  with <5 pts OR conf <0.6 OR out-of-range). On the subset that is the ~6-8
  sparse-text papers; in production it could be more. Mitigation: vision only
  runs when text is *not clearly dominant*, and the strong-text fast path (§2.4)
  keeps 1808.02340-style cases at one API call. Open question: should
  `should_consider_vision` also require the paper to *have* a figure (reuse
  `extract_figures_from_pdf` non-empty) before spending the vision call? Likely
  yes — flagged for implementation.
- **`coupling_recoverable` (T2) double-counts with the validator.** T2 re-runs
  the `_choose_discrete_factor` search that `_validate_extracted_range` also runs.
  Risk of divergence if one is changed without the other. Mitigation: factor the
  search into one helper in `transform_guard.py` that both call (it is pure).
- **In-range but wrong-convention reads (1902.04246).** A text read in the
  *wrong coupling convention* (`C_e/F_a` eV⁻¹ vs dimensionless `g_ae`) can land
  *inside* `VALID_RANGES` (5e-16 ∈ `(1e-20, 1e0)`), so T0 won't catch it and the
  selector may pick it over a correct figure read. This is **explicitly P3's
  job** (convention normalizer); P2 must not be expected to fix it. Note it so
  P4's CI gate doesn't attribute a 1902.04246 regression to P2.
- **Ties producing arbitrary winners.** If two candidates tie through T6
  (identical validity, shape, tier, corroboration, confidence, point count),
  the lexical-source tie-break decides. This should be vanishingly rare
  (table/text/figure/cv are different tiers at T3), but the tie-break must be
  documented so a reviewer isn't surprised by `cv_trace` losing a coin-flip it
  shouldn't have entered.
- **Threshold calibration N is small.** `n_points >= 5`, `conf >= 0.6`, the
  0.05-dex `y_const` and 1.0-decade `span_dex` are set from ~6-8 papers. They are
  shared with P0 (R4) and the project's `LOW_CONFIDENCE_THRESHOLD`, which
  constrains drift; P4's CI gate keeps them honest as the subset grows. P2 should
  land **with** the eval re-run, not on argument alone.
- **Where the selector lives.** Proposed inside the P0 `transform_guard.py` (one
  module, shared `ConsistencyScore`/`quality`). Alternative: a separate
  `pipeline/selector.py`. The shared module is preferred so the signals have a
  single source of truth and P0/P2 thresholds can't silently diverge.

---

## 6. Draft sub-issue body

> **Title:** `[extractor][P2] Principled best-extraction selector: replace point-count source routing with a validity-first score`

**Parent:** #566 (governs). Builds on **P0** (`transform_guard.py`, required
prerequisite). Refines #550 Phase 2.

### Why
The pipeline chooses its data source by **point count** in two places, and point
count is uncorrelated with correctness. The 41-paper fan-out
(`evaluation/eval_runs/per_paper_findings.md`) gives four opposite-direction
failures a single selector resolves:
- **2102.08764**: a 268-point CV trace of the **DFSZ benchmark line** overrode a
  correct 2-point **text** limit because `268 >= max(8, 2)` (extractor.py:1031);
  spot-check `ratio=1.6e10` ignored. resid → 10.455.
- **1905.13650**: a 293-point **degenerate** trace (all couplings at the identical
  `log10 y=−8.9258`, one mass decade) overrode a correct 27-point LLM boundary,
  because `293 >= max(8, 27)`.
- **1808.02340**: a correct 5-point **text** read (in-range window) was demoted to
  vision, which traced the wrong panel → 8 decades low, zero-overlap.
- **2204.01454**: a 4-point conf-0.42 **text** read short-circuited a correct
  29-point **vision** read (`MIN_DATA_POINTS_TEXT = 3`, conf gate 0.4); the text
  coupling `8.6e+13` was unrecoverable. Also **1704.05189**, **2007.04899**
  (sparse text at conf 0.55 blocking 40-point vision).

### What
Collect **all** candidate extractions (`text`, `figure_vision`, `cv_trace`) and
select the argmax of a **validity-first lexicographic score** in P0's
`pipeline/transform_guard.py` (no new file, no new deps):

```
quality(c) = (
  in_valid_ranges,                 # T0 HARD validity (config.py:320-334)
  not y_const and span_dex>=1.0,   # T1 non-degenerate (P0 R4 constants)
  coupling_recoverable,            # T2 some decade factor lands median in range
  SOURCE_TIER[source],             # T3 table=3>text=2>figure_vision=1>cv_trace=0
  corroborated,                    # T4 benchmark/spot-check in [1/3,3]
  round(extraction_confidence,2),  # T5
  n_points,                        # T6 — point count, LAST resort only
)
```

This ranks all five discriminating cases the **opposite direction from point
count**: 2204.01454 (text out of range → T0), 1905.13650 (CV trace degenerate →
T1), 2102.08764 (text > cv_trace → T3), 1808.02340 / 2007.04899 (text fast-path /
vision-wins).

**Replace the Stage-1 short-circuit** (`stage1_ok`, extractor.py:988-992) with
`should_consider_vision`: run vision unless text is already strong (in-range AND
`n_points >= 5` AND `confidence >= 0.6` AND non-degenerate). Thresholds:
`>= 5` (not `MIN_DATA_POINTS_TEXT = 3`) and `>= 0.6` (the existing
`LOW_CONFIDENCE_THRESHOLD`, config.py:341), so 2204.01454 (4 pts), 1704.05189 /
2007.04899 (conf 0.55) no longer suppress vision.

**Delete the CV-trace override gate** `len(traced) >= max(8, stage1_points)`
(extractor.py:1031) and the two adjacent count branches; all subsumed by
`select_best`. `_attempt_cv_curve_trace` is unchanged (still *proposes*).

### Composition
- **P0** is required: `select_best` reads P0's `ConsistencyScore` and reuses its
  `quality()` tuple/thresholds (R4 shape, R5 floor, corroboration band).
- **P1** strengthens T0/T4 for figure candidates (OCR-corroborated axes).
- **P3** owns in-range wrong-**convention** reads (1902.04246) that T0 can't
  catch.
- **P4** asserts the selector invariant in CI.

### Acceptance (on `evaluation/subset_eval.py`, vs master baseline)
1. 2102.08764 → `text`/compared; 1905.13650 → figure_vision LLM-boundary/compared
   (resid ≤ 0.342); 1808.02340 → `text`/compared (≤ 1.003); 2204.01454 →
   `figure_vision`/compared (≤ 3.816); 2007.04899 → `figure_vision`/not-ZO.
2. Zero-overlap papers ≤ **14**; `too_few_points` ≤ **7**.
3. **0** committed `cv_trace` when an in-range higher-tier candidate existed
   (selector invariant; check script over `after_p2` JSONs).
4. figure_vision row: median resid ≤ **0.642 dex**, `≤0.3 dex` ≥ **33.3%**.
5. text row: median resid ≤ **0.375 dex**, `≤0.3 dex` ≥ **47.8%**.
6. The 5 genuine wins preserved (1207.3275, 2402.12892, 1708.06367, 1508.01798,
   1804.05750), not re-routed away.
7. `--repeats 3` mean coupling-scale std ≤ **0.230 dex** (selector is pure
   deterministic).

### Non-goals
No new models/deps. Axis OCR (P1), read determinism + convention normalizer
(P3 — owns 1902.04246/1403.1290), and the CI gate (P4) are separate issues. P2
selects among candidates; P0 guards each transform within a candidate.

### Artifacts
`evaluation/eval_runs/comparison.md`,
`evaluation/eval_runs/per_paper_findings.md`,
`evaluation/eval_runs/roadmap_design/P0_failsafe_contract.md`,
`evaluation/eval_runs/roadmap_design/P2_extraction_selector.md`.
