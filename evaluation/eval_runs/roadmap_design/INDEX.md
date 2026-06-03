# #566 Extractor-Correctness Roadmap — INDEX

Parent: **#566** (governs) — structural follow-up to #550 (visual-extraction) + #561
(deterministic correction). Root cause: every metrology/correction layer overwrites
the prior result with **no check it is better** and **no floor against impossible
output** (33/41 changed papers regressed; `comparison.md`). Harness for every phase:
`evaluation/subset_eval.py` + `evaluation/subset_compare.py`.

## Phases

| Phase | One-line summary | Draft sub-issue |
|---|---|---|
| **P0** `P0_failsafe_contract.md` | `transform_guard.py`: a propose→score→commit-or-revert contract (R1–R5) around every axis/curve/correction transform + the `float(None)` crash fix. | Body drafted (§6); **not yet filed** |
| **P1** `P1_axis_metrology.md` | New `axis_ocr.py`: OCR'd tick labels as a third channel; override the LLM axis only when OCR+geometry agree *and* contradict the LLM. Feeds P0's `corroborated`. | Body drafted (§6); **not yet filed** |
| **P2** `P2_extraction_selector.md` | Replace point-count source routing with one validity-first lexicographic `select_best()` over all candidates (in P0's module). | Body drafted (§6); **not yet filed** |
| **P3** `P3_read_determinism.md` | `temperature=0` on every `messages.create` + N=3 median/vote for the two noisy reads + per-coupling convention normalizer. | Body drafted (§6); **not yet filed** |
| **P4** `P4_ci_gate.md` | New `extraction_regression.yml` + `gate.py`: wire `subset_eval` as a required before/after CI gate against a committed baseline snapshot. | Body drafted (§6); **not yet filed** |

> All five sub-issue bodies exist as §6 of each design doc; **none are open on GitHub
> yet** (`gh issue list` empty). File P0 and P4 first (see sequencing).

## Sequencing & dependency order

```
        ┌──────────────── P4 (CI gate) ── lands EARLY, guards everything ┐
        │                                                                │
   P0 (contract) ──► P1 (axis OCR, feeds P0.corroborated)                │
        │        ──► P2 (selector, reuses P0 quality()/ConsistencyScore) │
        │        ──► P3 (every read-aggregate/convert is a P0 candidate)  │
   P3 (read determinism) — independent of P1/P2; only needs P0 ──────────┘
```

1. **P0 is the hard prerequisite.** It owns all commit logic and the
   `transform_guard.py` module/`ConsistencyScore`/`quality()` that P1 (corroboration
   slot), P2 (`select_best`), and P3 (guarded candidates) all import. **P0 must land
   first.**
2. **P4 should land early** (right after P0, possibly alongside). Its gate logic is
   validatable **today** against the committed `before/`/`after/` snapshots with **no
   API calls**, so it can be merged before P1–P3 to guard them. P0 re-pins the
   baseline; P1–P3 can then only improve from that floor.
3. **P1 and P2 both depend only on P0** and are mutually independent — either order.
   P1 strengthens P2's `corroborated`/`in_valid_ranges` signals if it lands first,
   but neither blocks the other.
4. **P3 is independent of P1/P2** (different failure class D — read/convention
   nondeterminism, where there is *no transform* for P0 to revert). It needs only P0.
5. Recommended merge order: **P0 → P4 → {P1, P2, P3 in any order}.**

## Cross-cutting concerns / overlaps / conflicts

- **Shared module = single source of truth.** P0, P2, and P3's normalizer all live
  in `pipeline/transform_guard.py`. P2's 7-tuple `quality()` is a strict superset of
  P0's 5-tuple (adds T1 non-degenerate, T2 recoverable); both read the same
  `ConsistencyScore`. **Risk:** P2's extended tuple and P0's tuple must not diverge —
  keep one `quality()`, with P0's `guard_transform` using the 5-tuple subset.
- **Shared thresholds.** The R4 constants (`y_const` 0.05 dex, `span_dex` 1.0),
  corroboration band `[1/3,3]`, and the 0.32 dex noise floor recur across P0/P2/P3/P4.
  Define each once (P0 owns R-constants; P4 owns the floor) — duplication is a drift
  hazard.
- **`coupling_recoverable` double-counts** the validator's `_choose_discrete_factor`
  (P2 §5) — factor into one shared helper.
- **P4 must be re-pinned after each of P0–P3 lands** (the baseline floor moves), and
  P1/P3 new module filenames (`axis_ocr.py`, normalizer) must be added to P4's `paths`
  trigger. P4's `pull_request_target`/fork-secret handling is an open question.
- **No genuine conflicts.** The 5 protected wins (1207.3275, 1508.01798, 1708.06367,
  1804.05750, 2402.12892) are a regression guard every phase must not break;
  2402.12892 (0.62-dex axis override) is the binding constraint forcing P0's R2 floor
  *above* 0.62 dex and P1's corroboration to commit it.

## Phase → acceptance metric moved

| Phase | Primary eval metric it moves (vs master `before`) |
|---|---|
| **P0** | Overall median resid 0.842→**≤0.485**; zero-overlap 32→**≤14** (`unit_offset` 12→0); `status==error` →**0** (crash gone); HARD-floor violations →**0**. |
| **P1** | figure_vision row: median **≤0.55 dex**, `≤0.3 dex` **≥40%**; recovers the 4 class-A axis papers (1506.08082, 1907.05475, 2212.01139, 1907.11485). |
| **P2** | Routing: 2102.08764/1905.13650/1808.02340/2204.01454/2007.04899 → correct source/compared; `too_few_points` →**≤7**; text/figure_vision rows non-regressing. |
| **P3** | Determinism mean coupling-scale std 0.230→**≤0.150 dex**; convention fix 1902.04246 (5.79→≤0.5), drift 1403.1290, classifier-flip 2209.13588. |
| **P4** | None directly — it *locks in* P0–P3's gains as a permanent CI invariant (reproduces the known FAIL on the #550/#561 snapshot, PASS on a no-op). |
