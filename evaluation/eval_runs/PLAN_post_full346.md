# Plan: post-full346 failure remediation

> **STATUS (2026-07-02): IMPLEMENTED AND MERGED.** Phases 1a–1d, 2a–2c, the
> Phase-3 design, and the #594 declaration contract landed as PRs
> #650/#652/#653/#654/#655/#656/#657 (squash-merged to master the same day);
> the #648 state repair is pushed (`chore/update-pipeline-state` @ f113d120).
> Post-merge rescore of the same snapshots: micro median 0.331 → **0.245 dex**,
> zero-overlap 28 → **10**, top-bin overconfidence +0.138 → **+0.0004**.
> Still gated on the new API key: Haiku-subset validation of #655/#657,
> Phase 2d (re-validate the 9 wrong-curve papers vs the lever-D branch first),
> and the Phase 4A full-pool re-baseline. Macro−micro follow-up: issue #658.

Source analysis: `failure_analysis_full346.md` + `failure_analysis_full346_detail.md` (67 per-paper diagnoses of the 2026-06-08 full-pool report). Implementation is expected to happen in a **fresh session** — this file is the handoff. Status trackers: issue #648 (billing outage), #636 (GPD escalation tier), #613 (lever D branch `fix/613-leverd-curve-selection` @ f931446e).

Standing constraints:
- Every eval PR reports before/after impact on the same snapshots (canon-on-vs-off isolation — never mix extraction drift into a scoring comparison).
- GT = O'Hare repo limit_data files only; benchmark exclusions must be documented, visible in the report, and reversible — never silent deletions.
- Extraction benchmarks run on Haiku (`EXTRACTOR_MODEL`) when the API key returns.

---

## Phase 1 — Benchmark repair (eval-side only; NO API needed; fully runnable in Claude Code now)

All changes touch `evaluation/` only; the extractor is untouched, so the cached full346 snapshots (`evaluation/results/*.json`, 346 papers) remain valid and the **new evaluation round is a local rescoring**: re-run `python -m evaluation.evaluate --metrics --report` before/after on the SAME snapshots. Zero API calls. This answers the "can the new eval run locally in Claude Code?" question: **yes for this phase, fully.**

### 1a. GT exclusion mechanism + documentation (the "exclude with reasons" ask)
- Schema: add `excluded: true` + `exclusion_reason` (one line) + `exclusion_evidence` (pointer into `failure_analysis_full346_detail.md`) to entries in `evaluation/ground_truth/papers.json`.
- New doc `evaluation/ground_truth/EXCLUSIONS.md`: one section per excluded entry — what the paper reports, why the GT entry cannot grade it, and what would un-exclude it.
- `evaluation/metrics.py` / `report.py`: skip excluded entries from residual statistics but list them in a dedicated report table ("Excluded GT entries (n): reason") so exclusions stay visible.
- Entries to exclude (from the analysis):
  - **AxionMass prediction-band files (11)**: 1202.5851, 1505.07455, 1509.00026, 1606.03145, 1608.05414, 1705.00676, 1906.00967, 2007.04990, 2108.05368, 2206.11598, 2412.08699 — GT files are (mass, mass) prediction bands, no coupling column; a correct extraction cannot match them. (Alternative kept open: a mass-band-overlap score instead of exclusion; decide at review.)
  - **Unfixable (3)**: 2005.14694 (limit only in published Nature version), 2011.08693 (GT digitizes private-communication data), 2112.03439 (paper reports decay rates, not couplings).

### 1b. GT repairs (fix, not exclude)
- Per-entry GT data files in `evaluation/build_ground_truth.py` when several repo files share one arXiv id (fixes 1110.2895, 2112.12116, 1806.00310 structurally).
- Convert MonopoleDipole λ[m] x-columns to eV (m = 1.973e-7/λ) at GT ingestion (1712.00483, 2011.07100, 2302.09096).
- Re-key wrong mappings: 1003.0964 (LSW_UWA.txt belongs to 1105.6169), 1410.5244 (docs/dp.md refs swapped), 2411.13701 (file stores ε, ×1e-11), 2312.11608 (→ `Projections/21cm.txt`).

### 1c. Scoring unification (port subset_compare.py logic into evaluate.py --metrics)
- Canonicalize BOTH sides via `classify_reported_convention` + `file_source_convention` → `to_canonical` (verified: 2103.03783 18.3→1.06 dex, 2401.18076 20.8→2.3 dex).
- Port `_expand_mass_independent` (1406.6053 is accurate to 0.003 dex yet scored zero-overlap).
- Score n_ext==1 via reverse pass instead of ∞ (5 papers).
- Lower-envelope preprocessing for closed-contour GT curves (2306.01048: 3.51→0.02).
- UNCONVERTIBLE guard → `convention_mismatch` status (2208.07293).

### 1d. Convention converters round 2
- Input: `GPD/explanations/coupling-convention-conversions-round2-EXPLAIN.md` (**complete**, numerically spot-checked against GT scales) + its citation audit (in flight). Only implement rows the doc marks VETTED.
- **Vetted closed forms (5 families):** f_a[GeV] reciprocal (0.02 dex on 2105.13963); Γ/τ → g_aγγ = √(64π·ħ·Γ/m³), per-point mass-dependent, 64π prefactor verified in the papers (0.10 dex on 2301.06560, 0.32 dex on 2312.11608); TWO distinct squared-axis tokens — √(4π·y) for g²/4π axes (0809.4700) vs plain √y (1508.02463), 0.55 dex apart so they must not share a token (both land at 0.02 dex); thermal-axion ξ: g = 1.395e-10·ξ·m_eV (medium confidence, model-locked to QCD axion); QSNET Λ: d = √2·M_Pl/Λ.
- **UNCONVERTIBLE:** oscillating-EDM amplitude family (ρ_DM-dependent + residual ×1.6–2.2 analysis factor) → sentinel, not a formula. The deterministic constant 1 e·cm = 1.5346e13 GeV⁻¹ (reproduces the repo's 3.7e-3 from di Cortona) belongs in the **extractor**, not the eval registry.
- **Round-1 registry BUG found while vetting (must fix in the same PR):** the family default `g_aNN_inv_gev` (×2m_N) is wrong for 6 AxionNeutron + 5 AxionProton GT files that already store dimensionless g_aN (verified: no in-code multiplier in PlotFuncs.py), incl. 0809.4700's own `K-3He_Comagnetometer.txt` → add per-file canonical overrides to `_FILE_CONVENTION`.
- **Two mandatory structural guards:** (a) magnitude guards on every token — refuse conversion when input values are outside the token's plausible range (also disambiguates Λ-in-GeV vs GeV/Λ and rejects anchor-snap-corrupted snapshots); (b) a "converted from …" declaration rule to prevent double conversion (0809.4700 declared converted but emitted raw squared values — string-keyed converters cannot fix mislabeled declarations; that is the #594 extractor contract's job).
- Guard: suppress anchor-snap when declared convention is non-canonical (convention failures were compounded by spurious snaps). Tests mirror the round-1 suite.

**Phase-1 deliverable:** one PR (or a small stack: 1a+1b, 1c, 1d) each with the before/after local rescoring block: headline metrics table + per-affected-paper residual diff. Expected effect: ~35 of 67 failure-tail papers resolve; zero-overlap count and the +0.78 dex macro–micro gap should drop sharply.

---

## Phase 2 — Pipeline fixes (need the NEW API key to validate; code can be written before)

### 2a. Billing fail-fast + state repair (#648)
- Fatal-error class for API-availability errors (400 credit-balance, 401, 403) in the extractor's retry helper; orchestrator aborts run non-zero WITHOUT marking papers processed. Optional 1-token preflight ping step in the workflow.
- State repair: remove the 85 burned ids (list in #648) from `processed_ids` on `chore/update-pipeline-state`; drain via temporarily raised `--max-papers` or a few manual dispatches. Order: fail-fast first, then repair.

### 2b. AxionMass ranges (tiny, Lever 4)
- Widen `VALID_RANGES['AxionMass']` mass floor to ~1e-24 eV; add `_EXPECTED_MASS_ANCHOR_EV['AxionMass']`; make `_validate_extracted_range` revert snaps that don't restore validity. Fixes 5 superradiance/ultralight papers.

### 2c. Text-path hygiene (Lever 6) + sparse guard (Lever 7)
- Widen `_result_excerpts` context; demote "analytic-reconstruction/approximate read" text candidates below vision in the vote; Stage-1 flat-bound 2-point encoding; Hz-only scope for the 4.136e-15 factor; scale `R4_MIN_SPAN_DEX` to figure extent.

### 2d. Vision multi-curve (Lever 5 / #613 continuation)
- FIRST re-validate the 9 wrong-curve papers against the committed lever-D branch (the full346 run may have predated parts of the prompt). Then add deterministic gates: projection-vs-named-other-experiment rejection, axis-unit/mass-regime consistency vs declared coupling + abstract window, panel targeting. Chart-extraction model remains the backstop.

**Validation:** standard Haiku subset before/after re-extraction once the key exists (see Phase 4 alternative below if we want it keyless).

---

## Phase 3 — #636: GPD escalation tier for unfamiliar conventions (the user's forward ask)

Runtime design that keeps the pipeline deterministic (GPD never runs inside a production extraction):
1. **Prerequisite (blind-spot closure):** route undeclared conventions and same-unit/different-physics cases into `convention_review_needed` (today they pass silently — the 4.6-dex AxionEDM case).
2. **Queue, don't call:** when `convention_review_needed` fires in production, the pipeline appends a structured escalation request to `pipeline/state/convention_queue.json` (paper id, declared-convention string, coupling type, sample points, target repo file) and the PR keeps its `[CONVENTION REVIEW]` cap.
3. **Local GPD drain:** a Claude Code skill (e.g. `.claude/skills/convention-triage`) reads the queue and, per NEW convention token, runs the exact workflow used twice now: `gpd:explain` (code-verified derivation, dimensional check, numeric spot-check vs GT scale) → `gpd-bibliographer` audit → on PASS, a PR that promotes the vetted converter into `evaluation/conventions.py` + `pipeline/transform_guard.py` and clears the queue entry. On FAIL → stays flagged for human.
4. **Once per convention, not per paper:** the registry grows monotonically; steady-state cost ≈ 0. Cache key = normalized convention token + coupling type.
5. Metric: `[CONVENTION REVIEW]` flag rate and convention_mismatch/misconvert count per 100 papers, tracked in the report.

---

## Phase 4 — Re-baseline evaluation round

Two options, decide at review:
- **A (default, needs new key):** fresh Haiku full-pool extraction → new report.md → update the JINST draft's calibration numbers (the TECHNICAL_REPORT calibration table is stale).
- **B (keyless, exploratory): Claude-Code-backed extraction.** Add `EXTRACTOR_BACKEND=claude_cli` to `pipeline/extractor.py` routing `messages.create` through headless `claude -p --output-format json` (subscription-billed, supports image inputs). Verdict on feasibility: **technically doable but NOT a drop-in replacement for the benchmark** — different model tier/limits than the Haiku baseline, so all before/after comparisons must be run entirely within one backend; and subscription rate limits make a 346-paper pool slow. Recommended use: unblock extractor-change validation while keyless, and as the substrate for automating Phase 3 — not for the official re-baseline.

Note the practical implication of A-vs-B: **Phase 1's rescoring round is the only evaluation that is both fully local and directly comparable to the existing report** — it should ship first regardless.

---

## Suggested execution order for the implementation session(s)

1. Phase 1a+1b (GT hygiene + exclusions, pure data) → local rescore → PR.
2. Phase 1c (scoring unification) → local rescore → PR.
3. Phase 1d once the round-2 EXPLAIN doc passes its citation audit → local rescore → PR.
4. Phase 2a when the new key lands (unblocks the daily pipeline — highest operational urgency).
5. Phase 2b–2d with Haiku subset validation.
6. Phase 3 (design PR first, then the skill).
7. Phase 4 re-baseline; refresh paper/JOURNAL_PLAN.md numbers.
