# Failure Analysis — `after_roadmap` Extraction Benchmark

## 1. Headline

- **Total failures analyzed:** 38
- **Genuine extractor faults:** 20 (53%)
- **Ground-truth / benchmark issues (not extractor fault):** 18 (47%)
- **Recoverable (fixable=true):** 33 (87%)
- **Unrecoverable / not fixable now:** 5 (13%) — `1110.2895`, `1708.06367`, `2102.08764`, `2112.03439`, `2208.07293`

Of the 18 non-extractor-fault cases, all are GT labeling/convention problems except the two structural data-version mismatches (`2102.08764`) and contradictory observable metadata (`1708.06367`, `2208.07293`, `2112.03439`, `1110.2895`). The benchmark is roughly evenly split between "the extractor read the wrong thing" and "the GT told it to compare against the wrong thing."

## 2. Failures by Root Cause (ranked by papers recoverable)

| Root cause | Count | Recoverable | arXiv IDs |
|---|---|---|---|
| gt_benchmark_issue | 14 | 10 | 1110.2895*, 1509.00476, 1706.00209, 1708.06367*, 1711.08999, 1806.00310, 2111.06883, 2112.03439*, 2207.11968, 2208.06519, 2208.07293*, 2209.12901, 2211.08439, 2307.03878, 2309.07995 |
| source_misrouting | 12 | 11 | 0807.2926, 1406.6053, 1508.01798, 1606.03145, 1709.00009, 1712.00483, 1808.02340, 1907.11485, 2007.04899, 2007.04990, 2102.08764*, 2112.12116 |
| convention_units | 6 | 6 | 1410.7267, 1606.07001, 2006.07055, 2101.01241, 2204.01454, 2410.19902 |
| figure_extraction | 3 | 3 | 1403.1290, 2110.14406, 2209.12901 |
| comparator_encoding | 1 | 1 | 2007.03694 |
| text_truncation | 1 | 1 | 1804.05750 |
| vision_trace_drift | 1 | 1 | 2401.18076 |

\* not recoverable (fixable=false)

Note: `gt_benchmark_issue` count (14) excludes `2209.12901`, which is classified `figure_extraction`; the list above shows 15 IDs because the diagnosis text for `2209.12901` is reproduced under figure_extraction. The authoritative tally is **gt_benchmark_issue = 14, figure_extraction = 3**.

## 3. Highest-Yield Next Levers (ranked)

### Lever A — Quality-tier source selection beats point-count voting (recovers ~6)
**Papers:** 1406.6053, 1808.02340, 2007.04899, 1712.00483, 0807.2926, 2204.01454
**Fix:** Make `read_vote.select_consensus()` (`pipeline/read_vote.py`) rank candidates by `transform_guard.quality()` lexicographic tier instead of raw point count, so sparse text/reference benchmarks (≤3 pts) are demoted below a valid multi-point `figure_vision` curve. Concretely: raise `_SPARSE_POINT_LIMIT_MAX` from 2→3 (`transform_guard.py` ~line 112) and lock `data_source_expected='table'` papers to table tier (never demoted below vision). This is the single largest recoverable cluster — it directly addresses "wrong source won the vote" failures.

### Lever B — GT benchmark repair / convention metadata fixes (recovers ~10)
**Papers:** 1509.00476, 1706.00209, 1711.08999, 1806.00310, 2111.06883, 2207.11968, 2208.06519, 2211.08439, 2307.03878, 2309.07995
**Fix:** Targeted edits to `evaluation/ground_truth/papers.json` and regenerated data files: correct mismatched `coupling_type`/`coupling_convention`/`reference_repo_file` entries, add missing entries (ScalarNucleon for 2111.06883, VectorBL for 2309.07995, AxionElectron for 2307.03878), regenerate corrupted GT data (2207.11968 via `populate_data_from_repo()`), and add single-point reference routing (`gt_point_reference`) for 1706.00209/1806.00310/2208.06519. Zero extractor changes — pure data hygiene, high confidence, mechanical.

### Lever C — Convention converters in `evaluation/conventions.py` (recovers ~5)
**Papers:** 1410.7267, 2006.07055, 2101.01241, 2204.01454, 2410.19902
**Fix:** Add bidirectional converters to `to_canonical()` and tokens to `classify_reported_convention()`: Yukawa alpha→d (`500*sqrt(alpha)`), d_e_large↔d_me for ScalarElectron, d_n [e·cm]↔g_d [GeV^-2] for AxionEDM, and 1/f_a [GeV]↔m_a [eV] axis mapping. Also add per-coupling ceilings to `_COUPLING_CEILINGS` in `metrics.py` (AxionEDM). Unblocks convention_mismatch/no_comparable_gt that are pure mathematical-gap, not extraction errors.

### Lever D — Multi-curve disambiguation in vision (recovers ~5)
**Papers:** 1403.1290, 1508.01798, 2110.14406, 2209.12901, 1606.07001
**Fix:** In Stage 2 vision (`pipeline/extractor.py` curve-selection ~line 525-640): when multiple curves coexist, prefer projection/sensitivity curves over existing-exclusion boundaries (1403.1290, 2209.12901), prefer single-experiment over compilation figures (2110.14406, 1709.00009), and explicitly name the spot-check curve in `_run_vision_verify()` to stop cross-talk calibration (1606.07001). Add a post-hoc mass-range cross-check against `reference_repo_file` metadata.

### Lever E — Mass-independent / flat-bound encoding (recovers 1, cheap)
**Paper:** 2007.03694
**Fix:** Encode mass-independent bounds as `[(1e-30, g),(1e4, g)]` instead of `[(0,g1),(0,g2)]` so boundary filtering keeps positive masses (per issue #558). Trivial, isolated.

### Lever F — Read-vote mass-range sanity guard (recovers 2)
**Papers:** 1804.05750, 2401.18076
**Fix:** Add a mass-range consistency gate to medoid selection in `pipeline/read_vote.py`: reject candidate curves whose mass range drifts >0.5 dex from the paper-stated frequency/mass band (parsed from text). Stops both the tail-rescue duplicate-band confusion and the horizontal vision drift.

**Priority order: B → A → C → D → F → E.** B and A together cover ~21 of 33 recoverable papers (64%). B requires no model changes (lowest risk, immediate); A is the highest-yield extractor fix.

## 4. Appendix — Per-Paper Lines

| arXiv | status | root_cause | extractor_fault | fix (summary) |
|---|---|---|---|---|
| 0807.2926 | no_comparable_gt | source_misrouting | true | Veto rule in read-vote: force AxionElectron when vision notes say "electron Yukawa"/"g_ee" |
| 1110.2895 | no_extracted_points | gt_benchmark_issue | false | GT mislabels region-only paper as tabulated; reclassify as unextractable (not fixable) |
| 1403.1290 | compared | figure_extraction | true | Vision must trace PROJECTION curve, not existing exclusion; add curve-type verify question |
| 1406.6053 | zero_overlap | source_misrouting | false | Reject vision curve on non-mass x-axis (Y param); prefer figure over 1-pt text benchmark |
| 1410.7267 | convention_mismatch | convention_units | true | Add Yukawa alpha→d converter (500*sqrt(alpha)) to conventions.py |
| 1508.01798 | compared | source_misrouting | true | Multi-curve disambiguation: favor d_me/DUAL curve over d_e/EP via coupling hint + mass range |
| 1509.00476 | compared | gt_benchmark_issue | false | Regenerate GT from paper Fig 4; current GT data mismatched |
| 1606.03145 | no_extracted_points | source_misrouting | true | Emit synthetic AxionMass prediction point for lattice mass-window papers |
| 1606.07001 | compared | convention_units | true | Name the spot-check curve in _run_vision_verify; reject ratio>3 cross-talk calibration |
| 1706.00209 | no_comparable_gt | gt_benchmark_issue | false | Stale run; re-run eval — status should become gt_point_reference |
| 1708.06367 | compared | gt_benchmark_issue | false | Contradictory GT metadata (d_n vs g_agamman); paper shows CG/fa (not fixable) |
| 1709.00009 | compared | source_misrouting | true | figure_select: prefer single-experiment over "combined"/"lower envelope" figures |
| 1711.08999 | no_comparable_gt | gt_benchmark_issue | false | GT points to f_a file w/ wrong convention; fix to g_an or remove |
| 1712.00483 | zero_overlap | source_misrouting | true | Figure extraction failed; add MICROSCOPE/WEP keywords to _LIMIT_KEYWORDS, debug stage2 |
| 1804.05750 | zero_overlap | text_truncation | true | Dedup tail-rescue excerpts; add mass-range outlier rejection before medoid |
| 1806.00310 | zero_overlap | gt_benchmark_issue | false | GT is composite QUAX curve vs paper's single text point; split GT or flag composite |
| 1808.02340 | zero_overlap | source_misrouting | true | Quality-tier selection in read-vote so figure curve beats 1-pt text |
| 1907.11485 | compared | source_misrouting | true | Two GT entries share one g_ae file; give DarkPhoton its own epsilon GT file |
| 2006.07055 | convention_mismatch | convention_units | false | d_e_large↔d_me convention gap; add converter or relabel GT |
| 2007.03694 | zero_overlap | comparator_encoding | true | Encode flat bound as [(1e-30,g),(1e4,g)] not [(0,g1),(0,g2)] (issue #558) |
| 2007.04899 | compared | source_misrouting | true | Demote 3-pt Eot-Wash reference text below vision (_SPARSE_POINT_LIMIT_MAX 2→3) |
| 2007.04990 | no_extracted_points | source_misrouting | true | Stage1 wrongly set is_new_limit=false; allow AxionMass cosmological bounds |
| 2101.01241 | no_comparable_gt | convention_units | false | Add d_n↔g_d converter + AxionEDM ceiling in metrics.py |
| 2102.08764 | zero_overlap | source_misrouting | false | GT precise masses (33.1157) vs paper-rounded (33.117) version mismatch (not fixable) |
| 2110.14406 | no_comparable_gt | figure_extraction | true | Prefer experiment-named figure (GrAHal Fig 4) over compilation Fig 2 |
| 2111.06883 | no_comparable_gt | gt_benchmark_issue | false | Add missing ScalarNucleon GT entry (paper's primary QN·d constraint) |
| 2112.03439 | no_prediction | gt_benchmark_issue | false | GT auto-expanded from wrong file; paper has decay rates not coupling (not fixable) |
| 2112.12116 | compared | source_misrouting | true | Tie-break GT selection to prefer non-solar halo curve (XENON1T_SE not Solar_SE) |
| 2204.01454 | compared | convention_units | true | Improve table d_n parsing; lock table tier when data_source_expected='table' |
| 2207.11968 | compared | gt_benchmark_issue | false | Regenerate corrupted GT data via populate_data_from_repo() |
| 2208.06519 | no_comparable_gt | gt_benchmark_issue | false | GT has duplicate single mass; replace with 29-pt resonance curve or flag 1-pt ref |
| 2208.07293 | zero_overlap | gt_benchmark_issue | false | GT claims d_n but file is g_angamma (photon); fix/remove entry (not fixable) |
| 2209.12901 | no_comparable_gt | figure_extraction | true | Vision must extract projection haloscope curve, not Sun exclusion boundary |
| 2211.08439 | no_comparable_gt | gt_benchmark_issue | false | AxionElectron entry points to AxionNeutron file; create AE file or remove |
| 2307.03878 | no_comparable_gt | gt_benchmark_issue | false | Relabel GT entry AxionPhoton→AxionElectron, swap reference file |
| 2309.07995 | no_comparable_gt | gt_benchmark_issue | false | Add missing VectorBL GT entry (HELIOS.txt, 58 pts) |
| 2401.18076 | compared | vision_trace_drift | true | Mass-range sanity check in medoid selection vs paper-stated band |
| 2410.19902 | zero_overlap | convention_units | true | Fix 1/f_a vs m_a axis mapping to AxionMass; fix shared GT placeholder file |
