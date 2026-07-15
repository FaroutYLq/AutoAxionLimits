# Phase 4 (#625) — GT multi-plane / convention audit findings

**Date:** 2026-07-15. Scope chosen by the user: *minimal verified subset* — only
evidence-backed GT edits that make a pair cleanly comparable; no fabricated GT,
no unverified conversions. This doc records the full audit so the (small)
actionable set is auditable and the (large) non-actionable set is not silently
re-attempted.

## Part 1 — multi-plane GT ingestion: **0 actionable targets**

The plan's premise was that the confusable-pair papers (the B−L↔DarkPhoton
quartet, CASPEr/torsion, EP multi-charge) publish a second plane that upstream
already tracks but the GT failed to ingest. **The repository evidence
contradicts this.** For all 19 `no_comparable_gt` papers, checked two ways
(`limit_data/` grep and `evaluation/ground_truth/data/` file scan):

- Upstream (`limit_data/`) holds only ONE plane per paper — never the predicted
  plane. E.g. 2205.03617 → only `VectorB-L/DMStability.txt`; 2012.05427 → only
  `DarkPhoton/NeutronStarCooling.txt`; 1902.04644 → only `AxionNeutron/`
  (no AxionProton CASPEr-ZULF file exists upstream).
- The GT registers every plane whose data file exists — there is **no ingestion
  gap** (registered ⊇ data-file planes for every paper).

Therefore every `no_comparable_gt` paper is a **coupling-type misclassification**
(the extractor predicted a plane the repo does not track for that paper), NOT a
GT-coverage gap. These belong to **Phase 3** (the axis→plane cross-check), which
is designed to correct classification. Fabricating a second-plane GT to "rescue"
them would violate the exclusions discipline (never paper over an extractor
failure) and is not done.

## Part 2 — `convention_mismatch` audit (18 papers)

| class | papers | disposition |
|---|---|---|
| Phase-1 garbage exclusions | 2204.01454, 2410.02218, 2301.06560 | correct — leave excluded |
| genuine ≥15-dex convention gap (extraction in a small g_e/d_me scale vs GT large-valued curve) | 2303.00778, 2201.02042, 2108.04746, 2312.13723, 2006.07055 | correct — leave excluded (retag would create 20+ dex garbage) |
| extraction-side unconvertible / guard-refused (GT already canonical) | 1712.00483, 1708.02111, 1906.11844, 2104.12772, astro-ph/0611502, 2012.12790 | not a GT issue — extraction declared an unconvertible/out-of-guard convention |
| wrong-type (DarkPhoton pred, no scalar GT) | 1611.05852 | classification (Phase 3) |
| **false `d_e_large` exclusion — FIXED** | **2010.08107, 2212.04413** | **un-excluded (below)** |

### The two fixes (evidence-linked, reversible)

`ScalarPhoton/HQuartzSapphire.txt` (2010.08107) and `ScalarPhoton/DyQuartz.txt`
(2212.04413) carry the header `# mass [eV]  d_e` — they ARE canonical `d_e`.
Their `fill_between` polygon TOP wall (7.6e5 / 1e10, a plotting artifact) trips
the `ymax > 1e3 → d_e_large` range heuristic, so the comparator was excluding a
**canonical-d_e vs canonical-d_e** pair as a fake convention gap. That is exactly
the forbidden "exclude because the extractor fails" pattern: both papers have a
genuine (if mediocre, ~1.6–1.9 dex) `d_e` extraction that the false tag hid.

Fix: an explicit header-canonical override (`infer_convention` +
`_HEADER_CANONICAL_DE_FILES`, and the two `papers.json` entries retagged
`d_e_large → d_e`). Distinguished from a genuinely large-valued curve
(`ScalarElectron/WhiteDwarfs.txt`, uniformly ~2.6e6, extraction 22 dex off —
correctly kept excluded) by the header declaration + wide fill-region dynamic
range.

Impact (free rescore): both papers `convention_mismatch → compared` at
~1.6–1.9 dex. This nudges micro/macro **up** slightly (full346 micro
0.247→0.254, macro 0.437→0.463) because it stops hiding two real mediocre
extractions — an **integrity correction**, not a regression: the benchmark now
counts extractions it was previously suppressing. `ct-accuracy` unchanged
(already correct).
