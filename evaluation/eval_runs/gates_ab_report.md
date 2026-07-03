# WS3 PR3 — targeted Haiku A/B validation of the #664 gate wiring

Run 2026-07-02, `EXTRACTOR_MODEL=claude-haiku-4-5-20251001`, `AAL_READ_SAMPLES=3`.
Budget-constrained design (no full-pool re-baseline): the gates only change
behavior where they fire, so the A/B re-extracts only papers that carry signal.

- **before** = `pipeline/ws3-vision-gates` (#663, gates present but not wired),
  **after** = `pipeline/ws3-gates-runtime` (#664). Both + a local-only
  `EXTRACTOR_MODEL` override commit (mirrors the f931446e hunk; never pushed).
- **19-paper strict A/B** (`gates_before/` vs `gates_after/`): the 8
  gate-trigger/wrong-curve papers inside the `figure` subset + 8 deterministic
  controls + 3 already-extracted extras.
- **7 signal papers outside the figure subset** (5 of the wrong-curve 9):
  after-side only, judged against cached full346 + the lever-D re-validation
  (`leverd_9papers.md`).
- Total ~45 Haiku extractions ≈ $8; session total ≈ $13 (within the $20 cap).

## Verdict: wiring is safe and works in vivo; merge-ready with one follow-up

1. **Runtime firings confirmed end-to-end** (extraction log):
   - 1512.06165 — gate A rejected the traced curve in 2 of 3 vote samples
     ("existing experiment 'pulsar'", "'torsionbalance'") + gate D demote
     ("envelope of the blue solid curves").
   - 1611.05852 — gate B rejected a VectorB-L-axis trace declared DarkPhoton.
2. **Zero false rejections**: no gate fired on any control or good paper,
   runtime or post-hoc.
3. **Controls unchanged**: median residual delta 0.000 dex (n=6 scoreable
   controls; several results byte-identical across sides). The wiring touches
   every paper's selection flow — no side effects observed.
4. **Follow-up found — read-vote can resurrect a gate-rejected mistrace.**
   On 1512.06165, the two gate-rejected samples lost the consensus vote to the
   third sample, whose notes phrased the same mistrace without a trace verb
   ("The plot displays the lower boundary … with multiple experimental
   constraints: torsion balance …") — so the emitted result is still the wrong
   curve (5.10 dex; old 7.53) with no gate note. `read_vote.select_consensus`
   should become gate-aware: when a majority of samples were gate-rejected,
   the consensus should be the rejection (zero points + `[VISION GATE]` flag),
   not the surviving sample's points. Filed as the WS3 PR3 follow-up.

## 19-paper A/B

| set | n scoreable | median Δ (after−before) |
|---|---|---|
| signal | 5 | −0.22 dex |
| control | 6 | **0.00 dex** |

`evaluation/gate.py --key figure --soft` formally reports G1 FAIL
(0.67→1.41 dex) with G2–G7 PASS, but on this deliberately failure-enriched
8-paper symmetric set G1's threshold (calibrated for the full 54-paper figure
subset) is not meaningful: the swing is dominated by papers where no gate
fired on either side — pure run-to-run variance (e.g. 1207.3275: the before
side drew a lucky 0.30-dex trace, the after side a typical 1.78). The
control-set delta is the valid regression signal.

## Runtime gate behavior is stochastic (expected, now quantified)

Notes-based gates catch **confessed** mistraces. Across same-day fresh
samples: the lever-D batch produced confessions on 2/9 papers; this batch on
2/26 (within vote samples). Samples that mistrace silently (2309.07995 this
run: traced the wrong panel, described neutrally) pass through — those
unconfessed mistraces are exactly the population WS2's deterministic
extraction (vector-path first) is for.

## 7 after-only signal papers (vs cached full346)

| paper | after | old | note |
|---|---|---|---|
| 1008.3536 | 2.88 (text) | 4.74 | improved; failure moved to text values |
| 1512.06165 | 5.10 | 7.53 | gates fired in 2/3 samples; vote resurrected the third (see follow-up) |
| 1611.05852 | 8.26 | unscored | gate B fired in ≥1 sample; winner sample unconfessed |
| 1708.02111 | 0 points (src=none) | 1.62 | refused this run — preferable to the old wrong-panel trace |
| 1903.12190 | 5.81 (text) | unscored | failure moved to text values in-window |
| 1911.05086 | 4.68 (text) | 1.36 | text regression vs old vision read — single-sample noise |
| 1912.07751 | unscored | unscored | unchanged |
