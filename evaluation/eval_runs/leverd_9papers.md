# Lever-D re-validation of the 9 wrong-curve papers (Haiku, single sample)

Plan-mandated check (post-full346 §Lever 5 / WS3 plan): re-run the 9
`wrong_curve_vision` papers against `fix/613-leverd-curve-selection` @ f931446e
(the committed lever-D draft the full346 run partially exercised) BEFORE any
further prompt work. Run 2026-07-02: `EXTRACTOR_MODEL=claude-haiku-4-5-20251001`,
`AAL_READ_SAMPLES=3`, isolated outdir `evaluation/eval_runs/leverd_9papers/`
(snapshots not committed; this report is the artifact). Gate columns show what
`pipeline/vision_gates.py` (#663) fires on the NEW winning sample.

| paper | old (full346) | lever-D result | verdict | gates on new sample |
|---|---|---|---|---|
| 1508.01798 | 3.74 dex (traced Eot-Wash) | **0.95 dex**, vision, cov 1.0 | **fixed** by lever-D prompt | — |
| 1207.3275 | 3.85 dex (envelope) | 1.78 dex, vision, cov 1.0 | improved, still >1 dex | — |
| 1808.02340 | inf (nominal-mass text pt) | 2.50 dex, text 9 pts, cov 1.0 | improved from unscored; masses now in-window (gate C correctly silent) | — |
| 1512.06165 | 7.53 dex (traced static-EP) | 5.10 dex, ct drifted to ScalarBaryon | **still wrong** | **A:reject + D:demote** |
| 2309.07995 | 4.46 dex (traced Eot-Wash/LIGO) | 7.20 dex | **still wrong** | **A:reject** |
| 1903.12190 | inf (millicharge panel) | 5.27 dex, now a TEXT read with in-window masses | still wrong — failure MOVED to text, couplings ~5 dex off | — (C correctly silent: masses now overlap the abstract window) |
| 1008.3536 | 4.74 dex (compilation envelope) | 5.00 dex, now a 5-pt text read | still wrong — failure moved to text | — |
| 1708.02111 | 1.62 dex (g_aγ panel as AxionElectron) | unscored — now DECLARES AxionPhoton (matches the axis), GT expects AxionElectron | morphed into a coupling-type mismatch | — (B correctly silent: declaration now matches axis) |
| 1912.07751 | inf (dual-axis) | unscored | still failing | — |

## Conclusions

1. **Lever-D alone fixes 1/9 outright (1508.01798) and materially improves 2
   more** (1207.3275, 1808.02340). It does NOT resolve the family.
2. **The #663/#664 gates remain necessary and correctly targeted**: on the new
   winning samples they still catch the two clearest surviving mistraces
   (1512.06165, 2309.07995). At runtime (#664) they see every candidate's own
   notes — strictly more than this winner-only check.
3. **Two failures moved from vision to text winners** (1008.3536, 1903.12190):
   the lever-D vision prompt changed candidate quality enough to flip
   selection, and the surviving error is now wrong text values in the right
   mass window — outside the wrong-curve gates' remit by design (gate C is
   correctly silent when masses overlap the abstract window). These two need a
   different lever (text-value verification), not more vision prompt work.
4. Caveat: single Haiku extraction per paper; these hard papers have real
   run-to-run variance. Treat per-paper residuals as indicative, the
   family-level pattern as the finding.
