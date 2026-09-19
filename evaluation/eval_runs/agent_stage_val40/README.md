# Agent extraction stage: production validation (40 papers, 2026-09-18)

**Question.** Is the production agent stage (`AAL_EXTRACTOR=agent`, `pipeline/agent_extractor.py`,
task card = AxionLimitBench `docs/TASK.md` verbatim + production additions) the benchmarked system?

**Method.** 40 papers drawn at random (seed 20260918) from the 291 scorable AxionLimitBench papers
(`ids.json`), extracted through the full production path (`evaluation/benchmark/extract_driver.py`
-> `run_extraction_agent` -> agent session -> `finalize_extraction`), Fable 5 on the Claude Code
subscription, 3 workers, budget 5 USD/paper. Snapshots scored with the frozen AxionLimitBench scorer
(`python -m scorer.score`) and compared paper by paper with the benchmark's `agent_claude_code/fable5`
run (same card core, same model). Success = median residual within 10% (0.041 dex).

**Result.** Production reproduces the benchmark within run-to-run scatter: 23/40 vs 24/40 hits
(paired: 22 both, 2 benchmark-only, 1 production-only), 0 catastrophic on both sides, coupling type
100%, cond. median 0.029 dex. The two benchmark-only papers are raster-figure reads whose residual moved
0.04 -> 0.15 (1808.02340, spikes traced to line edges) and 0.041 -> 0.083 (2503.11753); no systematic
cause. Every session wrote `alternatives` (40/40), `overlay.png` (40/40), `headline_check` (26/40);
mean cost 2.04 USD (max 4.66), median 16 turns.

**One defect found and fixed in the shared guard tail.** 2105.13963 (AxionMass, GW170817): the agent's
correct 1/f_a = 6.4e-17 GeV^-1 was decade-snapped x1e12 by `VALID_RANGES["AxionMass"]` (coupling floor
1e-12, sized for the repo's normalised f_a plane), giving 12.0 dex. Floor widened to 1e-20 (f_a up to
~M_Pl); rerun 0.004 dex. Pinned by `test_inverse_fa_read_is_not_decade_snapped`.

Full snapshots, transcripts and overlays: `~/.aal_bench/agent_stage_val40/` (local, not committed).

```
sample n=40: benchmark fable5 hits 24/40; production agent stage hits 23/40
paired: both 22, benchmark-only 2, production-only 1
catastrophic (>1 dex): benchmark 0, production 0

arxiv               bench     prod  status_prod        src_prod        ct_prod/ct_bench
1503.06886          0.049    0.051  compared           figure_vector   ScalarPhoton/ScalarPhoton
1506.08082          0.010    0.007  compared           figure_vector   AxionPhoton/AxionPhoton
1604.08514          0.065    0.063  compared           figure_vector   ScalarPhoton/ScalarPhoton
1610.02580          0.052    0.052  compared           figure_vision   AxionPhoton/AxionPhoton
1808.02340          0.040    0.148  compared           figure_vision   AxionElectron/AxionElectron  <-- lost
1902.04246          0.010    0.010  compared           figure_vector   AxionElectron/AxionElectron
1903.05101          0.030    0.029  compared           figure_vector   DarkPhoton/DarkPhoton
1906.08814          0.000    0.000  compared           figure_vision   DarkPhoton/DarkPhoton
1907.05475          0.095    0.096  compared           figure_vector   AxionPhoton/AxionPhoton
2003.03348          0.003    0.005  compared           figure_vision   AxionPhoton/AxionPhoton
2003.13698          0.000    0.002  compared           ancillary_file  DarkPhoton/DarkPhoton
2004.02733          0.039    0.039  compared           figure_vector   AxionNeutron/AxionNeutron
2008.03305          0.003    0.017  compared           figure_vector   AxionPhoton/AxionPhoton
2008.10141          0.018    0.018  compared           figure_vision   AxionPhoton/AxionPhoton
2010.08107          0.065    0.065  compared           figure_vector   ScalarPhoton/ScalarPhoton
2102.00379          0.045    0.050  compared           figure_vector   AxionPhoton/AxionPhoton
2102.08764          0.000    0.000  compared           text            AxionElectron/AxionElectron
2105.13963          0.004    0.004  compared           figure_vector   AxionMass/AxionMass
2109.08822          0.302    0.277  compared           figure_vision   VectorBL/VectorBL
2111.06883              -        -  no_comparable_gt   figure_vector   ScalarNucleon/ScalarNucleon
2111.08025          0.010    0.007  compared           figure_vector   AxionPhoton/AxionPhoton
2202.08274          0.046    0.184  compared           figure_vision   AxionPhoton/AxionPhoton
2203.04319          0.004    0.004  compared           figure_vector   AxionPhoton/AxionPhoton
2205.01079          0.012    0.013  compared           figure_vector   AxionPhoton/AxionPhoton
2207.11968          0.010    0.007  compared           figure_vision   AxionElectron/AxionElectron
2209.09917          0.034    0.002  compared           figure_vector   AxionPhoton/AxionPhoton
2211.03414          0.006    0.003  compared           figure_vector   AxionPhoton/AxionPhoton
2301.03622          0.243    0.236  compared           figure_vector   DarkPhoton/DarkPhoton
2301.08736          0.904    0.897  compared           figure_vision   VectorBL/VectorBL
2302.10206          0.822    0.542  compared           figure_vision   AxionPhoton/AxionPhoton
2303.07370          0.046    0.046  compared           figure_vector   AxionPhoton/AxionPhoton
2308.06339          0.017    0.015  compared           figure_vector   AxionPhoton/AxionPhoton
2310.06017          0.010    0.026  compared           figure_vector   VectorBL/VectorBL
2407.18586          0.012    0.007  compared           figure_vision   AxionPhoton/AxionPhoton
2408.02368          0.020    0.010  compared           ancillary_file  DarkPhoton/DarkPhoton
2409.01805          0.129    0.129  compared           figure_vector   AxionPhoton/AxionPhoton
2409.08998          0.044    0.039  compared           figure_vector   AxionPhoton/AxionPhoton  <-- gained
2503.11753          0.041    0.083  compared           figure_vector   AxionPhoton/AxionPhoton  <-- lost
2503.14582          0.365    0.363  compared           figure_vector   AxionPhoton/AxionPhoton
2504.07559          0.023    0.021  compared           figure_vector   AxionPhoton/AxionPhoton
```
