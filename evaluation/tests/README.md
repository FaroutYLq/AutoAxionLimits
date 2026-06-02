# Evaluation metric tests

Unit tests for `evaluation/metrics.py` (and the summary helper in
`evaluation/evaluate.py`). They run on synthetic curves with known answers and
make **no** Anthropic API calls.

## Run

```bash
pytest evaluation/tests/
# or
python -m pytest evaluation/tests/test_metrics.py -v
```

## What is covered

- **Forward interpolation residual** (`compute_interpolation_metrics`):
  identical curves → residual 0 / 100% coverage; a vertical scale by factor `k`
  → median residual `log10(k)`; disjoint mass ranges → infinite residual / 0%
  coverage.
- **Boundary-closure sentinel filtering** (`_filter_boundary`,
  `_COUPLING_CEILINGS`): the exact ceilings the PR #535/#536 bugs touched —
  `AxionMass`/`DarkPhoton`/`VectorBL`/`AxionCPV`/`MonopoleDipole` = `1.0`,
  `Scalar*` = `1e19`, default = `1e-2`. Includes **regression tests** that fail
  on the old buggy ceilings (`AxionMass=1e6` kept the `1.0` sentinels;
  `Scalar=1e0` discarded the whole curve), plus an end-to-end check that the
  retained sentinel blows up the residual.
- **Single-mass GT routing**: `_usable_gt_stats` reports `< 2` distinct masses,
  which is the guard `evaluate.py` uses to route to `gt_point_reference`.
- **#541 metrics**: reverse pass (identical → 0; over-claim lowers reverse
  coverage), area-between-curves (identical → 0, known vertical shift → known
  area), and mass-range Jaccard (identical → 1, disjoint → 0, partial → 0.25).
- **#542 calibration**: the accuracy threshold constant is the noise floor
  (`NOISE_FLOOR_RESIDUAL_DEX == 0.32`); a confident-but-wrong set yields a
  positive overconfidence gap; the continuous `P(residual < tau)` fields.
- **Summary builder**: `build_metrics_summary` shape + the overconfidence gap.

These mirror (and fold in) the `python -m evaluation.metrics` synthetic
self-check.
