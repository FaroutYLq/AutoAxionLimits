# Gold set — limits digitized DIRECTLY from source papers

This directory holds a small, hand-curated **gold set**: experimental limit
curves digitized **directly from the source papers** (their published numeric
tables/text, or their figures), *independently* of cajohare's already-processed
repo curves.

## Why this exists (issue #537)

The main evaluation scores extractions against `evaluation/ground_truth/papers.json`,
whose ground truth is **cajohare's repo curve** — itself digitized, rescaled, and
convention-normalized from the same source papers. So a *perfect* extraction
still inherits the upstream digitization/convention gap, and the headline
residual cannot separate **"the extractor is wrong"** from **"the extractor and
cajohare made different choices about the same `f(x)`."**

The gold set is a reference that does **not** come from the repo, so we can:

1. **Quantify the upstream gap** — `gold` vs `repo-GT` residual (the part of the
   headline residual that is *not* extraction error).
2. **Isolate extraction error** — `extraction` vs `gold` residual, scored
   separately from `extraction` vs `repo-GT`.

## Independence tiers (kept distinguishable in the manifest)

| `digitized_by` | `source_kind`     | independence                                   |
|----------------|-------------------|------------------------------------------------|
| `gold_table`   | `table` / `text`  | **independent** — transcribed published numbers |
| `gold_vision`  | `figure`          | **semi-independent** — vision reads a figure     |

`gold_table` curves are the hard bound (truly repo- and vision-independent).
`gold_vision` curves are still vision-based, so the floor they give is a *softer*
bound — that is why the diff reports the two tiers separately.

### `gold_table` tier expansion (#537 follow-up, #542)

The original #537 build had only ~1 *usable* same-convention `gold_table`↔repo
pair, far too thin to pin the truly-independent digitization floor that #542
needs for its confidence-accuracy threshold. The tier was grown to **17
`gold_table` entries → 10 usable same-convention pairs** spanning AxionPhoton,
AxionElectron, AxionProton, AxionNeutron, DarkPhoton, ScalarPhoton and
ScalarNucleon. Every added paper publishes its limit as **numeric values in the
PDF text** (a stellar/cosmological *flat* bound stated over an explicit mass
range counts — transcribe both range endpoints). Candidates that turned out to
be **figure-only** (no transcribable numbers) were dropped rather than
vision-digitized, to keep the tier truly independent (e.g. `1804.10777` TEXONO).

Flat-bound curves are the same physical line regardless of which range endpoints
each side transcribes, so `gold_diff._median_residual` measures the gold↔repo gap
on whichever interpolation direction has shared mass support (forward, else the
swapped reverse). The result is direction-symmetric over the overlap.

The digitizer (`evaluation/gold_build.py`) uses the **strongest** available model
(`claude-opus-4-5-20251101`) with a prompt **distinct** from the production
extractor (`pipeline/extractor.py`, which runs Haiku). It does **not** reuse
`run_extraction_agent`.

## Files

- `gold.json` — the manifest (schema below). One entry per gold curve.
- `data/<arxiv_id>.txt` — two-column `mass_eV  coupling` points for that curve
  (comment header lines start with `#`).
- `README.md` — this file.

## Manifest schema (`gold.json`)

```jsonc
{
  "schema_version": 1,
  "digitize_model": "claude-opus-4-5-20251101",
  "gold_curves": [
    {
      "arxiv_id": "0802.2350",
      "entry_key": "0802.2350",            // manifest key + data-file stem;
                                           // defaults to arxiv_id, set explicitly
                                           // when one paper yields >1 gold curve
                                           // (e.g. 2111.09892_proton / _neutron)
      "paper_title": "...",
      "coupling_type": "ScalarNucleon",     // pipeline.config coupling key
      "coupling_convention": "coupling",    // inferred from the curve's range
      "coupling_units": "...",              // human-readable y-axis units
      "source_kind": "table",               // table | text | figure
      "digitized_by": "gold_table",         // gold_table | gold_vision
      "independence": "independent (published numbers)",
      "provenance": "which figure/panel/table/line, any in-paper rescaling",
      "reference_repo_file": "limit_data/.../X.txt",  // repo curve to diff against
      "gold_data_file": "0802.2350.txt",    // file in data/
      "num_points": 6,
      "status": "digitized",                // digitized | cached | no_points | failed
      "digitize_model": "claude-opus-4-5-20251101",
      "digitizer_confidence": 0.9,          // model self-report (figure/table)
      "digitizer_published_convention": "alpha vs lambda",
      "digitizer_notes": "unit conversions, which table/section, caveats",
      "digitizer_x_axis": "...",            // (figure mode) axis as read
      "digitizer_y_axis": "..."             // (figure mode) axis as read
    }
  ]
}
```

The `coupling_convention` / `coupling_units` fields reuse the logic added in
#538 (`evaluation/conventions.py`): the convention is **inferred from the
digitized curve's own value range** (e.g. `d_e` vs `d_e_large`, `f_a_GeV` vs
`f_a_norm`), so a gold curve and a repo curve in different conventions are not
silently compared.

## Building / refreshing

```bash
export ANTHROPIC_API_KEY=...        # required (Opus digitization)

python -m evaluation.gold_build --mode table     # table/text papers (independent)
python -m evaluation.gold_build --mode figure    # figure-only papers (vision)
python -m evaluation.gold_build --mode all       # everything (idempotent)
python -m evaluation.gold_build --arxiv-id 0802.2350 --force   # re-digitize one
```

Runs are idempotent: a curve whose `data/<id>.txt` already exists is kept
(convention/units are re-derived from the points) unless `--force` is given.
The human-curated **selection** (which papers, coupling, figure/table, expected
convention, page hints) lives in `GOLD_SELECTION` in `gold_build.py` — that is
the actual ground truth; the model only fills in the numeric points.

## Scoring against the gold set

```bash
python -m evaluation.gold_diff                          # print summary
python -m evaluation.gold_diff --report evaluation/gold_report.md --json /tmp/gold_diff.json
```

`gold_diff.py` reuses `compute_interpolation_metrics` / `_filter_boundary` from
`evaluation/metrics.py`, so gold / repo / extraction all see the same boundary
filtering and log-log interpolation. It reports, per curve and as medians:

- **gold ↔ repo-GT** — the upstream digitization gap (the key science number),
- **extraction ↔ gold** — extraction error against the paper itself,
- **extraction ↔ repo-GT** — the headline residual, on the SAME papers.

Pairs whose residual exceeds the plausibility cutoff (3 dex) are flagged as
likely units/convention gaps the coarse convention inference could not resolve
(e.g. `MonopoleDipole` `g_s*g_p` vs `g_p`) and excluded from the headline; the
raw (unfiltered) median is reported alongside for transparency.
```
