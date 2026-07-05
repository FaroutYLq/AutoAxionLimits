# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **NEVER push directly to master.** Always create a branch and open a PR. Wait for the user to review and merge. No exceptions.
- **NEVER merge a PR produced by the daily/weekly/backfill pipeline on your own** (the new-limit / science PRs, `[NEEDS REVIEW]`, `[LOW CONFIDENCE]`, `[BACKFILL]`, etc.). These contain scientific content that only the user may accept. If the user explicitly asks you to merge one, double-check with them before doing so. (Merging your *own* infrastructure/fix PRs when the user asks is fine.)
- **NEVER close or delete branches for PRs created by the daily/weekly pipelines** (#17, #19, #80, and similar). Only clean up PRs clearly from backfill cascades (`[BACKFILL]` prefix or `chore: update backfill state`).

## Project Overview

This is **AutoAxionLimits** — a fork of `cajohare/AxionLimits` with three automated pipelines built on top:

1. **Daily arXiv digest** (`pipeline/orchestrator.py`): Monitors arXiv for new dark matter limit papers, extracts data via Claude agents, and opens a GitHub PR per new limit.
2. **Weekly preprint checker** (`pipeline/preprint_checker.py`): Scans existing data files for arXiv IDs, detects updated preprint versions with changed results, and opens PRs. Also flags published papers whose published version yields no extractable data (`[NEEDS REVIEW]` PRs).
3. **Historical backfill** (`pipeline/backfill.py`): Manual-only workflow that searches Semantic Scholar for older papers in a date range, filters by citation count, and processes them through the extraction pipeline. Automatically splits large jobs across multiple daily runs.

All updates go through PRs — nothing merges to master automatically.

The upstream repository `cajohare/AxionLimits` is a scientific visualization repository that compiles experimental and theoretical limits on axions, dark photons, and other ultralight boson searches. It produces publication-quality constraint plots used by the particle physics community.

## Pipeline

### Running the pipelines

```bash
# Daily digest (dry run to test extraction without writing files)
python -m pipeline.orchestrator --dry-run

# Force-process a specific arXiv paper
python -m pipeline.orchestrator --arxiv-id 2412.12345

# Weekly preprint checker: initialize state (no PRs, just baseline versions)
python -m pipeline.preprint_checker --init-only

# Weekly preprint checker: dry run
python -m pipeline.preprint_checker --dry-run

# Historical backfill: discover candidates only (review before extracting)
python -m pipeline.backfill --date-from 2020-01-01 --date-to 2024-12-31 --min-citations 10 --discover-only

# Historical backfill: process queued candidates
python -m pipeline.backfill --resume --max-papers 10

# Historical backfill: dry run on specific coupling types
python -m pipeline.backfill --date-from 2023-01-01 --date-to 2023-12-31 --coupling-types AxionPhoton,DarkPhoton --dry-run
```

### Pipeline dependencies

```bash
pip install -r requirements_pipeline.txt
```

Required env vars:
- `ANTHROPIC_API_KEY` — Claude API key
- `GH_TOKEN` — GitHub token (provided automatically by Actions)

### Pipeline directory structure

```
pipeline/
  config.py           # Coupling registry, keywords, correction factors
  monitor.py          # arXiv querying + state (daily digest)
  extractor.py        # Claude extraction agent: PDF → ExtractionResult
  reviewer.py         # Claude reviewer agent: ExtractionResult → repo artifacts
  preprint_checker.py # Weekly: scan existing files, detect updated preprints
  backfill.py         # Historical backfill via Semantic Scholar search
  plot_regen.py       # Headless notebook execution + highlighted plot generation
  pr_creator.py       # git branch + commit + gh pr create
  orchestrator.py     # Daily digest entrypoint
  state/
    processed.json           # Git-tracked: processed arXiv IDs
    preprint_versions.json   # Git-tracked: known arXiv versions per data file
    backfill_state.json      # Git-tracked: backfill queue and progress
  logs/               # .gitignore'd
```

### GitHub Actions

- `.github/workflows/arxiv_daily.yml` — runs daily at 9 AM UTC
- `.github/workflows/preprint_weekly.yml` — runs every Monday at 10 AM UTC
- `.github/workflows/backfill.yml` — manual-only (`workflow_dispatch`); auto re-triggers itself daily until the queue is empty
- Daily and weekly workflows also support `workflow_dispatch` for manual triggering

### Key design decisions

- **Human-in-the-loop**: Every update becomes a PR; nothing merges automatically.
- **Low confidence PRs**: Created but titled `[LOW CONFIDENCE]`.
- **State persistence**: `processed.json` and `preprint_versions.json` are git-tracked and committed back to `master` by the Actions workflow after each run; no external storage needed.
- **Single reusable state branch**: State updates use a fixed branch per type (`chore/update-pipeline-state`, `chore/update-preprint-state`) that is force-updated each run, so only one state PR per type is ever open. State files are cumulative snapshots — only the latest is useful — so superseded snapshots are not worth keeping as separate PRs. This is an explicit exception to the "never close pipeline PRs" rule, which protects the *science/limit* PRs (and `[NEEDS REVIEW]` flags), not the `chore: update … state` chores. Force-pushing the fixed branch is the supersede mechanism; the old timestamped-PR scheme created one PR per run and never merged, piling them up.
- **State baseline carried forward without merging** (#547): each run does `actions/checkout` of `master`, so it would read the *frozen* baseline and never see the un-merged state PR's progress — re-extracting and re-proposing the same paper every day until it aged out of the arXiv window. Fix: before running, the daily/weekly workflows restore the cumulative state file (`processed.json` / `preprint_versions.json`) from the open state branch (`chore/update-pipeline-state` / `chore/update-preprint-state`) into the working tree, so `filter_new_papers()` already knows everything handled by previous runs. The restore is per-file (not the whole `pipeline/state/` dir) to avoid cross-contaminating other state types, and falls back to the `master` baseline if the branch (or the file in it) does not yet exist. No merge to `master` is required and human-in-the-loop is preserved; the commit step still force-pushes the freshest accumulated snapshot to the same branch. Merging the surviving state PR is now optional housekeeping, not a correctness requirement.
- **AST-based insertion**: New methods in `PlotFuncs.py` are inserted using the last `FunctionDef.end_lineno` inside the target class — never regex.
- **Text-first extraction**: Tables/text → vision fallback reduces API cost.
- **Prompt injection defence**: PDF text is sanitized (control chars stripped) and wrapped in `===PAPER_CONTENT===` delimiters before being sent to Claude.
- **API retry**: All Claude calls use exponential backoff on rate-limit and HTTP 529 errors.
- **DM density single-owner convention (2026-07-05)**: the stored data file is kept **paper-native**; the DM-density `sqrt(rho)` rescale is applied **once, by the plotting method** at plot time (matching the repo's own `PlotFuncs.py` convention), deterministically injected into the generated method by `reviewer.generate_plotfuncs_method` after the `loadtxt` line. The pipeline no longer bakes density into the file — doing that *and* letting the method rescale double-counted on re-extraction of an already-curated experiment. Pinned by `pipeline/tests/test_dm_density_correction.py`.
- **DM density correction scope & direction**: `sqrt(rho_DM)` rescaling is only applied to coupling types that have a `dm_density` entry in `PHYSICAL_CORRECTIONS` (i.e. haloscope/DM-search experiments), never to stellar, cosmological, or collider bounds. **Direction**: a DM-search limit scales as `g ∝ 1/sqrt(rho_DM)`, so `g_repo = g_paper * sqrt(rho_paper/rho_repo)` — a higher assumed density gives a *stronger* (lower) limit. This matches `PlotFuncs.py` DM-search methods (`*sqrt(0.3/0.45)`). Pinned by `pipeline/tests/test_dm_density_correction.py` (an earlier inverted `sqrt(rho_repo/rho_paper)` weakened every density-corrected limit).
- **Notebook selection**: `_select_notebook()` picks the target notebook by mass range — ultralight (< 1 μeV), collider (> 10 keV), or primary.
- **Out-of-range axis auto-extension**: when a new limit's mass range falls outside the target figure's conventional axis window, `insert_notebook_call()` widens the cell's `FigSetup(...)` call (`_extend_figsetup_range()`, using `_figsetup_defaults()` parsed from `PlotFuncs.py`) so the limit is plotted rather than silently dropped off the plot edge. It only ever *extends* (never shrinks), only the side the data exceeds, and rounds out to the enclosing decade — an out-of-range limit should always be visible even if the plot looks ugly. Because `FigSetup` builds the top frequency axis from `m_min`/`m_max`, the call args must be edited (not `set_xlim` after the fact).
- **Shell injection prevention**: `workflow_dispatch` inputs are passed to shell scripts via env vars, never interpolated directly into the command string.
- **`@staticmethod` guarantee**: A post-generation guard in `reviewer.py` prepends `@staticmethod` to any LLM-generated method that omits it.
- **Highlighted plots**: `execute_notebook_highlighted()` in `plot_regen.py` generates a greyed-out version of the constraint plot with only the new limit in red. Theoretical benchmarks (QCD axion band) are preserved in their original colours. The highlighted plot is shown prominently in the PR body.
- **Published paper handling**: When a tracked preprint transitions to published, the checker still runs extraction and comparison (no early short-circuit). If the published version yields no data, a `[NEEDS REVIEW]` flag PR is created instead of silently skipping. Papers are only marked `"published": true` in state after the transition is fully processed.
- **Projections out of scope (2026-07-04)**: the daily orchestrator skips papers the extractor flags `is_projection` (marks them `projection_out_of_scope`, opens no PR). Projected sensitivities carry detector-scenario-dependent assumptions (often several forecast curves per figure), are of less community interest than measured bounds, and are hard to record as a single ground-truth curve. The benchmark applies the same rule via `AAL_EXCLUDE_PROJECTIONS` (measured-limits-only scope, PR #698). The `[PROJECTION]` PR-title prefix in `pr_creator.py` remains for the weekly/backfill paths, which still surface projections; only the daily digest suppresses them.
- **PR separation**: Pipeline/infrastructure changes and new limit proposals must always be in separate PRs. Never mix technical updates with science content.
- **Backfill search strategy**: Uses INSPIRE-HEP API (not arXiv API or Semantic Scholar) for historical searches because INSPIRE is purpose-built for HEP literature, provides citation counts (`topcite`), date filtering, and has no aggressive rate limiting. Papers are pre-filtered through stages (citation threshold → duplicate detection → keyword classification → LLM batch relevance check) before expensive Claude extraction.
- **API-availability fail-fast (#648)**: billing 400 / 401 / 403 raise `FatalAPIError` in `_call_with_retry`; stage handlers re-raise it (never fail closed to `is_new_limit=False`), the orchestrator runs a 1-token preflight ping and aborts (exit 2) WITHOUT marking the current paper processed/failed, backfill re-queues the candidate, the weekly checker aborts red. An availability error is a property of the run, not the paper.
- **Truthful-declaration contract (#594 follow-up)**: `coupling_convention` must describe the EMITTED values. On a vision win the stage-2a axis read-back (a measurement) overrides a canonical-claiming declaration; candidates whose declared convention failed convention review are demoted in the selector; "converted from …" declarations are never re-converted by the registry.
- **Plotted-values contract (#684)**: the vision channel never converts — it emits raw axis values and declares the plotted quantity; conversion happens exactly once, in the registry. Asking the model to convert invites double conversion (model + registry) or zero conversion; both produced ~8 dex errors on correct readings (1508.02463: 8.5 → 0.22 dex after the contract).
- **Text–vision corroboration gate (#683)**: a vision trace deviating >2 dex from an in-range text anchor over shared mass support is rejected (`[TEXT-VISION DISAGREEMENT]`). Like all guards it is a pure function over the model's own declared outputs — no extra API calls, fully unit-testable. 19 firings in the definitive 346-paper run.
- **Foreign-quantity convention screen (#683)**: declarations naming physics outside a coupling's vetted vocabulary fail CLOSED — runtime flags `[CONVENTION REVIEW]` for a human; the eval side excludes the paper as a convention gap. Matching is against the vetted vocabulary, never by substring heuristics on free-text declarations (those silently suppress review flags).
- **N=1 production default (#685/#686)**: consensus voting is retired — a transport-matched probe measured N=3 − N=1 = +0.003 dex on Opus (voting stabilises the traced curve; the variance lives in channel routing). Production extraction is single-read.
- **Opus-production / Haiku-eval model split (#686)**: daily/weekly/backfill extraction runs Opus (at 1–3 papers/day the ~5× price delta is cents while the catastrophic-tail difference is 3×: 44 vs 19 on the definitive benchmark); benchmarks/evals default to Haiku per the standing cost rule. The extraction driver asserts the resolved model at startup — never trust env alone (the silent-model confound incident).
- **Benchmark GT exclusions (post-full346)**: `papers.json` entries can carry `excluded`/`exclusion_reason`/`exclusion_evidence` — used ONLY when the GT itself can grade no extraction (prediction-band files, published-only/private data), never because the extractor currently fails a paper. Documented in `evaluation/ground_truth/EXCLUSIONS.md`, listed in every report, reversible. GT data files are per-entry and ingested with header-declared unit conversions (λ[m]→eV etc.).
- **Convention-escalation design (#636)**: `pipeline/DESIGN_convention_escalation.md` — unknown conventions get queued (`convention_queue.json`, planned) and drained locally by a GPD skill once per convention token; GPD never runs inside a production extraction.
- **Backfill auto-scheduling**: When the candidate queue is larger than one run can handle, the workflow automatically re-triggers itself to process the next batch. The `concurrency` setting prevents overlap.

## Running the Code

There are no build steps or tests. Plots are generated by running Jupyter notebooks:

```bash
jupyter notebook
```

Each notebook corresponds to a coupling type (e.g., `DarkPhoton.ipynb`, `AxionPhoton.ipynb`) and produces PDF/PNG output in `plots/` and `plots/plots_png/`.

To regenerate all plots, run all cells in the relevant notebook. The final cell calls `MySaveFig(fig, 'PlotName')` which saves to `plots/`.

## Dependencies

Plotting dependencies (no requirements file — inferred from imports):
- `numpy`, `scipy`, `matplotlib` (with `patheffects`)

Pipeline dependencies (see `requirements_pipeline.txt`):
- `anthropic`, `arxiv`, `pymupdf`, `httpx`, `nbformat`, `nbconvert`, `numpy`

Install pipeline deps with `pip install -r requirements_pipeline.txt`. Python 3.11+ required.

## Architecture

### Core Library: `PlotFuncs.py`

The single large file (~4600 lines) contains all plotting logic. It is organized as:
- **Top-level helpers**: `FigSetup()`, `MySaveFig()`, `load_data()`, etc.
- **Particle classes**: Each coupling type has its own class (e.g., `AxionPhoton`, `DarkPhoton`, `AxionElectron`). Each class contains static methods, one per experimental constraint (e.g., `DarkPhoton.ADMX()`, `AxionPhoton.StellarBounds()`).

`PlotFuncs_ScalarVector.py` follows the same pattern for scalar and vector couplings.

### Notebooks

Each notebook imports from `PlotFuncs`, sets up a figure, calls experiment methods to draw filled exclusion regions, adds text annotations, and saves. The notebooks are the "main" files — all configuration of which experiments appear and plot aesthetics lives there.

### Data Files: `limit_data/`

Subdirectories correspond to coupling types (e.g., `limit_data/DarkPhoton/`, `limit_data/AxionPhoton/`). Each file is a two-column ASCII file: `mass [eV]`, `coupling strength`. Projections are in `limit_data/<Type>/Projections/`.

### Documentation: `docs/`

Markdown files (one per coupling, e.g., `docs/dp.md`, `docs/ap.md`) document all data sources with references and describe each experimental bound.

## Adding a New Experimental Limit

1. Add a data file to `limit_data/<CouplingType>/ExperimentName.txt`
2. Add a static method to the relevant class in `PlotFuncs.py`:
   ```python
   @staticmethod
   def ExperimentName(ax, col='color', text_on=True, lw=1.5, zorder=1.0):
       dat = loadtxt('limit_data/CouplingType/ExperimentName.txt')
       ax.fill_between(dat[:,0], dat[:,1], y2=1e99, ...)
   ```
   The `@staticmethod` decorator is required — methods without it will not be callable from notebook code.
3. Call the method in the relevant notebook
4. Add documentation in `docs/<type>.md`
