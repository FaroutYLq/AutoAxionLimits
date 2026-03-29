# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **AutoAxionLimits** — a fork of `cajohare/AxionLimits` with two automated pipelines built on top:

1. **Daily arXiv digest** (`pipeline/orchestrator.py`): Monitors arXiv for new dark matter limit papers, extracts data via Claude agents, and opens a GitHub PR per new limit.
2. **Weekly preprint checker** (`pipeline/preprint_checker.py`): Scans existing data files for arXiv IDs, detects updated preprint versions with changed results, and opens PRs. Also flags published papers whose published version yields no extractable data (`[NEEDS REVIEW]` PRs).

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
  plot_regen.py       # Headless notebook execution + highlighted plot generation
  pr_creator.py       # git branch + commit + gh pr create
  orchestrator.py     # Daily digest entrypoint
  state/
    processed.json           # Git-tracked: processed arXiv IDs
    preprint_versions.json   # Git-tracked: known arXiv versions per data file
  logs/               # .gitignore'd
```

### GitHub Actions

- `.github/workflows/arxiv_daily.yml` — runs daily at 9 AM UTC
- `.github/workflows/preprint_weekly.yml` — runs every Monday at 10 AM UTC
- Both support `workflow_dispatch` for manual triggering

### Key design decisions

- **Human-in-the-loop**: Every update becomes a PR; nothing merges automatically.
- **Low confidence PRs**: Created but titled `[LOW CONFIDENCE]`.
- **State persistence**: `processed.json` and `preprint_versions.json` are git-tracked and committed back to `master` by the Actions workflow after each run; no external storage needed.
- **AST-based insertion**: New methods in `PlotFuncs.py` are inserted using the last `FunctionDef.end_lineno` inside the target class — never regex.
- **Text-first extraction**: Tables/text → vision fallback reduces API cost.
- **Prompt injection defence**: PDF text is sanitized (control chars stripped) and wrapped in `===PAPER_CONTENT===` delimiters before being sent to Claude.
- **API retry**: All Claude calls use exponential backoff on rate-limit and HTTP 529 errors.
- **DM density correction scope**: `sqrt(rho_DM)` rescaling is only applied to coupling types that have a `dm_density` entry in `PHYSICAL_CORRECTIONS` (i.e. haloscope/DM-search experiments), never to stellar, cosmological, or collider bounds.
- **Notebook selection**: `_select_notebook()` picks the target notebook by mass range — ultralight (< 1 μeV), collider (> 10 keV), or primary.
- **Shell injection prevention**: `workflow_dispatch` inputs are passed to shell scripts via env vars, never interpolated directly into the command string.
- **`@staticmethod` guarantee**: A post-generation guard in `reviewer.py` prepends `@staticmethod` to any LLM-generated method that omits it.
- **Highlighted plots**: `execute_notebook_highlighted()` in `plot_regen.py` generates a greyed-out version of the constraint plot with only the new limit in red. Theoretical benchmarks (QCD axion band) are preserved in their original colours. The highlighted plot is shown prominently in the PR body.
- **Published paper handling**: When a tracked preprint transitions to published, the checker still runs extraction and comparison (no early short-circuit). If the published version yields no data, a `[NEEDS REVIEW]` flag PR is created instead of silently skipping. Papers are only marked `"published": true` in state after the transition is fully processed.
- **PR separation**: Pipeline/infrastructure changes and new limit proposals must always be in separate PRs. Never mix technical updates with science content.

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
