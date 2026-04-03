# AutoAxionLimits Pipeline

This document describes the three automated pipelines added to this fork of
[cajohare/AxionLimits](https://github.com/cajohare/AxionLimits).

---

## Overview

Three GitHub Actions workflows open pull requests when new or updated limits
are detected. **Nothing merges to master automatically** — every change
requires a human to review and merge the PR.

| Pipeline | Schedule | Entrypoint |
|----------|----------|------------|
| Daily arXiv digest | 9 AM UTC daily | `python -m pipeline.orchestrator` |
| Weekly preprint checker | Monday 10 AM UTC | `python -m pipeline.preprint_checker` |
| Historical backfill | Manual only | `python -m pipeline.backfill` |

The daily and weekly workflows also support `workflow_dispatch` for manual triggering.

---

## Pipeline 1: Daily arXiv Digest

Monitors arXiv for newly submitted papers that present experimental exclusion
limits on axions, dark photons, or other ultralight bosons. For each new limit
found, it extracts the data and opens a PR that adds the limit to the repo.

### What it does, step by step

1. **Fetch** — queries arXiv (`hep-ph`, `hep-ex`, `astro-ph.CO`, `astro-ph.HE`,
   `physics.ins-det`) for papers submitted in the last 3 days matching tracked keywords.
2. **Pre-filter** — cheap local keyword match against `ARXIV_KEYWORDS` in
   `pipeline/config.py` to skip obviously irrelevant papers before calling Claude.
3. **Extract** — two-stage Claude extraction:
   - *Stage 1 (text)*: sends sanitized PDF text to Claude; asks for coupling
     type, data points (mass [eV], coupling), DM density assumption, and a
     suggested experiment name.
   - *Stage 2 (vision)*: fallback when Stage 1 returns no data points, reports
     `is_new_limit=False`, or confidence < 0.4. Renders PDF pages to PNG and
     asks Claude to trace the exclusion boundary from the plot.
4. **Review** — applies deterministic physical corrections (see below), then
   asks Claude to generate a `PlotFuncs.py` static method following the exact
   style of existing methods.
5. **Write** — creates the data file, inserts the method into `PlotFuncs.py`
   via AST (never regex), adds a call to the appropriate notebook via
   `nbformat`, and appends a bullet to the relevant `docs/*.md` file.
6. **Regenerate** — executes the notebook headlessly with `nbconvert` to
   produce updated plot PDF/PNG files.
7. **PR** — creates a git branch, stages only the named changed files, and
   opens a PR titled:
   - `Add {Experiment} {CouplingType} limit (arXiv:{id})` — normal
   - `[LOW CONFIDENCE] Add ...` — extraction confidence < 60 %
   - `[PROJECTION] Add ...` — sensitivity projection rather than observed limit

### State file: `pipeline/state/processed.json`

Records every arXiv ID that has been processed (successfully or not). The
Actions workflow commits this file back to `master` after each run so the next
run does not re-process the same papers.

```json
{
  "schema_version": 1,
  "last_run": "2026-03-22T09:00:00Z",
  "processed_ids": ["2412.12345", "2501.99999"],
  "failed_ids": {"2412.22222": "timeout"}
}
```

---

## Pipeline 2: Weekly Preprint Checker

Scans every file in `limit_data/**/*.txt` for arXiv IDs embedded in header
comment lines, checks whether a newer version of each paper has been posted,
and opens a PR if the numerical data changed.

### What it does, step by step

1. **Scan** — reads the first 10 comment lines of each `.txt` data file and
   extracts arXiv IDs from URLs matching `arxiv.org/abs/{id}` or `arXiv:{id}`.
2. **Check version** — queries the arXiv API for the latest version number and
   whether the paper has a `journal_ref` (i.e. is no longer a preprint).
3. **Decision logic**:
   - *Published* (`journal_ref` present): mark as published, stop tracking.
   - *First time seen*: record current version as baseline — no PR created.
   - *Version unchanged*: update `last_checked` timestamp only.
   - *New version*: download the new PDF, run the extraction agent, compare
     data numerically (sorted by mass, relative tolerance 1 × 10⁻⁶).
4. **PR** — if data changed, writes the updated data file and opens a PR
   titled `Update {Experiment} {CouplingType}: arXiv:{id} v{old}→v{new}`,
   with a Claude-generated summary of what likely changed.

### State file: `pipeline/state/preprint_versions.json`

Records the known version and publication status of every tracked paper. The
Actions workflow commits this file back to `master` after each run.

```json
{
  "schema_version": 1,
  "last_checked": "2026-03-17T10:00:00Z",
  "files": {
    "limit_data/DarkPhoton/FUNK.txt": {
      "arxiv_id": "2003.13144",
      "known_version": 2,
      "last_checked": "2026-03-17T10:00:00Z",
      "published": false
    }
  }
}
```

**Initial population**: run `python -m pipeline.preprint_checker --init-only`
once to scan all existing data files and record their current arXiv versions as
the baseline. No PRs are created on the first scan.

---

## Pipeline 3: Historical Backfill

Searches [INSPIRE-HEP](https://inspirehep.net) for older papers in a
user-specified date range that are relevant to this repo but predate the daily
pipeline. Filters by citation count and passes candidates through the same
extraction pipeline used by the daily digest.

### Why INSPIRE-HEP?

- Purpose-built for HEP literature — much less noise than general academic APIs
- Provides citation counts via `topcite` operator
- Supports date filtering and sorting by citations
- Free, no authentication required, no aggressive rate limiting

### What it does, step by step

1. **Search** — queries INSPIRE-HEP with curated per-coupling-type keyword
   phrases (defined in `INSPIRE_SEARCH_QUERIES` in `pipeline/config.py`),
   filtering by date range and minimum citation count.
2. **Dedup** — skips papers already in `processed.json`, `backfill_state.json`,
   or embedded as arXiv IDs in existing `limit_data/**/*.txt` file headers.
3. **Keyword classify** — same local keyword matching as the daily pipeline
   (`ARXIV_KEYWORDS`); papers matching no coupling type are dropped.
4. **LLM batch filter** — sends batches of 20 title+abstracts to Claude Haiku
   asking which present NEW experimental exclusion limits. Filters out
   theory-only, review, and phenomenology papers. Cost: ~$0.001 per batch.
5. **Queue** — surviving candidates are sorted by citation count (highest first)
   and saved to `pipeline/state/backfill_state.json`.
6. **Process** — takes up to `--max-papers` (default 10) from the queue per run,
   runs the full extraction → review → PR pipeline (same as daily digest).
7. **Auto re-trigger** — if items remain in the queue after processing, the
   GitHub Actions workflow dispatches itself to continue the next day.

### CLI

```bash
# Discover candidates only (review queue before extracting)
python -m pipeline.backfill --date-from 2020-01-01 --date-to 2024-12-31 --min-citations 10 --discover-only

# Process queued candidates (max 10 per run)
python -m pipeline.backfill --resume --max-papers 10

# Dry run on specific coupling types
python -m pipeline.backfill --date-from 2023-01-01 --date-to 2023-12-31 \
  --coupling-types AxionPhoton,DarkPhoton --dry-run

# Full run (discover + process in one go)
python -m pipeline.backfill --date-from 2023-01-01 --date-to 2023-12-31 --min-citations 20
```

| Argument | Description |
|----------|-------------|
| `--date-from` / `--date-to` | Date range for INSPIRE search (required unless `--resume`) |
| `--min-citations` | Minimum citation count (default 10) |
| `--max-papers` | Max papers to process this run (default 10) |
| `--coupling-types` | Comma-separated subset of coupling types (default: all) |
| `--dry-run` | Log actions without writing files or creating PRs |
| `--discover-only` | Search + filter + save queue, don't extract |
| `--resume` | Skip discovery, process existing queue |

### State file: `pipeline/state/backfill_state.json`

Tracks the candidate queue, processed IDs, and run history. The queue persists
across runs, allowing large backfill jobs to be split across multiple days.

```json
{
  "schema_version": 1,
  "config": {"date_from": "2020-01-01", "date_to": "2024-12-31", "min_citations": 10},
  "queue": [{"arxiv_id": "2303.08666", "title": "...", "citations": 65, "coupling_guess": "DarkPhoton"}],
  "processed_ids": ["2205.67890"],
  "skipped_ids": {"2301.11111": "not_new_limit"},
  "runs": [{"timestamp": "2026-04-02T10:00:00Z", "processed": 10, "prs_created": 3, "remaining": 54}]
}
```

### Multi-day scheduling

The GitHub Actions workflow (`backfill.yml`) uses a `schedule-next` job that
checks the `queue_remaining` output. If the queue is non-empty and this is not a
dry-run or discover-only run, it dispatches itself with `--resume`. A
`concurrency` setting prevents overlapping runs.

### PR format

```
[BACKFILL] Add {ExperimentName} {CouplingType} limit (arXiv:{id})
[BACKFILL] [LOW CONFIDENCE] Add ...    ← extraction confidence < 60%
[BACKFILL] [PROJECTION] Add ...        ← sensitivity projection
```

Body includes citation count and a note that the paper was discovered via
historical backfill.

---

## Physical Corrections

Corrections are defined in `pipeline/config.py` under `PHYSICAL_CORRECTIONS`
and applied deterministically before any data is written.

### DM density rescaling

Haloscope and DM-absorption experiments quote limits that scale with the assumed
local dark matter density ρ_DM. This repo uses **ρ_repo = 0.45 GeV/cm³**. When
a paper assumes a different value ρ_paper, the coupling is rescaled:

```
coupling_corrected = coupling_paper × sqrt(ρ_repo / ρ_paper)
```

This is applied automatically only when:
- Claude reports a `dm_density_assumed` value for the paper, **and**
- the coupling type has a `dm_density` entry in `PHYSICAL_CORRECTIONS`
  (i.e. haloscope/DM-search types: DarkPhoton, AxionPhoton, AxionElectron,
  AxionNeutron, AxionProton).

Stellar, cosmological, and collider bounds are never rescaled.

### Polarization corrections

Some haloscopes assume a specific polarisation direction. When Claude detects a
polarisation assumption, it is flagged in the PR body for human review rather
than corrected automatically (the formula varies per experiment geometry).

---

## Configuration Reference (`pipeline/config.py`)

### `COUPLING_TYPES`

Maps each coupling type to its repo artifacts:

```python
"DarkPhoton": {
    "class_name": "DarkPhoton",          # PlotFuncs.py class name
    "plotfuncs_file": "PlotFuncs.py",    # or "PlotFuncs_ScalarVector.py"
    "data_dir": "limit_data/DarkPhoton",
    "notebooks": ["DarkPhoton.ipynb"],   # first entry = primary notebook
    "docs_file": "docs/dp.md",
}
```

Supported coupling types: DarkPhoton, AxionPhoton, AxionElectron, AxionNeutron,
AxionProton, AxionEDM, AxionCPV, AxionMass, MonopoleDipole, ScalarPhoton,
ScalarElectron, ScalarBaryon, ScalarNucleon, VectorBL.

### `ARXIV_KEYWORDS`

Per-coupling keyword lists used for cheap pre-filtering before calling Claude.
Edit these to tune sensitivity vs. noise.

### `PHYSICAL_CORRECTIONS`

Per-coupling correction metadata. Add a `dm_density` block here to enable
automatic density rescaling for a new coupling type.

### Top-level constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `LOW_CONFIDENCE_THRESHOLD` | `0.6` | Extractions below this confidence get a `[LOW CONFIDENCE]` PR title |
| `MAX_PAPERS_PER_RUN` | `5` | Maximum papers processed per daily digest run |
| `ARXIV_CATEGORIES` | see config | arXiv categories searched: `hep-ph`, `hep-ex`, `astro-ph.CO`, `astro-ph.HE`, `physics.ins-det` |
| `BACKFILL_MAX_PAPERS_PER_RUN` | `10` | Maximum papers processed per backfill run |
| `BACKFILL_DEFAULT_MIN_CITATIONS` | `10` | Default minimum citation count for backfill |
| `INSPIRE_SEARCH_QUERIES` | see config | Per-coupling-type keyword queries for INSPIRE-HEP search |

---

## Security Notes

- **Prompt injection**: PDF text is sanitized (control characters stripped) and
  enclosed in `===PAPER_CONTENT===` delimiters before being sent to Claude.
  The system prompt instructs Claude to treat content inside those markers as
  untrusted data only.
- **Shell injection**: `workflow_dispatch` inputs are passed to shell scripts
  via env vars (`$INPUT_ARXIV_ID`), never interpolated directly into commands.
- **Minimal git staging**: only explicitly named files are staged per commit —
  `git add -A` is never used.

---

## First-time Setup

Follow these steps once after forking the repository.

### 1. Clone and configure remotes

```bash
# Clone your fork
git clone https://github.com/<your-username>/AutoAxionLimits.git
cd AutoAxionLimits

# Point origin at your fork, upstream at the original repo
git remote rename origin upstream   # cajohare/AxionLimits → upstream
git remote add origin https://github.com/<your-username>/AutoAxionLimits.git
```

### 2. Install dependencies

Python 3.11 or later is required.

```bash
pip install -r requirements_pipeline.txt
```

Pipeline dependencies (`requirements_pipeline.txt`):

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API — extraction and reviewer agents |
| `arxiv` | arXiv search and paper metadata |
| `pymupdf` | PDF text and page-image extraction |
| `httpx` | PDF download |
| `nbformat` | Read/write Jupyter notebooks |
| `nbconvert` | Headless notebook execution |
| `numpy` | Data file loading and comparison in preprint checker |
| `scipy` | Required by `PlotFuncs.py` during headless notebook execution |
| `matplotlib` | Required by `PlotFuncs.py` during headless notebook execution |

All packages above must be in the same environment. `scipy` and `matplotlib`
are needed because the pipeline executes notebooks headlessly via `nbconvert`,
which imports `PlotFuncs.py`.

### 3. Configure git identity (local runs)

The orchestrator creates commits locally. Set your identity if not already configured:

```bash
git config user.email "you@example.com"
git config user.name "Your Name"
```

In GitHub Actions this is handled automatically by the workflow.

### 4. Authenticate the GitHub CLI

PR creation uses the `gh` CLI. Authenticate once:

```bash
gh auth login
```

Select **GitHub.com → HTTPS → Login with a web browser** (or paste a token).
Verify with `gh auth status`.

### 5. Set the Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Add this to your shell profile (`.bashrc` / `.zshrc`) for persistence.

### 5. Add the GitHub Actions secret

In your fork on GitHub: **Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|------|-------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |

`GITHUB_TOKEN` is provided automatically by Actions with `contents: write` and
`pull-requests: write` permissions — no manual setup needed.

### 6. Initialize the preprint version baseline

This scans all existing `limit_data/**/*.txt` files, records their current
arXiv versions, and writes `pipeline/state/preprint_versions.json`. Run this
**once** before the first scheduled workflow run. No PRs are created.

```bash
python -m pipeline.preprint_checker --init-only
git add pipeline/state/preprint_versions.json
git commit -m "chore: initialize preprint version baseline"
git push origin master
```

### 7. Verify with a dry run

```bash
# Check the daily digest finds papers and extracts data (writes nothing)
python -m pipeline.orchestrator --dry-run

# Force-process a known paper to test the full extraction path
python -m pipeline.orchestrator --arxiv-id 2003.13144 --dry-run
```

### 8. Enable the Actions workflows

GitHub disables scheduled workflows on forks by default. Go to
**Actions → (select a workflow) → Enable workflow** for:
- `Daily arXiv Digest`
- `Weekly Preprint Update Checker`
- `Backfill Historical Papers` (manual-only — no schedule to enable, but the
  workflow must be present on the default branch for `workflow_dispatch` to work)

Then trigger a manual run via **Run workflow** to confirm everything works
end-to-end before the first scheduled execution.

---

## Running Locally

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Dry run: print what would happen, write nothing
python -m pipeline.orchestrator --dry-run

# Process a specific paper end-to-end (creates branch + PR)
python -m pipeline.orchestrator --arxiv-id 2412.12345

# Initialize preprint version baseline (first-time only)
python -m pipeline.preprint_checker --init-only

# Run preprint check without opening PRs
python -m pipeline.preprint_checker --dry-run

# Historical backfill: discover candidates (no extraction)
python -m pipeline.backfill --date-from 2020-01-01 --date-to 2024-12-31 --min-citations 10 --discover-only

# Historical backfill: process queued candidates
python -m pipeline.backfill --resume --max-papers 10
```

`GH_TOKEN` (or `GITHUB_TOKEN` in Actions) must be set when creating PRs.

### `[skip ci]` on state commits

Both Actions workflows commit state files back to `master` with the message
suffix `[skip ci]`. This prevents the state commit from re-triggering other
workflows that watch for pushes to `master`. Do not remove that suffix if you
add push-triggered workflows.

---

## PR Formats

### Daily digest PR

```
Add {ExperimentName} {CouplingType} limit (arXiv:{id})
[LOW CONFIDENCE] Add ...    ← extraction confidence < 60%
[PROJECTION] Add ...        ← sensitivity projection
```

Body includes: paper title and link, data source (table / text / vision),
mass and coupling range, corrections applied, corrections flagged for review,
files changed, embedded updated plot PNG.

### Preprint update PR

```
Update {ExperimentName} {CouplingType}: arXiv:{id} v{old}→v{new}
```

Body includes: links to old and new arXiv versions, Claude-generated summary
of what changed, old vs. new data comparison, corrections re-applied, note
that the paper is still a preprint.

### Backfill PR

```
[BACKFILL] Add {ExperimentName} {CouplingType} limit (arXiv:{id})
[BACKFILL] [LOW CONFIDENCE] Add ...
[BACKFILL] [PROJECTION] Add ...
```

Body includes: paper title and link, citation count, data source, extraction
confidence, corrections applied/flagged, files changed, highlighted plot, and
a note that this was discovered via historical backfill.
