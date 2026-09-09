# AutoAxionLimits

[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21213786.svg)](https://doi.org/10.5281/zenodo.21213786)

AutoAxionLimits keeps axion and other light dark-matter limit plots up to date
from research papers. It extends [AxionLimits](https://github.com/cajohare/AxionLimits).
Every update goes through a pull request for human review.

Use these skills in Claude Code or Codex:

- **Find new papers:** `daily-arxiv-digest`.
- **Check revised papers:** `weekly-preprint-check`.
- **Import older results:** `backfill-extraction`.

After [setup](docs/local-pipelines.md#before-running), preview one new paper:

```bash
bash scripts/run_pipeline.sh run daily -- --dry-run --max-papers 1
```

Previews preserve saved progress. Local runs use a separate copy of the repository.

[Setup and usage](docs/local-pipelines.md) ·
[Pipeline details](docs/pipeline.md) ·
[Plots, data and credits](docs/plots.md) ·
[References](refs/README.md)
