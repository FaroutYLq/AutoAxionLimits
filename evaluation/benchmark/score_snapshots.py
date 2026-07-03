"""Score the final_full346 snapshot dir with the OFFICIAL evaluate.py metrics
path (RESULTS_DIR redirected — the old baseline cache stays untouched), write
metrics + report into the final dir, and print the headline comparison."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/Users/lanqingyuan/Documents/GitHub/AutoAxionLimits")
R = Path("/Users/lanqingyuan/Documents/GitHub/AutoAxionLimits")
FINAL = R / "evaluation/eval_runs/final_full346"

import evaluation.evaluate as ev

# redirect the module's cache dir to the fresh snapshots
ev.RESULTS_DIR = FINAL
shutil.copyfile(R / "evaluation/results/metadata_cache.json",
                FINAL / "metadata_cache.json")

from evaluation.ground_truth import load_ground_truth

entries, results = [], []
for e in load_ground_truth():
    p = FINAL / f"{e.arxiv_id.replace('/', '_')}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    if d.get("status") == "error":
        continue
    entries.append(e)
    results.append(d)

print(f"scoring {len(results)} snapshots / {len(entries)} GT entries")
m = ev.compute_all_metrics(entries, results)
(FINAL / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
ev.write_metrics_summary(m, FINAL / "metrics_summary.json")
ev.generate_report(m, str(FINAL / "report.md"))

old = json.loads((R / "evaluation/results/metrics.json").read_text())


def head(mm, tag):
    ia = mm.get("interpolation_aggregate") or {}
    cl = mm.get("classification") or {}
    print(f"[{tag}] micro median {ia.get('micro_median_residual_dex')}, "
          f"macro {ia.get('macro_median_residual_dex')}, "
          f"compared {ia.get('n_papers_compared')}, "
          f"zero-overlap {ia.get('n_zero_overlap')}, "
          f"ct-acc {(cl.get('coupling_type') or {}).get('accuracy')}")


head(old, "OLD full346")
head(m, "FINAL")
print("SCORING DONE")
