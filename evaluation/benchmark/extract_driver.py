"""Parallel Haiku validation extraction into an isolated outdir.

Runs evaluate.run_extraction (the same driver subset_eval uses) from a given
WORKTREE over an id list or subset key, fanning out over threads (API-bound).
Writes <outdir>/<id>.json — never touches the main repo's results cache.
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--worktree", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--ids", default=None, help="comma-separated arxiv ids")
ap.add_argument("--key", default=None, help="subset key from evaluation/subset/subset.json")
ap.add_argument("--workers", type=int, default=5)
args = ap.parse_args()

wt = Path(args.worktree).resolve()
os.chdir(wt)
sys.path.insert(0, str(wt))

from evaluation.evaluate import run_extraction  # noqa: E402

# Silent-Opus guard (2026-07-03 incident): never trust the env var alone —
# assert the constant the extractor will actually use.
from pipeline.extractor import CLAUDE_MODEL as _RESOLVED_MODEL
_want = os.environ.get("EXTRACTOR_MODEL")
if _want and _RESOLVED_MODEL != _want:
    sys.exit(f"FATAL: extractor resolved model {_RESOLVED_MODEL!r} != EXTRACTOR_MODEL {_want!r}")
print(f"extractor model: {_RESOLVED_MODEL}", flush=True)
from evaluation.ground_truth import load_ground_truth  # noqa: E402

if args.ids:
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
else:
    subset = json.loads((wt / "evaluation" / "subset" / "subset.json").read_text())
    ids = subset[args.key]

entries = {e.arxiv_id: e for e in load_ground_truth()}
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

def one(aid: str):
    dest = outdir / f"{aid.replace(chr(47), chr(95))}.json"
    if dest.exists():
        return aid, "cached"
    entry = entries.get(aid)
    if entry is None:
        return aid, "no-gt"
    t0 = time.time()
    try:
        result = run_extraction(entry)
    except Exception as e:
        result = {"arxiv_id": aid, "status": "error", "error": str(e)[:400]}
    err = str(result.get("error", ""))
    if "credit balance" in err or "billing" in err.lower():
        # availability error: abort the whole run instead of burning the
        # remaining ids into stubs (#648 semantics at the driver level)
        print(f"FATAL availability error at {aid}: {err[:120]}", flush=True)
        import os as _os
        _os._exit(2)
    dest.write_text(json.dumps(result, indent=1))
    return aid, f"{time.time()-t0:.0f}s src={result.get('data_source')} n={result.get('num_points')}"

print(f"extracting {len(ids)} papers from {wt} -> {outdir} "
      f"(model={os.environ.get('EXTRACTOR_MODEL','<default>')}, "
      f"samples={os.environ.get('AAL_READ_SAMPLES','1')})", flush=True)
done = 0
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(one, aid): aid for aid in ids}
    for f in as_completed(futs):
        aid, msg = f.result()
        done += 1
        print(f"[{done}/{len(ids)}] {aid}: {msg}", flush=True)
print("ALL DONE", flush=True)
