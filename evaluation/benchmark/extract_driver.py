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

# Load triage helpers from the driver's own directory, not the target
# worktree's sys.path: the driver is routinely pointed at older pinned
# worktrees (code-matched repeats) that predate snapshot_triage.py.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "snapshot_triage", Path(__file__).resolve().parent / "snapshot_triage.py")
_triage = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_triage)
classify_snapshot = _triage.classify_snapshot
is_environmental_error = _triage.is_environmental_error


def one(aid: str):
    dest = outdir / f"{aid.replace(chr(47), chr(95))}.json"
    if dest.exists():
        # Only a GOOD snapshot counts as done. An error stub or husk left by
        # an earlier run is re-extracted on resume rather than silently
        # counted as coverage (the 2026-07-30 lesson: 45 husks passed a
        # naive exists() check and deflated coverage unnoticed).
        try:
            kind = classify_snapshot(json.loads(dest.read_text()))
        except Exception:
            kind = "error"
        if kind == "good":
            return aid, "cached"
        dest.unlink()
    entry = entries.get(aid)
    if entry is None:
        return aid, "no-gt"
    t0 = time.time()
    try:
        result = run_extraction(entry)
    except Exception as e:
        result = {"arxiv_id": aid, "status": "error", "error": str(e)[:400]}
    err = str(result.get("error", ""))
    if is_environmental_error(err):
        # Availability/quota error: a property of the RUN, not the paper.
        # Abort instead of burning the remaining ids into stubs (#648
        # semantics at the driver level); nothing is saved for this paper,
        # so a later resume retries it for free.
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
