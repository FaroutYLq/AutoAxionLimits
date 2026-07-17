"""Parallel full-pool extraction runner for the evaluation benchmark.

The stock ``evaluate.py --extract`` loop is sequential (one paper at a time,
plus a 2 s nap), so a full 346-paper run at ``AAL_READ_SAMPLES=3`` on Opus is
~9 h. Each paper is independent, so this runner fans the extractions out over a
thread pool (the work is API-I/O-bound, not CPU-bound) and writes into the same
``evaluation/results/`` cache that ``--metrics``/``--report`` read.

Two parallel-safety hazards in the sequential path are handled here:

1. **Shared metadata cache.** ``evaluate._fetch_paper_metadata`` does an unlocked
   read-modify-write of ``results/metadata_cache.json`` and hits the
   rate-limited arXiv metadata API. Concurrent writes would corrupt it. Phase 1
   below pre-warms that cache *serially* (and injects a ground-truth-title
   fallback for any ID the arXiv API refuses), so the parallel phase only ever
   reads it — no writes, no arXiv calls.
2. **Per-paper output.** ``_save_result`` writes one ``results/<id>.json`` per
   paper, so there is no write contention between workers.

Resumable: completed IDs are appended (under a lock) to
``results/.parallel_done.json``; relaunch with ``--resume`` to skip them.

Usage:
    AAL_READ_SAMPLES=3 python -m evaluation.parallel_extract --workers 10 --force
    AAL_READ_SAMPLES=3 python -m evaluation.parallel_extract --workers 10 --resume
    AAL_READ_SAMPLES=3 python -m evaluation.parallel_extract --limit 5   # probe
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("parallel_extract")

from evaluation.evaluate import (  # noqa: E402
    RESULTS_DIR,
    _fetch_paper_metadata,
    _load_cached_result,
    _save_result,
    run_extraction,
)
from evaluation.ground_truth import load_ground_truth  # noqa: E402

DONE_PATH = RESULTS_DIR / ".parallel_done.json"


def _unique_entries(include_projections=False):
    """One ground-truth entry per arXiv ID (extraction is per-paper).

    **Measured-limits-only by default.** Projection papers are out of scope for
    the benchmark: the production pipeline skips ``is_projection`` papers, and
    the paper reports the ``*_noproj`` (measured-limits-only) metrics, so there
    is no reason to spend extraction on them. A paper is kept iff it has at least
    one measured (non-projection) ground-truth entry, and its representative
    entry is a measured one when available (a paper may carry both a measured
    curve and a projection curve). Projection-ONLY papers are dropped.

    Pass ``include_projections=True`` to restore the full labelled pool (e.g. to
    measure ``is_projection`` decline accuracy).
    """
    seen = {}
    for e in load_ground_truth():
        prev = seen.get(e.arxiv_id)
        # keep the first entry seen, but upgrade to a measured entry if one turns
        # up later, so a paper that carries both curves is represented (and
        # extracted) as a measured-limit paper rather than a projection.
        if prev is None or (prev.is_projection and not e.is_projection):
            seen[e.arxiv_id] = e
    entries = list(seen.values())
    if not include_projections:
        entries = [e for e in entries if not e.is_projection]
    return entries


def _prewarm_metadata(entries, fetch=True):
    """Serially ensure every ID has a metadata-cache entry (no parallel writes).

    Cache hits are instant. For misses, when ``fetch`` is True we try arXiv
    (rate-limited, with backoff); when arXiv is in a 429 storm this is far too
    slow, so ``fetch=False`` skips the network entirely and injects a
    ground-truth-title fallback (empty abstract). The extractor still reads the
    full PDF text — which contains the abstract — so only the lightweight
    title+abstract pre-classifier sees the degraded context. Either way every ID
    ends with a cache entry, so the parallel phase only reads the shared file.
    """
    cache_path = RESULTS_DIR / "metadata_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [e for e in entries if e.arxiv_id not in cache]
    logger.info("metadata pre-warm: %d cached, %d missing (fetch=%s)",
                len(entries) - len(missing), len(missing), fetch)
    for i, e in enumerate(missing, 1):
        if fetch:
            try:
                _fetch_paper_metadata(e.arxiv_id, cache_path)
            except Exception as exc:  # never fatal
                logger.warning("metadata warm failed for %s: %s", e.arxiv_id, exc)
            cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if e.arxiv_id not in cache:
            cache[e.arxiv_id] = {"title": e.paper_title, "abstract": ""}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
        if i % 20 == 0:
            logger.info("metadata pre-warm: %d/%d", i, len(missing))
    logger.info("metadata pre-warm complete")


def _load_done(resume: bool) -> set[str]:
    if resume and DONE_PATH.exists():
        return set(json.loads(DONE_PATH.read_text()))
    return set()


def main():
    ap = argparse.ArgumentParser(description="Parallel full-pool extraction")
    ap.add_argument("--workers", type=int, default=10, help="thread pool size")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if a results/<id>.json already exists")
    ap.add_argument("--resume", action="store_true",
                    help="skip IDs recorded in results/.parallel_done.json")
    ap.add_argument("--limit", type=int, default=0, help="probe: only first N papers")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip arXiv metadata fetch for uncached IDs (inject GT-title "
                         "fallback); use when arXiv is 429-storming")
    ap.add_argument("--include-projections", action="store_true",
                    help="restore the full labelled pool (default is measured-limits "
                         "only; projections are out of benchmark scope)")
    args = ap.parse_args()

    entries = _unique_entries(include_projections=args.include_projections)
    if args.limit:
        entries = entries[: args.limit]
    logger.info("%s pool: %d unique papers; workers=%d AAL_READ_SAMPLES=%s",
                "full (incl. projections)" if args.include_projections else "measured-limits",
                len(entries), args.workers, os.environ.get("AAL_READ_SAMPLES", "1"))

    _prewarm_metadata(entries, fetch=not args.no_fetch)

    done = _load_done(args.resume)
    if not args.resume:
        # fresh run for this invocation's done-tracking
        DONE_PATH.write_text(json.dumps(sorted(done)))

    todo = []
    for e in entries:
        if e.arxiv_id in done:
            continue
        if not args.force and _load_cached_result(e.arxiv_id) is not None:
            done.add(e.arxiv_id)
            continue
        todo.append(e)
    logger.info("to extract: %d (skipping %d already done/cached)",
                len(todo), len(entries) - len(todo))

    lock = threading.Lock()
    counter = {"n": 0, "err": 0}
    t0 = time.time()

    def _work(entry):
        result = run_extraction(entry)
        _save_result(entry.arxiv_id, result)
        return entry.arxiv_id, result.get("error")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_work, e): e for e in todo}
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                aid, err = fut.result()
            except Exception as exc:
                aid, err = entry.arxiv_id, f"worker crash: {exc}"
            with lock:
                counter["n"] += 1
                if err:
                    counter["err"] += 1
                done.add(aid)
                DONE_PATH.write_text(json.dumps(sorted(done)))
                n, total = counter["n"], len(todo)
                rate = n / max(time.time() - t0, 1e-9)
                eta = (total - n) / rate if rate > 0 else float("inf")
            tag = f"ERROR: {err}" if err else "ok"
            logger.info("[%d/%d] %s -> %s (%.1f/min, ETA %.0f min)",
                        n, total, aid, tag, rate * 60, eta / 60)

    logger.info("DONE: %d extracted, %d errored, %.0f min elapsed",
                counter["n"], counter["err"], (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
