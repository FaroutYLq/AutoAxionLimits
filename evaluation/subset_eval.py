"""Subset before/after evaluation harness for issues #550 and #561.

Re-extracts a fixed subset of papers into an isolated output directory (so the
global results/ cache and the 212-paper aggregate report are untouched), then
compares two such directories with the *same* pairing / interpolation /
zero-overlap logic used by the main report.

Usage:
    # Extract the union subset into a snapshot dir (calls Claude API)
    python -m evaluation.subset_eval extract --key union --outdir evaluation/eval_runs/before

    # Determinism repeats on a key (e.g. unit_offset), N runs each
    python -m evaluation.subset_eval extract --key unit_offset --repeats 3 \
        --outdir evaluation/eval_runs/before_repeats

    # Compare two snapshot dirs on the union subset
    python -m evaluation.subset_eval compare \
        --before evaluation/eval_runs/before --after evaluation/eval_runs/after
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("subset_eval")

SUBSET_PATH = Path(__file__).parent / "subset" / "subset.json"


def _load_subset(key: str) -> list[str]:
    data = json.loads(SUBSET_PATH.read_text())
    if key not in data:
        raise SystemExit(f"key {key!r} not in subset.json (have {list(data)})")
    return data[key]


def cmd_extract(args):
    from evaluation.evaluate import run_extraction
    from evaluation.ground_truth import load_ground_truth

    ids = _load_subset(args.key)
    entries = {e.arxiv_id: e for e in load_ground_truth()}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, aid in enumerate(ids, 1):
        entry = entries.get(aid)
        if entry is None:
            logger.warning("[%d/%d] %s not in ground truth; skipping", i, len(ids), aid)
            continue
        for r in range(args.repeats):
            suffix = "" if args.repeats == 1 else f"_r{r}"
            dest = outdir / f"{aid}{suffix}.json"
            if dest.exists() and not args.force:
                logger.info("[%d/%d] cached %s", i, len(ids), dest.name)
                continue
            logger.info("[%d/%d] extracting %s%s", i, len(ids), aid, suffix)
            result = run_extraction(entry)
            dest.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Done; wrote snapshots to %s", outdir)


def cmd_compare(args):
    # Implemented by the comparison harness; imported lazily so `extract`
    # works even before that half lands.
    from evaluation.subset_compare import run_compare
    run_compare(Path(args.before), Path(args.after), _load_subset(args.key),
                out=Path(args.out) if args.out else None)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="re-extract a subset into a snapshot dir")
    pe.add_argument("--key", default="union", help="subset.json key (default: union)")
    pe.add_argument("--outdir", required=True)
    pe.add_argument("--repeats", type=int, default=1)
    pe.add_argument("--force", action="store_true")
    pe.set_defaults(func=cmd_extract)

    pc = sub.add_parser("compare", help="compare two snapshot dirs")
    pc.add_argument("--before", required=True)
    pc.add_argument("--after", required=True)
    pc.add_argument("--key", default="union")
    pc.add_argument("--out", default=None, help="write markdown report here")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
