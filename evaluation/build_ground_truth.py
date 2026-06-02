"""Expand the evaluation ground-truth pool from repo-linked limit sources.

Parses docs/*.md, which link each limit_data/**/*.txt file to a
[reference](...) URL on the same bullet line. For every measured-limit file
(non-projection) that has a clear arXiv reference, an entry is emitted.

Existing entries in papers.json are PRESERVED VERBATIM (keyed by
reference_repo_file) because several were hand-corrected in prior PRs
(multi-coupling labels, misclassification fixes). Only genuinely new files
are appended.

Usage:
    python -m evaluation.build_ground_truth --dry-run   # report only
    python -m evaluation.build_ground_truth             # write papers.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
LIMIT_DATA = ROOT / "limit_data"
PAPERS_JSON = ROOT / "evaluation" / "ground_truth" / "papers.json"

# dir basename -> canonical coupling_type label (from pipeline/config.py).
# Identity for most; only VectorB-L differs. Dirs absent from the registry
# (fa, AxionTop) fall back to the dir basename.
try:
    sys.path.insert(0, str(ROOT))
    from pipeline.config import COUPLING_TYPES
    DIR_TO_COUPLING = {
        Path(meta["data_dir"]).name: key for key, meta in COUPLING_TYPES.items()
    }
except Exception:  # pragma: no cover - config import is best-effort
    DIR_TO_COUPLING = {"VectorB-L": "VectorBL"}

# The `fa` directory is the "Mass vs Peccei-Quinn scale" (m_a vs f_a) plane,
# which the pipeline classifies as AxionMass (axes f_a [GeV] / m_a [eV]).
DIR_TO_COUPLING.setdefault("VectorB-L", "VectorBL")
DIR_TO_COUPLING["fa"] = "AxionMass"

FULL_TXT_RE = re.compile(r"limit_data/[A-Za-z0-9_/\-\.]+\.txt")
ARXIV_RE = re.compile(
    r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", re.I
)


def parse_docs_sources() -> dict[str, str]:
    """Map repo file path -> arXiv id, from same-line [reference] links."""
    file_to_arxiv: dict[str, str] = {}
    for md in sorted(DOCS.glob("*.md")):
        for line in md.read_text(errors="replace").splitlines():
            files = FULL_TXT_RE.findall(line)
            if not files:
                continue
            ax = ARXIV_RE.search(line)
            if not ax:
                continue
            for f in files:
                file_to_arxiv.setdefault(f, ax.group(1))
    return file_to_arxiv


def coupling_for(repo_file: str) -> str:
    parts = Path(repo_file).parts
    # limit_data / <CouplingDir> / [Projections/] file.txt
    if len(parts) >= 2 and parts[0] == "limit_data":
        d = parts[1]
        return DIR_TO_COUPLING.get(d, d)
    return "Unknown"


def count_points(repo_file: str) -> int | None:
    path = ROOT / repo_file
    if not path.exists():
        return None
    n = 0
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def make_entry(repo_file: str, arxiv_id: str) -> dict:
    coupling = coupling_for(repo_file)
    npts = count_points(repo_file)
    stem = Path(repo_file).stem
    return {
        "arxiv_id": arxiv_id,
        "paper_title": stem,
        "coupling_type": coupling,
        "is_new_limit": True,
        "is_projection": False,
        "data_source_expected": "table",
        "confidence_level": 0.9,
        "dm_density_assumed": None,
        "difficulty": "medium",
        "tags": ["auto_expanded"],
        "notes": f"Auto-selected from {repo_file}",
        "ground_truth_data_file": f"{arxiv_id.replace('/', '_')}.txt",
        "reference_repo_file": repo_file,
        "ground_truth_mass_range_eV": None,
        "ground_truth_coupling_range": None,
        "ground_truth_num_points": npts,
        "verified_by": "repo_upstream",
        "verification_date": "2026-06-02",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be added without writing.")
    args = ap.parse_args()

    doc = json.loads(PAPERS_JSON.read_text())
    existing = doc["papers"]
    existing_files = {p.get("reference_repo_file") for p in existing}

    file_to_arxiv = parse_docs_sources()

    # Universe: arXiv-sourced, non-projection, file exists on disk.
    candidates = {
        f: a for f, a in file_to_arxiv.items()
        if "Projections/" not in f and (ROOT / f).exists()
    }

    new_entries = []
    for repo_file in sorted(candidates):
        if repo_file in existing_files:
            continue
        new_entries.append(make_entry(repo_file, candidates[repo_file]))

    total_arxiv_after = len(
        {p["arxiv_id"] for p in existing}
        | {e["arxiv_id"] for e in new_entries}
    )

    print(f"Existing entries:            {len(existing)}")
    print(f"Sourced non-proj limit files:{len(candidates)}")
    print(f"New entries to add:          {len(new_entries)}")
    print(f"Total entries after:         {len(existing) + len(new_entries)}")
    print(f"Distinct arXiv after:        {total_arxiv_after}")

    # Coupling-type breakdown of new entries
    from collections import Counter
    bd = Counter(e["coupling_type"] for e in new_entries)
    print("New-entry coupling breakdown:")
    for k, v in sorted(bd.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    doc["papers"] = existing + new_entries
    PAPERS_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nWrote {PAPERS_JSON} ({len(doc['papers'])} entries).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
