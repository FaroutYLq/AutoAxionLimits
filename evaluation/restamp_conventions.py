"""Re-stamp ``coupling_convention`` / ``coupling_units`` in papers.json.

The convention fields were stamped once (#538) and then preserved verbatim by
``label_ground_truth.py``. Since then the classifier in ``evaluation/conventions.py``
gained the ``coupling_large`` branch (#591) and sentinel-aware range
discrimination (#596), so several entries carry a STALE label that no longer
matches what the classifier would derive today. A stale label silently changes
the comparator's convention guard: e.g. ``ScalarNucleon/IUPUI.txt`` (1410.7267,
fifth-force ``(lambda[m], alpha)``, 7e3..2e17) is stored ``coupling`` == the
canonical expectation, so the guard COMPARES it and scores a ~16-dex
units/convention gap as extraction error instead of excluding it.

This script recomputes (convention, units) for every papers.json entry using the
same resolver the original stamping used — ``infer_convention_for_repo_file``,
which keys off the data file's ``limit_data/<dir>/`` directory, NOT the entry's
``coupling_type`` field (10 entries disagree; the dir is authoritative). It is
idempotent: re-running after the labels are aligned is a no-op.

HOLD set (see ``_HOLD``)
-----------------------
A correct label is normally desirable, but for a paper currently in an eval
subset, flipping a ``*_large`` (excluded) label to its comparable canonical
form would UN-exclude the paper into a residual that is really a separate
(extraction-side convention) problem, regressing the benchmark until that fix
lands. Such entries are held at their prior label with a documented reason
rather than introducing a known regression in this slice.

Usage::

    python -m evaluation.restamp_conventions --dry-run   # report, write nothing
    python -m evaluation.restamp_conventions             # apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.conventions import infer_convention_for_repo_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_JSON = PROJECT_ROOT / "evaluation" / "ground_truth" / "papers.json"

# arXiv IDs whose correct (sentinel-aware) label is intentionally NOT applied in
# this re-stamp, with the reason. 2401.18076 (LIGO O3): the GT files
# (Scalar{Photon,Electron}/LIGO.txt) are genuine d_e (interior 0.16..7.6; the
# 1e20 row is a fill sentinel), so the correct label is `d_e`. But the extractor
# emits this paper as g_phi-gamma [GeV^-1] (~5e-20 = d_e/(sqrt2 M_Pl)), so
# un-excluding it would score an ~18-dex convention gap as error. Recovering it
# needs the GeV^-1 -> d_e conversion (x sqrt2 M_Pl) on the extraction side, which
# is the deferred convention-conversion follow-up. Held until then.
_HOLD: set[str] = {"2401.18076"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the changes and write nothing.")
    args = ap.parse_args()

    doc = json.loads(PAPERS_JSON.read_text())
    papers = doc["papers"]

    changed, held, skipped_no_ref = [], [], 0
    for e in papers:
        ref = e.get("reference_repo_file")
        if not ref:
            skipped_no_ref += 1
            continue
        ct = e.get("coupling_type")
        new_conv, new_units = infer_convention_for_repo_file(
            ref, ct, PROJECT_ROOT / ref)
        old_conv = e.get("coupling_convention")
        if new_conv == old_conv:
            continue
        rec = (e["arxiv_id"], ct, Path(ref).name, old_conv, new_conv)
        if e["arxiv_id"] in _HOLD:
            held.append(rec)
            continue
        changed.append(rec)
        if not args.dry_run:
            e["coupling_convention"] = new_conv
            e["coupling_units"] = new_units

    print(f"Entries:                 {len(papers)}")
    print(f"No reference_repo_file:  {skipped_no_ref}")
    print(f"Held (documented):       {len(held)}")
    for r in held:
        print(f"    HOLD {r[0]:12s} {r[1]:14s} {r[2]:22s} {r[3]} -> {r[4]} (kept {r[3]})")
    print(f"Re-stamped:              {len(changed)}")
    for r in changed:
        print(f"    {r[0]:12s} {r[1]:14s} {r[2]:22s} {r[3]} -> {r[4]}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    PAPERS_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nWrote {PAPERS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
