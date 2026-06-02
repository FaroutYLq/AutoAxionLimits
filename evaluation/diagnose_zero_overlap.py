"""Diagnose the zero-overlap bucket (issue #540).

After PR #535, ~20% of compared papers have ZERO mass-range overlap between the
extracted curve and the ground-truth (GT) curve: the GT masses all fall outside
the extracted mass range, so the interpolation metric returns a median residual
of inf and an interpolation coverage of 0%. The aggregate metric only *counts*
these papers; it never says *why* the curves miss each other.

This standalone diagnostic reproduces the SAME extracted-vs-GT pairing that
``evaluate.compute_all_metrics`` uses (same authoritative-coupling logic, same
candidate-selection, same boundary-sentinel filtering), recomputes the
interpolation metric to find the zero-overlap papers, and classifies each one:

  unit_offset    -- extracted and GT mass ranges are displaced by a roughly
                    constant multiplicative factor (a unit-conversion error).
                    The factor is reported and flagged clean/suspicious against
                    known conversion constants.
  too_few_points -- only 1-2 usable extracted points; cannot span the GT range
                    no matter where it sits (interp1d needs >= 2 points).
  wrong_window   -- enough points, ranges do NOT differ by a constant factor;
                    a genuinely different mass region (figure/panel misread).

It writes a markdown report (breakdown counts + per-paper table) and prints the
same summary.  No API calls -- it reads only the cached extraction results in
``evaluation/results/`` and the GT data shipped in ``evaluation/ground_truth/``.

Usage::

    python -m evaluation.diagnose_zero_overlap
    python evaluation/diagnose_zero_overlap.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path (mirrors evaluate.py so pipeline.* imports resolve).
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import (  # read-only reuse
    RESULTS_DIR,
    _authoritative_coupling,
    _load_cached_result,
    _normalize_predicted_coupling,
    _usable_gt_stats,
)
from evaluation.ground_truth import GroundTruthEntry, load_ground_truth
from evaluation.metrics import (
    _COUPLING_CEILINGS,
    _DEFAULT_COUPLING_CEIL,
    _filter_boundary,
    compute_interpolation_metrics,
)

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).parent / "zero_overlap_report.md"

# Known / suspicious mass-unit conversion factors. If the inferred offset
# between the extracted and GT mass ranges lands near one of these (in log10
# space, within HALF a decade unless the constant is itself a round power of
# ten), the offset is almost certainly a unit-conversion bug rather than a
# coincidence. Tagged so the report can call out a recurring culprit.
SUSPICIOUS_FACTORS: list[tuple[float, str]] = [
    (4.136e-15, "eV per Hz (frequency->energy: E = h*f)"),
    (1.0 / 4.136e-15, "Hz per eV (energy->frequency, inverse of h)"),  # ~2.418e14
    (4.136e-6, "eV per GHz (GHz->eV)"),
    (1.0 / 4.136e-6, "GHz per eV (inverse)"),  # ~2.418e5
    (4.136e-9, "eV per MHz (MHz->eV)"),
    (1.0 / 4.136e-9, "MHz per eV (inverse)"),
    (1e9, "GeV per eV (1e9)"),
    (1e-9, "eV per GeV (1e-9)"),
    (1e6, "1e6 power-of-ten (MeV<->eV / wrong exponent)"),
    (1e-6, "1e-6 power-of-ten (eV<->MeV / micro prefix)"),
    (1e3, "1e3 power-of-ten (keV<->eV / wrong exponent)"),
    (1e-3, "1e-3 power-of-ten (eV<->keV / milli prefix)"),
]

# Tolerance (in log10 units) for matching an inferred offset to a known factor.
# Round powers of ten are matched tightly (0.15 dex) since they are exact when
# the bug is a misplaced exponent; the physical constants get a looser window
# (0.4 dex) to absorb the fact that the offset is estimated from range *edges*,
# not a true per-point ratio.
_POW10_TOL = 0.15
_CONST_TOL = 0.40


def _is_pow10(factor: float) -> bool:
    log = np.log10(factor)
    return abs(log - round(log)) < 1e-6


@dataclass
class ZeroOverlapCase:
    arxiv_id: str
    coupling: str
    ext_mass_lo: float
    ext_mass_hi: float
    gt_mass_lo: float
    gt_mass_hi: float
    num_ext_points: int  # usable extracted points after boundary filtering
    num_gt_points: int
    offset_factor: Optional[float]  # GT / extracted, geometric, None if N/A
    offset_log10: Optional[float]
    matched_factor_label: Optional[str]
    classification: str  # unit_offset | too_few_points | wrong_window
    note: str


def _ceil_for(coupling: str) -> float:
    return _COUPLING_CEILINGS.get(coupling, _DEFAULT_COUPLING_CEIL)


def _mass_range(data: np.ndarray) -> Optional[tuple[float, float]]:
    """Min/max positive mass in an Nx2 array, or None if empty."""
    if data is None or len(data) == 0:
        return None
    masses = data[:, 0]
    masses = masses[masses > 0]
    if len(masses) == 0:
        return None
    return float(masses.min()), float(masses.max())


def _match_known_factor(offset_log10: float) -> Optional[str]:
    """Return a label if the (log10) offset matches a known conversion factor."""
    best: Optional[tuple[float, str]] = None
    for factor, label in SUSPICIOUS_FACTORS:
        target = np.log10(factor)
        tol = _POW10_TOL if _is_pow10(factor) else _CONST_TOL
        dist = abs(offset_log10 - target)
        if dist <= tol and (best is None or dist < best[0]):
            best = (dist, f"{factor:.3e} = {label}")
    return best[1] if best else None


def _classify(
    coupling: str,
    ext_range: Optional[tuple[float, float]],
    gt_range: Optional[tuple[float, float]],
    n_ext: int,
    n_gt: int,
) -> ZeroOverlapCase:
    """Classify one zero-overlap paper. Caller has already established that the
    interpolation metric produced zero overlap for this (ext, gt) pair."""
    ext_lo, ext_hi = ext_range if ext_range else (float("nan"), float("nan"))
    gt_lo, gt_hi = gt_range if gt_range else (float("nan"), float("nan"))

    offset_factor: Optional[float] = None
    offset_log10: Optional[float] = None
    matched: Optional[str] = None

    # Geometric-centre offset: how far must the extracted window slide (in log
    # space) to sit on the GT window? A near-constant multiplicative factor is
    # the signature of a unit-conversion bug.
    if ext_range and gt_range and ext_lo > 0 and gt_lo > 0:
        ext_centre = np.sqrt(ext_lo * ext_hi)
        gt_centre = np.sqrt(gt_lo * gt_hi)
        offset_factor = gt_centre / ext_centre
        offset_log10 = float(np.log10(offset_factor))
        matched = _match_known_factor(offset_log10)

    # Decision order:
    #  1. Too few extracted points -> the curve physically cannot span a range,
    #     so it can never overlap regardless of position. Report first because
    #     the "offset" of a 1-2 point cloud is not meaningful.
    #  2. A clean match to a known conversion factor -> unit_offset.
    #  3. Otherwise a real different window -> wrong_window. (We still report
    #     the offset factor so a reviewer can spot a near-miss conversion.)
    if n_ext < 2:
        classification = "too_few_points"
        note = f"only {n_ext} usable extracted point(s); interp1d needs >= 2"
    elif matched is not None:
        classification = "unit_offset"
        note = f"offset ~ {matched}"
    else:
        classification = "wrong_window"
        if offset_factor is not None:
            note = (f"ranges differ by x{offset_factor:.3e} "
                    f"({offset_log10:+.2f} dex); no clean conversion match")
        else:
            note = "could not compute offset (degenerate range)"

    return ZeroOverlapCase(
        arxiv_id="",  # filled by caller
        coupling=coupling,
        ext_mass_lo=ext_lo, ext_mass_hi=ext_hi,
        gt_mass_lo=gt_lo, gt_mass_hi=gt_hi,
        num_ext_points=n_ext, num_gt_points=n_gt,
        offset_factor=offset_factor, offset_log10=offset_log10,
        matched_factor_label=matched,
        classification=classification, note=note,
    )


def find_zero_overlap_cases(
    entries: list[GroundTruthEntry],
) -> list[ZeroOverlapCase]:
    """Reproduce evaluate.py's pairing, find papers with zero mass-range overlap,
    classify each. One result per paper (results are identical across a paper's
    GT entries)."""
    # Group GT entries by paper, mirroring compute_all_metrics.
    by_id: dict[str, list[GroundTruthEntry]] = {}
    for e in entries:
        by_id.setdefault(e.arxiv_id, []).append(e)

    cases: list[ZeroOverlapCase] = []

    for arxiv_id, paper_entries in by_id.items():
        result = _load_cached_result(arxiv_id)
        if result is None or "error" in result:
            continue

        predicted_ct = _normalize_predicted_coupling(result.get("coupling_type"))
        if predicted_ct is None:
            continue

        true_couplings = {_authoritative_coupling(e) for e in paper_entries}
        if predicted_ct not in true_couplings:
            continue  # no comparable GT -> not a curve comparison

        extracted_points = result.get("data_points", [])
        if not extracted_points:
            continue
        ext_array = np.array(extracted_points, dtype=float, ndmin=2)

        # Same candidate selection as evaluate.py: same-coupling GT entries with
        # >= 2 distinct usable masses; the richest GT curve wins.
        candidates = []
        for e in paper_entries:
            if _authoritative_coupling(e) != predicted_ct:
                continue
            gt = e.load_data()
            if gt is None:
                gt = e.load_reference_data(PROJECT_ROOT)
            if gt is None:
                continue
            _, n_mass = _usable_gt_stats(gt, predicted_ct)
            if n_mass >= 2:
                candidates.append((n_mass, gt))
        if not candidates:
            continue  # gt_point_reference / gt_unusable -> not "compared"
        candidates.sort(key=lambda t: -t[0])
        gt_data = candidates[0][1]

        # Recompute the SAME interpolation metric to identify zero overlap.
        im = compute_interpolation_metrics(
            arxiv_id, ext_array, gt_data, coupling_type=predicted_ct
        )
        if im.num_interpolatable > 0:
            continue  # this paper DID overlap -> not in the bucket

        # Zero overlap. Classify using the SAME boundary-filtered arrays.
        ceil = _ceil_for(predicted_ct)
        ext_f = _filter_boundary(ext_array, ceil)
        gt_f = _filter_boundary(gt_data, ceil)
        ext_range = _mass_range(ext_f)
        gt_range = _mass_range(gt_f)

        case = _classify(
            predicted_ct, ext_range, gt_range, len(ext_f), len(gt_f)
        )
        case.arxiv_id = arxiv_id
        cases.append(case)

    cases.sort(key=lambda c: (c.classification, c.arxiv_id))
    return cases


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.3e}"


def build_report(cases: list[ZeroOverlapCase]) -> str:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.classification] = counts.get(c.classification, 0) + 1

    lines: list[str] = []
    lines.append("# Zero-overlap diagnostic (issue #540)\n")
    lines.append(
        "Papers where the extracted curve and the ground-truth curve share "
        "**zero mass-range overlap** (interpolation coverage = 0%, median "
        "residual = inf). Pairing, boundary filtering, and the interpolation "
        "metric are reproduced exactly from `evaluate.py` / `metrics.py`.\n"
    )
    lines.append(f"**Total zero-overlap papers:** {len(cases)}\n")
    lines.append("## Breakdown by cause\n")
    lines.append("| Cause | Count |")
    lines.append("|---|---|")
    for cause in ("unit_offset", "wrong_window", "too_few_points"):
        lines.append(f"| `{cause}` | {counts.get(cause, 0)} |")
    lines.append(f"| **total** | **{len(cases)}** |\n")

    # Recurring conversion factor: among unit_offset cases, which known factor
    # label dominates?
    factor_tally: dict[str, int] = {}
    for c in cases:
        if c.classification == "unit_offset" and c.matched_factor_label:
            factor_tally[c.matched_factor_label] = (
                factor_tally.get(c.matched_factor_label, 0) + 1
            )
    if factor_tally:
        lines.append("## Recurring conversion factors (unit_offset cases)\n")
        lines.append("| Inferred factor | Papers |")
        lines.append("|---|---|")
        for label, n in sorted(factor_tally.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {label} | {n} |")
        top_label, top_n = max(factor_tally.items(), key=lambda kv: kv[1])
        if top_n >= 3:
            lines.append(
                f"\n> **Recurring culprit:** {top_n} papers are off by "
                f"`{top_label}`. This is strong evidence of a single "
                f"mass-unit conversion bug in the extractor (out of scope for "
                f"this diagnostic — flagged for a follow-up fix).\n"
            )
    else:
        lines.append(
            "\n*No `unit_offset` cases matched a known conversion factor.*\n"
        )

    lines.append("## Per-paper detail\n")
    lines.append(
        "| arxiv_id | coupling | extracted mass [eV] | GT mass [eV] | "
        "offset (GT/ext) | n_ext | classification | note |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in cases:
        ext_rng = f"{_fmt(c.ext_mass_lo)} – {_fmt(c.ext_mass_hi)}"
        gt_rng = f"{_fmt(c.gt_mass_lo)} – {_fmt(c.gt_mass_hi)}"
        off = (f"x{c.offset_factor:.2e} ({c.offset_log10:+.2f} dex)"
               if c.offset_factor is not None else "—")
        lines.append(
            f"| {c.arxiv_id} | {c.coupling} | {ext_rng} | {gt_rng} | "
            f"{off} | {c.num_ext_points} | `{c.classification}` | {c.note} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not RESULTS_DIR.exists():
        logger.error("No cached results at %s. Run evaluate.py --extract first.",
                     RESULTS_DIR)
        return

    entries = load_ground_truth()
    logger.info("Loaded %d ground-truth entries", len(entries))

    cases = find_zero_overlap_cases(entries)
    logger.info("Identified %d zero-overlap papers", len(cases))

    report = build_report(cases)
    REPORT_PATH.write_text(report)
    logger.info("Report written to %s", REPORT_PATH)

    # Echo the breakdown to stdout.
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.classification] = counts.get(c.classification, 0) + 1
    print()
    print("=" * 60)
    print(f"ZERO-OVERLAP DIAGNOSTIC — {len(cases)} papers")
    print("=" * 60)
    for cause in ("unit_offset", "wrong_window", "too_few_points"):
        print(f"  {cause:16s}: {counts.get(cause, 0)}")
    print("-" * 60)
    print(report)


if __name__ == "__main__":
    main()
