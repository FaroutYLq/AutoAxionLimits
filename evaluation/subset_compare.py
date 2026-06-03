"""Before/after comparison harness for the #550 + #561 subset eval.

Given two snapshot directories of extraction results (see ``subset_eval.py``),
compute — on the fixed subset only — the same metrics the main report uses, so
"before" and "after" are directly comparable:

  * per-source breakdown (table / text / figure_vision): #papers, #compared,
    #zero-overlap, median residual, %≤0.3 dex  — this is the #550 scoreboard
  * zero-overlap bucket size + cause breakdown (unit_offset / wrong_window /
    too_few_points)  — shared #550/#561 scoreboard (reuses diagnose logic)
  * overall median residual + mean interpolation coverage

It reuses the *exact* pairing / boundary-filter / interpolation logic from
``evaluate.py`` + ``metrics.py`` + ``diagnose_zero_overlap.py`` (no
reimplementation), only swapping the result source from the global cache to an
arbitrary snapshot directory.

Determinism (#561) is scored separately from repeat snapshots (files named
``<id>_r{k}.json``) via :func:`determinism_report`.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import (
    _authoritative_coupling,
    _normalize_predicted_coupling,
    _usable_gt_stats,
)
from evaluation.metrics import _filter_boundary, compute_interpolation_metrics
from evaluation.diagnose_zero_overlap import _classify, _ceil_for, _mass_range
from evaluation.ground_truth import GroundTruthEntry, load_ground_truth


def _load_result_dir(d: Path) -> dict[str, dict]:
    """Load <id>.json (single-run) result files from a snapshot directory."""
    out: dict[str, dict] = {}
    for f in sorted(d.glob("*.json")):
        name = f.stem
        if "_r" in name:  # repeat file, handled by determinism_report
            continue
        try:
            out[name] = json.loads(f.read_text())
        except Exception:
            continue
    return out


def _paper_record(arxiv_id: str, result: dict,
                  paper_entries: list[GroundTruthEntry]) -> dict:
    """Reproduce evaluate.py pairing for one paper; return a comparison record.

    status ∈ {compared, zero_overlap, no_comparable_gt, no_extracted_points,
              no_prediction, error}. For compared/zero_overlap papers, include
      interp metrics and (for zero_overlap) the diagnose classification.
    """
    rec = {"arxiv_id": arxiv_id, "data_source": result.get("data_source"),
           "coupling": result.get("coupling_type"), "status": None,
           "median_resid": None, "coverage": None, "frac_0_3": None,
           "n_ext": None, "zo_cause": None}

    if "error" in result:
        rec["status"] = "error"
        return rec
    predicted_ct = _normalize_predicted_coupling(result.get("coupling_type"))
    if predicted_ct is None:
        rec["status"] = "no_prediction"
        return rec
    true_couplings = {_authoritative_coupling(e) for e in paper_entries}
    if predicted_ct not in true_couplings:
        rec["status"] = "no_comparable_gt"
        return rec
    extracted_points = result.get("data_points", [])
    if not extracted_points:
        rec["status"] = "no_extracted_points"
        return rec
    ext_array = np.array(extracted_points, dtype=float, ndmin=2)

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
        rec["status"] = "no_comparable_gt"  # gt_point_reference / gt_unusable
        return rec
    candidates.sort(key=lambda t: -t[0])
    gt_data = candidates[0][1]

    im = compute_interpolation_metrics(arxiv_id, ext_array, gt_data,
                                       coupling_type=predicted_ct)
    rec["coverage"] = im.interpolation_coverage
    rec["frac_0_3"] = im.frac_within_0_3dex
    rec["n_ext"] = im.num_extracted
    if im.num_interpolatable > 0:
        rec["status"] = "compared"
        rec["median_resid"] = im.median_residual_dex
    else:
        rec["status"] = "zero_overlap"
        rec["median_resid"] = float("inf")
        ceil = _ceil_for(predicted_ct)
        ext_f = _filter_boundary(ext_array, ceil)
        gt_f = _filter_boundary(gt_data, ceil)
        case = _classify(predicted_ct, _mass_range(ext_f), _mass_range(gt_f),
                         len(ext_f), len(gt_f))
        rec["zo_cause"] = case.classification
    return rec


def _summarize(records: list[dict]) -> dict:
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_src[r.get("data_source") or "none"].append(r)

    def _src_row(rs: list[dict]) -> dict:
        compared = [r for r in rs if r["status"] == "compared"]
        zo = [r for r in rs if r["status"] == "zero_overlap"]
        resids = [r["median_resid"] for r in compared
                  if r["median_resid"] is not None and math.isfinite(r["median_resid"])]
        f03 = [r["frac_0_3"] for r in compared if r["frac_0_3"] is not None]
        return {
            "papers": len(rs),
            "compared": len(compared),
            "zero_overlap": len(zo),
            "median_resid": statistics.median(resids) if resids else None,
            "frac_0_3": (sum(f03) / len(f03)) if f03 else None,
        }

    sources = {s: _src_row(rs) for s, rs in sorted(by_src.items())}
    compared_all = [r for r in records if r["status"] == "compared"]
    overlap_resids = [r["median_resid"] for r in compared_all
                      if r["median_resid"] is not None and math.isfinite(r["median_resid"])]
    zo_all = [r for r in records if r["status"] == "zero_overlap"]
    zo_causes: dict[str, int] = defaultdict(int)
    for r in zo_all:
        zo_causes[r["zo_cause"] or "unknown"] += 1
    coverages = [r["coverage"] for r in records
                 if r["coverage"] is not None]
    return {
        "n_total": len(records),
        "n_compared": len(compared_all),
        "n_zero_overlap": len(zo_all),
        "zo_causes": dict(zo_causes),
        "overall_median_resid": statistics.median(overlap_resids) if overlap_resids else None,
        "mean_coverage": (sum(coverages) / len(coverages)) if coverages else None,
        "sources": sources,
    }


def _records_for(results: dict[str, dict],
                 entries_by_id: dict[str, list[GroundTruthEntry]],
                 ids: list[str]) -> list[dict]:
    recs = []
    for aid in ids:
        if aid not in results:
            continue
        recs.append(_paper_record(aid, results[aid], entries_by_id.get(aid, [])))
    return recs


def determinism_report(repeats_dir: Path, ids: list[str]) -> dict:
    """For each id with files <id>_r{k}.json, compute the spread of the
    coupling-value SCALE across repeats. Scale = median(log10(coupling)) over
    a run's data points. Returns per-paper {n_runs, scale_std_dex, scale_range_dex}.
    """
    out: dict[str, dict] = {}
    for aid in ids:
        runs = sorted(repeats_dir.glob(f"{aid}_r*.json"))
        scales = []
        for f in runs:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            pts = d.get("data_points") or []
            gs = [p[1] for p in pts if len(p) >= 2 and p[1] and p[1] > 0]
            if gs:
                scales.append(statistics.median([math.log10(g) for g in gs]))
        if len(scales) >= 2:
            out[aid] = {
                "n_runs": len(scales),
                "scale_std_dex": statistics.pstdev(scales),
                "scale_range_dex": max(scales) - min(scales),
            }
        elif scales:
            out[aid] = {"n_runs": len(scales), "scale_std_dex": 0.0,
                        "scale_range_dex": 0.0}
    return out


def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "∞"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def run_compare(before: Path, after: Path, ids: list[str],
                out: Optional[Path] = None,
                before_repeats: Optional[Path] = None,
                after_repeats: Optional[Path] = None) -> str:
    entries_by_id: dict[str, list[GroundTruthEntry]] = defaultdict(list)
    for e in load_ground_truth():
        entries_by_id[e.arxiv_id].append(e)

    b_results = _load_result_dir(before)
    a_results = _load_result_dir(after)
    b_recs = _records_for(b_results, entries_by_id, ids)
    a_recs = _records_for(a_results, entries_by_id, ids)
    b = _summarize(b_recs)
    a = _summarize(a_recs)

    L: list[str] = []
    L.append("# Subset before/after comparison — issues #550 & #561\n")
    L.append(f"Subset: {len(ids)} papers. "
             f"Before snapshot: `{before}` ({b['n_total']} loaded). "
             f"After snapshot: `{after}` ({a['n_total']} loaded).\n")

    L.append("## Headline\n")
    L.append("| Metric | Before | After | Δ |")
    L.append("|---|---|---|---|")
    def _delta(bv, av, better="down"):
        if bv is None or av is None:
            return "—"
        d = av - bv
        arrow = "✅" if ((d < 0) == (better == "down")) and abs(d) > 1e-9 else (
            "▪️" if abs(d) < 1e-9 else "⚠️")
        return f"{d:+.3f} {arrow}"
    L.append(f"| Overall median residual (dex) | {_fmt(b['overall_median_resid'])} | "
             f"{_fmt(a['overall_median_resid'])} | "
             f"{_delta(b['overall_median_resid'], a['overall_median_resid'], 'down')} |")
    L.append(f"| Zero-overlap papers | {b['n_zero_overlap']} | {a['n_zero_overlap']} | "
             f"{a['n_zero_overlap'] - b['n_zero_overlap']:+d} "
             f"{'✅' if a['n_zero_overlap'] < b['n_zero_overlap'] else ('▪️' if a['n_zero_overlap']==b['n_zero_overlap'] else '⚠️')} |")
    L.append(f"| Mean interp. coverage | {_pct(b['mean_coverage'])} | {_pct(a['mean_coverage'])} | "
             f"{_delta(b['mean_coverage'], a['mean_coverage'], 'up')} |")
    L.append(f"| Papers compared | {b['n_compared']} | {a['n_compared']} | "
             f"{a['n_compared'] - b['n_compared']:+d} |\n")

    L.append("## Zero-overlap causes\n")
    causes = sorted(set(b["zo_causes"]) | set(a["zo_causes"]))
    L.append("| Cause | Before | After |")
    L.append("|---|---|---|")
    for c in causes:
        L.append(f"| {c} | {b['zo_causes'].get(c, 0)} | {a['zo_causes'].get(c, 0)} |")
    L.append("")

    L.append("## Per-source breakdown (#550 scoreboard)\n")
    srcs = sorted(set(b["sources"]) | set(a["sources"]))
    L.append("| Source | Papers (B/A) | Compared (B/A) | Zero-ovl (B/A) | "
             "Med.Resid B→A | ≤0.3dex B→A |")
    L.append("|---|---|---|---|---|---|")
    for s in srcs:
        bs = b["sources"].get(s, {})
        as_ = a["sources"].get(s, {})
        L.append(
            f"| {s} | {bs.get('papers','—')}/{as_.get('papers','—')} | "
            f"{bs.get('compared','—')}/{as_.get('compared','—')} | "
            f"{bs.get('zero_overlap','—')}/{as_.get('zero_overlap','—')} | "
            f"{_fmt(bs.get('median_resid'))} → {_fmt(as_.get('median_resid'))} | "
            f"{_pct(bs.get('frac_0_3'))} → {_pct(as_.get('frac_0_3'))} |")
    L.append("")

    # Determinism (#561)
    if before_repeats or after_repeats:
        L.append("## Determinism (#561) — coupling-scale spread across repeats\n")
        L.append("Scale = median(log10 coupling) per run; lower std = more stable. "
                 "Noise floor = 0.32 dex.\n")
        bd = determinism_report(before_repeats, ids) if before_repeats else {}
        ad = determinism_report(after_repeats, ids) if after_repeats else {}
        keys = sorted(set(bd) | set(ad))
        L.append("| arXiv | Before std (dex) | After std (dex) | Before range | After range |")
        L.append("|---|---|---|---|---|")
        for k in keys:
            L.append(f"| {k} | {_fmt(bd.get(k,{}).get('scale_std_dex'))} | "
                     f"{_fmt(ad.get(k,{}).get('scale_std_dex'))} | "
                     f"{_fmt(bd.get(k,{}).get('scale_range_dex'))} | "
                     f"{_fmt(ad.get(k,{}).get('scale_range_dex'))} |")
        if bd:
            L.append(f"\nMean before std: {statistics.mean(v['scale_std_dex'] for v in bd.values()):.3f} dex")
        if ad:
            L.append(f"  Mean after std: {statistics.mean(v['scale_std_dex'] for v in ad.values()):.3f} dex")
        L.append("")

    # Per-paper movement table (papers that changed status or residual)
    L.append("## Per-paper detail\n")
    L.append("| arXiv | Source B→A | Status B→A | Med.Resid B→A | Cov B→A |")
    L.append("|---|---|---|---|---|")
    b_by = {r["arxiv_id"]: r for r in b_recs}
    a_by = {r["arxiv_id"]: r for r in a_recs}
    for aid in ids:
        rb, ra = b_by.get(aid), a_by.get(aid)
        if rb is None and ra is None:
            continue
        rb = rb or {}
        ra = ra or {}
        L.append(
            f"| {aid} | {rb.get('data_source','—')}→{ra.get('data_source','—')} | "
            f"{rb.get('status','—')}→{ra.get('status','—')} | "
            f"{_fmt(rb.get('median_resid'))}→{_fmt(ra.get('median_resid'))} | "
            f"{_pct(rb.get('coverage'))}→{_pct(ra.get('coverage'))} |")
    L.append("")

    report = "\n".join(L)
    if out:
        out.write_text(report)
    print(report)
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--key", default="union")
    p.add_argument("--out", default=None)
    p.add_argument("--before-repeats", default=None)
    p.add_argument("--after-repeats", default=None)
    args = p.parse_args()
    subset = json.loads((Path(__file__).parent / "subset" / "subset.json").read_text())
    run_compare(
        Path(args.before), Path(args.after), subset[args.key],
        out=Path(args.out) if args.out else None,
        before_repeats=Path(args.before_repeats) if args.before_repeats else None,
        after_repeats=Path(args.after_repeats) if args.after_repeats else None,
    )
