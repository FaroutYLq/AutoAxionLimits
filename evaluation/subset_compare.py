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
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import (
    _authoritative_coupling,
    _normalize_predicted_coupling,
    _usable_gt_stats,
)
from evaluation.conventions import (
    UNCONVERTIBLE,
    canonical_convention,
    classify_reported_convention,
    file_source_convention,
    to_canonical,
)
from evaluation.metrics import (
    _deduplicate_mass,
    _filter_boundary,
    compute_interpolation_metrics,
)
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


def _canonicalize_curve(coupling_type, arr: np.ndarray, token) -> np.ndarray:
    """Apply a vetted `to_canonical` conversion to an Nx2 curve (no-op if token is
    None / unknown). Returns the (possibly converted) array."""
    if token is None or arr is None or len(arr) == 0:
        return arr
    pts = [(float(m), float(g)) for m, g in arr]
    out, _note = to_canonical(coupling_type, pts, token)
    if not out:
        return arr
    return np.array(out, dtype=float, ndmin=2)


# Single-point-GT comparison (#612). Many results are a SINGLE limit value at one
# operating mass (e.g. 1706.00209 ORGAN: g_ag<2.02e-12 @ 110 ueV; 2208.06519
# QuantumCyclotron; the 2020 QUAX point in 1806.00310). The O'Hare file then has
# one distinct mass (a point reference, not a curve), so curve interpolation
# cannot run and the paper is discarded as no_comparable_gt / zero_overlap even
# when the extracted coupling matches. We score it as a single point: pair each
# extracted point with the nearest-in-log-mass GT point and, if they sit at the
# same operating mass (within tolerance), take |Δ log10 coupling|.
_SINGLE_POINT_MASS_TOL_DEX = 0.3   # ~factor 2 in mass = the same experimental point
_SINGLE_POINT_MAX_EXT_MASSES = 3   # fallback only for sparse single-value extractions


def _maybe_canonicalize(result: dict, predicted_ct: str, ext_array: np.ndarray,
                        gt_entry, gt_data: np.ndarray):
    """Apply both-sides convention canonicalization (#536/#587 registry).

    Returns ``(ext_array, gt_data, unconvertible)``. ``unconvertible`` is True
    when the extraction declares a recognized but non-convertible convention
    (#604) — caller should treat as convention_mismatch. No-op (and never
    unconvertible) when the extraction does not declare a convention, so
    field-less old snapshots stay raw.
    """
    if not result.get("coupling_convention"):
        return ext_array, gt_data, False
    ext_token = classify_reported_convention(predicted_ct, result.get("coupling_convention"))
    if ext_token == UNCONVERTIBLE:
        return ext_array, gt_data, True
    gt_token = file_source_convention(gt_entry.reference_repo_file, predicted_ct)
    return (_canonicalize_curve(predicted_ct, ext_array, ext_token),
            _canonicalize_curve(predicted_ct, gt_data, gt_token), False)


def _residuals_at(curve: np.ndarray, reference: np.ndarray,
                  tol_dex: float) -> np.ndarray | None:
    """|Δ log10 coupling| of ``curve`` evaluated at each ``reference`` point.

    Both Nx2 (mass, coupling), already boundary-filtered & positive. The curve is
    interpolated in log-log when it has >= 2 distinct masses; for reference
    masses outside the curve's range (or a single-point curve) the nearest curve
    point is used, but only if it is within ``tol_dex`` of the reference mass
    (the same operating mass). Returns the residuals at the matched reference
    points, or ``None`` if none match.
    """
    cm, cc = _deduplicate_mass(np.log10(curve[:, 0]), np.log10(curve[:, 1]))
    # Dedup the reference to one (strongest) point per operating mass: duplicate
    # rows at the same mass are the same point, not extra coverage.
    rm, rc = _deduplicate_mass(np.log10(reference[:, 0]), np.log10(reference[:, 1]))
    interp = None
    if len(cm) >= 2:
        interp = interp1d(cm, cc, kind="linear", bounds_error=False,
                          fill_value=np.nan)
    out = []
    for k in range(len(rm)):
        val = float(interp(rm[k])) if interp is not None else float("nan")
        if not np.isfinite(val):
            j = int(np.argmin(np.abs(cm - rm[k])))
            if abs(cm[j] - rm[k]) <= tol_dex:
                val = cc[j]
        if np.isfinite(val):
            out.append(abs(val - rc[k]))
    return np.array(out) if out else None


def _single_point_compare(curve: np.ndarray, reference: np.ndarray,
                          predicted_ct: str, require_sparse_ref: bool = False):
    """Single-point residual: evaluate ``curve`` at the ``reference`` masses.

    ``reference`` is the side with the trustworthy operating mass(es) — the
    single GT point (single-mass GT), or the sparse single-value extraction (the
    fallback when a GT curve cannot be interpolated against a 1-point read).
    Returns ``(median_resid, n_matched, coverage)`` over the reference points, or
    ``None`` if nothing matches.

    ``require_sparse_ref`` refuses to score a rich multi-point reference this way
    — used in the curve fallback so a genuine wrong-window curve failure is not
    masked by one lucky near-mass extracted point.
    """
    ceil = _ceil_for(predicted_ct)
    c = _filter_boundary(curve, ceil)
    r = _filter_boundary(reference, ceil)
    if len(c) == 0 or len(r) == 0:
        return None
    n_ref_masses = int(np.unique(r[:, 0]).size)
    if require_sparse_ref and n_ref_masses > _SINGLE_POINT_MAX_EXT_MASSES:
        return None
    res = _residuals_at(c, r, _SINGLE_POINT_MASS_TOL_DEX)
    if res is None:
        return None
    return float(np.median(res)), len(res), len(res) / n_ref_masses


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

    # Convention guard (#536): a GT curve whose coupling_convention differs from
    # the extraction's expected (canonical) convention is NOT comparable — the
    # residual would be a units/convention gap (e.g. d_e vs the large-valued
    # d_e_large scalar files, 2401.18076/2006.07055 at ~18 dex), not extraction
    # error. We mirror evaluate.py's guard exactly (it had this; the subset
    # comparator did not, so the gate/eval kept scoring these false negatives).
    # None on either side = unknown convention, treated as comparable.
    # Convention canonicalization (#536/#587) is vetted only for select families
    # (axion-nucleon x2 m_N / SNO x m_N, DarkPhoton eps^2->chi, AxionEDM). Scalars
    # are NOT converted here — they remain governed by the convention_mismatch
    # guard. Back-compat: only canonicalize when the extraction DECLARES its
    # convention; field-less snapshots are left raw so converting one side alone
    # cannot break a shared-convention match.
    expected_conv, _ = canonical_convention(predicted_ct)
    multi_candidates = []   # n_mass >= 2 : a comparable curve
    single_candidates = []  # n_mass == 1 : a single-point (operating-mass) reference
    has_convention_mismatch = False
    for e in paper_entries:
        if _authoritative_coupling(e) != predicted_ct:
            continue
        if (expected_conv is not None
                and e.coupling_convention is not None
                and e.coupling_convention != expected_conv):
            has_convention_mismatch = True
            continue
        gt = e.load_data()
        if gt is None:
            gt = e.load_reference_data(PROJECT_ROOT)
        if gt is None:
            continue
        _, n_mass = _usable_gt_stats(gt, predicted_ct)
        if n_mass >= 2:
            multi_candidates.append((n_mass, e, gt))
        elif n_mass == 1:
            single_candidates.append((1, e, gt))

    if multi_candidates:
        multi_candidates.sort(key=lambda t: -t[0])
        _, gt_entry, gt_data = multi_candidates[0]
        ext_c, gt_c, unconvertible = _maybe_canonicalize(
            result, predicted_ct, ext_array, gt_entry, gt_data)
        if unconvertible:
            rec["status"] = "convention_mismatch"
            return rec
        im = compute_interpolation_metrics(arxiv_id, ext_c, gt_c,
                                           coupling_type=predicted_ct)
        rec["coverage"] = im.interpolation_coverage
        rec["frac_0_3"] = im.frac_within_0_3dex
        rec["n_ext"] = im.num_extracted
        if im.num_interpolatable > 0:
            rec["status"] = "compared"
            rec["median_resid"] = im.median_residual_dex
            return rec
        # Curve interpolation found no overlap. A single-value-limit extraction
        # (sparse) that sits at one of the GT curve's operating masses is scored
        # as a single point rather than discarded as zero_overlap (#612). Guarded
        # to sparse extractions so a genuine wrong-window curve failure is not
        # masked by one lucky near-mass point.
        # reference = the sparse single-value extraction; evaluate the GT CURVE at
        # those operating mass(es). Guarded to a sparse reference so a wrong-window
        # multi-point curve failure is not masked by one lucky near-mass point.
        sp = _single_point_compare(gt_c, ext_c, predicted_ct, require_sparse_ref=True)
        if sp is not None:
            rec["status"] = "compared"
            rec["median_resid"], _n, rec["coverage"] = sp
            rec["single_point"] = True
            return rec
        rec["status"] = "zero_overlap"
        rec["median_resid"] = float("inf")
        ceil = _ceil_for(predicted_ct)
        ext_f = _filter_boundary(ext_c, ceil)
        gt_f = _filter_boundary(gt_c, ceil)
        case = _classify(predicted_ct, _mass_range(ext_f), _mass_range(gt_f),
                         len(ext_f), len(gt_f))
        rec["zo_cause"] = case.classification
        return rec

    # No multi-point GT curve. A single-mass GT is a point reference: compare the
    # extracted coupling at that operating mass (single-point mode, #612).
    if single_candidates:
        _, gt_entry, gt_data = single_candidates[0]
        ext_c, gt_c, unconvertible = _maybe_canonicalize(
            result, predicted_ct, ext_array, gt_entry, gt_data)
        if unconvertible:
            rec["status"] = "convention_mismatch"
            return rec
        ext_f = _filter_boundary(ext_c, _ceil_for(predicted_ct))
        rec["n_ext"] = int(np.unique(ext_f[:, 0]).size) if len(ext_f) else 0
        # reference = the single GT operating point; evaluate the EXTRACTION at it
        # (interpolating a curve, or matching a single read).
        sp = _single_point_compare(ext_c, gt_c, predicted_ct)
        if sp is not None:
            rec["status"] = "compared"
            rec["median_resid"], _n, rec["coverage"] = sp
            rec["single_point"] = True
            return rec
        rec["status"] = "zero_overlap"
        rec["median_resid"] = float("inf")
        rec["zo_cause"] = "single_point_no_match"
        return rec

    # Distinguish a pure convention gap (excluded from residuals, a benchmark
    # units artifact) from a genuinely missing/unusable GT curve.
    rec["status"] = "convention_mismatch" if has_convention_mismatch else "no_comparable_gt"
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
