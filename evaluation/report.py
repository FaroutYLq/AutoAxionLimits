"""Generate evaluation report (markdown + optional calibration plots)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt(val, precision: int = 3) -> str:
    """Format a float, handling inf/None gracefully."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and (val == float("inf") or val != val):  # inf or nan
        return "∞"
    return f"{val:.{precision}f}"


def _pct(val) -> str:
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def generate_report(metrics: dict, output_path: str):
    """Generate a markdown evaluation report."""
    lines: list[str] = []

    lines.append("# AutoAxionLimits Extraction Pipeline — Evaluation Report\n")

    # --- Summary ---
    clf = metrics["classification"]
    agg_interp = metrics.get("interpolation_aggregate", {})
    agg_symmetric = metrics.get("symmetric_aggregate", {})
    agg_curve = metrics.get("curve_aggregate", {})

    cov = metrics.get("comparison_coverage", {})

    lines.append("## Summary\n")
    lines.append(f"- **Papers evaluated**: {metrics.get('n_papers', clf['coupling_type']['total'])}")
    lines.append(f"- **Papers with curve comparison**: {agg_interp.get('n_papers', agg_curve.get('n_papers_with_curves', 0))}")
    lines.append("")

    # --- Comparison coverage ---
    status_counts = cov.get("status_counts", {})
    if status_counts:
        lines.append("## Curve-Comparison Coverage\n")
        lines.append("A curve is scored only against a ground-truth curve of the **same coupling**. "
                     "Papers whose extracted coupling has no matching GT curve are not comparable "
                     "and are excluded from residual statistics (this is not an extraction failure).\n")
        lines.append("| Status | Papers | Meaning |")
        lines.append("|--------|--------|---------|")
        _meaning = {
            "compared": "scored against a same-coupling GT curve",
            "no_comparable_gt": "extracted coupling has no GT curve in the pool (usually a coupling misclassification)",
            "convention_mismatch": "same coupling but the GT curve uses a different convention/units (e.g. f_a [GeV] vs normalized, or d_e vs a large-valued variable) — excluded as a units gap, not extraction error",
            "gt_point_reference": "GT is a single-mass prediction/projection, not a curve (not comparable)",
            "gt_unusable": "GT curve has <2 usable points after boundary filtering",
            "no_prediction": "pipeline returned no coupling type",
            "no_extracted_points": "pipeline returned no data points",
            "extraction_failed": "extraction errored (download/parse/API)",
        }
        for status in ["compared", "no_comparable_gt", "convention_mismatch",
                       "gt_point_reference", "gt_unusable",
                       "no_extracted_points", "no_prediction", "extraction_failed"]:
            if status not in status_counts:
                continue
            lines.append(f"| {status} | {status_counts[status]} | {_meaning.get(status, '')} |")
        lines.append("")

    # --- Classification ---
    lines.append("## Classification Accuracy\n")
    lines.append("| Field | Accuracy | N |")
    lines.append("|-------|----------|---|")
    for field_name in ["coupling_type", "is_new_limit", "is_projection", "data_source"]:
        entry = clf[field_name]
        if entry["total"] == 0:
            lines.append(f"| {field_name} | N/A — no human-verified labels | 0 |")
        else:
            lines.append(f"| {field_name} | {_pct(entry['accuracy'])} | {entry['total']} |")
    lines.append("")
    if any(clf[f]["total"] == 0 for f in ["is_new_limit", "is_projection", "data_source"]):
        lines.append("> `is_new_limit`, `is_projection`, and `data_source` are scored only against "
                     "human-verified ground-truth entries. The current pool is entirely repo-sourced "
                     "(placeholder labels), so these are reported as N/A rather than against placeholders.")
        lines.append("")

    # Classification errors
    coupling_errors = clf["coupling_type"].get("errors", [])
    if coupling_errors:
        lines.append("### Coupling Type Misclassifications\n")
        lines.append("| arXiv ID | Predicted | Expected |")
        lines.append("|----------|-----------|----------|")
        for err in coupling_errors:
            lines.append(f"| {err['arxiv_id']} | {err['predicted']} | {err['expected']} |")
        lines.append("")

    # --- Interpolation Quality (primary) ---
    if agg_interp.get("n_papers", 0) > 0:
        n_all = agg_interp.get("n_papers", 0)
        n_zero = agg_interp.get("n_zero_overlap", 0)
        n_finite = agg_interp.get("n_finite", n_all - n_zero)
        lines.append("## Extraction Quality — Interpolation Metric (primary)\n")
        lines.append("Build log-log interpolation from extracted points, evaluate at ground-truth masses.\n")
        lines.append(f"- **Papers compared**: {n_all} "
                     f"({n_finite} with mass-range overlap, {n_zero} with zero overlap)")
        lines.append("")
        lines.append("**Coupling-value accuracy** (papers with mass-range overlap):")
        lines.append(f"- **Median residual across papers**: {_fmt(agg_interp.get('median_median_residual_dex'))} dex "
                     f"(IQR {_fmt(agg_interp.get('p25_median_residual_dex'))}–{_fmt(agg_interp.get('p75_median_residual_dex'))})")
        lines.append(f"- **Mean residual across papers** (outlier-sensitive): {_fmt(agg_interp.get('mean_median_residual_dex'))} dex")
        lines.append(f"- **Mean fraction within 0.3 dex (factor 2)**: {_pct(agg_interp.get('mean_frac_within_0_3dex'))}")
        lines.append(f"- **Mean fraction within 0.5 dex (factor 3)**: {_pct(agg_interp.get('mean_frac_within_0_5dex'))}")
        lines.append("")
        lines.append("**Mass-range coverage** (a separate failure mode):")
        lines.append(f"- **Mean interpolation coverage**: {_pct(agg_interp.get('mean_interpolation_coverage'))}")
        lines.append(f"- **Zero-overlap papers**: {n_zero}/{n_all} "
                     f"({_pct(n_zero / n_all if n_all else 0)}) — extracted masses miss the GT range entirely "
                     "(usually 1–2 extracted points or the wrong mass window)")
        lines.append("")
        # Reverse pass + symmetric shape metrics (issue #541).
        lines.append("**Reverse pass** (GT interpolated onto the *extracted* masses):")
        lines.append("- Mirrors the forward pass. A large forward-vs-reverse gap, or a reverse "
                     "coverage well below the forward coverage, flags an extraction whose mass "
                     "*extent* or shape disagrees with the GT (e.g. running past the GT range).")
        lines.append(f"- **Median reverse residual across papers**: "
                     f"{_fmt(agg_interp.get('median_median_residual_dex_reverse'))} dex "
                     f"(forward: {_fmt(agg_interp.get('median_median_residual_dex'))} dex)")
        lines.append(f"- **Mean reverse interpolation coverage**: "
                     f"{_pct(agg_interp.get('mean_interpolation_coverage_reverse'))} "
                     f"(forward: {_pct(agg_interp.get('mean_interpolation_coverage'))})")
        lines.append("")

    # --- Per-coupling-type breakdown + macro vs micro (issue #543) ---
    pt = metrics.get("per_type_aggregate", {})
    if pt.get("n_types", 0) > 0:
        thr = pt.get("small_sample_threshold", 5)
        micro = pt.get("micro_median_residual_dex")
        macro = pt.get("macro_median_residual_dex")
        gap = pt.get("macro_minus_micro_dex")
        lines.append("## Residual by Coupling Type — Micro vs Macro Average (issue #543)\n")
        lines.append(
            "The compared-paper pool is dominated by one coupling type "
            "(AxionPhoton), so the per-paper **micro-average** headline is "
            "largely that one type's number. The **macro-average** weights each "
            "coupling type equally (mean of the per-type medians), exposing how "
            "the pipeline does across the *range* of couplings rather than on the "
            "most common one.\n"
        )
        lines.append(f"- **Micro-average median residual** (per paper, {pt.get('n_papers_compared', 0)} papers): "
                     f"{_fmt(micro)} dex")
        lines.append(f"- **Macro-average median residual** (equal weight per type, "
                     f"{pt.get('n_types', 0)} types): {_fmt(macro)} dex")
        if gap is not None:
            direction = "worse" if gap > 0 else "better"
            lines.append(f"- **Macro − micro gap**: {'+' if gap >= 0 else ''}{_fmt(gap)} dex "
                         f"(macro is {direction}; a positive gap means the rarer couplings "
                         f"are harder than the AxionPhoton-dominated micro-average implies)")
        lines.append("")
        lines.append(f"Per-type medians carry a bootstrap 95% CI (1000 resamples). "
                     f"Rows with **N < {thr}** are flagged small-sample — their median "
                     f"and CI are unstable and should not be read as a reliable per-type score.\n")
        lines.append("| Coupling Type | N | Median Resid. (dex) | 95% CI (dex) | Flag |")
        lines.append("|---------------|---|---------------------|--------------|------|")
        for ct, d in pt.get("per_type", {}).items():
            ci_lo = d.get("ci95_lo")
            ci_hi = d.get("ci95_hi")
            ci_str = (f"[{_fmt(ci_lo)}, {_fmt(ci_hi)}]"
                      if ci_lo is not None and ci_hi is not None else "—")
            flag = f"⚠ small-sample (N<{thr})" if d.get("small_sample") else ""
            lines.append(f"| {ct} | {d['n']} | {_fmt(d['median_residual_dex'])} | {ci_str} | {flag} |")
        lines.append("")

    # --- Symmetric / 2-D shape + mass-range agreement (issue #541) ---
    if agg_symmetric.get("n_papers", 0) > 0:
        lines.append("## Shape & Mass-Range Agreement — Symmetric Metrics (complementary)\n")
        lines.append("These are symmetric, 2-D complements to the (asymmetric, vertical-only) "
                     "interpolation residual. **Area-between-curves** integrates "
                     "|Δ log10 coupling| over the overlapping log-mass range and normalises by "
                     "the overlap width (a single shape+offset number, in dex; a pure mass shift "
                     "inflates it even when the vertical residual looks fine). **Mass-range "
                     "Jaccard** is the Jaccard index of the extracted vs GT log-mass intervals "
                     "(1.0 = identical extent; small = over-/under-claimed mass range), reported "
                     "separately from interpolation coverage.\n")
        lines.append(f"- **Papers scored**: {agg_symmetric.get('n_papers', 0)} "
                     f"({agg_symmetric.get('n_finite_area', 0)} with mass overlap for area)")
        lines.append(f"- **Median area-between-curves**: "
                     f"{_fmt(agg_symmetric.get('median_area_between_log'))} dex "
                     f"(mean {_fmt(agg_symmetric.get('mean_area_between_log'))} dex)")
        lines.append(f"- **Median mass-range Jaccard**: "
                     f"{_fmt(agg_symmetric.get('median_mass_jaccard'))} "
                     f"(mean {_fmt(agg_symmetric.get('mean_mass_jaccard'))})")
        lines.append("")

    # --- Per-paper ---
    per_paper = metrics.get("per_paper", [])
    if per_paper:
        lines.append("## Per-Paper Results\n")
        lines.append("| arXiv ID | Coupling | Conf. | Interp. Cov. | Med. Resid. | Rev. Resid. | Area (dex) | Mass Jaccard | ≤0.3 dex | Points |")
        lines.append("|----------|----------|-------|--------------|-------------|-------------|------------|--------------|----------|--------|")
        for p in per_paper:
            if p.get("status") != "extracted":
                lines.append(f"| {p['arxiv_id']} | — | — | FAILED | — | — | — | — | — | — |")
                continue
            coupling_ok = "✓" if p.get("coupling_type_correct") else f"✗ ({p.get('coupling_type_predicted', '?')})"
            conf = _fmt(p.get("extraction_confidence"), 2)
            im = p.get("interp_metrics")
            sm = p.get("symmetric_metrics")
            if im:
                cov_col = _pct(im["interpolation_coverage"])
                med = _fmt(im["median_residual_dex"])
                rev = _fmt(im.get("median_residual_dex_reverse"))
                f03 = _pct(im["frac_within_0_3dex"])
                pts = f"{im['num_extracted']}/{im['num_ground_truth']}"
            else:
                cov_col = p.get("comparison_status", "—")
                med = rev = f03 = pts = "—"
            if sm:
                area = _fmt(sm.get("area_between_log"))
                jacc = _fmt(sm.get("mass_jaccard"))
            else:
                area = jacc = "—"
            lines.append(f"| {p['arxiv_id']} | {coupling_ok} | {conf} | {cov_col} | {med} | {rev} | {area} | {jacc} | {f03} | {pts} |")
        lines.append("")

    # --- Data source breakdown (the meaningful one) ---
    src_bd = metrics.get("source_breakdown", {})
    if src_bd:
        lines.append("## Breakdown by Extraction Source\n")
        lines.append("Median residual is over papers with mass-range overlap; "
                     "zero-overlap papers are listed separately.\n")
        lines.append("| Source | Papers | Compared | Zero-overlap | Med. Resid. | ≤0.3 dex |")
        lines.append("|--------|--------|----------|--------------|-------------|----------|")
        for src in ["table", "figure_vision", "text"]:
            if src not in src_bd:
                continue
            s = src_bd[src]
            lines.append(
                f"| {src} | {s['total']} | {s.get('n_compared', '—')} | "
                f"{s.get('n_zero_overlap', '—')} | "
                f"{_fmt(s.get('median_residual_dex'))} dex | "
                f"{_pct(s.get('mean_frac_within_0_3dex'))} |"
            )
        lines.append("")

    # --- Difficulty breakdown (placeholder labels — informational only) ---
    diff_bd = metrics.get("difficulty_breakdown", {})
    if diff_bd:
        lines.append("## Breakdown by Difficulty\n")
        lines.append("> Difficulty is a placeholder label for the repo-sourced pool "
                     "(nearly all `medium`); this table is informational only.\n")
        lines.append("| Difficulty | Papers | Coupling Acc. | Med. Resid. | ≤0.3 dex |")
        lines.append("|------------|--------|---------------|-------------|----------|")
        for diff in ["easy", "medium", "hard"]:
            if diff not in diff_bd:
                continue
            d = diff_bd[diff]
            lines.append(
                f"| {diff} | {d['total']} | "
                f"{_pct(d['coupling_type_accuracy'])} | "
                f"{_fmt(d.get('median_residual_dex'))} dex | "
                f"{_pct(d.get('mean_frac_within_0_3dex'))} |"
            )
        lines.append("")

    # --- Confidence calibration ---
    cal = metrics.get("confidence_calibration", [])
    non_empty_bins = [b for b in cal if b["n_papers"] > 0]
    if non_empty_bins:
        lines.append("## Confidence Calibration\n")
        lines.append("| Bin | N | Mean Conf. | Actual Acc. | Gap |")
        lines.append("|-----|---|------------|-------------|-----|")
        for b in non_empty_bins:
            gap = b["mean_confidence"] - b["actual_accuracy"]
            lines.append(
                f"| [{_fmt(b['bin_lo'], 1)}–{_fmt(b['bin_hi'], 1)}) | {b['n_papers']} | "
                f"{_pct(b['mean_confidence'])} | {_pct(b['actual_accuracy'])} | "
                f"{'+' if gap >= 0 else ''}{_fmt(gap, 2)} |"
            )
        lines.append("")
        lines.append("> **Interpretation**: Gap > 0 means the pipeline is overconfident; "
                      "Gap < 0 means underconfident.")
        lines.append("")

    # --- Methodology ---
    lines.append("## Methodology\n")
    lines.append("### Curve selection (what each extraction is compared against)")
    lines.append("- A paper usually produces several repo curves (one per coupling). The single "
                 "extraction is compared **only** against the GT curve whose coupling matches the "
                 "extracted coupling type (taken from the data file's `limit_data/<dir>/`).")
    lines.append("- Papers whose extracted coupling has no matching GT curve, or whose GT curve has "
                 "<2 usable points, are reported under Curve-Comparison Coverage and excluded from "
                 "residual statistics — they do not measure extraction quality.")
    lines.append("")
    lines.append("### Caveats on the residual floor")
    lines.append("- The ground truth `g(x_i)` is the **upstream-curated** repo curve (itself digitised "
                 "and rescaled from the same papers), not the paper's raw numbers. A perfect extraction "
                 "still shows a nonzero residual equal to the upstream digitisation/convention gap, so "
                 "the ~0.5–0.7 dex typical residual is an upper bound on true extraction error.")
    lines.append("- `is_new_limit`, `is_projection`, `data_source`, and `difficulty` are placeholder "
                 "labels in the repo-sourced pool and are not scored (shown as N/A / informational).")
    lines.append("")
    lines.append("### Interpolation metric (primary)")
    lines.append("1. Filter boundary-closure sentinel points (coupling >= 1e-2) from both extracted and GT data")
    lines.append("2. Build `scipy.interpolate.interp1d` from extracted points in log10(mass) → log10(coupling) space")
    lines.append("3. Evaluate the interpolation at each ground-truth mass value")
    lines.append("4. Compute residual = |log10(g_interpolated) - log10(g_ground_truth)| at each GT point")
    lines.append("5. Only GT points inside the extracted mass range are used (no extrapolation)")
    lines.append("")
    lines.append("**Key statistics:**")
    lines.append("- **Interpolation coverage**: fraction of GT points inside the extracted mass range")
    lines.append("- **Median/P90 residual**: summary of coupling errors in dex (0.3 dex ≈ factor 2)")
    lines.append("- **Fraction within threshold**: what % of GT points have residual below 0.1/0.3/0.5/1.0 dex")
    lines.append("")
    lines.append("When multiple extracted points share the same mass, the strongest constraint (lowest coupling) is kept.")
    lines.append("")
    lines.append("### Symmetric / 2-D metrics (complementary)")
    lines.append("- **Reverse pass**: the same interpolation, swapped — build the interp from the "
                 "GT points and evaluate at the *extracted* masses. The forward pass cannot see an "
                 "extraction that runs past the GT range; the reverse pass surfaces it as a low "
                 "reverse coverage / large reverse residual.")
    lines.append("- **Area-between-curves**: sample both log-log curves on a common grid over their "
                 "overlapping log-mass range, integrate |Δ log10 coupling| (trapezoid), normalise by "
                 "the overlap width → mean dex offset. Penalises both vertical offset and horizontal "
                 "(mass) shift.")
    lines.append("- **Mass-range Jaccard**: Jaccard index of the extracted vs GT log-mass intervals "
                 "(intersection / union). Penalises over- and under-claimed mass extent independently "
                 "of interpolation density.")
    lines.append("")
    lines.append("### Confidence calibration")
    lines.append('- A paper is "accurate" if median residual < 0.3 dex AND interpolation coverage ≥ 50%')
    lines.append("- Papers binned by extraction_confidence; actual accuracy computed per bin")
    lines.append("- Perfect calibration: actual accuracy = mean confidence in each bin")
    lines.append("")

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text)

    # Try to generate calibration plot
    try:
        _generate_calibration_plot(cal, output_path)
    except Exception as e:
        logger.warning("Could not generate calibration plot: %s", e)


def _generate_calibration_plot(calibration: list[dict], report_path: str):
    """Generate a calibration plot (confidence vs actual accuracy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    non_empty = [b for b in calibration if b["n_papers"] > 0]
    if len(non_empty) < 2:
        return

    x = [b["mean_confidence"] for b in non_empty]
    y = [b["actual_accuracy"] for b in non_empty]
    sizes = [b["n_papers"] * 50 for b in non_empty]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.scatter(x, y, s=sizes, alpha=0.7, zorder=5)
    for b in non_empty:
        ax.annotate(f"n={b['n_papers']}", (b["mean_confidence"], b["actual_accuracy"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Mean extraction confidence")
    ax.set_ylabel("Actual accuracy (coverage>80% & error<0.5 dex)")
    ax.set_title("Confidence Calibration")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = str(Path(report_path).with_suffix(".png"))
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration plot saved to %s", plot_path)
