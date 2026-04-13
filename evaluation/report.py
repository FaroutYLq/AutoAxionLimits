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
    agg_curve = metrics.get("curve_aggregate", {})

    lines.append("## Summary\n")
    lines.append(f"- **Papers evaluated**: {clf['coupling_type']['total']}")
    lines.append(f"- **Papers with curve comparison**: {agg_interp.get('n_papers', agg_curve.get('n_papers_with_curves', 0))}")
    lines.append("")

    # --- Classification ---
    lines.append("## Classification Accuracy\n")
    lines.append("| Field | Accuracy | N |")
    lines.append("|-------|----------|---|")
    for field_name in ["coupling_type", "is_new_limit", "is_projection", "data_source"]:
        entry = clf[field_name]
        lines.append(f"| {field_name} | {_pct(entry['accuracy'])} | {entry['total']} |")
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
        lines.append("## Extraction Quality — Interpolation Metric (primary)\n")
        lines.append("Build log-log interpolation from extracted points, evaluate at ground-truth masses.\n")
        lines.append(f"- **Mean interpolation coverage**: {_pct(agg_interp.get('mean_interpolation_coverage'))}")
        lines.append(f"- **Mean median residual**: {_fmt(agg_interp.get('mean_median_residual_dex'))} dex")
        lines.append(f"- **Mean P90 residual**: {_fmt(agg_interp.get('mean_p90_residual_dex'))} dex")
        lines.append(f"- **Mean fraction within 0.3 dex (factor 2)**: {_pct(agg_interp.get('mean_frac_within_0_3dex'))}")
        lines.append(f"- **Mean fraction within 0.5 dex (factor 3)**: {_pct(agg_interp.get('mean_frac_within_0_5dex'))}")
        lines.append("")

    # --- Per-paper ---
    per_paper = metrics.get("per_paper", [])
    if per_paper:
        lines.append("## Per-Paper Results\n")
        lines.append("| arXiv ID | Coupling | Conf. | Interp. Cov. | Med. Resid. | ≤0.3 dex | Points |")
        lines.append("|----------|----------|-------|--------------|-------------|----------|--------|")
        for p in per_paper:
            if p.get("status") != "extracted":
                lines.append(f"| {p['arxiv_id']} | — | — | FAILED | — | — | — |")
                continue
            coupling_ok = "✓" if p.get("coupling_type_correct") else f"✗ ({p.get('coupling_type_predicted', '?')})"
            conf = _fmt(p.get("extraction_confidence"), 2)
            im = p.get("interp_metrics")
            if im:
                cov = _pct(im["interpolation_coverage"])
                med = _fmt(im["median_residual_dex"])
                f03 = _pct(im["frac_within_0_3dex"])
                pts = f"{im['num_extracted']}/{im['num_ground_truth']}"
            else:
                cov = med = f03 = pts = "—"
            lines.append(f"| {p['arxiv_id']} | {coupling_ok} | {conf} | {cov} | {med} | {f03} | {pts} |")
        lines.append("")

    # --- Difficulty breakdown ---
    diff_bd = metrics.get("difficulty_breakdown", {})
    if diff_bd:
        lines.append("## Breakdown by Difficulty\n")
        lines.append("| Difficulty | Papers | Coupling Acc. | Med. Resid. | ≤0.3 dex |")
        lines.append("|------------|--------|---------------|-------------|----------|")
        for diff in ["easy", "medium", "hard"]:
            if diff not in diff_bd:
                continue
            d = diff_bd[diff]
            lines.append(
                f"| {diff} | {d['total']} | "
                f"{_pct(d['coupling_type_accuracy'])} | "
                f"{_fmt(d.get('mean_median_residual_dex'))} dex | "
                f"{_pct(d.get('mean_frac_within_0_3dex'))} |"
            )
        lines.append("")

    # --- Data source breakdown ---
    src_bd = metrics.get("source_breakdown", {})
    if src_bd:
        lines.append("## Breakdown by Extraction Source\n")
        lines.append("| Source | Papers | Med. Resid. | ≤0.3 dex |")
        lines.append("|--------|--------|-------------|----------|")
        for src in ["table", "figure_vision", "text"]:
            if src not in src_bd:
                continue
            s = src_bd[src]
            lines.append(
                f"| {src} | {s['total']} | "
                f"{_fmt(s.get('mean_median_residual_dex'))} dex | "
                f"{_pct(s.get('mean_frac_within_0_3dex'))} |"
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
