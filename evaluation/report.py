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
    agg = metrics["curve_aggregate"]

    lines.append("## Summary\n")
    lines.append(f"- **Papers evaluated**: {clf['coupling_type']['total']}")
    lines.append(f"- **Papers with curve comparison**: {agg.get('n_papers_with_curves', 0)}")
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

    # --- Curve Quality ---
    if agg.get("n_papers_with_curves", 0) > 0:
        lines.append("## Curve Quality (aggregate)\n")
        lines.append(f"- **Mean coverage (0.5 dex)**: {_pct(agg.get('mean_coverage_0_5dex'))}")
        lines.append(f"- **Mean coverage (1.0 dex)**: {_pct(agg.get('mean_coverage_1_0dex'))}")
        lines.append(f"- **Mean median coupling error**: {_fmt(agg.get('mean_median_coupling_log_error'))} dex")
        lines.append(f"- **Mean mass range overlap**: {_pct(agg.get('mean_mass_range_overlap'))}")
        lines.append("")

    # --- Per-paper ---
    per_paper = metrics.get("per_paper", [])
    if per_paper:
        lines.append("## Per-Paper Results\n")
        lines.append("| arXiv ID | Status | Coupling | Conf. | Coverage (1 dex) | Med. Error | Points |")
        lines.append("|----------|--------|----------|-------|------------------|------------|--------|")
        for p in per_paper:
            status = p.get("status", "?")
            coupling_ok = "✓" if p.get("coupling_type_correct") else f"✗ ({p.get('coupling_type_predicted', '?')})"
            conf = _fmt(p.get("extraction_confidence"), 2)
            cm = p.get("curve_metrics")
            if cm:
                cov = _pct(cm["coverage_at_1_0dex"])
                err = _fmt(cm["median_coupling_log_error"])
                pts = f"{cm['num_extracted']}/{cm['num_ground_truth']}"
            else:
                cov = err = pts = "—"
            lines.append(f"| {p['arxiv_id']} | {status} | {coupling_ok} | {conf} | {cov} | {err} | {pts} |")
        lines.append("")

    # --- Difficulty breakdown ---
    diff_bd = metrics.get("difficulty_breakdown", {})
    if diff_bd:
        lines.append("## Breakdown by Difficulty\n")
        lines.append("| Difficulty | Papers | Extracted | Coupling Acc. | Coverage (1 dex) |")
        lines.append("|------------|--------|-----------|---------------|------------------|")
        for diff in ["easy", "medium", "hard"]:
            if diff not in diff_bd:
                continue
            d = diff_bd[diff]
            lines.append(
                f"| {diff} | {d['total']} | {d['extracted']} | "
                f"{_pct(d['coupling_type_accuracy'])} | "
                f"{_pct(d.get('mean_coverage_1_0dex'))} |"
            )
        lines.append("")

    # --- Data source breakdown ---
    src_bd = metrics.get("source_breakdown", {})
    if src_bd:
        lines.append("## Breakdown by Extraction Source\n")
        lines.append("| Source | Papers | Coverage (1 dex) | Med. Error |")
        lines.append("|--------|--------|------------------|------------|")
        for src in ["table", "figure_vision", "text"]:
            if src not in src_bd:
                continue
            s = src_bd[src]
            lines.append(
                f"| {src} | {s['total']} | "
                f"{_pct(s.get('mean_coverage_1_0dex'))} | "
                f"{_fmt(s.get('mean_median_coupling_error'))} dex |"
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
    lines.append("### Curve comparison")
    lines.append("- All comparisons done in log10 space (mass, coupling)")
    lines.append("- Boundary closure points (coupling ≥ 1e-2) filtered before comparison")
    lines.append("- **Coverage**: fraction of ground-truth points with an extracted point within N dex")
    lines.append("- **Median coupling error**: for mass-matched points (within 0.1 dex), absolute log10 error on coupling")
    lines.append("- **Mass range overlap**: fraction of ground-truth mass range covered by extracted data")
    lines.append("- **Hausdorff distance**: max of (max min-distance from ext→GT, max min-distance from GT→ext)")
    lines.append("")
    lines.append("### Confidence calibration")
    lines.append('- A paper is "accurate" if coverage(1 dex) ≥ 80% AND median coupling error < 0.5 dex')
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
