"""Generate per-paper comparison plots: extracted vs ground-truth curves.

Shows ground truth, extracted points, interpolated curve from extracted
points, and a residual panel.

Usage:
    python -m evaluation.plot_comparison                # all papers with cached results
    python -m evaluation.plot_comparison --arxiv-id 2208.03183  # single paper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ground_truth import load_ground_truth
from evaluation.metrics import _filter_boundary

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"


def plot_comparison(
    arxiv_id: str,
    gt_data: np.ndarray,
    ext_data: np.ndarray | None,
    title: str,
    coupling_type: str,
    output_path: Path,
    extraction_confidence: float | None = None,
):
    """Plot ground truth vs extracted data with interpolation and residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_ext = ext_data is not None and len(ext_data) > 0

    if has_ext:
        fig, (ax_main, ax_resid) = plt.subplots(
            2, 1, figsize=(8, 8), height_ratios=[3, 1], sharex=True,
        )
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(8, 6))
        ax_resid = None

    # Filter boundary closure points
    gt_clean = _filter_boundary(gt_data)

    # Ground truth
    ax_main.scatter(
        gt_clean[:, 0], gt_clean[:, 1],
        c="blue", s=10, alpha=0.5, zorder=3, label="Ground truth",
    )

    if has_ext:
        ext_clean = _filter_boundary(ext_data)

        # Extracted points
        ax_main.scatter(
            ext_clean[:, 0], ext_clean[:, 1],
            c="red", s=20, alpha=0.6, marker="x", zorder=5,
            label=f"Extracted ({len(ext_clean)} pts)",
        )

        # Build interpolation in log-log space
        if len(ext_clean) >= 2:
            log_m = np.log10(ext_clean[:, 0])
            log_c = np.log10(ext_clean[:, 1])
            sort_idx = np.argsort(log_m)
            log_m, log_c = log_m[sort_idx], log_c[sort_idx]

            # Deduplicate: keep strongest constraint at each mass
            unique_m, inv = np.unique(log_m, return_inverse=True)
            min_c = np.full(len(unique_m), np.inf)
            for i, idx in enumerate(inv):
                if log_c[i] < min_c[idx]:
                    min_c[idx] = log_c[i]

            if len(unique_m) >= 2:
                interp_fn = interp1d(unique_m, min_c, kind="linear",
                                     bounds_error=False, fill_value=np.nan)

                # Dense curve for smooth visualization
                m_dense = np.linspace(unique_m.min(), unique_m.max(), 500)
                c_dense = interp_fn(m_dense)
                valid = ~np.isnan(c_dense)
                ax_main.plot(
                    10**m_dense[valid], 10**c_dense[valid],
                    "r-", linewidth=1.5, alpha=0.7, zorder=4,
                    label="Interpolated curve",
                )

                # Residuals at GT points
                log_gt_m = np.log10(gt_clean[:, 0])
                log_gt_c = np.log10(gt_clean[:, 1])
                interp_at_gt = interp_fn(log_gt_m)
                gt_valid = ~np.isnan(interp_at_gt)

                if gt_valid.any() and ax_resid is not None:
                    residuals = interp_at_gt[gt_valid] - log_gt_c[gt_valid]
                    gt_masses = gt_clean[gt_valid, 0]

                    ax_resid.scatter(gt_masses, residuals, c="black", s=6, alpha=0.5)
                    ax_resid.axhline(0, color="black", linewidth=0.5)
                    ax_resid.axhline(0.3, color="orange", linewidth=0.8, linestyle="--", alpha=0.6, label="±0.3 dex")
                    ax_resid.axhline(-0.3, color="orange", linewidth=0.8, linestyle="--", alpha=0.6)
                    ax_resid.axhline(0.5, color="red", linewidth=0.8, linestyle=":", alpha=0.4, label="±0.5 dex")
                    ax_resid.axhline(-0.5, color="red", linewidth=0.8, linestyle=":", alpha=0.4)

                    ax_resid.set_ylabel("Residual [dex]")
                    ax_resid.set_xlabel("Mass [eV]")
                    ax_resid.set_xscale("log")
                    y_lim = max(1.0, np.max(np.abs(residuals)) * 1.2)
                    ax_resid.set_ylim(-y_lim, y_lim)
                    ax_resid.legend(fontsize=7, loc="upper right")
                    ax_resid.grid(True, alpha=0.2)

                    med_r = np.median(np.abs(residuals))
                    n_within = np.sum(np.abs(residuals) <= 0.3)
                    ax_resid.set_title(
                        f"Residuals: median |r| = {med_r:.3f} dex, "
                        f"{n_within}/{len(residuals)} within 0.3 dex",
                        fontsize=9,
                    )

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    if ax_resid is None:
        ax_main.set_xlabel("Mass [eV]")
    ax_main.set_ylabel("Coupling")

    conf_str = f" (conf={extraction_confidence:.2f})" if extraction_confidence is not None else ""
    ax_main.set_title(f"arXiv:{arxiv_id} — {coupling_type}{conf_str}\n{title}", fontsize=10)
    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot: %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-id", type=str, default=None)
    args = parser.parse_args()

    entries = load_ground_truth()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        if args.arxiv_id and entry.arxiv_id != args.arxiv_id:
            continue

        gt_data = entry.load_data()
        if gt_data is None:
            gt_data = entry.load_reference_data(PROJECT_ROOT)
        if gt_data is None:
            logger.warning("No ground truth data for %s, skipping", entry.arxiv_id)
            continue

        # Load cached extraction result
        result_path = RESULTS_DIR / f"{entry.arxiv_id}.json"
        ext_data = None
        confidence = None
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            pts = result.get("data_points", [])
            if pts:
                ext_data = np.array(pts, dtype=float, ndmin=2)
            confidence = result.get("extraction_confidence")

        output = PLOTS_DIR / f"{entry.arxiv_id}.png"
        plot_comparison(
            arxiv_id=entry.arxiv_id,
            gt_data=gt_data,
            ext_data=ext_data,
            title=entry.paper_title[:80],
            coupling_type=entry.coupling_type,
            output_path=output,
            extraction_confidence=confidence,
        )


if __name__ == "__main__":
    main()
