"""Generate per-paper comparison plots: extracted vs ground-truth curves.

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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ground_truth import load_ground_truth

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
    """Plot ground truth vs extracted data for one paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Filter boundary closure points for cleaner visualization
    gt_mask = gt_data[:, 1] < 1e-2
    gt_plot = gt_data[gt_mask] if gt_mask.any() else gt_data

    ax.plot(
        gt_plot[:, 0], gt_plot[:, 1],
        "b-", linewidth=1.5, alpha=0.8, label="Ground truth (repo)", zorder=3,
    )
    ax.scatter(
        gt_plot[:, 0], gt_plot[:, 1],
        c="blue", s=8, alpha=0.5, zorder=4,
    )

    if ext_data is not None and len(ext_data) > 0:
        ext_mask = ext_data[:, 1] < 1e-2
        ext_plot = ext_data[ext_mask] if ext_mask.any() else ext_data

        ax.plot(
            ext_plot[:, 0], ext_plot[:, 1],
            "r--", linewidth=1.2, alpha=0.8, label="Extracted (pipeline)", zorder=5,
        )
        ax.scatter(
            ext_plot[:, 0], ext_plot[:, 1],
            c="red", s=12, alpha=0.6, marker="x", zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mass [eV]")
    ax.set_ylabel("Coupling")

    conf_str = f" (conf={extraction_confidence:.2f})" if extraction_confidence is not None else ""
    ax.set_title(f"arXiv:{arxiv_id} — {coupling_type}{conf_str}\n{title}", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

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
