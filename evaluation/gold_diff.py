"""Gold-vs-repo-GT diff + extraction-vs-gold scoring (issue #537).

The evaluation's repo-sourced ground truth ``g(x_i)`` is cajohare's
ALREADY-PROCESSED curve (digitized + rescaled + convention-normalized from the
same source papers). A perfect extraction therefore inherits the upstream
digitization/convention gap. The hand-curated gold set
(``evaluation/ground_truth/gold/``) is digitized DIRECTLY from the source
papers, so:

  1. gold-vs-repo residual   == the UPSTREAM digitization/convention gap
                                 (how different cajohare's curve is from the
                                  paper's own numbers/figure).
  2. extraction-vs-gold      == the extractor's error against the paper itself.
  3. extraction-vs-repo      == the headline residual the main evaluation reports.

This is a STANDALONE script (no edits to evaluate.py needed for the science) so
it stays out of the way of the parallel work on evaluate.py/report.py. It reuses
``compute_interpolation_metrics`` and ``_filter_boundary`` from
``evaluation.metrics`` so gold/repo/extraction all see the same boundary
filtering and log-log interpolation.

Usage:
    # gold-vs-repo upstream gap + extraction-vs-gold vs extraction-vs-repo.
    python -m evaluation.gold_diff

    # also write a markdown report.
    python -m evaluation.gold_diff --report evaluation/gold_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.conventions import (  # noqa: E402
    infer_convention,
    infer_convention_for_repo_file,
)
from evaluation.metrics import (  # noqa: E402
    _COUPLING_CEILINGS,
    _filter_boundary,
    compute_interpolation_metrics,
)

logger = logging.getLogger(__name__)

GOLD_DIR = Path(__file__).parent / "ground_truth" / "gold"
GOLD_JSON = GOLD_DIR / "gold.json"
GOLD_DATA_DIR = GOLD_DIR / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _load_gold() -> list[dict]:
    with open(GOLD_JSON) as f:
        manifest = json.load(f)
    return manifest["gold_curves"]


def _load_gold_data(curve: dict) -> Optional[np.ndarray]:
    f = GOLD_DATA_DIR / curve["gold_data_file"]
    if not f.exists() or f.stat().st_size == 0:
        return None
    arr = np.loadtxt(str(f), ndmin=2)
    return arr if arr.size and arr.ndim == 2 and arr.shape[1] >= 2 else None


def _load_repo(curve: dict) -> Optional[np.ndarray]:
    ref = curve.get("reference_repo_file")
    if not ref:
        return None
    p = PROJECT_ROOT / ref
    if not p.exists():
        logger.warning("repo file missing: %s", p)
        return None
    try:
        arr = np.loadtxt(str(p), ndmin=2)
    except Exception:
        return None
    return arr if arr.size and arr.ndim == 2 and arr.shape[1] >= 2 else None


def _load_extraction(arxiv_id: str) -> Optional[np.ndarray]:
    p = RESULTS_DIR / f"{_safe_id(arxiv_id)}.json"
    if not p.exists():
        return None
    with open(p) as f:
        r = json.load(f)
    if "error" in r:
        return None
    pts = r.get("data_points") or []
    if not pts:
        return None
    try:
        return np.array(pts, dtype=float, ndmin=2)
    except Exception:
        return None


def _usable(arr: Optional[np.ndarray], coupling_type: str) -> bool:
    """A curve must survive boundary filtering with >= 2 distinct masses to be
    interpolatable (same gate compute_interpolation_metrics applies)."""
    if arr is None:
        return False
    ceil = _COUPLING_CEILINGS.get(coupling_type, 1e-2)
    f = _filter_boundary(arr, ceil)
    return len(f) >= 2 and int(np.unique(f[:, 0]).size) >= 2


def _median_residual(src: np.ndarray, tgt: np.ndarray, coupling_type: str
                     ) -> Optional[float]:
    """Median |Δlog10 coupling| of ``tgt`` interpolated onto ``src``'s masses,
    via compute_interpolation_metrics (``src`` plays the role of "extracted",
    ``tgt`` of "ground truth"). Returns None if no mass overlap."""
    im = compute_interpolation_metrics(
        "diff", src, tgt, coupling_type=coupling_type,
    )
    if im.median_residual_dex == float("inf"):
        return None
    return im.median_residual_dex


def compute_gold_diff() -> dict:
    """For every gold curve: gold-vs-repo, extraction-vs-gold, extraction-vs-repo
    median residuals (dex), kept SEPARATE by digitization independence."""
    gold = _load_gold()
    rows: list[dict] = []

    for c in gold:
        aid = c["arxiv_id"]
        ct = c["coupling_type"]
        digitized_by = c.get("digitized_by")
        ref = c.get("reference_repo_file")

        gold_arr = _load_gold_data(c)
        repo_arr = _load_repo(c)
        ext_arr = _load_extraction(aid)

        # Conventions (the #538 trap): the gold curve carries its own (inferred)
        # convention; the repo file's convention is inferred from its value
        # range; the extraction's expected convention is the CANONICAL one for
        # the coupling. A residual across two different conventions is a UNITS
        # gap, not a digitization/extraction error, so we flag and exclude it
        # from the headline rather than letting a 20-dex outlier dominate.
        gold_conv = c.get("coupling_convention")
        if gold_conv is None and gold_arr is not None:
            gold_conv, _ = infer_convention(
                ct, GOLD_DATA_DIR / c["gold_data_file"])
        repo_path = (PROJECT_ROOT / ref) if ref else None
        repo_conv, _ = infer_convention_for_repo_file(
            ref, ct, repo_path if (repo_path and repo_path.exists()) else None)
        canon_conv, _ = infer_convention(ct, None)  # extraction expected conv

        conv_gold_repo_ok = (gold_conv is None or repo_conv is None
                             or gold_conv == repo_conv)
        conv_gold_ext_ok = (gold_conv is None or canon_conv is None
                            or gold_conv == canon_conv)
        conv_repo_ext_ok = (repo_conv is None or canon_conv is None
                            or repo_conv == canon_conv)

        row: dict = {
            "arxiv_id": aid,
            "coupling_type": ct,
            "digitized_by": digitized_by,
            "independence": c.get("independence"),
            "source_kind": c.get("source_kind"),
            "reference_repo_file": ref,
            "gold_num_points": c.get("num_points", 0),
            "gold_convention": gold_conv,
            "repo_convention": repo_conv,
            "gold_usable": _usable(gold_arr, ct),
            "repo_usable": _usable(repo_arr, ct),
            "ext_usable": _usable(ext_arr, ct),
            "conv_gold_repo_ok": conv_gold_repo_ok,
            "conv_gold_ext_ok": conv_gold_ext_ok,
            "conv_repo_ext_ok": conv_repo_ext_ok,
        }

        # gold-vs-repo (the upstream gap). Interpolate the repo curve onto the
        # gold masses (gold is the reference here). Only when conventions agree.
        if row["gold_usable"] and row["repo_usable"] and conv_gold_repo_ok:
            row["gold_vs_repo_dex"] = _median_residual(gold_arr, repo_arr, ct)
        else:
            row["gold_vs_repo_dex"] = None
            if not conv_gold_repo_ok:
                row["gold_vs_repo_excluded"] = "convention_mismatch"

        # extraction-vs-gold (true extraction error against the paper).
        if row["gold_usable"] and row["ext_usable"] and conv_gold_ext_ok:
            row["ext_vs_gold_dex"] = _median_residual(ext_arr, gold_arr, ct)
        else:
            row["ext_vs_gold_dex"] = None
            if not conv_gold_ext_ok:
                row["ext_vs_gold_excluded"] = "convention_mismatch"

        # extraction-vs-repo (the headline residual, for the SAME papers).
        if row["repo_usable"] and row["ext_usable"] and conv_repo_ext_ok:
            row["ext_vs_repo_dex"] = _median_residual(ext_arr, repo_arr, ct)
        else:
            row["ext_vs_repo_dex"] = None
            if not conv_repo_ext_ok:
                row["ext_vs_repo_excluded"] = "convention_mismatch"

        rows.append(row)

    # A residual > this many dex is implausible as a digitization difference on
    # the SAME physical curve, so it is almost certainly a units/convention gap
    # the (coarse) convention inference could not resolve — e.g. MonopoleDipole
    # g_s*g_p vs g_p, or an f_a-plane axis ambiguity. We report BOTH the raw
    # median (no filter) and the plausible-only median (these flagged), so the
    # headline reflects the digitization-scale difference, not unit gaps.
    PLAUSIBLE_DEX = 3.0

    def _med(key: str, subset: list[dict], plausible: bool = False
             ) -> tuple[Optional[float], int]:
        vals = [r[key] for r in subset if r.get(key) is not None]
        if plausible:
            vals = [v for v in vals if v <= PLAUSIBLE_DEX]
        return (float(np.median(vals)) if vals else None, len(vals))

    # Mark the implausible (likely-units-gap) pairs for the per-curve report.
    for r in rows:
        for key in ("gold_vs_repo_dex", "ext_vs_gold_dex", "ext_vs_repo_dex"):
            v = r.get(key)
            if isinstance(v, (int, float)) and v > PLAUSIBLE_DEX:
                r[key.replace("_dex", "_likely_units_gap")] = True

    table_rows = [r for r in rows if r["digitized_by"] == "gold_table"]
    vision_rows = [r for r in rows if r["digitized_by"] == "gold_vision"]

    gvr_all, n_gvr = _med("gold_vs_repo_dex", rows)
    gvr_pl, n_gvr_pl = _med("gold_vs_repo_dex", rows, plausible=True)
    gvr_tab, n_gvr_tab = _med("gold_vs_repo_dex", table_rows, plausible=True)
    gvr_vis, n_gvr_vis = _med("gold_vs_repo_dex", vision_rows, plausible=True)
    evg_all, n_evg = _med("ext_vs_gold_dex", rows, plausible=True)
    evr_all, n_evr = _med("ext_vs_repo_dex", rows, plausible=True)

    # Paired comparison: extraction-vs-gold vs extraction-vs-repo on the SAME
    # papers (only papers where BOTH are defined and plausible), so the two
    # numbers are apples-to-apples.
    paired = [r for r in rows
              if r.get("ext_vs_gold_dex") is not None
              and r.get("ext_vs_repo_dex") is not None
              and r["ext_vs_gold_dex"] <= PLAUSIBLE_DEX
              and r["ext_vs_repo_dex"] <= PLAUSIBLE_DEX]
    evg_paired = (float(np.median([r["ext_vs_gold_dex"] for r in paired]))
                  if paired else None)
    evr_paired = (float(np.median([r["ext_vs_repo_dex"] for r in paired]))
                  if paired else None)

    return {
        "n_gold_curves": len(rows),
        "plausible_dex_cutoff": PLAUSIBLE_DEX,
        "summary": {
            # KEY SCIENCE NUMBER: upstream digitization gap (gold vs repo-GT),
            # restricted to plausible (<=3 dex) same-convention pairs.
            "gold_vs_repo_median_dex": gvr_pl,
            "gold_vs_repo_n": n_gvr_pl,
            "gold_vs_repo_median_dex_raw": gvr_all,
            "gold_vs_repo_n_raw": n_gvr,
            "gold_vs_repo_median_dex_table": gvr_tab,
            "gold_vs_repo_n_table": n_gvr_tab,
            "gold_vs_repo_median_dex_vision": gvr_vis,
            "gold_vs_repo_n_vision": n_gvr_vis,
            "ext_vs_gold_median_dex": evg_all,
            "ext_vs_gold_n": n_evg,
            "ext_vs_repo_median_dex": evr_all,
            "ext_vs_repo_n": n_evr,
            # Paired (same-paper) extraction comparison.
            "paired_n": len(paired),
            "ext_vs_gold_median_dex_paired": evg_paired,
            "ext_vs_repo_median_dex_paired": evr_paired,
        },
        "per_curve": rows,
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def render_report(diff: dict) -> str:
    s = diff["summary"]
    lines = [
        "# Gold-set diff report",
        "",
        "Hand-curated gold curves digitized DIRECTLY from source papers vs "
        "cajohare's repo curves and the cached pipeline extractions.",
        "",
        f"- Gold curves: **{diff['n_gold_curves']}**",
        "",
        "## Upstream digitization gap (gold vs repo-GT) — the key number",
        "",
        f"Pairs with a residual > {diff['plausible_dex_cutoff']:.0f} dex are "
        f"flagged as likely units/convention gaps the convention inference could "
        f"not resolve and excluded from the headline (raw shown for reference).",
        "",
        f"- **median gold-vs-repo = {_fmt(s['gold_vs_repo_median_dex'])} dex** "
        f"(N={s['gold_vs_repo_n']})",
        f"  - raw (no plausibility filter): "
        f"{_fmt(s['gold_vs_repo_median_dex_raw'])} dex "
        f"(N={s['gold_vs_repo_n_raw']})",
        f"  - table/text-digitized (independent): "
        f"{_fmt(s['gold_vs_repo_median_dex_table'])} dex "
        f"(N={s['gold_vs_repo_n_table']})",
        f"  - figure-digitized (semi-independent, vision): "
        f"{_fmt(s['gold_vs_repo_median_dex_vision'])} dex "
        f"(N={s['gold_vs_repo_n_vision']})",
        "",
        "## Extraction error: vs gold vs vs repo (same papers)",
        "",
        f"- extraction-vs-gold (all):  {_fmt(s['ext_vs_gold_median_dex'])} dex "
        f"(N={s['ext_vs_gold_n']})",
        f"- extraction-vs-repo (all):  {_fmt(s['ext_vs_repo_median_dex'])} dex "
        f"(N={s['ext_vs_repo_n']})",
        f"- paired (N={s['paired_n']}): "
        f"vs-gold {_fmt(s['ext_vs_gold_median_dex_paired'])} dex, "
        f"vs-repo {_fmt(s['ext_vs_repo_median_dex_paired'])} dex",
        "",
        "## Per-curve",
        "",
        "| arXiv | coupling | digitized_by | gold↔repo | ext↔gold | ext↔repo |",
        "|---|---|---|---|---|---|",
    ]
    for r in diff["per_curve"]:
        lines.append(
            f"| {r['arxiv_id']} | {r['coupling_type']} | {r['digitized_by']} | "
            f"{_fmt(r.get('gold_vs_repo_dex'))} | "
            f"{_fmt(r.get('ext_vs_gold_dex'))} | "
            f"{_fmt(r.get('ext_vs_repo_dex'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Gold-vs-repo diff + extraction scoring")
    ap.add_argument("--report", default=None, help="Write a markdown report here")
    ap.add_argument("--json", default=None, help="Write the raw diff JSON here")
    args = ap.parse_args()

    diff = compute_gold_diff()
    s = diff["summary"]

    print("=== Gold-set diff ===")
    print(f"gold curves: {diff['n_gold_curves']} "
          f"(plausibility cutoff {diff['plausible_dex_cutoff']:.0f} dex)")
    print(f"gold-vs-repo (upstream gap): {_fmt(s['gold_vs_repo_median_dex'])} dex "
          f"(N={s['gold_vs_repo_n']}); raw {_fmt(s['gold_vs_repo_median_dex_raw'])} "
          f"(N={s['gold_vs_repo_n_raw']})")
    print(f"  table/text: {_fmt(s['gold_vs_repo_median_dex_table'])} "
          f"(N={s['gold_vs_repo_n_table']}); "
          f"vision: {_fmt(s['gold_vs_repo_median_dex_vision'])} "
          f"(N={s['gold_vs_repo_n_vision']})")
    print(f"extraction-vs-gold: {_fmt(s['ext_vs_gold_median_dex'])} dex "
          f"(N={s['ext_vs_gold_n']})")
    print(f"extraction-vs-repo: {_fmt(s['ext_vs_repo_median_dex'])} dex "
          f"(N={s['ext_vs_repo_n']})")
    print(f"paired (N={s['paired_n']}): vs-gold "
          f"{_fmt(s['ext_vs_gold_median_dex_paired'])}, vs-repo "
          f"{_fmt(s['ext_vs_repo_median_dex_paired'])}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(diff, f, indent=2, default=str)
        logger.info("Wrote diff JSON to %s", args.json)
    if args.report:
        Path(args.report).write_text(render_report(diff))
        logger.info("Wrote report to %s", args.report)


if __name__ == "__main__":
    main()
