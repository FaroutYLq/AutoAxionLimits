"""WS2 keyless ceiling survey: vector-path figure extraction vs ground truth.

For every benchmark paper with a cached e-print source tree, open each figure
PDF, attempt text-layer axis calibration (``pipeline/vector_trace.py``), map
every colour-grouped vector curve into data space, and score EVERY candidate
against the paper's GT curves (oracle selection — this bounds the channel's
ceiling; runtime selection is a later, cheap LLM/heuristic step).

No Anthropic calls, no network (uses the ``AAL_SOURCE_CACHE`` e-prints).

Usage:
    python -m evaluation.vector_ceiling --workdir <survey dir> \
        --out evaluation/eval_runs/vector_ceiling.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ground_truth import load_ground_truth  # noqa: E402
from evaluation.metrics import compute_interpolation_metrics  # noqa: E402
from pipeline.source_data import download_source, extract_source  # noqa: E402
from pipeline.vector_trace import curve_to_data, trace_figure_pdf  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
MAX_FIGURES_PER_PAPER = 20
MAX_CURVES_SCORED = 10
MIN_COVERAGE = 0.3


def survey_paper(aid: str, workdir: Path, entries: list) -> dict:
    out = {"status": "no_source", "n_figs": 0, "n_calibrated": 0,
           "n_outlined": 0, "entries": []}
    src_dir = workdir / f"{aid.replace('/', '_')}_src"
    if not src_dir.is_dir():
        try:
            blob = download_source(aid, workdir, max_retries=2)
            extract_source(blob, src_dir)
        except Exception:
            return out
    figs = sorted(src_dir.rglob("*.pdf"))[:MAX_FIGURES_PER_PAPER]
    out["n_figs"] = len(figs)
    if not figs:
        out["status"] = "no_figure_pdfs"
        return out

    traced = []  # (fig_name, cal, curves)
    for f in figs:
        cal, curves = trace_figure_pdf(f)
        if cal is None:
            out["n_outlined"] += 1
            continue
        out["n_calibrated"] += 1
        if curves:
            traced.append((f.name, cal, curves))
    if not traced:
        out["status"] = "no_calibrated_figures"
        return out
    out["status"] = "traced"

    for e in entries:
        if e.excluded:
            continue
        gt = e.load_data()
        if gt is None or not len(gt):
            continue
        best = None
        for fig_name, cal, curves in traced:
            for c in curves[:MAX_CURVES_SCORED]:
                pts = curve_to_data(c, cal)
                if len(pts) < 3:
                    continue
                try:
                    m = compute_interpolation_metrics(
                        aid, np.array(pts), gt, coupling_type=e.coupling_type)
                except Exception:
                    continue
                r, cov = m.median_residual_dex, m.interpolation_coverage
                if not np.isfinite(r) or cov < MIN_COVERAGE:
                    continue
                if best is None or (r, -cov) < (best[0], -best[1]):
                    best = (r, cov, fig_name, str(c.color), c.n_points)
        out["entries"].append({
            "coupling_type": e.coupling_type,
            "best_median": best[0] if best else None,
            "best_cov": best[1] if best else None,
            "best_fig": best[2] if best else None,
            "best_color": best[3] if best else None,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    per_paper = {p["arxiv_id"]: p for p in metrics["per_paper"]}
    gt_by_id: dict[str, list] = {}
    for e in load_ground_truth():
        gt_by_id.setdefault(e.arxiv_id, []).append(e)

    ids = sorted(per_paper)
    if args.limit:
        ids = ids[: args.limit]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    t0 = time.time()
    for i, aid in enumerate(ids):
        results[aid] = survey_paper(aid, workdir, gt_by_id.get(aid, []))
        if (i + 1) % 50 == 0:
            print(f"progress {i+1}/{len(ids)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    def cur_resid(aid):
        im = per_paper[aid].get("interp_metrics") or {}
        return im.get("median_residual_dex")

    n = len(ids)
    n_any_fig = sum(1 for r in results.values() if r["n_figs"] > 0)
    n_cal = sum(1 for r in results.values() if r["n_calibrated"] > 0)
    scored = [(aid, e) for aid, r in results.items() for e in r["entries"]
              if e["best_median"] is not None]
    lt03 = [(a, e) for a, e in scored if e["best_median"] < 0.3]
    lt10 = [(a, e) for a, e in scored if e["best_median"] < 1.0]
    vision_ids = {aid for aid, p in per_paper.items()
                  if p.get("data_source") in ("figure_vision", "vision")}
    rescued = []
    for aid, e in scored:
        if e["best_median"] >= 0.3:
            continue
        cr = cur_resid(aid)
        if cr is None or not np.isfinite(cr) or cr > 1.0:
            rescued.append((aid, e))

    lines = []
    w = lines.append
    w("# WS2 vector-path ceiling survey (keyless, oracle selection)")
    w("")
    w("Generated by `python -m evaluation.vector_ceiling` — no API, cached e-prints.")
    w("Tracer: `pipeline/vector_trace.py` (text-layer tick calibration + colour-")
    w("grouped vector curves; outlined-text figures are counted as the raster/OCR")
    w("fallback residue, not traced here).")
    w("")
    w("> **Oracle ceiling**: best figure/curve picked BY COMPARING TO GT. Runtime")
    w("> selection (label heuristics + cheap LLM pick) is the follow-up; this")
    w("> bounds what that selection can achieve.")
    w("")
    w("## Headline")
    w("")
    w(f"- Papers surveyed: **{n}**; with figure PDFs in source: **{n_any_fig}**")
    w(f"- Papers with >= 1 text-calibratable figure: **{n_cal}** ({100*n_cal/n:.0f}%)")
    w(f"- GT entries scored (coverage >= {MIN_COVERAGE}): **{len(scored)}**")
    w(f"- best curve < 0.3 dex: **{len(lt03)}**; < 1.0 dex: **{len(lt10)}**")
    if scored:
        med = float(np.median([e["best_median"] for _, e in scored]))
        w(f"- median best-candidate residual: **{med:.3f} dex**")
    w(f"- papers currently > 1 dex (or unscored) with a < 0.3 dex vector curve: "
      f"**{len(rescued)}**")
    w(f"- current figure_vision papers among the scored: "
      f"{sum(1 for a, _ in scored if a in vision_ids)}")
    w("")
    w("## Rescues (current > 1 dex or unscored -> vector < 0.3 dex)")
    w("")
    w("| paper | entry | vector best (dex) | cov | figure | current (dex) |")
    w("|---|---|---|---|---|---|")
    for aid, e in sorted(rescued, key=lambda t: (t[0], t[1]['coupling_type'])):
        cr = cur_resid(aid)
        cr_s = f"{cr:.2f}" if cr is not None and np.isfinite(cr) else "unscored/inf"
        w(f"| {aid} | {e['coupling_type']} | {e['best_median']:.3f} | "
          f"{e['best_cov']:.2f} | {e['best_fig']} | {cr_s} |")
    w("")
    w("## All scored entries")
    w("")
    w("| paper | entry | best (dex) | cov | figure | colour | current (dex) |")
    w("|---|---|---|---|---|---|---|")
    for aid, e in sorted(scored, key=lambda t: (t[1]['best_median'], t[0])):
        cr = cur_resid(aid)
        cr_s = f"{cr:.2f}" if cr is not None and np.isfinite(cr) else "—"
        w(f"| {aid} | {e['coupling_type']} | {e['best_median']:.3f} | "
          f"{e['best_cov']:.2f} | {e['best_fig']} | {e['best_color']} | {cr_s} |")
    w("")
    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    print(f"surveyed={n} calibratable={n_cal} scored={len(scored)} "
          f"lt0.3={len(lt03)} lt1.0={len(lt10)} rescued={len(rescued)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
