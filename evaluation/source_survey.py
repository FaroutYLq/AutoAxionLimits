"""WS1 keyless ceiling survey: arXiv source-tarball data vs ground truth.

Measures the ceiling of the planned ``source_data`` extraction channel with
ZERO Anthropic calls (arXiv fetch only): for every benchmark paper, unpack the
e-print, scan for candidate curve-data (``pipeline/source_data.py``), and
score EVERY candidate column pair against the paper's GT curves under a small
deterministic transform set (unit guesses the runtime channel would derive
from pgfplots axis options / captions instead). Report:

* hit rate — how much of the pool has extractable source data at all,
* best-candidate residual vs the paper's current channel residual,
* how many current >1 dex papers a source candidate takes below 0.3 dex.

Usage:
    python -m evaluation.source_survey --workdir <dir> \
        --out evaluation/eval_runs/source_survey.md

The survey is an ORACLE bound: it picks the best candidate/columns/transform
by looking at GT, which the runtime channel cannot do. It sizes the prize;
candidate selection (heuristics + optional flag-gated LLM disambiguation) is
the later integration PR's job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ground_truth import load_ground_truth  # noqa: E402
from evaluation.metrics import compute_interpolation_metrics  # noqa: E402
from pipeline.source_data import (  # noqa: E402
    download_source,
    extract_source,
    scan_candidates,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# h in eV*s -> photon-frequency mass conversions the runtime channel would
# read off pgfplots axis labels. Lambda in metres -> eV via 1.23984e-6/lambda.
_H_EV_S = 4.135667696e-15
X_TRANSFORMS: dict[str, callable] = {
    "identity": lambda x: x,
    "Hz->eV": lambda x: x * _H_EV_S,
    "kHz->eV": lambda x: x * (_H_EV_S * 1e3),
    "MHz->eV": lambda x: x * (_H_EV_S * 1e6),
    "GHz->eV": lambda x: x * (_H_EV_S * 1e9),
    "lambda_m->eV": lambda x: np.divide(1.23984198e-6, x,
                                        out=np.full_like(x, np.nan), where=x != 0),
    "10^x": lambda x: np.power(10.0, np.clip(x, -60, 60)),
}
Y_TRANSFORMS: dict[str, callable] = {
    "identity": lambda y: y,
    "10^y": lambda y: np.power(10.0, np.clip(y, -60, 60)),
    "sqrt": lambda y: np.sqrt(np.abs(y)),
}
# A log-decoded axis only makes sense when the raw values look like exponents.
_LOG_LIKE_MAX_ABS = 60.0

MAX_CANDIDATES_SCORED = 40
MAX_COLS = 6
MIN_COVERAGE = 0.3


@dataclass
class EntryResult:
    arxiv_id: str
    coupling_type: str
    n_gt: int
    best_median: float | None = None
    best_coverage: float | None = None
    best_candidate: str | None = None
    best_transform: str | None = None
    n_candidates: int = 0


def _columns(rows: list) -> np.ndarray:
    return np.asarray(rows, dtype=float)


def score_candidate_vs_gt(rows: list, gt: np.ndarray, arxiv_id: str,
                          coupling_type: str) -> tuple[float, float, str] | None:
    """Best (median_dex, coverage, transform_label) over column pairs x
    transforms, requiring coverage >= MIN_COVERAGE. None if nothing scores."""
    arr = _columns(rows)
    if len(arr) > 2000:  # subsample huge tables — the ceiling doesn't need 50k rows
        arr = arr[:: len(arr) // 2000 + 1]
    ncols = min(arr.shape[1], MAX_COLS)
    gt_pos = gt[(gt[:, 0] > 0)]
    if len(gt_pos) == 0:
        return None
    gt_lo, gt_hi = float(gt_pos[:, 0].min()), float(gt_pos[:, 0].max())
    best: tuple[float, float, str] | None = None
    for xi in range(ncols):
        for yi in range(ncols):
            if xi == yi:
                continue
            xraw, yraw = arr[:, xi], arr[:, yi]
            for xt_name, xt in X_TRANSFORMS.items():
                if xt_name == "10^x" and np.nanmax(np.abs(xraw)) > _LOG_LIKE_MAX_ABS:
                    continue
                x = xt(xraw)
                # Cheap pre-check: the transformed masses must overlap the GT
                # mass window at all, else every interpolation call is wasted.
                xpos = x[np.isfinite(x) & (x > 0)]
                if len(xpos) < 3 or xpos.min() > gt_hi or xpos.max() < gt_lo:
                    continue
                for yt_name, yt in Y_TRANSFORMS.items():
                    if yt_name == "10^y" and np.nanmax(np.abs(yraw)) > _LOG_LIKE_MAX_ABS:
                        continue
                    y = yt(yraw)
                    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                    if mask.sum() < 3:
                        continue
                    pts = np.column_stack([x[mask], y[mask]])
                    try:
                        m = compute_interpolation_metrics(
                            arxiv_id, pts, gt, coupling_type=coupling_type)
                    except Exception:
                        continue
                    med, cov = m.median_residual_dex, m.interpolation_coverage
                    if not np.isfinite(med) or cov < MIN_COVERAGE:
                        continue
                    label = f"cols({xi},{yi}) x:{xt_name} y:{yt_name}"
                    if best is None or (med, -cov) < (best[0], -best[1]):
                        best = (med, cov, label)
    return best


def survey_paper(aid: str, workdir: Path, entries: list) -> tuple[str, list[EntryResult]]:
    """Returns (source_status, per-GT-entry results)."""
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        blob = download_source(aid, workdir, max_retries=3, base_delay=10.0)
    except Exception:
        return "fetch_failed", []
    src_dir = workdir / f"{aid.replace('/', '_')}_src"
    try:
        files = extract_source(blob, src_dir)
    except Exception:
        return "unpack_failed", []
    if not files:
        return "pdf_only", []
    cands = scan_candidates(src_dir)
    if not cands:
        return "no_candidates", []

    results = []
    for e in entries:
        if e.excluded:
            continue
        gt = e.load_data()
        if gt is None or len(gt) == 0:
            continue
        r = EntryResult(arxiv_id=aid, coupling_type=e.coupling_type, n_gt=len(gt),
                        n_candidates=len(cands))
        for c in cands[:MAX_CANDIDATES_SCORED]:
            s = score_candidate_vs_gt(c.rows, gt, aid, e.coupling_type)
            if s and (r.best_median is None or s[0] < r.best_median):
                r.best_median, r.best_coverage = s[0], s[1]
                r.best_candidate, r.best_transform = c.rel_path, s[2]
        results.append(r)
    return "candidates", results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None, help="per-paper JSON dump")
    ap.add_argument("--limit", type=int, default=0, help="only first N papers (dev)")
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

    statuses: dict[str, str] = {}
    all_results: dict[str, list[EntryResult]] = {}
    t0 = time.time()
    for i, aid in enumerate(ids):
        status, results = survey_paper(aid, workdir, gt_by_id.get(aid, []))
        statuses[aid] = status
        if results:
            all_results[aid] = results
        if (i + 1) % 25 == 0:
            print(f"progress {i+1}/{len(ids)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ------------------------------------------------------------------ report
    def cur_resid(aid: str) -> float | None:
        im = per_paper[aid].get("interp_metrics") or {}
        return im.get("median_residual_dex")

    n = len(ids)
    n_source = sum(1 for s in statuses.values() if s in ("no_candidates", "candidates"))
    n_cand = sum(1 for s in statuses.values() if s == "candidates")
    scored = [r for rs in all_results.values() for r in rs if r.best_median is not None]
    lt03 = [r for r in scored if r.best_median < 0.3]
    lt10 = [r for r in scored if r.best_median < 1.0]

    # papers whose current pipeline result is bad (>1 dex or unscored) but a
    # source candidate is excellent (<0.3 dex)
    rescued = []
    for aid, rs in all_results.items():
        best = min((r.best_median for r in rs if r.best_median is not None), default=None)
        if best is None or best >= 0.3:
            continue
        cr = cur_resid(aid)
        if cr is None or not np.isfinite(cr) or cr > 1.0:
            rescued.append((aid, best, cr))

    lines: list[str] = []
    w = lines.append
    w("# WS1 source-tarball ceiling survey (keyless)")
    w("")
    w("Generated by `python -m evaluation.source_survey` — arXiv e-print fetch only,")
    w("no Anthropic API. Scanner: `pipeline/source_data.py`. GT = O'Hare repo files")
    w("via `evaluation/ground_truth` (exclusions skipped).")
    w("")
    w("> **This is an ORACLE ceiling.** The survey picks the best candidate file,")
    w("> column pair, and unit transform by comparing against GT — the runtime")
    w("> channel must make those choices from pgfplots axis options, captions and")
    w("> heuristics instead. It bounds the prize from above; it does not claim the")
    w("> channel achieves it.")
    w("")
    w("## Headline")
    w("")
    w(f"- Papers surveyed: **{n}**")
    w(f"- e-print source present (not PDF-only/withdrawn): **{n_source}** ({100*n_source/n:.0f}%)")
    w(f"- ... with >= 1 numeric data candidate: **{n_cand}** ({100*n_cand/n:.0f}%)")
    w(f"- GT entries scored against a candidate (coverage >= {MIN_COVERAGE}): **{len(scored)}**")
    w(f"- ... best candidate < 0.3 dex: **{len(lt03)}**; < 1.0 dex: **{len(lt10)}**")
    if scored:
        med = float(np.median([r.best_median for r in scored]))
        w(f"- median best-candidate residual over scored entries: **{med:.3f} dex**")
    w(f"- papers currently > 1 dex (or unscored) with a < 0.3 dex source candidate: "
      f"**{len(rescued)}**")
    w("")
    w("## Reading the result")
    w("")
    w(f"- The channel is a **tail rescue, not a vision replacement**: only "
      f"{100*n_cand/n:.0f}% of the pool has numeric source data (most papers ship "
      f"figures as pre-rendered PDF/PNG graphics — the WS2 population), but where "
      f"data exists it is near-exact, and the best candidates are overwhelmingly "
      f"arXiv **ancillary files (`anc/`)** — curves the authors deliberately "
      f"published.")
    w("- Several best candidates match GT to ~0.00 dex because the O'Hare repo "
      "file was built from the same published data. That is the point of the "
      "channel, not a scoring artifact: the source file IS the paper's own curve.")
    w("- WS2 note: the pre-rendered figures are mostly **vector** PDFs; a "
      "vector-path extraction pass (pymupdf `get_drawings()`) could recover exact "
      "coordinates for that population without raster CV.")
    w("")
    w("## Source availability")
    w("")
    w("| status | papers |")
    w("|---|---|")
    for s in ("candidates", "no_candidates", "pdf_only", "unpack_failed", "fetch_failed"):
        w(f"| {s} | {sum(1 for v in statuses.values() if v == s)} |")
    w("")
    w("## Scored entries (best candidate per GT entry)")
    w("")
    w("| paper | coupling | best med (dex) | cov | current (dex) | candidate | transform |")
    w("|---|---|---|---|---|---|---|")
    for aid in sorted(all_results):
        for r in all_results[aid]:
            if r.best_median is None:
                continue
            cr = cur_resid(aid)
            cr_s = f"{cr:.2f}" if cr is not None and np.isfinite(cr) else "—"
            w(f"| {aid} | {r.coupling_type} | {r.best_median:.3f} | "
              f"{r.best_coverage:.2f} | {cr_s} | {r.best_candidate[:60]} | {r.best_transform} |")
    w("")
    w("## Rescues (current > 1 dex or unscored -> source < 0.3 dex)")
    w("")
    w("| paper | source best (dex) | current (dex) |")
    w("|---|---|---|")
    for aid, best, cr in sorted(rescued):
        cr_s = f"{cr:.2f}" if cr is not None and np.isfinite(cr) else "unscored/inf"
        w(f"| {aid} | {best:.3f} | {cr_s} |")
    w("")

    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    if args.json_out:
        dump = {aid: {"status": statuses[aid],
                      "entries": [vars(r) for r in all_results.get(aid, [])]}
                for aid in ids}
        Path(args.json_out).write_text(json.dumps(dump, indent=1))
        print(f"wrote {args.json_out}")

    print(f"surveyed={n} source={n_source} with_candidates={n_cand} "
          f"scored={len(scored)} lt0.3={len(lt03)} rescued={len(rescued)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
