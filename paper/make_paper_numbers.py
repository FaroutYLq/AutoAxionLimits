#!/usr/bin/env python3
"""Pin every number cited in the JINST manuscript to the benchmark artifacts.

Single source of truth for the manuscript: every quantitative claim in
paper/jinst_autoaxionlimits.tex must appear in this script's output
(paper/numbers.json + paper/numbers.md). Definitions are pinned HERE so a
referee can reproduce each value from the committed metrics files.

Benchmark scope (2026-07-04): MEASURED LIMITS ONLY — GT entries flagged
is_projection are excluded before scoring (see PR #698). Projections embed
detector-scenario assumptions that make a single GT curve ill-defined and
are of less community interest; the *_noproj.json artifacts are the citable
metrics.

Inputs (committed):
  evaluation/eval_runs/final2_opus_n1/metrics_noproj.json   (definitive Opus arm)
  evaluation/eval_runs/final2_haiku_n1/metrics_noproj.json  (definitive Haiku arm)
  evaluation/eval_runs/final2_*/<id>.json                   (extraction snapshots)
  evaluation/eval_runs/baseline_metrics_noproj.json         (old-code N=3 baseline, rescored)
  evaluation/ground_truth/papers.json                       (GT pool)

Pinned definitions:
  compared paper   comparison_status == "compared" AND finite forward
                   interpolation median residual (== interpolation_aggregate
                   .n_finite).
  residual         per-paper median |Δlog10 g| by forward interpolation over
                   shared mass support (interp_metrics.median_residual_dex).
  catastrophic     residual > 3 dex among compared papers.
  >1 dex           residual > 1 dex among compared papers.
  paired set       papers compared (finite) in BOTH arms; delta = Haiku − Opus
                   per-paper residual; better/worse threshold 0.05 dex.
  accuracy (calib) residual < 0.32 dex AND interpolation coverage >= 50%
                   (the scorer's confidence_calibration definition).
  macro statistic  the scorer's "macro_median_residual_dex" key is the MEAN
                   of per-type medians (np.mean in evaluation/evaluate.py),
                   not a median of medians; the key name is historical. Cite
                   it as a macro average / mean of per-type medians.

Run from repo root:  python paper/make_paper_numbers.py
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ARMS = {
    "opus": "evaluation/eval_runs/final2_opus_n1",
    "haiku": "evaluation/eval_runs/final2_haiku_n1",
}
OLD_BASELINE = "evaluation/eval_runs/baseline_metrics_noproj.json"
GT_POOL = "evaluation/ground_truth/papers.json"

GUARD_MARKERS = {
    "corroboration_rejections": "TEXT-VISION DISAGREEMENT",
    "convention_review_flags": "CONVENTION REVIEW",
}


def load(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return json.load(f)


def finite_residuals(metrics):
    """arxiv_id -> forward-interp median residual, compared+finite papers only."""
    out = {}
    for pp in metrics["per_paper"]:
        if pp.get("comparison_status") != "compared":
            continue
        med = (pp.get("interp_metrics") or {}).get("median_residual_dex")
        if med is not None and math.isfinite(med):
            out[pp["arxiv_id"]] = med
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def arm_summary(arm_dir):
    m = load(os.path.join(arm_dir, "metrics_noproj.json"))
    pta = m["per_type_aggregate"]
    ia = m["interpolation_aggregate"]
    resid = finite_residuals(m)

    # channel mix: winning data_source across ALL per-paper records (346)
    mix = {}
    for pp in m["per_paper"]:
        src = pp.get("data_source") or "none"
        mix[src] = mix.get(src, 0) + 1

    # per-channel residuals over compared+finite papers, from per_paper directly
    # (the scorer's source_breakdown omits vector_trace/source_data rows)
    by_src = {}
    for pp in m["per_paper"]:
        r = resid.get(pp["arxiv_id"])
        if r is None:
            continue
        by_src.setdefault(pp.get("data_source") or "none", []).append(r)
    per_channel = {
        src: {"n_compared": len(rs), "median_residual_dex": round(median(rs), 4)}
        for src, rs in sorted(by_src.items())
    }

    # guard firings recomputed from the snapshot notes, restricted to the
    # papers in this metrics pool (i.e. excluding projection papers)
    pool_ids = {pp["arxiv_id"] for pp in m["per_paper"]}
    guards = {}
    for key, marker in GUARD_MARKERS.items():
        n = 0
        for path in glob.glob(os.path.join(ROOT, arm_dir, "*.json")):
            base = os.path.basename(path)
            if base.startswith("metrics"):
                continue
            pid = base[:-5].replace("_", "/", 1) if base[0].isalpha() else base[:-5]
            if pid not in pool_ids:
                continue
            with open(path) as f:
                if marker in f.read():
                    n += 1
        guards[key] = n

    # anatomy quantities cited in the manuscript: residual-mass share and
    # catastrophic counts by channel (over compared+finite papers)
    tot_mass = sum(resid.values())
    vis_mass = sum(r for pp in m["per_paper"]
                   if (r := resid.get(pp["arxiv_id"])) is not None
                   and pp.get("data_source") == "figure_vision")
    cat_by_src = {}
    for pp in m["per_paper"]:
        r = resid.get(pp["arxiv_id"])
        if r is not None and r > 3:
            src = pp.get("data_source") or "none"
            cat_by_src[src] = cat_by_src.get(src, 0) + 1

    calib = [
        {
            "bin": f"[{b['bin_lo']:.1f}-{b['bin_hi']:.1f})",
            "n_papers": b["n_papers"],
            "mean_confidence": round(b["mean_confidence"], 3),
            "actual_accuracy": round(b["actual_accuracy"], 3),
        }
        for b in m["confidence_calibration"]
        if b["n_papers"] > 0
    ]

    cls = m.get("classification") or {}
    classification = {
        k: {"accuracy": round(v["accuracy"], 4), "n": v["total"]}
        for k, v in cls.items()
        if isinstance(v, dict) and "accuracy" in v
    }

    return {
        "n_papers": m["n_papers"],
        "micro_median_residual_dex": round(pta["micro_median_residual_dex"], 4),
        "macro_median_residual_dex": round(pta["macro_median_residual_dex"], 4),
        "n_compared_status": m["comparison_coverage"]["n_compared"],
        "n_zero_overlap": ia["n_zero_overlap"],
        "n_compared_finite": ia["n_finite"],
        "catastrophic_gt3dex": sum(1 for r in resid.values() if r > 3),
        "gt1dex": sum(1 for r in resid.values() if r > 1),
        "frac_within_0_3dex_mean": round(ia["mean_frac_within_0_3dex"], 4),
        "frac_within_0_5dex_mean": round(ia["mean_frac_within_0_5dex"], 4),
        "mean_interpolation_coverage": round(ia["mean_interpolation_coverage"], 4),
        "channel_mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
        "per_channel_residuals": per_channel,
        "guard_firings": guards,
        "vision_residual_mass_share": round(vis_mass / tot_mass, 4) if tot_mass else None,
        "catastrophic_by_channel": dict(sorted(cat_by_src.items(), key=lambda kv: -kv[1])),
        "confidence_calibration": calib,
        "classification": classification,
        "comparison_status_counts": m["comparison_coverage"]["status_counts"],
        "_residuals": resid,  # stripped before output; used for pairing
    }


def paired_comparison(opus, haiku, tie_dex=0.05):
    shared = sorted(set(opus["_residuals"]) & set(haiku["_residuals"]))
    deltas = [haiku["_residuals"][a] - opus["_residuals"][a] for a in shared]
    return {
        "n_shared_papers": len(shared),
        "median_delta_haiku_minus_opus_dex": round(median(deltas), 4),
        "tie_threshold_dex": tie_dex,
        "opus_better": sum(1 for d in deltas if d > tie_dex),
        "haiku_better": sum(1 for d in deltas if d < -tie_dex),
        "tied": sum(1 for d in deltas if abs(d) <= tie_dex),
    }


def gt_pool():
    gt = load(GT_POOL)
    entries = gt["papers"] if isinstance(gt, dict) and "papers" in gt else gt
    if isinstance(entries, dict):
        entries = list(entries.values())
    n_excluded = sum(1 for e in entries if e.get("excluded"))
    ids = {e.get("arxiv_id") for e in entries if e.get("arxiv_id")}
    return {
        "n_entries": len(entries),
        "n_unique_papers": len(ids),
        "n_excluded_entries": n_excluded,
        "note": (
            "Scope: measured limits only — 25 is_projection GT entries "
            "(18 papers) are excluded (PR #698), leaving 409 entries / 329 "
            "ids and a 328-paper benchmark pool. One further GT id "
            "(1007.3766, re-keyed from 1410.5244 post-remediation) has no "
            "extraction in this run."
        ),
    }


def old_baseline():
    m = load(OLD_BASELINE)
    pta = m["per_type_aggregate"]
    return {
        "note": "Opus 4.8, OLD code, N=3 consensus (pre extraction-channels arc), rescored measured-limits-only",
        "micro_median_residual_dex": round(pta["micro_median_residual_dex"], 4),
        "macro_median_residual_dex": round(pta["macro_median_residual_dex"], 4),
    }


def to_markdown(out):
    lines = ["# Manuscript numbers (generated by make_paper_numbers.py — do not edit)", ""]
    lines.append("| quantity | Opus 4.8 (N=1, fixed) | Haiku 4.5 (N=1, fixed) |")
    lines.append("|---|---|---|")
    rows = [
        ("micro-median residual [dex]", "micro_median_residual_dex"),
        ("macro-median residual [dex]", "macro_median_residual_dex"),
        ("papers compared (finite)", "n_compared_finite"),
        ("catastrophic (>3 dex)", "catastrophic_gt3dex"),
        (">1 dex", "gt1dex"),
        ("mean frac within 0.3 dex", "frac_within_0_3dex_mean"),
        ("mean mass-range coverage", "mean_interpolation_coverage"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {out['arms']['opus'][key]} | {out['arms']['haiku'][key]} |")
    p = out["paired"]
    lines += [
        "",
        f"**Paired (both arms, {p['n_shared_papers']} papers):** Haiku − Opus median "
        f"= **{p['median_delta_haiku_minus_opus_dex']:+} dex**; Opus better {p['opus_better']}, "
        f"Haiku better {p['haiku_better']}, tied {p['tied']} (±{p['tie_threshold_dex']} dex).",
        "",
        f"**Old baseline** ({out['old_baseline']['note']}): "
        f"{out['old_baseline']['micro_median_residual_dex']} micro / "
        f"{out['old_baseline']['macro_median_residual_dex']} macro dex.",
        "",
        "## Opus-arm detail", "",
        "**Guard firings (recomputed from snapshots):** "
        + ", ".join(f"{k} = {v}" for k, v in out["arms"]["opus"]["guard_firings"].items()),
        "",
        "**Classification:** "
        + ", ".join(f"{k} {v['accuracy']*100:.1f}% (n={v['n']})"
                    for k, v in out["arms"]["opus"]["classification"].items()),
        "",
        "| channel | n compared | median residual [dex] |", "|---|---|---|",
    ]
    for src, d in out["arms"]["opus"]["per_channel_residuals"].items():
        lines.append(f"| {src} | {d['n_compared']} | {d['median_residual_dex']} |")
    lines += ["", "| confidence bin | n | mean conf | accuracy |", "|---|---|---|---|"]
    for b in out["arms"]["opus"]["confidence_calibration"]:
        lines.append(f"| {b['bin']} | {b['n_papers']} | {b['mean_confidence']} | {b['actual_accuracy']} |")
    g = out["gt_pool"]
    lines += [
        "",
        f"**GT pool:** {g['n_unique_papers']} unique papers, {g['n_entries']} entries "
        f"({g['n_excluded_entries']} excluded).",
        "",
        "Definitions are pinned in the module docstring of `make_paper_numbers.py`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    arms = {name: arm_summary(d) for name, d in ARMS.items()}
    out = {
        "generated_by": "paper/make_paper_numbers.py",
        "arms": arms,
        "paired": paired_comparison(arms["opus"], arms["haiku"]),
        "old_baseline": old_baseline(),
        "gt_pool": gt_pool(),
    }
    for a in arms.values():
        del a["_residuals"]
    with open(os.path.join(HERE, "numbers.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(HERE, "numbers.md"), "w") as f:
        f.write(to_markdown(out))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    for name, a in arms.items():
        print(f"[{name}] micro {a['micro_median_residual_dex']} macro {a['macro_median_residual_dex']} "
              f"finite {a['n_compared_finite']} cat {a['catastrophic_gt3dex']} gt1 {a['gt1dex']} "
              f"guards {a['guard_firings']}")


if __name__ == "__main__":
    main()
