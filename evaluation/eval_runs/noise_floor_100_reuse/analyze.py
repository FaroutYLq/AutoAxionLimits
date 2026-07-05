#!/usr/bin/env python3
"""Noise-floor analysis: pair repeat-1 (final2 cached) vs repeat-2 (fresh re-read)
on the frozen 100 ids, per model. Produces the three deliverables of issue #701:

  (1) per-paper std histogram  -- std at n=2 = |delta|/sqrt(2); report the MEDIAN
      (robust; the mean is dominated by the routing-flip tail).  "how much one
      paper wobbles."
  (2) aggregate-median floor + bootstrap CI -- resample which of the 2 reads wins
      per paper, recompute the pool median, take percentiles.  The ONLY floor that
      goes next to the reported medians / the paired delta.
  (3) paired-delta noise -- run-to-run noise on the haiku-opus paired delta.

Per-paper metric == metrics_noproj.json per_paper[].interp_metrics.median_residual_dex
(the same quantity behind numbers.json).  Deterministic: bootstrap uses a fixed seed.
No API calls.
"""
import json, os, statistics, math
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
FROZEN = set(json.load(open(BASE / "frozen_ids.json"))["ids"])
BOOT_SEED = 701
BOOT_N = 20000

ARMS = {
    "opus":  (REPO / "evaluation/eval_runs/final2_opus_n1/metrics_noproj.json",
              BASE / "opus_repeat2/metrics_noproj.json"),
    "haiku": (REPO / "evaluation/eval_runs/final2_haiku_n1/metrics_noproj.json",
              BASE / "haiku_repeat2/metrics_noproj.json"),
}


def per_paper_residuals(metrics_path):
    """id -> forward-interp median residual (dex), compared+finite, frozen-100 only."""
    m = json.load(open(metrics_path))
    out = {}
    for p in m["per_paper"]:
        pid = p.get("arxiv_id") or p.get("id")
        if pid not in FROZEN:
            continue
        med = (p.get("interp_metrics") or {}).get("median_residual_dex")
        if med is not None and math.isfinite(med):
            out[pid] = float(med)
    return out


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def bootstrap_read_selection(pairs, seed=BOOT_SEED, B=BOOT_N):
    """pairs: list of (r1, r2). Each iter picks r1 or r2 per paper, recompute pool
    median. Returns (lo95, hi95, boot_median_of_medians)."""
    import random
    rng = random.Random(seed)
    meds = []
    for _ in range(B):
        pool = [ (p[0] if rng.random() < 0.5 else p[1]) for p in pairs ]
        meds.append(median(pool))
    meds.sort()
    lo = meds[int(0.025 * B)]
    hi = meds[int(0.975 * B)]
    return lo, hi, median(meds)


def analyze_arm(name, r1_path, r2_path):
    if not Path(r2_path).exists():
        return {"arm": name, "status": f"repeat-2 metrics missing: {r2_path}"}
    r1 = per_paper_residuals(r1_path)   # repeat 1 = final2 (benchmark run itself)
    r2 = per_paper_residuals(r2_path)   # repeat 2 = fresh matched-config re-run
    shared = sorted(set(r1) & set(r2))
    pairs = [(r1[i], r2[i]) for i in shared]

    # (1) per-paper std at n=2 = |delta|/sqrt(2)
    per_paper_std = [abs(a - b) / math.sqrt(2) for a, b in pairs]
    abs_delta = [abs(a - b) for a, b in pairs]

    # (2) aggregate-median floor: pool medians of each read + read-selection bootstrap CI
    med_r1 = median([r1[i] for i in shared])
    med_r2 = median([r2[i] for i in shared])
    lo, hi, boot_med = bootstrap_read_selection(pairs)

    # histogram bins for per-paper std
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 1e9]
    hist = [0] * (len(bins) - 1)
    for s in per_paper_std:
        for k in range(len(bins) - 1):
            if bins[k] <= s < bins[k + 1]:
                hist[k] += 1
                break

    return {
        "arm": name,
        "n_frozen": len(FROZEN),
        "n_compared_r1": len(r1),
        "n_compared_r2": len(r2),
        "n_paired": len(pairs),
        "pool_median_repeat1_dex": round(med_r1, 4),
        "pool_median_repeat2_dex": round(med_r2, 4),
        "abs_pool_median_shift_dex": round(abs(med_r1 - med_r2), 4),
        # deliverable 1
        "per_paper_std_median_dex": round(median(per_paper_std), 4),
        "per_paper_std_mean_dex": round(statistics.fmean(per_paper_std), 4) if per_paper_std else None,
        "per_paper_absdelta_median_dex": round(median(abs_delta), 4),
        "per_paper_std_hist_bins": bins,
        "per_paper_std_hist_counts": hist,
        # deliverable 2  (the headline-guard floor)
        "aggregate_floor_boot_median_dex": round(boot_med, 4),
        "aggregate_floor_ci95_dex": [round(lo, 4), round(hi, 4)],
        "aggregate_floor_ci_halfwidth_dex": round((hi - lo) / 2, 4),
        "_pairs": pairs,  # kept for paired-delta stage; stripped before dumping
    }


def paired_delta_noise(res_opus, res_haiku):
    """Deliverable 3: run-to-run noise on the haiku-opus paired median delta.
    delta_run = median(haiku_read) - median(opus_read); compare the two runs."""
    if "_pairs" not in res_opus or "_pairs" not in res_haiku:
        return {"status": "need both arms"}
    # per-run pool medians already computed; the paired delta is (haiku - opus)
    d1 = res_haiku["pool_median_repeat1_dex"] - res_opus["pool_median_repeat1_dex"]
    d2 = res_haiku["pool_median_repeat2_dex"] - res_opus["pool_median_repeat2_dex"]
    return {
        "delta_repeat1_dex": round(d1, 4),
        "delta_repeat2_dex": round(d2, 4),
        "delta_run_to_run_shift_dex": round(abs(d1 - d2), 4),
        "reported_paired_delta_dex": 0.2098,
        "note": "run-to-run shift in the haiku-opus median gap; compare to the 0.21 dex headline",
    }


def main():
    results = {}
    for name, (r1p, r2p) in ARMS.items():
        results[name] = analyze_arm(name, r1p, r2p)

    summary = {"seed": BOOT_SEED, "bootstrap_iters": BOOT_N, "arms": {}}
    for name, res in results.items():
        if "_pairs" in res:
            paired = res.pop("_pairs")
            res["_had_pairs"] = len(paired)
            res["_pairs"] = paired  # keep for paired-delta below
        summary["arms"][name] = {k: v for k, v in res.items() if k != "_pairs"}

    if all("_pairs" in results[a] for a in ("opus", "haiku") if a in results):
        summary["paired_delta_noise"] = paired_delta_noise(results["opus"], results["haiku"])

    out = BASE / "noise_floor_results.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary["arms"], indent=2))
    if "paired_delta_noise" in summary:
        print("\npaired-delta noise:", json.dumps(summary["paired_delta_noise"], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
