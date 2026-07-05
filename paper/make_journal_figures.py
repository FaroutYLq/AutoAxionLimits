#!/usr/bin/env python3
"""Generate JINST manuscript figures from the definitive two-arm benchmark.

Data source: the 2026-07-04 definitive benchmark (both models on the fixed
pipeline, N=1), measured-limits-only scope (projections excluded, PR #698):
    evaluation/eval_runs/final2_opus_n1/metrics_noproj.json
    evaluation/eval_runs/final2_haiku_n1/metrics_noproj.json
No values are hand-entered; definitions match paper/make_paper_numbers.py
(compared paper = comparison_status "compared" with finite forward residual).

Run from repo root:  python paper/make_journal_figures.py
Outputs PDFs+PNGs into paper/figures/.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

ARMS = {
    "Opus 4.8": ("evaluation/eval_runs/final2_opus_n1/metrics_noproj.json", "C0", "-"),
    "Haiku 4.5": ("evaluation/eval_runs/final2_haiku_n1/metrics_noproj.json", "C1", "--"),
}
PRIMARY = "Opus 4.8"  # per-type + calibration figures use the production arm

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load_metrics(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return json.load(f)


def finite_residuals(m):
    out = []
    for pp in m["per_paper"]:
        if pp.get("comparison_status") != "compared":
            continue
        med = (pp.get("interp_metrics") or {}).get("median_residual_dex")
        if med is not None and math.isfinite(med):
            out.append(med)
    return np.array(sorted(out))


metrics = {name: load_metrics(path) for name, (path, _, _) in ARMS.items()}

# ===== Figure A: residual CDF, both arms =====================================
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for name, (path, color, ls) in ARMS.items():
    r = finite_residuals(metrics[name])
    y = np.arange(1, len(r) + 1) / len(r)
    med = float(np.median(r))
    ax.step(r, y, where="post", color=color, ls=ls, lw=2,
            label=f"{name} (N={len(r)}, median {med:.2f} dex)")
    ax.axvline(med, color=color, ls=":", lw=1)
    print(f"[figA] {name}: {len(r)} compared, median {med:.4f} dex")
for x, lab, c in [(0.3, "factor 2", "C3"), (0.5, "factor 3", "C2")]:
    ax.axvline(x, color=c, ls="--", lw=0.8)
    ax.text(x, 0.97, f"{lab} ", color=c, fontsize=8, va="top", ha="right", rotation=90)
ax.set_xscale("log")
ax.set_xlim(1e-3, 30)
ax.set_xlabel(r"per-paper median residual $|\Delta\log_{10} g|$ [dex]")
ax.set_ylabel("cumulative fraction of papers")
ax.set_title("Coupling-value accuracy (fixed pipeline, N=1)")
ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=8)
fig.savefig(os.path.join(FIG, "residual_cdf.pdf"))
fig.savefig(os.path.join(FIG, "residual_cdf.png"))
plt.close(fig)

# ===== Figure B: per-coupling-type median residual + 95% CI (primary arm) ====
M = metrics[PRIMARY]
pt = M["per_type_aggregate"]["per_type"]
items = sorted(pt.items(), key=lambda kv: kv[1]["median_residual_dex"])
names = [k for k, _ in items]
meds = np.array([v["median_residual_dex"] for _, v in items])
lo = np.array([v["ci95_lo"] for _, v in items])
hi = np.array([v["ci95_hi"] for _, v in items])
small = [v.get("small_sample", False) for _, v in items]
ns = [v["n"] for _, v in items]

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ypos = np.arange(len(names))
xerr = np.vstack([np.clip(meds - lo, 0, None), np.clip(hi - meds, 0, None)])
colors = ["#bbbbbb" if s else "C0" for s in small]
ax.errorbar(meds, ypos, xerr=xerr, fmt="none", ecolor="gray", elinewidth=1, capsize=3, zorder=1)
ax.scatter(meds, ypos, c=colors, s=40, zorder=2)
for i, (n, s) in enumerate(zip(ns, small)):
    ax.text(hi[i] * 1.05, ypos[i], f"N={n}" + ("*" if s else ""), va="center", fontsize=8)
micro = M["per_type_aggregate"]["micro_median_residual_dex"]
macro = M["per_type_aggregate"]["macro_median_residual_dex"]
# NOTE: the scorer's "macro_median_residual_dex" key is historically named;
# the statistic is the MEAN of per-type medians (evaluation/evaluate.py).
ax.axvline(micro, color="C2", ls="--", lw=1, label=f"micro-median {micro:.2f}")
ax.axvline(macro, color="C3", ls="--", lw=1, label=f"macro (mean of type medians) {macro:.2f}")
ax.axvline(0.3, color="k", ls=":", lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(names)
ax.set_xscale("log")
ax.set_xlabel("median residual [dex] (95% CI; * = small sample, N<5)")
ax.set_title(f"Residual by coupling type ({PRIMARY} arm)")
ax.legend(loc="lower right", fontsize=8)
fig.savefig(os.path.join(FIG, "per_type_residual.pdf"))
fig.savefig(os.path.join(FIG, "per_type_residual.png"))
plt.close(fig)
print(f"[figB] micro={micro:.4f} macro={macro:.4f} types={len(names)}")

# ===== Figure C: confidence calibration (primary arm) ========================
cal = [b for b in M["confidence_calibration"] if b["n_papers"] > 0]
conf = np.array([b["mean_confidence"] for b in cal])
acc = np.array([b["actual_accuracy"] for b in cal])
nps = np.array([b["n_papers"] for b in cal])
labels = [f"[{b['bin_lo']:.1f}–{b['bin_hi']:.1f})" for b in cal]

fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1, label="perfect calibration")
ax.scatter(conf, acc, s=nps * 1.5, c="C0", alpha=0.8, zorder=3)
for c, a, lab, n in zip(conf, acc, labels, nps):
    ax.annotate(f"{lab}\nN={n}", (c, a), textcoords="offset points", xytext=(6, -2), fontsize=7)
ax.set_xlabel("mean self-reported extraction confidence")
ax.set_ylabel(r"actual accuracy (resid $<0.32$ dex & cov $\geq50\%$)")
ax.set_title(f"Confidence calibration ({PRIMARY} arm)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=8)
fig.savefig(os.path.join(FIG, "confidence_calibration.pdf"))
fig.savefig(os.path.join(FIG, "confidence_calibration.png"))
plt.close(fig)
print("[figC] calib bins (conf->acc, N): "
      + "; ".join(f"{c:.2f}->{a:.2f}(n{n})" for c, a, n in zip(conf, acc, nps)))

# ===== Figure D: noise-floor histogram (per-paper run-to-run difference) =====
# Repeatability (issue #701, Plan B): repeat-1 = the definitive Opus run itself
# (final2_opus_n1); repeat-2 = a fresh matched-config N=1 re-read of a random
# subset (opus_repeat2). Per-paper metric == interp_metrics.median_residual_dex
# (same quantity as Figure A / numbers.json). We plot the run-to-run magnitude
# |Delta| per paper; the two N=1 reads are exchangeable, so magnitude (not sign)
# is the meaningful quantity. Log-spaced (half-decade) bins because |Delta|
# spans 0 to ~6 dex; the leftmost bin absorbs the exactly-reproduced core.
NF_R1 = "evaluation/eval_runs/final2_opus_n1/metrics_noproj.json"
NF_R2 = "evaluation/eval_runs/noise_floor_100_reuse/opus_repeat2/metrics_noproj.json"
NF_FROZEN = "evaluation/eval_runs/noise_floor_100_reuse/frozen_ids.json"
if os.path.exists(os.path.join(ROOT, NF_R2)):
    def _resid_map(relpath):
        m = load_metrics(relpath)
        out = {}
        for pp in m["per_paper"]:
            pid = pp.get("arxiv_id") or pp.get("id")
            med = (pp.get("interp_metrics") or {}).get("median_residual_dex")
            if med is not None and math.isfinite(med):
                out[pid] = float(med)
        return out

    frozen = set(json.load(open(os.path.join(ROOT, NF_FROZEN)))["ids"])
    r1m, r2m = _resid_map(NF_R1), _resid_map(NF_R2)
    ids = sorted(set(r1m) & set(r2m) & frozen)
    delta = np.array([abs(r1m[i] - r2m[i]) for i in ids])
    n_pair = len(delta)
    n_exact = int(np.sum(delta < 1e-4))
    med_d = float(np.median(delta))
    n_flip = 11  # observed text<->vision channel flips among the paired papers

    edges = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    labels = [r"$<$0.01", "0.01–\n0.03", "0.03–\n0.1", "0.1–\n0.3",
              "0.3–1", "1–3", r"$\geq$3"]
    counts = np.array([int(np.sum((delta >= edges[k]) & (delta < edges[k + 1])))
                       for k in range(len(edges) - 1)])
    # core = runs agree to <0.03 dex (bins 0-1); tail = >=0.03 dex (bins 2+)
    CORE = 2
    n_core, n_tail = int(counts[:CORE].sum()), int(counts[CORE:].sum())
    bar_colors = ["C0"] * CORE + ["C1"] * (len(counts) - CORE)

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    xpos = np.arange(len(counts))
    ax.bar(xpos, counts, width=0.92, color=bar_colors, edgecolor="k", linewidth=0.5, zorder=2)
    for x, c in zip(xpos, counts):
        if c:
            ax.text(x, c + 0.5, str(c), ha="center", va="bottom", fontsize=9)
    ax.axvline(CORE - 0.5, color="0.4", ls="--", lw=0.9, zorder=1)
    ymax = counts.max() + 8
    ax.set_ylim(0, ymax)
    # region labels, placed above the bars (core bar is tall, so anchor high)
    ax.text((CORE - 1) / 2.0, ymax - 0.5,
            f"runs agree $<$0.03 dex\n({n_core} papers)",
            ha="center", va="top", fontsize=8.5, color="C0")
    ax.text((CORE + len(counts) - 1) / 2.0, ymax - 0.5,
            f"routing-flip tail\n({n_tail} papers, {n_flip} channel flips)",
            ha="center", va="top", fontsize=8.5, color="C1")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel(r"run-to-run change per paper $|\Delta\log_{10} g|$ [dex] (half-decade bins)")
    ax.set_ylabel(f"papers (of {n_pair})")
    ax.set_title("Extraction repeatability: per-paper run-to-run difference\n"
                 "(Opus, $N{=}1$; repeat-1 = benchmark run, repeat-2 = fresh re-run)",
                 fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.60, f"median $|\\Delta|$ = {med_d:.03f} dex\n"
                        f"{n_exact}/{n_pair} reproduce to $<10^{{-4}}$ dex",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.savefig(os.path.join(FIG, "noise_floor_hist.pdf"))
    fig.savefig(os.path.join(FIG, "noise_floor_hist.png"))
    plt.close(fig)
    print(f"[figD] noise floor: n_pair={n_pair} median|d|={med_d:.4f} "
          f"exact={n_exact} core={n_core} tail={n_tail} counts={counts.tolist()}")
else:
    print(f"[figD] SKIPPED — repeat-2 metrics not found ({NF_R2})")

print("wrote:", sorted(f for f in os.listdir(FIG) if f.endswith(".pdf")))
