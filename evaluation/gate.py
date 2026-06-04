"""Extraction-regression gate — P4 of the #566 roadmap (issue #569).

Turns the before/after subset eval that **caught the #550+#561 regression**
(`evaluation/eval_runs/comparison.md`) into a pass/fail check. Loads a committed
BEFORE (baseline) snapshot and an AFTER (PR) snapshot of subset extractions,
reuses `subset_compare._records_for` / `_summarize` (NO metric reimplementation —
the gate can never disagree with the report a human reads), and asserts the
gate rules G1-G7 below. Exits non-zero on any failure so the CI step fails;
`--soft` prints the table but always exits 0 (for the nightly comment-only run).

No new dependencies: stdlib + the existing `subset_compare` summary functions
(which pull numpy, already a CI dep). The HARD-floor scan (P0 R5 invariant)
reuses `pipeline.transform_guard.in_valid_ranges` + `pipeline.config.VALID_RANGES`
— both anthropic-free — so the whole gate runs in the no-API `eval_tests` job.

**Symmetric-set counting:** the rules are evaluated only over papers that
extracted *without an infrastructure error on either side* (`_symmetric_ids`). An
arXiv 429/timeout drops a paper to a download-`error` on one side only; comparing
the asymmetric sets would shrink one side's compared/zero-overlap counts and
false-fail G2/G7 on flakiness rather than a real regression. Dropping such papers
from BOTH sides makes infra drops cancel. A real logic error is kept (G6 gates
it). This is the robustness piece that lets the gate become auto-required.

CLI:
    python -m evaluation.gate --before evaluation/eval_runs/baseline \
        --after evaluation/eval_runs/after --key figure [--soft] [--out report.md]

Gate rules (thresholds calibrated from comparison.md; the noise floor is the
single source of the residual tolerance):

| # | Rule | Threshold |
|---|------|-----------|
| G1 | overall median residual must not regress beyond the noise floor | after <= before + 0.32 dex |
| G2 | zero-overlap count must not increase | after <= before + 2 (flap slack) |
| G3 | no new `unit_offset` zero-overlap cause | after <= before + 2 (flap slack) |
| G4 | figure_vision per-source residual must not regress beyond noise floor | after <= before + 0.32 dex |
| G5 | figure_vision <=0.3 dex fraction must not drop > 5 pp | after >= before - 0.05 |
| G6 | zero logic errors AND zero P0 HARD-floor violations | strict |
| G7 | papers-compared must not collapse | after >= before - 3 |
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.subset_compare import _load_result_dir, _records_for, _summarize
from evaluation.ground_truth import GroundTruthEntry, load_ground_truth
from pipeline.config import VALID_RANGES

# --- Thresholds (single source of truth; unit-tested at the boundaries) ------

NOISE_FLOOR_DEX = 0.32      # run-to-run LLM noise floor (metrics.py / PR #545)
FRAC_0_3_SLACK = 0.05       # G5: allowed drop in figure_vision <=0.3 dex fraction
COMPARED_SLACK = 3          # G7: allowed drop in papers-compared (absorbs flap)
FLOOR_SLACK = 3             # G6: allowed increase in HARD-floor violations.
ZO_SLACK = 2                # G2: allowed increase in zero-overlap count (flap).
UNIT_OFFSET_SLACK = 2       # G3: allowed increase in unit_offset cause (flap).
# G2/G3 were strict, but a no-op N=3-voted repeat pair still flaps a single paper
# across the zero-overlap / unit_offset boundary (observed +1 each on the union
# key). N=3 read voting (#584) damps the residual rules below the floor, but these
# COUNT rules still need a small flap slack — same rationale as COMPARED_SLACK /
# FLOOR_SLACK. A real regression adds DOZENS (the #550 fixture: +18 zero-overlap,
# +12 unit_offset, and blows up G1/G4), far above the slack, so sensitivity holds.
# Logic errors (crashes) are deterministic per code version, so G6 gates them
# strictly (no increase). HARD-floor violation membership, by contrast, flaps on
# the same LLM noise as every residual (a median can cross VALID_RANGES between
# two identical-code repeats — observed 0->2 on the 11-paper unit_offset key), so
# the floor count carries the same count-flap slack as G7. A real #550-style
# regression adds far more than FLOOR_SLACK violations, so protection is retained.

# An "error" whose message matches one of these is an infrastructure failure
# (arXiv/Anthropic flakiness), NOT an extraction-logic bug — it must not fail the
# gate (it should be retried / re-run). Everything else is a real logic error.
_INFRA_ERROR_PATTERNS = (
    "download", "429", "timed out", "timeout", "connection",
    "read operation", "temporarily", "rate limit", "overloaded",
)


def _is_infra_error(msg: str) -> bool:
    m = (msg or "").lower()
    return any(p in m for p in _INFRA_ERROR_PATTERNS)


@dataclass
class GateRow:
    rule: str
    description: str
    before: str
    after: str
    threshold: str
    passed: bool

    def fmt(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (f"| {self.rule} | {mark} | {self.before} | {self.after} | "
                f"{self.threshold} | {self.description} |")


def _f(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "∞"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2] if s else 0.0


def _median_in_valid_ranges(data_points: list, ct: str | None) -> bool:
    """True iff the candidate's median mass/coupling lie in strict VALID_RANGES.

    Inlined (rather than importing P0's `transform_guard.in_valid_ranges`) so the
    gate is independent of P0's merge order; both share the single `VALID_RANGES`
    source. When P0 has landed this is the same predicate as its R5 hard floor.
    """
    valid = VALID_RANGES.get(ct) if ct else None
    if not data_points or not valid:
        return True
    masses = [float(m) for m, _ in data_points if float(m) > 0]
    couplings = [float(g) for _, g in data_points if float(g) > 0]
    if not masses or not couplings:
        return True
    mass_lo, mass_hi = valid["mass"]
    coup_lo, coup_hi = valid["coupling"]
    return (mass_lo <= _median(masses) <= mass_hi
            and coup_lo <= _median(couplings) <= coup_hi)


def _hard_floor_violations(results: dict[str, dict], ids: list[str]) -> list[tuple]:
    """P0 R5 no-API invariant: after medians must lie in strict VALID_RANGES.

    Returns [(arxiv_id, coupling_type)] for each paper whose committed
    data_points have a median mass/coupling outside `VALID_RANGES[ct]`.
    """
    viol = []
    for aid in ids:
        r = results.get(aid)
        if not r or r.get("error"):
            continue
        pts = r.get("data_points") or []
        ct = r.get("coupling_type")
        if not pts or not ct:
            continue
        if not _median_in_valid_ranges(pts, ct):
            viol.append((aid, ct))
    return viol


def _logic_errors(results: dict[str, dict], ids: list[str]) -> tuple[list, list]:
    """Split error papers into (logic_errors, infra_errors)."""
    logic, infra = [], []
    for aid in ids:
        r = results.get(aid)
        if not r or not r.get("error"):
            continue
        (infra if _is_infra_error(str(r["error"])) else logic).append(aid)
    return logic, infra


def _symmetric_ids(b_results: dict[str, dict], a_results: dict[str, dict],
                   ids: list[str]) -> tuple[list[str], list[str]]:
    """Keep only papers assessable on BOTH sides; drop ones with an INFRA error
    on either side.

    arXiv 429s / timeouts drop a paper to a download-`error` on one side only,
    which would otherwise shrink that side's compared/zero-overlap set and
    false-fail G2/G7 (and skew the residual aggregates) on infrastructure rather
    than a real regression. Comparing only the symmetric set makes those drops
    cancel. A LOGIC error (a real extraction bug) is NOT dropped — it stays in the
    set so G6 gates it and its lost comparability is visible.

    Returns ``(kept, dropped)``.
    """
    kept, dropped = [], []
    for aid in ids:
        br, ar = b_results.get(aid), a_results.get(aid)
        b_infra = bool(br and br.get("error") and _is_infra_error(str(br["error"])))
        a_infra = bool(ar and ar.get("error") and _is_infra_error(str(ar["error"])))
        (dropped if (b_infra or a_infra) else kept).append(aid)
    return kept, dropped


def apply_rules(b: dict, a: dict, *, b_logic_n: int, a_logic_n: int,
                b_floor_n: int, a_floor_n: int) -> list[GateRow]:
    """Pure G1-G7 evaluation over two `_summarize` dicts. No IO — unit-testable
    at the exact threshold boundaries (the #544-style guarantee).

    ``b`` / ``a`` are `subset_compare._summarize` outputs; the ``*_logic_n`` /
    ``*_floor_n`` are the logic-error and HARD-floor-violation COUNTS for the
    before/after snapshots (from :func:`_logic_errors` / :func:`_hard_floor_violations`).

    G6 is a **no-increase (delta)** rule, not absolute-zero: the committed master
    baseline already carries some out-of-`VALID_RANGES` raw reads (P0's final
    floor *flags* them low-confidence, it does not delete the data), so the gate
    forbids *adding* logic errors or floor violations rather than requiring zero.
    """
    bf = b["sources"].get("figure_vision", {})
    af = a["sources"].get("figure_vision", {})
    b_uo = b["zo_causes"].get("unit_offset", 0)
    a_uo = a["zo_causes"].get("unit_offset", 0)

    rows: list[GateRow] = []

    # G1 — overall median residual.
    bm, am = b["overall_median_resid"], a["overall_median_resid"]
    if bm is None:
        g1 = True  # no baseline median to regress from
    elif am is None:
        g1 = False  # after lost all comparable papers
    else:
        g1 = am <= bm + NOISE_FLOOR_DEX + 1e-9
    rows.append(GateRow("G1", "overall median residual (dex)", _f(bm), _f(am),
                        f"after <= before + {NOISE_FLOOR_DEX}", g1))

    # G2 — zero-overlap count (small flap slack).
    g2 = a["n_zero_overlap"] <= b["n_zero_overlap"] + ZO_SLACK
    rows.append(GateRow("G2", "zero-overlap papers", str(b["n_zero_overlap"]),
                        str(a["n_zero_overlap"]), f"after <= before + {ZO_SLACK}", g2))

    # G3 — unit_offset zero-overlap cause (small flap slack).
    g3 = a_uo <= b_uo + UNIT_OFFSET_SLACK
    rows.append(GateRow("G3", "unit_offset zero-overlap cause", str(b_uo),
                        str(a_uo), f"after <= before + {UNIT_OFFSET_SLACK}", g3))

    # G4 — figure_vision per-source median residual.
    bfm, afm = bf.get("median_resid"), af.get("median_resid")
    if bfm is None:
        g4 = True
    elif afm is None:
        g4 = False
    else:
        g4 = afm <= bfm + NOISE_FLOOR_DEX + 1e-9
    rows.append(GateRow("G4", "figure_vision median residual (dex)", _f(bfm),
                        _f(afm), f"after <= before + {NOISE_FLOOR_DEX}", g4))

    # G5 — figure_vision <=0.3 dex fraction.
    bff, aff = bf.get("frac_0_3"), af.get("frac_0_3")
    if bff is None:
        g5 = True
    elif aff is None:
        g5 = False
    else:
        g5 = aff >= bff - FRAC_0_3_SLACK - 1e-9
    rows.append(GateRow("G5", "figure_vision <=0.3 dex fraction", _f(bff),
                        _f(aff), f"after >= before - {FRAC_0_3_SLACK}", g5))

    # G6 — no NEW logic errors (strict) and no run-away HARD-floor violations
    # (slacked, since membership flaps on LLM noise).
    g6 = (a_logic_n <= b_logic_n) and (a_floor_n <= b_floor_n + FLOOR_SLACK)
    rows.append(GateRow(
        "G6", "logic errors / HARD-floor violations",
        f"{b_logic_n}/{b_floor_n}",
        f"{a_logic_n}/{a_floor_n}",
        f"logic after<=before; floor after<=before+{FLOOR_SLACK}", g6))

    # G7 — papers-compared must not collapse.
    g7 = a["n_compared"] >= b["n_compared"] - COMPARED_SLACK
    rows.append(GateRow("G7", "papers compared", str(b["n_compared"]),
                        str(a["n_compared"]), f"after >= before - {COMPARED_SLACK}", g7))

    return rows


def evaluate_gate(before_dir: Path, after_dir: Path, ids: list[str]) -> tuple[list[GateRow], dict]:
    """Load both snapshot dirs, pair against ground truth, and apply the rules.

    Pure function of the two directories' contents (the property the offline
    fixtures rely on).
    """
    entries_by_id: dict[str, list[GroundTruthEntry]] = defaultdict(list)
    for e in load_ground_truth():
        entries_by_id[e.arxiv_id].append(e)

    b_results = _load_result_dir(before_dir)
    a_results = _load_result_dir(after_dir)

    # Symmetric-set counting: compare only papers assessable on BOTH sides, so an
    # arXiv-throttle drop on one side cancels instead of false-failing the gate.
    kept, dropped = _symmetric_ids(b_results, a_results, ids)

    b = _summarize(_records_for(b_results, entries_by_id, kept))
    a = _summarize(_records_for(a_results, entries_by_id, kept))

    b_logic, _b_infra = _logic_errors(b_results, kept)
    a_logic, a_infra = _logic_errors(a_results, kept)
    b_floor = _hard_floor_violations(b_results, kept)
    a_floor = _hard_floor_violations(a_results, kept)

    rows = apply_rules(b, a, b_logic_n=len(b_logic), a_logic_n=len(a_logic),
                       b_floor_n=len(b_floor), a_floor_n=len(a_floor))
    detail = {
        "before": b, "after": a,
        "logic_errors": a_logic, "infra_errors": a_infra,
        "floor_violations": a_floor,
        "dropped_infra": dropped, "n_kept": len(kept),
    }
    return rows, detail


def render(rows: list[GateRow], detail: dict, ids: list[str]) -> str:
    passed = all(r.passed for r in rows)
    n_dropped = len(detail.get("dropped_infra", []))
    n_kept = detail.get("n_kept", len(ids))
    sym = (f" ({n_kept} compared symmetrically, {n_dropped} dropped for an "
           f"infra error on one side)" if n_dropped else f" ({n_kept} compared)")
    L = ["# Extraction-regression gate (P4 / #569)\n",
         f"Subset: {len(ids)} papers{sym}. Overall: "
         f"**{'PASS' if passed else 'FAIL'}**.\n",
         "| Rule | Result | Before | After | Threshold | Metric |",
         "|---|---|---|---|---|---|"]
    L += [r.fmt() for r in rows]
    if detail.get("dropped_infra"):
        L.append(f"\n_Symmetric-set: dropped {n_dropped} paper(s) with an infra "
                 f"error on one side (not gated): "
                 f"{', '.join(detail['dropped_infra'])}_")
    if detail["infra_errors"]:
        L.append(f"\n_Note: {len(detail['infra_errors'])} infrastructure error(s) "
                 f"(not gated): {', '.join(detail['infra_errors'])}_")
    if detail["floor_violations"]:
        L.append(f"\n_HARD-floor violations: "
                 f"{', '.join(f'{a} ({c})' for a, c in detail['floor_violations'])}_")
    return "\n".join(L)


def run_gate(before_dir: Path, after_dir: Path, ids: list[str], *,
             soft: bool = False, out: Path | None = None) -> int:
    rows, detail = evaluate_gate(before_dir, after_dir, ids)
    report = render(rows, detail, ids)
    print(report)
    if out:
        out.write_text(report)
    passed = all(r.passed for r in rows)
    if soft:
        return 0
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", required=True, help="baseline snapshot dir")
    p.add_argument("--after", required=True, help="PR snapshot dir")
    p.add_argument("--key", default="union", help="subset key (default: union)")
    p.add_argument("--soft", action="store_true",
                   help="print the table but always exit 0 (nightly comment-only)")
    p.add_argument("--out", default=None, help="write the markdown report here")
    args = p.parse_args()
    subset = json.loads((Path(__file__).parent / "subset" / "subset.json").read_text())
    if args.key not in subset:
        raise SystemExit(f"key {args.key!r} not in subset.json (have {list(subset)})")
    return run_gate(Path(args.before), Path(args.after), subset[args.key],
                    soft=args.soft, out=Path(args.out) if args.out else None)


if __name__ == "__main__":
    raise SystemExit(main())
