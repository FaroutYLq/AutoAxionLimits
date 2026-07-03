"""Replay the WS3 wrong-curve vision gates over the cached benchmark snapshots.

Keyless scoreboard for ``pipeline/vision_gates.py`` (WS3 PR 1): runs the four
deterministic gates over all cached full346 extraction snapshots
(``evaluation/results/<id>.json``) plus the cached abstracts
(``evaluation/results/metadata_cache.json``) and reports:

* per-gate trigger list with the note excerpt that fired it,
* catch rate on the 9 known wrong-curve papers (target >= 5/9),
* false-trigger rate on the good vision papers (target <= 2%).

No Anthropic API calls, no network. Usage:

    python -m evaluation.replay_gates --out evaluation/eval_runs/gate_replay.md

IMPORTANT caveat (stated in the report too): cached snapshot ``notes`` are the
WINNING sample's notes post-selection. At runtime the gates see each
candidate's own stage-2 notes — strictly more information — so replay is a
LOWER bound on the catch rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.vision_gates import (  # noqa: E402
    GateResult,
    VISION_SOURCES,
    build_experiment_lexicon,
    check_vision_gates,
    extract_vision_segment,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# The wrong_curve_vision failure family from the full346 per-paper digest
# (evaluation/eval_runs/failure_analysis_full346.md §2 + detail table).
KNOWN_WRONG_CURVE: dict[str, str] = {
    "1512.06165": "A: projection paper, traced the existing 'static EP tests' region",
    "1508.01798": "A: projection paper, traced the existing Eot-Wash EP limit",
    "2309.07995": "A: projection paper, traced the existing Eot-Wash/LIGO-Virgo bounds",
    "1708.02111": "B: declared AxionElectron but traced a g_agamma (IAXO) panel",
    "1903.12190": "C: traced the GeV-scale millicharge panel, not the eV-scale hidden-photon one",
    "1808.02340": "C: kept a nominal-mass text point ~6 dex below the paper's 0.8-500 keV range",
    "1008.3536": "D: traced the union of all regions in a compilation plot",
    "1207.3275": "D: traced the LSW exclusion merged with surrounding context",
    "1912.07751": "(notes-silent: dual-axis miscalibration; not expected catchable by replay)",
}

# Good-vision population: compared against GT with a sub-1-dex median residual.
GOOD_VISION_MAX_RESID_DEX = 1.0


def load_snapshot(arxiv_id: str) -> dict | None:
    p = RESULTS_DIR / f"{arxiv_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def replay_paper(snap: dict, abstract: str | None, lexicon) -> list[GateResult]:
    """Run the gates on one cached snapshot (the winning sample only)."""
    return check_vision_gates(
        source=snap.get("data_source"),
        is_projection=bool(snap.get("is_projection")),
        coupling_type=snap.get("coupling_type"),
        vision_notes=extract_vision_segment(snap.get("notes")),
        data_points=snap.get("data_points") or [],
        abstract=abstract,
        suggested_experiment_name=snap.get("suggested_experiment_name"),
        paper_title=snap.get("paper_title"),
        lexicon=lexicon,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args(argv)

    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    metadata = json.loads((RESULTS_DIR / "metadata_cache.json").read_text())
    per_paper = {p["arxiv_id"]: p for p in metrics["per_paper"]}
    lexicon = build_experiment_lexicon(PROJECT_ROOT / "limit_data")

    triggers: dict[str, list[GateResult]] = {}
    replayed: list[str] = []
    for aid in sorted(per_paper):
        snap = load_snapshot(aid)
        if snap is None:
            continue
        replayed.append(aid)
        fired = replay_paper(snap, (metadata.get(aid) or {}).get("abstract"), lexicon)
        if fired:
            triggers[aid] = fired

    # Populations ---------------------------------------------------------
    def resid(aid: str) -> float | None:
        im = per_paper[aid].get("interp_metrics") or {}
        return im.get("median_residual_dex")

    def is_vision(aid: str) -> bool:
        return per_paper[aid].get("data_source") in VISION_SOURCES

    good_vision = [aid for aid in replayed
                   if aid not in KNOWN_WRONG_CURVE
                   and is_vision(aid)
                   and per_paper[aid].get("comparison_status") == "compared"
                   and resid(aid) is not None
                   and resid(aid) < GOOD_VISION_MAX_RESID_DEX]

    caught = {aid: triggers[aid] for aid in KNOWN_WRONG_CURVE if aid in triggers}
    # False triggers: a REJECT on a good vision paper. Demotes (gate D) are
    # soft — reported separately, they only reorder the selector.
    false_rejects = {aid: [r for r in triggers.get(aid, []) if r.action == "reject"]
                     for aid in good_vision}
    false_rejects = {k: v for k, v in false_rejects.items() if v}
    good_demotes = {aid: [r for r in triggers.get(aid, []) if r.action == "demote"]
                    for aid in good_vision}
    good_demotes = {k: v for k, v in good_demotes.items() if v}

    # Report ---------------------------------------------------------------
    lines: list[str] = []
    w = lines.append
    w("# WS3 gate replay — wrong-curve vision gates over the cached full346 snapshots")
    w("")
    w("Generated by `python -m evaluation.replay_gates` (keyless: no API, no network).")
    w("Gates: `pipeline/vision_gates.py`. Snapshots: `evaluation/results/*.json`;")
    w("abstracts: `evaluation/results/metadata_cache.json`.")
    w("")
    w("> **Replay is a LOWER bound on the runtime catch rate.** Cached snapshot")
    w("> `notes` are the WINNING sample's notes post-selection; at runtime each")
    w("> gate sees every candidate's own stage-2 notes, which is strictly more")
    w("> information.")
    w("")
    w("## Headline")
    w("")
    w(f"- Snapshots replayed: **{len(replayed)}**")
    n_catchable = len(KNOWN_WRONG_CURVE)
    w(f"- Known wrong-curve papers caught: **{len(caught)}/{n_catchable}** (target ≥ 5/9)")
    fr = 100.0 * len(false_rejects) / len(good_vision) if good_vision else 0.0
    w(f"- False rejects on the {len(good_vision)} good vision papers "
      f"(compared, median < {GOOD_VISION_MAX_RESID_DEX} dex): "
      f"**{len(false_rejects)} ({fr:.1f}%)** (target ≤ 2%)")
    w(f"- Soft demotes (gate D) on good vision papers: {len(good_demotes)}")
    w(f"- Papers with any trigger: {len(triggers)}")
    w("")

    extra = sorted(aid for aid in triggers
                   if aid not in KNOWN_WRONG_CURVE and aid not in good_vision)
    if extra:
        w(f"Triggers on {len(extra)} papers outside both the known wrong-curve "
          f"set and the good-vision set: {', '.join(extra)}. These are candidate "
          f"wrong-curve papers the failure digest did not enumerate (high "
          f"residual or no comparable GT) — at runtime the gates would make "
          f"selection fall back rather than emit these traces.")
        w("")

    w("## Catches on the 9 known wrong-curve papers")
    w("")
    w("| paper | expected failure | gate(s) fired | excerpt |")
    w("|---|---|---|---|")
    for aid, why in KNOWN_WRONG_CURVE.items():
        fired = triggers.get(aid, [])
        gates = ", ".join(f"{r.gate} ({r.action})" for r in fired) or "—"
        exc = (fired[0].excerpt if fired else "").replace("|", "\\|")
        w(f"| {aid} | {why} | {gates} | {exc[:160]} |")
    w("")

    for gate_name in ("A_projection_target", "B_axis_vs_coupling",
                      "C_mass_regime", "D_compilation_envelope"):
        rows = [(aid, r) for aid, rs in sorted(triggers.items())
                for r in rs if r.gate == gate_name]
        w(f"## Gate {gate_name} — {len(rows)} trigger(s)")
        w("")
        if rows:
            w("| paper | source | resid (dex) | known? | excerpt |")
            w("|---|---|---|---|---|")
            for aid, r in rows:
                rd = resid(aid)
                rd_s = f"{rd:.2f}" if rd is not None else "—"
                known = "**wrong-curve**" if aid in KNOWN_WRONG_CURVE else (
                    "good" if aid in good_vision else "other")
                exc = r.excerpt.replace("|", "\\|")[:200]
                w(f"| {aid} | {per_paper[aid].get('data_source')} | {rd_s} | {known} | {exc} |")
        w("")

    if false_rejects:
        w("## False rejects on good vision papers (must stay ≤ 2%)")
        w("")
        w("| paper | resid (dex) | gate | excerpt |")
        w("|---|---|---|---|")
        for aid, rs in sorted(false_rejects.items()):
            for r in rs:
                w(f"| {aid} | {resid(aid):.2f} | {r.gate} | "
                  f"{r.excerpt.replace('|', chr(92) + '|')[:200]} |")
        w("")

    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)

    # Machine-readable summary on stderr for the PR impact block.
    print(f"caught={len(caught)}/{n_catchable} false_rejects={len(false_rejects)}"
          f"/{len(good_vision)} ({fr:.1f}%) triggers={len(triggers)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
