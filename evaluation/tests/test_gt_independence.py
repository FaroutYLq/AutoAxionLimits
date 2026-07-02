"""Ground-truth independence guard.

PROJECT INVARIANT (user decision): the extraction benchmark's ground truth is
cajohare/O'Hare's own ``limit_data/<Type>/*.txt`` repo files, and NOTHING else.
Claude-extracted / Claude-digitized data must NEVER be used as GT in the scoring
path — the pipeline is itself a Claude extractor, so grading it against
Claude-made GT would measure self-agreement (circularity), not correctness.

Concretely:
  * ``evaluation/ground_truth/gold/gold.json`` is Claude-digitized
    (``digitize_model: claude-opus-4-*``). It may exist only as the opt-in
    ``--gold`` / ``gold_diff`` diagnostic; it must stay OUT of the scoring path
    (``subset_compare`` / ``gate``).
  * Every benchmark GT curve actually scored must be an O'Hare ``limit_data``
    file (loaded via the repo-copied ``ground_truth/data/<id>.txt`` or the
    entry's ``reference_repo_file``).

If a future change tries to route Claude/gold data into scoring, these tests
fail. No API/network.

Run:
    pytest evaluation/tests/test_gt_independence.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUBSET = json.loads((PROJECT_ROOT / "evaluation/subset/subset.json").read_text())
PAPERS: dict[str, list] = {}
for _e in json.loads(
    (PROJECT_ROOT / "evaluation/ground_truth/papers.json").read_text()
)["papers"]:
    PAPERS.setdefault(_e["arxiv_id"], []).append(_e)
DATA_DIR = PROJECT_ROOT / "evaluation/ground_truth/data"


# ---------------------------------------------------------------------------
# 1. The scoring path must not read the Claude-digitized gold set.
# ---------------------------------------------------------------------------

def _refs(arxiv_id: str) -> list[str]:
    return [e.get("reference_repo_file") for e in PAPERS.get(arxiv_id, [])]


def test_scoring_modules_do_not_read_gold():
    # The two modules that score the pipeline (subset comparator + CI gate) must
    # never reference the Claude-digitized gold set.
    for mod in ("evaluation/subset_compare.py", "evaluation/gate.py"):
        src = (PROJECT_ROOT / mod).read_text().lower()
        assert "gold/gold.json" not in src and "gold_diff" not in src, (
            f"{mod} references the Claude-digitized gold set — GT must be "
            f"O'Hare repo data only (see feedback_ground_truth_definition)."
        )


def test_gold_set_is_claude_digitized_and_thus_excluded():
    # Documents WHY gold.json is barred from scoring: it is model-digitized.
    gold = PROJECT_ROOT / "evaluation/ground_truth/gold/gold.json"
    if not gold.exists():
        return  # nothing to guard
    meta = json.loads(gold.read_text())
    model = (meta.get("digitize_model") or "").lower()
    assert "claude" in model or "gpt" in model or "opus" in model or "sonnet" in model, (
        "gold.json digitize_model changed; re-confirm it is model-digitized "
        "(and therefore must stay out of the scoring path)."
    )


# ---------------------------------------------------------------------------
# 2. Every scored benchmark GT curve must BE an O'Hare repo limit_data file.
# ---------------------------------------------------------------------------

def _sig(path: Path):
    """(n_rows, min_y, max_y) signature of a 2-column data file, or None."""
    try:
        a = np.loadtxt(str(path), ndmin=2)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] == 0:
        return None
    return (a.shape[0], float(a[:, 1].min()), float(a[:, 1].max()))


def _sig_of_lines(lines: list[str]):
    """Signature of data lines as produced by ``_ingest_reference_file``."""
    import io
    try:
        a = np.loadtxt(io.StringIO("\n".join(lines)), ndmin=2)
    except Exception:
        return None
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] == 0:
        return None
    return (a.shape[0], float(a[:, 1].min()), float(a[:, 1].max()))


def test_union_gt_data_matches_an_ohare_repo_file():
    # For every union entry with a local GT data file, its content must equal
    # the DETERMINISTIC ingestion of that entry's own O'Hare
    # reference_repo_file (limit_data/... run through the header-declared unit
    # conversions in evaluation.ground_truth._ingest_reference_file — the
    # lambda[m]→eV x-axis and the per-file y-scales). A mismatch means
    # foreign/Claude/hand-built data leaked into the GT — forbidden.
    from evaluation.ground_truth import _ingest_reference_file

    foreign = []
    checked = 0
    for aid in SUBSET["union"]:
        for e in PAPERS.get(aid, []):
            f = e.get("ground_truth_data_file")
            r = e.get("reference_repo_file")
            if not f or not r:
                continue
            local = DATA_DIR / f
            rp = PROJECT_ROOT / r
            if not local.exists() or not rp.exists():
                continue
            sl = _sig(local)
            if sl is None:
                continue
            checked += 1
            expected = _sig_of_lines(_ingest_reference_file(rp, r))
            if sl != expected:
                foreign.append((aid, sl, expected, Path(r).name))
    assert checked > 0, "no GT data files found — subset/paths broken?"
    assert not foreign, (
        "benchmark GT data files that do NOT match the deterministic ingestion "
        f"of their O'Hare repo file (foreign/Claude/hand-built data is forbidden): {foreign}"
    )


def test_reference_repo_files_live_under_limit_data():
    # Every scored entry's reference must be an O'Hare repo file, never gold/.
    bad = []
    for aid in SUBSET["union"]:
        for r in _refs(aid):
            if r and not r.startswith("limit_data/"):
                bad.append((aid, r))
    assert not bad, f"reference_repo_file outside limit_data/ (must be O'Hare repo data): {bad}"
