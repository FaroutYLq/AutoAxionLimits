"""Independent LLM labeler for evaluation ground-truth scalar labels.

Most ground-truth entries in ``evaluation/ground_truth/papers.json`` were
auto-expanded from the repo's docs (#534) and carry *placeholder* scalar labels:

    is_new_limit=True, is_projection=False, data_source_expected="table",
    difficulty="medium"

Scoring the extraction pipeline against those placeholders measured the
placeholder, not the pipeline, so #535 stopped scoring them (they show N/A).
This script generates **real** labels for those fields.

Independence (this is the whole point — see issue #539)
-------------------------------------------------------
Ground truth must be independent of the extractor it grades. The extractor
(``pipeline/extractor.py``) emits ``is_projection`` / ``data_source`` /
``is_new_limit`` only as a *side effect* of extracting a limit curve. This
labeler is a DISTINCT pass:

  * a different prompt whose PRIMARY task is "classify this paper's properties",
  * read the whole paper and adjudicate (no curve extraction),
  * a different, stronger model (``claude-opus-4-5`` vs the extractor's haiku),
  * it does NOT call ``run_extraction_agent``.

Scoring the extractor against these labels is therefore a fair test, not
self-agreement.

``difficulty`` is derived MECHANICALLY from the (LLM) ``data_source_expected``
and the number of ground-truth data points (see ``derive_difficulty``); we do
not ask the LLM for it.

Outputs (written back into papers.json, idempotent / re-runnable):
  * the four label fields (is_new_limit, is_projection, data_source_expected,
    difficulty),
  * ``verified_by = "llm_labeler:<model-id>"``,
  * the ``auto_expanded`` tag is removed (so the evaluator's non-placeholder
    filter lights the entry up). All other fields (including the
    coupling_convention / coupling_units added in #538) are preserved verbatim.

Usage:
    # Smoke / inspect without writing the file (still calls the API):
    python -m evaluation.label_ground_truth --dry-run --limit 5

    # Label a single paper:
    python -m evaluation.label_ground_truth --arxiv-id 2208.03183

    # Label a test batch of 40 papers (spanning coupling types):
    python -m evaluation.label_ground_truth --limit 40

    # Label everything still carrying placeholder labels:
    python -m evaluation.label_ground_truth

Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the pipeline / evaluator paper-fetch + text-extraction helpers rather
# than reinventing them (issue #539 requirement).
from pipeline.extractor import (  # noqa: E402
    _call_with_retry,
    _sanitize_pdf_text,
    download_pdf,
    extract_text_from_pdf,
)

logger = logging.getLogger(__name__)

PAPERS_JSON = PROJECT_ROOT / "evaluation" / "ground_truth" / "papers.json"

# Independent labeler model: distinct from (and stronger than) the extractor's
# claude-haiku-4-5, so scoring the extractor against these labels is a fair
# cross-model test rather than self-agreement.
LABELER_MODEL = "claude-opus-4-5-20251101"

VALID_DATA_SOURCES = {"table", "text", "figure_vision"}

LABELER_SYSTEM = (
    "You are a scientific-literature classifier for a dark-matter / axion "
    "constraint database. Your ONLY job is to read a physics paper and "
    "adjudicate a small set of properties about the limit it reports. You do "
    "NOT extract any numbers, curves, or data points. Be a careful, skeptical "
    "reader: base every answer on what the paper actually says, and respond "
    "with strict JSON only."
)

LABELER_PROMPT = """\
Read the paper below and classify the PRIMARY new constraint on a dark-matter
or axion-like coupling that it reports. Answer about the paper as a whole.

Return ONLY a JSON object with exactly these keys:

  "is_new_limit": boolean
      true  -> the paper reports a NEW experimental/observational EXCLUSION
               limit (an actual measured constraint, even if it is a
               re-analysis of existing data).
      false -> the paper reports ONLY a projected/forecast/sensitivity curve
               for a future or hypothetical experiment, or only a theory
               prediction, with no realized measured limit.

  "is_projection": boolean
      true  -> the headline curve is a PROJECTION / forecast / sensitivity
               reach (future experiment, "expected", "projected sensitivity").
      false -> the headline curve is a realized measured exclusion limit.
      (is_projection is essentially the logical opposite of "this is a real
       measured limit"; a paper can occasionally show both, in which case
       answer for the PRIMARY result the paper is about.)

  "data_source_expected": one of "table" | "text" | "figure_vision"
      Where in the paper would a reader most naturally obtain the numeric
      limit curve?
        "table"        -> the limit is given as an explicit data TABLE of
                          (mass, coupling) values.
        "text"         -> the key numbers are stated inline in prose (e.g. a
                          single bound quoted in a sentence), not tabulated.
        "figure_vision"-> the limit exists only as a PLOTTED curve in a figure,
                          with no table and no inline numeric listing, so it
                          would have to be read off the plot.

  "reasoning": a one-sentence justification (<= 240 chars).

Known metadata (use only as weak context; judge from the paper text):
  coupling_type (database label): {coupling_type}
  title: {title}

Respond with the JSON object and nothing else.

===PAPER_CONTENT===
{paper_text}
===END_PAPER_CONTENT===
"""


def derive_difficulty(data_source: str, num_points: int | None) -> str:
    """Mechanically derive difficulty from data source + GT curve size.

    Rationale: extraction difficulty is driven by (a) where the data lives and
    (b) how much of it there is. Reading a tabulated/inline curve is easy;
    tracing a curve off a plot is hard; a sparse figure is the hardest. We do
    NOT ask the LLM for this (keeps difficulty deterministic and auditable).

    Rule:
      * figure_vision  -> "hard"  if few points (< 15), else "medium"
      * table / text   -> "easy"  if many points (>= 15), else "medium"
      * anything else  -> "medium"
    """
    n = num_points if isinstance(num_points, int) else 0
    if data_source == "figure_vision":
        return "hard" if n < 15 else "medium"
    if data_source in ("table", "text"):
        return "easy" if n >= 15 else "medium"
    return "medium"


def _is_placeholder(entry: dict) -> bool:
    """An entry still carrying auto-generated placeholder scalar labels."""
    return ("auto_expanded" in (entry.get("tags") or [])) or (
        entry.get("verified_by") == "repo_upstream"
    )


def _parse_json_response(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def label_paper(
    client, arxiv_id: str, coupling_type: str, title: str
) -> dict:
    """Run the independent classification pass on one paper.

    Returns {is_new_limit, is_projection, data_source_expected, reasoning}.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = download_pdf(arxiv_id, Path(tmpdir))
        raw = extract_text_from_pdf(pdf_path)
    paper_text = _sanitize_pdf_text(raw)
    if not paper_text.strip():
        raise ValueError("empty paper text after extraction/sanitization")

    prompt = LABELER_PROMPT.format(
        coupling_type=coupling_type,
        title=title or "(unknown)",
        paper_text=paper_text,
    )

    def _do_call():
        return client.messages.create(
            model=LABELER_MODEL,
            max_tokens=1024,
            system=LABELER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = _call_with_retry(_do_call)
    out = _parse_json_response(resp.content[0].text)

    is_new_limit = bool(out["is_new_limit"])
    is_projection = bool(out["is_projection"])
    data_source = str(out["data_source_expected"]).strip()
    if data_source not in VALID_DATA_SOURCES:
        raise ValueError(f"invalid data_source_expected: {data_source!r}")

    return {
        "is_new_limit": is_new_limit,
        "is_projection": is_projection,
        "data_source_expected": data_source,
        "reasoning": str(out.get("reasoning", ""))[:240],
    }


def _select_targets(papers: list[dict], arxiv_id: str | None, limit: int | None):
    """Pick entries to label.

    Default: every entry still carrying placeholder labels, ordered to span
    coupling types first (so a --limit batch is a diverse cross-section rather
    than one coupling). With --arxiv-id, restrict to that paper.
    """
    if arxiv_id:
        return [p for p in papers if p.get("arxiv_id") == arxiv_id]

    pending = [p for p in papers if _is_placeholder(p)]

    if limit is None:
        return pending

    # Round-robin across coupling types so a small batch is diverse and
    # spans table/figure sources.
    from collections import OrderedDict, deque

    buckets: "OrderedDict[str, deque]" = OrderedDict()
    for p in pending:
        buckets.setdefault(p.get("coupling_type", "?"), deque()).append(p)

    ordered: list[dict] = []
    while buckets and len(ordered) < limit:
        for ct in list(buckets.keys()):
            if not buckets[ct]:
                del buckets[ct]
                continue
            ordered.append(buckets[ct].popleft())
            if len(ordered) >= limit:
                break
    return ordered


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arxiv-id", type=str, default=None,
                    help="Only label this arXiv ID.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Label at most N (placeholder) entries, spanning "
                         "coupling types.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be labeled and don't write the file. "
                         "Still calls the API so labels can be inspected; use "
                         "--limit to keep it cheap.")
    args = ap.parse_args()

    doc = json.loads(PAPERS_JSON.read_text())
    papers = doc["papers"]

    targets = _select_targets(papers, args.arxiv_id, args.limit)
    if not targets:
        logger.error("No matching entries to label.")
        return 1

    # One API call per distinct arXiv id; a paper may have several entries
    # (one per coupling/file) that all share the same paper-level labels.
    by_id: dict[str, list[dict]] = {}
    for p in targets:
        by_id.setdefault(p["arxiv_id"], []).append(p)

    logger.info(
        "Labeling %d entries across %d papers with %s",
        len(targets), len(by_id), LABELER_MODEL,
    )

    import anthropic

    client = anthropic.Anthropic()

    labeled_papers = 0
    entries_affected = 0
    failures: list[tuple[str, str]] = []

    for arxiv_id, entries in by_id.items():
        rep = entries[0]
        title = rep.get("paper_title", "")
        coupling = rep.get("coupling_type", "")
        try:
            labels = label_paper(client, arxiv_id, coupling, title)
        except Exception as e:  # noqa: BLE001 — keep batch going
            logger.error("Labeling failed for %s: %s", arxiv_id, e)
            failures.append((arxiv_id, str(e)))
            continue

        for entry in entries:
            npts = entry.get("ground_truth_num_points")
            difficulty = derive_difficulty(labels["data_source_expected"], npts)

            logger.info(
                "%s [%s] new_limit=%s projection=%s source=%s -> %s  (%s)",
                arxiv_id, entry.get("coupling_type"),
                labels["is_new_limit"], labels["is_projection"],
                labels["data_source_expected"], difficulty,
                labels["reasoning"],
            )

            if args.dry_run:
                continue

            entry["is_new_limit"] = labels["is_new_limit"]
            entry["is_projection"] = labels["is_projection"]
            entry["data_source_expected"] = labels["data_source_expected"]
            entry["difficulty"] = difficulty
            entry["verified_by"] = f"llm_labeler:{LABELER_MODEL}"
            # Light up the evaluator's non-placeholder filter.
            entry["tags"] = [t for t in (entry.get("tags") or [])
                             if t != "auto_expanded"]
            if "llm_labeled" not in entry["tags"]:
                entry["tags"].append("llm_labeled")
            entries_affected += 1

        labeled_papers += 1
        time.sleep(1)  # be nice to the API

    print(f"\nPapers labeled:   {labeled_papers}/{len(by_id)}")
    print(f"Entries affected: {entries_affected}")
    if failures:
        print(f"Failures:         {len(failures)}")
        for aid, err in failures:
            print(f"  {aid}: {err}")

    if args.dry_run:
        print("\n[dry-run] no file written.")
        return 0

    PAPERS_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nWrote {PAPERS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
