"""Convention-escalation queue (issue #636, design in
``pipeline/DESIGN_convention_escalation.md``).

When production extraction flags a paper ``[CONVENTION REVIEW]`` — the declared
coupling convention is outside the vetted vocabulary and cannot be converted at
runtime — the token is appended here so a human-invoked GPD drain skill
(``.claude/skills/convention-triage``) can derive the conversion offline, once
per convention token, and promote it into the registry via a reviewable PR.

Production side is deterministic and cheap: one JSON append (or counter bump)
per flagged paper, deduplicated by ``cache_key`` (normalized declaration +
coupling type). Nothing here converts or guesses — GPD never runs inside a
production extraction. Same git-tracked lifecycle as ``processed.json``: the
daily/weekly workflow restores it from the state branch (#547) and force-pushes
the updated snapshot back.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# Default is the git-tracked state file; AAL_CONVENTION_QUEUE overrides it so a
# benchmark/behavioral run can isolate its writes (mirrors AAL_RESULTS_DIR).
QUEUE_PATH = Path(os.environ.get("AAL_CONVENTION_QUEUE")
                  or (Path(__file__).parent / "state" / "convention_queue.json"))

_MAX_SAMPLE_POINTS = 5

# Valid entry lifecycle states.
STATUS_QUEUED = "queued"
STATUS_NEEDS_HUMAN = "needs_human"
STATUS_PROMOTED = "promoted"


def _empty_queue() -> dict:
    return {"version": 1, "entries": []}


def load_queue(path: Path = QUEUE_PATH) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                q = json.load(f)
            if isinstance(q, dict) and isinstance(q.get("entries"), list):
                return q
        except (json.JSONDecodeError, OSError):
            pass
    return _empty_queue()


def save_queue(queue: dict, path: Path = QUEUE_PATH) -> None:
    """Atomic write via .tmp rename (mirrors monitor.save_state)."""
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(queue, f, indent=2)
    tmp.replace(path)


def normalize_declaration(declared: str | None) -> str:
    """Reduce a free-text convention declaration to a stable dedup token:
    lowercase, punctuation dropped, whitespace collapsed. Deterministic and
    order-preserving so the same declaration always maps to the same key."""
    text = (declared or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def cache_key(coupling_type: str | None, declared: str | None) -> str:
    """Dedup unit: normalized declaration scoped by coupling type."""
    return f"{(coupling_type or '').lower()}::{normalize_declaration(declared)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_flag(
    *,
    coupling_type: str | None,
    declared_convention: str | None,
    arxiv_id: str | None,
    sample_points=None,
    target_repo_file: str | None = None,
    pr_url: str | None = None,
    path: Path = QUEUE_PATH,
    now: str | None = None,
) -> dict:
    """Append a ``[CONVENTION REVIEW]`` firing to the queue, or bump the counter
    of an existing entry with the same ``cache_key``. Returns the (new or
    updated) entry. A token already ``promoted``/``needs_human`` is only
    counter-bumped, never reopened. Idempotent per (cache_key, arxiv_id)."""
    key = cache_key(coupling_type, declared_convention)
    ts = now or _now_iso()
    queue = load_queue(path)
    for entry in queue["entries"]:
        if entry.get("cache_key") == key:
            entry["count"] = int(entry.get("count", 1)) + 1
            entry["last_seen"] = ts
            seen = entry.setdefault("arxiv_ids", [])
            if arxiv_id and arxiv_id not in seen:
                seen.append(arxiv_id)
            save_queue(queue, path)
            return entry
    pts = [[float(m), float(g)] for m, g in (sample_points or [])][:_MAX_SAMPLE_POINTS]
    entry = {
        "cache_key": key,
        "coupling_type": coupling_type,
        "declared_convention": declared_convention,
        "arxiv_id": arxiv_id,
        "arxiv_ids": [arxiv_id] if arxiv_id else [],
        "sample_points": pts,
        "target_repo_file": target_repo_file,
        "pr_url": pr_url,
        "first_seen": ts,
        "last_seen": ts,
        "count": 1,
        "status": STATUS_QUEUED,
    }
    queue["entries"].append(entry)
    save_queue(queue, path)
    return entry


def record_convention_flag(
    coupling_type: str | None,
    declared_convention: str | None,
    arxiv_id: str | None,
    data_points=None,
    *,
    target_repo_file: str | None = None,
    path: Path = QUEUE_PATH,
) -> None:
    """Thin, exception-swallowing wrapper for the extractor's flag site: queue
    plumbing must never break a production extraction. Sampling ≤5 points gives
    the drain skill a numeric anchor without bloating the state file."""
    try:
        append_flag(
            coupling_type=coupling_type,
            declared_convention=declared_convention,
            arxiv_id=arxiv_id,
            sample_points=data_points,
            target_repo_file=target_repo_file,
            path=path,
        )
    except Exception:  # pragma: no cover - defensive; never fail the extraction
        pass
