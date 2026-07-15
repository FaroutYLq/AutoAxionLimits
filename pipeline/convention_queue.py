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
# Verdict for convention classes with NO constant conversion by established
# physics (documented, not awaiting derivation) — the drain skill skips these
# instead of burning a GPD derivation that must FAIL. The [CONVENTION REVIEW]
# cap on the PR still stands; only the queue disposition is pre-decided.
STATUS_UNCONVERTIBLE = "unconvertible"

# Known-unconvertible classes: (coupling_type, declaration-token tuple).
# NOTE (Phase 2, #625): the AxionEDM e*cm class was REMOVED — the oscillating-
# EDM amplitude d_n/d_AC [e*cm] is now converted by the mass-dependent d_n_ecm
# registry converter (g_angamma = C * d_n[e*cm] * m_a[eV], via a_0 =
# sqrt(2 rho)/m_a; note convention-dn-ecm-g_angamma.md), so those papers are
# convertible, not pre-verdicted unconvertible. No known-unconvertible class
# remains; the dict is kept for future documented no-conversion classes.
_KNOWN_UNCONVERTIBLE: dict = {}


def known_unconvertible(coupling_type: str | None, declared: str | None) -> bool:
    """True when the declared convention belongs to a documented
    no-constant-conversion class for this coupling type."""
    if not coupling_type or not declared:
        return False
    d = str(declared).lower()
    if "converted" in d:
        return False
    tokens = _KNOWN_UNCONVERTIBLE.get(coupling_type, ())
    return any(t in d for t in tokens)


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
    counter-bumped, never reopened. Idempotent per (cache_key, arxiv_id).

    Tokens in a documented no-constant-conversion class enter directly as
    ``STATUS_UNCONVERTIBLE`` so the drain never queues a derivation that must
    fail; an existing entry's status is never changed here (drain owns it)."""
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
        "status": (STATUS_UNCONVERTIBLE
                   if known_unconvertible(coupling_type, declared_convention)
                   else STATUS_QUEUED),
    }
    queue["entries"].append(entry)
    save_queue(queue, path)
    return entry


# --- Blind-spot #1 (design §1 prerequisite): undeclared + suspicious values --
# An EMPTY declaration is deliberately treated as canonical by
# convention_review_needed (the common case must not flag), so a paper whose
# values are in a novel convention but whose declaration is empty passes
# silently today. When the emitted couplings sit far above the type's
# physically-reasonable ceiling (novel conventions store LARGE values: the
# d_e_large scalar files ~1e30, e*cm amplitudes ~1e-22 vs g_d ~1e-30), that
# magnitude alone is the review signal. High side only, wide margin — a
# strong (small) limit is never suspicious.
_SUSPICIOUS_MARGIN_DEX = 3.0

# The dedup token for undeclared-suspicious firings: one queue entry per
# coupling type (the declaration is empty, so there is no token to group by).
UNDECLARED_TOKEN = "<undeclared: suspicious magnitude>"


def undeclared_suspicious(coupling_type: str | None, declared: str | None,
                          data_points, valid_ranges: dict | None) -> bool:
    """True when the declaration is empty/canonical-claimed BUT the median
    emitted coupling exceeds the type's VALID_RANGES ceiling by more than
    ``_SUSPICIOUS_MARGIN_DEX`` decades. Pure and deterministic."""
    if declared and str(declared).strip().lower() not in (
            "", "canonical", "standard", "none", "n/a"):
        return False  # a real declaration goes through convention_review_needed
    if not coupling_type or not valid_ranges:
        return False
    rng = (valid_ranges.get(coupling_type) or {}).get("coupling")
    if not rng:
        return False
    vals = sorted(float(g) for _, g in (data_points or ()) if float(g) > 0)
    if not vals:
        return False
    median = vals[len(vals) // 2]
    ceiling = float(rng[1]) * (10.0 ** _SUSPICIOUS_MARGIN_DEX)
    return median > ceiling


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
