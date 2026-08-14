"""Classify extraction snapshots and error messages for benchmark drivers.

Two lessons from operating the benchmark are encoded here so every runner
shares them instead of re-learning them:

* **Husk vs error (2026-07-30 / 2026-08-05).** A *husk* is a stochastic empty
  extraction: ``data_source`` absent or ``'none'``, zero points, and NO error
  field. A snapshot carrying ``status == 'error'`` or a non-empty ``error`` is
  an ERROR, not a husk. A triage loop that lumps the two together will
  silently delete-and-retry genuine errors forever (it cost ~3h of no-op
  attempts on 2026-08-05) or, worse, count error stubs as coverage.

* **Environmental errors are a property of the run, not the paper (#648).**
  Quota windows, rate limits, overload, and billing failures must never
  consume a per-paper retry budget or mark a paper failed; the run should
  back off (or abort) and the paper be retried for free later.
"""
from __future__ import annotations

# Substrings identifying availability/quota failures. Matched case-insensitively.
ENVIRONMENTAL_MARKERS = (
    "credit balance",
    "billing",
    "usage limit",
    "session limit",
    "rate limit",
    "overloaded",
    "529",
)


def is_environmental_error(message: str | None) -> bool:
    """True if the error text describes run-level availability, not the paper."""
    if not message:
        return False
    m = str(message).lower()
    return any(marker in m for marker in ENVIRONMENTAL_MARKERS)


def classify_snapshot(snapshot: dict) -> str:
    """Classify a saved extraction snapshot.

    Returns one of:
      ``'error'`` — carries ``status == 'error'`` or a non-empty ``error``;
      ``'husk'``  — empty extraction with NO error (data_source none/absent,
                    zero points);
      ``'good'``  — everything else (including legitimately sparse reads).
    """
    if snapshot.get("status") == "error" or snapshot.get("error"):
        return "error"
    source = snapshot.get("data_source")
    n_points = snapshot.get("num_points")
    points = snapshot.get("data_points") or []
    if source in (None, "none") and n_points in (None, 0) and len(points) == 0:
        return "husk"
    return "good"
