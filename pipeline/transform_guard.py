"""Fail-safe transform contract — P0 of the figure-extraction roadmap (#568 / #566).

Every metrology/correction transform in the extractor (vision calibration,
range/unit snap, and — once P1/P2 land — CV axis calibration and curve-trace
override) must declare a *candidate* output. This module scores that candidate
against objective, paper-internal consistency checks and accepts it **only if it
violates no reject rule**; otherwise the pre-transform state is restored
(``revert``). This turns the pipeline's blind ``overwrite-and-trust`` layers into
``propose -> score -> commit-or-revert`` envelopes, so no single layer can make a
result worse on its own consistency signal or emit physically impossible output.

Pure Python (``math`` only) — no new dependencies. This is the **single source of
truth** for the consistency thresholds (R1-R5) and the ``quality()`` ordering
shared across the roadmap:

* P1 plugs its OCR-tick / geometric-fit agreement into the ``corroborated`` slot
  of :func:`guard_transform` for axis overrides.
* P2 imports :func:`quality` / :class:`ConsistencyScore` and extends the 5-tuple
  to a multi-candidate selector (P0's two-way "augment-not-override" is the
  degenerate 2-candidate case).
* P3's anchor / convention-normalizer changes are *gated by* the R5 hard floor
  here, so a wrong anchor can no longer emit out-of-range data.
* P4 asserts the P0 invariants (no R5 violations; never raises) as CI checks.

Thresholds are calibrated from the #550+#561 subset eval; see
``evaluation/eval_runs/roadmap_design/P0_failsafe_contract.md`` §2.3 for the
evidence anchoring each band. All thresholds are in dex (log10) so they compare
orders of magnitude, not raw values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Reject thresholds (calibrated from evidence — see P0_failsafe_contract.md §2.3)
# ---------------------------------------------------------------------------

# R1 — spot-check verify/stage2 ratio must lie in this (broad) band to even be
# *considered*. Catastrophic blow-ups (2102.08764 ratio 1.6e10; 2207.11968
# ratio 3.3e26) fall outside and hard-reject the candidate.
R1_SPOTCHECK_BAND: tuple[float, float] = (1e-2, 1e2)

# R3 — benchmark expected/reported ratio must lie within this tight band
# (<=0.48 dex) for a benchmark-driven correction to commit. A full-decade
# benchmark disagreement (2008.10141 ratio 0.16 = 0.80 dex) means the curve is
# mis-scaled, not that a x0.1 fix is warranted, so it is reverted; a near-1 ratio
# (before-run 1.4 = 0.15 dex) idles correctly.
R3_BENCHMARK_BAND: tuple[float, float] = (1.0 / 3.0, 3.0)

# A benchmark/spot ratio inside this band positively *corroborates* an axis
# override (same band as R3 by construction — 0.48 dex).
CORROBORATION_BAND: tuple[float, float] = (1.0 / 3.0, 3.0)

# R2 — a CV axis endpoint disagreeing with the LLM axis by more than this many
# dex *without corroboration* is reverted to the LLM axis. The threshold sits
# ABOVE the genuine 0.62-dex win (2402.12892, x-max 30->124, corroborated) and
# below the catastrophic 12-24 dex blow-ups (1907.05475 ~12 dex; 1506.08082
# ~24 dex). Any override beyond R2_AXIS_TRIGGER_DEX requires corroboration.
R2_AXIS_TRIGGER_DEX: float = 0.5   # below this, an override needs no corroboration
R2_AXIS_REVERT_DEX: float = 1.0    # above this, corroboration is mandatory

# R4 — degenerate curve trace. All couplings within this many dex => a
# floor-pinned constant line; a curve candidate spanning fewer than this many
# mass decades is likewise degenerate (1905.13650: 293 points at constant
# log10 y = -8.9258 over a 0.997-dex x-span).
R4_YCONST_DEX: float = 0.05
R4_MIN_SPAN_DEX: float = 1.0

# Source-tier ranking for quality(): semantics-trust. A table read is trusted
# over a text point-limit, which is trusted over a figure-vision / CV trace, so a
# CV trace (tier 1) can never override a text point-limit (tier 2).
SOURCE_TIER: dict[str, int] = {
    "table": 3,
    "text": 2,
    "figure_vision": 1,
    "vision": 1,
}


def _median(values) -> float:
    """Median over *sorted* values (order-independent by construction).

    Mirrors ``extractor._sorted_median`` but kept local so this module has no
    import dependency on the extractor (P2/P3 import *from* here, not vice
    versa). Returns ``0.0`` for an empty sequence.
    """
    s = sorted(v for v in values)
    if not s:
        return 0.0
    return s[len(s) // 2]


def in_valid_ranges(data_points, valid_for_ct) -> bool:
    """True iff the candidate's median mass and coupling lie in strict ranges.

    ``valid_for_ct`` is a ``VALID_RANGES[ct]`` mapping ``{"mass": (lo, hi),
    "coupling": (lo, hi)}``. Uses the strict window — NOT the x0.1/x10 widened
    window used for *choosing* a correction factor. Empty or unknown inputs are
    treated as in-range (the floor only rejects positively out-of-range data).
    """
    if not data_points or not valid_for_ct:
        return True
    masses = [float(m) for m, _ in data_points if float(m) > 0]
    couplings = [float(g) for _, g in data_points if float(g) > 0]
    if not masses or not couplings:
        return True
    mass_lo, mass_hi = valid_for_ct["mass"]
    coup_lo, coup_hi = valid_for_ct["coupling"]
    mm, mc = _median(masses), _median(couplings)
    return (mass_lo <= mm <= mass_hi) and (coup_lo <= mc <= coup_hi)


def span_dex(masses) -> float:
    """log10(mass_hi / mass_lo) over the positive masses (0.0 if degenerate)."""
    pos = [float(m) for m in masses if float(m) > 0]
    if len(pos) < 2:
        return 0.0
    lo, hi = min(pos), max(pos)
    if lo <= 0:
        return 0.0
    return math.log10(hi / lo)


def couplings_y_const(couplings) -> bool:
    """True iff all positive couplings agree within ``R4_YCONST_DEX`` (flat line)."""
    pos = [float(g) for g in couplings if float(g) > 0]
    if len(pos) < 2:
        return False
    logs = [math.log10(g) for g in pos]
    return (max(logs) - min(logs)) <= R4_YCONST_DEX


@dataclass(frozen=True)
class ConsistencyScore:
    """Paper-internal consistency signals for a candidate transform output.

    Every field is *optional* (``None``) except :attr:`in_valid_ranges`: a rule
    is evaluated only when its driving field is populated, so the same contract
    serves axis-override, curve-trace, calibration, and range-snap candidates
    without one candidate's irrelevant fields tripping another's rule.

    * :attr:`in_valid_ranges` — HARD floor (R5): candidate median in VALID_RANGES.
    * :attr:`benchmark_ratio` — expected/reported on a known model line (R3).
    * :attr:`spotcheck_ratio` — verify/stage2 at the spot-check mass (R1).
    * :attr:`axis_disagree_dex` — max |log10(CV axis) - log10(LLM axis)| over
      endpoints (R2).
    * :attr:`span_dex` / :attr:`y_const` — curve-degeneracy signals (R4).
    * :attr:`n_points` — informational; last tie-break in :func:`quality`.
    """

    in_valid_ranges: bool
    benchmark_ratio: float | None = None
    spotcheck_ratio: float | None = None
    axis_disagree_dex: float | None = None
    n_points: int = 0
    span_dex: float | None = None
    y_const: bool = False


def _in_band(value: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= value <= hi


def passes_contract(score: ConsistencyScore, *, corroborated: bool = False) -> tuple[bool, str]:
    """Return ``(accept, reason)``. Accept iff the candidate violates no rule.

    Rules are checked in order of severity. The first violation returns
    ``(False, reason)``; if none fire, returns ``(True, "ok")``. A rule whose
    driving field is ``None`` is skipped (not applicable to this candidate).

    ``corroborated`` is the external agreement signal (P1's OCR-tick match, or —
    pre-P1 — a passing spot-check / r2>=0.95 axis fit). It only affects R2.
    """
    # R5 — HARD floor, no corroboration escape hatch.
    if not score.in_valid_ranges:
        return False, "R5 hard floor: median outside VALID_RANGES"

    # R1 — spot-check blow-up.
    if score.spotcheck_ratio is not None:
        if score.spotcheck_ratio <= 0 or not _in_band(score.spotcheck_ratio, R1_SPOTCHECK_BAND):
            return False, f"R1 spot-check ratio {score.spotcheck_ratio:.3g} outside {R1_SPOTCHECK_BAND}"

    # R3 — benchmark disagreement.
    if score.benchmark_ratio is not None:
        if score.benchmark_ratio <= 0 or not _in_band(score.benchmark_ratio, R3_BENCHMARK_BAND):
            return False, f"R3 benchmark ratio {score.benchmark_ratio:.3g} outside {R3_BENCHMARK_BAND}"

    # R2 — axis override without corroboration.
    if score.axis_disagree_dex is not None:
        d = abs(score.axis_disagree_dex)
        if d > R2_AXIS_TRIGGER_DEX and not corroborated:
            return False, (
                f"R2 axis disagreement {d:.2f} dex > {R2_AXIS_TRIGGER_DEX} "
                f"without corroboration"
            )

    # R4 — degenerate trace.
    if score.y_const:
        return False, f"R4 degenerate trace: couplings constant within {R4_YCONST_DEX} dex"
    if score.span_dex is not None and score.span_dex < R4_MIN_SPAN_DEX:
        return False, f"R4 degenerate trace: x-span {score.span_dex:.3f} dex < {R4_MIN_SPAN_DEX}"

    return True, "ok"


def guard_transform(before, after, *, score_after: ConsistencyScore,
                    corroborated: bool = False, label: str = "transform"):
    """Commit ``after`` iff it passes the contract, else revert to ``before``.

    Returns ``(committed_value, note)``. Pure and **never raises** — any internal
    error reverts to ``before`` with an explanatory note, so the contract layer
    can wrap a transform that itself might throw.
    """
    try:
        accept, reason = passes_contract(score_after, corroborated=corroborated)
    except Exception as e:  # defensive: a guard must never break the pipeline
        return before, f"{label}: reverted (contract error: {e})"
    if accept:
        return after, f"{label}: committed"
    return before, f"{label}: reverted ({reason})"


def quality(*, source: str, in_valid_ranges: bool, corroborated: bool,
            confidence: float, n_points: int) -> tuple:
    """Ordered quality tuple for comparing extraction candidates.

    Higher tuples are better. Quality is decided by *semantics-trust first*, not
    raw point count: a CV trace (``source_tier=1``) can never outrank a text
    point-limit (``source_tier=2``) on point count alone — exactly the
    point-count routing bug (``len(traced) >= max(8, stage1_points)``) P0
    replaces. ``n_points`` is only the final tie-break.

    P2 extends this 5-tuple to a 7-tuple; the two must not diverge — there is
    exactly one ``quality()`` and P0's :func:`guard_transform` uses this subset.
    """
    return (
        SOURCE_TIER.get(source, 1),
        1 if in_valid_ranges else 0,
        1 if corroborated else 0,
        float(confidence or 0.0),
        int(n_points or 0),
    )
