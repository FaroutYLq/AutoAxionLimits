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
import re
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
# over a text point-limit, over an LLM figure read, over a raw CV pixel trace, so
# a CV trace can never override a text point-limit on point count alone. `cv_trace`
# is split BELOW `figure_vision` because a pixel trace with no LLM semantic check
# (which-curve/which-panel) is the least-trusted when it ties on validity — it is
# exactly the source that produced the 2102.08764 / 1905.13650 regressions. On the
# current master there is no CV-trace producer; the tier is reserved for when CV
# metrology returns (P1+), the same way P0's R2/R4 ship ahead of their wiring.
#
# The integers are spaced (not 0..3) to leave a slot BETWEEN cv_trace (0) and
# figure_vision (2) for a *sparse* point-limit (see `_SPARSE_POINT_LIMIT_TIER`).
#
# `source_data` (WS1, #665) sits ABOVE `table`: numeric curve files shipped in
# the paper's own e-print (pgfplots .dat / anc/ ancillary data) are the
# authors' published coordinates — deterministic and exact where they exist
# (survey: median 0.055 dex on hits), so no LLM read should outrank them.
# `vector_trace` (WS2) is the LLM-SELECTED deterministic vector-path trace:
# exact geometry from the figure PDF, curve identity chosen by the cheap
# selection call. It slots at 3.5 — above a text point-limit (its geometry is
# measured, not read), below a table (the identity choice is still a model
# judgement). This realises the tier the plan reserved for verified traces.
SOURCE_TIER: dict[str, float] = {
    "source_data": 5,
    "table": 4,
    "vector_trace": 3.5,
    "text": 3,
    "figure_vision": 2,
    "vision": 2,
    "cv_trace": 0,
}

# Sources that are point-limits (a sparse set of (mass, coupling) bounds), not a
# traced curve — they are exempt from the R4 degenerate-shape penalty in quality()
# (a 2-point text limit is legitimately sparse; only a *traced curve* that
# collapsed to a flat line or a single mass decade is degenerate).
_POINT_LIMIT_SOURCES = frozenset({"text", "table"})

# P-A1 (#587): a point-limit (text/table) carrying this many points or fewer is a
# lone headline/benchmark value, NOT a digitized contour. The #1 theme in the
# full-scale failure digest (family A, ~9 papers: 1207.3275, 1907.11485,
# 2110.03679, 2211.12699, 2311.16364, …) was the selector preferring such a
# single — and often physically *correct* — text point over the figure curve the
# ground truth was built from, so it could never overlap a 15-129-point GT curve.
# A sparse point-limit is therefore demoted to a tier BELOW figure_vision (2) but
# still ABOVE an unverified cv_trace (0), which preserves the 2102.08764
# protection (a raw pixel trace must not override a clean text point) while
# letting a traceable figure curve win. The demotion changes the outcome ONLY
# when a *valid, non-degenerate* figure curve is present: T0 (validity) and T1
# (non-degeneracy) rank above T3, so a degenerate or out-of-range vision read
# still loses to the point — the curve wins only when it is trustworthy
# GT-class evidence. A genuine multi-point contour (> _SPARSE_POINT_LIMIT_MAX
# points, e.g. a digitized table) keeps its full source tier.
#
# Lever A (#606): raised 2 -> 3 after the post-roadmap digest found a third cluster
# of sparse-text wins at exactly 3 points (1406.6053, 1808.02340, 2007.04899,
# 1712.00483, 0807.2926, 2204.01454). The demotion stays MONOTONE — it only flips
# a <=3-pt text/table -> figure_vision when a VALID, NON-DEGENERATE figure curve
# exists (T0/T1 outrank T3), so a real 3-row table still beats a degenerate/
# out-of-range vision read; only a trustworthy GT-class curve wins.
_SPARSE_POINT_LIMIT_MAX: int = 3
_SPARSE_POINT_LIMIT_TIER: int = 1   # below figure_vision (2), above cv_trace (0)


def _source_tier(source: str, n_points: int) -> float:
    """Effective T3 source tier, demoting a *sparse* point-limit below figure_vision.

    A text/table candidate with ``<= _SPARSE_POINT_LIMIT_MAX`` points is a lone
    headline/benchmark value, not a digitized contour, so it drops to
    ``_SPARSE_POINT_LIMIT_TIER`` (below figure_vision, above cv_trace). Every other
    source keeps its :data:`SOURCE_TIER` rank. See P-A1 (#587).
    """
    base = SOURCE_TIER.get(source, SOURCE_TIER["figure_vision"])
    if source in _POINT_LIMIT_SOURCES and n_points <= _SPARSE_POINT_LIMIT_MAX:
        return _SPARSE_POINT_LIMIT_TIER
    return base


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
    # Log10 extent of the source FIGURE's x-axis, when stage 2a read it. Lets
    # the R4 span floor scale to the plot: a narrow-band resonance search whose
    # whole figure spans <1 decade cannot be asked for a 1-decade trace
    # (2110.10497 — post-full346 Lever 7). None = unknown -> flat floor.
    axis_extent_dex: float | None = None


@dataclass(frozen=True)
class Candidate:
    """One scored extraction candidate for the best-extraction selector (P2, #571).

    The selector ranks candidates by :func:`quality` — a validity-first ordered
    tuple — so the *source choice* (text vs figure-vision vs CV trace) is decided
    by which read is most valid / consistent / confident, NOT by which produced
    more points. ``recoverable`` (T2) is precomputed by the caller (it needs
    ``VALID_RANGES`` + the decade-factor search, which live in the extractor) so
    this module stays dependency-free.
    """

    source: str
    data_points: tuple
    coupling_type: str | None
    extraction_confidence: float
    score: ConsistencyScore
    recoverable: bool = False
    # The candidate's own notes admit its values were analytically
    # reconstructed / approximately read rather than taken from the paper
    # (post-full346 Lever 6). Demoted below figure_vision in T3 — LLM
    # arithmetic must not outrank a real trace — without renaming ``source``
    # (which flows into the emitted data_source enum).
    reconstruction: bool = False
    # The candidate's effective declared convention failed convention review
    # (unknown / unconvertible). #594 follow-up: values in an unknown
    # convention cannot be used canonically, so a candidate in a known
    # convention outranks them regardless of source (1708.06367: the flagged
    # e*cm text read beat the vision trace of the figure already in g_d).
    convention_flagged: bool = False


def _in_band(value: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= value <= hi


def _corroborated(score: ConsistencyScore) -> bool:
    """True iff a benchmark or spot-check ratio lands in the corroboration band."""
    for r in (score.benchmark_ratio, score.spotcheck_ratio):
        if r is not None and _in_band(r, CORROBORATION_BAND):
            return True
    return False


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
    floor = _r4_span_floor(score)
    if score.span_dex is not None and score.span_dex < floor:
        return False, f"R4 degenerate trace: x-span {score.span_dex:.3f} dex < {floor:.3g}"

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


def _r4_span_floor(score: ConsistencyScore) -> float:
    """Effective R4 minimum x-span, scaled to the figure's own axis extent.

    A faithful full-width trace of a narrow-band haloscope figure spans exactly
    the figure's extent; demanding a fixed 1-dex span rejects it (2110.10497,
    post-full346 Lever 7). When stage 2a read the axis extent, the floor is
    half that extent, capped by the flat R4_MIN_SPAN_DEX; unknown extent keeps
    the flat floor.
    """
    if score.axis_extent_dex is not None and score.axis_extent_dex > 0:
        return min(R4_MIN_SPAN_DEX, 0.5 * score.axis_extent_dex)
    return R4_MIN_SPAN_DEX


def quality(c: Candidate) -> tuple:
    """Validity-first ordered quality tuple for a :class:`Candidate` (P2, #571).

    Higher tuples are better; the selector takes the lexicographic argmax. The
    tiers, in order:

      T0  in_valid_ranges   — HARD validity (P0 R5 floor) promoted to the top.
      T1  non-degenerate     — a *traced curve* that collapsed to a flat line
                               (y_const) or a single mass decade (span < 1) is
                               degenerate; point-limit sources (text/table) are
                               exempt (a 2-point limit is legitimately sparse).
      T2  recoverable        — coupling lands in VALID_RANGES under some decade
                               factor (precomputed by the caller).
      T2b convention known   — a candidate whose effective declared convention
                               failed convention review (unknown/unconvertible)
                               ranks below one in a known convention (#594).
      T3  source tier        — table > text > figure_vision > cv_trace, EXCEPT a
                               *sparse* (<= _SPARSE_POINT_LIMIT_MAX-pt) text/table
                               point-limit is demoted below figure_vision (P-A1,
                               #587), so a lone headline value can no longer
                               outrank a traceable figure curve.
      T4  corroborated       — benchmark/spot-check ratio in [1/3, 3].
      T5  confidence         — the LLM self-confidence (rounded so float jitter
                               cannot reorder near-ties).
      T6  n_points           — raw point count, the ONLY place it appears, as the
                               last resort. This is the gate P0/P2 demote from
                               being the *primary* routing signal.

    Decided by *semantics-trust first, point count last* — the inversion of the
    `len(traced) >= max(8, stage1_points)` / `stage2_points > stage1_points`
    routing bugs. There is exactly one ``quality()``; P0's R-thresholds
    (`R4_*`, `CORROBORATION_BAND`) are reused so P0 and P2 never diverge.
    """
    s = c.score
    point_limit = c.source in _POINT_LIMIT_SOURCES
    non_degenerate = 1 if point_limit else (
        1 if (not s.y_const and (s.span_dex is None or s.span_dex >= _r4_span_floor(s)))
        else 0
    )
    n_pts = int(s.n_points or len(c.data_points) or 0)

    return (
        1 if s.in_valid_ranges else 0,             # T0
        non_degenerate,                            # T1
        1 if c.recoverable else 0,                 # T2
        0 if c.convention_flagged else 1,          # T2b (#594)
        (_SPARSE_POINT_LIMIT_TIER if c.reconstruction
         else _source_tier(c.source, n_pts)),      # T3 (Lever 6: reconstructions rank with sparse points, below figure_vision)
        1 if _corroborated(s) else 0,              # T4
        round(float(c.extraction_confidence or 0.0), 2),  # T5
        n_pts,                                     # T6
    )


def should_consider_vision(text_cand: Candidate | None) -> bool:
    """Whether to spend the vision API call to produce a figure candidate (P2).

    Returns True unless the text candidate is *already clearly dominant* — valid
    window, not sparse (>= 5 points), confident (>= 0.6, the project's
    ``LOW_CONFIDENCE_THRESHOLD``), and non-degenerate. A sparse/low-confidence/
    out-of-range text read (2204.01454 conf 0.42; 1704.05189 / 2007.04899
    conf 0.55) must NOT suppress vision; a strong 5-point text read
    (1808.02340) keeps the one-API-call fast path. This only decides whether to
    *spend the call*, never the winner — the selector picks on merit.
    """
    if text_cand is None or not text_cand.data_points:
        return True
    s = text_cand.score
    strong = (
        s.in_valid_ranges
        and (s.n_points or len(text_cand.data_points)) >= 5
        and float(text_cand.extraction_confidence or 0.0) >= 0.6
        and not s.y_const
    )
    return not strong


def select_best(candidates: list[Candidate]) -> tuple[Candidate | None, str]:
    """Return ``(winner, reason)`` — the validity-first argmax over candidates.

    Pure and deterministic: ties keep the earlier candidate (callers append in a
    stable order — text before figure_vision before cv_trace), and T5 rounding +
    integer T6 prevent float-jitter reorders. Returns ``(None, ...)`` for an empty
    list.
    """
    if not candidates:
        return None, "no candidates"
    best = candidates[0]
    best_q = quality(best)
    for c in candidates[1:]:
        q = quality(c)
        if q > best_q:
            best, best_q = c, q
    if len(candidates) == 1:
        return best, f"selector: {best.source} (sole candidate)"
    others = ", ".join(sorted({c.source for c in candidates if c is not best}))
    return best, f"selector: {best.source} over {others} (quality {best_q})"

# ---------------------------------------------------------------------------
# Coupling-convention normalizer (issue #572, P3) — the 5.6-dex fix
# ---------------------------------------------------------------------------
# A value read CORRECTLY but in the wrong convention can swing the result by many
# decades and neither #561's decade-snap nor P0's R5 floor can catch it (the
# converted value is "in range", just wrong). Example (1902.04246, AxionElectron):
# the LLM reported the raw paper number C_e/F_a = 5.00e-16 eV^-1 instead of the
# canonical dimensionless g_ae; both sit inside VALID_RANGES (1e-20, 1e0), so only
# a convention conversion recovers it: g_ae = 2 m_e (C_e/F_a) = 1.022e6 * 5e-16
# = 5.11e-10, landing exactly on the gold curve.
#
# Scope: the well-defined "dimensionless = 2 * mass * (C/F_a)[eV^-1]" family
# (AxionElectron evidenced; AxionProton/Neutron the same physics form). The
# AxionPhoton eV^-1 <-> GeV^-1 unit rescaling and the AxionEDM convention are
# deliberately NOT auto-converted here (no eval evidence + sign-ambiguity risk).

_M_E_EV = 511000.0          # electron mass [eV] (CODATA) — g_ae = 2 m_e C_e/F_a
_M_P_EV = 9.382720813e8     # proton mass [eV]
_M_N_EV = 9.395654205e8     # neutron mass [eV]

# Inverse-energy unit markers for a "C/F_a"-style (eV^-1) convention. The ``eV``
# must be BARE — not a prefixed unit (GeV/keV/MeV/TeV), which are different energy
# scales. The old substring test (``"ev^-1" in label``) wrongly fired on
# ``"GeV^-1"`` (and keV/MeV), making :func:`normalize_convention` apply a spurious
# 2*m_e / 2*m_N factor (~1e6-1e9) to a GeV^-1 axion-electron/proton/neutron
# coupling — a real misdetection bug (#587 P-B). The negative lookbehind guards
# the prefix while still matching a bare eV^-1 / eV-1 / eV⁻¹ / eV**-1 / 1/eV.
_INV_EV_RE = re.compile(
    r"(?<![gkmt])ev\s*(?:\^?\s*-\s*1|⁻¹|\*\*\s*-\s*1)"   # eV^-1, eV-1, eV⁻¹, eV**-1
    r"|/\s*(?<![gkmt])ev"                                  # 1/eV, /eV
)


def _has_inv_ev(label: str) -> bool:
    return bool(_INV_EV_RE.search((label or "").lower()))


def normalize_convention(coupling_type, data_points, axis_unit_label="", notes=""):
    """Map a coupling reported in a paper-native convention to the repo canonical.

    Pure, deterministic, no LLM. Returns ``(data_points', note)``; ``note`` is ""
    when no conversion applied. Only converts on POSITIVE detection (unit label,
    then a note regex), so it can never introduce a wrong-direction error; the
    result still flows through the downstream R5 floor.

    Detection (deterministic, conservative — convert ONLY on explicit evidence):
      1. the axis unit label contains an eV^-1 token, or
      2. the notes name the inverse-energy convention (``C_e/F_a`` / ``eV^-1``).

    A range-based fallback was considered and rejected: an ultralight-mass
    *experimental limit* (e.g. 1902.04246's g_ae ~ 5e-10 at m_a ~ 1e-20 eV) sits
    many decades ABOVE the DFSZ benchmark line, so a "converted value near the
    benchmark" test never fires, and a bare "median in the eV^-1 band" test would
    wrongly convert a legitimately small dimensionless g_ae. Requiring the stated
    convention keeps the normalizer one-directional and safe (1902.04246's read
    note literally says "C_e/F_a in eV^-1").
    """
    if not data_points or not coupling_type:
        return data_points, ""
    couplings = [float(g) for _, g in data_points if float(g) > 0]
    if not couplings:
        return data_points, ""
    note_l = (notes or "").lower()
    label_inv = _has_inv_ev(axis_unit_label)
    note_inv = _has_inv_ev(notes) or "c_e/f_a" in note_l or "c_n/f_a" in note_l \
        or "c_p/f_a" in note_l or "c_e / f_a" in note_l

    # Magnitude guard (double-conversion audit, 2026-07-16). The notes regex
    # cannot distinguish "my emitted values are C/F_a [eV^-1]" from "the PAPER
    # publishes C/F_a (which I already converted)": 1902.04246 emitted the
    # already-canonical g_ae ~ 2.55e-10 while its notes discussed the paper's
    # Ce/Fa convention, and the blind x2 m_e re-converted a correct value into
    # 2.6e-4 (5.9 dex off). A genuine C/F_a [eV^-1] input is tiny — g/(2m) ~
    # canonical/1e6-1e9 (the founding case 1902.04246 v1 read 5.00e-16) —
    # while an already-canonical coupling sits >= ~1e-12. Convert ONLY when
    # the median is below the C/F_a ceiling; otherwise the values are already
    # dimensionless-scale and conversion would double-apply.
    med = _median(couplings)
    _CFA_INPUT_CEILING = 1e-12
    if med >= _CFA_INPUT_CEILING and (label_inv or note_inv):
        return data_points, ""

    if coupling_type == "AxionElectron":
        if label_inv or note_inv:
            factor = 2.0 * _M_E_EV  # g_ae = 2 m_e (C_e/F_a)
            out = [(m, g * factor) for m, g in data_points]
            return out, (f"convention: AxionElectron C_e/F_a [eV^-1] -> "
                         f"dimensionless g_ae (x{factor:.3e})")

    elif coupling_type in ("AxionProton", "AxionNeutron"):
        if label_inv or note_inv:
            m_n = _M_P_EV if coupling_type == "AxionProton" else _M_N_EV
            factor = 2.0 * m_n
            out = [(m, g * factor) for m, g in data_points]
            return out, (f"convention: {coupling_type} C_N/F_a [eV^-1] -> "
                         f"dimensionless g_aN (x{factor:.3e})")

    return data_points, ""



# ---------------------------------------------------------------------------
# Convention review flag (#536 / #587 runtime hybrid) — escalate-on-UNKNOWN
# ---------------------------------------------------------------------------
# The registry canonicalizes KNOWN conventions deterministically; for an UNKNOWN
# convention we do not silently emit a limit — we flag the PR for human review
# (the cheap escalation half of the hybrid). This is the production safety net.
#
# Canonical / repo-standard convention phrasings per coupling type (a declared
# output convention matching one of these needs no review). Lowercased substring
# match. Kept deliberately conservative: a non-empty declaration is flagged ONLY
# when it is NEITHER canonical NOR a registry-convertible alternate, so the common
# cases (canonical, or a vetted alternate) are never flagged.
_CANONICAL_DECL: dict[str, tuple[str, ...]] = {
    "AxionPhoton":    ("gev^-1", "gev-1", "g_agamma", "g_a\\gamma", "g_agg", "gev$^{-1}$"),
    "AxionElectron":  ("dimensionless", "g_ae"),
    "AxionNeutron":   ("dimensionless", "g_an"),
    "AxionProton":    ("dimensionless", "g_ap"),
    "DarkPhoton":     ("dimensionless", "chi", "kinetic mixing", "epsilon", "eps"),
    # NOTE: e*cm deliberately NOT listed — the repo AxionEDM files are all
    # g_angamma/g_d [GeV^-2] (#604); an oscillating-EDM amplitude in e*cm is
    # UNCONVERTIBLE and must be review-flagged, not silently accepted. (It
    # previously escaped flagging only when spelled "e*cm"; "e cm" slipped
    # through as "canonical" — post-full346 #594 follow-up.)
    "AxionEDM":       ("g_d", "g_angamma", "gev^-2", "gev-2"),
    "AxionCPV":       ("dimensionless", "coupling"),
    # gev^{-1} / f_a^{-1}: LaTeX-braced spellings of the canonical inverse-GeV
    # plane (2302.00685 "f_a^{-1} in GeV^{-1}" was runtime-flagged while the
    # eval registry's inv_gev — which lists the braced variant — treated it as
    # canonical; mirror drift, 2026-07-14).
    "AxionMass":      ("gev^-1", "gev^{-1}", "1/f_a", "f_a^{-1}", "fa^{-1}",
                       "f_a_norm", "dimensionless", "normalized"),
    "MonopoleDipole": ("dimensionless", "coupling"),
    "ScalarPhoton":   ("d_e", "dimensionless"),
    "ScalarElectron": ("d_me", "d_{m_e}", "dimensionless"),
    "ScalarNucleon":  ("d_e", "coupling", "dimensionless"),
    "ScalarBaryon":   ("d_e", "coupling", "dimensionless"),
    "VectorBL":       ("dimensionless", "g_bl", "g_b-l"),
}


# --- Foreign-quantity screen (registry hardening, 2026-07-04) -----------------
# MIRRORS evaluation.conventions._foreign_quantity_declared — keep in sync.
# A declaration naming physics outside the coupling's canonical/vetted
# vocabulary (a different coupling symbol like g_p when canonical is g_ae, an
# hbar-c-normalized dipole-dipole strength, an interaction potential) is
# NEITHER canonical NOR convertible: substring rules alone were fooled across
# couplings (1508.02463 AxionElectron "(g_p^e)^2/(hbar c)" matched "squared"
# AND its "dimensionless" matched the canonical tokens, so a ~9-dex mis-scaled
# vision curve shipped with the review flag suppressed). Fails closed to
# [CONVENTION REVIEW]. "converted from ..." declarations are exempt (#594) and
# negation clauses ("NOT canonical g_ae") are dropped before scanning.
_FOREIGN_CLASS_TOKENS = (
    "v_dd", "vdd", "potential", "force strength", "torque",
    "cross section", "count rate",
)
# Per-COUPLING foreign-class tokens (catastrophic-tail audit, 2026-07-15):
# planes that share a coupling's "dimensionless" vocabulary but are different
# physics FOR THAT COUPLING (VectorBL fifth-force alpha-tilde; AxionMass
# epsilon-bias). Scoped per coupling so the scalar alpha_fifthforce converter
# is untouched. MIRRORS evaluation.conventions._FOREIGN_CLASS_TOKENS_BY_CT.
_FOREIGN_CLASS_TOKENS_BY_CT: dict = {
    "VectorBL": ("alpha", "α"),
    "AxionMass": ("epsilon", "bias parameter", "(m_a/m_a"),
}
_FOREIGN_CLASS_EXEMPT = ("MonopoleDipole", "AxionCPV")
_EXPECTED_SYMBOL_STEMS: dict = {
    # Vocabulary hardening (2026-07-14, final2_opus_n1 flag audit): g_phi/g_chi
    # (the ALP field named phi/chi — g_phigammagamma IS the photon coupling)
    # and g_ksvz (benchmark-provenance parenthetical, "derived from
    # 0.93*g_KSVZ") are photon-coupling family notation, not foreign physics;
    # likewise g_z (the Z' gauge boson IS the B-L vector, 2304.12907) and g_e
    # for ScalarElectron (the Yukawa in the canonical d_me definition gloss,
    # 2201.02042; precedent: AxionElectron already lists g_e).
    # ScalarNucleon d_g/d_mhat are deliberately ABSENT: adding them would let
    # combined-coupling declarations (|d_mhat - d_g|, 1807.04512) pass the
    # foreign screen and then match "dimensionless" as canonical, suppressing
    # a genuine review flag (#683 — vocabulary must never suppress flags).
    "AxionPhoton":    ("g_ag", "g_a\\gamma", "g_aγ", "g_phi", "g_chi", "g_ksvz"),
    "AxionElectron":  ("g_ae", "g_p", "g_e"),
    "AxionNeutron":   ("g_an", "g_ann", "g_n", "g_p", "g_ag", "c_n"),
    "AxionProton":    ("g_ap", "g_ann", "g_an", "g_n", "g_p", "g_ag", "c_n", "c_p"),
    "DarkPhoton":     (),
    "AxionEDM":       ("g_d", "g_ang", "g_{a", "c_g", "d_n", "d_d", "d_ac", "f_a"),
    "AxionMass":      ("f_a", "m_a"),
    "ScalarPhoton":   ("d_e", "d_gamma", "g_phi", "g_ph"),
    "ScalarElectron": ("d_me", "d_{m", "d_e", "g_phi", "g_ph", "g_e"),
    "ScalarNucleon":  ("d_e", "d_n"),
    "ScalarBaryon":   ("d_e", "d_b"),
    "VectorBL":       ("g_b", "g_z"),
}
_NOT_CLAUSE_RE = None

# Clock-comparison sensitivity combination (promoted 2026-07-14, drained token
# 1604.08514; note GPD/explanations/convention-1604.08514-clock-combo-d_e.md):
# d_e-leading with an explicit sub-unity decimal coefficient reduces
# identically to d_e under the compilation's one-coupling-dominance
# convention. Scoped: no-d_e combinations (|d_mhat - d_g|) keep failing
# closed (#683). MIRRORS evaluation.conventions._is_clock_combo_de.
_CLOCK_COMBO_DE_RE = None


def _is_clock_combo_de(decl_lower: str) -> bool:
    global _CLOCK_COMBO_DE_RE
    if _CLOCK_COMBO_DE_RE is None:
        import re as _re
        _CLOCK_COMBO_DE_RE = _re.compile(r"\bd_?e\s*\+\s*0?\.\d+\s*\*?\s*\(")
    return bool(_CLOCK_COMBO_DE_RE.search(decl_lower))


# "(equivalent to g_agamma in GeV^-1)" — the model asserts the emitted values
# ARE the coupling's own canonical quantity; under the truthful-declaration
# contract (#594/#657) that assertion governs the emitted values, so a
# foreign-looking symbol elsewhere in the declaration (c_gamma/Lambda,
# 1903.03586) is presentation, not different physics. The asserted symbol must
# itself be an expected stem — "equivalent to V_dd" exempts nothing.
_EQUIV_CLAUSE_RE = None


def _asserts_canonical_equivalence(coupling_type: str, decl_lower: str) -> bool:
    """True when the declaration asserts equivalence to an expected-stem symbol
    ("equivalent to g_agamma", "same as g_bl"). MIRRORS
    evaluation.conventions._asserts_canonical_equivalence — keep in sync."""
    global _EQUIV_CLAUSE_RE
    import re as _re
    if _EQUIV_CLAUSE_RE is None:
        _EQUIV_CLAUSE_RE = _re.compile(
            r"(?:equivalent to|equal to|same as)\s+([gdc]_\{?\\?[a-z0-9γ]+)")
    expected = _EXPECTED_SYMBOL_STEMS.get(coupling_type) or ()
    for m in _EQUIV_CLAUSE_RE.finditer(decl_lower):
        stem = m.group(1).replace("{", "")
        if any(stem.startswith(e) for e in expected):
            return True
    return False


def _foreign_quantity_declared(coupling_type: str, decl_lower: str) -> bool:
    global _NOT_CLAUSE_RE
    import re as _re
    if _NOT_CLAUSE_RE is None:
        _NOT_CLAUSE_RE = _re.compile(r"\bnot\b[^,;.]*")
    if "converted" in decl_lower:
        return False
    d = _NOT_CLAUSE_RE.sub(" ", decl_lower)
    if coupling_type not in _FOREIGN_CLASS_EXEMPT \
            and any(t in d for t in _FOREIGN_CLASS_TOKENS):
        return True
    # Per-coupling foreign planes (catastrophic-tail audit, 2026-07-15).
    if any(t in d for t in _FOREIGN_CLASS_TOKENS_BY_CT.get(coupling_type, ())):
        return True
    if _asserts_canonical_equivalence(coupling_type, d):
        return False
    if coupling_type == "ScalarPhoton" and _is_clock_combo_de(d):
        return False    # vetted clock combination — reduces to d_e (see above)
    expected = _EXPECTED_SYMBOL_STEMS.get(coupling_type)
    if expected is None:
        return False
    import re as _re2
    for sym in _re2.findall(r"(?<![a-z0-9\\])[gdc]_\{?\\?[a-z0-9γ]+", d):
        stem = sym.replace("{", "")
        if not any(stem.startswith(e) for e in expected):
            return True
    return False


def _declared_convertible(coupling_type: str, decl_lower: str) -> bool:
    """True iff the eval registry has a vetted conversion for this declared output
    convention. MIRRORS ``evaluation.conventions.classify_reported_convention`` —
    keep the two in sync (a convertible alternate must not be flagged for review).

    Updated for the round-2 registry (post-full346 Phase 1d, #653): decay-rate/
    lifetime planes, squared axes, thermal xi, f_a-in-GeV, and the round-1
    scalar GeV^-1 branch (#600) that this mirror had never picked up.
    Foreign-quantity declarations are never convertible (2026-07-04 hardening).
    """
    if _foreign_quantity_declared(coupling_type, decl_lower):
        return False
    inv_gev = any(t in decl_lower for t in
                  ("gev^-1", "gev-1", "gev^{-1}", "1/gev", "gev$^{-1}$"))
    squared = ("^2" in decl_lower or "squared" in decl_lower) \
        and "converted" not in decl_lower
    if coupling_type in ("AxionNeutron", "AxionProton"):
        return inv_gev or squared
    if coupling_type == "AxionElectron":
        return squared
    if coupling_type == "DarkPhoton":
        return any(t in decl_lower for t in ("eps^2", "epsilon^2", "chi^2", "squared"))
    if coupling_type == "AxionEDM":
        # Mirror-drift fix (2026-07-14): the eval registry's inv_fa branch also
        # accepts spelled-out "c_g/f_a" and "c_g/(f_a" (2204.01454 declared
        # "C_G/f_a in GeV^-1" and was runtime-flagged while the eval converted
        # it) — keep this token list identical to evaluation.conventions.
        if any(t in decl_lower for t in ("1/f_a", "invfa", "1/fa", "cg/fa",
                                         "c_g/f_a", "c_g/(f_a", "gluon")):
            return True
        # Oscillating-EDM amplitude in e*cm is now the mass-dependent d_n_ecm
        # converter (Phase 2, #625) — convertible, not a review flag. Its
        # magnitude is guarded by convertible_out_of_profile at extraction time.
        return any(t in decl_lower for t in ("e*cm", "e·cm", "ecm", "e.cm", "e cm"))
    if coupling_type == "AxionPhoton":
        if any(t in decl_lower for t in ("s^-1", "s-1", "decay rate", "1/s")):
            return True
        if decl_lower in ("s", "sec", "seconds") or "lifetime" in decl_lower \
                or "tau" in decl_lower:
            return True
        import re as _re
        if _re.search(r"\bxi\b", decl_lower):
            return True
        return False
    if coupling_type == "AxionMass":
        return any(t in decl_lower for t in
                   ("f_a in gev", "fa in gev", "f_a [gev", "fa [gev", "f [gev"))
    if coupling_type in ("ScalarPhoton", "ScalarElectron"):
        # Higgs-portal sin theta -> d_me (drained token 2303.00778; eval
        # registry converts via sin_theta_higgs, x sqrt2 M_Pl/v). Scoped to
        # ScalarElectron — a ScalarPhoton sin-theta would carry a different
        # constant and must keep flagging.
        if coupling_type == "ScalarElectron" and "sin" in decl_lower \
                and ("theta" in decl_lower or "θ" in decl_lower) \
                and ("higgs" in decl_lower or "mixing" in decl_lower):
            return True
        return inv_gev or "lambda" in decl_lower or "λ" in decl_lower
    return False


# ---------------------------------------------------------------------------
# Unit-PLANE screen (Phase 1a, #625) — fail closed on the unit POWER
# ---------------------------------------------------------------------------
# MIRRORS evaluation.conventions._CANONICAL_PLANE / _explicit_unit_plane /
# _unit_plane_contradicts — keep the two in sync. Token matching sees a plane's
# vocabulary but not its unit POWER; a declaration whose explicit unit power
# contradicts the coupling's canonical plane (and that no vetted converter
# serves) is flagged for review even when it also carries a canonical SYMBOL
# ("epsilon in GeV^-1" for DarkPhoton, "g_agamma in GeV^-2" for AxionPhoton).
# Acts ONLY on an explicit unit token — a unit-less declaration is untouched.
# AxionMass is deliberately ABSENT — legitimately multi-plane (dimensionless
# normalized f_a_norm, 1/f_a [GeV^-1], and f_a [GeV] are all canonical); a
# single-plane screen would false-flag a valid normalized read. MIRRORS
# evaluation.conventions._CANONICAL_PLANE.
_CANONICAL_PLANE: dict = {
    "AxionPhoton":    "gev^-1",
    "AxionEDM":       "gev^-2",
    "AxionElectron":  "dimensionless",
    "AxionNeutron":   "dimensionless",
    "AxionProton":    "dimensionless",
    "AxionCPV":       "dimensionless",
    "DarkPhoton":     "dimensionless",
    "VectorBL":       "dimensionless",
    "MonopoleDipole": "dimensionless",
    "ScalarPhoton":   "dimensionless",
    "ScalarElectron": "dimensionless",
    "ScalarNucleon":  "dimensionless",
    "ScalarBaryon":   "dimensionless",
}


def _explicit_unit_plane(decl_lower: str):
    """The explicit unit-plane token named in a declaration, else None. Most
    specific power wins (GeV^-2 before GeV^-1 before bare GeV). MIRRORS
    evaluation.conventions._explicit_unit_plane."""
    u = (decl_lower or "").replace(" ", "")
    if any(t in u for t in ("gev^-2", "gev-2", "gev^{-2}", "gev**-2", "1/gev^2")):
        return "gev^-2"
    if any(t in u for t in ("gev^-1", "gev-1", "gev^{-1}", "gev**-1",
                            "gev$^{-1}$", "1/gev")):
        return "gev^-1"
    if any(t in u for t in ("e*cm", "e·cm", "ecm", "e.cm")) or "e cm" in decl_lower:
        return "ecm"
    if any(t in u for t in ("s^-1", "s-1", "1/s", "decayrate")):
        return "s^-1"
    if "gev" in u or "[gev]" in u:
        return "gev"
    if "dimensionless" in u:
        return "dimensionless"
    return None


def _unit_plane_contradicts(coupling_type, decl_lower: str) -> bool:
    """True iff the declaration names an explicit unit plane contradicting the
    coupling's canonical plane. MIRRORS evaluation.conventions._unit_plane_contradicts."""
    canon = _CANONICAL_PLANE.get(coupling_type or "")
    if canon is None:
        return False
    got = _explicit_unit_plane(decl_lower)
    if got is None:
        return False
    return got != canon


def convention_review_needed(coupling_type, declared_convention) -> bool:
    """Whether an extraction's model-declared output convention should be flagged
    for human review (escalate-on-UNKNOWN, #536/#587).

    Conservative by design: returns True ONLY for a non-empty declaration that is
    neither a canonical/repo-standard phrasing nor a registry-convertible
    alternate. An empty/absent declaration is treated as canonical (no flag), so
    this never flags the common case where the model did not populate the field.
    """
    if not coupling_type or not declared_convention:
        return False
    d = str(declared_convention).strip().lower()
    if d in ("", "canonical", "standard", "none", "n/a"):
        return False
    # Foreign-quantity screen FIRST (2026-07-04): a foreign-physics declaration
    # must flag for review even when it contains a canonical token in passing
    # (1508.02463's "(g_p^e)^2/(hbar c) dimensionless ..." matched
    # AxionElectron's "dimensionless" and dodged the flag).
    if _foreign_quantity_declared(coupling_type, d):
        return True
    # Unit-plane screen (Phase 1a): an explicit unit power contradicting the
    # canonical plane, with NO vetted converter, fails closed even when a
    # canonical symbol appears ("epsilon in GeV^-1"). A convertible contradiction
    # (AxionEDM "C_G/f_a in GeV^-1" -> inv_fa) is NOT flagged here — the converter
    # owns it and Phase 1b magnitude-guards it at extraction time.
    if _unit_plane_contradicts(coupling_type, d) \
            and not _declared_convertible(coupling_type, d):
        return True
    if any(t in d for t in _CANONICAL_DECL.get(coupling_type, ())):
        return False
    if _declared_convertible(coupling_type, d):
        return False
    return True


# Plausible-input band for the AxionEDM 1/f_a [GeV^-1] plane (Phase 1b, #625).
# A genuine decay constant f_a in [1e9,1e18] GeV gives 1/f_a in [1e-18,1e-9];
# widened one decade each side (f_a in [1e7,1e20] GeV) so no real conversion is
# refused. Values ~10 dex below the floor are the tiny oscillating-EDM d_n
# [e*cm] amplitude mislabeled as GeV^-1. MIRRORS the inv_fa / d_n_ecm /
# f_a_gev_axis guard bands in evaluation.conventions.to_canonical.
_INV_FA_BAND = (1e-20, 1e-7)
_DN_ECM_BAND = (1e-30, 1e-18)
# A MEASURED f_a [GeV] axis must carry decay-constant-scale values (f_a in
# [1e5,1e18] GeV, widened) — canonical-scale values contradict the read-back.
_FA_GEV_AXIS_BAND = (1e3, 1e20)


def convertible_out_of_profile(coupling_type, declared_convention,
                               data_points) -> bool:
    """Runtime side of Phase 1b/2 + the catastrophic-tail audit: a declaration
    that IS registry-convertible (so :func:`convention_review_needed` returns
    False) but whose emitted magnitude cannot be the declared quantity, so the
    vetted conversion would produce out-of-plane garbage. Scoped to the planes
    whose converters carry a magnitude guard — AxionEDM inv_fa (the measured
    16-dex hole 2204.01454/2410.02218), AxionEDM d_n_ecm (Phase 2), and the
    AxionMass f_a-axis-read-back contradiction (1708.08464/2105.13963: a
    measured f_a[GeV] axis with canonical-scale values). Returns True -> the
    caller flags [CONVENTION REVIEW] + queues, exactly as for an unknown
    convention.

    Pure over the model's own declared output; never raises."""
    if not coupling_type or not declared_convention or not data_points:
        return False
    d = str(declared_convention).strip().lower()
    if coupling_type == "AxionMass":
        # Only the axis-read-back provenance — a model-declared f_a keeps the
        # registry's mislabel escape (1708.07521 compares at 0.13 dex).
        if "axis read-back" in d and any(t in d.replace(" ", "") for t in
                                         ("f_aingev", "faingev", "f_a[gev",
                                          "fa[gev", "f[gev")):
            band = _FA_GEV_AXIS_BAND
        else:
            return False
    elif coupling_type != "AxionEDM":
        return False
    elif any(t in d for t in ("1/f_a", "invfa", "1/fa", "cg/fa",
                              "c_g/f_a", "c_g/(f_a", "gluon")):
        band = _INV_FA_BAND
    elif any(t in d for t in ("e*cm", "e·cm", "ecm", "e.cm", "e cm")):
        band = _DN_ECM_BAND
    else:
        return False
    try:
        couplings = [float(g) for _m, g in data_points if float(g) > 0]
    except Exception:
        return False
    if not couplings:
        return False
    med = _median(couplings)
    lo, hi = band
    return not (lo <= med <= hi)


# ---------------------------------------------------------------------------
# Axis-plane consistency cross-check (Phase 3, #625)
# ---------------------------------------------------------------------------
# The 19 remaining wrong-type predictions and the plane-declaration errors are
# the SAME axis question: g_B-L vs ε, f_a vs g_aγ, d_g vs d_e, e·cm vs GeV^-2 are
# all written on the figure y-axis. Figure selection uses the type prior, so this
# is a CROSS-CHECK AFTER the axis read-back, never a blind reorder (no circularity
# — #625's referee negative result). A stage-2a axis label that unambiguously
# names a DIFFERENT plane within the SAME confusable family overrides the
# classified type (re-label, no figure re-selection); a cross-family
# contradiction is review-flagged; an ambiguous/unreadable axis is a no-op
# (fail open to current behaviour).

# Coupling types that are genuinely confusable from text alone but distinguished
# by their figure axis (from the Phase-1 confusion matrix / issue #625).
_CONFUSABLE_FAMILIES: tuple = (
    frozenset({"DarkPhoton", "VectorBL"}),                 # ε vs g_B-L
    frozenset({"ScalarPhoton", "ScalarElectron",
               "ScalarNucleon", "ScalarBaryon"}),          # d_e vs d_me vs d_g
    frozenset({"AxionMass", "AxionEDM", "AxionCPV"}),      # f_a/m_a vs e·cm/g_d vs CPV
    frozenset({"AxionProton", "AxionNeutron"}),            # g_ap vs g_an
    frozenset({"AxionElectron", "AxionPhoton"}),           # g_ae vs g_aγ
)


def same_confusable_family(a, b) -> bool:
    """True iff a and b sit in the same confusable family (override guardrail)."""
    if not a or not b or a == b:
        return a == b
    return any({a, b} <= fam for fam in _CONFUSABLE_FAMILIES)


def in_confusable_family(ct) -> bool:
    """True iff ``ct`` belongs to any confusable family (the 3c axis-peek gate:
    only spend a vision call when the classified type could plausibly be
    corrected by its axis)."""
    return bool(ct) and any(ct in fam for fam in _CONFUSABLE_FAMILIES)


# Distinct coupling-symbol families an axis label can name, used by the
# multi-symbol ambiguity screen below. Each entry is (family_key, tokens);
# a label naming symbols from >=2 DISTINCT families is a product/combination
# plane (sqrt(g_ae*g_aγ), d_e + 0.043(d_m̂ − d_g)) — neither single plane, so
# the map must fail OPEN. Measured on the final347 run: two of the four
# override-caused wrong types were exactly this class (1708.02111, 1604.08514).
_AXIS_SYMBOL_FAMILIES: tuple = (
    ("photon",   ("g_agamma", "g_aγ", "g_a\\gamma")),
    ("electron", ("g_ae",)),
    ("proton",   ("g_ap",)),
    ("neutron",  ("g_an",)),
    ("edm",      ("d_n", "d_ac", "e*cm", "e·cm", "e.cm")),
    ("sc_el",    ("d_me", "d_m_e", "d_mhat", "d_m̂")),
    ("sc_glu",   ("d_g",)),
    ("sc_ph",    ("d_e",)),
    ("bl",       ("g_b-l", "g_b−l", "g_bl")),
    ("eps",      ("epsilon", "ε", "\\chi", "χ")),
)


def _distinct_symbol_families(u: str, raw: str) -> set:
    """Set of distinct coupling-symbol families named in an axis label."""
    found = set()
    for key, tokens in _AXIS_SYMBOL_FAMILIES:
        for t in tokens:
            if t in u or t in raw:
                found.add(key)
                break
    # d_me contains no 'd_e' substring, but 'd_mhat'/'d_m̂' labels often ALSO
    # write d_e in the same combination — that is exactly the >=2 case.
    return found


def axis_implies_coupling_type(axis_label):
    """The coupling type an EXPLICIT figure y-axis label names, else None.

    High-precision by construction — fires only on unambiguous plane tokens, and
    orders the most specific symbol first so g_agamma is never read as g_ae, d_me
    never as d_e, etc. Two ambiguity screens (measured on the final347 run,
    where each caused 2 override-driven wrong types):

    * **multi-symbol combinations** — a label naming >=2 distinct coupling
      symbols (``sqrt(g_ae*g_aγ)``, ``d_e + 0.043(d_m̂ − d_g)``) is a
      product/combination plane, not either single plane -> None (fail open);
    * **bare epsilon** — B−L papers write their gauge coupling ε_B-L / ε too,
      so ε alone cannot distinguish DarkPhoton from VectorBL. ε with a B−L
      qualifier -> VectorBL; ε with kinetic-mixing context (or the unambiguous
      χ symbol) -> DarkPhoton; bare ε -> None.

    An unreadable / ambiguous label returns None (the caller then no-ops).
    Pure, runtime-only (the eval scorer reads emitted snapshots and never
    re-classifies, so there is no eval mirror)."""
    if not axis_label:
        return None
    raw = str(axis_label).lower()
    u = raw.replace(" ", "").replace("{", "").replace("}", "").replace("$", "")
    # --- multi-symbol combination screen (fail open) ---
    if len(_distinct_symbol_families(u, raw)) >= 2:
        return None
    # --- vector / dark photon ---
    # Any B−L qualifier claims the label for VectorBL — including a B−L-
    # subscripted epsilon (2403.03004 wrote its gauge coupling 'epsilon_{B-L}').
    if "g_b-l" in u or "g_b−l" in u or "b-lgauge" in u or "b−lgauge" in u \
            or "g_bl" in u or "b-l" in u or "b−l" in u:
        return "VectorBL"
    # χ is unambiguous kinetic mixing; ε needs kinetic/mixing context (B−L
    # papers use bare ε for their gauge coupling — 2112.07687 wrote 'epsilon^2
    # (squared dimensionless coupling constant)' for ε_B-L^2).
    if "\\chi" in u or "χ" in raw or "kineticmixing" in u:
        return "DarkPhoton"
    if ("epsilon" in u or "ε" in raw or ("eps" in u and "eps^" not in u)) \
            and ("kinetic" in raw or "mixing" in raw):
        return "DarkPhoton"
    # --- axion-fermion (specific symbol before generic) ---
    if "g_agamma" in u or "g_aγ" in u or "g_a\\gamma" in u or "g_aγ" in raw:
        return "AxionPhoton"
    if "g_ae" in u:
        return "AxionElectron"
    if "g_ap" in u:
        return "AxionProton"
    if "g_an" in u:
        return "AxionNeutron"
    # --- axion-EDM (oscillating amplitude plane) ---
    if any(t in u for t in ("e*cm", "e·cm", "ecm", "e.cm")) or "e cm" in raw \
            or "d_n" in u or "d_ac" in u:
        return "AxionEDM"
    # --- scalar / dilaton sub-types (specific before generic) ---
    if "d_me" in u or "d_m_e" in u or "d_mhat" in u or "d_m̂" in raw:
        return "ScalarElectron"
    if "d_g" in u:
        return "ScalarNucleon"
    if "d_e" in u:
        return "ScalarPhoton"
    # --- axion mass / decay-constant plane ---
    if ("f_a" in u or "1/f_a" in u) and "gev" in u:
        return "AxionMass"
    return None


def axis_plane_crosscheck(predicted_ct, axis_label):
    """Cross-check a classified coupling type against its figure axis label.

    Returns ``(resolved_ct, action, note)`` where action is:
      * ``"noop"``     — axis unreadable/ambiguous or already consistent;
      * ``"override"`` — axis names a DIFFERENT plane in the SAME confusable
                         family; ``resolved_ct`` is the axis's type (re-label);
      * ``"review"``   — axis names a plane in a DIFFERENT family (contradiction
                         we will not silently act on); ``resolved_ct`` unchanged.

    Pure and deterministic; never raises. Figure selection is NOT re-run by the
    caller — this only re-labels the already-chosen candidate (#625: bounded, no
    loop)."""
    implied = axis_implies_coupling_type(axis_label)
    if implied is None or implied == predicted_ct:
        return predicted_ct, "noop", ""
    if same_confusable_family(predicted_ct, implied):
        return implied, "override", (
            f"coupling type {predicted_ct} -> {implied} by axis read-back "
            f"'{axis_label}' (same confusable family, #625 Phase 3)")
    return predicted_ct, "review", (
        f"[COUPLING REVIEW] axis '{axis_label}' implies {implied} but classified "
        f"{predicted_ct} (cross-family contradiction — flagged, not overridden)")


# ---------------------------------------------------------------------------
# Gauge-group type correction — U(1)_B / U(1)_{B-L} dark-photon-DM searches
# ---------------------------------------------------------------------------
# A search for a gauged U(1)_B or U(1)_{B-L} vector boson dark matter (the
# LIGO/LISA-Pathfinder/PPTA class) writes its coupling as a dimensionless
# epsilon / epsilon^2 normalized to e — identical in FORM to kinetic-mixing chi.
# The figure axis is therefore a bare epsilon (axis_implies_coupling_type
# deliberately refuses to override on it — see the 2112.07687 note there), so
# the ONLY signal separating these from a real dark photon is the gauge group,
# which the model names in its own convention declaration. The classifier
# routinely still emits DarkPhoton, producing the internal contradiction
# "coupling_type=DarkPhoton but coupling_convention says U(1)_{B-L}". The repo
# folds both baryonic-vector planes into VectorBL (limit_data/VectorB-L holds
# these very searches), so we re-label to VectorBL.
#
# Gated on the DECLARED CONVENTION (the model's own assertion about the emitted
# quantity), never on free-text notes: the trigger is the type/convention
# contradiction, not an incidental mention. A genuine kinetic-mixing chi
# emission never declares a B/B-L gauge coupling, so this cannot demote a real
# dark photon. Unlike convention_review_needed this does not read notes and does
# not fire on a B-L mentioned only in the surrounding discussion (that path is
# owned by the extractor prompt, which is the primary fix; this guard is the
# deterministic, unit-testable net for the clear-declaration cases).
_GAUGE_BL_DECL_TOKENS: tuple = (
    "u(1)_b", "u(1)b", "u(1) b",           # gauged baryon number (prefix of B-L too)
    "u(1)_{b-l}", "u(1)_{b−l}", "u(1)_b-l", "u(1)_b−l", "u(1)b-l",
    "b-l gauge", "b−l gauge", "gauged b-l", "gauged b−l", "gauged baryon",
    "baryon minus lepton", "baryon-lepton", "baryon number",
    "b-l coupling", "b−l coupling", "g_bl", "g_b-l", "g_b−l",
    # A dark photon "coupled to baryons" IS a gauged-baryon-number vector boson
    # by definition — kinetic mixing couples to the EM current (electric charge),
    # never to baryon content. So a convention describing the emitted value as a
    # coupling to baryons is a reliable VectorBL signal even without "U(1)_B".
    # (Measured: with the strengthened prompt, 2301.08736's own convention became
    # "squared coupling of the dark photon to baryons" while it still typed
    # DarkPhoton — the model understood the physics but filled the wrong enum.)
    "to baryon", "coupled to baryon", "coupling to baryon", "baryonic vector",
)


def gauge_group_type_correction(coupling_type, declared_convention):
    """``"VectorBL"`` if a DarkPhoton emission's own declared convention names a
    gauged U(1)_B / U(1)_{B-L} coupling, else ``None``.

    Pure and deterministic; never raises. Only ever maps DarkPhoton -> VectorBL,
    so it is a no-op for every other classified type (including already-correct
    VectorBL) and cannot regress a correct dark photon (whose convention
    describes kinetic mixing / chi, not a B/B-L gauge coupling)."""
    if coupling_type != "DarkPhoton" or not declared_convention:
        return None
    d = str(declared_convention).strip().lower()
    if any(t in d for t in _GAUGE_BL_DECL_TOKENS):
        return "VectorBL"
    return None
