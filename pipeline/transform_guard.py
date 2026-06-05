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
SOURCE_TIER: dict[str, int] = {
    "table": 4,
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


def _source_tier(source: str, n_points: int) -> int:
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
        1 if (not s.y_const and (s.span_dex is None or s.span_dex >= R4_MIN_SPAN_DEX))
        else 0
    )
    n_pts = int(s.n_points or len(c.data_points) or 0)

    return (
        1 if s.in_valid_ranges else 0,             # T0
        non_degenerate,                            # T1
        1 if c.recoverable else 0,                 # T2
        _source_tier(c.source, n_pts),             # T3
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
    "AxionEDM":       ("e cm", "e.cm", "ecm", "g_d", "gev^-2", "gev-2"),
    "AxionCPV":       ("dimensionless", "coupling"),
    "AxionMass":      ("gev^-1", "1/f_a", "f_a_norm", "dimensionless", "normalized"),
    "MonopoleDipole": ("dimensionless", "coupling"),
    "ScalarPhoton":   ("d_e", "dimensionless"),
    "ScalarElectron": ("d_me", "d_{m_e}", "dimensionless"),
    "ScalarNucleon":  ("d_e", "coupling", "dimensionless"),
    "ScalarBaryon":   ("d_e", "coupling", "dimensionless"),
    "VectorBL":       ("dimensionless", "g_bl", "g_b-l"),
}


def _declared_convertible(coupling_type: str, decl_lower: str) -> bool:
    """True iff the eval registry has a vetted conversion for this declared output
    convention. MIRRORS ``evaluation.conventions.classify_reported_convention`` —
    keep the two in sync (a convertible alternate must not be flagged for review).
    """
    if coupling_type in ("AxionNeutron", "AxionProton"):
        return any(t in decl_lower for t in ("gev^-1", "gev-1", "gev^{-1}", "1/gev", "gev$^{-1}$"))
    if coupling_type == "DarkPhoton":
        return any(t in decl_lower for t in ("eps^2", "epsilon^2", "chi^2", "squared"))
    if coupling_type == "AxionEDM":
        return any(t in decl_lower for t in ("1/f_a", "invfa", "1/fa"))
    return False


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
    if any(t in d for t in _CANONICAL_DECL.get(coupling_type, ())):
        return False
    if _declared_convertible(coupling_type, d):
        return False
    return True
