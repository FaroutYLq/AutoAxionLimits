"""Deterministic wrong-curve gates for vision extraction candidates (WS3, Lever 5).

Post-hoc checks on extraction OUTPUTS that catch the ``wrong_curve_vision``
failure family from the full346 digest (9 papers: the vision stage traced an
existing exclusion, a compilation envelope, or a wrong-panel/wrong-regime
curve instead of the paper's own new result). Every input the gates need is
already produced by the pipeline — the candidate's vision notes, its points,
the declared coupling, ``is_projection`` — plus the paper abstract, so the
gates are pure functions with no API calls and are measurable by replay over
the cached benchmark snapshots (``evaluation/replay_gates.py``).

The four gates (see ``evaluation/eval_runs/PLAN_extraction_channels.md``, WS3):

* **A projection-target** — the paper's main result is a projection, yet the
  vision notes say the traced curve is an *existing* experiment's bound
  (1512.06165, 1508.01798, 2309.07995). Action: reject.
* **B axis-vs-coupling** — the vision notes themselves assert that the y-axis
  quantity belongs to a different coupling family than the declared coupling
  type (1708.02111: declared AxionElectron, notes report a g_agamma axis).
  Extends the #657 truthful-declaration contract: instead of only re-declaring,
  the candidate is rejected so selection falls back. Action: reject.
* **C mass-regime** — the extracted mass window sits >= ~3 dex outside the
  abstract-stated mass window (1903.12190: 14+ dex; 1808.02340: ~6 dex).
  Fires only when the abstract parse is unambiguous (exactly one two-sided
  mass range). Applies to any source, not just vision — a lone text point at
  a nominal mass is the same wrong-regime failure. Action: reject.
* **D compilation-envelope** — the notes admit tracing an envelope/union of
  several bounds or a compilation plot rather than one curve (1008.3536,
  1207.3275). Action: demote (same pattern as the ``reconstruction`` flag on
  :class:`pipeline.transform_guard.Candidate`).

Rejection is never silent: each :class:`GateResult` carries a ``note`` in the
``[VISION GATE]`` format (mirroring ``[CONVENTION REVIEW]``) that the runtime
integration appends to the emitted notes, and the caller falls back to the
next candidate or zero points with a confidence cap.

Fail-open discipline: a gate that cannot evaluate its inputs (missing notes,
unparseable abstract, no points) returns nothing — the gates only ever act on
a *positive* signal, and none of them may raise.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

# Sources whose data came from an LLM figure read; gates A/B/D only apply to
# these (a text/table read has no vision notes to interrogate).
VISION_SOURCES = frozenset({"figure_vision", "vision"})

# Gate C: minimum log10 separation between the extracted mass interval and the
# abstract-stated mass interval before the candidate is rejected. 3 dex per
# the WS3 plan — far above unit sloppiness (the known catches are 6 and 14+
# dex away), far below any plausible same-figure disagreement.
GATE_C_MIN_GAP_DEX: float = 3.0

# ---------------------------------------------------------------------------
# Notes plumbing
# ---------------------------------------------------------------------------

# The extractor assembles composite notes as "stage1 | selector: ... |
# Vision: <stage-2 notes> | Calibration: ... | read-vote ...". The gates must
# only see the VISION segment: stage-1 notes legitimately enumerate existing
# constraints from the text (1508.01798 lists Eot-Wash numbers in stage 1),
# so running gate A over the whole composite would fire on good papers.
_VISION_SEG_RE = re.compile(r"(?:^|\|)\s*Vision:\s*(.*?)(?=\s*\|\s*(?:Calibration:|read-vote|selector:|\[)|$)",
                            re.DOTALL)


def extract_vision_segment(notes: str | None) -> str:
    """The ``Vision: ...`` segment of a composite notes string, or ``""``.

    Used by the replay harness (cached snapshots only keep the composite) and
    by any caller that does not hold the stage-2 notes separately. Returns the
    empty string when there is no vision segment — the gates then skip
    (fail-open), they never fall back to the stage-1 text.
    """
    if not notes:
        return ""
    m = _VISION_SEG_RE.search(notes)
    return m.group(1).strip() if m else ""


def _sentences(text: str) -> list[str]:
    """Split on '.' boundaries only — ';' must NOT split, because the known
    catches pack the existing-bound admission and the trace verb into one
    sentence joined by ';' (1512.06165)."""
    return [s.strip() for s in re.split(r"(?<=\.)\s+|\.\s*$", text) if s.strip()]


_TRACE_RE = re.compile(r"\btrac(?:e[ds]?|ing)\b", re.IGNORECASE)
# A sentence that explicitly says the named curve was NOT used must not fire.
_NEGATION_RE = re.compile(r"\bnot (?:used|traced|selected)\b|\bwere not used\b|\binstead of\b",
                          re.IGNORECASE)


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """One fired gate. ``action`` is ``"reject"`` or ``"demote"``; ``note`` is
    the ``[VISION GATE]``-formatted string the runtime appends to the emitted
    notes (rejection is never silent)."""

    gate: str      # "A_projection_target" | "B_axis_vs_coupling" | "C_mass_regime" | "D_compilation_envelope"
    action: str    # "reject" | "demote"
    reason: str    # short machine-grep-able cause
    excerpt: str   # the note/abstract excerpt that fired the gate (for the replay report)

    @property
    def note(self) -> str:
        return f"[VISION GATE {self.gate.split('_')[0]}] {self.reason}"


# ---------------------------------------------------------------------------
# Gate A — projection-target: lexicon of existing experiments
# ---------------------------------------------------------------------------

# Normalized names that are too generic to identify an existing experiment
# (they name a bound *category* or appear as ordinary prose words).
_LEXICON_STOPWORDS = frozenset({
    "cosmology", "projections", "readme", "misc", "data", "limits", "bounds",
    "combined", "comment", "solar", "stellar", "other", "dark", "axion",
    "cast",  # collides with the English verb; CAST traces still surface via
             # the existing-bound phrase branch of gate A
})

# Names the file stems don't yield in a matchable form (stems are camel-case
# concatenations; notes write them hyphenated/spaced) plus canonical
# astro/lab bounds every compilation plot carries.
_COMMON_EXISTING_NAMES = (
    "eotwash", "staticep", "sn1987a", "ligo", "virgo", "ligovirgo",
    "torsionbalance", "equivalenceprinciple", "fifthforce",
)


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens ('Eot-Wash' -> ['eot', 'wash'])."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _ngram_joins(tokens: list[str], n_max: int = 4) -> set[str]:
    """All 1..n_max-gram concatenations ('eot','wash' -> 'eotwash', ...)."""
    joins: set[str] = set()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            joins.add("".join(tokens[i:i + n]))
    return joins


def build_experiment_lexicon(limit_data_dir: str | Path = "limit_data") -> frozenset[str]:
    """Normalized existing-experiment names from ``limit_data/**`` file stems.

    ``Projections/`` subdirectories are excluded on purpose: a projection
    paper's own experiment (IAXO, DMRadio, CASPEr phases, ...) lives there,
    and gate A must not reject a projection for naming *itself*. Stems are
    split at the first ``_`` (``COBEFIRAS_Chluba`` -> ``COBEFIRAS``) and both
    forms are kept when >= 4 chars and not stop-listed.
    """
    names: set[str] = set(_COMMON_EXISTING_NAMES)
    root = Path(limit_data_dir)
    if root.is_dir():
        for f in root.glob("*/**/*.txt"):
            if "Projections" in f.parts:
                continue
            stem = f.stem
            for cand in (stem, stem.split("_")[0]):
                norm = "".join(_normalize_tokens(cand))
                if len(norm) >= 4 and norm not in _LEXICON_STOPWORDS:
                    names.add(norm)
    return frozenset(names)


_DEFAULT_LEXICON: frozenset[str] | None = None


def _default_lexicon() -> frozenset[str]:
    global _DEFAULT_LEXICON
    if _DEFAULT_LEXICON is None:
        _DEFAULT_LEXICON = build_experiment_lexicon(
            Path(__file__).resolve().parents[1] / "limit_data")
    return _DEFAULT_LEXICON


def _lexicon_hit(text: str, lexicon: frozenset[str]) -> str | None:
    """First lexicon name whose normalized form appears as a token n-gram."""
    joins = _ngram_joins(_normalize_tokens(text))
    for name in joins & lexicon:
        return name
    return None


_EXISTING_BOUND_RE = re.compile(
    r"\b(?:existing|current|previous(?:ly)?\s+published|published|prior|already[- ]excluded)\s+"
    r"(?:bounds?|exclusions?|constraints?|limits?|exclusion\s+regions?)\b",
    re.IGNORECASE)


def gate_projection_target(*, is_projection: bool, source: str | None,
                           vision_notes: str | None,
                           suggested_experiment_name: str | None = None,
                           paper_title: str | None = None,
                           lexicon: frozenset[str] | None = None) -> GateResult | None:
    """Gate A: a projection paper whose traced curve is an existing bound."""
    try:
        if not is_projection or source not in VISION_SOURCES:
            return None
        if lexicon is None:
            lexicon = _default_lexicon()
        # Own-experiment guard: an experiment with published limits releasing a
        # projection of ITSELF (CASPEr 1711.08999) collides with its own
        # limit_data stem — a lexicon hit that also appears in the paper title
        # is the paper's own experiment, not an existing-bound mistrace.
        title_joins = _ngram_joins(_normalize_tokens(paper_title or ""))

        # Only the notes-branch fires. Matching ``suggested_experiment_name``
        # against the lexicon was tried and removed: a projection paper
        # legitimately names its OWN experiment (DARWIN 1606.07001), and papers
        # that are themselves the source of a limit_data file collide with
        # their own stem (superradiance 2011.11646) — all three known gate-A
        # catches fire via the trace-sentence branch below anyway.
        for sent in _sentences(vision_notes or ""):
            if not _TRACE_RE.search(sent) or _NEGATION_RE.search(sent):
                continue
            phrase = _EXISTING_BOUND_RE.search(sent)
            lex = _lexicon_hit(sent, lexicon)
            if lex in title_joins:
                lex = None
            if phrase or lex:
                what = phrase.group(0) if phrase else f"existing experiment '{lex}'"
                return GateResult(
                    gate="A_projection_target", action="reject",
                    reason=(f"is_projection=true but the vision notes say the traced curve "
                            f"is {what} — not the paper's own projected curve"),
                    excerpt=sent[:240])
        return None
    except Exception:
        return None  # gates fail open, never raise


# ---------------------------------------------------------------------------
# Gate B — axis quantity vs declared coupling family
# ---------------------------------------------------------------------------

# Quantity-symbol regexes -> the coupling families they canonically label.
# Only unmistakable symbols are listed: a family token must identify the axis,
# not merely appear in prose ("GeV^-1" alone is shared by g_agamma / f_a /
# Lambda and is NOT a token). Couplings without a distinctive symbol simply
# never fire gate B (fail-open).
_AXIS_QUANTITY_FAMILIES: tuple[tuple[re.Pattern, frozenset[str]], ...] = (
    (re.compile(r"\bg_?\{?a\\?(?:gamma|γ)\}?", re.IGNORECASE), frozenset({"AxionPhoton"})),
    (re.compile(r"\bg_?\{?ae\}?\b", re.IGNORECASE), frozenset({"AxionElectron"})),
    (re.compile(r"\bg_?\{?an\}?\b", re.IGNORECASE), frozenset({"AxionNeutron"})),
    (re.compile(r"\bg_?\{?ap\}?\b", re.IGNORECASE), frozenset({"AxionProton"})),
    # Bare "chi"/"epsilon" are NOT DarkPhoton tokens: epsilon also names the
    # B-L gauge coupling (2403.03004) and chi is a generic fit statistic. Only
    # the unambiguous phrase counts.
    (re.compile(r"\bkinetic\s+mixing\b", re.IGNORECASE), frozenset({"DarkPhoton"})),
    (re.compile(r"\bd_?\{?m_?e\}?\b", re.IGNORECASE), frozenset({"ScalarElectron"})),
    (re.compile(r"\bg_?\{?b[-\s]?l\}?\b", re.IGNORECASE), frozenset({"VectorBL", "VectorB-L"})),
    (re.compile(r"\bg_?d\b", re.IGNORECASE), frozenset({"AxionEDM"})),
    # "f_a"/"1/f_a" is deliberately absent: an f_a-axis assertion against an
    # AxionMass/fa declaration is unit notation the #653 convention converters
    # already handle truthfully (2408.07740, 2410.19902, 2410.21590,
    # 2412.03655 — all good extractions whose notes flag the 1/f_a axis).
)

# A sentence must positively ASSERT what the plotted axis is before a foreign
# symbol counts — incidental mentions of other couplings are routine in notes.
_AXIS_ASSERT_RE = re.compile(
    r"\by[- ]axis\b|\bvertical axis\b|\baxis (?:is|labeled|label)\b|"
    r"\bis actually an?\b|\bplot of\b|\bvs\.?\s+m_", re.IGNORECASE)


def _families_in(sent: str) -> set[str]:
    fams: set[str] = set()
    for pat, families in _AXIS_QUANTITY_FAMILIES:
        if pat.search(sent):
            fams |= families
    return fams


def gate_axis_vs_coupling(*, source: str | None, coupling_type: str | None,
                          vision_notes: str | None) -> GateResult | None:
    """Gate B: the notes assert a y-axis quantity foreign to the declared coupling."""
    try:
        if source not in VISION_SOURCES or not coupling_type or not vision_notes:
            return None
        for sent in _sentences(vision_notes):
            if not _AXIS_ASSERT_RE.search(sent):
                continue
            fams = _families_in(sent)
            if fams and coupling_type not in fams:
                return GateResult(
                    gate="B_axis_vs_coupling", action="reject",
                    reason=(f"vision notes assert the plotted axis belongs to "
                            f"{'/'.join(sorted(fams))} but the declared coupling is "
                            f"{coupling_type} — wrong panel/curve traced"),
                    excerpt=sent[:240])
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gate C — extracted mass window vs abstract-stated mass window
# ---------------------------------------------------------------------------

_SI_EV = {
    "y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9,
    "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3,
    "M": 1e6, "G": 1e9, "T": 1e12,
}

# One number in abstract notation: "0.8", "500", "10^{-23}", "1.5\times10^{-3}",
# "3×10^-5". LaTeX wrappers ($, {}) are stripped before matching. The caret in
# the power form is REQUIRED: with it optional, plain "100" parses as 10^0.
_NUM = r"(?:\d+(?:\.\d+)?(?:\s*(?:\\times|×|x|\*)\s*10\^\{?[+-]?\d+\}?)?|10\^\{?[+-]?\d+\}?)"
# Case-sensitive SI prefix: meV and MeV differ by 9 dex.
_UNIT = r"([yzafpnµμumkMGT]?)\s*eV(?:\s*/\s*c\$?\^?\{?2\}?\$?)?"

_RANGE_RE = re.compile(
    rf"({_NUM})\s*(?:-|–|—|\\?-|to|and)\s*({_NUM})\s*{_UNIT}")
# "$10^{-23} \lesssim m_DM \lesssim 10^{-10}$ eV" sandwiches.
_SANDWICH_RE = re.compile(
    rf"({_NUM})\s*(?:\\lesssim|\\leq|\\le|<|≤|≲)\s*m[_a-zA-Z\\{{}}\s]*?"
    rf"(?:\\lesssim|\\leq|\\le|<|≤|≲)\s*({_NUM})\s*{_UNIT}")
_MASS_CONTEXT_RE = re.compile(r"\bmass(?:es)?\b|/\s*c\$?\^?\{?2\}?|\bm_", re.IGNORECASE)


def _parse_num(tok: str) -> float | None:
    tok = tok.replace("$", "").replace("{", "").replace("}", "").strip()
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*(?:\\times|×|x|\*)\s*)?10\^([+-]?\d+)", tok)
    if m:
        return float(m.group(1) or 1.0) * 10.0 ** int(m.group(2))
    try:
        return float(tok)
    except ValueError:
        return None


def parse_abstract_mass_window(abstract: str | None) -> tuple[float, float] | None:
    """The single unambiguous two-sided mass window [eV] stated in an abstract.

    Two accepted forms: an explicit range with a unit ("mass range $0.8 - 500$
    keV/c$^2$") and a LaTeX inequality sandwich ("$10^{-23} \\lesssim m_DM
    \\lesssim 10^{-10}$ eV"). A plain range additionally needs mass context
    (the word "mass", an ``m_`` symbol, or a ``/c^2``) within the surrounding
    text, so detector energy windows ("events between 1 and 7 keV") do not
    count. If zero or more than one distinct window parses, returns ``None``
    (gate C only fires on an unambiguous abstract, per the WS3 plan).
    """
    if not abstract:
        return None
    text = abstract.replace("$", "")
    windows: list[tuple[float, float]] = []
    for pat, needs_context in ((_SANDWICH_RE, False), (_RANGE_RE, True)):
        for m in pat.finditer(text):
            lo, hi = _parse_num(m.group(1)), _parse_num(m.group(2))
            scale = _SI_EV.get(m.group(3) or "", None)
            if lo is None or hi is None or scale is None or lo <= 0 or hi <= 0:
                continue
            if needs_context:
                ctx = text[max(0, m.start() - 80):m.end() + 30]
                if not _MASS_CONTEXT_RE.search(ctx):
                    continue
            lo_ev, hi_ev = sorted((lo * scale, hi * scale))
            if hi_ev / lo_ev < 1.0001:  # degenerate "range"
                continue
            windows.append((lo_ev, hi_ev))
    # Collapse duplicates (the same window quoted twice), then demand exactly one.
    distinct: list[tuple[float, float]] = []
    for w in windows:
        if not any(abs(math.log10(w[0] / d[0])) < 0.05 and abs(math.log10(w[1] / d[1])) < 0.05
                   for d in distinct):
            distinct.append(w)
    return distinct[0] if len(distinct) == 1 else None


def mass_window_gap_dex(data_points, window: tuple[float, float]) -> float | None:
    """log10 separation between the extracted mass interval and ``window`` (0 if
    they overlap); ``None`` when there are no positive extracted masses."""
    masses = [float(m) for m, *_ in (data_points or ()) if float(m) > 0]
    if not masses:
        return None
    ext_lo, ext_hi = min(masses), max(masses)
    abs_lo, abs_hi = window
    if ext_lo > abs_hi:
        return math.log10(ext_lo / abs_hi)
    if abs_lo > ext_hi:
        return math.log10(abs_lo / ext_hi)
    return 0.0


def gate_mass_regime(*, data_points, abstract: str | None,
                     min_gap_dex: float = GATE_C_MIN_GAP_DEX) -> GateResult | None:
    """Gate C: extracted masses >= ``min_gap_dex`` outside the abstract window.

    Source-agnostic: a wrong-panel vision trace (1903.12190) and a lone text
    point at a nominal mass (1808.02340) are the same wrong-regime failure.
    """
    try:
        window = parse_abstract_mass_window(abstract)
        if window is None:
            return None
        gap = mass_window_gap_dex(data_points, window)
        if gap is None or gap < min_gap_dex:
            return None
        return GateResult(
            gate="C_mass_regime", action="reject",
            reason=(f"extracted mass window is {gap:.1f} dex outside the abstract-stated "
                    f"range [{window[0]:.3g}, {window[1]:.3g}] eV — wrong figure/panel/"
                    f"regime traced"),
            excerpt=f"abstract window [{window[0]:.3g}, {window[1]:.3g}] eV, gap {gap:.1f} dex")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gate-C rescue — constant mass-axis unit-prefix / frequency correction
# ---------------------------------------------------------------------------
#
# A constant-factor unit misread shifts EVERY extracted mass by the same power
# of ten: the axis was μeV but read as eV (×1e6 too large), or a frequency axis
# (GHz/MHz) was reported in eV without conversion. Gate C rejects these outright
# today; a wrong-window gate that could recognise the offset would recover the
# curve rather than discard it (issue-driven future-work item, JINST §7).
#
# This is NOT the blind #561 corrector, and the difference is the whole safety
# argument. That corrector snapped to whichever factor got closest to a
# geometric anchor with no absolute constraint, which is how a CORRECT 6–100 GeV
# collider reading collapsed by ~14 dex when the frequency factor won the snap
# (#587). Here every safeguard the blind path lacked is present:
#   1. It only runs when Gate C is ALREADY rejecting the candidate (a correct
#      in-window reading never reaches it — gap 0 < 3 dex, gate silent).
#   2. It rescues ONLY if EXACTLY ONE factor lands the whole extracted interval
#      inside the abstract-stated window (widened by ``tol_dex``). Zero or ≥2
#      qualifying factors → no rescue, reject unchanged.
#   3. Frequency→eV factors are candidates ONLY when the candidate's own notes
#      declare a frequency axis (``allow_frequency``); a bare GeV reading with no
#      frequency wording can never be reinterpreted as a frequency.
# The #587 collider case is structurally excluded by (1) — its correct reading
# and its abstract window coincide, so Gate C never fires — and is pinned by a
# regression test.

_H_EV = 4.135667696e-15  # Planck constant [eV·s]; 1 Hz ↔ h eV.

# (factor, label): multiply every extracted mass by ``factor`` to correct a
# constant unit-prefix misread. Both directions of the common prefix errors.
_MASS_RESCUE_PREFIX_FACTORS: tuple[tuple[float, str], ...] = (
    (1e-9, "GeV→eV (÷1e9)"),
    (1e-6, "μeV→eV (÷1e6)"),
    (1e-3, "meV→eV (÷1e3)"),
    (1e3, "keV→eV (×1e3)"),
    (1e6, "MeV→eV (×1e6)"),
    (1e9, "GeV→eV (×1e9)"),
)
# Frequency-axis values reported as eV: multiply by h·(prefix) to reach eV.
_MASS_RESCUE_FREQ_FACTORS: tuple[tuple[float, str], ...] = (
    (_H_EV, "Hz→eV (×h)"),
    (_H_EV * 1e3, "kHz→eV (×h·1e3)"),
    (_H_EV * 1e6, "MHz→eV (×h·1e6)"),
    (_H_EV * 1e9, "GHz→eV (×h·1e9)"),
    (_H_EV * 1e12, "THz→eV (×h·1e12)"),
)

_FREQ_AXIS_RE = re.compile(r"\b(?:frequenc(?:y|ies)|[kMGT]?Hz)\b")


def mass_axis_is_frequency(notes: str | None) -> bool:
    """True when the extraction notes describe a frequency mass axis (Hz/GHz or
    the word 'frequency'). Gates whether the frequency→eV rescue factors are in
    play, so a bare GeV reading is never reinterpreted as a frequency."""
    return bool(notes and _FREQ_AXIS_RE.search(notes))


def rescue_mass_regime(
    *,
    data_points,
    window: tuple[float, float] | None,
    allow_frequency: bool = False,
    tol_dex: float = 0.5,
) -> tuple[float, str] | None:
    """Find the UNIQUE constant factor that maps every extracted mass inside the
    abstract ``window`` (widened by ``tol_dex`` each side). Returns
    ``(factor, label)`` when exactly one candidate qualifies, else ``None``.

    Pure and deterministic: no API, order-independent (min/max only), fixed
    discrete candidate set. Intended to run only on a candidate Gate C is
    already rejecting (see the module note above). ``allow_frequency`` adds the
    frequency→eV factors; without it only unit-prefix powers of ten are tried.

    ``tol_dex`` is deliberately tight (half a decade): the correct rescue is an
    exact power of ten that lands the interval where the true masses live, so a
    non-power-of-ten offset (e.g. ×1e2) whose nearest candidate leaves a
    fractional-decade residual is rejected rather than mis-snapped. The slack
    only absorbs a real curve extending slightly past the abstract's round
    headline window. Because the μeV (÷1e6) and GHz (×h·1e9) factors sit ~0.6
    dex apart, a haloscope whose window admits both is (correctly) refused as
    ambiguous — either correction is within ~0.6 dex, but we never guess."""
    try:
        masses = [float(m) for m, *_ in (data_points or ()) if float(m) > 0]
        if not masses or window is None:
            return None
        lo, hi = min(masses), max(masses)
        abs_lo, abs_hi = window
        if abs_lo <= 0 or abs_hi <= 0:
            return None
        w_lo = abs_lo / (10.0 ** tol_dex)
        w_hi = abs_hi * (10.0 ** tol_dex)
        factors = list(_MASS_RESCUE_PREFIX_FACTORS)
        if allow_frequency:
            factors += list(_MASS_RESCUE_FREQ_FACTORS)
        hits = [(f, label) for f, label in factors
                if w_lo <= lo * f and hi * f <= w_hi]
        return hits[0] if len(hits) == 1 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gate D — compilation-envelope trace
# ---------------------------------------------------------------------------

# Every pattern must signal an envelope OF MULTIPLE bounds. Bare "envelope"
# and singular "combined ... exclusion" were tried and removed: tracing the
# "lower envelope" of the paper's OWN noisy curve (1910.08638, 2005.14184) or
# the paper's own "combined >3sigma exclusion" over its channels (2203.04332)
# is routine, correct phrasing — plural constituents are what distinguish a
# compilation-envelope mistrace.
_ENVELOPE_RES = (
    re.compile(r"\bunion of\b", re.IGNORECASE),
    re.compile(r"\bcompilation\b", re.IGNORECASE),
    re.compile(r"\bsummary plot\b", re.IGNORECASE),
    re.compile(r"\bsurrounding context\b", re.IGNORECASE),
    re.compile(r"\b(?:existing|constraint|exclusion)\s+envelope\b", re.IGNORECASE),
    re.compile(r"\benvelope of\b[^.]*\b(?:constraints|bounds|regions|curves|limits|exclusions)\b",
               re.IGNORECASE),
    re.compile(r"\bcombin(?:ed|ing)\b[^.]*\b(?:constraints|bounds|regions|exclusions)\b",
               re.IGNORECASE),
    re.compile(r"\bmerg(?:ed|ing)\b[^.]*\b(?:constraints|bounds|regions|curves|exclusions)\b",
               re.IGNORECASE),
)


def gate_compilation_envelope(*, source: str | None,
                              vision_notes: str | None) -> GateResult | None:
    """Gate D: the notes admit tracing an envelope/union/compilation boundary.

    The envelope token must appear in a sentence with a trace verb: notes
    routinely *describe* the figure as a compilation or the paper's own
    combined channels (1709.00009, 2301.03433) while correctly tracing a
    single curve — only "traced the union/envelope/..." is an admission.
    """
    try:
        if source not in VISION_SOURCES or not vision_notes:
            return None
        for sent in _sentences(vision_notes):
            if not _TRACE_RE.search(sent) or _NEGATION_RE.search(sent):
                continue
            for pat in _ENVELOPE_RES:
                m = pat.search(sent)
                if m:
                    return GateResult(
                        gate="D_compilation_envelope", action="demote",
                        reason=(f"vision notes admit tracing a compilation envelope "
                                f"('{m.group(0)}') rather than the paper's own single curve"),
                        excerpt=sent[:240])
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------

def check_vision_gates(*, source: str | None, is_projection: bool,
                       coupling_type: str | None, vision_notes: str | None,
                       data_points=None, abstract: str | None = None,
                       suggested_experiment_name: str | None = None,
                       paper_title: str | None = None,
                       lexicon: frozenset[str] | None = None) -> list[GateResult]:
    """Run all four gates; returns every fired gate (possibly empty), rejects first."""
    results = [
        gate_projection_target(is_projection=is_projection, source=source,
                               vision_notes=vision_notes,
                               suggested_experiment_name=suggested_experiment_name,
                               paper_title=paper_title, lexicon=lexicon),
        gate_axis_vs_coupling(source=source, coupling_type=coupling_type,
                              vision_notes=vision_notes),
        gate_mass_regime(data_points=data_points, abstract=abstract),
        gate_compilation_envelope(source=source, vision_notes=vision_notes),
    ]
    fired = [r for r in results if r is not None]
    return sorted(fired, key=lambda r: (r.action != "reject", r.gate))
