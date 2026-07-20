"""Inline convention-derivation tier (#724).

When production/benchmark extraction hits a NEW coupling-convention token that
the static registry cannot convert, this module attempts to derive the
conversion *on the fly* — a single focused LLM derivation — verifies it with
deterministic gates (numeric plausibility against the paper's own extracted
extremes + dimensional sanity), and, ONLY if the gates pass, applies it
**provisionally** to the emitted ``data_points``.

Design (issue #724 — "inline-derive, gate-apply"):

* GPD-flavoured, not the full multi-agent GPD stack: one derivation call + pure
  verification gates. Runs inline in Claude Code / CLI runs only; the Actions
  daily/backfill path keeps flag-and-queue (no GPD there).
* **The gates own the decision, not the LLM.** A derivation is *applied
  provisionally*, never promoted to the permanent registry — a human PR does
  that (the C_N/AxionEDM same-units-different-physics incident is why).
* Conversions are a single safe monomial ``g_out = C * g_in**p * m_eV**q`` — no
  ``eval``/``exec`` of model text, so there is no security surface. This form
  covers every case seen: linear factor, sqrt families (α→g_B, g²/4π→g),
  squared, and the mass-dependent d_n·m_a/√(2ρ) EDM converter.
* Cached per token (``pipeline/state/derived_conventions.json``); the second
  paper with the same convention pays nothing.
* Opt-in via ``AAL_INLINE_CONVENTION=1`` (default OFF — no behaviour change to
  the paused arm, the definitive benchmark, or production until validated).
* Model overridable via ``AAL_CONVENTION_MODEL`` so derivations do not compete
  with the extraction model's quota.

The whole tier is exception-swallowing: any failure returns ``None`` and the
caller falls back to today's flag-and-queue behaviour.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Reuse the queue's dedup key so a derived converter and a queued flag for the
# same token share an identity.
from .convention_queue import cache_key  # noqa: E402

CACHE_PATH = Path(__file__).parent / "state" / "derived_conventions.json"

# Canonical target convention/units per coupling type — the derivation is asked
# to land the value in this plane, and the dimensional gate checks against it.
# Deliberately terse; the numeric gate is the real teeth.
CANONICAL_TARGET: dict[str, str] = {
    "AxionPhoton": "g_agamma [GeV^-1]",
    "AxionElectron": "g_ae [dimensionless]",
    "AxionNeutron": "g_an [dimensionless]",
    "AxionProton": "g_ap [dimensionless]",
    "AxionEDM": "g_angamma [GeV^-2]",
    "DarkPhoton": "kinetic mixing epsilon [dimensionless]",
    "VectorBL": "g_B-L [dimensionless]",
    "ScalarPhoton": "d_e [dimensionless]",
    "ScalarElectron": "d_me [dimensionless]",
    "ScalarNucleon": "d_g [dimensionless]",
    "ScalarBaryon": "g_B [dimensionless]",
    "AxionMass": "g_agamma [GeV^-1]",
}


def inline_enabled() -> bool:
    """The tier is strictly opt-in; default OFF preserves all current behaviour."""
    return os.environ.get("AAL_INLINE_CONVENTION", "").lower() in ("1", "true", "yes")


def _convention_model() -> str:
    """Model for the derivation call. Defaults to a non-extraction model so a
    derivation never competes with the extraction model's quota."""
    return (os.environ.get("AAL_CONVENTION_MODEL")
            or os.environ.get("REVIEWER_MODEL")
            or "claude-opus-4-8")


# ---------------------------------------------------------------------------
# Safe conversion: a single monomial, applied by pure arithmetic
# ---------------------------------------------------------------------------

@dataclass
class Monomial:
    """g_out = C * g_in**p * m_eV**q. No eval, ever."""
    C: float
    p: float
    q: float

    def apply(self, data_points):
        out = []
        for m, g in data_points:
            m = float(m); g = float(g)
            if g <= 0 or m <= 0:
                # a non-positive input cannot be raised to a fractional power;
                # drop the point rather than emit NaN.
                continue
            out.append((m, self.C * (g ** self.p) * (m ** self.q)))
        return out


@dataclass
class InlineConversionResult:
    ok: bool
    converted_points: list = field(default_factory=list)
    provisional_declaration: str = ""
    monomial: Optional[Monomial] = None
    target_units: str = ""
    derivation: str = ""
    checks: dict = field(default_factory=dict)
    cached: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# Deterministic verification gates (these own the decision)
# ---------------------------------------------------------------------------

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def numeric_gate(coupling_type, converted_points, valid_ranges, margin_dex=1.0):
    """Strong gate: the converted couplings' median must land inside the type's
    physically-valid coupling band (±``margin_dex`` decades). A wrong-factor or
    double-conversion derivation lands out-of-plane and is rejected here.

    Pure; returns (passed: bool, detail: dict)."""
    rng = (valid_ranges or {}).get(coupling_type, {}).get("coupling")
    gs = [g for _m, g in converted_points if g > 0 and math.isfinite(g)]
    if not rng or not gs:
        return False, {"reason": "no range or no positive converted points",
                       "n": len(gs)}
    lo, hi = float(rng[0]), float(rng[1])
    lo_m = lo / (10 ** margin_dex)
    hi_m = hi * (10 ** margin_dex)
    med = _median(gs)
    passed = lo_m <= med <= hi_m
    return passed, {"median": med, "band": [lo_m, hi_m],
                    "raw_band": [lo, hi], "n": len(gs)}


def dimensional_gate(coupling_type, target_units):
    """Secondary sanity: the derivation's declared target units must name the
    canonical plane for the type (dimension token match). Soft — the numeric
    gate is authoritative."""
    want = CANONICAL_TARGET.get(coupling_type, "")
    if not want:
        return True, {"reason": "no canonical target on record; skipped"}
    t = (target_units or "").lower().replace(" ", "")
    # match on the dimensional core (GeV^-1 / GeV^-2 / dimensionless)
    for tok in ("gev^-2", "gev^-1", "dimensionless"):
        if tok in want.lower().replace(" ", ""):
            return (tok in t), {"want_token": tok, "got": target_units}
    return True, {"reason": "unrecognized canonical token; skipped"}


# ---------------------------------------------------------------------------
# Cache (once per token)
# ---------------------------------------------------------------------------

def _load_cache(path: Path = CACHE_PATH) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {"version": 1, "entries": {}}


def _save_cache(cache: dict, path: Path = CACHE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2))
    except Exception:  # pragma: no cover - never fail extraction on cache I/O
        pass


# ---------------------------------------------------------------------------
# The derivation call
# ---------------------------------------------------------------------------

_DERIVATION_SYSTEM = """You are a physics unit-conversion deriver for a dark-matter \
constraint repository. You are given a coupling type, its CANONICAL target \
convention, a paper's DECLARED (non-canonical) convention, and sample (mass_eV, \
value) points the paper actually plots.

Derive the conversion from the declared quantity to the canonical coupling as a \
SINGLE MONOMIAL:  g_canonical = C * (value_declared ** p) * (m_eV ** q)

- Do the physics: relate the declared quantity to the canonical coupling, show \
the algebra, and give a dimensional check.
- If the declared quantity CANNOT be converted to the canonical coupling with a \
known physical relation (e.g. it is a different observable such as a lifetime, a \
decay constant on an inverted-bound plane, or a prediction band), set \
"convertible": false. Do NOT invent a conversion.
- Return STRICT JSON only, no prose outside it:
{"convertible": true|false, "C": <float>, "p": <float>, "q": <float>, \
"target_units": "<canonical units>", "derivation": "<short derivation + \
dimensional check>", "confidence": <0..1>}"""


def _run_derivation(coupling_type, declared, data_points, client):
    """One LLM derivation call. Returns the parsed dict or None on any failure."""
    from .extractor import _call_with_retry, _create, _parse_json_response, CLAUDE_MODEL  # noqa: E402

    sample = [[float(m), float(g)] for m, g in data_points[:8]]
    user = (
        f"coupling_type: {coupling_type}\n"
        f"canonical_target: {CANONICAL_TARGET.get(coupling_type, '(unknown)')}\n"
        f"declared_convention: {declared}\n"
        f"sample_points (mass_eV, value): {json.dumps(sample)}\n\n"
        "Derive g_canonical = C * value**p * m_eV**q, or convertible:false."
    )
    model = _convention_model() or CLAUDE_MODEL
    resp = _call_with_retry(lambda: _create(
        client,
        model=model,
        max_tokens=1024,
        system=_DERIVATION_SYSTEM,
        messages=[{"role": "user", "content": user}],
    ))
    text = resp.content[0].text if resp and resp.content else ""
    return _parse_json_response(text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_convention_inline(coupling_type, declared_convention, data_points,
                              client, *, arxiv_id=None,
                              valid_ranges=None) -> Optional[InlineConversionResult]:
    """Try to derive + verify + apply a conversion for an unknown convention.

    Returns an :class:`InlineConversionResult` with ``ok=True`` when a derivation
    passed the gates and the points were converted; returns ``None`` to fall back
    to the caller's existing flag-and-queue behaviour. Never raises."""
    try:
        if not coupling_type or not declared_convention or not data_points:
            return None
        if valid_ranges is None:
            from .config import VALID_RANGES as valid_ranges  # noqa: E402

        key = cache_key(coupling_type, declared_convention)
        cache = _load_cache()
        hit = cache.get("entries", {}).get(key)

        if hit and hit.get("convertible"):
            mono = Monomial(hit["C"], hit["p"], hit["q"])
            target_units = hit.get("target_units", "")
            derivation = hit.get("derivation", "")
            cached = True
        elif hit and hit.get("convertible") is False:
            # a prior derivation already judged this token unconvertible; don't
            # re-spend — fall back to flag.
            return None
        else:
            der = _run_derivation(coupling_type, declared_convention, data_points, client)
            if not der or not isinstance(der, dict):
                return None
            if not der.get("convertible"):
                cache.setdefault("entries", {})[key] = {
                    "convertible": False, "coupling_type": coupling_type,
                    "declared": declared_convention,
                    "derivation": der.get("derivation", "") if isinstance(der, dict) else "",
                }
                _save_cache(cache)
                return None
            try:
                mono = Monomial(float(der["C"]), float(der["p"]), float(der["q"]))
            except (KeyError, TypeError, ValueError):
                return None
            target_units = str(der.get("target_units", ""))
            derivation = str(der.get("derivation", ""))
            cached = False

        converted = mono.apply(data_points)
        num_ok, num_detail = numeric_gate(coupling_type, converted, valid_ranges)
        dim_ok, dim_detail = dimensional_gate(coupling_type, target_units)
        checks = {"numeric": num_detail, "numeric_pass": num_ok,
                  "dimensional": dim_detail, "dimensional_pass": dim_ok}

        if not (num_ok and dim_ok and converted):
            # A live derivation that fails the gates is cached as unconvertible so
            # sibling papers do not re-spend on the same losing derivation. A
            # cached converter that now fails (data-dependent) simply falls back.
            if not cached:
                cache.setdefault("entries", {})[key] = {
                    "convertible": False, "coupling_type": coupling_type,
                    "declared": declared_convention,
                    "derivation": derivation, "gate_failed": checks,
                }
                _save_cache(cache)
            return None

        # Passed. Cache the converter (once per token) and build the result.
        if not cached:
            cache.setdefault("entries", {})[key] = {
                "convertible": True, "C": mono.C, "p": mono.p, "q": mono.q,
                "target_units": target_units, "derivation": derivation,
                "coupling_type": coupling_type, "declared": declared_convention,
            }
            _save_cache(cache)

        summary = (f"g=C·g^p·m^q with C={mono.C:.3g}, p={mono.p:g}, q={mono.q:g} "
                   f"→ {target_units}"
                   + (" [cached]" if cached else ""))
        declaration = (f"[PROVISIONAL CONVERSION] converted from {declared_convention!r} "
                       f"via {summary}; canonical {target_units}")
        return InlineConversionResult(
            ok=True, converted_points=converted, provisional_declaration=declaration,
            monomial=mono, target_units=target_units, derivation=derivation,
            checks=checks, cached=cached, summary=summary,
        )
    except Exception as e:  # pragma: no cover - never fail extraction
        logger.warning("inline convention derivation failed for %s (%s): %s",
                       arxiv_id, coupling_type, e)
        return None
