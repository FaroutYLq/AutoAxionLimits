"""Unit tests for read-layer determinism (P3, issue #572).

No API calls. Covers the two deterministic, zero-extra-cost components of P3:

* temperature-0 decoding — the `_create` wrapper injects `temperature=0.0` into
  every `messages.create` (extractor tests; need anthropic, so they self-skip in
  the minimal no-API job),
* the coupling-convention normalizer — `normalize_convention` (pure; the 5.6-dex
  fix for 1902.04246's C_e/F_a [eV^-1] -> dimensionless g_ae).

Run:
    pytest evaluation/tests/test_read_determinism.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.transform_guard import normalize_convention


# ---------------------------------------------------------------------------
# Convention normalizer (pure)
# ---------------------------------------------------------------------------

_GAE_FACTOR = 2.0 * 511000.0  # g_ae = 2 m_e (C_e/F_a) = 1.022e6


def test_axion_electron_converts_on_note():
    # 1902.04246: read note states the eV^-1 convention -> convert to g_ae.
    pts = [(1e-20, 5e-16), (1e-19, 5e-16)]
    out, note = normalize_convention("AxionElectron", pts,
                                     notes="y-axis of Fig.5 is C_e/F_a in eV^-1")
    assert out[0][1] == pytest.approx(5e-16 * _GAE_FACTOR)  # ~5.11e-10
    assert out[0][1] == pytest.approx(5.11e-10, rel=1e-3)
    assert "g_ae" in note

def test_axion_electron_converts_on_axis_unit_label():
    pts = [(1e-20, 5e-16)]
    out, note = normalize_convention("AxionElectron", pts, axis_unit_label="eV^-1")
    assert out[0][1] == pytest.approx(5e-16 * _GAE_FACTOR)
    assert note

@pytest.mark.parametrize("unit", ["eV^-1", "eV⁻¹", "1/eV", "eV-1", "/eV"])
def test_inverse_ev_unit_tokens_detected(unit):
    out, note = normalize_convention("AxionElectron", [(1e-20, 5e-16)],
                                     axis_unit_label=unit)
    assert note != ""

def test_no_conversion_without_explicit_convention():
    # A legitimately small dimensionless g_ae (no note/label) must NOT be touched
    # (the rejected range-fallback would have wrongly converted it).
    pts = [(1e-20, 5e-16)]
    out, note = normalize_convention("AxionElectron", pts)
    assert out == pts and note == ""

def test_dimensionless_gae_not_converted_even_in_band():
    pts = [(1e-20, 5e-10)]  # already canonical
    out, note = normalize_convention("AxionElectron", pts, notes="dimensionless g_ae")
    assert out == pts and note == ""

def test_axion_proton_converts():
    pts = [(1e-20, 1e-16)]
    out, note = normalize_convention("AxionProton", pts, axis_unit_label="eV^-1")
    assert out[0][1] == pytest.approx(1e-16 * 2.0 * 9.382720813e8)
    assert "g_aN" in note

def test_axion_neutron_uses_neutron_mass():
    pts = [(1e-20, 1e-16)]
    out, _ = normalize_convention("AxionNeutron", pts, notes="C_N/F_a")
    assert out[0][1] == pytest.approx(1e-16 * 2.0 * 9.395654205e8)

def test_dark_photon_is_noop():
    pts = [(1e-10, 1e-6)]
    out, note = normalize_convention("DarkPhoton", pts, axis_unit_label="eV^-1")
    assert out == pts and note == ""

def test_axion_edm_not_auto_converted():
    pts = [(1e-10, 1e-26)]
    out, note = normalize_convention("AxionEDM", pts, axis_unit_label="eV^-1")
    assert out == pts and note == ""

def test_empty_and_none_inputs():
    assert normalize_convention("AxionElectron", []) == ([], "")
    assert normalize_convention(None, [(1e-20, 5e-16)]) == ([(1e-20, 5e-16)], "")
    assert normalize_convention("AxionElectron", [(0.0, 0.0)], notes="eV^-1")[1] == ""


# --- GeV^-1 must NOT be misread as eV^-1 (substring bug, #587 P-B) ------------

from pipeline.transform_guard import _has_inv_ev


@pytest.mark.parametrize("unit", ["GeV^-1", "keV^-1", "MeV^-1", "TeV^-1",
                                  "GeV$^{-1}$", "GeV-1"])
def test_prefixed_inverse_energy_not_detected_as_inv_ev(unit):
    assert _has_inv_ev(unit) is False

@pytest.mark.parametrize("unit", ["eV^-1", "eV-1", "eV⁻¹", "eV**-1", "1/eV",
                                  "C_e/F_a in eV^-1"])
def test_bare_inverse_ev_still_detected(unit):
    assert _has_inv_ev(unit) is True

def test_gev_inverse_does_not_trigger_spurious_conversion():
    # The bug: a GeV^-1 axion-nucleon coupling was misread as eV^-1 and multiplied
    # by ~2*m_N (~1e9). It must now pass through untouched.
    pts = [(1e-15, 9e-12)]
    out, note = normalize_convention("AxionNeutron", pts, axis_unit_label="GeV^-1")
    assert out == pts and note == ""


# ---------------------------------------------------------------------------
# Deterministic decoding wrapper (needs anthropic transitively)
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic  # noqa: F401
    _HAVE_EXTRACTOR = True
except Exception:
    _HAVE_EXTRACTOR = False

requires_extractor = pytest.mark.skipif(
    not _HAVE_EXTRACTOR, reason="pipeline.extractor needs anthropic (minimal no-API job)")


class _FakeMessages:
    def __init__(self):
        self.last = None

    def create(self, **kw):
        self.last = kw
        return kw


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


@requires_extractor
def test_create_injects_temperature_zero():
    from pipeline.extractor import _create, _READ_TEMPERATURE
    assert _READ_TEMPERATURE == 0.0
    kw = _create(_FakeClient(), model="m", max_tokens=10)
    assert kw["temperature"] == 0.0

@requires_extractor
def test_create_respects_explicit_temperature():
    from pipeline.extractor import _create
    kw = _create(_FakeClient(), model="m", temperature=0.4)
    assert kw["temperature"] == 0.4


class _RejectsTemperatureMessages:
    """A client that 400s when sent `temperature` (mimics claude-opus-4-8)."""

    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(dict(kw))
        if "temperature" in kw:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': "
                "'`temperature` is deprecated for this model.'}}"
            )
        return kw


class _RejectsTemperatureClient:
    def __init__(self):
        self.messages = _RejectsTemperatureMessages()


@requires_extractor
def test_create_falls_back_when_temperature_rejected():
    # #580: a model that deprecated `temperature` must not break extraction.
    from pipeline.extractor import _create, _TEMPERATURE_UNSUPPORTED
    _TEMPERATURE_UNSUPPORTED.discard("rej")
    c = _RejectsTemperatureClient()
    kw = _create(c, model="rej", max_tokens=10)
    assert "temperature" not in kw                 # final successful call omitted it
    assert len(c.messages.calls) == 2              # one rejected (with temp), one without
    assert "rej" in _TEMPERATURE_UNSUPPORTED
    # subsequent calls skip temperature entirely (no repeated wasted 400)
    c2 = _RejectsTemperatureClient()
    kw2 = _create(c2, model="rej", max_tokens=10)
    assert "temperature" not in kw2
    assert len(c2.messages.calls) == 1
    _TEMPERATURE_UNSUPPORTED.discard("rej")


@requires_extractor
def test_all_reads_route_through_create():
    # No bare `client.messages.create(` remains outside the `_create` wrapper, so
    # every read carries temperature=0.0.
    import re
    src = (PROJECT_ROOT / "pipeline" / "extractor.py").read_text()
    bare = re.findall(r"client\.messages\.create\(", src)
    # exactly one occurrence: the call inside _create itself
    assert len(bare) == 1, f"found {len(bare)} bare messages.create calls; expected 1 (in _create)"
