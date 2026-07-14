"""Unit tests for the Gate-C mass-axis unit rescue (JINST §7 wrong-window item).

Gate C (``vision_gates.gate_mass_regime``) rejects a candidate whose extracted
mass window sits >=3 dex outside the abstract-stated range. A frequent, benign
cause of that offset is a *constant* unit-prefix misread (μeV read as eV) or an
unconverted frequency axis (GHz reported as eV) — the curve shape is right, only
its mass column is scaled by a fixed power of ten. ``rescue_mass_regime`` finds
the UNIQUE factor that maps the whole interval back inside the window and lets
the runtime correct rather than discard the curve.

Crucially this is NOT the blind #561 corrector: it runs only when Gate C is
already rejecting, and only rescues on a unique in-window factor. The #587
regression (a correct 6-100 GeV collider reading collapsed ~14 dex by a
frequency factor) is pinned here to prove the new path cannot reintroduce it.

Pure functions only — no network/API.

Run:
    python -m pytest pipeline/tests/test_mass_unit_rescue.py -v
"""

from __future__ import annotations

import random

import pytest

from pipeline.vision_gates import (
    GATE_C_MIN_GAP_DEX,
    gate_mass_regime,
    mass_axis_is_frequency,
    parse_abstract_mass_window,
    rescue_mass_regime,
)


def _pts(masses, coupling=1e-12):
    return tuple((m, coupling) for m in masses)


# ---------------------------------------------------------------------------
# mass_axis_is_frequency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("notes", [
    "traced the exclusion in GHz along the x-axis",
    "axis in MHz",
    "the frequency axis spans 1-10",
    "resonator frequencies from 100 kHz",
    "converted from THz",
])
def test_frequency_axis_detected(notes):
    assert mass_axis_is_frequency(notes) is True


@pytest.mark.parametrize("notes", [
    "",
    None,
    "read the g_agamma curve in GeV vs eV mass",  # 'GeV'/'eV' are not Hz
    "mass axis in micro-eV, converted to eV",
])
def test_non_frequency_notes_not_flagged(notes):
    assert mass_axis_is_frequency(notes) is False


# ---------------------------------------------------------------------------
# rescue_mass_regime — the happy path
# ---------------------------------------------------------------------------

def test_unique_prefix_factor_rescues():
    # Abstract window 1e-6..1e-5 eV; extracted masses are ×1e6 too large (a μeV
    # axis read as eV). Only the ÷1e6 factor lands them back in-window.
    window = (1e-6, 1e-5)
    pts = _pts([1.0, 10.0])  # i.e. should have been 1e-6..1e-5
    res = rescue_mass_regime(data_points=pts, window=window)
    assert res is not None
    factor, label = res
    assert factor == pytest.approx(1e-6)
    assert "μeV" in label


def test_frequency_factor_rescues_only_when_allowed():
    # A kHz axis reported as eV: true mass = f * h. 1-10 kHz ~ 4.1e-12..4.1e-11
    # eV. kHz is chosen because no unit-prefix power of ten sits within a decade
    # of h*1e3, so the rescue is unambiguous (unlike GHz, which brushes μeV).
    window = (1e-12, 1e-10)
    pts = _pts([1.0, 10.0])  # 1-10 "eV" that are really 1-10 kHz
    # Without allow_frequency, no prefix power of ten lands in-window.
    assert rescue_mass_regime(data_points=pts, window=window) is None
    # With allow_frequency, the kHz factor uniquely rescues.
    res = rescue_mass_regime(data_points=pts, window=window, allow_frequency=True)
    assert res is not None
    factor, label = res
    assert "kHz" in label


def test_haloscope_freq_vs_uev_ambiguity_refused():
    # μeV (÷1e6) and GHz (×h·1e9) offsets differ by only ~0.6 dex, so a window
    # admitting both must be refused rather than guessed.
    window = (1e-6, 1e-5)
    pts = _pts([1.0, 5.0])
    res = rescue_mass_regime(data_points=pts, window=window, allow_frequency=True)
    assert res is None


# ---------------------------------------------------------------------------
# rescue_mass_regime — refuses when ambiguous or inapplicable
# ---------------------------------------------------------------------------

def test_two_in_window_factors_refuse():
    # A very wide window (12 decades) admits more than one prefix factor for a
    # single-decade interval, so the rescue must refuse (ambiguous).
    window = (1e-9, 1e3)
    pts = _pts([1e5, 1e6])
    res = rescue_mass_regime(data_points=pts, window=window)
    assert res is None


def test_no_factor_in_window_refuses():
    # Extracted interval is offset by a factor that is not near any candidate
    # power of ten (×10^4.5), so with the tight tolerance nothing lands in-window.
    window = (1.0, 10.0)
    pts = _pts([10 ** 4.5, 10 ** 5.5])
    assert rescue_mass_regime(data_points=pts, window=window) is None


def test_none_window_or_empty_points_refuse():
    assert rescue_mass_regime(data_points=_pts([1.0]), window=None) is None
    assert rescue_mass_regime(data_points=(), window=(1e-6, 1e-5)) is None
    assert rescue_mass_regime(data_points=None, window=(1e-6, 1e-5)) is None


def test_nonpositive_masses_ignored():
    # Zero/negative masses are dropped; the surviving interval still rescues.
    window = (1e-6, 1e-5)
    pts = ((0.0, 1e-12), (-3.0, 1e-12), (1.0, 1e-12), (10.0, 1e-12))
    res = rescue_mass_regime(data_points=pts, window=window)
    assert res is not None and res[0] == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# Determinism / order independence
# ---------------------------------------------------------------------------

def test_order_independent():
    window = (1e-6, 1e-5)
    base = [1.0, 3.0, 10.0]
    r1 = rescue_mass_regime(data_points=_pts(base), window=window)
    shuffled = base[:]
    random.Random(0).shuffle(shuffled)
    r2 = rescue_mass_regime(data_points=_pts(shuffled), window=window)
    assert r1 == r2


# ---------------------------------------------------------------------------
# #587 regression — a CORRECT collider reading must never be "rescued"
# ---------------------------------------------------------------------------

def test_correct_collider_reading_never_reaches_rescue():
    # 2008.05355-style: 6-100 GeV correctly read as 6e9-1e11 eV, and the
    # abstract states the same 6-100 GeV window. Gate C does NOT fire (gap 0),
    # so the rescue is never invoked in the runtime.
    abstract = "We set limits on ALPs with mass $6 - 100$ GeV/c$^2$ produced in..."
    window = parse_abstract_mass_window(abstract)
    assert window is not None
    pts = _pts([6e9, 1e11])
    assert gate_mass_regime(data_points=pts, abstract=abstract) is None


def test_correct_collider_reading_not_falsely_rescued_even_if_forced():
    # Defence in depth: even if the rescue were called directly on a correct
    # in-window collider reading, no non-identity factor pulls it further into a
    # window it already fills, and the frequency factor (×h~4e-15) would shove
    # 6e9 eV down to ~2.5e-5 eV — far below the 6-100 GeV window — so it is
    # rejected, not applied.
    window = (6e9, 1e11)
    pts = _pts([6e9, 1e11])
    assert rescue_mass_regime(data_points=pts, window=window, allow_frequency=True) is None


# ---------------------------------------------------------------------------
# End-to-end through gate_mass_regime: rescue exactly the cases Gate C rejects
# ---------------------------------------------------------------------------

def test_gate_c_fires_then_rescue_recovers():
    # A μeV haloscope axis read as eV: abstract says 1-5 μeV, extracted 1-5 "eV".
    abstract = "The cavity probes axion dark matter of mass $1 - 5$ μeV/c$^2$."
    window = parse_abstract_mass_window(abstract)
    assert window is not None
    pts = _pts([1.0, 5.0])
    fired = gate_mass_regime(data_points=pts, abstract=abstract)
    assert fired is not None and fired.action == "reject"  # Gate C would reject
    res = rescue_mass_regime(data_points=pts, window=window)
    assert res is not None and res[0] == pytest.approx(1e-6)
    # Corrected masses now fall inside the window (gate would be silent).
    corrected = _pts([1.0 * res[0], 5.0 * res[0]])
    assert gate_mass_regime(data_points=corrected, abstract=abstract) is None
