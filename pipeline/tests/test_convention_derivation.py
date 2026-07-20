"""Deterministic tests for the inline convention-derivation tier (#724).

No network: the derivation LLM call is replaced by a mock client. These pin the
parts that own the decision — the safe monomial application, the numeric and
dimensional gates, the cache, and the end-to-end resolve with a canned
derivation — none of which depend on the model.

Run:
    python -m pytest pipeline/tests/test_convention_derivation.py -v
"""

from __future__ import annotations

import json

import pytest

from pipeline.convention_derivation import (
    Monomial,
    numeric_gate,
    dimensional_gate,
    resolve_convention_inline,
    CACHE_PATH,
)

VR = {
    "ScalarBaryon": {"mass": [1e-24, 1e9], "coupling": [1e-25, 1.0]},
    "AxionPhoton": {"mass": [1e-24, 1e12], "coupling": [1e-25, 1e-3]},
    "AxionEDM": {"mass": [1e-24, 1e9], "coupling": [1e-40, 1e-15]},
}


# ---- safe monomial application ------------------------------------------------

def test_monomial_linear():
    m = Monomial(C=2.0, p=1.0, q=0.0)
    assert m.apply([(1.0, 3.0), (10.0, 4.0)]) == [(1.0, 6.0), (10.0, 8.0)]


def test_monomial_sqrt_and_mass_dependent():
    # sqrt family (alpha -> g): g = sqrt(alpha)
    assert Monomial(1.0, 0.5, 0.0).apply([(1.0, 4.0)]) == [(1.0, 2.0)]
    # mass-dependent (d_n * m): q=1
    assert Monomial(3.0, 1.0, 1.0).apply([(2.0, 5.0)]) == [(2.0, 30.0)]


def test_monomial_drops_nonpositive():
    # a non-positive coupling can't be raised to a fractional power -> dropped
    assert Monomial(1.0, 0.5, 0.0).apply([(1.0, -1.0), (1.0, 4.0)]) == [(1.0, 2.0)]


# ---- numeric gate (the strong gate) ------------------------------------------

def test_numeric_gate_accepts_in_band():
    pts = [(1.0, 1e-3), (10.0, 2e-3)]  # median ~1.5e-3, ScalarBaryon band ok
    ok, detail = numeric_gate("ScalarBaryon", pts, VR)
    assert ok, detail


def test_numeric_gate_rejects_out_of_band():
    # converted values ~1e6 — far above the ScalarBaryon ceiling of 1.0
    pts = [(1.0, 1e6), (10.0, 2e6)]
    ok, detail = numeric_gate("ScalarBaryon", pts, VR)
    assert not ok
    assert detail["median"] > detail["band"][1]


def test_numeric_gate_rejects_empty():
    assert numeric_gate("ScalarBaryon", [], VR)[0] is False


# ---- dimensional gate ---------------------------------------------------------

def test_dimensional_gate_matches_token():
    assert dimensional_gate("AxionPhoton", "g_agamma [GeV^-1]")[0] is True
    assert dimensional_gate("AxionEDM", "g [GeV^-2]")[0] is True


def test_dimensional_gate_rejects_wrong_dimension():
    # AxionPhoton wants GeV^-1; declaring dimensionless is wrong
    assert dimensional_gate("AxionPhoton", "dimensionless")[0] is False


# ---- end-to-end resolve with a mock client -----------------------------------

class _MockResp:
    def __init__(self, text):
        self.content = [type("B", (), {"text": text})()]


class _MockClient:
    """Stands in for the anthropic/CLI client: returns a canned derivation."""
    def __init__(self, payload):
        self._payload = payload
        self.messages = self

    def create(self, **kwargs):
        return _MockResp(json.dumps(self._payload))


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    # redirect the module cache to a temp file so tests never touch real state
    import pipeline.convention_derivation as cd
    monkeypatch.setattr(cd, "CACHE_PATH", tmp_path / "derived.json")
    # also patch the functions that default to the module constant
    orig_load, orig_save = cd._load_cache, cd._save_cache
    monkeypatch.setattr(cd, "_load_cache", lambda path=cd.CACHE_PATH: orig_load(path))
    monkeypatch.setattr(cd, "_save_cache", lambda c, path=cd.CACHE_PATH: orig_save(c, path))
    yield


def test_resolve_applies_passing_derivation():
    # alpha (Yukawa strength) -> g_B for ScalarBaryon: g = sqrt(alpha), values ~1e-11
    payload = {"convertible": True, "C": 1.0, "p": 0.5, "q": 0.0,
               "target_units": "g_B [dimensionless]",
               "derivation": "alpha = g^2 => g = sqrt(alpha)", "confidence": 0.8}
    pts = [(1e-15, 1e-22), (1e-13, 4e-22)]  # sqrt -> ~1e-11, in ScalarBaryon band
    res = resolve_convention_inline("ScalarBaryon", "dimensionless Yukawa |alpha|",
                                    pts, _MockClient(payload), valid_ranges=VR)
    assert res is not None and res.ok
    assert res.monomial.p == 0.5
    assert "PROVISIONAL CONVERSION" in res.provisional_declaration
    # converted first point = sqrt(1e-22) = 1e-11
    assert abs(res.converted_points[0][1] - 1e-11) / 1e-11 < 1e-6


def test_resolve_rejects_convertible_false():
    payload = {"convertible": False, "derivation": "lifetime axis, not a coupling"}
    res = resolve_convention_inline("AxionPhoton", "tau_phi lifetime [s]",
                                    [(1e5, 1e3)], _MockClient(payload), valid_ranges=VR)
    assert res is None


def test_resolve_rejects_gate_failing_derivation():
    # model claims a conversion, but it lands wildly out of band -> gate rejects
    payload = {"convertible": True, "C": 1e30, "p": 1.0, "q": 0.0,
               "target_units": "g_B [dimensionless]", "derivation": "bogus"}
    pts = [(1.0, 1.0), (10.0, 2.0)]  # * 1e30 -> ~1e30, far above ceiling 1.0
    res = resolve_convention_inline("ScalarBaryon", "some novel token",
                                    pts, _MockClient(payload), valid_ranges=VR)
    assert res is None  # rejected by the numeric gate


def test_cache_round_trip_second_call_is_free():
    payload = {"convertible": True, "C": 1.0, "p": 0.5, "q": 0.0,
               "target_units": "g_B [dimensionless]", "derivation": "d", "confidence": 0.8}
    pts = [(1e-15, 1e-22)]
    c1 = _MockClient(payload)
    r1 = resolve_convention_inline("ScalarBaryon", "|alpha|", pts, c1, valid_ranges=VR)
    assert r1 and r1.ok and not r1.cached
    # second call: a client that would RAISE if hit proves the cache short-circuits
    class _Boom:
        messages = property(lambda self: (_ for _ in ()).throw(AssertionError("hit API")))
    r2 = resolve_convention_inline("ScalarBaryon", "|alpha|", pts, _Boom(), valid_ranges=VR)
    assert r2 and r2.ok and r2.cached
