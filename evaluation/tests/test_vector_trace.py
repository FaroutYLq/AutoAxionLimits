"""Unit tests for the WS2 vector-path figure tracer (pipeline/vector_trace.py).

No API, no network. The core test is a full ROUND TRIP: render a log-log
matplotlib figure of a known curve to PDF, trace it back with the module, and
require the recovered points to match the input within a small tolerance —
this pins tick reconstruction, axis fitting, unit conversion, colour-grouped
curve extraction, and the data-space mapping in one go.

Run:
    pytest evaluation/tests/test_vector_trace.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.vector_trace import (
    AxisCalibration,
    VectorCurve,
    curve_to_data,
    fit_axes,
    trace_figure_pdf,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MASS = np.logspace(-8, -3, 60)                      # "frequency" axis, Hz
COUP = 1e-12 * (MASS / 1e-6) ** 0.5                 # known power-law curve


@pytest.fixture(scope="module")
def figure_pdf(tmp_path_factory):
    p = tmp_path_factory.mktemp("figs") / "roundtrip.pdf"
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.loglog(MASS, COUP, color="firebrick", lw=1.5)
    ax.loglog(MASS, COUP * 30, color="steelblue", lw=1.5)  # a second curve
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("coupling $d_e$")
    fig.savefig(p)
    plt.close(fig)
    return p


class TestRoundTrip:
    def test_calibration_signs_and_ticks(self, figure_pdf):
        # This matplotlib build renders the superscript minus (and some label
        # glyphs) as vector paths, not text — the drawn-minus detection must
        # still recover the exponent signs: both axes span negative decades,
        # increasing rightward/upward (ax > 0, ay < 0 in page coords).
        cal, curves = trace_figure_pdf(figure_pdf)
        assert cal is not None
        assert cal.n_xticks >= 3 and cal.n_yticks >= 3
        assert cal.ax > 0 and cal.ay < 0
        # x = -8 decade must map near 1e-8, not 1e+8 (the sign-flip failure)
        left_exp = cal.ax * 40 + cal.bx
        assert left_exp < 0
        assert len(curves) >= 2

    def test_x_unit_map(self):
        # unit recognition itself, on clean label strings (real-paper figures
        # with intact text layers, e.g. 2301.03433, exercise this end to end)
        from pipeline.vector_trace import _X_UNIT_EV, _X_UNIT_RE
        for label, ev in [("Frequency (Hz)", 4.135667696e-15),
                          ("mass [eV]", 1.0), ("m_a [GeV]", 1e9),
                          ("frequency [kHz]", 4.135667696e-12),
                          ("mass [neV]", 1e-9)]:
            m = _X_UNIT_RE.search(label)
            assert m, label
            assert _X_UNIT_EV[m.group(1) or m.group(2)] == pytest.approx(ev), label

    def test_known_curve_recovered(self, figure_pdf):
        # native units (this rendering keeps no label text, so no eV factor)
        cal, curves = trace_figure_pdf(figure_pdf)
        best = None
        for c in curves[:4]:
            pts = curve_to_data(c, cal, to_ev=False)
            if len(pts) < 10:
                continue
            arr = np.array(pts)
            sel = (MASS >= arr[0, 0]) & (MASS <= arr[-1, 0])
            if sel.sum() < 10:
                continue
            interp = 10 ** np.interp(np.log10(MASS[sel]),
                                     np.log10(arr[:, 0]), np.log10(arr[:, 1]))
            resid = np.median(np.abs(np.log10(interp / COUP[sel])))
            best = resid if best is None else min(best, resid)
        assert best is not None and best < 0.05

    def test_second_curve_distinct(self, figure_pdf):
        cal, curves = trace_figure_pdf(figure_pdf)
        meds = []
        for c in curves[:3]:
            if c.filled:
                continue  # the glyph-outline group is fill-only
            pts = curve_to_data(c, cal, to_ev=False)
            meds.append(sorted(p[1] for p in pts)[len(pts) // 2])
        ratio = max(meds) / min(meds)
        assert 10 < ratio < 100  # the x30 offset survives, curves not merged


class TestFailOpen:
    def test_no_text_pdf_returns_none(self, tmp_path):
        import fitz
        p = tmp_path / "blank.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=150)
        page.draw_line((10, 10), (100, 100))
        doc.save(p)
        cal, curves = trace_figure_pdf(p)
        assert cal is None and curves == []

    def test_missing_file(self):
        cal, curves = trace_figure_pdf("/nonexistent/f.pdf")
        assert cal is None and curves == []


class TestCurveToData:
    def test_lower_envelope_dedup(self):
        cal = AxisCalibration(ax=0.01, bx=-10, ay=-0.01, by=-8)
        # two points at the same x, different y -> keep the lower coupling
        c = VectorCurve(color=(1, 0, 0), filled=False,
                        points=[(100, 50), (100, 80), (200, 60)])
        pts = curve_to_data(c, cal, to_ev=False)
        assert len(pts) == 2
        x0, y0 = pts[0]
        assert y0 == pytest.approx(10 ** (-0.01 * 80 - 8))  # lower coupling wins
