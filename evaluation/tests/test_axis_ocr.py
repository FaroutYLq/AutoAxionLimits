"""Unit tests for the P1 axis-OCR channel (issue #570, P1 of #566).

Covers the parts that need no system binary at all:

* :func:`pipeline.axis_ocr.parse_positive_label` — the four printed label forms
  (power-of-ten, mantissa×10ⁿ, plain decimal/integer, superscript Unicode),
* :func:`pipeline.axis_ocr.interpret_decade_axis` — the 1207.3275 win (bare
  exponents ``-10..0`` on a log axis → ``10**k``),
* optional-import discipline: with the OCR stack absent, ``ocr_axis_labels``
  returns ``[]`` and never raises.

A small real-OCR smoke test runs only when the ``tesseract`` binary +
``pytesseract`` are present (installed in the eval_tests CI job; self-skips
otherwise), so the suite stays green everywhere.

Run:
    pytest evaluation/tests/test_axis_ocr.py -v
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import axis_ocr
from pipeline.axis_ocr import (
    TickLabel,
    interpret_decade_axis,
    parse_int_label,
    parse_positive_label,
)

_HAVE_PYTESS = False
try:  # pragma: no cover
    import pytesseract as _pt  # noqa: F401
    _HAVE_PYTESS = True
except Exception:
    _HAVE_PYTESS = False

requires_ocr = pytest.mark.skipif(
    not (_HAVE_PYTESS and shutil.which("tesseract") is not None),
    reason="pytesseract / tesseract binary not available",
)


# ---------------------------------------------------------------------------
# Numeric label parser — the four printed forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        # power-of-ten, ASCII caret and superscript Unicode
        ("10^-15", 1e-15),
        ("10^{-15}", 1e-15),
        ("10⁻¹⁵", 1e-15),
        ("10-15", 1e-15),
        ("10^5", 1e5),
        ("10^0", 1.0),
        # mantissa × 10ⁿ
        ("2×10⁻⁶", 2e-6),
        ("2*10^-6", 2e-6),
        ("3.0e-9", 3e-9),
        ("1.5x10^3", 1.5e3),
        ("2.10e-33", 2.10e-33),
        # plain decimal / integer — must NOT be read as powers of ten
        ("0.1", 0.1),
        ("30", 30.0),
        ("100", 100.0),
        ("1000", 1000.0),
        ("500", 500.0),
    ],
)
def test_parse_positive_label_forms(text, expected):
    got = parse_positive_label(text)
    assert got is not None, f"{text!r} failed to parse"
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("text", ["", "abc", "-", "x10", "1e", "--5", None])
def test_parse_positive_label_rejects_garbage(text):
    assert parse_positive_label(text) is None


def test_parse_positive_label_rejects_nonpositive():
    # bare negatives / zero are not positive values (the decade axis handles -10)
    assert parse_positive_label("-5") is None
    assert parse_positive_label("0") is None


def test_parse_int_label():
    assert parse_int_label("-10") == -10
    assert parse_int_label("0") == 0
    assert parse_int_label("7") == 7
    assert parse_int_label("10^-3") is None
    assert parse_int_label("1.5") is None


# ---------------------------------------------------------------------------
# Decade-exponent axis (the 1207.3275 win)
# ---------------------------------------------------------------------------

def test_interpret_decade_axis_recovers_powers():
    # bare exponents -10..-6 ordered by pixel -> 1e-10..1e-6
    pix_text = [
        (100.0, "-10", 0.9),
        (200.0, "-9", 0.9),
        (300.0, "-8", 0.9),
        (400.0, "-7", 0.9),
        (500.0, "-6", 0.9),
    ]
    labels = interpret_decade_axis(pix_text)
    assert labels is not None
    assert len(labels) == 5
    vals = sorted(l.value for l in labels)
    assert vals[0] == pytest.approx(1e-10)
    assert vals[-1] == pytest.approx(1e-6)


def test_interpret_decade_axis_rejects_nonmonotonic():
    pix_text = [(100.0, "-10", 0.9), (200.0, "5", 0.9), (300.0, "-3", 0.9)]
    assert interpret_decade_axis(pix_text) is None


def test_interpret_decade_axis_needs_two():
    assert interpret_decade_axis([(100.0, "-10", 0.9)]) is None
    assert interpret_decade_axis([]) is None


def test_interpret_decade_axis_rejects_large_gaps():
    # steps of 5 are not decade ticks
    pix_text = [(100.0, "-20", 0.9), (200.0, "-15", 0.9), (300.0, "-10", 0.9)]
    assert interpret_decade_axis(pix_text) is None


# ---------------------------------------------------------------------------
# Optional-import discipline
# ---------------------------------------------------------------------------

def test_ocr_axis_labels_no_op_without_stack(monkeypatch):
    # Force the unavailable path; must return [] and never raise.
    monkeypatch.setattr(axis_ocr, "_OCR_AVAILABLE", False)
    out = axis_ocr.ocr_axis_labels(object(), (0, 0, 10, 10), "x", [1.0, 2.0, 3.0])
    assert out == []


def test_ocr_axis_labels_guards_short_ticklist():
    out = axis_ocr.ocr_axis_labels(object(), (0, 0, 10, 10), "x", [1.0])
    assert out == []
    out = axis_ocr.ocr_axis_labels(object(), (0, 0, 10, 10), "x", None)
    assert out == []


# ---------------------------------------------------------------------------
# PaddleOCR backend token selection (no real paddle needed)
# ---------------------------------------------------------------------------

def test_paddle_read_prefers_power_of_ten_token(monkeypatch):
    # PaddleOCR may return a stray fragment alongside the real label
    # ("3 10-18"); _paddle_read must pick the power-of-ten token -> 1e-18.
    class _FakeOCR:
        def predict(self, path):
            return [{"rec_texts": ["3", "10-18"], "rec_scores": [0.90, 0.95]}]

    monkeypatch.setattr(axis_ocr, "_get_paddle", lambda: _FakeOCR())
    txt, conf = axis_ocr._paddle_read("/tmp/whatever.png")
    assert txt == "10-18"
    assert axis_ocr.parse_positive_label(txt) == pytest.approx(1e-18)
    assert conf == pytest.approx(0.925, abs=1e-3)


def test_paddle_read_no_op_without_engine(monkeypatch):
    monkeypatch.setattr(axis_ocr, "_get_paddle", lambda: None)
    assert axis_ocr._paddle_read("/tmp/x.png") == ("", 0.0)


# ---------------------------------------------------------------------------
# Real-OCR smoke test (tesseract present only)
# ---------------------------------------------------------------------------

def _find_ttf():
    """Locate a real TrueType font; PIL's default bitmap font OCRs poorly."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    try:  # matplotlib bundles DejaVuSans if it happens to be installed
        from matplotlib import font_manager
        return font_manager.findfont("DejaVu Sans")
    except Exception:
        return None


@requires_ocr
def test_ocr_reads_plain_numeric_crop(tmp_path):
    """Render numeric labels under an x axis at realistic proportions and OCR
    them end to end (genuine tesseract path; runs only when the binary + a real
    TTF font are present)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    ttf = _find_ttf()
    if ttf is None:
        pytest.skip("no TrueType font available for a reliable OCR fixture")
    font = ImageFont.truetype(ttf, 22)

    W, H = 760, 420
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 90, 40, 700, 330
    d.rectangle([x0, y0, x1, y1], outline="black", width=2)
    # Multi-digit labels OCR far more robustly than a bare "1" (-> I/l).
    ticks = [(180, "10"), (340, "100"), (500, "1000"), (640, "5000")]
    for xp, label in ticks:
        d.line([(xp, y1), (xp, y1 + 6)], fill="black", width=2)
        d.text((xp, y1 + 12), label, fill="black", font=font, anchor="ma")
    arr = np.array(img)
    tick_pixels = [xp for xp, _ in ticks]
    labels = axis_ocr.ocr_axis_labels(arr, (x0, y0, x1, y1), "x",
                                      tick_pixels, scale="log")
    # Tesseract should recover at least two of the four labels.
    assert len(labels) >= 2
    for lab in labels:
        assert lab.value > 0
        assert isinstance(lab, TickLabel)
