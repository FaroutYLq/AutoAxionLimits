"""Wiring tests for the P1 axis-OCR integration in the extractor (#570, 2/2).

Pins the guard semantics of :func:`pipeline.extractor._attach_cv_calibration`
(no API, no real OpenCV/OCR — ``calibrate_figure_axes`` is monkeypatched):

* a corroborated OCR read that contradicts the LLM axis (> R2 trigger) commits
  the corrected range and records a ``corroborated=Y`` note;
* an uncorroborated read, or one that merely agrees with the LLM (<= R2 trigger),
  keeps the LLM axis untouched;
* the P1 invariant: an axis is overwritten ONLY when corroborated;
* OCR/CV absent => complete no-op.

Run:
    pytest evaluation/tests/test_extractor_axis_ocr.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_HAVE_STACK = True
try:
    from pipeline import extractor
    from pipeline import plot_calibration as pc
    from pipeline.plot_calibration import AxisCalibration
except Exception:
    _HAVE_STACK = False

requires_stack = pytest.mark.skipif(
    not _HAVE_STACK,
    reason="pipeline.extractor / plot_calibration (anthropic, cv2) unavailable",
)


def _cal(*, corroborated, dex, ocr_min, ocr_max, phys=True):
    """Build an AxisCalibration whose .corroborated equals `corroborated`."""
    return AxisCalibration(
        transform=None, ocr_min=ocr_min, ocr_max=ocr_max,
        ocr_geom_agree=corroborated, ocr_vs_llm_dex=dex, n_labels=5,
        fit_r2=0.99 if corroborated else 0.5, endpoint_phys_ok=phys,
        note=f"ocr=5 labels, geom_agree={'Y' if corroborated else 'N'}",
    )


def _axis_info():
    return {
        "found_exclusion_plot": True,
        "plot_page_index": 0,
        "x_axis_min": 1.0, "x_axis_max": 30.0, "x_axis_scale": "log",
        "y_axis_min": 1e-6, "y_axis_max": 1e-2, "y_axis_scale": "log",
    }


@requires_stack
def test_corroborated_contradiction_commits(monkeypatch):
    # 2402.12892 shape: OCR says x-max ~124 (here 1e4), LLM eyeballed 30.
    monkeypatch.setattr(pc, "_CV_AVAILABLE", True)
    monkeypatch.setattr(
        pc, "calibrate_figure_axes",
        lambda *a, **k: {"panel": Path("p"),
                         "x": _cal(corroborated=True, dex=2.5, ocr_min=1.0, ocr_max=1e4),
                         "y": None},
    )
    ai = _axis_info()
    extractor._attach_cv_calibration(ai, [Path("fig.png")], "AxionPhoton")
    assert ai["x_axis_max"] == pytest.approx(1e4)
    assert ai["x_axis_min"] == pytest.approx(1.0)
    assert "corroborated=Y" in ai["cv_calibration_note"]


@requires_stack
def test_uncorroborated_keeps_llm(monkeypatch):
    monkeypatch.setattr(pc, "_CV_AVAILABLE", True)
    monkeypatch.setattr(
        pc, "calibrate_figure_axes",
        lambda *a, **k: {"panel": Path("p"),
                         "x": _cal(corroborated=False, dex=2.5, ocr_min=1.0, ocr_max=1e4),
                         "y": None},
    )
    ai = _axis_info()
    extractor._attach_cv_calibration(ai, [Path("fig.png")], "AxionPhoton")
    assert ai["x_axis_max"] == pytest.approx(30.0)   # unchanged
    assert "corroborated=N" in ai["cv_calibration_note"]


@requires_stack
def test_corroborated_but_agrees_is_noop(monkeypatch):
    # OCR agrees with the LLM within the R2 trigger -> no override.
    monkeypatch.setattr(pc, "_CV_AVAILABLE", True)
    monkeypatch.setattr(
        pc, "calibrate_figure_axes",
        lambda *a, **k: {"panel": Path("p"),
                         "x": _cal(corroborated=True, dex=0.2, ocr_min=1.0, ocr_max=32.0),
                         "y": None},
    )
    ai = _axis_info()
    extractor._attach_cv_calibration(ai, [Path("fig.png")], "AxionPhoton")
    assert ai["x_axis_max"] == pytest.approx(30.0)   # kept LLM (agree)


@requires_stack
def test_no_op_without_cv(monkeypatch):
    monkeypatch.setattr(pc, "_CV_AVAILABLE", False)
    ai = _axis_info()
    extractor._attach_cv_calibration(ai, [Path("fig.png")], "AxionPhoton")
    assert ai["x_axis_max"] == pytest.approx(30.0)
    assert "cv_calibration_note" not in ai


@requires_stack
def test_no_op_when_no_calibration(monkeypatch):
    monkeypatch.setattr(pc, "_CV_AVAILABLE", True)
    monkeypatch.setattr(pc, "calibrate_figure_axes", lambda *a, **k: None)
    ai = _axis_info()
    extractor._attach_cv_calibration(ai, [Path("fig.png")], "AxionPhoton")
    assert ai["x_axis_max"] == pytest.approx(30.0)
    assert "cv_calibration_note" not in ai


@requires_stack
def test_missing_figures_is_noop(monkeypatch):
    monkeypatch.setattr(pc, "_CV_AVAILABLE", True)
    ai = _axis_info()
    extractor._attach_cv_calibration(ai, [], "AxionPhoton")
    assert ai["x_axis_max"] == pytest.approx(30.0)
