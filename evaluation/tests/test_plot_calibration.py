"""Unit tests for the P1 axis CV base + OCR calibrator (issue #570, P1 of #566).

Two layers, both needing only ``opencv-python-headless`` + ``numpy`` + ``Pillow``
(no ``tesseract`` binary):

* CV metrology: :func:`detect_plot_region`, :func:`detect_axis_ticks`,
  :func:`build_log_transform`, :func:`split_panels` on synthetic PIL-drawn plots.
* The OCR cross-check logic: :func:`calibrate_axis_ocr` /
  :class:`AxisCalibration`, exercised by **monkeypatching** ``ocr_axis_labels`` to
  return controlled ``TickLabel``\\ s. This pins the corroboration verdict
  (clean→corroborated, one mislocated tick→RANSAC-dropped→still corroborated,
  absurd endpoint→phys-veto→uncorroborated, contradict-LLM→still commits,
  no-consensus→uncorroborated, <2 labels→None) deterministically and portably.

Run:
    pytest evaluation/tests/test_plot_calibration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from pipeline import plot_calibration as pc  # noqa: E402
from pipeline.axis_ocr import TickLabel  # noqa: E402
from pipeline.plot_calibration import AxisCalibration  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _framed_plot(tmp_path, n_ticks=5):
    """Draw a single-panel plot frame with clear x/y tick marks -> path + bbox."""
    W, H = 600, 400
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 100, 60, 520, 330
    d.rectangle([x0, y0, x1, y1], outline="black", width=2)
    xs = [x0 + int((i + 1) * (x1 - x0) / (n_ticks + 1)) for i in range(n_ticks)]
    ys = [y0 + int((i + 1) * (y1 - y0) / (n_ticks + 1)) for i in range(n_ticks)]
    for xp in xs:                       # x ticks: short marks below the bottom spine
        d.line([(xp, y1), (xp, y1 + 9)], fill="black", width=2)
    for yp in ys:                       # y ticks: short marks left of the left spine
        d.line([(x0 - 9, yp), (x0, yp)], fill="black", width=2)
    p = tmp_path / "plot.png"
    img.save(p)
    return p, (x0, y0, x1, y1), xs, ys


def _dummy_panel(h=400, w=600):
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# build_log_transform
# ---------------------------------------------------------------------------

def test_build_log_transform_recovers_log_axis():
    t = pc.build_log_transform([100.0, 200.0, 300.0], [1e0, 1e1, 1e2], "log")
    assert t is not None
    assert t(100.0) == pytest.approx(1e0, rel=1e-6)
    assert t(300.0) == pytest.approx(1e2, rel=1e-6)
    assert t(250.0) == pytest.approx(10 ** 1.5, rel=1e-6)


def test_build_log_transform_rejects_inconsistent_ticks():
    # not collinear in log space -> r2 < 0.95 -> None
    assert pc.build_log_transform([100.0, 200.0, 300.0], [1.0, 10.0, 5.0], "log") is None


def test_build_log_transform_rejects_nonpositive_on_log():
    assert pc.build_log_transform([100.0, 200.0], [1.0, -10.0], "log") is None


def test_build_log_transform_needs_two_points():
    assert pc.build_log_transform([100.0], [1.0], "log") is None


# ---------------------------------------------------------------------------
# CV detection on synthetic plots
# ---------------------------------------------------------------------------

def test_detect_plot_region(tmp_path):
    p, bbox, _xs, _ys = _framed_plot(tmp_path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    got = pc.detect_plot_region(img)
    assert got is not None
    gx0, gy0, gx1, gy1 = got
    bx0, by0, bx1, by1 = bbox
    # within ~12 px of the drawn frame
    assert abs(gx0 - bx0) < 12 and abs(gy0 - by0) < 12
    assert abs(gx1 - bx1) < 12 and abs(gy1 - by1) < 12


def test_detect_axis_ticks(tmp_path):
    p, bbox, xs, ys = _framed_plot(tmp_path, n_ticks=5)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    ticks = pc.detect_axis_ticks(img, bbox)
    assert ticks is not None
    assert isinstance(ticks["x"], list) and isinstance(ticks["y"], list)
    assert len(ticks["x"]) >= 3
    assert len(ticks["y"]) >= 3


def test_split_panels_single_panel_passthrough(tmp_path):
    p, _bbox, _xs, _ys = _framed_plot(tmp_path)
    out = pc.split_panels(p)
    assert out == [p]


# ---------------------------------------------------------------------------
# _robust_fit (RANSAC) — drops a single bad tick
# ---------------------------------------------------------------------------

def test_robust_fit_clean():
    px = [100.0, 200.0, 300.0, 400.0, 500.0]
    vals = [1e0, 1e1, 1e2, 1e3, 1e4]
    fit = pc._robust_fit(px, vals, "log")
    assert fit is not None
    a, b, r2, inliers = fit
    assert r2 == pytest.approx(1.0, abs=1e-6)
    assert len(inliers) == 5


def test_robust_fit_drops_one_outlier():
    px = [100.0, 200.0, 300.0, 400.0, 500.0]
    vals = [1e0, 1e1, 10 ** 2.6, 1e3, 1e4]   # 300px tick is 0.6 dex off
    fit = pc._robust_fit(px, vals, "log")
    assert fit is not None
    a, b, r2, inliers = fit
    assert len(inliers) == 4                 # the outlier is excluded
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_robust_fit_no_consensus():
    px = [100.0, 200.0, 300.0, 400.0, 500.0]
    vals = [1e0, 1e1, 1e3, 1e6, 1e10]        # convex, no 3 collinear
    assert pc._robust_fit(px, vals, "log") is None


# ---------------------------------------------------------------------------
# calibrate_axis_ocr — corroboration verdict (OCR monkeypatched)
# ---------------------------------------------------------------------------

def _patch_ocr(monkeypatch, labels):
    monkeypatch.setattr(pc.axis_ocr, "ocr_axis_labels", lambda *a, **k: labels)


def test_calibrate_clean_axis_corroborated(monkeypatch):
    labels = [
        TickLabel(100.0, 1e0, "1", 0.9),
        TickLabel(200.0, 1e1, "10", 0.9),
        TickLabel(300.0, 1e2, "100", 0.9),
        TickLabel(400.0, 1e3, "1000", 0.9),
        TickLabel(500.0, 1e4, "10000", 0.9),
    ]
    _patch_ocr(monkeypatch, labels)
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "x", "log",
        [100.0, 200.0, 300.0, 400.0, 500.0], (1e0, 1e4),
        valid_range=(1e-25, 1e10),
    )
    assert cal is not None
    assert cal.corroborated is True
    assert cal.ocr_geom_agree is True
    assert cal.n_labels == 5
    assert cal.ocr_vs_llm_dex == pytest.approx(0.0, abs=1e-6)
    assert cal.ocr_min == pytest.approx(1e0, rel=1e-6)
    assert cal.ocr_max == pytest.approx(1e4, rel=1e-6)


def test_calibrate_drops_mislocated_tick(monkeypatch):
    # 2212.01139 mechanism: one bottom tick mislocated; RANSAC drops it.
    labels = [
        TickLabel(100.0, 1e0, "1", 0.9),
        TickLabel(200.0, 1e1, "10", 0.9),
        TickLabel(300.0, 10 ** 2.6, "100?", 0.9),   # off by 0.6 dex
        TickLabel(400.0, 1e3, "1000", 0.9),
        TickLabel(500.0, 1e4, "10000", 0.9),
    ]
    _patch_ocr(monkeypatch, labels)
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "x", "log",
        [100.0, 200.0, 300.0, 400.0, 500.0], (1e0, 1e4),
        valid_range=(1e-25, 1e10),
    )
    assert cal is not None
    assert cal.corroborated is True
    # endpoints still come out clean despite the bad middle tick
    assert cal.ocr_max == pytest.approx(1e4, rel=1e-3)


def test_calibrate_absurd_endpoint_vetoed(monkeypatch):
    # 1506.08082 mechanism: collinear labels but an impossible endpoint.
    labels = [
        TickLabel(50.0, 1e2, "100", 0.9),
        TickLabel(150.0, 1e0, "1", 0.9),
        TickLabel(250.0, 1e-2, "0.01", 0.9),
        TickLabel(350.0, 1e-4, "1e-4", 0.9),
        TickLabel(450.0, 1e-6, "1e-6", 0.9),
    ]
    _patch_ocr(monkeypatch, labels)
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "y", "log",
        [50.0, 150.0, 250.0, 350.0, 450.0], (1e-6, 1e2),
        valid_range=(1e-26, 1e-2),   # widened AxionPhoton coupling: 1e2 is impossible
    )
    assert cal is not None
    assert cal.ocr_geom_agree is True       # the fit is self-consistent...
    assert cal.endpoint_phys_ok is False    # ...but the endpoint is unphysical
    assert cal.corroborated is False        # so it does NOT corroborate


def test_calibrate_contradicts_llm_but_commits(monkeypatch):
    # 2402.12892 mechanism: OCR contradicts an eyeballed LLM axis, and commits.
    labels = [
        TickLabel(100.0, 1e0, "1", 0.9),
        TickLabel(200.0, 1e1, "10", 0.9),
        TickLabel(300.0, 1e2, "100", 0.9),
        TickLabel(400.0, 1e3, "1000", 0.9),
        TickLabel(500.0, 1e4, "10000", 0.9),
    ]
    _patch_ocr(monkeypatch, labels)
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "x", "log",
        [100.0, 200.0, 300.0, 400.0, 500.0], (1e0, 30.0),  # LLM eyeballed max 30
        valid_range=(1e-25, 1e10),
    )
    assert cal is not None
    assert cal.ocr_vs_llm_dex > 0.5         # contradicts the LLM
    assert cal.corroborated is True         # but OCR+geom agree + phys-ok => commit


def test_calibrate_no_consensus_uncorroborated(monkeypatch):
    labels = [
        TickLabel(100.0, 1e0, "1", 0.9),
        TickLabel(200.0, 1e1, "10", 0.9),
        TickLabel(300.0, 1e3, "1000", 0.9),
        TickLabel(400.0, 1e6, "1e6", 0.9),
        TickLabel(500.0, 1e10, "1e10", 0.9),
    ]
    _patch_ocr(monkeypatch, labels)
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "x", "log",
        [100.0, 200.0, 300.0, 400.0, 500.0], (1e0, 1e10),
        valid_range=(1e-25, 1e15),
    )
    assert cal is not None
    assert cal.transform is None
    assert cal.corroborated is False
    assert "no consensus" in cal.note


def test_calibrate_too_few_labels_returns_none(monkeypatch):
    _patch_ocr(monkeypatch, [TickLabel(100.0, 1e0, "1", 0.9)])
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "x", "log",
        [100.0, 200.0], (1e0, 1e4),
    )
    assert cal is None


# ---------------------------------------------------------------------------
# AxisCalibration.corroborated truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "geom,n,r2,phys,expected",
    [
        (True, 5, 0.99, True, True),
        (False, 5, 0.99, True, False),   # geom disagree
        (True, 1, 0.99, True, False),    # too few labels
        (True, 5, 0.90, True, False),    # poor fit
        (True, 5, 0.99, False, False),   # unphysical endpoint
    ],
)
def test_axis_calibration_corroborated_truth_table(geom, n, r2, phys, expected):
    cal = AxisCalibration(
        transform=None, ocr_min=1.0, ocr_max=1e4,
        ocr_geom_agree=geom, ocr_vs_llm_dex=0.1, n_labels=n,
        fit_r2=r2, endpoint_phys_ok=phys, note="",
    )
    assert cal.corroborated is expected


# ---------------------------------------------------------------------------
# Optional-import discipline
# ---------------------------------------------------------------------------

def test_calibrate_no_op_without_cv(monkeypatch):
    monkeypatch.setattr(pc, "_CV_AVAILABLE", False)
    cal = pc.calibrate_axis_ocr(
        _dummy_panel(), (100, 50, 500, 450), "x", "log", [100.0, 200.0], (1e0, 1e4)
    )
    assert cal is None
