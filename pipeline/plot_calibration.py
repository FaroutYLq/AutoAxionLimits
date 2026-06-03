"""
Classical computer-vision support for figure-axis metrology (roadmap #566, P1 / #570).

Scope on master = the *axis-metrology* slice only. The #550 curve tracer
(``trace_curve`` and the CV-trace override gate) is deliberately NOT revived here:
that path drove the #550 blow-ups and is not needed to read an axis. This module
provides exactly what axis calibration needs:

  * ``detect_plot_region`` / ``detect_axis_ticks`` — pixel positions of the plot
    frame and the tick marks (metrology).
  * ``build_log_transform`` — the validated pixel -> data fit (``r2>=0.95`` guard).
  * ``split_panels`` — multi-panel figure splitting.
  * ``calibrate_axis_ocr`` + :class:`AxisCalibration` — fuse the OCR'd tick labels
    (``axis_ocr``) with the tick pixels into a *corroborated* axis verdict that
    ``transform_guard.guard_transform`` (P0's R2) commits-or-reverts.

The #550 calibrator trusted the LLM tick *values* and paired them to CV pixels by
index (``n = min(len(pix), len(vals))``); a mispaired or mislocated tick still fit
a line with ``r2>=0.95``, so a 12-/24-decade endpoint shift passed. P1 fixes this
by reading the labels themselves (OCR) so each (pixel, value) pair comes from the
same physical tick, and by a robust (RANSAC-style) fit that drops a single bad
tick instead of letting it drag the whole axis.

Every public function is defensive: it returns ``None`` (or passes the input
through, for ``split_panels``) on any failure and never raises. The caller in
``extractor.py`` must always be able to fall back to the pure-LLM path with no
regression below P0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import axis_ocr

logger = logging.getLogger(__name__)

# OpenCV / numpy are optional at import time so that importing this module never
# breaks the pipeline if the wheel is missing. All functions degrade to a no-op.
try:  # pragma: no cover - exercised implicitly
    import cv2  # type: ignore
    import numpy as np

    _CV_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV_AVAILABLE = False
    logger.warning("opencv/numpy unavailable; plot_calibration is a no-op: %s", _e)


# Type aliases for readability
BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixel coords

# Self-consistency tolerance for the OCR labels vs the geometric fit (dex).
GEOM_AGREE_DEX = 0.15
# RANSAC inlier residual (dex) and minimum inlier fraction.
RANSAC_INLIER_DEX = 0.15
RANSAC_MIN_INLIER_FRAC = 0.60


def _read_gray(image_path: Path):
    """Load an image as a grayscale uint8 array, or None on failure."""
    if not _CV_AVAILABLE:
        return None
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Panel splitting
# ---------------------------------------------------------------------------

def split_panels(image_path: Path, min_panel_frac: float = 0.18) -> list[Path]:
    """Split a multi-panel figure into single-panel crops.

    Detects panel boundaries by looking for wide, near-white gutters in the
    horizontal and vertical projection profiles of the inked (dark) pixels.
    Returns ``[image_path]`` unchanged if the figure looks single-panel or on
    ANY failure (never raises).
    """
    if not _CV_AVAILABLE:
        return [image_path]
    try:
        gray = _read_gray(image_path)
        if gray is None:
            return [image_path]
        h, w = gray.shape
        if h < 60 or w < 60:
            return [image_path]

        ink = (gray < 200).astype(np.uint8)

        def _find_gutters(profile, length, axis_min_frac=min_panel_frac):
            thresh = max(1, int(0.004 * profile.max())) if profile.max() > 0 else 1
            empty = profile <= thresh
            cuts = []
            i = 0
            min_gutter = max(8, int(0.02 * length))
            while i < length:
                if empty[i]:
                    j = i
                    while j < length and empty[j]:
                        j += 1
                    run_len = j - i
                    if run_len >= min_gutter and i > 0 and j < length:
                        cuts.append((i + j) // 2)
                    i = j
                else:
                    i += 1
            return cuts

        col_ink = ink.sum(axis=0)
        row_ink = ink.sum(axis=1)
        v_cuts = _find_gutters(col_ink, w)
        h_cuts = _find_gutters(row_ink, h)

        def _segments(cuts, length):
            bounds = [0] + sorted(cuts) + [length]
            segs = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                if (b - a) >= int(min_panel_frac * length):
                    segs.append((a, b))
            return segs

        x_segs = _segments(v_cuts, w)
        y_segs = _segments(h_cuts, h)
        if len(x_segs) <= 1 and len(y_segs) <= 1:
            return [image_path]
        x_segs = x_segs or [(0, w)]
        y_segs = y_segs or [(0, h)]

        color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if color is None:
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        out_paths: list[Path] = []
        idx = 0
        stem = image_path.stem
        for (y0, y1) in y_segs:
            for (x0, x1) in x_segs:
                crop = color[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                cink = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 200).mean()
                if cink < 0.002:
                    continue
                out = image_path.parent / f"{stem}_panel{idx:02d}.png"
                cv2.imwrite(str(out), crop)
                out_paths.append(out)
                idx += 1

        if len(out_paths) <= 1:
            return [image_path]
        logger.info("split_panels: %s -> %d panels", image_path.name, len(out_paths))
        return out_paths
    except Exception as e:
        logger.warning("split_panels failed for %s: %s", image_path, e)
        return [image_path]


# ---------------------------------------------------------------------------
# Plot region detection
# ---------------------------------------------------------------------------

def detect_plot_region(img) -> Optional[BBox]:
    """Find the bounding box of the main plot/axes rectangle.

    Detects long horizontal/vertical line segments (the axis spines / frame) via
    morphology on the binarised image, then takes the bounding box of the
    dominant frame. Falls back to the largest near-rectangular contour. Returns
    ``(x0, y0, x1, y1)`` or ``None`` on failure.
    """
    if not _CV_AVAILABLE or img is None:
        return None
    try:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = gray.shape
        if h < 40 or w < 40:
            return None

        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        hor_len = max(15, w // 3)
        ver_len = max(15, h // 3)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_kernel)
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_kernel)
        frame = cv2.bitwise_or(horiz, vert)

        ys, xs = np.where(frame > 0)
        if xs.size >= 8 and ys.size >= 8:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if (x1 - x0) / w > 0.3 and (y1 - y0) / h > 0.3:
                return (x0, y0, x1, y1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        x, y, bw_, bh_ = cv2.boundingRect(best)
        if bw_ < 0.3 * w or bh_ < 0.3 * h:
            return None
        return (x, y, x + bw_, y + bh_)
    except Exception as e:
        logger.warning("detect_plot_region failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Tick detection
# ---------------------------------------------------------------------------

def detect_axis_ticks(img, bbox: BBox) -> Optional[dict]:
    """Locate tick-mark pixel positions along the x and y axes.

    Projection profile of ink in a thin band adjacent to each spine, with peak
    detection to recover tick centres. Returns ``{"x": [...], "y": [...]}``
    (sorted pixel positions) or ``None`` on failure. Lists may be empty.
    """
    if not _CV_AVAILABLE or img is None or bbox is None:
        return None
    try:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = gray.shape
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None

        ink = (gray < 128).astype(np.uint8)

        def _peaks(profile, min_sep):
            if profile.max() <= 0:
                return []
            thr = max(1, 0.35 * profile.max())
            cand = np.where(profile >= thr)[0]
            if cand.size == 0:
                return []
            peaks = []
            grp_start = cand[0]
            prev = cand[0]
            for c in cand[1:]:
                if c - prev > min_sep:
                    peaks.append(int(round((grp_start + prev) / 2)))
                    grp_start = c
                prev = c
            peaks.append(int(round((grp_start + prev) / 2)))
            return peaks

        band = max(3, int(0.03 * (y1 - y0)))
        xb0, xb1 = y1, min(h, y1 + band)
        if xb1 > xb0:
            xband = ink[xb0:xb1, x0:x1].sum(axis=0)
            x_ticks = [x0 + p for p in _peaks(xband, min_sep=max(3, (x1 - x0) // 60))]
        else:
            x_ticks = []

        yb0, yb1 = max(0, x0 - band), x0
        if yb1 > yb0:
            yband = ink[y0:y1, yb0:yb1].sum(axis=1)
            y_ticks = [y0 + p for p in _peaks(yband, min_sep=max(3, (y1 - y0) // 60))]
        else:
            y_ticks = []

        if not x_ticks:
            inner = ink[y0:y1, x0:x1].sum(axis=0)
            x_ticks = [x0 + p for p in _peaks(inner, min_sep=max(3, (x1 - x0) // 60))]
        if not y_ticks:
            inner = ink[y0:y1, x0:x1].sum(axis=1)
            y_ticks = [y0 + p for p in _peaks(inner, min_sep=max(3, (y1 - y0) // 60))]

        return {"x": sorted(x_ticks), "y": sorted(y_ticks)}
    except Exception as e:
        logger.warning("detect_axis_ticks failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Pixel -> data transform
# ---------------------------------------------------------------------------

def build_log_transform(
    tick_pixels: list[float],
    tick_values: list[float],
    scale: str,
) -> Optional[Callable[[float], float]]:
    """Fit a pixel -> data-coordinate mapping from measured ticks.

    ``scale == "log"``: linear fit of ``log10(value)`` vs ``pixel`` -> callable
    maps a pixel to ``10 ** (a*pixel + b)``. ``scale == "linear"``: linear fit of
    ``value`` vs ``pixel``. Returns ``None`` on inconsistent inputs, a degenerate
    fit, or a fit that does not track the data (``r2 < 0.95``).
    """
    if not _CV_AVAILABLE:
        return None
    try:
        if tick_pixels is None or tick_values is None:
            return None
        n = min(len(tick_pixels), len(tick_values))
        if n < 2:
            return None
        px = np.asarray(tick_pixels[:n], dtype=float)
        val = np.asarray(tick_values[:n], dtype=float)
        good = np.isfinite(px) & np.isfinite(val)
        px, val = px[good], val[good]
        if px.size < 2:
            return None

        scale = (scale or "log").lower()
        is_log = scale.startswith("log")
        if is_log:
            if np.any(val <= 0):
                return None
            y = np.log10(val)
        else:
            y = val

        if np.ptp(px) < 1e-6 or np.ptp(y) < 1e-9:
            return None

        A = np.vstack([px, np.ones_like(px)]).T
        (a, b), residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
        if rank < 2 or not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12:
            return None

        pred = a * px + b
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
            if r2 < 0.95:
                logger.warning("build_log_transform: poor fit (R^2=%.3f); rejecting", r2)
                return None

        if is_log:
            def _transform(pixel: float, _a=a, _b=b) -> float:
                return float(10.0 ** (_a * float(pixel) + _b))
        else:
            def _transform(pixel: float, _a=a, _b=b) -> float:
                return float(_a * float(pixel) + _b)

        return _transform
    except Exception as e:
        logger.warning("build_log_transform failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# OCR-corroborated axis calibration (P1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisCalibration:
    """An OCR-corroborated pixel->data axis fit + the corroboration verdict.

    ``.corroborated`` is the signal P0's ``guard_transform`` consumes: P1 changes
    the *meaning* of corroborated from the #550 weak ``r2>=0.95`` proxy to "the
    OCR'd labels and the geometric fit agree with each other and land in a
    physically possible range".
    """

    transform: Optional[Callable[[float], float]]  # pixel -> data (validated fit) or None
    ocr_min: Optional[float]        # OCR-implied axis minimum (data coords)
    ocr_max: Optional[float]        # OCR-implied axis maximum (data coords)
    ocr_geom_agree: bool            # OCR labels vs geometric fit: median |Δ| <= GEOM_AGREE_DEX
    ocr_vs_llm_dex: float           # max endpoint |log10(OCR) - log10(LLM)|; inf if unknown
    n_labels: int                   # parseable OCR labels used
    fit_r2: float                   # fit quality on the inlier set
    endpoint_phys_ok: bool          # both OCR endpoints inside the widened VALID_RANGES
    note: str

    @property
    def corroborated(self) -> bool:
        return bool(
            self.ocr_geom_agree
            and self.n_labels >= 2
            and self.fit_r2 >= 0.95
            and self.endpoint_phys_ok
        )


def _robust_fit(pixels, values, scale: str):
    """RANSAC-style consensus line fit over (pixel, value) pairs.

    Returns ``(a, b, r2, inlier_idx)`` in the transformed space (``log10(value)``
    for a log axis), or ``None``. Drops a single mislocated/misread tick (the
    2212.01139 mechanism) instead of letting it drag the fit. Requires >=2
    inliers and >= ``RANSAC_MIN_INLIER_FRAC`` of the labels as inliers.
    """
    if not _CV_AVAILABLE:
        return None
    px = np.asarray(pixels, dtype=float)
    val = np.asarray(values, dtype=float)
    is_log = (scale or "log").lower().startswith("log")
    if is_log:
        if np.any(val <= 0):
            return None
        t = np.log10(val)
    else:
        t = val
    n = px.size
    if n < 2 or np.ptp(px) < 1e-6:
        return None

    best_inliers = None
    for i in range(n):
        for j in range(i + 1, n):
            if abs(px[j] - px[i]) < 1e-6:
                continue
            a = (t[j] - t[i]) / (px[j] - px[i])
            b = t[i] - a * px[i]
            resid = np.abs(t - (a * px + b))
            inliers = np.where(resid <= RANSAC_INLIER_DEX)[0]
            if best_inliers is None or inliers.size > best_inliers.size:
                best_inliers = inliers
    if best_inliers is None or best_inliers.size < 2:
        return None
    if best_inliers.size < math.ceil(RANSAC_MIN_INLIER_FRAC * n):
        return None

    ip, it = px[best_inliers], t[best_inliers]
    A = np.vstack([ip, np.ones_like(ip)]).T
    (a, b), _, rank, _ = np.linalg.lstsq(A, it, rcond=None)
    if rank < 2 or not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12:
        return None
    pred = a * ip + b
    ss_res = float(np.sum((it - pred) ** 2))
    ss_tot = float(np.sum((it - it.mean()) ** 2))
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2), best_inliers


def calibrate_axis_ocr(
    panel_img,
    bbox: BBox,
    axis: str,
    scale: str,
    tick_pixels: list[float],
    llm_values: tuple[float, float],
    valid_range: Optional[tuple[float, float]] = None,
) -> Optional["AxisCalibration"]:
    """OCR the tick labels, robustly fit pixel->value, and cross-validate.

    ``llm_values`` is the LLM-supplied ``(axis_min, axis_max)`` for this axis (the
    channel we are checking). ``valid_range`` is the *already widened*
    ``(lo, hi)`` physical bound for this axis quantity (caller widens via
    ``VALID_RANGES``); ``None`` skips the physical-plausibility check.

    Returns an :class:`AxisCalibration` carrying the validated transform and the
    corroboration verdict, or ``None`` when OCR finds < 2 parseable labels (no
    independent channel -> caller keeps the LLM axis with ``corroborated=False``).
    """
    if not _CV_AVAILABLE or panel_img is None or bbox is None:
        return None
    try:
        labels = axis_ocr.ocr_axis_labels(panel_img, bbox, axis, tick_pixels, scale)
        if len(labels) < 2:
            return None
        n_labels = len(labels)
        pixels = [lab.pixel for lab in labels]
        values = [lab.value for lab in labels]

        fit = _robust_fit(pixels, values, scale)
        if fit is None:
            # Labels could not agree on one line (the 1506.08082 / 1907.05475
            # mechanism): report an uncorroborated calibration so P0's R2 reverts.
            return AxisCalibration(
                transform=None, ocr_min=None, ocr_max=None,
                ocr_geom_agree=False, ocr_vs_llm_dex=float("inf"),
                n_labels=n_labels, fit_r2=0.0, endpoint_phys_ok=False,
                note=f"ocr={n_labels} labels, geom_agree=N (no consensus fit)",
            )
        a, b, r2, _inliers = fit
        is_log = (scale or "log").lower().startswith("log")

        def _val(pixel: float) -> float:
            return 10.0 ** (a * pixel + b) if is_log else (a * pixel + b)

        # ocr_geom_agree: median over ALL labels of |log10(value) - log10(fit(px))|.
        diffs = []
        for lab in labels:
            fv = _val(lab.pixel)
            if is_log and fv > 0 and lab.value > 0:
                diffs.append(abs(math.log10(lab.value) - math.log10(fv)))
            elif not is_log:
                span = max(abs(max(values) - min(values)), 1e-9)
                diffs.append(abs(lab.value - fv) / span)
        diffs.sort()
        median_diff = diffs[len(diffs) // 2] if diffs else float("inf")
        ocr_geom_agree = median_diff <= GEOM_AGREE_DEX

        # OCR-implied axis endpoints from the plot frame extent.
        x0, y0, x1, y1 = bbox
        if axis == "x":
            e_a, e_b = _val(x0), _val(x1)
        else:  # y image-pixels are inverted (top = max)
            e_a, e_b = _val(y0), _val(y1)
        ocr_lo, ocr_hi = (min(e_a, e_b), max(e_a, e_b))

        # ocr_vs_llm_dex (log endpoints).
        llm_lo, llm_hi = sorted(float(v) for v in llm_values)
        if all(v > 0 for v in (ocr_lo, ocr_hi, llm_lo, llm_hi)):
            ocr_vs_llm_dex = max(
                abs(math.log10(ocr_lo) - math.log10(llm_lo)),
                abs(math.log10(ocr_hi) - math.log10(llm_hi)),
            )
        else:
            ocr_vs_llm_dex = float("inf")

        # endpoint_phys_ok against the (already widened) valid range.
        if valid_range is None:
            endpoint_phys_ok = True
        else:
            lo_w, hi_w = valid_range
            endpoint_phys_ok = (ocr_lo >= lo_w) and (ocr_hi <= hi_w)

        transform = build_log_transform(
            [pixels[i] for i in _inliers], [values[i] for i in _inliers], scale
        )

        cal = AxisCalibration(
            transform=transform, ocr_min=ocr_lo, ocr_max=ocr_hi,
            ocr_geom_agree=ocr_geom_agree, ocr_vs_llm_dex=float(ocr_vs_llm_dex),
            n_labels=n_labels, fit_r2=r2, endpoint_phys_ok=endpoint_phys_ok,
            note=(
                f"ocr={n_labels} labels, geom_agree={'Y' if ocr_geom_agree else 'N'}, "
                f"vs_llm={ocr_vs_llm_dex:.2f}dex, phys_ok="
                f"{'Y' if endpoint_phys_ok else 'N'}, r2={r2:.3f}"
            ),
        )
        return cal
    except Exception as e:
        logger.warning("calibrate_axis_ocr failed: %s", e)
        return None
