"""
Independent OCR channel for axis tick LABELS (roadmap #566, phase P1 / #570).

This is the third, independent channel in the axis-metrology cross-check:

  * The LLM (vision model) supplies the tick *values* (semantics).
  * ``plot_calibration`` (OpenCV) supplies the tick *pixel positions* (metrology).
  * THIS module reads the tick *labels* straight from the figure pixels (OCR),
    so the (pixel, value) pairing is *measured*, not taken on faith.

Reading the labels directly removes the index-alignment step that let a spurious
or mislocated tick corrupt the pixel<->value pairing in the #550 calibrator
(`_calibrate_axis`'s ``n = min(len(pix), len(vals))`` prefix pairing), which is
how a 12- or 24-decade axis blow-up sailed through an ``r2>=0.95`` gate. The
calibrator in ``plot_calibration.calibrate_axis_ocr`` consumes the
``TickLabel`` pairs this module returns.

Optional-dependency discipline (mirrors ``plot_calibration``): ``pytesseract`` /
``Pillow`` are imported at module top behind a ``_OCR_AVAILABLE`` flag. If either
the wheel or the system ``tesseract`` binary is missing, every public function
degrades to ``[]`` / ``None`` and never raises, so the caller falls back to the
pure-LLM axis path with no regression below P0.
"""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# pytesseract + Pillow are optional at import time. Pillow ships as a pytesseract
# dependency, so they are present or absent together; numpy comes with the CV
# stack. Any import failure makes this module a no-op.
try:  # pragma: no cover - exercised implicitly
    import numpy as np
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    np = None  # type: ignore
    pytesseract = None  # type: ignore
    Image = None  # type: ignore
    _OCR_AVAILABLE = False
    logger.warning("pytesseract/Pillow unavailable; axis_ocr is a no-op: %s", _e)


@dataclass(frozen=True)
class TickLabel:
    """One OCR-read tick label paired with the tick it sits under.

    ``pixel`` and ``value`` come from the *same physical tick* (the label is
    cropped at that tick), so there is no index-alignment step to corrupt.
    """

    pixel: float        # tick centre pixel (from detect_axis_ticks)
    value: float        # parsed numeric value from OCR (finite, positive)
    raw_text: str       # raw OCR string (for the note / debugging)
    ocr_conf: float     # tesseract per-label confidence, 0..1


# Default confidence floor; labels below this are discarded.
CONF_FLOOR = 0.50

# Superscript Unicode -> ASCII (for "10⁻¹⁵" style power-of-ten labels).
_SUPERSCRIPT = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
}
_SUPERSCRIPT_TABLE = {ord(k): v for k, v in _SUPERSCRIPT.items()}


def _normalize_text(text: str) -> str:
    """NFKC-normalize and map superscripts/× to ASCII for parsing."""
    s = unicodedata.normalize("NFKC", text or "")
    s = s.translate(_SUPERSCRIPT_TABLE)
    s = s.replace("×", "x").replace("−", "-")  # × and Unicode minus
    s = s.replace("{", "").replace("}", "").replace(" ", "")
    return s.strip()


# 2x10^-6 / 3.0e-9 / 2*10^-6 / 10^-15 / 10-15 / 0.1 / 30
_RE_POW10 = re.compile(r"^([+-]?\d*\.?\d*)(?:x|\*)?10\^?([+-]?\d+)$", re.IGNORECASE)
_RE_BARE10 = re.compile(r"^10\^?([+-]?\d+)$")
_RE_PLAIN = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")
_RE_INT = re.compile(r"^([+-]?\d+)$")


def parse_positive_label(text: str) -> Optional[float]:
    """Parse a tick label to a finite *positive* float, or ``None``.

    Handles power-of-ten (``10^-15``, ``10⁻¹⁵``), mantissa×10ⁿ
    (``2×10⁻⁶``, ``3.0e-9``) and plain decimal/integer (``0.1``, ``30``) forms.
    Bare exponent axes (``-10..0``) are handled at the axis level in
    :func:`interpret_decade_axis`, not here, because a bare ``-10`` is a
    legitimate *exponent*, not a value of ``-10``.
    """
    s = _normalize_text(text)
    if not s:
        return None
    try:
        # Plain decimal / integer / scientific FIRST (3.0e-9, 0.1, 30, 100, 1000).
        # This must precede the power forms so "100" parses as 100, not 10**0.
        if "^" not in s and "x" not in s.lower():
            if _RE_PLAIN.match(s):
                v = float(s)
                return v if math.isfinite(v) and v > 0 else None
        # mantissa x 10^n  (2x10-6 from "2×10⁻⁶", 2*10^-6, and bare 10^-15 / 10-15)
        m = _RE_POW10.match(s)
        if m and "10" in s:
            mant = m.group(1)
            mant_f = 1.0 if mant in ("", "+", "-") else float(mant)
            if mant == "-":
                mant_f = -1.0
            v = mant_f * (10.0 ** int(m.group(2)))
            return v if math.isfinite(v) and v > 0 else None
        # 10^-15 (no mantissa) fallback.
        m = _RE_BARE10.match(s)
        if m:
            v = float(10.0 ** int(m.group(1)))
            return v if math.isfinite(v) and v > 0 else None
    except (ValueError, OverflowError):
        return None
    return None


def parse_int_label(text: str) -> Optional[int]:
    """Parse a bare signed integer (the decade-exponent case), or ``None``."""
    s = _normalize_text(text)
    m = _RE_INT.match(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def interpret_decade_axis(
    pix_text: list[tuple[float, str, float]],
) -> Optional[list[TickLabel]]:
    """Reproduce the 1207.3275 win: bare exponents ``-10..0`` on a log axis.

    When the tick labels are bare integers whose values, ordered by pixel, are
    monotonic and consecutive-ish, interpret each integer ``k`` as ``10**k``.
    ``pix_text`` is ``[(pixel, raw_text, conf), ...]``. Returns labels only when
    at least 2 bare ints are recovered and they are monotonic in pixel order;
    otherwise ``None`` so the caller keeps the LLM axis.
    """
    ints: list[tuple[float, int, str, float]] = []
    for px, raw, conf in pix_text:
        k = parse_int_label(raw)
        if k is None or abs(k) > 40:
            continue
        ints.append((px, k, raw, conf))
    if len(ints) < 2:
        return None
    ints.sort(key=lambda t: t[0])
    ks = [t[1] for t in ints]
    diffs = [b - a for a, b in zip(ks[:-1], ks[1:])]
    # Monotonic (all same sign, non-zero) and unit-ish steps (decade ticks).
    if any(d == 0 for d in diffs):
        return None
    if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
        return None
    if max(abs(d) for d in diffs) > 2:
        return None
    out = []
    for px, k, raw, conf in ints:
        v = float(10.0 ** k)
        if math.isfinite(v) and v > 0:
            out.append(TickLabel(pixel=float(px), value=v, raw_text=raw, ocr_conf=conf))
    return out if len(out) >= 2 else None


def _median_spacing(pixels: list[float]) -> float:
    sp = sorted(float(p) for p in pixels)
    diffs = [b - a for a, b in zip(sp[:-1], sp[1:]) if b - a > 0]
    if not diffs:
        return 0.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def _label_band(bbox, axis: str, tick: float, spacing: float, shape) -> Optional[tuple]:
    """Pixel rectangle (x0, y0, x1, y1) of the label band beside ``tick``."""
    x0, y0, x1, y1 = bbox
    h, w = shape[0], shape[1]
    gap = 2
    if spacing <= 0:
        spacing = (x1 - x0) / 6.0 if axis == "x" else (y1 - y0) / 6.0
    if axis == "x":
        half_w = max(6.0, 0.5 * spacing)
        band_h = max(12.0, min(1.2 * spacing, 0.18 * (y1 - y0)))
        cx0 = int(max(0, tick - half_w))
        cx1 = int(min(w, tick + half_w))
        cy0 = int(min(h - 1, y1 + gap))
        cy1 = int(min(h, y1 + gap + band_h))
    else:  # y axis
        half_h = max(5.0, 0.45 * spacing)
        band_w = max(18.0, min(2.0 * spacing, 0.22 * (x1 - x0)))
        cy0 = int(max(0, tick - half_h))
        cy1 = int(min(h, tick + half_h))
        cx0 = int(max(0, x0 - gap - band_w))
        cx1 = int(max(0, x0 - gap))
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    return (cx0, cy0, cx1, cy1)


def _ocr_crop(arr, upscale: int = 4) -> tuple[str, float]:
    """OCR a numeric crop (numpy array) -> (joined_text, mean_conf 0..1).

    The crop is written to a temp PNG and tesseract is invoked on the file path
    rather than handed a PIL object: some environments mis-handle pytesseract's
    internal in-memory temp-save (a libpng/leptonica mismatch surfaces as a
    decode error), and the file-path invocation is robust across them.
    """
    try:
        img = Image.fromarray(arr)
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        if upscale > 1:
            img = img.resize(
                (img.width * upscale, img.height * upscale), Image.LANCZOS
            )
        config = (
            "--psm 7 -c tessedit_char_whitelist="
            "0123456789.eExX+-^"
        )
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp = tf.name
            img.save(tmp)
            data = pytesseract.image_to_data(
                tmp, config=config, output_type=pytesseract.Output.DICT
            )
        finally:
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        texts, confs = [], []
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            t = (txt or "").strip()
            if not t:
                continue
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if c < 0:
                continue
            texts.append(t)
            confs.append(c / 100.0)
        if not texts:
            return "", 0.0
        return "".join(texts), float(sum(confs) / len(confs))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("_ocr_crop failed: %s", e)
        return "", 0.0


def ocr_axis_labels(
    panel_img,
    bbox,
    axis: str,
    tick_pixels: list[float],
    scale: str = "log",
    conf_floor: float = CONF_FLOOR,
) -> list[TickLabel]:
    """Crop a label band beside each detected tick, OCR it, parse to a float.

    Returns ``TickLabel``\\ s that parse to a finite positive number with
    ``ocr_conf >= conf_floor``, paired to the tick pixel they were read at. On a
    *log* axis whose labels turn out to be bare decade exponents (``-10..0``),
    falls back to :func:`interpret_decade_axis`. Never raises; ``[]`` on any
    failure or when OCR is unavailable.
    """
    if not _OCR_AVAILABLE or panel_img is None or bbox is None:
        return []
    if not tick_pixels or len(tick_pixels) < 2:
        return []
    try:
        shape = panel_img.shape
        spacing = _median_spacing(tick_pixels)
        positives: list[TickLabel] = []
        raw_for_decade: list[tuple[float, str, float]] = []
        for tick in tick_pixels:
            band = _label_band(bbox, axis, float(tick), spacing, shape)
            if band is None:
                continue
            cx0, cy0, cx1, cy1 = band
            crop = panel_img[cy0:cy1, cx0:cx1]
            if crop is None or crop.size == 0:
                continue
            text, conf = _ocr_crop(crop)
            if not text:
                continue
            raw_for_decade.append((float(tick), text, conf))
            if conf < conf_floor:
                continue
            v = parse_positive_label(text)
            if v is not None:
                positives.append(
                    TickLabel(pixel=float(tick), value=v, raw_text=text, ocr_conf=conf)
                )
        if len(positives) >= 2:
            return positives
        # Decade-exponent fallback (1207.3275): only on a log axis.
        if (scale or "log").lower().startswith("log"):
            decade = interpret_decade_axis(
                [(px, raw, conf) for (px, raw, conf) in raw_for_decade if conf >= conf_floor]
            )
            if decade:
                return decade
        return positives
    except Exception as e:
        logger.warning("ocr_axis_labels failed: %s", e)
        return []
