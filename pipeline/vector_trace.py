"""Vector-path curve extraction from figure PDFs (WS2, vector-first).

Most benchmark papers ship figures as embedded VECTOR PDFs (matplotlib /
pgfplots output): the curves are stroke paths with exact coordinates, the
tick labels are real text spans, and the axis units are written in the axis
label — all machine-readable with pymupdf, no OCR, no LLM, no pixels. This
module turns one figure PDF into calibrated data-space curve candidates:

1. **Ticks** (:func:`extract_text_ticks`): matplotlib renders a log tick as a
   base span ``10`` plus superscript exponent spans (smaller font, raised
   baseline); linear ticks are plain numeric spans. Bottom-row spans are
   x ticks, left-column spans are y ticks; the axis *labels* ("Frequency
   (Hz)", "mass [eV]") ride along for unit conversion.
2. **Calibration** (:func:`fit_axes`): least-squares ``page-coord ->
   log10(value)`` per axis, >= 3 ticks and near-perfect linearity required —
   a failed fit means the figure needs the raster/OCR fallback (outlined-text
   PDFs), it never guesses.
3. **Curves** (:func:`extract_curves`): drawing paths grouped by stroke/fill
   colour; each colour group ordered along the path is one candidate curve.
   Filled exclusion *regions* yield their boundary path, which contains the
   limit contour.

The caller (ceiling survey now, extractor integration later) maps candidates
to data space via the calibration and the x-axis unit. Selection among
candidates is NOT this module's job — the survey scores all of them against
GT (oracle ceiling); the runtime channel will use label heuristics + a cheap
LLM pick. Everything here is deterministic and fail-open: any structural
surprise returns ``None``/empty rather than raising.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# x-axis unit label -> factor to eV. Frequency axes convert via h; mass axes
# via SI prefix. Matching is case-sensitive where it must be (mHz/MHz).
_H_EV_S = 4.135667696e-15
_X_UNIT_EV: dict[str, float] = {
    "Hz": _H_EV_S, "kHz": _H_EV_S * 1e3, "MHz": _H_EV_S * 1e6,
    "GHz": _H_EV_S * 1e9, "mHz": _H_EV_S * 1e-3,
    "eV": 1.0, "keV": 1e3, "MeV": 1e6, "GeV": 1e9, "TeV": 1e12,
    "meV": 1e-3, "µeV": 1e-6, "μeV": 1e-6, "ueV": 1e-6, "neV": 1e-9,
    "peV": 1e-12, "feV": 1e-15, "aeV": 1e-18, "zeV": 1e-21,
}
_X_UNIT_RE = re.compile(
    r"[\[\(]\s*([kMGTmµμunpfaz]?(?:Hz|eV))\s*[\]\)]|(?:^|\s)([kMGTmµμunpfaz]?(?:Hz|eV))(?:\s|$)")

_MIN_TICKS = 3
_MAX_FIT_RESID = 0.05   # max |fit residual| in exponent units — ticks are exact


@dataclass
class AxisCalibration:
    """Linear page->log10(value) maps for both axes + label metadata."""

    ax: float; bx: float          # log10(x_value) = ax * page_x + bx
    ay: float; by: float          # log10(y_value) = ay * page_y + by
    x_label: str = ""
    y_label: str = ""
    x_to_ev: Optional[float] = None   # None = unit not recognized
    n_xticks: int = 0
    n_yticks: int = 0


@dataclass
class VectorCurve:
    """One colour-grouped candidate curve, in page coordinates."""

    color: tuple
    filled: bool
    points: list = field(repr=False, default_factory=list)  # [(px, py), ...]

    @property
    def n_points(self) -> int:
        return len(self.points)


def _spans(page) -> list[dict]:
    out = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                t = sp["text"].strip()
                if t:
                    out.append({"t": t, "x": sp["origin"][0], "y": sp["origin"][1],
                                "size": sp["size"], "bbox": sp["bbox"]})
    return out


def _parse_plain(t: str) -> Optional[float]:
    try:
        return float(t.replace("−", "-").replace("−", "-"))
    except ValueError:
        return None


def _minus_marks(page) -> list:
    """Centres of small horizontal strokes/fills that render a minus sign.

    Mathtext-mode matplotlib draws the superscript minus as a PATH, not a
    text span — without this, every negative exponent silently reads
    positive and the calibration is sign-flipped. A minus mark is a tiny,
    nearly-flat drawing item (2-9 pt wide, < 2.5 pt tall)."""
    marks = []
    try:
        for dr in page.get_drawings():
            r = dr["rect"]
            if 1.5 <= r.width <= 9 and r.height <= 2.5:
                marks.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    except Exception:
        pass
    return marks


def extract_text_ticks(page):
    """(x_ticks, y_ticks, x_label, y_label); ticks are (page_pos, log10_value).

    Handles the two matplotlib renderings: ``10``-base + superscript-exponent
    span pairs (log axes) and plain numeric spans (linear/scalar ticks, which
    must be positive to be usable on a log-log limit plot). Bottom-row spans
    become x ticks, left-column spans y ticks, classified by clustering
    against the page's drawing-free margins.
    """
    spans = _spans(page)
    rect = page.rect
    minus = _minus_marks(page)
    ticks = []  # (cx, cy, log10_value)
    used = set()
    for i, b in enumerate(spans):
        if b["t"] not in ("10",) or b["size"] < 6:
            continue
        parts = [(j, s) for j, s in enumerate(spans)
                 if j != i and s["size"] < b["size"]
                 and 0 < s["x"] - b["x"] < 30 and 0 < b["y"] - s["y"] < b["size"]]
        parts.sort(key=lambda js: js[1]["x"])
        txt = "".join(p["t"] for _, p in parts).replace("−", "-").replace("−", "-")
        m = re.fullmatch(r"-?\d{1,3}", txt)
        if not m:
            continue
        val = float(txt)
        if val > 0 and parts:
            # sign may be a drawn glyph in the gap between the base and the
            # first exponent digit (mathtext minus) — see _minus_marks.
            first = parts[0][1]
            gap_x0, gap_x1 = b["bbox"][2] - 1, first["bbox"][0] + 1
            band_y0, band_y1 = first["bbox"][1] - 2, first["bbox"][3] + 2
            if any(gap_x0 <= mx <= gap_x1 and band_y0 <= my <= band_y1
                   for mx, my in minus):
                val = -val
        cx = (b["bbox"][0] + b["bbox"][2]) / 2 + 3
        cy = b["y"] - 3
        ticks.append((cx, cy, val))
        used.add(i)
        used.update(j for j, _ in parts)
    for i, b in enumerate(spans):
        if i in used:
            continue
        v = _parse_plain(b["t"])
        if v is None or v <= 0:
            continue
        ticks.append(((b["bbox"][0] + b["bbox"][2]) / 2, b["y"], math.log10(v)))

    if not ticks:
        return [], [], "", ""
    # x ticks share a baseline row near the bottom; y ticks share a column
    # near the left. Use medians of the extreme clusters.
    ys = sorted(t[1] for t in ticks)
    xs = sorted(t[0] for t in ticks)
    bottom = ys[-1]
    left = xs[0]
    x_ticks = [(t[0], t[2]) for t in ticks if abs(t[1] - bottom) < 5]
    y_ticks = [(t[1], t[2]) for t in ticks if abs(t[0] - left) < 12
               and abs(t[1] - bottom) >= 5]

    # axis labels: the lowest text row (below x ticks) and the leftmost
    # rotated/vertical text; approximate with the largest non-tick spans.
    non_tick = [s for j, s in enumerate(spans) if j not in used
                and _parse_plain(s["t"]) is None]

    def _join(row):
        row = sorted(row, key=lambda s: s["x"])
        out = ""
        prev_end = None
        for s in row:
            if prev_end is not None and s["bbox"][0] - prev_end > 2:
                out += " "
            out += s["t"]
            prev_end = s["bbox"][2]
        return out.strip()

    x_label = _join([s for s in non_tick
                     if s["y"] > bottom + 3 and s["y"] > rect.height * 0.7])
    y_label = _join([s for s in non_tick
                     if s["x"] < left + 5 and s["y"] <= bottom + 3])
    return x_ticks, y_ticks, x_label, y_label


def _fit(ticks):
    """(slope, intercept) of page->exponent, or None if not cleanly linear."""
    if len(ticks) < _MIN_TICKS:
        return None
    xs = [p for p, _ in ticks]
    es = [e for _, e in ticks]
    n = len(xs)
    mx, me = sum(xs) / n, sum(es) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    a = sum((x - mx) * (e - me) for x, e in zip(xs, es)) / sxx
    b = me - a * mx
    if a == 0:
        return None
    resid = max(abs(a * x + b - e) for x, e in zip(xs, es))
    if resid > _MAX_FIT_RESID:
        return None
    return a, b


def fit_axes(page) -> Optional[AxisCalibration]:
    """Text-layer axis calibration, or ``None`` (-> raster/OCR fallback)."""
    try:
        x_ticks, y_ticks, x_label, y_label = extract_text_ticks(page)
        fx, fy = _fit(x_ticks), _fit(y_ticks)
        if fx is None or fy is None:
            return None
        unit = None
        m = _X_UNIT_RE.search(x_label)
        if m:
            unit = _X_UNIT_EV.get(m.group(1) or m.group(2))
        return AxisCalibration(ax=fx[0], bx=fx[1], ay=fy[0], by=fy[1],
                               x_label=x_label, y_label=y_label, x_to_ev=unit,
                               n_xticks=len(x_ticks), n_yticks=len(y_ticks))
    except Exception as e:  # fail open
        logger.debug("fit_axes failed: %s", e)
        return None


def extract_curves(page, *, min_items: int = 20,
                   max_curves: int = 12) -> list[VectorCurve]:
    """Colour-grouped drawing paths, largest first. Black/greyscale groups are
    kept too (frames are filtered later by the min_items/shape of real curves,
    and monochrome plots exist), but pure axis furniture (tiny groups) is not.
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    try:
        for dr in page.get_drawings():
            col = tuple(round(c, 2) for c in (dr.get("color") or ()))
            fill = tuple(round(c, 2) for c in (dr.get("fill") or ())) if dr.get("fill") else None
            key = (col or fill or (), fill is not None)
            for it in dr["items"]:
                if it[0] == "l":
                    groups[key] += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
                elif it[0] == "c":
                    groups[key] += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
                elif it[0] == "re":
                    pass  # rectangles are frames/legend boxes, not curves
    except Exception as e:
        logger.debug("extract_curves failed: %s", e)
        return []
    curves = [VectorCurve(color=k[0], filled=k[1], points=v)
              for k, v in groups.items() if len(v) >= min_items]
    curves.sort(key=lambda c: -c.n_points)
    return curves[:max_curves]


def curve_to_data(curve: VectorCurve, cal: AxisCalibration,
                  *, to_ev: bool = True) -> list[tuple[float, float]]:
    """Map a page-space curve into (mass_eV_or_native, coupling) data space,
    deduplicated to the lower envelope per x (a filled region's boundary
    traces up AND down; the limit is the lower edge)."""
    scale = cal.x_to_ev if (to_ev and cal.x_to_ev) else 1.0
    best: dict = {}
    for px, py in curve.points:
        ex = cal.ax * px + cal.bx
        ey = cal.ay * py + cal.by
        if not (-150 < ex < 150 and -150 < ey < 150):
            continue  # bogus calibration/off-axes point; 10**e would overflow
        x = 10 ** ex * scale
        y = 10 ** ey
        if not (math.isfinite(x) and math.isfinite(y)) or x <= 0 or y <= 0:
            continue
        k = round(math.log10(x), 4)
        if k not in best or y < best[k][1]:
            best[k] = (x, y)
    return sorted(best.values())


def trace_figure_pdf(pdf_path: str | Path):
    """One figure PDF -> (calibration, [VectorCurve]) or (None, [])."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        if doc.page_count < 1:
            return None, []
        page = doc[0]
        cal = fit_axes(page)
        if cal is None:
            return None, []
        return cal, extract_curves(page)
    except Exception as e:
        logger.debug("trace_figure_pdf(%s) failed: %s", pdf_path, e)
        return None, []
