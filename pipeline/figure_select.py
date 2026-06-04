"""Limit-figure page selection — P-A2 of the failure-digest fixes (#587).

Pure, dependency-free page ranking for :func:`extractor.extract_figures_from_pdf`.

The figure extractor used to deliver the WRONG images to the vision stage in two
ways (diagnosed in ``evaluation/eval_runs/failure_analysis.md`` + the #587 P-A2
investigation):

1. **Strategy-1 short-circuit.** It extracted only *embedded raster* images
   (``page.get_images``) and, if any existed, returned them and SKIPPED the
   page-render fallback. Exclusion plots are usually VECTOR (matplotlib, no
   embedded bitmap), so a paper whose plot is vector but which also has decorative
   rasters (a detector schematic, an artist's rendition, a logo) got the
   decorative rasters cropped and the real plot was never rendered. Proven on
   2305.01002: the two cropped "figures" were a NS-merger geometry schematic and
   an artist's rendition; the Fig. 4 exclusion plot (vector, page 4) was never
   delivered.
2. **First-N-pages cap.** The page-render fallback rendered only the first
   ``max_figures`` pages in page order, missing plots later in long papers. Proven
   on 1810.04602: 41-page CMS paper, limit plot in Fig. 7 past page 10.

Either way stage2 got no exclusion plot, returned ``found_limit_plot=False`` /
0 points, produced no ``figure_vision`` candidate, and the selector fell back to a
sparse text point (digest family A) or emitted nothing (family B).

This module ranks pages by how likely they CONTAIN a limit/exclusion figure, using
the figure CAPTION (which sits on the figure's own page) together with
limit/coupling language. The extractor renders the top-ranked pages and ADDS them
to the raster crops — additive, so a paper whose plot genuinely is a bitmap does
not regress.
"""

from __future__ import annotations

import re

# Figure caption marker. Captions live on the same page as the figure, so a page
# carrying a caption is a candidate to physically contain that figure. Matches
# "FIG. 4", "Figure 4:", "Fig 4".
_CAPTION_RE = re.compile(r"\bfig(?:ure)?\.?\s*\d+", re.IGNORECASE)

# Limit / exclusion language marking a constraint figure (matched lowercased).
_LIMIT_KEYWORDS: tuple[str, ...] = (
    "exclusion", "excluded", "exclude", "upper limit", "upper bound",
    "lower limit", "95% c", "90% c", "99% c", "c.l.", "confidence level",
    "constraint", "constrain", "sensitivity", "projected", "excluded region",
    "ruled out",
)

# Coupling / observable tokens typical of these plots' axes (matched lowercased).
_COUPLING_TOKENS: tuple[str, ...] = (
    "g_a", "g_{a", "gae", "gagamma", "ga\\gamma", "g_{a\\gamma", "g_a\\gamma",
    "g_{ae", "g_{an", "g_{ap", "coupling", "kinetic mixing", "\\chi", " chi",
    "decay constant", "f_a", "1/f_a", "d_e", "\\epsilon", "mixing parameter",
)


def score_page(text: str) -> int:
    """How likely this page CONTAINS a limit/exclusion figure (0 = not a candidate).

    A page scores only if it carries a figure caption AND limit-or-coupling
    language — the caption of an exclusion plot reads e.g. "FIG. 4. 95% CL
    exclusion on g_agamma vs m_a". Pure: text in, int out. Higher is more likely.
    """
    if not text:
        return 0
    if not _CAPTION_RE.search(text):
        return 0
    t = text.lower()
    kw = sum(t.count(k) for k in _LIMIT_KEYWORDS)
    coup = any(tok in t for tok in _COUPLING_TOKENS)
    if kw == 0 and not coup:
        return 0
    return 2 + min(kw, 5) + (1 if coup else 0)


def rank_limit_pages(page_texts, max_pages: int = 6) -> list[int]:
    """Page indices likely to hold a limit figure, best first (at most ``max_pages``).

    Sorted by descending :func:`score_page`, ties broken by page order (stable).
    Returns ``[]`` when no page matches — the caller then keeps its raster crops
    or first-N-pages fallback.
    """
    scored = [(score_page(t or ""), i) for i, t in enumerate(page_texts)]
    scored = [(s, i) for s, i in scored if s > 0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scored[:max_pages]]
