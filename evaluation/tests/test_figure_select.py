"""Unit tests for limit-figure page selection (P-A2, issue #587).

No API calls, no PDF, no anthropic — :mod:`pipeline.figure_select` is a pure text
ranker, so these run in the minimal no-API CI job. They pin the behaviour that
fixes the two figure-delivery bugs the digest's family A/B traced to (a vector
exclusion plot the raster-only path missed; a plot on a page beyond the first N).

Run:
    pytest evaluation/tests/test_figure_select.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.figure_select import rank_limit_pages, score_page

# A realistic exclusion-plot caption (cf. 2305.01002 Fig. 4).
_PLOT_CAPTION = (
    "FIG. 4. The 95% C.L. exclusion region on the ALP-photon coupling g_agamma "
    "as a function of the ALP mass m_a, derived from GW170817. The excluded "
    "region is shaded; the projected sensitivity is shown dashed."
)
# Decorative figures from the same paper that the old extractor wrongly cropped.
_SCHEMATIC_CAPTION = "FIG. 2. Schematic of the NS-merger decay geometry. The angle theta..."
_ARTIST_CAPTION = "FIG. 1. An artist's rendition of our main idea. The NS merger emits..."
_BODY_TEXT = (
    "In this section we describe the analysis. As shown in Fig. 4, the bound "
    "improves on previous work. We use the standard formalism throughout."
)


# ---------------------------------------------------------------------------
# score_page
# ---------------------------------------------------------------------------

def test_exclusion_caption_scores_high():
    assert score_page(_PLOT_CAPTION) > 0

def test_caption_without_limit_or_coupling_scores_zero():
    # A figure caption that is purely decorative (no limit/coupling language).
    assert score_page(_SCHEMATIC_CAPTION) == 0
    assert score_page(_ARTIST_CAPTION) == 0

def test_limit_keywords_without_a_caption_score_zero():
    # Body text that *references* a figure but holds no caption is not the plot page.
    assert score_page(_BODY_TEXT) == 0

def test_empty_text_scores_zero():
    assert score_page("") == 0
    assert score_page(None) == 0  # type: ignore[arg-type]

def test_plot_page_outscores_decorative_caption_page():
    assert score_page(_PLOT_CAPTION) > score_page(_SCHEMATIC_CAPTION)

def test_coupling_alone_with_caption_scores():
    # Caption + coupling token but no explicit limit word still qualifies.
    assert score_page("FIG. 5. The kinetic mixing chi versus dark photon mass.") > 0


# ---------------------------------------------------------------------------
# rank_limit_pages
# ---------------------------------------------------------------------------

def test_picks_the_plot_page_among_decorative_pages():
    # 2305.01002 shape: artist rendition (p0), schematic (p14), plot (p4).
    pages = [_ARTIST_CAPTION, _BODY_TEXT, _BODY_TEXT, _BODY_TEXT, _PLOT_CAPTION,
             _BODY_TEXT] + [_BODY_TEXT] * 8 + [_SCHEMATIC_CAPTION, _BODY_TEXT]
    ranked = rank_limit_pages(pages)
    assert ranked[0] == 4  # the exclusion-plot page wins

def test_finds_plot_page_beyond_first_ten():
    # 1810.04602 shape: 41-page paper, the limit plot caption is on a late page.
    pages = [_BODY_TEXT] * 30
    pages[27] = _PLOT_CAPTION
    ranked = rank_limit_pages(pages)
    assert 27 in ranked and ranked[0] == 27

def test_returns_empty_when_no_limit_figure():
    pages = [_BODY_TEXT, _SCHEMATIC_CAPTION, _ARTIST_CAPTION]
    assert rank_limit_pages(pages) == []

def test_respects_max_pages_cap():
    pages = [_PLOT_CAPTION] * 10
    assert len(rank_limit_pages(pages, max_pages=6)) == 6

def test_ties_break_by_page_order():
    pages = [_BODY_TEXT, _PLOT_CAPTION, _BODY_TEXT, _PLOT_CAPTION]
    ranked = rank_limit_pages(pages)
    assert ranked == [1, 3]  # equal score -> earlier page first
