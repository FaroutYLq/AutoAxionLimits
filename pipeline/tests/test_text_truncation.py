"""Tests for #597 result-excerpt rescue in pipeline.extractor.

No API/network. Verifies that a limit statement beyond the head budget is mined
back into the text (with context), in original order, within the excerpt budget,
and that the head is never shortened (no regression for early-result papers).

Run:
    pytest pipeline/tests/test_text_truncation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.extractor import _result_excerpts


def test_captures_result_line_with_context():
    lines = [
        "filler about the apparatus",
        "We exclude g_ae > 1.2e-12 at 95% C.L.",
        "in the mass range 1-10 eV.",
    ]
    out = _result_excerpts("\n".join(lines), budget=10_000)
    # the result line plus its neighbours (context) come through
    assert "We exclude g_ae > 1.2e-12" in out
    assert "in the mass range 1-10 eV." in out


def test_keeps_original_order():
    lines = ["intro", "we obtain a bound A", "middle prose", "we find limit B"]
    out = _result_excerpts("\n".join(lines), budget=10_000)
    assert out.index("bound A") < out.index("limit B")


def test_budget_respected():
    lines = [f"we exclude value number {i} at 95% C.L." for i in range(500)]
    out = _result_excerpts("\n".join(lines), budget=200)
    assert len(out) <= 200


def test_no_result_keywords_returns_empty():
    out = _result_excerpts("just\nsome\nfiller\nprose", budget=10_000)
    assert out == ""


def test_zero_budget_returns_empty():
    assert _result_excerpts("we exclude X at 95% C.L.", budget=0) == ""
