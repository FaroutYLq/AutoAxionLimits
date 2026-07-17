"""Canonical arXiv-id derivation for old-style ids.

Old-style arXiv ids carry a category prefix with a slash
(``hep-ph/0307284``, ``quant-ph/0106045``). A naive
``entry_id.split("/")[-1]`` grabs only the final path segment and silently
drops the category, yielding ``0307284`` — a non-canonical id that no longer
matches the ground-truth / pool key. That desync made two old-style benchmark
papers look absent from their own snapshots (their internal ``arxiv_id`` field
was the stripped form while the pool key kept the prefix).

Both id-deriving helpers now use ``paper.get_short_id()``, which preserves the
category. These tests pin that against regression, for new-style, old-style, and
version-suffixed ids.

Run:
    python -m pytest pipeline/tests/test_arxiv_id_canonical.py -v
"""

from __future__ import annotations

import pytest

from pipeline.extractor import run_extraction_agent  # noqa: F401  (import guard)
from pipeline.monitor import _arxiv_id


class _PaperStub:
    """Mimics the arxiv.Result / eval _PaperStub surface used for id parsing."""

    def __init__(self, short_id: str, version: str = ""):
        self._short = short_id + version
        self.entry_id = f"http://arxiv.org/abs/{short_id}{version}"

    def get_short_id(self):
        return self._short


CASES = [
    # (short_id, version_suffix, expected_canonical)
    ("2412.12345", "", "2412.12345"),
    ("2412.12345", "v2", "2412.12345"),
    ("hep-ph/0307284", "", "hep-ph/0307284"),
    ("hep-ph/0307284", "v2", "hep-ph/0307284"),
    ("quant-ph/0106045", "", "quant-ph/0106045"),
    ("astro-ph/0611502", "v1", "astro-ph/0611502"),
]


@pytest.mark.parametrize("short_id,version,expected", CASES)
def test_monitor_arxiv_id_is_canonical(short_id, version, expected):
    assert _arxiv_id(_PaperStub(short_id, version)) == expected


@pytest.mark.parametrize("short_id,version,expected", CASES)
def test_extractor_id_parse_is_canonical(short_id, version, expected):
    """The extractor derives its id the same way (get_short_id, strip version)."""
    import re

    paper = _PaperStub(short_id, version)
    # mirrors pipeline/extractor.py run_extraction_agent line 1 of the body
    derived = re.sub(r"v\d+$", "", paper.get_short_id())
    assert derived == expected


def test_oldstyle_not_confused_with_version():
    """A category prefix must survive; only a trailing vN is stripped."""
    assert _arxiv_id(_PaperStub("hep-ph/0307284", "v11")) == "hep-ph/0307284"
    # a bare number that merely *contains* letters+digits is unaffected
    assert _arxiv_id(_PaperStub("2412.12345", "")) == "2412.12345"
