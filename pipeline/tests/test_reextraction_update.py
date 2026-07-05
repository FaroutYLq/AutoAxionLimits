"""Regression tests for the re-extraction / update path.

Re-running the pipeline on a paper whose limit is ALREADY curated in the repo
(showcase run of arXiv:2209.03419 → QUALIPHIDE, 2026-07-04) exposed a bug family:

  1. ``write_repo_files`` inserted a SECOND ``def QUALIPHIDE`` into
     ``PlotFuncs.py::DarkPhoton``. Because the duplicate landed at the end of the
     class it shadowed the hand-curated method (whose signature has ``edge_on``),
     so the notebook call ``DarkPhoton.QUALIPHIDE(ax, …, edge_on=True)`` raised
     ``TypeError: unexpected keyword argument 'edge_on'``.
  2. It appended a duplicate notebook call.
  3. It appended a duplicate docs bullet (in the wrong section).
  4. ``create_feature_branch`` reused the branch name of a long-closed PR whose
     branch still lived on origin, so the push failed non-fast-forward and the
     paper was marked failed after a SUCCESSFUL extraction.
  5. ``execute_notebook_highlighted`` appended ``col='red', lw=3`` without
     stripping an existing ``col=``/``lw=`` → ``SyntaxError: keyword argument
     repeated`` that silently killed the highlighted plot.

These tests pin the fixes. They are pure (no network / no Anthropic API).

Run:
    python -m pytest pipeline/tests/test_reextraction_update.py -v
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pipeline.reviewer import (
    ReviewResult,
    class_has_method,
    docs_has_entry,
    notebook_has_call,
    write_repo_files,
)
from pipeline.pr_creator import _pick_unique_branch
from pipeline.plot_regen import _build_highlight_call, _strip_kwargs


# ---------------------------------------------------------------------------
# Fixtures — a tiny repo with a hand-curated QUALIPHIDE artifact set
# ---------------------------------------------------------------------------

# The curated method mirrors the real one: signature has ``edge_on`` that a
# freshly-generated method would NOT have. If the generated method shadows it,
# the notebook call below raises TypeError.
_CURATED_PLOTFUNCS = '''\
from numpy import loadtxt

class DarkPhoton:
    @staticmethod
    def SomethingElse(ax, col='r'):
        return

    @staticmethod
    def QUALIPHIDE(ax, col='r', fs=9, text_on=True, edge_on=False, lw=0.8, zorder=0):
        dat = loadtxt("limit_data/DarkPhoton/QUALIPHIDE.txt", ndmin=2)
        return
'''

_CURATED_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, ax = plt.subplots()\n",
                "DarkPhoton.QUALIPHIDE(ax, text_on=False, col='darkred', edge_on=True)\n",
                "MySaveFig(fig, 'DarkPhoton')\n",
            ],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}

_CURATED_DOCS = (
    "# Dark photon\n\n"
    "## Haloscopes\n\n"
    "* QUALIPHIDE: [limit](https://example/QUALIPHIDE.txt), "
    "[reference](https://arxiv.org/abs/2209.03419)\n"
)


def _review(repo_root: Path, experiment_name: str = "QUALIPHIDE") -> ReviewResult:
    return ReviewResult(
        arxiv_id="2209.03419",
        data_file_path="limit_data/DarkPhoton/%s.txt" % experiment_name,
        data_file_content=(
            "# QUALIPHIDE\n"
            "# mass [eV]    chi\n"
            "1.000000e-05   1.000000e-12\n"
            "2.000000e-05   2.000000e-12\n"
        ),
        plotfuncs_method=(
            "    @staticmethod\n"
            "    def %s(ax, col='crimson', fs=15, text_on=True, lw=1.5):\n"
            "        dat = loadtxt('limit_data/DarkPhoton/%s.txt', ndmin=2)\n"
            "        return\n" % (experiment_name, experiment_name)
        ),
        plotfuncs_file="PlotFuncs.py",
        plotfuncs_class="DarkPhoton",
        notebook_path="DarkPhoton.ipynb",
        notebook_call="DarkPhoton.%s(ax)\n" % experiment_name,
        docs_entry="- **%s**: [Paper](https://arxiv.org/abs/2209.03419)\n" % experiment_name,
        docs_file="docs/dp.md",
        corrections_applied=[],
        corrections_flagged=[],
        extraction_confidence=0.9,
        low_confidence=False,
        is_projection=False,
        paper_title="QUALIPHIDE",
        arxiv_url="https://arxiv.org/abs/2209.03419",
        experiment_name=experiment_name,
    )


def _make_curated_repo(tmp_path: Path) -> Path:
    (tmp_path / "PlotFuncs.py").write_text(_CURATED_PLOTFUNCS)
    (tmp_path / "DarkPhoton.ipynb").write_text(json.dumps(_CURATED_NOTEBOOK))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "dp.md").write_text(_CURATED_DOCS)
    (tmp_path / "limit_data" / "DarkPhoton").mkdir(parents=True)
    (tmp_path / "limit_data" / "DarkPhoton" / "QUALIPHIDE.txt").write_text("# old\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def test_class_has_method_detects_curated(tmp_path):
    pf = tmp_path / "PlotFuncs.py"
    pf.write_text(_CURATED_PLOTFUNCS)
    assert class_has_method(pf, "DarkPhoton", "QUALIPHIDE") is True
    assert class_has_method(pf, "DarkPhoton", "NotThere") is False
    # Same-named method in a DIFFERENT class must not count.
    assert class_has_method(pf, "AxionPhoton", "QUALIPHIDE") is False


def test_class_has_method_missing_file_is_false(tmp_path):
    assert class_has_method(tmp_path / "nope.py", "DarkPhoton", "QUALIPHIDE") is False


def test_notebook_has_call_matches_call_not_path(tmp_path):
    nb = tmp_path / "DarkPhoton.ipynb"
    nb.write_text(json.dumps(_CURATED_NOTEBOOK))
    assert notebook_has_call(nb, "DarkPhoton", "QUALIPHIDE") is True
    assert notebook_has_call(nb, "DarkPhoton", "SENSEI") is False


def test_notebook_has_call_ignores_loadtxt_path(tmp_path):
    # A bare filename mention (loadtxt path, comment) must NOT read as a call.
    nb_obj = {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ['dat = loadtxt("limit_data/DarkPhoton/QUALIPHIDE.txt")\n'],
        }],
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
    }
    nb = tmp_path / "DarkPhoton.ipynb"
    nb.write_text(json.dumps(nb_obj))
    assert notebook_has_call(nb, "DarkPhoton", "QUALIPHIDE") is False


def test_docs_has_entry_both_formats(tmp_path):
    docs = tmp_path / "dp.md"
    # Curated star-colon format.
    docs.write_text(_CURATED_DOCS)
    assert docs_has_entry(docs, "QUALIPHIDE") is True
    assert docs_has_entry(docs, "SENSEI") is False
    # Generated bold-dash format.
    docs.write_text("- **SENSEI**: [Paper](x)\n")
    assert docs_has_entry(docs, "SENSEI") is True
    # Substring in prose is not a bullet → not an entry.
    docs.write_text("The QUALIPHIDE experiment is described in the text.\n")
    assert docs_has_entry(docs, "QUALIPHIDE") is False


# ---------------------------------------------------------------------------
# write_repo_files — update semantics
# ---------------------------------------------------------------------------

def test_update_path_refreshes_data_only(tmp_path):
    repo = _make_curated_repo(tmp_path)
    review = _review(repo)
    pf_before = (repo / "PlotFuncs.py").read_text()
    nb_before = (repo / "DarkPhoton.ipynb").read_text()
    docs_before = (repo / "docs" / "dp.md").read_text()

    write_repo_files(review, repo_root=repo)

    # Data file refreshed with the new content.
    data = (repo / "limit_data" / "DarkPhoton" / "QUALIPHIDE.txt").read_text()
    assert "1.000000e-05" in data
    assert data != "# old\n"

    # PlotFuncs/notebook/docs untouched — byte-for-byte identical.
    assert (repo / "PlotFuncs.py").read_text() == pf_before
    assert (repo / "DarkPhoton.ipynb").read_text() == nb_before
    assert (repo / "docs" / "dp.md").read_text() == docs_before


def test_update_path_no_duplicate_method(tmp_path):
    repo = _make_curated_repo(tmp_path)
    write_repo_files(_review(repo), repo_root=repo)
    tree = ast.parse((repo / "PlotFuncs.py").read_text())
    defs = [
        n.name
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "DarkPhoton"
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
    ]
    assert defs.count("QUALIPHIDE") == 1  # not shadowed by a second def

    # The surviving method keeps the curated signature (has edge_on).
    qp = next(
        n for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "DarkPhoton"
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "QUALIPHIDE"
    )
    assert "edge_on" in {a.arg for a in qp.args.args}


def test_update_path_no_duplicate_notebook_call(tmp_path):
    repo = _make_curated_repo(tmp_path)
    write_repo_files(_review(repo), repo_root=repo)
    nb = json.loads((repo / "DarkPhoton.ipynb").read_text())
    joined = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert joined.count("DarkPhoton.QUALIPHIDE(") == 1


def test_update_path_no_duplicate_docs_entry(tmp_path):
    repo = _make_curated_repo(tmp_path)
    write_repo_files(_review(repo), repo_root=repo)
    docs = (repo / "docs" / "dp.md").read_text()
    # Exactly one bullet naming QUALIPHIDE.
    bullets = [ln for ln in docs.splitlines() if docs_has_entry_line(ln, "QUALIPHIDE")]
    assert len(bullets) == 1


def docs_has_entry_line(line: str, name: str) -> bool:
    import re
    return bool(re.match(rf"^\s*[-*]\s*\*{{0,2}}\s*{re.escape(name)}\b", line))


def test_fresh_experiment_still_inserts_everything(tmp_path):
    """A genuinely new experiment must still get method + call + docs appended."""
    repo = _make_curated_repo(tmp_path)
    review = _review(repo, experiment_name="BRANDNEW")
    write_repo_files(review, repo_root=repo)

    assert class_has_method(repo / "PlotFuncs.py", "DarkPhoton", "BRANDNEW")
    assert notebook_has_call(repo / "DarkPhoton.ipynb", "DarkPhoton", "BRANDNEW")
    assert docs_has_entry(repo / "docs" / "dp.md", "BRANDNEW")
    # PlotFuncs still parses (fail-closed AST guard held).
    ast.parse((repo / "PlotFuncs.py").read_text())


# ---------------------------------------------------------------------------
# pr_creator — branch uniquification
# ---------------------------------------------------------------------------

def test_pick_unique_branch_free_name():
    assert _pick_unique_branch("pipeline/arxiv-1-X", lambda b: False) == "pipeline/arxiv-1-X"


def test_pick_unique_branch_suffixes_on_collision():
    taken = {"pipeline/arxiv-2209-03419-QUALIPHIDE"}
    assert (
        _pick_unique_branch("pipeline/arxiv-2209-03419-QUALIPHIDE", lambda b: b in taken)
        == "pipeline/arxiv-2209-03419-QUALIPHIDE-2"
    )


def test_pick_unique_branch_walks_multiple_collisions():
    base = "pipeline/arxiv-2209-03419-QUALIPHIDE"
    taken = {base, f"{base}-2", f"{base}-3"}
    assert _pick_unique_branch(base, lambda b: b in taken) == f"{base}-4"


# ---------------------------------------------------------------------------
# plot_regen — highlight call de-duplication
# ---------------------------------------------------------------------------

def test_build_highlight_call_bare():
    assert _build_highlight_call("DarkPhoton.QUALIPHIDE(ax)") == \
        "DarkPhoton.QUALIPHIDE(ax, col='red', lw=3)"


def test_build_highlight_call_strips_existing_col_lw():
    out = _build_highlight_call(
        "DarkPhoton.QUALIPHIDE(ax,text_on=False,col=QUALIPHIDE_col,edge_on=True,lw=0.8)"
    )
    # Exactly one col= and one lw= — the injected ones.
    assert out.count("col=") == 1
    assert out.count("lw=") == 1
    assert out.endswith("col='red', lw=3)")
    # Unrelated kwargs are preserved.
    assert "text_on=False" in out
    assert "edge_on=True" in out
    # The result is syntactically valid (this is what the repeated-kwarg bug broke).
    ast.parse(out)


def test_build_highlight_call_non_call_passthrough():
    assert _build_highlight_call("# not a call") == "# not a call"


def test_strip_kwargs_leaves_others():
    assert _strip_kwargs(",a=1,col=x,b=2,lw=3", ("col", "lw")) == ",a=1,b=2"
