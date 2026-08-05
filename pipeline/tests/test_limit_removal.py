"""Tests for removal of a curated limit's repository artifacts.

When a source paper is withdrawn from arXiv the weekly checker proposes the
removal as a reviewable diff instead of handing the reviewer a to-do list. A
curated limit lives in four places (data file, PlotFuncs method, notebook call,
docs bullet) and all four have to come out together, structurally: ``ast`` for
PlotFuncs and ``nbformat`` for notebooks, mirroring the insertion side.

The dangerous failure here is a half-applied removal that leaves PlotFuncs.py
unparseable, because that breaks every notebook import and the plots silently
fall back to stale images. Those cases are pinned below as fail-closed.

Run:
    python -m pytest pipeline/tests/test_limit_removal.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pipeline.removal import (
    coupling_type_from_data_path,
    describe_limit_artifacts,
    remove_docs_entry,
    remove_limit_artifacts,
    remove_method_from_plotfuncs,
    remove_notebook_call,
)


_PLOTFUNCS = '''\
from numpy import loadtxt


class DarkPhoton:
    @staticmethod
    def Keeper(ax, col='r'):
        dat = loadtxt("limit_data/DarkPhoton/Keeper.txt")
        return dat

    # A comment that documents the doomed method.
    @staticmethod
    def QUALIPHIDE_FIR(ax, col='crimson', fs=15, text_on=True, lw=1.5):
        dat = loadtxt("limit_data/DarkPhoton/QUALIPHIDE_FIR.txt", ndmin=2)
        ax.fill_between(dat[:, 0], dat[:, 1], y2=1e99)
        return dat

    @staticmethod
    def AlsoKeeper(ax):
        return None


class Lonely:
    @staticmethod
    def OnlyChild(ax):
        return None
'''

_DOCS = """\
# Dark photon

## Haloscopes
- **Keeper**: [some paper](https://arxiv.org/abs/1234.56789)
- **QUALIPHIDE_FIR**: [Dark matter searches with a 13 meV threshold](https://arxiv.org/abs/2607.19319)
* INTEGRAL: [limit](https://example.org/INTEGRAL.txt), [reference1](https://arxiv.org/abs/2406.19445)
"""

_CELL = """\
fig, ax = DarkPhoton.FigSetup()
DarkPhoton.Keeper(ax)
DarkPhoton.QUALIPHIDE_FIR(ax)
MySaveFig(fig,'DarkPhoton')"""


@pytest.fixture
def repo(tmp_path):
    """A miniature repo with all four artifacts of a curated limit."""
    import nbformat

    (tmp_path / "limit_data" / "DarkPhoton").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "limit_data" / "DarkPhoton" / "QUALIPHIDE_FIR.txt").write_text(
        "# arXiv: https://arxiv.org/abs/2607.19319\n1.0e-2  1.3e-11\n"
    )
    (tmp_path / "limit_data" / "DarkPhoton" / "Keeper.txt").write_text("1.0 2.0\n")
    (tmp_path / "PlotFuncs.py").write_text(_PLOTFUNCS)
    (tmp_path / "docs" / "dp.md").write_text(_DOCS)

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(source="import PlotFuncs"),
        nbformat.v4.new_code_cell(source=_CELL),
    ]
    nbformat.write(nb, str(tmp_path / "DarkPhoton.ipynb"))
    return tmp_path


# ---------------------------------------------------------------------------
# PlotFuncs
# ---------------------------------------------------------------------------

def test_removes_method_with_its_decorator_and_comment(repo):
    path = repo / "PlotFuncs.py"
    assert remove_method_from_plotfuncs(path, "DarkPhoton", "QUALIPHIDE_FIR") is True

    src = path.read_text()
    assert "QUALIPHIDE_FIR" not in src
    assert "doomed method" not in src, "the method's own comment should go with it"
    # The @staticmethod above the def must not be orphaned.
    assert src.count("@staticmethod") == 3

    tree = ast.parse(src)
    dp = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DarkPhoton")
    assert [c.name for c in dp.body] == ["Keeper", "AlsoKeeper"]


def test_touches_nothing_outside_the_removed_block(repo):
    """A global whitespace pass would churn blank-line runs across the whole file.

    Caught in review of the first live removal PR: collapsing blank lines with a
    file-wide regex rewrote unrelated lines hundreds of lines away, burying the
    real change in whitespace noise.
    """
    path = repo / "PlotFuncs.py"
    before = path.read_text().splitlines()
    assert remove_method_from_plotfuncs(path, "DarkPhoton", "QUALIPHIDE_FIR") is True
    after = path.read_text().splitlines()

    # Every surviving line must appear verbatim, in order, in the original: the
    # edit may only DELETE a contiguous block, never reflow anything.
    assert len(after) < len(before)
    cut = len(before) - len(after)
    for start in range(len(before) - cut + 1):
        if before[:start] + before[start + cut:] == after:
            break
    else:
        pytest.fail("removal changed lines outside one contiguous deleted block")


def test_leaves_the_original_method_separation(repo):
    """The gap left behind should match the file's method separation, not widen."""
    path = repo / "PlotFuncs.py"
    remove_method_from_plotfuncs(path, "DarkPhoton", "QUALIPHIDE_FIR")
    src = path.read_text()
    assert "\n\n\n\n" not in src, "removal left a widened blank-line run"
    assert "return dat\n    @staticmethod" not in src, "survivors were left adjacent"


def test_removal_is_the_exact_inverse_of_insertion(tmp_path):
    """Insert a method the way the reviewer does, remove it, get the file back.

    This is the case that matters in production: the limits being removed are
    the ones this pipeline inserted. Byte-identity means the removal PR shows
    only the limit coming out, with no whitespace churn for a reviewer to read
    past.
    """
    from pipeline.reviewer import insert_method_into_plotfuncs

    original = '''\
class DarkPhoton:
    @staticmethod
    def Existing(ax):
        return None




#=============================================================================#
def helper():
    return 1
'''
    path = tmp_path / "PlotFuncs.py"
    path.write_text(original)

    method = (
        "@staticmethod\n"
        "def Doomed(ax, col='crimson'):\n"
        "    dat = loadtxt('limit_data/DarkPhoton/Doomed.txt')\n"
        "    return dat\n"
    )
    insert_method_into_plotfuncs(path, "DarkPhoton", method)
    assert "Doomed" in path.read_text()

    assert remove_method_from_plotfuncs(path, "DarkPhoton", "Doomed") is True
    assert path.read_text() == original


def test_absent_method_is_a_noop(repo):
    path = repo / "PlotFuncs.py"
    before = path.read_text()
    assert remove_method_from_plotfuncs(path, "DarkPhoton", "NeverExisted") is False
    assert path.read_text() == before


def test_refuses_to_empty_a_class(repo):
    """Removing a class's only member would produce invalid Python."""
    path = repo / "PlotFuncs.py"
    before = path.read_text()
    assert remove_method_from_plotfuncs(path, "Lonely", "OnlyChild") is False
    assert path.read_text() == before
    ast.parse(path.read_text())


def test_unparseable_file_fails_closed(tmp_path):
    bad = tmp_path / "PlotFuncs.py"
    bad.write_text("class Broken:\n    def oops(:\n")
    assert remove_method_from_plotfuncs(bad, "Broken", "oops") is False
    assert bad.read_text() == "class Broken:\n    def oops(:\n"


def test_missing_class_is_a_noop(repo):
    path = repo / "PlotFuncs.py"
    before = path.read_text()
    assert remove_method_from_plotfuncs(path, "NoSuchClass", "QUALIPHIDE_FIR") is False
    assert path.read_text() == before


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------

def test_removes_only_the_target_call(repo):
    import nbformat

    assert remove_notebook_call(repo / "DarkPhoton.ipynb", "DarkPhoton", "QUALIPHIDE_FIR") is True
    nb = nbformat.read(str(repo / "DarkPhoton.ipynb"), as_version=4)
    src = nb.cells[1].source
    assert "QUALIPHIDE_FIR" not in src
    assert "DarkPhoton.Keeper(ax)" in src
    assert "MySaveFig" in src


def test_notebook_call_absent_is_a_noop(repo):
    assert remove_notebook_call(repo / "DarkPhoton.ipynb", "DarkPhoton", "NeverExisted") is False


def test_bare_name_mention_is_not_a_call(tmp_path):
    """A loadtxt path mentioning the name must not be mistaken for a call."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(
        source='dat = loadtxt("limit_data/DarkPhoton/QUALIPHIDE_FIR.txt")'
    )]
    p = tmp_path / "n.ipynb"
    nbformat.write(nb, str(p))

    assert remove_notebook_call(p, "DarkPhoton", "QUALIPHIDE_FIR") is False
    assert "loadtxt" in nbformat.read(str(p), as_version=4).cells[0].source


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

def test_removes_generated_bullet_only(repo):
    docs = repo / "docs" / "dp.md"
    assert remove_docs_entry(docs, "QUALIPHIDE_FIR") is True
    text = docs.read_text()
    assert "QUALIPHIDE_FIR" not in text
    assert "**Keeper**" in text and "INTEGRAL" in text


def test_removes_curated_star_bullet(repo):
    """Curated entries use `* Name: …` rather than `- **Name**: …`."""
    docs = repo / "docs" / "dp.md"
    assert remove_docs_entry(docs, "INTEGRAL") is True
    assert "INTEGRAL" not in docs.read_text()


def test_docs_entry_absent_is_a_noop(repo):
    assert remove_docs_entry(repo / "docs" / "dp.md", "NeverExisted") is False


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

_DATA_REL = "limit_data/DarkPhoton/QUALIPHIDE_FIR.txt"


def test_coupling_type_from_path():
    assert coupling_type_from_data_path(_DATA_REL) == "DarkPhoton"
    assert coupling_type_from_data_path(
        "limit_data/AxionPhoton/Projections/X.txt"
    ) == "AxionPhoton"
    assert coupling_type_from_data_path("something/else.txt") is None


def test_removes_all_four_artifacts(repo):
    report = remove_limit_artifacts(_DATA_REL, repo_root=repo)

    assert not (repo / _DATA_REL).exists()
    assert "QUALIPHIDE_FIR" not in (repo / "PlotFuncs.py").read_text()
    assert "QUALIPHIDE_FIR" not in (repo / "docs" / "dp.md").read_text()
    assert "QUALIPHIDE_FIR" not in (repo / "DarkPhoton.ipynb").read_text()

    assert report.removed_anything
    assert report.missing == []
    assert set(report.changed_paths) == {
        _DATA_REL, "PlotFuncs.py", "DarkPhoton.ipynb", "docs/dp.md",
    }
    assert report.coupling_type == "DarkPhoton"
    assert report.experiment_name == "QUALIPHIDE_FIR"
    # Untouched neighbours survive.
    assert (repo / "limit_data" / "DarkPhoton" / "Keeper.txt").exists()


def test_partial_curation_is_reported_not_fatal(repo):
    """A hand-curated limit may lack a generated docs bullet; that is not an error."""
    (repo / "docs" / "dp.md").write_text("# Dark photon\n- **Keeper**: x\n")

    report = remove_limit_artifacts(_DATA_REL, repo_root=repo)

    assert report.removed_anything
    assert any("docs" in m for m in report.missing)
    assert "docs/dp.md" not in report.changed_paths


def test_plotfuncs_still_parses_after_end_to_end_removal(repo):
    remove_limit_artifacts(_DATA_REL, repo_root=repo)
    ast.parse((repo / "PlotFuncs.py").read_text())


def test_dry_run_description_matches_what_removal_does(repo):
    """The --dry-run listing must not drift from the real removal."""
    planned = describe_limit_artifacts(_DATA_REL, repo_root=repo)
    report = remove_limit_artifacts(_DATA_REL, repo_root=repo)
    assert planned == report.removed


def test_describe_is_non_destructive(repo):
    before = (repo / "PlotFuncs.py").read_text()
    describe_limit_artifacts(_DATA_REL, repo_root=repo)
    assert (repo / _DATA_REL).exists()
    assert (repo / "PlotFuncs.py").read_text() == before


def test_second_removal_is_a_clean_noop(repo):
    remove_limit_artifacts(_DATA_REL, repo_root=repo)
    again = remove_limit_artifacts(_DATA_REL, repo_root=repo)
    assert not again.removed_anything
    assert len(again.missing) >= 3
