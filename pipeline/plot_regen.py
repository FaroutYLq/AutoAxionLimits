"""
Headless notebook execution via nbconvert.
"""

from __future__ import annotations

import copy
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import json
import re

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def get_notebook_plot_names(notebook_path: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """
    Parse a notebook and return the plot names passed to MySaveFig().

    Returns a list of names (without extension), e.g. ['AxionPhoton_ColliderBounds'].
    Falls back to an empty list if the notebook cannot be read.
    """
    try:
        nb_text = (repo_root / notebook_path).read_text()
        nb = json.loads(nb_text)
    except Exception:
        return []
    names = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for m in re.finditer(r"MySaveFig\s*\(\s*\w+\s*,\s*['\"]([^'\"]+)['\"]", source):
            names.append(m.group(1))
    return names


def execute_notebook(
    notebook_path: str,
    repo_root: Path = REPO_ROOT,
    timeout_seconds: int = 300,
) -> tuple[bool, str]:
    """
    Execute a Jupyter notebook in-place using nbconvert.

    Returns (success, stderr_output).
    cwd=repo_root is critical: loadtxt("limit_data/...") uses relative paths.
    """
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout_seconds}",
        notebook_path,
    ]
    logger.info("Executing notebook: %s", notebook_path)
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Notebook %s executed successfully", notebook_path)
    else:
        logger.warning(
            "Notebook %s failed (rc=%d): %s",
            notebook_path,
            result.returncode,
            result.stderr[-2000:],
        )
    return result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Highlighted plot generation
# ---------------------------------------------------------------------------

# Monkey-patch cell injected at the start of the notebook.
# Intercepts Axes-level drawing calls so that all existing limits render in
# grey, while the new limit (guarded by _HIGHLIGHT_ACTIVE) renders in colour.
_HIGHLIGHT_PATCH_CODE = r'''
import matplotlib.axes as _mpl_axes
import matplotlib.figure as _mpl_figure

_orig_fill_between = _mpl_axes.Axes.fill_between
_orig_fill = _mpl_axes.Axes.fill
_orig_plot = _mpl_axes.Axes.plot
_orig_text = _mpl_axes.Axes.text
_orig_axhline = _mpl_axes.Axes.axhline
_orig_axvline = _mpl_axes.Axes.axvline
_orig_arrow = _mpl_axes.Axes.arrow
_orig_fig_text = _mpl_figure.Figure.text

_HIGHLIGHT_ACTIVE = False
_GREY_FACE = '#dddddd'
_GREY_EDGE = '#bbbbbb'

def _patched_fill_between(self, x, y1, y2=0, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs.pop('color', None)
        kwargs['facecolor'] = _GREY_FACE
        kwargs.pop('edgecolor', None)
        kwargs['edgecolor'] = None
    return _orig_fill_between(self, x, y1, y2=y2, **kwargs)

def _patched_fill(self, *args, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs.pop('color', None)
        kwargs['facecolor'] = _GREY_FACE
        kwargs.pop('edgecolor', None)
        kwargs['edgecolor'] = None
    return _orig_fill(self, *args, **kwargs)

def _patched_plot(self, *args, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        # Strip colour characters from any format-string arg (e.g. 'k-',
        # 'r--', 'b.') so we can safely pass our own color= kwarg without
        # triggering matplotlib's "duplicate colour" ValueError.
        _FMT_COLORS = set('bgrcmykwBGRCMYKW')
        cleaned = []
        for a in args:
            if isinstance(a, str) and len(a) <= 4:
                a = ''.join(ch for ch in a if ch not in _FMT_COLORS) or '-'
            cleaned.append(a)
        kwargs['color'] = _GREY_EDGE
        kwargs['alpha'] = 0.0
        kwargs.pop('path_effects', None)
        return _orig_plot(self, *cleaned, **kwargs)
    return _orig_plot(self, *args, **kwargs)

def _patched_text(self, *args, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs['alpha'] = 0.0
        kwargs.pop('path_effects', None)
    return _orig_text(self, *args, **kwargs)

def _patched_fig_text(self, *args, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs['alpha'] = 0.0
        kwargs.pop('path_effects', None)
    return _orig_fig_text(self, *args, **kwargs)

def _patched_axhline(self, y=0, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs['color'] = _GREY_EDGE
        kwargs['alpha'] = 0.3
    return _orig_axhline(self, y=y, **kwargs)

def _patched_axvline(self, x=0, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs['color'] = _GREY_EDGE
        kwargs['alpha'] = 0.3
    return _orig_axvline(self, x=x, **kwargs)

def _patched_arrow(self, *args, **kwargs):
    if not _HIGHLIGHT_ACTIVE:
        kwargs['alpha'] = 0.0
    return _orig_arrow(self, *args, **kwargs)

_mpl_axes.Axes.fill_between = _patched_fill_between
_mpl_axes.Axes.fill = _patched_fill
_mpl_axes.Axes.plot = _patched_plot
_mpl_axes.Axes.text = _patched_text
_mpl_axes.Axes.axhline = _patched_axhline
_mpl_axes.Axes.axvline = _patched_axvline
_mpl_axes.Axes.arrow = _patched_arrow
_mpl_figure.Figure.text = _patched_fig_text
'''


def _build_highlight_notebook(
    nb: dict,
    notebook_call: str,
    data_file_path: str | None = None,
) -> tuple[dict, list[str]]:
    """
    Pure transform of a notebook dict for highlighted-plot generation.

    Injects the grey-out monkey-patch cell, wraps the current run's call with
    ``_HIGHLIGHT_ACTIVE = True/False`` (plus a bright overlay of
    *data_file_path*), renames the target cell's MySaveFig outputs to
    ``*_highlighted``, and disables MySaveFig in every other cell.

    Targeting is line-exact and last-occurrence: the target cell may already
    contain earlier pipeline-inserted calls or a commented-out copy of this
    one, which a substring match could latch onto. The current run's call is
    always the LAST exact match — insert_notebook_call appends it immediately
    before MySaveFig, after everything already in the cell.

    Returns (patched deep copy, highlight plot names); names is empty when no
    cell contains the call.
    """
    nb = copy.deepcopy(nb)
    call_line = notebook_call.strip()

    # Build the highlighted call: force bright red colour and thick edges,
    # then overlay a prominent marker so the limit is unmissable even for
    # single-point data files.
    # Parse "CouplingClass.Method(ax)" or "CouplingClass.Method(ax, ...)"
    hl_match = re.match(r"(\w+\.\w+)\(ax(.*)\)", call_line)
    if hl_match:
        method_ref = hl_match.group(1)  # e.g. "AxionPhoton.DALI_Prototype"
        extra_args = hl_match.group(2)  # e.g. "" or ", fs=20"
        hl_call = f"{method_ref}(ax{extra_args}, col='red', lw=3)"
    else:
        hl_call = call_line

    # 1. Inject the monkey-patch cell at position 0
    patch_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _HIGHLIGHT_PATCH_CODE.strip().splitlines(keepends=True),
    }
    nb["cells"].insert(0, patch_cell)

    # 2. Find the cell containing the new method call, wrap it with
    #    _HIGHLIGHT_ACTIVE = True / False, and rename MySaveFig outputs.
    highlight_plots: list[str] = []
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.split("\n")
        call_idxs = [i for i, ln in enumerate(lines) if ln.strip() == call_line]
        if not call_idxs:
            continue

        target = call_idxs[-1]
        if len(call_idxs) > 1:
            logger.info(
                "Cell %d contains %d exact copies of %r; wrapping the last one "
                "(the current run's insertion)",
                cell_idx, len(call_idxs), call_line,
            )

        # Wrap the new limit call so it draws in bright red.
        # Re-draw the limit data as a bright overlay so it is unmissable.
        # For multi-point limits draw a single continuous fill_between;
        # for single-point limits draw a spike with finite width.
        spike_code = ""
        if data_file_path:
            spike_code = (
                f'import numpy as _hl_np\n'
                f'_hl_dat = _hl_np.loadtxt("{data_file_path}", ndmin=2)\n'
                f'_hl_y2 = ax.get_ylim()[1]\n'
                f'if len(_hl_dat) > 2:\n'
                f'    ax.fill_between(_hl_dat[:,0], _hl_dat[:,1], y2=_hl_y2,\n'
                f'        facecolor="red", edgecolor="darkred", '
                f'lw=1.5, zorder=1000, alpha=0.85)\n'
                f'else:\n'
                f'    for _hl_row in _hl_dat:\n'
                f'        _hl_m, _hl_g = _hl_row[0], _hl_row[1]\n'
                f'        _hl_w = _hl_m * 0.15\n'
                f'        ax.fill_between([_hl_m - _hl_w, _hl_m + _hl_w],\n'
                f'            [_hl_g, _hl_g], y2=_hl_y2,\n'
                f'            facecolor="red", edgecolor="darkred", '
                f'lw=1.5, zorder=1000, alpha=0.85)\n'
            )
        indent = lines[target][: len(lines[target]) - len(lines[target].lstrip())]
        block = f"_HIGHLIGHT_ACTIVE = True\n{hl_call}\n{spike_code}_HIGHLIGHT_ACTIVE = False"
        lines[target : target + 1] = [indent + ln for ln in block.split("\n")]
        source = "\n".join(lines)
        logger.info(
            "Highlight wraps %r (line %d of notebook cell %d)", call_line, target, cell_idx
        )

        # Keep theoretical benchmarks (QCD axion band, etc.) in their
        # original colours — only experimental constraints should be grey.
        _THEORY_PATTERNS = [".QCDAxion(", ".BlackHoleSpins("]
        new_lines = []
        for line in source.split("\n"):
            stripped = line.strip()
            if any(pat in stripped for pat in _THEORY_PATTERNS) and not stripped.startswith("#"):
                new_lines.append("_HIGHLIGHT_ACTIVE = True")
                new_lines.append(line)
                new_lines.append("_HIGHLIGHT_ACTIVE = False")
            else:
                new_lines.append(line)
        source = "\n".join(new_lines)

        # Rename MySaveFig outputs → *_highlighted
        def _rename_save(m: re.Match) -> str:
            prefix, name, suffix = m.group(1), m.group(2), m.group(3)
            highlight_plots.append(name + "_highlighted")
            return f"{prefix}{name}_highlighted{suffix}"

        source = re.sub(
            r"""(MySaveFig\s*\(\s*\w+\s*,\s*['"])([^'"]+)(['"])""",
            _rename_save,
            source,
        )

        cell["source"] = source.splitlines(keepends=True)
        break  # only patch the first matching cell

    # 3. For all OTHER cells that contain MySaveFig, comment them out so we
    #    don't waste time regenerating unrelated plots.
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "_HIGHLIGHT_ACTIVE" in src:
            continue  # this is the patched cell, skip
        if "MySaveFig" in src:
            # Replace MySaveFig calls with pass so the cell is still valid
            src = re.sub(r"^(MySaveFig\(.+\))", r"# \1  # skipped for highlight", src, flags=re.MULTILINE)
            cell["source"] = src.splitlines(keepends=True)

    return nb, highlight_plots


def _collect_fresh_outputs(
    repo_root: Path, expected: list[str], before: dict[str, int | None]
) -> list[str]:
    """
    Return the *expected* paths actually (re)written since the *before*
    mtime snapshot. Existence alone is not evidence of generation: a stale
    highlighted plot may already sit at the same path (on 2026-07-04 the
    working tree carried master's previous-showcase TEXONO-highlighted plot),
    and reporting it would attach the wrong plot to the PR.
    """
    produced: list[str] = []
    for rel in expected:
        p = repo_root / rel
        if not p.exists():
            continue
        if before.get(rel) is not None and p.stat().st_mtime_ns == before[rel]:
            logger.warning(
                "Highlighted output %s was not regenerated (stale pre-run copy); excluding it",
                rel,
            )
            continue
        produced.append(rel)
    return produced


def execute_notebook_highlighted(
    notebook_path: str,
    notebook_call: str,
    repo_root: Path = REPO_ROOT,
    timeout_seconds: int = 300,
    data_file_path: str | None = None,
) -> tuple[bool, str, list[str]]:
    """
    Execute a modified copy of the notebook that greys out all existing limits
    and highlights only the new one (identified by *notebook_call*).

    The resulting plot files are saved with a ``_highlighted`` suffix so they
    don't overwrite the standard plots.

    *data_file_path* (relative, e.g. "limit_data/AxionPhoton/X.txt") is used
    to overlay a bright marker at the limit's data points.

    Returns (success, stderr, list_of_highlight_plot_relative_paths).
    """
    nb_abs = repo_root / notebook_path
    try:
        nb = json.loads(nb_abs.read_text())
    except Exception as exc:
        return False, f"Cannot read notebook: {exc}", []

    nb, highlight_plots = _build_highlight_notebook(nb, notebook_call, data_file_path)

    if not highlight_plots:
        logger.warning("Could not find cell with %r for highlighting", notebook_call.strip())
        return False, "No matching cell found for highlight", []

    # Snapshot pre-run mtimes of the expected outputs so a stale copy already
    # in the working tree is never reported as this run's output.
    expected: list[str] = []
    for name in highlight_plots:
        expected.extend([f"plots/{name}.pdf", f"plots/plots_png/{name}.png"])
    before: dict[str, int | None] = {}
    for rel in expected:
        p = repo_root / rel
        before[rel] = p.stat().st_mtime_ns if p.exists() else None

    # Write to a temp notebook alongside the original (same directory so
    #    relative imports like `from PlotFuncs import *` still work).
    tmp_name = Path(notebook_path).stem + "_highlighted_tmp.ipynb"
    tmp_nb_path = repo_root / tmp_name
    try:
        tmp_nb_path.write_text(json.dumps(nb, indent=1))

        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={timeout_seconds}",
            tmp_name,
        ]
        logger.info("Executing highlighted notebook: %s", tmp_name)
        result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Highlighted notebook executed successfully")
        else:
            logger.warning(
                "Highlighted notebook failed (rc=%d): %s",
                result.returncode, result.stderr[-2000:],
            )

        # Collect the outputs actually (re)generated by THIS execution —
        # never stale pre-run copies (see _collect_fresh_outputs).
        produced = _collect_fresh_outputs(repo_root, expected, before)

        return result.returncode == 0, result.stderr, produced
    finally:
        # Clean up temporary notebook
        if tmp_nb_path.exists():
            tmp_nb_path.unlink()
