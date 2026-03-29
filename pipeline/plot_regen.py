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

    # Deep-copy so we don't mutate the original notebook on disk
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
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if call_line not in source:
            continue

        # Wrap the call so only it draws in colour, with bright red override.
        # For single-point or narrow limits the fill_between spike is
        # infinitely thin and invisible, so we re-draw the limit as a
        # visible red band with a small finite width in log-space.
        spike_code = ""
        if data_file_path:
            spike_code = (
                f'_hl_dat = loadtxt("{data_file_path}", ndmin=2)\n'
                f'_hl_y2 = ax.get_ylim()[1]\n'
                f'for _hl_row in _hl_dat:\n'
                f'    _hl_m, _hl_g = _hl_row[0], _hl_row[1]\n'
                f'    _hl_w = _hl_m * 0.15\n'
                f'    ax.fill_between([_hl_m - _hl_w, _hl_m + _hl_w],\n'
                f'        [_hl_g, _hl_g], y2=_hl_y2,\n'
                f'        facecolor="red", edgecolor="darkred", '
                f'lw=1.5, zorder=1000, alpha=0.85)\n'
            )
        source = source.replace(
            call_line,
            f"_HIGHLIGHT_ACTIVE = True\n{hl_call}\n{spike_code}_HIGHLIGHT_ACTIVE = False",
        )

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

    if not highlight_plots:
        logger.warning("Could not find cell with %r for highlighting", call_line)
        return False, "No matching cell found for highlight", []

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

    # 4. Write to a temp notebook alongside the original (same directory so
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

        # Collect the output file paths that were actually produced
        produced: list[str] = []
        for name in highlight_plots:
            for rel in [f"plots/{name}.pdf", f"plots/plots_png/{name}.png"]:
                if (repo_root / rel).exists():
                    produced.append(rel)

        return result.returncode == 0, result.stderr, produced
    finally:
        # Clean up temporary notebook
        if tmp_nb_path.exists():
            tmp_nb_path.unlink()
