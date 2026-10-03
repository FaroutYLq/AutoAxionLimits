"""
Headless notebook execution via nbconvert.
"""

from __future__ import annotations

import copy
import ast
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import json
import re

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


class PlotGenerationError(RuntimeError):
    """A review cannot be published without freshly generated comparison plots."""


def notebook_environment(repo_root: Path) -> dict[str, str]:
    """Jupyter puts inherited PYTHONPATH ahead of cwd: pin imports to this run."""
    env = os.environ.copy()
    root = str(repo_root.resolve())
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join([root, *paths])
    return env


def _output_snapshot(repo_root: Path, paths: list[str]) -> dict[str, int | None]:
    return {rel: (repo_root / rel).stat().st_mtime_ns
            if (repo_root / rel).exists() else None for rel in paths}


def generate_review_plots(notebook_path: str, notebook_call: str, repo_root: Path,
                          data_file_path: str) -> tuple[list[str], list[str]]:
    """Require successful renders and fresh full/highlighted PDF+PNG pairs.

    Check before creating a branch or marking a paper processed, so renderer
    outages remain retryable and inherited images never masquerade as results.
    """
    expected = [rel for name in get_notebook_plot_names(notebook_path, repo_root)
                for rel in (f"plots/{name}.pdf", f"plots/plots_png/{name}.png")]
    before = _output_snapshot(repo_root, expected)
    ok, err = execute_notebook(notebook_path, repo_root)
    if not ok:
        raise PlotGenerationError(f"Full plot failed for {notebook_path}: {err[-2000:]}")
    normal = _collect_fresh_outputs(repo_root, expected, before)
    ok, err, highlighted = execute_notebook_highlighted(
        notebook_path, notebook_call, repo_root, data_file_path=data_file_path)
    if not ok or not highlighted:
        raise PlotGenerationError(f"Highlighted plot failed for {notebook_path}: {err[-2000:]}")
    required = [p.replace("_highlighted.", ".") for p in highlighted]
    missing = sorted(set(required) - set(normal))
    if missing or not any(p.endswith(".png") for p in highlighted):
        raise PlotGenerationError(f"Missing fresh comparison outputs: {missing or highlighted}")
    return normal, highlighted


def find_plot_target(data_file_path: str, repo_root: Path) -> tuple[str, str]:
    """Resolve an existing data file to its actual method and notebook call.

    File and method names can differ (JWST_Pinetti.txt uses AxionPhoton.JWST).
    Fail explicitly if no direct plotted call exists rather than inventing one.
    """
    from .config import COUPLING_TYPES
    methods = []
    tree = ast.parse((repo_root / "PlotFuncs.py").read_text())
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and any(
                isinstance(n, ast.Constant) and n.value == data_file_path
                for n in ast.walk(method)
            ):
                methods.append(f"{cls.name}.{method.name}")
    coupling = Path(data_file_path).parts[1]
    # A legacy group may load the same data as a newer dedicated method.
    methods.sort(key=lambda name: name.rsplit(".", 1)[-1] != Path(data_file_path).stem)
    for method in methods:
        for notebook in COUPLING_TYPES.get(coupling, {}).get("notebooks", []):
            path = repo_root / notebook
            if not path.exists():
                continue
            for cell in json.loads(path.read_text()).get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if "MySaveFig(" not in source:
                    continue
                for line in source.splitlines():
                    if line.strip().startswith(method + "("):
                        return notebook, line.strip()
    # Some notebooks draw groups: DarkMatterDecay() calls JWST(), for example.
    # Keep the precise limit as the highlight target, never the whole group.
    for method in methods:
        parents = _calling_methods(method, repo_root)
        for notebook in COUPLING_TYPES.get(coupling, {}).get("notebooks", []):
            path = repo_root / notebook
            if not path.exists():
                continue
            for cell in json.loads(path.read_text()).get("cells", []):
                source = "".join(cell.get("source", []))
                if cell.get("cell_type") == "code" and "MySaveFig(" in source and any(
                    line.strip().startswith(parent + "(")
                    for line in source.splitlines() for parent in parents
                ):
                    return notebook, f"{method}(ax)"
    raise PlotGenerationError(f"No plotted notebook call found for {data_file_path}")


def _calling_methods(target: str, repo_root: Path) -> set[str]:
    """Class methods that transitively call the exact target (cycle safe)."""
    tree = ast.parse((repo_root / "PlotFuncs.py").read_text())
    calls = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if isinstance(method, ast.FunctionDef):
                    calls[f"{cls.name}.{method.name}"] = {
                        ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)}
    found = {target}
    while True:
        expanded = found | {name for name, children in calls.items() if children & found}
        if expanded == found:
            return found - {target}
        found = expanded


def _expose_grouped_target(nb: dict, notebook_call: str, repo_root: Path) -> dict:
    """Re-draw a grouped limit only in the temporary highlighted notebook."""
    call = notebook_call.strip()
    if any(call == line.strip() for cell in nb.get("cells", [])
           if cell.get("cell_type") == "code"
           for line in "".join(cell.get("source", [])).splitlines()):
        return nb
    parents = _calling_methods(call.split("(", 1)[0], repo_root)
    nb = copy.deepcopy(nb)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = "".join(cell.get("source", [])).splitlines(keepends=True)
        if not any(line.strip().startswith(parent + "(") for line in lines for parent in parents):
            continue
        for idx, line in enumerate(lines):
            if line.strip().startswith("MySaveFig("):
                lines.insert(idx, call + "\n")
                cell["source"] = lines
                return nb
    return nb


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
        for m in re.finditer(r"(?m)^[ \t]*MySaveFig\s*\(\s*\w+\s*,\s*['\"]([^'\"]+)['\"]", source):
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
    expected = [f"plots/{name}.pdf" for name in get_notebook_plot_names(notebook_path, repo_root)]
    before = _output_snapshot(repo_root, expected)
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=notebook_environment(repo_root),
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
    if result.returncode == 0:
        missing = set(expected) - set(_collect_fresh_outputs(repo_root, expected, before))
        if not expected or missing:
            return False, f"Notebook produced no fresh plots or missed outputs: {sorted(missing)}"
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


def _emphasize_new_artists(ax, before):
    # Restyle the method's actual artists. Never reconstruct exclusions from
    # raw table rows: that loses transformations, direction and topology.
    import math as _math
    import builtins as _builtins
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.text import Text as _Text
    from matplotlib import patheffects as _pe
    artists = [a for a in ax.get_children() if id(a) not in before]
    if not artists:
        raise RuntimeError('Target method produced no artists to highlight')
    background = [a.get_zorder() for a in ax.get_children()
                  if id(a) in before and _math.isfinite(a.get_zorder())]
    base = _builtins.max(background, default=0) + 10
    orders = sorted({a.get_zorder() for a in artists})
    for artist in artists:
        # Keep the target's internal layering, entirely above the background.
        artist.set_zorder(base + orders.index(artist.get_zorder()))
        if isinstance(artist, _Line2D) and len(artist.get_xdata()) == len(artist.get_ydata()) == 1:
            artist.set_marker('o')
            artist.set_markersize(_builtins.max(artist.get_markersize(), 8))
        if isinstance(artist, _Text):
            artist.set_zorder(base + len(orders) + 1)
            artist.set_color('darkred')
            artist.set_path_effects([_pe.withStroke(linewidth=3, foreground='white')])

_mpl_axes.Axes.fill_between = _patched_fill_between
_mpl_axes.Axes.fill = _patched_fill
_mpl_axes.Axes.plot = _patched_plot
_mpl_axes.Axes.text = _patched_text
_mpl_axes.Axes.axhline = _patched_axhline
_mpl_axes.Axes.axvline = _patched_axvline
_mpl_axes.Axes.arrow = _patched_arrow
_mpl_figure.Figure.text = _patched_fig_text
'''


def _strip_kwargs(arg_str: str, names: tuple[str, ...]) -> str:
    """Remove ``, name=<value>`` kwargs from a call-argument fragment.

    *arg_str* is everything after the leading ``ax`` positional (so every kwarg is
    comma-prefixed). Values are assumed simple (identifier or literal, no nested
    comma) — the convention throughout these notebooks — so a non-greedy
    ``[^,)]+`` value match is sufficient.
    """
    for name in names:
        arg_str = re.sub(rf",\s*{name}\s*=\s*[^,)]+", "", arg_str)
    return arg_str


def _build_highlight_call(call_line: str) -> str:
    """Rewrite ``Class.Method(ax, …)`` to force ``col='red', lw=3`` for the overlay.

    Any pre-existing ``col=``/``lw=`` in the source call is stripped first —
    otherwise appending our own produced ``…, col=X, col='red'`` and a SyntaxError
    ("keyword argument repeated") that silently killed the highlighted plot.
    Falls back to the original line if it isn't a recognisable method call.
    """
    m = re.match(r"(\w+\.\w+)\(ax(.*)\)", call_line)
    if not m:
        return call_line
    method_ref, extra_args = m.group(1), m.group(2)
    extra_args = _strip_kwargs(extra_args, ("col", "lw"))
    return f"{method_ref}(ax{extra_args}, col='red', lw=3)"


def _build_highlight_notebook(
    nb: dict,
    notebook_call: str,
    data_file_path: str | None = None,
) -> tuple[dict, list[str]]:
    """
    Pure transform of a notebook dict for highlighted-plot generation.

    Injects the grey-out monkey-patch cell, wraps the current run's call with
    ``_HIGHLIGHT_ACTIVE = True/False``, promotes its actual artists above
    the background, renames the target cell's MySaveFig outputs to
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

    # Build the highlighted call with bright red colour and thick edges.
    # Restyling actual artists preserves the method's scientific geometry.
    # _build_highlight_call de-dupes any existing
    # col=/lw= so the injected kwargs never collide (SyntaxError).
    hl_call = _build_highlight_call(call_line)

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

        # Capture and promote the actual target artists, including annotations.
        # A two-row table may describe a continuous bound; raw-data overlays
        # cannot infer its topology or any physical conversions in the method.
        indent = lines[target][: len(lines[target]) - len(lines[target].lstrip())]
        label = ""
        end = target + 1
        if end < len(lines) and lines[end].endswith("# AAL publication label"):
            label = lines[end].strip() + "\n"
            end += 1
        block = (f"_hl_before = {{id(a) for a in ax.get_children()}}\n"
                 f"_HIGHLIGHT_ACTIVE = True\n{hl_call}\n{label}"
                 f"_HIGHLIGHT_ACTIVE = False\n_emphasize_new_artists(ax, _hl_before)")
        lines[target : end] = [indent + ln for ln in block.split("\n")]
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
        if not p.exists() or p.stat().st_size == 0:
            continue
        if before.get(rel) is not None and p.stat().st_mtime_ns == before[rel]:
            logger.warning(
                "Plot output %s was not regenerated (stale pre-run copy); excluding it",
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

    nb = _expose_grouped_target(nb, notebook_call, repo_root)
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
        result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True,
                                env=notebook_environment(repo_root))

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

        missing = sorted(set(expected) - set(produced))
        if result.returncode != 0 or missing:
            return False, result.stderr + f"\nMissing fresh highlighted outputs: {missing}", []
        return True, result.stderr, produced
    finally:
        # Clean up temporary notebook
        if tmp_nb_path.exists():
            tmp_nb_path.unlink()
