"""Removal of a curated limit's repository artifacts.

The mirror image of ``reviewer.write_repo_files``. When a curated limit loses
its source (the paper is withdrawn from arXiv), the weekly checker proposes the
removal as a reviewable diff rather than only flagging it: the reviewer should
see exactly what would come out and merge or close, not be handed a to-do list.

A curated limit occupies four places, and all four must come out together:

  1. ``limit_data/<Type>/<Name>.txt``   the data file
  2. ``PlotFuncs.py``                   the static method on the coupling class
  3. ``<Coupling>.ipynb``               the ``<Class>.<Name>(ax)`` call
  4. ``docs/<type>.md``                 the bullet documenting the source

Every edit here is structural, matching the insertion side: ``ast`` for
PlotFuncs (never regex), ``nbformat`` for notebooks. Removal is *fail closed* —
if an edit would leave PlotFuncs.py unparseable, or would not actually remove
the method, nothing is written. A corrupt PlotFuncs.py breaks every notebook
import, which is a far worse outcome than a limit that outlives its paper.

Partial removal is reported, never silently accepted: the caller puts the
``missing`` list in the PR body so a human sees which artifacts were already
absent (a hand-curated experiment may never have had a generated docs bullet).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import COUPLING_TYPES

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


@dataclass
class RemovalReport:
    """What a removal actually did, for the PR body and for tests."""

    experiment_name: str
    coupling_type: str
    removed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)

    @property
    def removed_anything(self) -> bool:
        return bool(self.changed_paths)


def coupling_type_from_data_path(data_file_rel: str) -> Optional[str]:
    """``limit_data/DarkPhoton/X.txt`` (or ``.../Projections/X.txt``) -> ``DarkPhoton``."""
    parts = Path(data_file_rel).parts
    if len(parts) < 3 or parts[0] != "limit_data":
        return None
    return parts[1]


# ---------------------------------------------------------------------------
# PlotFuncs.py — AST-located method excision
# ---------------------------------------------------------------------------

def remove_method_from_plotfuncs(
    plotfuncs_path: Path,
    class_name: str,
    method_name: str,
) -> bool:
    """Delete ``class_name.method_name`` from the file. True if it was removed.

    The span deleted runs from the first decorator line (``@staticmethod`` sits
    ABOVE the ``def``, so ``node.lineno`` alone would orphan it) through
    ``end_lineno``. Any immediately preceding comment lines indented at method
    level go too, since they document the method being removed.

    Fail closed: the result must still parse AND must no longer define the
    method, or the file is left untouched.
    """
    try:
        source = plotfuncs_path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        logger.warning("Cannot parse %s: %s", plotfuncs_path, exc)
        return False

    class_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    if class_node is None:
        logger.warning("Class '%s' not found in %s", class_name, plotfuncs_path.name)
        return False

    target = next(
        (
            c for c in class_node.body
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == method_name
        ),
        None,
    )
    if target is None:
        logger.info("%s.%s not present in %s", class_name, method_name, plotfuncs_path.name)
        return False

    # A class body cannot be empty; removing the only method would produce
    # invalid Python. Refuse rather than corrupt the file.
    methods = [
        c for c in class_node.body
        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(methods) == 1 and len(class_node.body) == 1:
        logger.warning(
            "Refusing to remove the only member of class %s (would leave an empty class body)",
            class_name,
        )
        return False

    start = min([target.lineno] + [d.lineno for d in target.decorator_list])
    end = target.end_lineno

    lines = source.splitlines(keepends=True)
    # 1-indexed -> 0-indexed
    start_idx, end_idx = start - 1, end

    # Absorb comment lines directly above that belong to this method.
    while start_idx > 0 and lines[start_idx - 1].strip().startswith("#"):
        start_idx -= 1

    new_lines = lines[:start_idx] + lines[end_idx:]
    new_source = "".join(new_lines)
    # Collapse the blank-line run left behind so the class stays tidy.
    new_source = re.sub(r"\n{4,}", "\n\n\n", new_source)

    try:
        new_tree = ast.parse(new_source)
    except SyntaxError as exc:
        logger.error(
            "Removing %s.%s would produce invalid Python (%s at line %s); aborting",
            class_name, method_name, exc.msg, exc.lineno,
        )
        return False

    still_there = any(
        isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == method_name
        for n in ast.walk(new_tree)
        if isinstance(n, ast.ClassDef) and n.name == class_name
        for c in n.body
    )
    if still_there:
        logger.error(
            "Post-removal check: %s.%s is still defined; aborting", class_name, method_name
        )
        return False

    plotfuncs_path.write_text(new_source)
    logger.info("Removed %s.%s from %s", class_name, method_name, plotfuncs_path.name)
    return True


# ---------------------------------------------------------------------------
# Notebooks — nbformat
# ---------------------------------------------------------------------------

def remove_notebook_call(
    notebook_path: Path,
    coupling_class: str,
    experiment_name: str,
) -> bool:
    """Drop every ``<coupling_class>.<experiment_name>(`` line. True if any went.

    Matches the call token (trailing ``(``) so a ``loadtxt`` path or a comment
    mentioning the name is not mistaken for a call, mirroring
    ``reviewer.notebook_has_call``.
    """
    import nbformat

    try:
        nb = nbformat.read(str(notebook_path), as_version=4)
    except Exception as exc:
        logger.warning("Cannot read notebook %s: %s", notebook_path, exc)
        return False

    needle = f"{coupling_class}.{experiment_name}("
    changed = False
    for cell in nb.cells:
        if cell.cell_type != "code" or needle not in cell.source:
            continue
        kept = [ln for ln in cell.source.splitlines() if needle not in ln]
        cell.source = "\n".join(kept)
        changed = True

    if changed:
        nbformat.write(nb, str(notebook_path))
        logger.info("Removed %s call from %s", needle, notebook_path.name)
    return changed


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

def remove_docs_entry(docs_path: Path, experiment_name: str) -> bool:
    """Drop the bullet documenting ``experiment_name``. True if one went.

    Uses the same bullet pattern as ``reviewer.docs_has_entry``, so both the
    generated ``- **Name**: …`` and the curated ``* Name: [limit](…)`` shapes
    are matched.
    """
    try:
        text = docs_path.read_text()
    except OSError:
        return False

    pat = re.compile(rf"^\s*[-*]\s*\*{{0,2}}\s*{re.escape(experiment_name)}\b")
    kept, dropped = [], 0
    for line in text.splitlines(keepends=True):
        if pat.match(line):
            dropped += 1
            continue
        kept.append(line)

    if not dropped:
        return False

    docs_path.write_text("".join(kept))
    logger.info("Removed %d docs bullet(s) for %s from %s", dropped, experiment_name, docs_path.name)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def describe_limit_artifacts(
    data_file_rel: str,
    repo_root: Path = REPO_ROOT,
    experiment_name: Optional[str] = None,
) -> list[str]:
    """Non-destructive: list the artifacts a removal WOULD touch.

    Used for ``--dry-run`` reporting, so a dry run says what it would delete
    instead of only that it would open a PR. Detection uses the same predicates
    as the insertion side (``reviewer.class_has_method`` and friends) so the
    dry-run listing cannot drift from what the real removal finds.
    """
    from .reviewer import class_has_method, docs_has_entry, notebook_has_call

    experiment_name = experiment_name or Path(data_file_rel).stem
    coupling_type = coupling_type_from_data_path(data_file_rel) or "Unknown"
    cfg = COUPLING_TYPES.get(coupling_type, {})
    class_name = cfg.get("class_name", coupling_type)
    present: list[str] = []

    if (repo_root / data_file_rel).exists():
        present.append(f"data file `{data_file_rel}`")

    plotfuncs_rel = cfg.get("plotfuncs_file", "PlotFuncs.py")
    if class_has_method(repo_root / plotfuncs_rel, class_name, experiment_name):
        present.append(f"`{class_name}.{experiment_name}()` in `{plotfuncs_rel}`")

    for nb_rel in cfg.get("notebooks", []):
        nb_path = repo_root / nb_rel
        if nb_path.exists() and notebook_has_call(nb_path, class_name, experiment_name):
            present.append(f"`{class_name}.{experiment_name}(ax)` call in `{nb_rel}`")

    docs_rel = cfg.get("docs_file")
    if docs_rel and docs_has_entry(repo_root / docs_rel, experiment_name):
        present.append(f"docs entry in `{docs_rel}`")

    return present


def remove_limit_artifacts(
    data_file_rel: str,
    repo_root: Path = REPO_ROOT,
    experiment_name: Optional[str] = None,
) -> RemovalReport:
    """Remove all repository artifacts for the limit stored at *data_file_rel*.

    Returns a RemovalReport listing what came out and what was already absent.
    Never raises for a missing artifact — a partially curated experiment is
    reported, not treated as an error.
    """
    experiment_name = experiment_name or Path(data_file_rel).stem
    coupling_type = coupling_type_from_data_path(data_file_rel) or "Unknown"
    cfg = COUPLING_TYPES.get(coupling_type, {})
    report = RemovalReport(experiment_name=experiment_name, coupling_type=coupling_type)

    # 1. Data file
    data_path = repo_root / data_file_rel
    if data_path.exists():
        data_path.unlink()
        report.removed.append(f"data file `{data_file_rel}`")
        report.changed_paths.append(data_file_rel)
    else:
        report.missing.append(f"data file `{data_file_rel}` (already absent)")

    # 2. PlotFuncs method
    plotfuncs_rel = cfg.get("plotfuncs_file", "PlotFuncs.py")
    class_name = cfg.get("class_name", coupling_type)
    plotfuncs_path = repo_root / plotfuncs_rel
    if remove_method_from_plotfuncs(plotfuncs_path, class_name, experiment_name):
        report.removed.append(f"`{class_name}.{experiment_name}()` in `{plotfuncs_rel}`")
        report.changed_paths.append(plotfuncs_rel)
    else:
        report.missing.append(f"`{class_name}.{experiment_name}()` in `{plotfuncs_rel}`")

    # 3. Notebook calls
    for nb_rel in cfg.get("notebooks", []):
        nb_path = repo_root / nb_rel
        if not nb_path.exists():
            continue
        if remove_notebook_call(nb_path, class_name, experiment_name):
            report.removed.append(f"`{class_name}.{experiment_name}(ax)` call in `{nb_rel}`")
            report.changed_paths.append(nb_rel)

    # 4. Docs bullet
    docs_rel = cfg.get("docs_file")
    if docs_rel:
        docs_path = repo_root / docs_rel
        if remove_docs_entry(docs_path, experiment_name):
            report.removed.append(f"docs entry in `{docs_rel}`")
            report.changed_paths.append(docs_rel)
        else:
            report.missing.append(f"docs entry in `{docs_rel}`")

    logger.info(
        "Removal of %s: %d artifact(s) removed, %d already absent",
        experiment_name, len(report.removed), len(report.missing),
    )
    return report
