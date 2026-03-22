"""
Reviewer agent: apply physical corrections, generate repo artifacts.

Takes an ExtractionResult and produces a ReviewResult containing:
  - corrected data file content
  - PlotFuncs.py static method code
  - notebook call line
  - docs entry
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import arxiv

from .config import COUPLING_TYPES, PHYSICAL_CORRECTIONS
from .extractor import ExtractionResult

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ReviewResult:
    arxiv_id: str
    data_file_path: str           # relative, e.g. "limit_data/DarkPhoton/SENSEI2024.txt"
    data_file_content: str        # 2-col ASCII with header comments
    plotfuncs_method: str         # Complete @staticmethod block
    plotfuncs_file: str           # "PlotFuncs.py" or "PlotFuncs_ScalarVector.py"
    plotfuncs_class: str          # e.g. "DarkPhoton"
    notebook_path: str            # first notebook for this coupling
    notebook_call: str            # e.g. "DarkPhoton.SENSEI2024(ax)\n"
    docs_entry: str               # Markdown bullet
    docs_file: str                # e.g. "docs/dp.md"
    corrections_applied: list[str]
    corrections_flagged: list[str]
    extraction_confidence: float
    low_confidence: bool
    is_projection: bool
    paper_title: str
    arxiv_url: str
    experiment_name: str


# ---------------------------------------------------------------------------
# Physical corrections (deterministic)
# ---------------------------------------------------------------------------

def apply_dm_density_correction(
    data_points: list[tuple[float, float]],
    rho_paper: float,
    rho_repo: float = 0.45,
) -> tuple[list[tuple[float, float]], str]:
    """Scale coupling values by sqrt(rho_repo / rho_paper)."""
    factor = math.sqrt(rho_repo / rho_paper)
    corrected = [(m, g * factor) for m, g in data_points]
    note = (
        f"DM density: paper={rho_paper} GeV/cm³ → repo={rho_repo} GeV/cm³; "
        f"factor=sqrt({rho_repo}/{rho_paper})={factor:.4f}"
    )
    return corrected, note


def apply_corrections(
    result: ExtractionResult,
) -> tuple[list[tuple[float, float]], list[str], list[str]]:
    """Apply deterministic corrections; flag others for human review."""
    corrections = PHYSICAL_CORRECTIONS.get(result.coupling_type or "", {})
    data = list(result.data_points)
    applied: list[str] = []
    flagged: list[str] = []

    # DM density correction
    rho_paper = result.dm_density_assumed
    if rho_paper is not None:
        rho_repo = corrections.get("dm_density", {}).get("repo_convention", 0.45)
        if abs(rho_paper - rho_repo) > 0.01:
            data, note = apply_dm_density_correction(data, rho_paper, rho_repo)
            applied.append(note)

    # Polarization — flag for human review
    if result.polarization_assumption:
        pol_desc = corrections.get("polarization", {}).get("description", "")
        flagged.append(
            f"Polarization assumption: '{result.polarization_assumption}'. {pol_desc}"
        )

    return data, applied, flagged


# ---------------------------------------------------------------------------
# Data file formatting
# ---------------------------------------------------------------------------

def format_data_file(
    data_points: list[tuple[float, float]],
    result: ExtractionResult,
    corrections_applied: list[str],
) -> str:
    """Format corrected data as 2-column ASCII with header comments."""
    header = (
        f"# {result.paper_title}\n"
        f"# arXiv: {result.arxiv_url}\n"
        f"# Coupling type: {result.coupling_type}\n"
        f"# Extracted from: {result.data_source}\n"
        f"# Confidence level: {result.confidence_level}\n"
        f"# Extraction confidence: {result.extraction_confidence:.2f}\n"
    )
    if corrections_applied:
        header += "# Corrections applied:\n"
        for c in corrections_applied:
            header += f"#   {c}\n"
    header += "# mass [eV]    coupling\n"
    rows = "\n".join(f"{m:.6e}   {g:.6e}" for m, g in sorted(data_points))
    return header + rows + "\n"


# ---------------------------------------------------------------------------
# PlotFuncs.py method generation via Claude
# ---------------------------------------------------------------------------

_METHOD_GEN_SYSTEM = """\
You are a Python expert specialising in matplotlib-based scientific visualisation.

You will be given:
1. Two example @staticmethod method bodies from PlotFuncs.py as style exemplars.
2. A new experiment name, coupling type, and data file path.

Generate a COMPLETE @staticmethod method following the EXACT same style (loadtxt, fill_between,
y2 = ax.get_ylim()[1], conditional text_on, etc.).

Return ONLY the Python code block — no explanations, no markdown fences.
The method signature must be:
    def {name}(ax, col='crimson', fs=15, text_on=True, lw=1.5):
"""

_EXEMPLAR_METHODS = [
    # A minimal single-dataset method (SENSEI pattern)
    textwrap.dedent("""\
        def SENSEI(ax,col='firebrick',fs=21,text_on=True,lw=1.5):
            y2 = ax.get_ylim()[1]
            dat = loadtxt("limit_data/DarkPhoton/SENSEI.txt")
            dat[:,1] = dat[:,1]*sqrt(0.3/0.45)
            plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,zorder=1)
            plt.plot(dat[:,0],dat[:,1],color='k',alpha=1,zorder=1,lw=lw)
            if text_on:
                plt.text(3e-3,1.5e-14,r'{\\bf SENSEI}',fontsize=fs,color=col,rotation=0,
                    rotation_mode='anchor',ha='center',va='center',clip_on=True)
            return
    """),
]


def generate_plotfuncs_method(
    experiment_name: str,
    data_file_path: str,
    coupling_type: str,
    client: anthropic.Anthropic,
) -> str:
    """Ask Claude to generate a PlotFuncs.py static method."""
    exemplars = "\n\n".join(f"# Example:\n{m}" for m in _EXEMPLAR_METHODS)
    prompt = (
        f"{exemplars}\n\n"
        f"# Now generate a new method:\n"
        f"# Experiment name: {experiment_name}\n"
        f"# Coupling type: {coupling_type}\n"
        f"# Data file path: {data_file_path}\n"
        f"# Class: {coupling_type}\n\n"
        f"Generate the complete @staticmethod method for {experiment_name}."
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=_METHOD_GEN_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    code = resp.content[0].text.strip()
    # Remove any markdown fences if present
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code


# ---------------------------------------------------------------------------
# Notebook insertion helper
# ---------------------------------------------------------------------------

def generate_notebook_call(
    experiment_name: str, coupling_type: str, notebook_path: str
) -> str:
    """Return the one-liner to add to the notebook."""
    return f"{coupling_type}.{experiment_name}(ax)\n"


# ---------------------------------------------------------------------------
# Docs entry generation
# ---------------------------------------------------------------------------

def generate_docs_entry(result: ExtractionResult, experiment_name: str) -> str:
    """Return a Markdown bullet point for the docs file."""
    projection_tag = " *(projection)*" if result.is_projection else ""
    return (
        f"- **{experiment_name}**{projection_tag}: "
        f"[{result.paper_title}]({result.arxiv_url})\n"
    )


# ---------------------------------------------------------------------------
# PlotFuncs.py insertion via AST
# ---------------------------------------------------------------------------

def insert_method_into_plotfuncs(
    plotfuncs_path: Path,
    class_name: str,
    method_code: str,
) -> None:
    """
    Insert a new static method at the end of class_name in plotfuncs_path.
    Uses ast.parse() to locate the class boundary — never regex.
    """
    source = plotfuncs_path.read_text()
    tree = ast.parse(source)

    # Find the class definition
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            break

    if class_node is None:
        raise ValueError(f"Class '{class_name}' not found in {plotfuncs_path}")

    # Find the last line of the class
    last_line = max(
        getattr(child, "end_lineno", class_node.end_lineno)
        for child in ast.walk(class_node)
        if hasattr(child, "end_lineno")
    )

    lines = source.splitlines(keepends=True)

    # Determine indentation from existing methods (4 spaces inside class)
    indent = "    "
    indented_method = textwrap.indent(method_code.rstrip(), indent) + "\n"

    # Insert after last_line (0-indexed: last_line - 1)
    insert_pos = last_line  # insert *after* the last line of the class
    lines.insert(insert_pos, "\n" + indented_method + "\n")

    plotfuncs_path.write_text("".join(lines))
    logger.info(
        "Inserted method into %s::%s at line %d", plotfuncs_path.name, class_name, insert_pos
    )


# ---------------------------------------------------------------------------
# Notebook insertion via nbformat
# ---------------------------------------------------------------------------

def insert_notebook_call(notebook_path: Path, notebook_call: str) -> None:
    """
    Find the first code cell that calls the coupling class and append the new call.
    Uses nbformat — never raw string manipulation.
    """
    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)
    coupling_class = notebook_call.split(".")[0]

    # Find the last code cell that already contains calls to this class
    target_cell_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and f"{coupling_class}." in cell.source:
            target_cell_idx = i

    if target_cell_idx is None:
        logger.warning(
            "No cell with %s.* found in %s; appending new cell", coupling_class, notebook_path
        )
        new_cell = nbformat.v4.new_code_cell(source=notebook_call)
        nb.cells.append(new_cell)
    else:
        nb.cells[target_cell_idx].source += f"\n{notebook_call}"

    nbformat.write(nb, str(notebook_path))
    logger.info("Updated notebook %s", notebook_path.name)


# ---------------------------------------------------------------------------
# Main reviewer entrypoint
# ---------------------------------------------------------------------------

def run_reviewer_agent(
    result: ExtractionResult,
    client: anthropic.Anthropic,
) -> ReviewResult:
    """Produce all repo artifacts from an ExtractionResult."""
    if result.coupling_type is None:
        raise ValueError(f"Cannot review paper {result.arxiv_id}: no coupling type")

    cfg = COUPLING_TYPES[result.coupling_type]
    experiment_name = _sanitize_name(result.suggested_experiment_name)

    # --- Apply physical corrections ---
    corrected_data, applied, flagged = apply_corrections(result)

    if not corrected_data:
        raise ValueError(f"No data points for {result.arxiv_id} after corrections")

    # --- Data file ---
    if result.is_projection:
        data_dir = f"{cfg['data_dir']}/Projections"
    else:
        data_dir = cfg["data_dir"]
    data_file_rel = f"{data_dir}/{experiment_name}.txt"
    data_file_content = format_data_file(corrected_data, result, applied)

    # --- PlotFuncs method ---
    method_code = generate_plotfuncs_method(
        experiment_name,
        data_file_rel,
        result.coupling_type,
        client,
    )

    # --- Notebook ---
    notebook_path = cfg["notebooks"][0]
    notebook_call = generate_notebook_call(
        experiment_name, result.coupling_type, notebook_path
    )

    # --- Docs ---
    docs_entry = generate_docs_entry(result, experiment_name)

    low_confidence = result.extraction_confidence < 0.6

    return ReviewResult(
        arxiv_id=result.arxiv_id,
        data_file_path=data_file_rel,
        data_file_content=data_file_content,
        plotfuncs_method=method_code,
        plotfuncs_file=cfg["plotfuncs_file"],
        plotfuncs_class=cfg["class_name"],
        notebook_path=notebook_path,
        notebook_call=notebook_call,
        docs_entry=docs_entry,
        docs_file=cfg["docs_file"],
        corrections_applied=applied,
        corrections_flagged=flagged,
        extraction_confidence=result.extraction_confidence,
        low_confidence=low_confidence,
        is_projection=result.is_projection,
        paper_title=result.paper_title,
        arxiv_url=result.arxiv_url,
        experiment_name=experiment_name,
    )


def write_repo_files(review: ReviewResult, repo_root: Path = REPO_ROOT) -> None:
    """Write data file, update PlotFuncs, update notebook, update docs."""
    # 1. Data file
    data_path = repo_root / review.data_file_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(review.data_file_content)
    logger.info("Wrote data file: %s", data_path)

    # 2. PlotFuncs method
    pf_path = repo_root / review.plotfuncs_file
    insert_method_into_plotfuncs(pf_path, review.plotfuncs_class, review.plotfuncs_method)

    # 3. Notebook
    nb_path = repo_root / review.notebook_path
    if nb_path.exists():
        insert_notebook_call(nb_path, review.notebook_call)
    else:
        logger.warning("Notebook not found: %s", nb_path)

    # 4. Docs
    docs_path = repo_root / review.docs_file
    if docs_path.exists():
        existing = docs_path.read_text()
        docs_path.write_text(existing + "\n" + review.docs_entry)
        logger.info("Updated docs: %s", docs_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convert a free-form experiment name into a valid Python identifier."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "Exp_" + name
    return name or "UnknownExp"
