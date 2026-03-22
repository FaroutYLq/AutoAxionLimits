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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import arxiv

from .config import COUPLING_TYPES, PHYSICAL_CORRECTIONS
from .extractor import ExtractionResult, _call_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coupling type normalization
# ---------------------------------------------------------------------------

# Maps common LLM free-form strings → canonical COUPLING_TYPES keys.
_COUPLING_ALIASES: dict[str, str] = {
    # AxionProton
    "axionproton": "AxionProton",
    "g_ap": "AxionProton",
    "alp-proton": "AxionProton",
    "alp proton": "AxionProton",
    "axion proton": "AxionProton",
    "axion-proton": "AxionProton",
    # AxionNeutron
    "axionneutron": "AxionNeutron",
    "g_an": "AxionNeutron",
    "alp-neutron": "AxionNeutron",
    "alp neutron": "AxionNeutron",
    "axion neutron": "AxionNeutron",
    # AxionElectron
    "axionelectron": "AxionElectron",
    "g_ae": "AxionElectron",
    "gaee": "AxionElectron",
    "alp-electron": "AxionElectron",
    "axion electron": "AxionElectron",
    # AxionPhoton
    "axionphoton": "AxionPhoton",
    "gagg": "AxionPhoton",
    "g_agamma": "AxionPhoton",
    "alp-photon": "AxionPhoton",
    "alp photon": "AxionPhoton",
    "axion photon": "AxionPhoton",
    # DarkPhoton
    "darkphoton": "DarkPhoton",
    "dark photon": "DarkPhoton",
    "kinetic mixing": "DarkPhoton",
    "hidden photon": "DarkPhoton",
    # AxionEDM
    "axionedm": "AxionEDM",
    "axion edm": "AxionEDM",
    # AxionCPV
    "axioncpv": "AxionCPV",
    "axion cpv": "AxionCPV",
    "axion cp violation": "AxionCPV",
    # AxionMass
    "axionmass": "AxionMass",
    "axion mass": "AxionMass",
    # MonopoleDipole
    "monopoledipole": "MonopoleDipole",
    "monopole dipole": "MonopoleDipole",
    "monopole-dipole": "MonopoleDipole",
    # ScalarPhoton
    "scalarphoton": "ScalarPhoton",
    "scalar photon": "ScalarPhoton",
    # ScalarElectron
    "scalarelectron": "ScalarElectron",
    "scalar electron": "ScalarElectron",
    # ScalarBaryon
    "scalarbaryon": "ScalarBaryon",
    "scalar baryon": "ScalarBaryon",
    # ScalarNucleon
    "scalarnucleon": "ScalarNucleon",
    "scalar nucleon": "ScalarNucleon",
    # VectorBL
    "vectorbl": "VectorBL",
    "vector b-l": "VectorBL",
    "b-l": "VectorBL",
}


def _normalize_coupling_type(raw: str) -> str:
    """
    Map a free-form LLM coupling type string to a canonical COUPLING_TYPES key.
    Returns the canonical key if found, otherwise raises KeyError.
    """
    # Exact match first
    if raw in COUPLING_TYPES:
        return raw
    # Case-insensitive alias lookup — also try stripping parenthetical suffixes
    key = raw.lower().strip()
    # Strip anything after '(' e.g. "g_ap (ALP-proton coupling)" → "g_ap"
    key_no_paren = key.split("(")[0].strip()
    for candidate in (key, key_no_paren):
        if candidate in _COUPLING_ALIASES:
            canonical = _COUPLING_ALIASES[candidate]
            logger.info("Normalized coupling type %r → %r", raw, canonical)
            return canonical
    raise KeyError(raw)

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

    # DM density correction — only for DM-search haloscope experiments.
    # Guard: coupling type must have dm_density in PHYSICAL_CORRECTIONS (filters out
    # stellar, cosmological, collider couplings) AND Claude must have reported a
    # dm_density_assumed (should only happen for DM-absorption/haloscope results).
    rho_paper = result.dm_density_assumed
    dm_corr_cfg = corrections.get("dm_density")
    if rho_paper is not None and dm_corr_cfg is not None:
        rho_repo = dm_corr_cfg.get("repo_convention", 0.45)
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
1. An example @staticmethod method from PlotFuncs.py as a style exemplar.
2. A new experiment name, coupling type, and data file path.

Generate a COMPLETE static method following the EXACT same style (loadtxt, fill_between,
y2 = ax.get_ylim()[1], conditional text_on, etc.).

IMPORTANT requirements:
- The output must start with the LITERAL decorator line `    @staticmethod`
- Then the `def` line indented by 4 spaces
- The method body indented by 8 spaces
- Return ONLY the code — no explanations, no markdown fences.

The method signature must be:
    def {name}(ax, col='crimson', fs=15, text_on=True, lw=1.5):
"""

_EXEMPLAR_METHODS = [
    # A minimal single-dataset method (SENSEI pattern) — includes @staticmethod decorator
    textwrap.dedent("""\
        @staticmethod
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
    resp = _call_with_retry(lambda: client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=_METHOD_GEN_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    ))
    code = resp.content[0].text.strip()
    # Remove any markdown fences if present
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    # Guarantee @staticmethod is present — if the LLM omitted it, prepend it
    if not re.search(r"^\s*@staticmethod", code, re.MULTILINE):
        code = "@staticmethod\n" + code
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
    Uses ast.parse() to locate the last method's end_lineno — never regex.

    Insertion is INSIDE the class (before its closing line), correctly indented,
    regardless of whether there is trailing whitespace after the last method.
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

    # Find the last direct method (FunctionDef) in the class body.
    # We insert AFTER the last method's end_lineno, which is guaranteed to be
    # inside the class regardless of trailing blank lines.
    last_method_end = class_node.end_lineno  # fallback: class end
    for child in class_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(child, "end_lineno"):
                last_method_end = max(last_method_end, child.end_lineno)

    lines = source.splitlines(keepends=True)

    # Indent: 4 spaces for class body
    indent = "    "
    # Ensure @staticmethod and def are both properly indented
    indented_method = textwrap.indent(method_code.rstrip(), indent) + "\n"

    # Insert after last_method_end (lines list is 0-indexed; line N is index N-1).
    # Inserting at index `last_method_end` places content after line last_method_end.
    insert_pos = last_method_end
    lines.insert(insert_pos, "\n" + indented_method + "\n")

    plotfuncs_path.write_text("".join(lines))
    logger.info(
        "Inserted method into %s::%s after line %d", plotfuncs_path.name, class_name, insert_pos
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
        source = nb.cells[target_cell_idx].source
        # Insert before MySaveFig if present, so the new limit appears in the saved figure
        save_match = re.search(r"\nMySaveFig\(", source)
        if save_match:
            insert_at = save_match.start()
            nb.cells[target_cell_idx].source = (
                source[:insert_at] + f"\n{notebook_call.rstrip()}" + source[insert_at:]
            )
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

    canonical = _normalize_coupling_type(result.coupling_type)
    cfg = COUPLING_TYPES[canonical]
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
    notebook_path = _select_notebook(cfg, result.data_points)
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


def _select_notebook(cfg: dict, data_points: list[tuple[float, float]]) -> str:
    """
    Select the most appropriate notebook for a new limit.

    For couplings with only one notebook, that's always the right choice.
    For AxionPhoton (4 notebooks), pick by mass range:
      - mass < 1e-6 eV  → AxionPhoton_Ultralight.ipynb
      - mass > 1e4 eV   → AxionPhoton_ColliderBounds.ipynb
      - otherwise       → AxionPhoton.ipynb  (the main plot)
    All other multi-notebook couplings default to the first (primary) notebook.
    """
    notebooks = cfg.get("notebooks", [])
    if len(notebooks) == 1 or not data_points:
        return notebooks[0]

    masses = [m for m, _ in data_points]
    min_mass = min(masses)
    max_mass = max(masses)

    # AxionPhoton-specific logic
    for nb in notebooks:
        if "Ultralight" in nb and min_mass < 1e-6:
            return nb
        if "Collider" in nb and max_mass > 1e4:
            return nb

    # Default: first notebook in list is always the main comprehensive plot
    return notebooks[0]
