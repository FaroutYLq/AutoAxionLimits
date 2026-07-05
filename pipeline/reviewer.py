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
import os
import math
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import arxiv

from .config import COUPLING_TYPES, PHYSICAL_CORRECTIONS, VALID_RANGES
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
    "g_app": "AxionProton",
    "alp-proton": "AxionProton",
    "alp proton": "AxionProton",
    "axion proton": "AxionProton",
    "axion-proton": "AxionProton",
    # AxionNeutron (also default for generic "nucleon" coupling)
    "axionneutron": "AxionNeutron",
    "axionnucleon": "AxionNeutron",
    "axion nucleon": "AxionNeutron",
    "axion-nucleon": "AxionNeutron",
    "alp-nucleon": "AxionNeutron",
    "alp nucleon": "AxionNeutron",
    "g_an": "AxionNeutron",
    "g_ann": "AxionNeutron",
    "alp-neutron": "AxionNeutron",
    "alp neutron": "AxionNeutron",
    "axion neutron": "AxionNeutron",
    "axion-neutron": "AxionNeutron",
    # AxionElectron
    "axionelectron": "AxionElectron",
    "g_ae": "AxionElectron",
    "gaee": "AxionElectron",
    "g_aee": "AxionElectron",
    "axion-electron": "AxionElectron",
    "axion-electron coupling": "AxionElectron",
    "alp-electron": "AxionElectron",
    "axion electron": "AxionElectron",
    # AxionPhoton
    "axionphoton": "AxionPhoton",
    "gagg": "AxionPhoton",
    "g_agamma": "AxionPhoton",
    "gaγγ": "AxionPhoton",
    "g_aγγ": "AxionPhoton",
    "axion-photon coupling": "AxionPhoton",
    "axion-diphoton coupling": "AxionPhoton",
    "alp-photon": "AxionPhoton",
    "alp photon": "AxionPhoton",
    "axion photon": "AxionPhoton",
    "axion photon coupling": "AxionPhoton",
    # DarkPhoton
    "darkphoton": "DarkPhoton",
    "dark photon": "DarkPhoton",
    "kinetic mixing": "DarkPhoton",
    "hidden photon": "DarkPhoton",
    "hidden sector photon": "DarkPhoton",
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
    "spin-mass": "MonopoleDipole",
    "spin-mass coupling": "MonopoleDipole",
    "g_s g_p": "MonopoleDipole",
    "fifth force axion": "MonopoleDipole",
    "exotic spin-dependent": "MonopoleDipole",
    # ScalarPhoton
    "scalarphoton": "ScalarPhoton",
    "scalar photon": "ScalarPhoton",
    "scalar-photon": "ScalarPhoton",
    "d_e photon": "ScalarPhoton",
    "d_gamma": "ScalarPhoton",
    "dilaton photon": "ScalarPhoton",
    "fine structure constant variation": "ScalarPhoton",
    "fine structure variation": "ScalarPhoton",
    "alpha variation": "ScalarPhoton",
    "clock comparison photon": "ScalarPhoton",
    # ScalarElectron
    "scalarelectron": "ScalarElectron",
    "scalar electron": "ScalarElectron",
    "scalar-electron": "ScalarElectron",
    "d_me": "ScalarElectron",
    "d_e electron": "ScalarElectron",
    "dilaton electron": "ScalarElectron",
    "electron mass variation": "ScalarElectron",
    "clock comparison electron": "ScalarElectron",
    # ScalarBaryon
    "scalarbaryon": "ScalarBaryon",
    "scalar baryon": "ScalarBaryon",
    "scalar-baryon": "ScalarBaryon",
    "d_baryon": "ScalarBaryon",
    "d_g": "ScalarBaryon",
    "fifth force baryon": "ScalarBaryon",
    "yukawa baryon": "ScalarBaryon",
    "dilaton baryon": "ScalarBaryon",
    "equivalence principle baryon": "ScalarBaryon",
    # ScalarNucleon
    "scalarnucleon": "ScalarNucleon",
    "scalar nucleon": "ScalarNucleon",
    "scalar-nucleon": "ScalarNucleon",
    "d_nucleon": "ScalarNucleon",
    "d_hat": "ScalarNucleon",
    "yukawa_interaction_strength": "ScalarNucleon",
    "yukawa interaction": "ScalarNucleon",
    "yukawa nucleon": "ScalarNucleon",
    "fifth force nucleon": "ScalarNucleon",
    "dilaton nucleon": "ScalarNucleon",
    "equivalence principle nucleon": "ScalarNucleon",
    # VectorBL
    "vectorbl": "VectorBL",
    "vector b-l": "VectorBL",
    "b-l gauge": "VectorBL",
    "b-l gauge boson": "VectorBL",
    "gauged b-l": "VectorBL",
    "g_bl": "VectorBL",
    "u(1)_b-l": "VectorBL",
    "u(1)_{b-l}": "VectorBL",
    "z prime b-l": "VectorBL",
    "baryon minus lepton": "VectorBL",
    "b minus l": "VectorBL",
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
    # Fuzzy: check if any alias is a substring of the input.
    # Use longest match to avoid e.g. "d_g" matching before "d_gamma".
    matches = [(alias, canonical) for alias, canonical in _COUPLING_ALIASES.items() if alias in key]
    if matches:
        alias, canonical = max(matches, key=lambda x: len(x[0]))
        logger.info("Normalized coupling type %r → %r (substring match on %r)", raw, canonical, alias)
        return canonical
    raise KeyError(raw)

# Reviewer/code-gen model. Overridable via REVIEWER_MODEL (workflows set it
# to Haiku for cost, 2026-07-03 user decision); falls back to EXTRACTOR_MODEL
# so a single env switch covers both agents, then to the Opus default.
CLAUDE_MODEL = os.environ.get(
    "REVIEWER_MODEL", os.environ.get("EXTRACTOR_MODEL", "claude-opus-4-8"))

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
    """Scale coupling values by sqrt(rho_paper / rho_repo).

    A DM-search coupling limit scales as g_limit ∝ 1/sqrt(rho_DM) (the excluded
    signal power ∝ rho·g²), so re-expressing a limit quoted at rho_paper in the
    repo's rho_repo=0.45 GeV/cm³ *strengthens* it: g_repo = g_paper·sqrt(rho_paper/rho_repo).
    A higher assumed density means more dark matter, hence a tighter bound. This
    matches the repository's own plotting convention (``PlotFuncs.py`` DM-search
    methods multiply stored paper-native data by ``sqrt(rho_paper/rho_repo)`` at
    plot time, e.g. ``sqrt(0.3/0.45)``). The earlier ``sqrt(rho_repo/rho_paper)``
    was inverted and weakened every density-corrected limit.
    """
    factor = math.sqrt(rho_paper / rho_repo)
    corrected = [(m, g * factor) for m, g in data_points]
    note = (
        f"DM density: paper={rho_paper} GeV/cm³ → repo={rho_repo} GeV/cm³; "
        f"factor=sqrt({rho_paper}/{rho_repo})={factor:.4f}"
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


def validate_data_ranges(
    data_points: list[tuple[float, float]],
    coupling_type: str,
) -> None:
    """Raise ValueError if extracted data falls outside physically reasonable ranges.

    Catches unit-conversion failures (e.g. mass in μeV reported as eV, coupling
    missing a 10^-14 prefactor) before artifacts are generated.
    """
    ranges = VALID_RANGES.get(coupling_type)
    if not ranges or not data_points:
        return
    masses = [m for m, _ in data_points]
    couplings = [abs(g) for _, g in data_points]
    m_lo, m_hi = ranges["mass"]
    g_lo, g_hi = ranges["coupling"]
    if min(masses) < m_lo or max(masses) > m_hi:
        raise ValueError(
            f"Mass range [{min(masses):.2e}, {max(masses):.2e}] eV outside "
            f"valid range [{m_lo:.0e}, {m_hi:.0e}] for {coupling_type} — "
            f"likely a unit conversion error"
        )
    if min(couplings) < g_lo or max(couplings) > g_hi:
        raise ValueError(
            f"Coupling range [{min(couplings):.2e}, {max(couplings):.2e}] outside "
            f"valid range [{g_lo:.0e}, {g_hi:.0e}] for {coupling_type} — "
            f"likely a unit conversion error"
        )


# ---------------------------------------------------------------------------
# Data file formatting
# ---------------------------------------------------------------------------

def format_data_file(
    data_points: list[tuple[float, float]],
    result: ExtractionResult,
    corrections_applied: list[str],
) -> str:
    """Format corrected data as 2-column ASCII with header comments.

    For narrow-band haloscope results, adds boundary closure points at
    coupling=1e0 at the first and last mass values. This ensures
    fill_between closes cleanly at the top of the plot (matching the
    convention used by existing CAPP data files).
    """
    # Use coupling-specific axis labels from config
    cfg = COUPLING_TYPES.get(result.coupling_type or "", {})
    axes = cfg.get("axes", {})
    x_label = axes.get("x", "mass [eV]")
    y_label = axes.get("y", "coupling")
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
    header += f"# {x_label}    {y_label}\n"

    sorted_pts = sorted(data_points)

    # Add boundary closure points for narrow-band results (haloscopes).
    # The convention (see CAPP-1.txt etc.) is to start and end the data at
    # coupling=1e0 so fill_between closes at the top of the plot.
    if len(sorted_pts) >= 2:
        first_mass = sorted_pts[0][0]
        last_mass = sorted_pts[-1][0]
        mass_span = last_mass / first_mass if first_mass > 0 else 1e99
        # "Narrow-band": mass range spans less than a factor of 10
        if mass_span < 10:
            sorted_pts = [(first_mass, 1e0)] + sorted_pts + [(last_mass, 1e0)]

    rows = "\n".join(f"{m:.6e}   {g:.6e}" for m, g in sorted_pts)
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

IMPORTANT: Do NOT draw a black outline with plt.plot() along the boundary — the data file
already includes closure points at coupling=1e0, so fill_between handles the shape correctly.
Only use plt.plot() for the lower boundary edge (dat[:,1]), not the full outline.

CRITICAL: Always use `loadtxt(..., ndmin=2)` to ensure the data array is 2D even for
single-row data files. Without ndmin=2, loadtxt returns a 1D array and dat[:,0] will crash.

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
    # ndmin=2 ensures loadtxt always returns a 2D array even for single-row data files
    textwrap.dedent("""\
        @staticmethod
        def SENSEI(ax,col='firebrick',fs=21,text_on=True,lw=1.5):
            y2 = ax.get_ylim()[1]
            dat = loadtxt("limit_data/DarkPhoton/SENSEI.txt",ndmin=2)
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
    # Guarantee ndmin=2 in loadtxt calls — single-row data files return 1D arrays without it
    code = re.sub(r'loadtxt\(([^)]*?)(?<!ndmin=2)\)', _ensure_ndmin2, code)
    return code


def _ensure_ndmin2(match: re.Match) -> str:
    """Add ndmin=2 to a loadtxt() call if not already present."""
    args = match.group(1)
    if "ndmin" in args:
        return match.group(0)
    return f"loadtxt({args.rstrip()},ndmin=2)"


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

def _normalize_method_indentation(method_code: str) -> str:
    """
    Re-anchor a method block to column 0 so it can be re-indented to class level.

    ``textwrap.dedent`` keys off the *common* minimum indentation, so a single
    column-0 line (e.g. a ``@staticmethod`` decorator prepended ahead of an
    already-indented ``def``) defeats it and leaves the decorator and ``def`` at
    mismatched columns — which is an ``IndentationError``. This normalizes
    structurally instead: decorator and ``def`` lines are anchored to column 0,
    and body lines are dedented by the ``def`` line's original indentation
    (clamped so they never go negative), preserving their relative nesting.
    """
    lines = method_code.rstrip().split("\n")

    def_indent = 0
    for ln in lines:
        m = re.match(r"(\s*)def\s", ln)
        if m:
            def_indent = len(m.group(1))
            break

    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        stripped = ln.lstrip()
        if stripped.startswith("@") or re.match(r"def\s", stripped):
            # Decorator or def header → class-body level (column 0 here).
            out.append(stripped)
        else:
            # Body (and any deeper blocks) → drop one def-level of indent.
            cur = len(ln) - len(stripped)
            out.append(ln[min(cur, def_indent):])
    return "\n".join(out)


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
    The result is validated with ast.parse() and the write is aborted (fail
    closed) if insertion would produce syntactically invalid Python.
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

    # Indent: 4 spaces for class body. Normalize structurally first — the
    # decorator may arrive at column 0 ahead of an indented def, which a plain
    # textwrap.dedent cannot fix (it would leave them at mismatched columns).
    indent = "    "
    indented_method = textwrap.indent(_normalize_method_indentation(method_code), indent) + "\n"

    # Insert after last_method_end (lines list is 0-indexed; line N is index N-1).
    # Inserting at index `last_method_end` places content after line last_method_end.
    insert_pos = last_method_end
    lines.insert(insert_pos, "\n" + indented_method + "\n")

    new_source = "".join(lines)
    # Fail closed: never write syntactically invalid Python — a corrupt
    # PlotFuncs.py breaks every notebook import, so the daily/highlighted plots
    # silently fall back to stale images instead of showing the new limit.
    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        raise ValueError(
            f"Inserting method into {class_name} produced invalid Python "
            f"({exc.msg} at line {exc.lineno}); aborting to avoid corrupting "
            f"{plotfuncs_path.name}."
        ) from exc

    plotfuncs_path.write_text(new_source)
    logger.info(
        "Inserted method into %s::%s after line %d", plotfuncs_path.name, class_name, insert_pos
    )


# ---------------------------------------------------------------------------
# Notebook insertion via nbformat
# ---------------------------------------------------------------------------

def _figsetup_defaults(plotfuncs_path: Path, class_name: str) -> tuple[float | None, float | None]:
    """Return the (m_min, m_max) default arguments of <class_name>.FigSetup, or (None, None)."""
    try:
        tree = ast.parse(plotfuncs_path.read_text())
    except (OSError, SyntaxError):
        return None, None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "FigSetup"):
                continue
            found: dict[str, float] = {}
            a = item.args
            # Defaults align to the tail of positional args.
            posargs = a.args
            for arg, dflt in zip(posargs[len(posargs) - len(a.defaults):], a.defaults):
                if arg.arg in ("m_min", "m_max") and isinstance(dflt, ast.Constant):
                    found[arg.arg] = float(dflt.value)
            for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
                if dflt is not None and arg.arg in ("m_min", "m_max") and isinstance(dflt, ast.Constant):
                    found[arg.arg] = float(dflt.value)
            return found.get("m_min"), found.get("m_max")
    return None, None


def _decade_literal(value: float) -> str:
    """Format a power-of-ten bound as a clean source literal, e.g. 1e7, 1e-18."""
    return f"1e{round(math.log10(value))}"


def _extend_figsetup_range(
    cell_source: str,
    data_min: float,
    data_max: float,
    default_min: float | None,
    default_max: float | None,
) -> tuple[str, tuple[float, float] | None]:
    """
    Widen the FigSetup(...) call in *cell_source* so the new limit's mass range
    is on-axis, even if it lies outside the conventional plot window.

    Only ever *extends* (never shrinks) the axis, and only the side that the data
    actually exceeds, so the conventional appearance is preserved for in-range
    limits. Bounds are rounded out to the enclosing decade ("ugly but visible").
    Returns (possibly-modified source, (new_min, new_max) | None).
    """
    m = re.search(r"FigSetup\s*\(", cell_source)
    if not m:
        return cell_source, None
    open_paren = m.end() - 1

    # Find the matching close paren (handles nested parens, quotes, line continuations).
    depth, i, n, in_str = 0, open_paren, len(cell_source), None
    close_paren = -1
    while i < n:
        c = cell_source[i]
        if in_str:
            if c == in_str and cell_source[i - 1] != "\\":
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                close_paren = i
                break
        i += 1
    if close_paren < 0:
        return cell_source, None

    args = cell_source[open_paren + 1:close_paren]

    def _arg(name: str, fallback: float | None) -> float | None:
        mm = re.search(rf"\b{name}\s*=\s*([0-9eE.+\-]+)", args)
        return float(mm.group(1)) if mm else fallback

    cur_min = _arg("m_min", default_min)
    cur_max = _arg("m_max", default_max)

    new_min = 10.0 ** math.floor(math.log10(data_min)) if data_min > 0 else None
    new_max = 10.0 ** math.ceil(math.log10(data_max)) if data_max > 0 else None

    # Only extend a side when its current bound is known AND the data exceeds it
    # — never clip (which would hide the conventional limits).
    set_min = new_min is not None and cur_min is not None and new_min < cur_min
    set_max = new_max is not None and cur_max is not None and new_max > cur_max
    if not set_min and not set_max:
        return cell_source, None

    def _set_kwarg(arg_str: str, name: str, literal: str) -> str:
        pat = re.compile(rf"(\b{name}\s*=\s*)[0-9eE.+\-]+")
        if pat.search(arg_str):
            return pat.sub(rf"\g<1>{literal}", arg_str, count=1)
        if not arg_str.strip():
            return f"{name}={literal}"
        sep = "" if arg_str.lstrip().startswith(",") else ","
        return f"{name}={literal}{sep}" + arg_str

    if set_min:
        args = _set_kwarg(args, "m_min", _decade_literal(new_min))
    if set_max:
        args = _set_kwarg(args, "m_max", _decade_literal(new_max))

    new_source = cell_source[:open_paren + 1] + args + cell_source[close_paren:]
    final_min = new_min if set_min else cur_min
    final_max = new_max if set_max else cur_max
    return new_source, (final_min, final_max)


def insert_notebook_call(
    notebook_path: Path,
    notebook_call: str,
    mass_range: tuple[float, float] | None = None,
    plotfuncs_path: Path | None = None,
) -> None:
    """
    Find the first code cell that calls the coupling class and append the new call.
    Uses nbformat — never raw string manipulation.

    If *mass_range* (and *plotfuncs_path*, for FigSetup defaults) is provided and
    the new limit falls outside the figure's conventional mass axis, the target
    cell's FigSetup(...) call is widened so the limit is visible — accepting an
    uglier plot rather than silently dropping an out-of-range limit.
    """
    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)
    coupling_class = notebook_call.split(".")[0]

    # Find the first code cell that calls this class AND contains MySaveFig
    # (i.e. a plot-generating cell, not just an import line).
    # Using the first cell ensures the call lands in the primary plot, not a
    # rescaled/secondary variant (e.g. AxionPhoton_Rescaled_NoProjections).
    target_cell_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and f"{coupling_class}." in cell.source and "MySaveFig" in cell.source:
            target_cell_idx = i
            break

    if target_cell_idx is None:
        logger.warning(
            "No cell with %s.* found in %s; appending new cell", coupling_class, notebook_path
        )
        new_cell = nbformat.v4.new_code_cell(source=notebook_call)
        nb.cells.append(new_cell)
    else:
        source = nb.cells[target_cell_idx].source

        # Widen the figure axis if the new limit falls outside the conventional
        # mass window, so an out-of-range limit is shown (ugly) rather than
        # silently dropped off the plot edge.
        if mass_range is not None:
            data_min, data_max = mass_range
            dflt_min, dflt_max = (
                _figsetup_defaults(plotfuncs_path, coupling_class)
                if plotfuncs_path is not None else (None, None)
            )
            source, widened = _extend_figsetup_range(source, data_min, data_max, dflt_min, dflt_max)
            if widened:
                logger.info(
                    "Extended %s FigSetup mass axis to [%.1e, %.1e] for out-of-range limit",
                    coupling_class, widened[0], widened[1],
                )

        # Insert before MySaveFig if present, so the new limit appears in the saved figure
        save_match = re.search(r"\nMySaveFig\(", source)
        if save_match:
            insert_at = save_match.start()
            nb.cells[target_cell_idx].source = (
                source[:insert_at] + f"\n{notebook_call.rstrip()}" + source[insert_at:]
            )
        else:
            nb.cells[target_cell_idx].source = source + f"\n{notebook_call}"

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

    # --- Validate data ranges (catch unit-conversion errors) ---
    validate_data_ranges(corrected_data, canonical)

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


def class_has_method(plotfuncs_path: Path, class_name: str, method_name: str) -> bool:
    """True if ``class_name`` already defines a method ``method_name`` in the file.

    Uses ``ast`` (never a substring/regex scan) so it can't be fooled by the name
    appearing in a comment, a string, or a *different* class. Returns False if the
    file is missing or unparseable — the caller then falls back to insertion.
    """
    try:
        tree = ast.parse(plotfuncs_path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return True
    return False


def notebook_has_call(notebook_path: Path, coupling_class: str, experiment_name: str) -> bool:
    """True if any code cell already calls ``<coupling_class>.<experiment_name>(``.

    Matches the *call* token specifically (trailing ``(``) so a bare mention in a
    ``loadtxt("…/Name.txt")`` path or a comment is not mistaken for an existing
    call. Returns False on any read/parse error (caller falls back to insertion).
    """
    try:
        nb = json.loads(notebook_path.read_text())
    except (OSError, ValueError):
        return False
    needle = f"{coupling_class}.{experiment_name}("
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if needle in "".join(cell.get("source", [])):
            return True
    return False


def docs_has_entry(docs_path: Path, experiment_name: str) -> bool:
    """True if a docs bullet already documents ``experiment_name``.

    Format-agnostic: matches a list bullet (``-``/``*``) whose first token is the
    experiment name, with or without surrounding ``**bold**``. Curated entries use
    ``* Name: [limit](…)`` while generated ones use ``- **Name**: …`` — both count.
    """
    try:
        text = docs_path.read_text()
    except OSError:
        return False
    pat = re.compile(rf"^\s*[-*]\s*\*{{0,2}}\s*{re.escape(experiment_name)}\b", re.MULTILINE)
    return bool(pat.search(text))


def write_repo_files(review: ReviewResult, repo_root: Path = REPO_ROOT) -> None:
    """Write data file, update PlotFuncs, update notebook, update docs.

    Re-extraction / update semantics: when the experiment is ALREADY curated in the
    repo (its PlotFuncs method, notebook call, or docs bullet exists), only the data
    file is refreshed and the existing artifacts are left untouched. Blindly
    appending a second ``def``/call/bullet used to shadow a hand-curated method with
    a richer signature (e.g. ``edge_on``) and break the notebook with a TypeError.
    Each artifact is guarded independently so a partially-curated state still
    converges correctly.
    """
    # 1. Data file — always refreshed (this IS the update on the re-extraction path)
    data_path = repo_root / review.data_file_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(review.data_file_content)
    logger.info("Wrote data file: %s", data_path)

    # 2. PlotFuncs method — skip if the class already defines it (don't shadow a
    #    hand-curated method whose signature we must not clobber).
    pf_path = repo_root / review.plotfuncs_file
    if class_has_method(pf_path, review.plotfuncs_class, review.experiment_name):
        logger.info(
            "%s.%s already defined — leaving PlotFuncs unchanged (update: data file only)",
            review.plotfuncs_class, review.experiment_name,
        )
    else:
        insert_method_into_plotfuncs(pf_path, review.plotfuncs_class, review.plotfuncs_method)

    # 3. Notebook — skip if a call already exists (avoid a duplicate plot call).
    nb_path = repo_root / review.notebook_path
    nb_class = review.notebook_call.split(".")[0]
    if not nb_path.exists():
        logger.warning("Notebook not found: %s", nb_path)
    elif notebook_has_call(nb_path, nb_class, review.experiment_name):
        logger.info(
            "Notebook %s already calls %s.%s — leaving notebook unchanged",
            nb_path.name, nb_class, review.experiment_name,
        )
    else:
        masses = [
            float(line.split()[0])
            for line in review.data_file_content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        mass_range = (min(masses), max(masses)) if masses else None
        insert_notebook_call(
            nb_path, review.notebook_call,
            mass_range=mass_range,
            plotfuncs_path=pf_path,
        )

    # 4. Docs — skip if an entry already documents this experiment.
    docs_path = repo_root / review.docs_file
    if not docs_path.exists():
        return
    if docs_has_entry(docs_path, review.experiment_name):
        logger.info(
            "Docs %s already document %s — leaving docs unchanged",
            docs_path.name, review.experiment_name,
        )
    else:
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
      - Ultralight only if the data fits entirely below 1e-6 eV
        (the Ultralight plot covers ~1e-24 to 1e-14 eV)
      - Collider only if the data fits entirely above 1e4 eV
      - otherwise → AxionPhoton.ipynb  (the main plot)
    All other multi-notebook couplings default to the first (primary) notebook.
    """
    notebooks = cfg.get("notebooks", [])
    if len(notebooks) == 1 or not data_points:
        return notebooks[0]

    masses = [m for m, _ in data_points]
    min_mass = min(masses)
    max_mass = max(masses)

    # AxionPhoton-specific logic: only pick a specialised notebook if the
    # limit's entire mass range fits within that notebook's axis bounds.
    for nb in notebooks:
        if "Ultralight" in nb and max_mass < 1e-6:
            return nb
        if "Collider" in nb and min_mass > 1e4:
            return nb

    # Default: first notebook in list is always the main comprehensive plot
    return notebooks[0]
