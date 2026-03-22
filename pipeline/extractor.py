"""
Extraction agent: download arXiv PDF and extract limit data via Claude.

Strategy:
  Stage 1 — send text/tables to Claude (cheap, accurate when tables present)
  Stage 2 — send figure images to Claude vision (fallback when no table found)
"""

from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
import arxiv
import httpx

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    arxiv_id: str
    paper_title: str
    arxiv_url: str
    coupling_type: Optional[str]           # e.g. "DarkPhoton"
    is_new_limit: bool                     # False → skip
    is_projection: bool                    # True → Projections/ subdirectory
    data_points: list[tuple[float, float]] # [(mass_eV, coupling), ...]
    data_source: str                       # "table" | "figure_vision" | "text"
    dm_density_assumed: Optional[float]    # GeV/cm^3
    polarization_assumption: Optional[str]
    confidence_level: float                # 0.90 or 0.95
    suggested_experiment_name: str
    extraction_confidence: float           # 0.0 – 1.0
    abstract: str = ""
    notes: str = ""                        # Free-form notes from Claude


# ---------------------------------------------------------------------------
# PDF download & parsing
# ---------------------------------------------------------------------------

def download_pdf(arxiv_id: str, workdir: Path) -> Path:
    """Download the arXiv PDF and return local path."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = workdir / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    logger.info("Downloading %s", pdf_url)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        resp = client.get(pdf_url)
        resp.raise_for_status()
    pdf_path.write_bytes(resp.content)
    return pdf_path


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 60_000) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("pymupdf not installed; text extraction unavailable")
        return ""
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    text = "\n".join(parts)
    return text[:max_chars]


def extract_figures_from_pdf(pdf_path: Path, max_figures: int = 10, dpi: int = 150) -> list[Path]:
    """Render PDF pages to PNG images (one per page, up to max_figures)."""
    try:
        import fitz
    except ImportError:
        logger.warning("pymupdf not installed; figure extraction unavailable")
        return []
    doc = fitz.open(str(pdf_path))
    out_dir = pdf_path.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc):
        if i >= max_figures:
            break
        pix = page.get_pixmap(matrix=mat)
        img_path = out_dir / f"page_{i:03d}.png"
        pix.save(str(img_path))
        paths.append(img_path)
    doc.close()
    return paths


# ---------------------------------------------------------------------------
# Extraction agent
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = """\
You are a particle physics expert helping to extract experimental exclusion limits \
from arXiv papers about axions, dark photons, and other ultralight dark matter searches.

Your task is to determine:
1. Whether the paper presents a NEW measured/observed exclusion limit or sensitivity projection.
2. What coupling type it constrains (DarkPhoton, AxionPhoton, AxionElectron, AxionNeutron,
   AxionProton, AxionEDM, AxionCPV, AxionMass, MonopoleDipole, ScalarPhoton, ScalarElectron,
   ScalarBaryon, ScalarNucleon, VectorBL — or null if none of these).
3. The actual numerical limit data as (mass_eV, coupling) pairs.
4. Any DM density assumption (GeV/cm^3).
5. A suggested experiment/detector name (e.g. "SENSEI2024", "ADMX_SLIC").

Respond ONLY with a JSON object with these keys:
{
  "is_new_limit": bool,
  "is_projection": bool,
  "coupling_type": str | null,
  "data_points": [[mass_eV, coupling], ...],
  "data_source": "table" | "text" | "none",
  "dm_density_assumed": float | null,
  "polarization_assumption": str | null,
  "confidence_level": 0.90 or 0.95,
  "suggested_experiment_name": str,
  "extraction_confidence": float,
  "notes": str
}

If you cannot find data, set data_points to [] and extraction_confidence < 0.3.
Use scientific notation in data_points (Python float literals accepted).
All masses must be in eV.
"""

_STAGE2_SYSTEM = """\
You are a particle physics expert reading exclusion limit plots from papers about dark matter.

I am providing images of paper pages. Your task is to trace the LOWER boundary of the \
exclusion/constraint region on any limit plot you find and return 20–50 (mass, coupling) pairs \
along that boundary.

The x-axis is the particle mass in eV (usually log scale).
The y-axis is the coupling constant (log scale).
The excluded region is ABOVE the boundary (higher coupling values are excluded).

Respond ONLY with a JSON object:
{
  "found_limit_plot": bool,
  "coupling_type": str | null,
  "data_points": [[mass_eV, coupling], ...],
  "dm_density_assumed": float | null,
  "polarization_assumption": str | null,
  "confidence_level": 0.90 or 0.95,
  "suggested_experiment_name": str,
  "extraction_confidence": float,
  "notes": str
}
"""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from Claude's response (handles markdown code blocks)."""
    # Try to find JSON block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        import json
        return json.loads(match.group(1))
    # Try raw JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        import json
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in response: {text[:300]}")


def run_extraction_agent(
    paper: arxiv.Result,
    pdf_path: Path,
    client: anthropic.Anthropic,
) -> ExtractionResult:
    """Run two-stage extraction: text first, vision fallback."""
    arxiv_id = re.sub(r"v\d+$", "", paper.entry_id.split("/")[-1])
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

    # --- Stage 1: text/table extraction ---
    pdf_text = extract_text_from_pdf(pdf_path)
    stage1_result = _run_stage1(paper, pdf_text, client)

    if (
        stage1_result.get("is_new_limit")
        and stage1_result.get("data_points")
        and stage1_result.get("extraction_confidence", 0) >= 0.4
    ):
        data_source = stage1_result.get("data_source", "table")
        logger.info(
            "Stage 1 succeeded for %s (%d points, conf=%.2f)",
            arxiv_id,
            len(stage1_result["data_points"]),
            stage1_result.get("extraction_confidence", 0),
        )
    else:
        # --- Stage 2: vision fallback ---
        logger.info("Stage 1 insufficient for %s; trying vision", arxiv_id)
        figure_paths = extract_figures_from_pdf(pdf_path)
        stage2_result = _run_stage2(paper, figure_paths, client) if figure_paths else {}
        if stage2_result.get("found_limit_plot") and stage2_result.get("data_points"):
            # Merge stage1 metadata with stage2 data
            stage1_result["data_points"] = stage2_result["data_points"]
            stage1_result["data_source"] = "figure_vision"
            stage1_result["extraction_confidence"] = stage2_result.get(
                "extraction_confidence", 0.4
            )
            stage1_result["is_new_limit"] = True
            if stage2_result.get("coupling_type"):
                stage1_result["coupling_type"] = stage2_result["coupling_type"]
            if stage2_result.get("dm_density_assumed"):
                stage1_result["dm_density_assumed"] = stage2_result["dm_density_assumed"]
            if stage2_result.get("suggested_experiment_name"):
                stage1_result["suggested_experiment_name"] = stage2_result[
                    "suggested_experiment_name"
                ]
            stage1_result["notes"] = (
                stage1_result.get("notes", "")
                + " | Vision: "
                + stage2_result.get("notes", "")
            )
        else:
            logger.info("Both stages failed for %s", arxiv_id)

    data_points = [
        (float(m), float(g)) for m, g in stage1_result.get("data_points", [])
    ]

    return ExtractionResult(
        arxiv_id=arxiv_id,
        paper_title=paper.title,
        arxiv_url=arxiv_url,
        coupling_type=stage1_result.get("coupling_type"),
        is_new_limit=bool(stage1_result.get("is_new_limit", False)),
        is_projection=bool(stage1_result.get("is_projection", False)),
        data_points=data_points,
        data_source=stage1_result.get("data_source", "none"),
        dm_density_assumed=stage1_result.get("dm_density_assumed"),
        polarization_assumption=stage1_result.get("polarization_assumption"),
        confidence_level=float(stage1_result.get("confidence_level", 0.9)),
        suggested_experiment_name=stage1_result.get("suggested_experiment_name", "Unknown"),
        extraction_confidence=float(stage1_result.get("extraction_confidence", 0.0)),
        abstract=paper.summary[:1000],
        notes=stage1_result.get("notes", ""),
    )


def _run_stage1(paper: arxiv.Result, pdf_text: str, client: anthropic.Anthropic) -> dict:
    prompt = (
        f"Title: {paper.title}\n\n"
        f"Abstract: {paper.summary[:2000]}\n\n"
        f"--- Paper text (truncated) ---\n{pdf_text}\n"
    )
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=_STAGE1_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json_response(resp.content[0].text)
    except Exception as e:
        logger.warning("Stage 1 failed: %s", e)
        return {"is_new_limit": False, "data_points": [], "extraction_confidence": 0.0}


def _run_stage2(
    paper: arxiv.Result, figure_paths: list[Path], client: anthropic.Anthropic
) -> dict:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Title: {paper.title}\nAbstract: {paper.summary[:500]}\n\n"
                "Please examine the following pages for exclusion limit plots "
                "and trace the constraint boundary."
            ),
        }
    ]
    for img_path in figure_paths[:8]:  # limit API payload
        img_data = base64.standard_b64encode(img_path.read_bytes()).decode()
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_data},
            }
        )
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=_STAGE2_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_json_response(resp.content[0].text)
    except Exception as e:
        logger.warning("Stage 2 failed: %s", e)
        return {"found_limit_plot": False, "data_points": [], "extraction_confidence": 0.0}
