"""Load and validate ground-truth dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
PAPERS_JSON = GROUND_TRUTH_DIR / "papers.json"
DATA_DIR = GROUND_TRUTH_DIR / "data"


@dataclass
class GroundTruthEntry:
    arxiv_id: str
    paper_title: str
    coupling_type: str
    is_new_limit: bool
    is_projection: bool
    data_source_expected: str  # "table" | "figure_vision" | "text"
    confidence_level: float
    dm_density_assumed: Optional[float]
    difficulty: str  # "easy" | "medium" | "hard"
    tags: list[str]
    notes: str
    ground_truth_data_file: Optional[str]  # filename in data/ dir
    reference_repo_file: Optional[str]  # path in repo (for auto-populating)
    ground_truth_mass_range_eV: Optional[tuple[float, float]]
    ground_truth_coupling_range: Optional[tuple[float, float]]
    ground_truth_num_points: Optional[int]
    verified_by: str
    verification_date: str

    def load_data(self) -> Optional[np.ndarray]:
        """Load ground-truth data as Nx2 array (mass_eV, coupling)."""
        if self.ground_truth_data_file is None:
            return None
        path = DATA_DIR / self.ground_truth_data_file
        if not path.exists():
            logger.warning("Ground truth data file not found: %s", path)
            return None
        return np.loadtxt(str(path), ndmin=2)

    def load_reference_data(self, repo_root: Path) -> Optional[np.ndarray]:
        """Load the reference data from the repo (upstream-curated)."""
        if self.reference_repo_file is None:
            return None
        path = repo_root / self.reference_repo_file
        if not path.exists():
            logger.warning("Reference repo file not found: %s", path)
            return None
        return np.loadtxt(str(path), ndmin=2)


def load_ground_truth(path: Path = PAPERS_JSON) -> list[GroundTruthEntry]:
    """Load all ground-truth entries from papers.json."""
    with open(path) as f:
        data = json.load(f)

    assert data["schema_version"] == 1, f"Unsupported schema version: {data['schema_version']}"

    entries = []
    for p in data["papers"]:
        mass_range = p.get("ground_truth_mass_range_eV")
        if mass_range is not None:
            mass_range = tuple(mass_range)
        coupling_range = p.get("ground_truth_coupling_range")
        if coupling_range is not None:
            coupling_range = tuple(coupling_range)

        entries.append(GroundTruthEntry(
            arxiv_id=p["arxiv_id"],
            paper_title=p["paper_title"],
            coupling_type=p["coupling_type"],
            is_new_limit=p["is_new_limit"],
            is_projection=p["is_projection"],
            data_source_expected=p["data_source_expected"],
            confidence_level=p["confidence_level"],
            dm_density_assumed=p.get("dm_density_assumed"),
            difficulty=p["difficulty"],
            tags=p.get("tags", []),
            notes=p.get("notes", ""),
            ground_truth_data_file=p.get("ground_truth_data_file"),
            reference_repo_file=p.get("reference_repo_file"),
            ground_truth_mass_range_eV=mass_range,
            ground_truth_coupling_range=coupling_range,
            ground_truth_num_points=p.get("ground_truth_num_points"),
            verified_by=p["verified_by"],
            verification_date=p["verification_date"],
        ))

    logger.info("Loaded %d ground-truth entries", len(entries))
    return entries


def populate_data_from_repo(repo_root: Path) -> int:
    """Copy reference repo files into ground_truth/data/ for entries that have
    reference_repo_file set but no local data file yet.

    Returns the number of files copied.
    """
    entries = load_ground_truth()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0

    for entry in entries:
        if entry.ground_truth_data_file is None or entry.reference_repo_file is None:
            continue
        dest = DATA_DIR / entry.ground_truth_data_file
        if dest.exists():
            continue
        src = repo_root / entry.reference_repo_file
        if not src.exists():
            logger.warning("Reference file %s not found, skipping", src)
            continue

        # Read, strip comments, write pure data
        lines = []
        with open(src) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(stripped)

        with open(dest, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("Copied %s → %s (%d data lines)", src.name, dest.name, len(lines))
        copied += 1

    return copied
