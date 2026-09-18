"""Pin the coupling -> docs page mapping in pipeline/config.py.

2026-09-18 daily run: VectorBL pointed at docs/vector.md (the page is
docs/gBL.md).  The reviewer silently skips a missing docs file, but the
publication step still ran `git add docs/vector.md`, which failed and
aborted the run with PublicationError.  Every docs_file must either exist or
be on the explicit list of couplings that have no docs page yet.
"""
from pathlib import Path

from pipeline.config import COUPLING_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]

# Couplings whose docs page does not exist upstream (yet).  The publication
# step skips `git add` for these; remove an entry here once its page lands.
KNOWN_MISSING_DOCS = {"MonopoleDipole", "ScalarBaryon", "ScalarNucleon"}


def test_docs_files_exist_or_are_known_missing():
    for coupling, cfg in COUPLING_TYPES.items():
        exists = (REPO_ROOT / cfg["docs_file"]).exists()
        if coupling in KNOWN_MISSING_DOCS:
            assert not exists, (
                f"{coupling}: {cfg['docs_file']} now exists; drop it from KNOWN_MISSING_DOCS"
            )
        else:
            assert exists, f"{coupling}: docs_file {cfg['docs_file']} does not exist"


def test_corrected_mappings():
    assert COUPLING_TYPES["VectorBL"]["docs_file"] == "docs/gBL.md"
    assert COUPLING_TYPES["AxionProton"]["docs_file"] == "docs/app.md"
    assert COUPLING_TYPES["AxionCPV"]["docs_file"] == "docs/cp.md"
    assert COUPLING_TYPES["ScalarPhoton"]["docs_file"] == "docs/phie.md"
    assert COUPLING_TYPES["ScalarElectron"]["docs_file"] == "docs/phime.md"
