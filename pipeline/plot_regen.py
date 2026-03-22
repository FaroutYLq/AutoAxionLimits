"""
Headless notebook execution via nbconvert.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


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
