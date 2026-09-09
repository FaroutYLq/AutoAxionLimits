#!/usr/bin/env bash
# Activate the project's Python environment before importing pipeline code.
set -euo pipefail
AAL_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "straxion" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is required. Activate the straxion environment and retry." >&2
        exit 1
    fi
    AAL_CONDA_BASE="$(conda info --base)"
    source "$AAL_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate straxion
fi
cd "$AAL_SCRIPT_ROOT"
exec python -m pipeline.local_runner "$@"
