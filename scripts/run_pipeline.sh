#!/usr/bin/env bash
# Activate the project's Python environment before importing pipeline code.
# The caller's working directory is preserved so relative --run-dir/--source
# arguments resolve where the user typed them, not inside the repository.
set -euo pipefail
AAL_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "straxion" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is required. Activate the straxion environment and retry." >&2
        exit 1
    fi
    AAL_CONDA_BASE="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
    AAL_CONDA_BASE="${AAL_CONDA_BASE:-$(conda info --base)}"
    source "$AAL_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate straxion
fi
export PYTHONPATH="$AAL_SCRIPT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m pipeline.local_runner "$@"
