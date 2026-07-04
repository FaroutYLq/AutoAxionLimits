"""Regression tests for highlighted-plot call targeting in pipeline.plot_regen.

Context (2026-07-04 showcase run, arXiv:2603.22390 / PR #694): the target
DarkPhoton.ipynb cell already contained an earlier pipeline-inserted call
(DarkPhoton.TEXONO_CsI_Tl_reactor_dark_photon_de_excitation_Compton_like), and
the PR was suspected of highlighting it instead of the newly inserted call.
The highlight must always wrap the call inserted by the CURRENT run — never a
pre-existing pipeline-style call, a commented-out copy, or (after a re-run
that duplicated the insertion) anything but the last exact occurrence.

The same incident exposed a second hazard: the run's output collection used to
report any existing plots/<name>_highlighted.* file, so a stale highlighted
plot already in the working tree (e.g. checked out from master) could be
attached to a PR as if this run had produced it.

No API/network; the notebook transform is a pure function.

Run:
    pytest pipeline/tests/test_highlight_targeting.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.plot_regen import _build_highlight_notebook, _collect_fresh_outputs

NEW_CALL = "DarkPhoton.Amorphous_phonon_detector_SiO2_SiNx_projection(ax)\n"
PREEXISTING_CALL = (
    "DarkPhoton.TEXONO_CsI_Tl_reactor_dark_photon_de_excitation_Compton_like(ax)"
)


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _nb(*cell_sources: str) -> dict:
    return {
        "cells": [_code_cell(s) for s in cell_sources],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _active_state_per_line(source: str) -> list[tuple[bool, str]]:
    """Walk a patched cell, tracking the _HIGHLIGHT_ACTIVE flag per line."""
    active = False
    out = []
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped == "_HIGHLIGHT_ACTIVE = True":
            active = True
            continue
        if stripped == "_HIGHLIGHT_ACTIVE = False":
            active = False
            continue
        out.append((active, stripped))
    return out


def _patched_source(nb: dict, cell_idx: int) -> str:
    return "".join(nb["cells"][cell_idx]["source"])


# The showcase cell layout: pre-existing pipeline-style call directly above
# the insertion point, exactly as on master during the 2603.22390 run.
SHOWCASE_CELL = (
    "fig,ax = DarkPhoton.FigSetup(m_max=1e7)\n"
    "DarkPhoton.TEXONO(ax)\n"
    f"{PREEXISTING_CALL}\n"
    f"{NEW_CALL.rstrip()}\n"
    "MySaveFig(fig,'DarkPhoton')"
)


def test_wraps_only_the_current_run_call():
    nb, plots = _build_highlight_notebook(_nb(SHOWCASE_CELL), NEW_CALL)
    assert plots == ["DarkPhoton_highlighted"]

    # cell 0 is the injected monkey-patch cell; cell 1 is the patched target
    states = _active_state_per_line(_patched_source(nb, 1))
    highlighted = [ln for active, ln in states if active and "DarkPhoton." in ln]
    greyed = [ln for active, ln in states if not active and "DarkPhoton." in ln]

    assert any("Amorphous_phonon_detector_SiO2_SiNx_projection(ax" in ln for ln in highlighted)
    assert all("TEXONO" not in ln for ln in highlighted)
    assert any(PREEXISTING_CALL in ln for ln in greyed)
    assert any("DarkPhoton.TEXONO(ax)" == ln for _, ln in states)


def test_highlighted_call_gets_red_override_and_save_renamed():
    nb, _ = _build_highlight_notebook(_nb(SHOWCASE_CELL), NEW_CALL)
    src = _patched_source(nb, 1)
    assert "Amorphous_phonon_detector_SiO2_SiNx_projection(ax, col='red', lw=3)" in src
    assert "MySaveFig(fig,'DarkPhoton_highlighted')" in src
    # the pre-existing call is untouched (drawn grey by the monkey-patch)
    assert f"{PREEXISTING_CALL}\n" in src


def test_duplicate_insertion_wraps_only_the_last_occurrence():
    # A re-run that inserted the same call twice: the current run's insertion
    # is the last one (insert_notebook_call appends just before MySaveFig).
    cell = (
        "fig,ax = DarkPhoton.FigSetup(m_max=1e7)\n"
        f"{NEW_CALL.rstrip()}\n"
        f"{PREEXISTING_CALL}\n"
        f"{NEW_CALL.rstrip()}\n"
        "MySaveFig(fig,'DarkPhoton')"
    )
    nb, plots = _build_highlight_notebook(_nb(cell), NEW_CALL)
    assert plots == ["DarkPhoton_highlighted"]

    states = _active_state_per_line(_patched_source(nb, 1))
    bare = [
        (active, ln) for active, ln in states if ln == NEW_CALL.strip()
    ]
    red = [
        (active, ln) for active, ln in states
        if "Amorphous_phonon_detector_SiO2_SiNx_projection(ax, col='red', lw=3)" in ln
    ]
    # exactly one wrapped (red, active) call; the earlier duplicate stays bare and grey
    assert len(red) == 1 and red[0][0] is True
    assert len(bare) == 1 and bare[0][0] is False


def test_commented_out_copy_does_not_select_the_wrong_cell():
    # A cell earlier in the notebook holds only a commented-out copy of the
    # call (plus its own MySaveFig). The old substring match latched onto it;
    # line-exact matching must skip it and patch the real cell.
    commented_cell = (
        "fig,ax = DarkPhoton.FigSetup(m_max=1e7)\n"
        f"# {NEW_CALL.rstrip()}\n"
        "MySaveFig(fig,'DarkPhoton_Other')"
    )
    nb, plots = _build_highlight_notebook(_nb(commented_cell, SHOWCASE_CELL), NEW_CALL)
    assert plots == ["DarkPhoton_highlighted"]
    # the decoy cell's MySaveFig is disabled, not renamed
    decoy_src = _patched_source(nb, 1)
    assert "# MySaveFig(fig,'DarkPhoton_Other')" in decoy_src


def test_no_matching_cell_returns_empty_plots():
    cell = f"fig,ax = DarkPhoton.FigSetup(m_max=1e7)\n{PREEXISTING_CALL}\nMySaveFig(fig,'DarkPhoton')"
    _, plots = _build_highlight_notebook(_nb(cell), NEW_CALL)
    assert plots == []


def test_stale_preexisting_output_is_not_reported(tmp_path):
    # A highlighted plot already in the tree (e.g. checked out from master)
    # must not be reported as produced unless this run actually rewrote it.
    (tmp_path / "plots" / "plots_png").mkdir(parents=True)
    stale_pdf = tmp_path / "plots" / "DarkPhoton_highlighted.pdf"
    stale_pdf.write_bytes(b"stale")
    expected = [
        "plots/DarkPhoton_highlighted.pdf",
        "plots/plots_png/DarkPhoton_highlighted.png",
    ]
    before = {
        rel: (tmp_path / rel).stat().st_mtime_ns if (tmp_path / rel).exists() else None
        for rel in expected
    }

    # Simulated failed run: nothing regenerated → nothing reported.
    assert _collect_fresh_outputs(tmp_path, expected, before) == []

    # Simulated successful run: the pdf is rewritten and the png appears.
    stale_pdf.write_bytes(b"fresh")
    os.utime(stale_pdf, ns=(before[expected[0]] + 10**9, before[expected[0]] + 10**9))
    (tmp_path / expected[1]).write_bytes(b"fresh")
    assert _collect_fresh_outputs(tmp_path, expected, before) == expected
