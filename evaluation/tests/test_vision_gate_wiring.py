"""Wiring tests for the WS3 wrong-curve gates inside the extractor selector
(pipeline/extractor._gate_candidates — #663 PR 2).

No API calls: `_gate_candidates` is pure. These pin the four contract points:
reject -> fall back to the next candidate; gate D -> demote below
figure_vision via the reconstruction flag; gate C also guards the text
candidate; nothing fires on clean inputs.

Run:
    pytest evaluation/tests/test_vision_gate_wiring.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.extractor import _gate_candidates, _make_candidate
from pipeline.transform_guard import quality, select_best

CURVE = [(10 ** (-6 + 0.1 * i), 10 ** (-11 - 0.05 * i)) for i in range(20)]
TEXT_PTS = [(1.1e-6, 2e-11), (5e-6, 3e-11), (2e-5, 6e-11), (9e-5, 2e-10), (3e-4, 8e-10)]

EXISTING_BOUND_NOTES = ("The lower boundary traced corresponds to the existing "
                        "bounds (Eot-Wash / LIGO-Virgo) overlapping with the "
                        "prototype reach.")
ENVELOPE_NOTES = ("Lower boundary traced across the union of all colored "
                  "(excluded) regions.")
CLEAN_NOTES = "Traced the paper's own red exclusion curve from Figure 3."


def test_gate_a_reject_falls_back_to_text():
    text_c = _make_candidate("text", TEXT_PTS, "VectorBL", 0.7)
    vis_c = _make_candidate("figure_vision", CURVE, "VectorBL", 0.6)
    cands, text_c2, vis_c2, _src, notes = _gate_candidates(
        [text_c, vis_c], text_c, vis_c,
        is_projection=True, vision_notes=EXISTING_BOUND_NOTES,
        suggested_experiment_name="Eot-Wash EP", paper_title="A New Proposal",
        abstract=None)
    assert vis_c2 is None and text_c2 is text_c
    assert cands == [text_c]
    assert any("[VISION GATE A]" in n for n in notes)
    chosen, _ = select_best(cands)
    assert chosen is text_c


def test_gate_d_demotes_below_clean_text():
    vis_c = _make_candidate("figure_vision", CURVE, "DarkPhoton", 0.6)
    # A sparse (2-pt) text point-limit ranks below a valid vision curve (P-A1),
    # so the un-gated vision candidate would win here.
    sparse_text = _make_candidate("text", TEXT_PTS[:2], "DarkPhoton", 0.7)
    assert quality(vis_c) > quality(sparse_text)
    cands, _, vis_c2, _src, notes = _gate_candidates(
        [sparse_text, vis_c], sparse_text, vis_c,
        is_projection=False, vision_notes=ENVELOPE_NOTES,
        suggested_experiment_name=None, paper_title=None, abstract=None)
    assert vis_c2 is not None and vis_c2.reconstruction
    assert any("[VISION GATE D]" in n for n in notes)
    # demoted vision now ranks below even the sparse text point-limit
    assert quality(vis_c2) < quality(sparse_text)
    chosen, _ = select_best(cands)
    assert chosen is sparse_text


def test_gate_c_rejects_nominal_mass_text_point():
    # 1808.02340: lone text point at a nominal 1e-3 eV mass vs an abstract
    # stating the 0.8-500 keV search range; the vision candidate remains.
    abstract = ("Limits are placed on the couplings of ALPs or hidden photon "
                "dark matter in the mass range $0.8 - 500$ keV/c$^2$.")
    text_c = _make_candidate("text", [(1e-3, 1.1e-11)], "AxionElectron", 0.45)
    vis_pts = [(10 ** (3 + 0.1 * i), 10 ** (-12 + 0.02 * i)) for i in range(25)]
    vis_c = _make_candidate("figure_vision", vis_pts, "AxionElectron", 0.5)
    cands, text_c2, vis_c2, _src, notes = _gate_candidates(
        [text_c, vis_c], text_c, vis_c,
        is_projection=False, vision_notes=CLEAN_NOTES,
        suggested_experiment_name=None, paper_title=None, abstract=abstract)
    assert text_c2 is None and vis_c2 is vis_c
    assert any("[VISION GATE C]" in n for n in notes)
    chosen, _ = select_best(cands)
    assert chosen is vis_c


def test_all_candidates_rejected_empties_pool():
    vis_c = _make_candidate("figure_vision", CURVE, "VectorBL", 0.6)
    cands, _, vis_c2, _src, notes = _gate_candidates(
        [vis_c], None, vis_c,
        is_projection=True, vision_notes=EXISTING_BOUND_NOTES,
        suggested_experiment_name=None, paper_title=None, abstract=None)
    assert cands == [] and vis_c2 is None and notes
    chosen, _ = select_best(cands)
    assert chosen is None


def test_clean_candidates_untouched():
    text_c = _make_candidate("text", TEXT_PTS, "AxionPhoton", 0.7)
    vis_c = _make_candidate("figure_vision", CURVE, "AxionPhoton", 0.6)
    cands, text_c2, vis_c2, _src, notes = _gate_candidates(
        [text_c, vis_c], text_c, vis_c,
        is_projection=True, vision_notes=CLEAN_NOTES,
        suggested_experiment_name="This Paper's Own Detector",
        paper_title="A New Proposal", abstract="No mass range stated here.")
    assert notes == []
    assert cands == [text_c, vis_c]
    assert text_c2 is text_c and vis_c2 is vis_c
