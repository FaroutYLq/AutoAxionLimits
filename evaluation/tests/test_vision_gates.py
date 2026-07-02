"""Unit tests for the WS3 wrong-curve vision gates (pipeline/vision_gates.py).

No network/API calls. Every positive case is anchored to the cached full346
snapshot of the eval paper that motivated it (excerpts quoted verbatim), and
every negative case pins a false-trigger class that replay tuning removed —
see evaluation/eval_runs/gate_replay.md for the population-level numbers.

Run:
    pytest evaluation/tests/test_vision_gates.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.vision_gates import (
    GATE_C_MIN_GAP_DEX,
    build_experiment_lexicon,
    check_vision_gates,
    extract_vision_segment,
    gate_axis_vs_coupling,
    gate_compilation_envelope,
    gate_mass_regime,
    gate_projection_target,
    mass_window_gap_dex,
    parse_abstract_mass_window,
)

LEXICON = build_experiment_lexicon(PROJECT_ROOT / "limit_data")


# ---------------------------------------------------------------------------
# Notes plumbing
# ---------------------------------------------------------------------------

class TestExtractVisionSegment:
    def test_composite_notes(self):
        notes = ("Stage-1 text mentioning Eot-Wash limits from the paper. | "
                 "selector: figure_vision (sole candidate) | "
                 "Vision: Traced the paper's own curve. | "
                 "Calibration: No calibration needed | read-vote N=3: ct=X (3/3)")
        assert extract_vision_segment(notes) == "Traced the paper's own curve."

    def test_no_vision_segment(self):
        assert extract_vision_segment("plain stage-1 notes only") == ""

    def test_none_and_empty(self):
        assert extract_vision_segment(None) == ""
        assert extract_vision_segment("") == ""


class TestLexicon:
    def test_common_names_present(self):
        assert "eotwash" in LEXICON
        assert "sn1987a" in LEXICON

    def test_stopwords_and_short_stems_absent(self):
        assert "cast" not in LEXICON     # English-verb collision, stop-listed
        assert "cmb" not in LEXICON      # < 4 chars

    def test_missing_dir_yields_common_names_only(self):
        lex = build_experiment_lexicon("/nonexistent/limit_data")
        assert "eotwash" in lex


# ---------------------------------------------------------------------------
# Gate A — projection-target
# ---------------------------------------------------------------------------

class TestGateA:
    def test_1512_06165_existing_exclusion_phrase(self):
        # ';' joins the admission and the trace verb into one sentence — the
        # sentence splitter must not split on ';'.
        r = gate_projection_target(
            is_projection=True, source="figure_vision",
            vision_notes=("Figure 2 shows B-L vector DM reach. The yellow shaded "
                          "'static EP tests' region is the existing exclusion; its "
                          "lower boundary is traced."),
            lexicon=LEXICON)
        assert r is not None and r.action == "reject"

    def test_1508_01798_lexicon_name_in_trace_sentence(self):
        r = gate_projection_target(
            is_projection=True, source="figure_vision",
            vision_notes=("Traced the orange dashed 'EP d_e' line (equivalence "
                          "principle 95% CL limit from Eot-Wash) as the lower "
                          "exclusion boundary for ScalarElectron coupling d_e."),
            lexicon=LEXICON)
        assert r is not None and r.action == "reject"

    def test_2309_07995_existing_bounds(self):
        r = gate_projection_target(
            is_projection=True, source="figure_vision",
            vision_notes=("The lower boundary traced corresponds to the existing "
                          "bounds (Eot-Wash / LIGO-Virgo) overlapping with the "
                          "HeLIOS prototype (red) reach."),
            lexicon=LEXICON)
        assert r is not None and r.action == "reject"

    def test_not_projection_never_fires(self):
        r = gate_projection_target(
            is_projection=False, source="figure_vision",
            vision_notes="Traced the existing bounds from Eot-Wash.",
            lexicon=LEXICON)
        assert r is None

    def test_text_source_never_fires(self):
        r = gate_projection_target(
            is_projection=True, source="text",
            vision_notes="Traced the existing bounds from Eot-Wash.",
            lexicon=LEXICON)
        assert r is None

    def test_own_experiment_title_guard_1711_08999(self):
        # CASPEr's own overview paper tracing its own combined sensitivity must
        # not be rejected for colliding with CASPEr's published-limit stem.
        r = gate_projection_target(
            is_projection=True, source="figure_vision",
            vision_notes=("4 (CASPEr Wind, page 7), y-axis = ALP nucleon coupling "
                          "g_aNN [GeV^-1], traced lower boundary of combined phase "
                          "I-LF/II sensitivity region."),
            paper_title="Overview of the Cosmic Axion Spin Precession Experiment (CASPEr)",
            lexicon=LEXICON)
        assert r is None

    def test_lexicon_name_outside_trace_sentence_no_fire(self):
        # Describing existing bounds without tracing them is routine.
        r = gate_projection_target(
            is_projection=True, source="figure_vision",
            vision_notes=("Eot-Wash bounds are shown in grey for context. Traced "
                          "the red projected reach of this proposal."),
            lexicon=LEXICON)
        assert r is None

    def test_negated_trace_sentence_no_fire(self):
        r = gate_projection_target(
            is_projection=True, source="figure_vision",
            vision_notes=("The existing Eot-Wash exclusion curves were not used "
                          "for tracing."),
            lexicon=LEXICON)
        assert r is None


# ---------------------------------------------------------------------------
# Gate B — axis vs declared coupling
# ---------------------------------------------------------------------------

class TestGateB:
    NOTES_1708 = ("The provided y-axis calibration (1e-13 to 1e-09) matches the "
                  "left panel of Figure 5, which has y-axis g_agamma [GeV^-1].")

    def test_1708_02111_foreign_axis_fires(self):
        r = gate_axis_vs_coupling(source="figure_vision",
                                  coupling_type="AxionElectron",
                                  vision_notes=self.NOTES_1708)
        assert r is not None and r.action == "reject"
        assert "AxionPhoton" in r.reason

    def test_matching_family_no_fire(self):
        r = gate_axis_vs_coupling(source="figure_vision",
                                  coupling_type="AxionPhoton",
                                  vision_notes=self.NOTES_1708)
        assert r is None

    def test_fa_axis_statement_is_not_foreign(self):
        # 2408.07740 / 2410.19902 / 2410.21590 / 2412.03655: a truthful "the
        # y-axis is 1/f_a [GeV^-1], not m_a" against an AxionMass declaration
        # is unit notation the convention converters handle — never a reject.
        r = gate_axis_vs_coupling(
            source="figure_vision", coupling_type="AxionMass",
            vision_notes="WARNING: The y-axis is 1/f_a [GeV^-1], NOT m_a [eV].")
        assert r is None

    def test_bare_epsilon_is_not_darkphoton(self):
        # 2403.03004: epsilon names the B-L gauge coupling there.
        r = gate_axis_vs_coupling(
            source="figure_vision", coupling_type="VectorBL",
            vision_notes="Y-axis labeled epsilon^{95%}_{B-L}, the B-L gauge "
                         "coupling constant.")
        assert r is None

    def test_foreign_symbol_without_axis_assertion_no_fire(self):
        r = gate_axis_vs_coupling(
            source="figure_vision", coupling_type="AxionElectron",
            vision_notes="The paper also derives g_agamma limits elsewhere.")
        assert r is None


# ---------------------------------------------------------------------------
# Gate C — mass regime vs abstract
# ---------------------------------------------------------------------------

class TestAbstractWindow:
    def test_latex_lesssim_sandwich_1903_12190(self):
        abstract = (r"stronger bounds than all the previous literature on "
                    r"ultra-light hidden photon DM for nearly all of the mass "
                    r"range $10^{-23}\lesssim m_\mathrm{DM} \lesssim 10^{-10}$ eV")
        assert parse_abstract_mass_window(abstract) == pytest.approx((1e-23, 1e-10))

    def test_kev_c2_range_1808_02340(self):
        abstract = ("Limits are placed on the couplings of ALPs or hidden photon "
                    "dark matter in the mass range $0.8 - 500$ keV/c$^2$.")
        assert parse_abstract_mass_window(abstract) == pytest.approx((800.0, 5e5))

    def test_meV_vs_MeV_case_sensitivity(self):
        assert parse_abstract_mass_window("axion masses between 1 and 10 meV") \
            == pytest.approx((1e-3, 1e-2))
        assert parse_abstract_mass_window("dark photon masses between 1 and 10 MeV") \
            == pytest.approx((1e6, 1e7))

    def test_detector_energy_range_without_mass_context_skipped(self):
        assert parse_abstract_mass_window(
            "We search for events depositing between 1 and 7 keV in the detector.") is None

    def test_multiple_distinct_windows_ambiguous(self):
        assert parse_abstract_mass_window(
            "masses between 1 and 10 ueV, and also masses between 1 and 100 keV") is None

    def test_duplicate_window_collapses(self):
        assert parse_abstract_mass_window(
            "masses between 1 and 10 ueV; we scan masses from 1 to 10 µeV") \
            == pytest.approx((1e-6, 1e-5))

    def test_no_window(self):
        assert parse_abstract_mass_window("We measure the electron lifetime.") is None
        assert parse_abstract_mass_window(None) is None


class TestGateC:
    ABSTRACT = ("Limits are placed on the couplings of ALPs or hidden photon "
                "dark matter in the mass range $0.8 - 500$ keV/c$^2$.")

    def test_1808_02340_nominal_mass_point_rejected(self):
        r = gate_mass_regime(data_points=[(1e-3, 1.1e-11)], abstract=self.ABSTRACT)
        assert r is not None and r.action == "reject"

    def test_in_window_no_fire(self):
        r = gate_mass_regime(data_points=[(1e3, 1e-11), (1e5, 1e-10)],
                             abstract=self.ABSTRACT)
        assert r is None

    def test_gap_below_threshold_no_fire(self):
        # 800 eV window edge; 8 eV is only 2 dex away — below the 3-dex floor.
        r = gate_mass_regime(data_points=[(8.0, 1e-11)], abstract=self.ABSTRACT)
        assert r is None

    def test_gap_dex_helper(self):
        assert mass_window_gap_dex([(1e-3, 1)], (800.0, 5e5)) == pytest.approx(5.9, abs=0.05)
        assert mass_window_gap_dex([(1e4, 1)], (800.0, 5e5)) == 0.0
        assert mass_window_gap_dex([], (800.0, 5e5)) is None

    def test_ambiguous_abstract_fails_open(self):
        assert gate_mass_regime(data_points=[(1e-3, 1)], abstract="no mass here") is None
        assert GATE_C_MIN_GAP_DEX == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Gate D — compilation envelope
# ---------------------------------------------------------------------------

class TestGateD:
    def test_1008_3536_union_of_regions(self):
        r = gate_compilation_envelope(
            source="figure_vision",
            vision_notes=("Figure 1 summary plot of combined hidden photon "
                          "exclusion regions. Lower boundary traced across the "
                          "union of all colored (excluded) regions."))
        assert r is not None and r.action == "demote"

    def test_1207_3275_surrounding_context(self):
        r = gate_compilation_envelope(
            source="figure_vision",
            vision_notes=("Boundary traced approximately combining the microwave "
                          "LSW exclusion and surrounding context; coordinates "
                          "approximate from log-log reading."))
        assert r is not None and r.action == "demote"

    def test_own_curve_lower_envelope_no_fire(self):
        # 2005.14184: the "lower envelope" of the paper's OWN noisy curve.
        r = gate_compilation_envelope(
            source="figure_vision",
            vision_notes=("Curve is noisy with many gamma-line peaks; values "
                          "traced represent the lower envelope of the GERDA "
                          "exclusion boundary."))
        assert r is None

    def test_own_combined_channels_no_fire(self):
        # 2301.03433: combining the paper's own two measurement channels,
        # described outside any trace sentence.
        r = gate_compilation_envelope(
            source="figure_vision",
            vision_notes=("Combined lower envelope of Yb+ E3/E2 (pink) and E3/Sr "
                          "(blue) regions in Fig. 2. The boundary was read at 30 "
                          "mass values."))
        assert r is None

    def test_descriptive_compilation_without_trace_verb_no_fire(self):
        # 2008.05355: the figure IS a compilation, but the note only describes it.
        r = gate_compilation_envelope(
            source="figure_vision",
            vision_notes="Confidence moderate due to log-scale reading of "
                         "compilation plot.")
        assert r is None


# ---------------------------------------------------------------------------
# Composite + fail-open discipline
# ---------------------------------------------------------------------------

class TestCheckVisionGates:
    def test_rejects_sort_before_demotes(self):
        results = check_vision_gates(
            source="figure_vision", is_projection=True,
            coupling_type="VectorBL",
            vision_notes=("The lower boundary traced corresponds to the existing "
                          "bounds (Eot-Wash / LIGO-Virgo). Boundary traced across "
                          "the union of all colored regions."),
            lexicon=LEXICON)
        assert [r.action for r in results] == ["reject", "demote"]
        assert results[0].note.startswith("[VISION GATE A]")

    def test_all_empty_inputs_fire_nothing(self):
        assert check_vision_gates(source=None, is_projection=False,
                                  coupling_type=None, vision_notes=None) == []

    def test_never_raises_on_garbage(self):
        assert check_vision_gates(
            source="figure_vision", is_projection=True, coupling_type=123,
            vision_notes=object(), data_points=[("x", None)],
            abstract=b"bytes", suggested_experiment_name=0.5,
            paper_title=(), lexicon=LEXICON) == []
