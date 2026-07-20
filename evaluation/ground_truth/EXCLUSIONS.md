# Ground-truth benchmark exclusions

Entries in `papers.json` carrying `"excluded": true`. An excluded entry stays in
`papers.json` (documented, visible, reversible — never a silent deletion), is
skipped by all scoring in `evaluation/evaluate.py`, and is listed in the
"Excluded GT Entries" table of every report. Source analysis:
`evaluation/eval_runs/failure_analysis_full346_detail.md` (per-paper sections),
summarized in `failure_analysis_full346.md` (Lever 2).

To un-exclude an entry: remove the `excluded`/`exclusion_reason`/
`exclusion_evidence` keys and delete its section here. Each section below states
what would justify that.

---

## AxionMass prediction-band files (13 entries, 11 papers)

**Papers:** 1202.5851 (Kawasaki12), 1505.07455 (Berkowitz15), 1509.00026
(Fleury15), 1606.03145 (Petreczky16), 1608.05414 (Ballesteros16), 1705.00676
(Dine17), 1906.00967 (Buschmann20), 2007.04990 (Gorghetto20 + GorghettoDW_6),
2108.05368 (Buschmann21), 2206.11598 (VISHnu), 2412.08699 (Benabou24 +
Benabou24_DW).

**What the papers report:** theoretical/lattice predictions of the QCD-axion
dark-matter mass window (e.g. "Predicted axion masses (in eV)" in the repo file
headers). These are cosmological mass *predictions*, not exclusion limits on
any coupling.

**Why the GT cannot grade them:** the repo files store `(m_lo, m_hi)` — *both
columns are masses*, consumed by `PlotTheoryMass` as horizontal bands. The GT
auto-expansion mis-ingested column 2 as a dimensionless coupling, producing a
single `(mass, coupling)` "point" that no correct extraction can ever match.
In the full346 run the extractor *correctly* identified each of these papers as
a mass prediction (its notes quote the exact band endpoints) and returned 0
points — correct refusals graded as failures.

**What would un-exclude them:** a dedicated mass-band-overlap comparator
(score the predicted `(m_lo, m_hi)` window against the extraction's reported
window) plus a matching extractor output schema for mass predictions. This
alternative was left open at plan review; until it exists, these entries are
not gradeable.

---

## 2005.14694 — BACON optical-clock network (ScalarPhoton/BACON.txt)

**What the paper reports (arXiv):** 18-digit-accuracy frequency-ratio
measurements. No dark-matter exclusion appears in any arXiv version (full-text
search: zero "boson"/"ultralight"/"d_e" hits).

**Why the GT cannot grade it:** the repo `BACON.txt` d_e curve is digitized
from a figure that exists only in the published Nature 591, 564 (2021) version.
The benchmark feeds the extractor the arXiv PDF, from which the limit is
unreachable.

**What would un-exclude it:** pointing the benchmark's input at the published
PDF (out of scope for the arXiv-PDF pipeline), or an arXiv revision that adds
the DM analysis.

---

## 2011.08693 — Type IIB superradiance landscape (fa/BlackHoleSpins_Mehta.txt)

**What the paper reports:** black-hole-superradiance exclusion statistics over
>2e5 Calabi–Yau compactifications (ensemble KDEs, per-BH exclusion-probability
examples, fraction-excluded vs Hodge number). No general BHSR exclusion curve
in the (m_a, 1/f_a) plane is published in the paper.

**Why the GT cannot grade it:** the 178-point repo curve's own header says
"private communication" — the data was never published in the paper and is
inherently non-extractable from the PDF.

**What would un-exclude it:** the authors publishing the compilation curve in a
paper version the pipeline can read.

---

## 2112.03439 — Breakthrough Listen radio search (AxionPhoton/BreakthroughListen.txt)

**What the paper reports:** model-independent DM decay-rate (λ ~ 1e-32 s⁻¹) and
annihilation cross-section limits over 1020–2700 MHz. It never quotes g_aγγ.

**Why the GT cannot grade it:** the repo g_aγγ = 4.1e-8 GeV⁻¹ value is a
physics conversion O'Hare derived from the λ limit (halo + stimulated-emission
assumptions); the number does not appear in the paper, so the GT demands a
quantity that is not extractable. The extractor's refusal (ct=None, 0 points)
was correct.

**What would un-exclude it:** a vetted decay-rate → g_aγγ converter applied on
*both* sides at scoring time (the Γ-plane converter of Phase 1d covers papers
that plot Γ curves, but this paper's λ is for χ→γγ of generic DM, and the GT is
a 2-point derived band — revisit after 1d lands).

---

## 1003.0964 — first UWA microwave-cavity LSW (tombstone, no repo file)

**What the paper reports:** hidden-photon kinetic mixing χ = 2.9e-5 peak at
m = 37.88 µeV, explicitly "within already established limits".

**Why the GT cannot grade it:** the entry's former mapping,
`DarkPhoton/LSW_UWA.txt`, belongs to arXiv:1410.5244 (per the file's own
header) — the `docs/dp.md` LSW-ADMX/LSW-UWA reference links are swapped
upstream. This paper's own limit has no repo data file, so after re-keying
LSW_UWA.txt to 1410.5244 there is nothing valid left to grade against. (The
LSW_ADMX.txt entry was likewise re-keyed to its header id 1007.3766.)

**What would un-exclude it:** digitizing this paper's own Fig. 11 sliver into
the repo (upstream science change).

---

## 2312.11608 — DM21cm forecast (tombstone for GammaRayDecayCompilation.txt)

**What the paper reports:** projected HERA 21-cm sensitivity to decaying DM
(lifetime vs mass); pre-existing gamma-ray constraints appear only as
background curves in its figure.

**Why this GT file cannot grade it:**
`AxionPhoton/GammaRayDecayCompilation.txt` is a compilation of PRE-EXISTING
limits (HEAO-1/COMPTEL/EGRET/Fermi) that O'Hare took from the paper's
*background* curves — not the paper's own result. The active 2312.11608 entry
now points at the paper's actual headline, `AxionPhoton/Projections/21cm.txt`
(`is_projection: true`).

**What would un-exclude it:** nothing — the file is genuinely not this paper's
result; the tombstone exists so the auto-expander does not silently re-add the
wrong mapping.

## 1609.00667 — NuSTAR sterile-neutrino window (AxionPhoton/NuSTAR.txt)

**What the paper reports:** sterile-neutrino dark-matter limits from the
NuSTAR X-ray line search — the active–sterile mixing angle sin²(2θ) and the
decay rate Γ versus sterile-neutrino mass (10–50 keV). It never quotes an
axion-photon coupling.

**Why the GT cannot grade it:** the repo `AxionPhoton/NuSTAR.txt` g_aγγ curve
is the maintainer's derived conversion of the same X-ray line-flux limit to
decaying-ALP dark matter; the number does not appear in the paper, so the GT
demands a quantity that is not extractable. Same class as the 2112.03439
Breakthrough Listen exclusion. The extractor's refusal (ct=None) was correct
and is reproduced independently by both the Opus (`final347_remediated`) and
Fable (`final347_fable`) arms.

**What would un-exclude it:** a vetted line-flux → g_aγγ converter applied on
both sides at scoring time (the same machinery the Breakthrough Listen entry
is waiting on).

---

## 2102.02207 — XMM-Newton blank-sky decaying DM (AxionPhoton/XMM-Newton.txt)

**What the paper reports:** decaying dark-matter limits via the
sterile-neutrino mixing angle sin²(2θ) versus m_χ (5–16 keV) from XMM-Newton
blank-sky observations. It never quotes an axion-photon coupling.

**Why the GT cannot grade it:** the repo `AxionPhoton/XMM-Newton.txt` g_aγγ
curve is the maintainer's derived conversion to decaying-ALP dark matter; the
GT demands a quantity that is not extractable from the paper. Same class as
the 2112.03439 Breakthrough Listen exclusion. The extractor's refusal
(ct=None) was correct.

**What would un-exclude it:** a vetted line-flux → g_aγγ converter applied on
both sides at scoring time.
