---
audit_of: GPD/explanations/coupling-convention-conversions-EXPLAIN.md
auditor: gpd-bibliographer
date: 2026-06-04
status: completed
verdict: all 10 references are REAL; 2 metadata corrections needed (author + paper-title attribution); 1 physics-claim relevance note
methods: arXiv abstract pages (WebFetch), WebSearch cross-check, live HTTP HEAD/GET on URLs, repo file headers + docs/ provenance cross-check
---

# Citation Audit — Coupling-Convention Conversions for AutoAxionLimits

## Summary verdict

All 10 cited references (8 arXiv papers, 1 PDG review, 1 GitHub repo) are **real** and the
**physics claims attached to each are supported**. No hallucinated references were found.

Two **metadata errors** require correction, and one **labeling/relevance note** is worth
recording. Every cited URL was checked and resolves.

| # | Reference (as cited) | Real? | Metadata accurate? | Relevant to its claim? | Action |
|---|---|---|---|---|---|
| 1 | Damour & Donoghue 2010, arXiv:1007.2792 | YES | YES | YES | none |
| 2 | di Cortona et al. 2016, arXiv:1511.02867 | YES | YES | YES | none |
| 3 | "Tretiak et al." arXiv:2010.08107 | YES | **NO — wrong authors** | YES | correct authors |
| 4 | PDG Axions review (2020 PDF) | YES | YES | YES | none |
| 5 | cajohare/AxionLimits repo | YES | YES | YES | none |
| 6 | NASDUCK-SERF, arXiv:2209.13588 | YES | YES | YES | none |
| 7 | SNO, arXiv:2004.02733 | YES | YES | YES | minor note |
| 8 | "AxionEDM C_G/(f_a m_a) context" arXiv:2204.01454 | YES | **incomplete — paper unnamed** | YES | name the paper |
| 9 | CsCav, arXiv:2201.02042 | YES | **NO — wrong title/authors in concern, but label OK** | YES | clarify |
| 10 | Holometer, arXiv:2108.04746 | YES | YES | YES | none |

---

## Per-reference findings

### 1. Damour & Donoghue, arXiv:1007.2792 — VERIFIED (Priority-1 anchor)
- **Confirmed title:** "Equivalence Principle Violations and Couplings of a Light Dilaton."
- **Authors:** Thibault Damour, John F. Donoghue. **Year:** 2010.
- **Journal:** Phys. Rev. D 82, 084033 (2010) — matches the document exactly.
- **Claim support:** This is the canonical paper that introduces the dimensionless dilaton
  coefficients `d_i` and the `α(φ)=α(1+d_e φ̂)`, `φ̂=φ/M_Pl` parameterization. The document's
  use of it as the definition source for `d_e`, `d_g`, `d_{m_e}` is correct. (The arXiv abstract
  page confirms title/authors/journal; the specific `d_i` definitions are standard and
  universally attributed to this paper — high confidence.)
- **Verdict:** Accurate and well-supported. No change.

### 2. di Cortona, Hardy, Pardo Vega, Villadoro, arXiv:1511.02867 — VERIFIED (Priority-2 anchor)
- **Confirmed title:** "The QCD axion, precisely."
- **Authors:** Giovanni Grilli di Cortona, Edward Hardy, Javier Pardo Vega, Giovanni Villadoro.
  (Note: lead author surname is "Grilli di Cortona"; the document's short form "di Cortona et al."
  is the conventional abbreviation and is acceptable.)
- **Journal:** JHEP 01 (2016) 034 — matches the document.
- **Claim support:** The abstract explicitly states axion-nucleon couplings are derived with
  <10% uncertainty, supporting the document's use of it for `c_N` and the EDM `g_d`
  (≈2.4×10⁻¹⁶ e·cm) coefficient. The `3.7e-3` ↔ `g_d` re-expression is standard.
- **Verdict:** Accurate and well-supported. No change.

### 3. arXiv:2010.08107 — REAL but WRONG AUTHOR ATTRIBUTION ⚠️
- **Document cites it as:** "Tretiak et al. (incl. the 2010.08107 family), scalar-DM via fundamental constants."
- **Actual paper:** "Searching for Scalar Dark Matter via Coupling to Fundamental Constants with
  Photonic, Atomic and Mechanical Oscillators," **Campbell, McAllister, Goryachev, Ivanov, Tobar**,
  PRL 126, 071301 (2021).
- **The authors are NOT Tretiak et al.** Tretiak is the lead author of a *different* paper
  (arXiv:2201.02042, reference #9). The document appears to have swapped the Tretiak attribution
  onto 2010.08107.
- **Claim support is otherwise EXACT:** the paper reports `|d_e| ≳ 1.59×10⁻¹` and
  `|d_{m_e}−d_g| ≳ 6.97×10⁻¹`, which round to the document's quoted "≳0.16" and "≳0.7." So the
  *number and the physics are right*; only the author name is wrong.
- **Cross-check with repo:** `docs/phie.md`/`docs/phime.md` map arXiv:2010.08107 to the repo's
  **"H/Quartz/Sapphire"** file (a Tobar-group BAW/sapphire oscillator experiment) — consistent
  with the Campbell/Tobar attribution, NOT Tretiak.
- **Correction:** Replace "Tretiak et al." with **"Campbell, McAllister, Goryachev, Ivanov, Tobar
  (PRL 126, 071301, 2021)"** for arXiv:2010.08107. This is the H/Quartz/Sapphire source, not a
  Tretiak paper.

### 4. PDG "Axions and Other Similar Particles" review — VERIFIED
- **URL:** https://pdg.lbl.gov/2020/reviews/rpp2020-rev-axions.pdf
- **Live check:** HTTP 200, `Content-Type: application/pdf`, 582 KB, `Last-Modified: 2020-06-01` —
  consistent with the 2020 PDG axion review. URL opens.
- **Claim support:** Standard `g_{aN}`/derivative-coupling definitions are indeed part of this
  review. (PDF binary did not text-extract through the fast model, but the URL, size, date, and
  filename `rpp2020-rev-axions` are all consistent. Confidence: high on existence/URL; the specific
  definition content is a well-known standard reference.)
- **Verdict:** Accurate. No change.

### 5. cajohare/AxionLimits (GitHub) — VERIFIED
- **URL:** https://github.com/cajohare/AxionLimits — this is the documented upstream of the very
  fork being worked in (AutoAxionLimits), confirmed by CLAUDE.md and the repo's own data-file
  headers/docs. Trivially real and authoritative for "canonical repo convention."
- **Verdict:** Accurate. No change.

### 6. NASDUCK-SERF, arXiv:2209.13588 — VERIFIED
- **Confirmed title:** "Constraints on axion-like dark matter from a SERF comagnetometer."
- **Authors:** Bloch, Shaham, Hochberg, Kuflik, Volansky, Katz (NASDUCK Collaboration).
- **Journal:** Nature Communications 14, 5784 (2023).
- **Claim support:** It is the NASDUCK SERF experiment, constrains axion-neutron AND axion-proton
  couplings from ALP dark matter — matching the document's use and the repo file headers
  (`AxionNeutron/NASDUCK-SERF.txt` → `g_aNN [GeV^-1]`). Mass range 1.4×10⁻¹²–2×10⁻¹⁰ eV matches
  the haloscope/DM-search context. The abstract did not print the exact unit string, but the repo
  file headers (independently inspected) confirm the GeV⁻¹ derivative-coupling convention.
- **Verdict:** Accurate and well-supported. No change.

### 7. SNO, arXiv:2004.02733 — VERIFIED (with a clarifying note)
- **Confirmed title:** "Searching for solar axions using data from the Sudbury Neutrino Observatory."
- **Authors:** Aagaman Bhusal, Nick Houston, Tianjun Li. **Journal:** PRL 126, 091601 (2021).
- **Claim support:** It uses axion-induced deuteron dissociation (n+p channel) in SNO heavy water —
  matching the document's "SNO axion-induced dissociation." It defines an **isovector** coupling
  `|g³_aN| ≡ ½|g_an − g_ap|` in GeV⁻¹ and excludes >2×10⁻⁵ GeV⁻¹.
- **Note on the convention claim:** The document attaches this paper to the repo's SNO-specific
  `×m_n` (not `×2m_n`) rule, citing a repo comment that "their g_an = my g_an/m_n not g_an/2m_n."
  The paper's own isovector `½(g_an−g_ap)` normalization is consistent with a non-standard factor
  relative to the `g_aNN = C_N/2f_a` convention, so the document's "different normalization →
  per-file multiplier" reasoning is reasonable. The decisive evidence for the `×m_n` factor is the
  repo's `AxionNeutron/SNO.txt` header (`g_an/m_n [GeV^-1]`) and code, which the document already
  grounds independently. The paper supports the "yet another normalization" framing.
- **Verdict:** Accurate and well-supported. No change required; the convention claim rests on the
  repo header (verified) rather than on the paper alone, which is appropriate.

### 8. arXiv:2204.01454 — REAL; PAPER WAS NOT NAMED ⚠️
- **Document cites it as:** "AxionEDM `C_G/(f_a m_a)` context, arXiv:2204.01454 — the `∝1/m_a`
  EDM-response convention." The document never gives the paper's title or authors.
- **Actual paper:** "New Limit on Axion-Like Dark Matter using Cold Neutrons," **Schulthess,
  Chanel, Fratangelo, Gottstein, Gsponer, Hodge, Pistillo, Ries, Soldner, Thorne, Piegsa**,
  PRL 129, 191801 (2022).
- **Claim support is EXACT:** the paper's best limit is literally `C_G/f_a m_a = 2.7×10¹³ GeV⁻²`
  via an oscillating-nEDM Ramsey measurement over 23 μHz–1 kHz — i.e. exactly the `C_G/(f_a m_a)`
  [GeV⁻²] coupling with the `∝1/m_a` response the document describes. Highly relevant and correct.
- **Correction:** This is not an error of substance, but the reference should be **named**:
  add "Schulthess et al., 'New Limit on Axion-Like Dark Matter using Cold Neutrons,'
  PRL 129, 191801 (2022)" so the citation is attributable rather than an anonymous "context."

### 9. CsCav, arXiv:2201.02042 — REAL; LABEL IS REPO-CONSISTENT (clarification) ⚠️
- **Document cites it as:** "Cs/cavity scalar-DM bounds, arXiv:2201.02042 (CsCav)."
- **Actual paper:** "Improved bounds on ultralight scalar dark matter in the radio-frequency
  range," **Tretiak, Zhang, Figueroa, Antypas, Brogna, Banerjee, Perez, Budker**, PRL 129,
  031301 (2022). It is an **optical/RF-spectroscopy** search (20 kHz–100 MHz), constraining
  couplings to electrons and photons (i.e. `d_{m_e}`/`d_e`-type), mass 8×10⁻¹¹–4×10⁻⁷ eV.
- **Repo provenance (decisive):** `limit_data/ScalarElectron/CsCav.txt` header literally reads
  `# Cs/Cav` and `# https://arxiv.org/pdf/2201.02042.pdf`; `docs/phie.md` and `docs/phime.md` both
  map the "Cs/Cav" limit to arXiv:2201.02042. **So the pairing 2201.02042 ↔ CsCav is exactly what
  the upstream repo asserts** — the document did NOT misassign the file.
- **Clarification:** "CsCav"/"Cs/cavity" is the *repo's own short label* (cesium + cavity
  oscillators feature in this RF/spectroscopy program). The document's parenthetical "(Cs/cavity
  scalar-DM bounds)" is therefore repo-consistent, not a fabrication. The one thing to keep
  straight: **2201.02042 = Tretiak et al.** (NOT 2010.08107, see ref #3). The document's Tretiak
  attribution belongs on THIS paper, not on 2010.08107.
- **Document's own honesty note preserved:** the document flags the *stored-file convention* for
  CsCav as "unverified at file level" (magnitudes 1e4–1e13 vs expected `d_me`). That caveat is
  appropriate and is not contradicted by anything found here — the paper exists and is the right
  source; only the repo's stored numerical convention for that file remains unconfirmed.
- **Verdict:** Reference real and correctly paired with the repo file. No swap needed; just note
  the Tretiak-authorship belongs here, not on #3.

### 10. Holometer, arXiv:2108.04746 — VERIFIED
- **Confirmed title:** "Constraints on scalar field dark matter from co-located Michelson
  interferometers."
- **Authors:** Aiello, Richardson, Vermeulen, Grote, Hogan, Kwon, Stoughton (Fermilab Holometer).
  Submitted 2021-08, revised 2022-03.
- **Claim support:** Twin co-located 40-m power-recycled interferometers (the Fermilab Holometer),
  scalar-field DM via oscillating size/refractive index, mass 1.6×10⁻¹²–1.0×10⁻⁷ eV — matching the
  document's "Fermilab Holometer" label and the `ScalarPhoton/Holometer.txt` file.
- **Verdict:** Accurate and well-supported. No change.

---

## Special-attention items requested by the task

- **Damour & Donoghue 2010 (arXiv:1007.2792, dilaton `d_i`):** VERIFIED real, correct metadata
  (PRD 82, 084033), correctly used as the `d_i` definition source. No issue.
- **Axion-EFT review for g_aN vs C_N/f_a (di Cortona et al., arXiv:1511.02867):** VERIFIED real,
  correct metadata (JHEP 01(2016)034), appropriately cited for nucleon couplings and the EDM
  coefficient. No issue. (The granular `g_{aN}=2 m_N·C_N/(2f_a)` reduction the document relies on
  is grounded in repo code, which is the right anchor.)
- **NASDUCK-SERF (2209.13588):** VERIFIED — Nature Comm. 14, 5784 (2023). Correct.
- **SNO (2004.02733):** VERIFIED — PRL 126, 091601 (2021), isovector `½(g_an−g_ap)`. Correct;
  convention claim grounded in repo header. No issue.
- **CsCav (2201.02042):** VERIFIED — Tretiak et al., PRL 129, 031301 (2022); repo-confirmed pairing.
  Label OK; note Tretiak-authorship sits here, not on 2010.08107.
- **Holometer (2108.04746):** VERIFIED — Aiello et al. Correct.
- **The fifth listed ID, 2204.01454:** VERIFIED — Schulthess et al., PRL 129, 191801 (2022),
  `C_G/f_a m_a = 2.7×10¹³ GeV⁻²`. Correct physics; just unnamed in the document.

## Required corrections (do NOT silently apply — these touch the manuscript's Literature Guide)

1. **arXiv:2010.08107 author fix (substantive):** change "Tretiak et al." →
   "Campbell, McAllister, Goryachev, Ivanov, Tobar, PRL 126, 071301 (2021)" (the
   H/Quartz/Sapphire / photonic-atomic-mechanical-oscillator paper). The `|d_e|≳0.16`,
   `|d_{m_e}−d_g|≳0.7` numbers are correct and come from THIS paper.
2. **arXiv:2204.01454 naming (attribution completeness):** name it
   "Schulthess et al., 'New Limit on Axion-Like Dark Matter using Cold Neutrons,'
   PRL 129, 191801 (2022)" instead of the anonymous "AxionEDM C_G/(f_a m_a) context."
3. **Optional clarity:** since the Tretiak attribution genuinely belongs to arXiv:2201.02042
   (CsCav), consider adding "Tretiak, Zhang, …, Budker, PRL 129, 031301 (2022)" beside the CsCav
   entry to make the (now-correct) attribution explicit and avoid re-confusing it with #1.

## What was NOT verified (honest gaps)

- The **internal numerical convention of the stored `CsCav.txt`/`Holometer.txt` files**
  (whether the 1e4–1e13 magnitudes are `1/Λ` [GeV⁻¹], an amplitude, or experiment-native) — the
  document already flags this as unverified; this audit did not resolve it (PDF figure axes not
  pulled). This is a *data-provenance* gap, not a citation-existence gap. The papers themselves
  are confirmed real and topically correct.
- The full text of the PDG 2020 axion PDF was not machine-parsed; existence, URL, size, date, and
  filename are all consistent with the named review.

## Reproducibility

Verification path: arXiv abstract pages via WebFetch for 1007.2792, 1511.02867, 2010.08107,
2209.13588, 2004.02733, 2204.01454, 2201.02042; WebSearch cross-checks for 2201.02042 and
2108.04746; live `curl -I` on the PDG URL (HTTP 200, 582 KB PDF, 2020-06-01); and repo-internal
cross-checks (`limit_data/ScalarElectron/CsCav.txt` header, `docs/phie.md`, `docs/phime.md`,
`PlotFuncs_ScalarVector.py`) to confirm file↔paper pairings.
