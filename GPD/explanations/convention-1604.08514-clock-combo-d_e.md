---
concept: Convention token 1604.08514 — clock sensitivity-coefficient combination "d_e + 0.043(d_m̂ − d_g)" vs canonical ScalarPhoton d_e
date: 2026-07-14
mode: project-context
project_context: AutoAxionLimits convention-escalation queue (pipeline/DESIGN_convention_escalation.md Phase 3 drain); target GT file limit_data/ScalarPhoton/RbCs.txt
citation_status: declaring paper verified at PDF level this session (Eqs. 3, 4, 8, Fig. 3 caption, k-coefficients, limit values read directly from arXiv:1604.08514v3); PRL 117, 061301 (2016) journal ref quoted from memory, unverified
---

# Convention Note: 1604.08514 (SYRTE Rb/Cs) — "d_e + 0.043(d_m̂ − d_g)" → canonical d_e

**Verdict: identity conversion (numeric factor 1) under the compilation's one-coupling-dominance
convention. Registry action: VOCABULARY promotion only — no converter added.**

## The declared token

Production extraction of arXiv 1604.08514 (Hees, Guéna, Abgrall, Bize, Wolf — SYRTE dual Rb/Cs
fountain, oscillating scalar DM) declared:

> "log10 of d_e + 0.043(d_m̂ - d_g); emitted as absolute coupling value (10^y)"

i.e. the y-axis of the paper's Fig. 3 is log₁₀ of the combination, and the extractor already
exponentiated — the emitted values are absolute (dimensionless) combination values, ~5e-8 at the
low-mass edge. Coupling type ScalarPhoton; canonical variable dimensionless d_e
(Damour–Donoghue normalization). GT file `limit_data/ScalarPhoton/RbCs.txt` (header: `mass [eV]  d_e`)
is plotted RAW by `PlotFuncs_ScalarVector.py::RbCs` (line 90: `loadtxt` → `FilledLimit`, no rescale).

## 1. Closed-form conversion and where 0.043 comes from (verified against the PDF)

**Framework.** The paper uses the Damour–Donoghue linear dilaton couplings verbatim: its Eq. (3)
is Eq. (12) of Damour–Donoghue, L_int = ϕ[d_e/(4µ₀)F² − (d_g β_g/2g₃)(F^A)² − Σ_i(d_mi+γ_mi d_g)m_i ψ̄ψ],
with five dimensionless coefficients d_e, d_me, d_mu, d_md, d_g; its Eq. (4) gives
α(ϕ) = α(1 + d_e ϕ), m_i(ϕ) = m_i(1 + d_mi ϕ), Λ₃(ϕ) = Λ₃(1 + d_g ϕ), and the mean-quark-mass
combination d_m̂ = (d_mu m_u + d_md m_d)/(m_u + m_d). Exactly the compilation's convention.

**Why a combination appears.** A ratio X of two atomic transition frequencies varies as
d ln X = k_α d ln α + k_µ d ln(m_e/m_p) + k_q d ln(m_q/Λ₃) (paper, ref [52] therein). Hyperfine
frequencies depend on the nuclear magnetic moment, hence on the quark-mass/QCD-scale ratio
m_q/Λ₃ — that is the physical origin of the k_q term; optical/electronic transitions have only
k_α ≠ 0. For the Rb/Cs ground-state hyperfine *ratio*, atomic-structure calculations (paper
refs [53–56]) give, verbatim from p.2–3:

  k_α = −0.49,  k_µ = 0,  k_q = −0.021.

(k_µ = 0 because the m_e/m_p dependence cancels between two hyperfine transitions — this is why
d_me never enters.) The paper's Eq. (8) then gives the oscillation amplitude of the ratio:

  A = [k_α d_e + k_q (d_m̂ − d_g)] ϕ₀ = [k_α d_e + k_q (d_m̂ − d_g)] (1/ω) √(8πG ρ_DM/c²),

with ρ_DM ≈ 0.4 GeV/cm³ and m_ϕ = ħω/c². Dividing through by k_α normalizes the d_e coefficient
to 1:

  **d_e + (k_q/k_α)(d_m̂ − d_g) = d_e + (0.021/0.49)(d_m̂ − d_g) = d_e + 0.0429(d_m̂ − d_g).**

The paper itself rounds 0.0429 → **0.043** — verbatim in the text ("limits on
d_e + k_q/k_α (d_m̂ − d_g) = d_e + 0.043(d_m̂ − d_g)") and in the Fig. 3 caption/axis
(log₁₀[d_e + (k_q/k_α)(d_m̂ − d_g)]). So the declared token reproduces the paper's own published
y-variable exactly.

**Why dominance makes it the identity.** The AxionLimits compilation plots every clock/EP bound
on the d_e plane under the standard one-coupling-dominance convention: all dilaton couplings
except the plotted one are set to zero (the same convention the paper invokes for its Dy
comparison, "if we assume that the scalar field is coupled only to electromagnetism (only
d_e ≠ 0)"). Setting d_m̂ = d_g = 0:

  d_e + 0.043(d_m̂ − d_g) → **d_e, identically. Conversion factor: 1. No constants enter.**

Status labels: the k-coefficients and Eq. (8) are established results of the paper; the
dominance reduction is a *project/compilation convention* (shared by the GT file, so both sides
carry the same assumption — the comparison is convention-consistent, not convention-free).

## 2. Dimensional check

All d_i are dimensionless by construction (Damour–Donoghue; the paper's footnote 1: ϕ is the
dimensionless field, ϕ = √(4πG/cħ) φ = √4π φ/M_Pl). k_q/k_α is a ratio of dimensionless
sensitivity coefficients. So declared side (10^y, dimensionless combination) → canonical side
(dimensionless d_e): dimensionless → dimensionless ✓. Identity is dimensionally trivial, as it
must be.

## 3. Numeric spot-check — same plane, same regime

GT file (505 rows, read this session): mass 4.302e-25 … 2.308e-18 eV, d_e 3.803e-09 … 4.306e-04.

**Anchor check (paper text ↔ GT file):** the paper quotes "most stringent limit … 3.8×10⁻⁹ at
m_ϕ = 1.4×10⁻²³ eV/c²" and "exclude couplings larger than 5.3×10⁻⁴ at any m_ϕ". The GT minimum
is 3.8032e-9 at 1.326e-23 eV — **0.00 dex** on the floor; file max 4.31e-4 vs quoted 5.3e-4
ceiling, 0.09 dex. The repo file is manifestly a raw digitization of Fig. 3's red 95%-CL curve.

**Extraction sample points vs GT (nearest-in-log-mass):**

| extraction (m [eV], value) | GT nearest (m, d_e) | residual |
|---|---|---|
| (3.16e-25, 5.0e-08) | (4.30e-25, 7.91e-08) — file edge | 0.20 dex (extraction sits just below GT's low-mass edge) |
| (5.60e-25, 3.2e-08) | (5.76e-25, 5.92e-08) | 0.27 dex |
| (1.00e-24, 1.6e-08) | (1.03e-24, 3.30e-08) | 0.31 dex |

Residuals of 0.2–0.3 dex on a spiky amplitude-spectrum curve (Fig. 3's blue/red traces oscillate
by ~0.5 dex point-to-point) are **vision-tracing/digitization noise, not a units-class gap** —
the two curves live on the same plane at the same scale, and the file floor sits within a factor
~2 of the sample values in the same decade band (~1e-8 vs 3.8e-9). Contrast with what a genuine
wrong-plane declaration looks like in this project: the AxionEDM e·cm cluster (round-2 note,
Family 5) produced ~6–15 dex gaps (1401.6460: 6.19 dex from one wrong unit constant;
raw-Γ-vs-g_aγγ misreads: 13 dex). Nothing of that kind is present here. Identity conversion is
confirmed by the data, not just the algebra.

## Registry consequence — VOCABULARY promotion, with a scope guard

**Action:** promote the declared-string family "clock sensitivity-coefficient combination
d_e + c(d_m̂ − d_g), |c| < 1" to *recognized-as-canonical d_e* for ScalarPhoton. No numeric
converter is added to `to_canonical`; `classify_reported_convention` should return the canonical
token (equivalently None/no-op) for such declarations instead of flagging an unknown convention.
GT side unchanged (`RbCs.txt` already canonical, plotted raw).

**Guard (must be enforced):**

- Scope to **ScalarPhoton clock-comparison declarations in which d_e appears with unit
  coefficient and the non-d_e terms carry a small coefficient (|c| < 1, here 0.043)**. The small
  coefficient is what makes the dominance reduction faithful — a d_e-dominated combination is
  numerically the d_e bound to within ~4% even for O(equal) couplings.
- It must **NOT generalize** to combinations with no d_e term — e.g. a declaration of
  |d_m̂ − d_g| alone (the plane constrained by pure-microwave/QCD-sensitive comparisons) is a
  *different plane*; mapping it to d_e would be a silent physics substitution, exactly the class
  of error the foreign-quantity screen (#683) fails CLOSED on. Match on the presence of the
  d_e (or "de") leading term, never on substring "d_m̂"/"d_g" alone.
- Per the truthful-declaration contract (#594): the token describes the emitted values (absolute
  10^y, dimensionless). A future declaration saying "converted to d_e" stays canonical-claimed;
  this recognition adds no re-conversion either way — consistent because the map is the identity.

## Literature

- **Hees, Guéna, Abgrall, Bize, Wolf**, "Searching for an oscillating massive scalar field as a
  dark matter candidate using atomic hyperfine frequency comparisons" — the declaring paper;
  Eqs. (3), (4), (8), Fig. 3, k-coefficients all verified from the PDF this session.
  https://arxiv.org/abs/1604.08514 (PRL 117, 061301 (2016) — journal ref unverified from memory)
- **Damour & Donoghue**, "Equivalence Principle Violations and Couplings of a Light Dilaton,"
  PRD 82, 084033 (2010) — the d_i normalization (round-1 audited anchor). https://arxiv.org/abs/1007.2792
- **Van Tilburg, Leefer, Bougas, Budker** (Dy spectroscopy) — the pure-d_e comparison bound the
  paper benchmarks against. https://arxiv.org/abs/1503.06886

**PASS** — closed form (identity) derived from the paper's own Eq. (8) and k-coefficients,
dimension-checked, and numerically confirmed on the GT floor (0.00 dex) and sample points
(≤0.31 dex, tracing noise).

## Citation audit

*Auditor: gpd-bibliographer, 2026-07-14. Methods: arXiv export API (metadata for 1604.08514,
1007.2792, 1503.06886), Crossref resolution of DOI 10.1103/PhysRevLett.117.061301, and
claim-content re-location in the cached PDF `~/.cache/aal_pdf_cache/1604.08514.pdf` (v3) for
every verbatim quote. Verdict: **PASS** — 0 hallucinations, 0 wrong attributions; the one
from-memory journal ref resolves correct.*

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | arXiv 1604.08514 = Hees, Guéna, Abgrall, Bize, Wolf, "Searching for an oscillating massive scalar field as a dark matter candidate using atomic hyperfine frequency comparisons"; journal ref PRL 117, 061301 (2016) | **PASS — unverified flag resolved, ref CORRECT** | arXiv journal_ref `Phys. Rev. Lett. 117, 061301 (2016)`, DOI 10.1103/PhysRevLett.117.061301; Crossref confirms PRL vol. 117, article 061301, published 2016-08-05, same 5 authors. The `citation_status`/Literature "unverified from memory" flag can be replaced with "verified 2026-07-14". |
| 2 | Damour & Donoghue, "Equivalence Principle Violations and Couplings of a Light Dilaton," PRD 82, 084033 (2010), arXiv 1007.2792 — source of the paper's Eq. (3) | **PASS** | arXiv metadata exact (journal_ref `Phys.Rev.D82:084033,2010`, DOI 10.1103/PhysRevD.82.084033). The Eq.-number bridge is the declaring paper's own verbatim statement: "The interacting part of the Lagrangian Lint is given by Eq. (12) of [40]", with bibliography entry [40] = "T. Damour and J. F. Donoghue, Phys. Rev. D 82, 084033 (2010), arXiv:1007.2792" — the note's "its Eq. (3) is Eq. (12) of Damour–Donoghue" is the paper's claim, correctly attributed. |
| 3 | k_α = −0.49, k_µ = 0, k_q = −0.021 for the Rb/Cs hyperfine ratio, attributed to refs [52]–[56] therein | **PASS — verbatim** | PDF: "Recent atomic structure calculations have shown that for the Rb/Cs ground state hyperfine transitions kα = −0.49, kµ = 0 and kq = −0.021 [53–56]"; the d ln X sensitivity relation carries "[52]" = V. V. Flambaum, D. B. Leinweber, A. W. Thomas, R. D. Young, Phys. Rev. D 69, 115006 (2004), hep-ph/0402098 — i.e. the Flambaum-et-al. sensitivity-coefficient literature, exactly as the note attributes. |
| 4 | Numeric constants and quotes: 0.043 (text + Fig. 3 caption), Eq. (8) amplitude form with ρ_DM ≈ 0.4 GeV/cm³, 3.8×10⁻⁹ at m_ϕ = 1.4×10⁻²³ eV/c², 5.3×10⁻⁴ ceiling, Eq. (4a–4c) + d_m̂ = (d_mu m_u + d_md m_d)/(m_u+m_d), footnote-1 ϕ = √(4πG/cℏ)φ, "coupled only to electromagnetism (only de ≠ 0)" | **PASS — all re-located verbatim in the cached PDF** | Every quoted string/number found; equation numbers (3), (4a–c), (8) match the PDF numbering; "de + 0.043(d ˆm −dg)" appears in both the text and the Fig. 3 caption as claimed. GT-file values are repo-internal (out of citation scope). |
| 5 | Van Tilburg, Leefer, Bougas, Budker (Dy spectroscopy), arXiv 1503.06886 — the pure-d_e benchmark | **PASS** | arXiv metadata exact ("Search for ultralight scalar dark matter with atomic spectroscopy"); the declaring paper's [2] is verbatim this reference, and its "improve those of [2] … only de ≠ 0" passage is the benchmarking the note describes. Optional completion (not an error — the note makes no journal claim): PRL 115, 011802 (2015), DOI 10.1103/PhysRevLett.115.011802. |
