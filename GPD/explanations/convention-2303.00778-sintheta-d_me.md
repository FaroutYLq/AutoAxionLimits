---
concept: Convention token "sin theta (Higgs-portal scalar-Higgs mixing angle)" -> canonical d_me (ScalarElectron), paper 2303.00778
date: 2026-07-14
mode: project-context
project_context: AutoAxionLimits convention-escalation queue (pipeline/DESIGN_convention_escalation.md Phase 3 drain); companion to GPD/explanations/coupling-convention-conversions-round2-EXPLAIN.md
citation_status: paper title/authors/abstract verified via arXiv this session; repo anchors (WhiteDwarfs.txt, RedGiants.txt, PlotFuncs_ScalarVector.py:233, evaluation/conventions.py) read from the working tree; Damour–Donoghue normalization carried over from the round-1/round-2 audited notes; journal refs from memory are flagged unverified
---

# Convention Derivation: sin θ (Higgs portal) → d_me — arXiv 2303.00778

**Verdict: PASS.** Closed-form, mass-independent, constant multiplier:

  **d_me = C · sin θ,  C = √2·M_Pl/v = 1.379×10¹⁶** (repo constants: `_M_PL_GEV = 2.4e18`, v = 246.22 GeV).

Spot-check residual vs the GT file: **+0.0067 dex** (exact-κ variant +0.0130 dex). The √2 is resolved from
first principles below — it comes from the Damour–Donoghue κ, *not* from the Yukawa convention.

## The Paper (verified)

**Bottaro, Caputo, Raffelt, Vitagliano**, "Stellar limits on scalars from electron-nucleus
bremsstrahlung," https://arxiv.org/abs/2303.00778 (JCAP, unverified volume). White-dwarf luminosity
function limits on light scalars via e–nucleus bremsstrahlung; quotes g_L ≲ 4×10⁻¹⁶ (leptophilic
Yukawa to electrons) and, for the Higgs portal, **sin θ ≲ 2×10⁻¹⁰**, valid for m_φ ≲ 1 keV (flat in
mass — consistent with both the flat extraction and the flat GT file).

## Token and Target

- Declared (verbatim from production extraction): `"sin theta (Higgs-portal scalar-Higgs mixing angle)"`
- Extraction sample points (paper-native): (1e-3 eV, 1.9e-10), (1e3 eV, 1.9e-10) — flat
- Canonical variable (ScalarElectron): dimensionless **d_me** (`_CANONICAL_TOKEN`, conventions.py:262)
- GT: `limit_data/ScalarElectron/WhiteDwarfs.txt` — header `# m_phi d_me`, flat **d_me = 2.5793e6**
  over 1e-30…1e2 eV; plotted RAW by `PlotFuncs_ScalarVector.py::WhiteDwarfs` (line 233, no rescale)

## Derivation of C (both parametrizations written out)

**Parametrization A — Damour–Donoghue canonical (round-1/round-2 vetted).** The dilaton-type
electron-mass coupling is defined by

  L ⊃ −κ · d_me · m_e · φ ē e,  κ = √(4πG_N) = 1/(√2·M_Pl,red),  1/κ = 3.444×10¹⁸ GeV,

(QSNET Eq. 1 wording, verified in round 2, Family 6; identical κ to the vetted `gev_inv_scalar`
branch). So the physical scalar–electron Yukawa magnitude is

  g_e = κ · d_me · m_e = d_me · m_e / (√2·M_Pl,red).      (A)

Equivalently g_e = d_me·√(4π)·m_e/M_Pl,full (M_Pl,full = 1.221e19 GeV; √(4π)/M_Pl,full = κ) — the
same statement, since M_Pl,full/√(4π) = √2 × M_Pl,red = 3.444e18 GeV. The repo convention uses the
**reduced** Planck mass at the notebook value 2.4e18 (`_M_PL_GEV`, conventions.py:203).

**Parametrization B — Higgs-portal mixing.** After mass mixing with angle θ, the light scalar
inherits the SM Higgs couplings scaled by sin θ. The SM h-ē-e vertex is
L = −(y_e/√2) h ē e with m_e = y_e v/√2, i.e. the physical vertex is **m_e/v** — the √2 in
y_e = √2·m_e/v cancels identically and contributes nothing to K. Hence

  g_e = sin θ · m_e / v,  v = 246.22 GeV.      (B)

The paper's own numbers confirm (B) with no stray √2: sin θ = 2e-10 × (m_e/v = 2.0755e-6)
= 4.15e-16 ≈ its quoted g_L ≲ 4e-16. ✓

**Equate (A) = (B):**

  d_me · m_e/(√2·M_Pl) = sin θ · m_e/v  ⇒  **d_me = sin θ · √2·M_Pl/v**,  so K = √2, sourced
entirely from κ = 1/(√2·M_Pl,red) — not fit to the data, and not a Yukawa-convention artifact.

  C = √2 × 2.4e18 / 246.22 = **1.3785×10¹⁶** (repo constants)
  C = 3.444e18 / 246.22 = 1.3987×10¹⁶ (exact κ; 0.006 dex above repo — keep the repo value for
  consistency with the existing `gev_inv_scalar` branch, same policy as round 2)

## Dimensional Check

sin θ dimensionless; √2·M_Pl/v = [GeV]/[GeV] dimensionless ⇒ d_me dimensionless ✓ (matches the
canonical dimensionless d_me axis; and (A), (B) each give g_e dimensionless ✓).

## Numeric Spot-Check (verified this session)

| quantity | value |
|---|---|
| extraction sin θ = 1.9e-10 × C_repo | d_me = 2.6191e6 |
| GT `WhiteDwarfs.txt` | d_me = 2.5793e6 |
| **residual** | **+0.0067 dex** (+0.0130 dex with exact κ) |

Back-conversions close the loop: the GT file implies g_e = 3.88e-16 (paper: g_L ≲ 4e-16 ✓) and
sin θ = 1.871e-10 (paper: ≲ 2e-10; extraction read 1.9e-10 ✓). Independent family cross-check:
`RedGiants.txt` (header `d_me`, 4.709e6) back-converts through the SAME (A) to g_e = 7.09e-16 —
the red-giant bound of its source (1611.05852) — confirming these stellar ScalarElectron files
store plain canonical d_me built with √2·M_Pl,red ≈ 2.4e18.

## Machine-Encodable Entry

| token | coupling_type | formula | constant | applicability guard | confidence |
|---|---|---|---|---|---|
| `sin_theta_higgs_portal` | ScalarElectron | `y' = C·y` | `C = sqrt(2)*_M_PL_GEV/246.22 = 1.3785e16` | declared contains `sin` + (`theta`/`θ`) or `mixing angle`, plus `higgs`/`portal`; values y < 1e-3 (a mixing angle, refuses already-converted d_me ≫ 1); does NOT contain `converted` (truthful-declaration contract #594) | High (0.007 dex; K derived, not fit) |

**Model lock (important):** this map exists only because the Higgs portal ties g_e to sin θ via the
SM electron Yukawa. Do **not** reuse it for ScalarPhoton (the mixed scalar's photon coupling goes
through loop functions, not sin θ·m_e/v), nor for a generic scalar whose paper happens to bound a
"mixing angle" with non-Higgs mixing. Mass-independent and ρ_DM-free (stellar bound), so it sits
cleanly in `to_canonical`; validity window m_φ ≲ 1 keV is automatically respected because the
curve itself ends at 1e2 eV.

## GT Label Flag (informational, not part of the conversion)

`papers.json` (entry at line ~9142) labels this file `coupling_convention: "d_e_large"` — assigned
by the magnitude heuristic in `infer_convention` (conventions.py:138, fires on ymax > 1e3). The
derivation shows the file values ARE plain canonical d_me: the header says `d_me`, the plot method
consumes them raw on the d_me axis, and 2.58e6 back-converts to the paper's own g_L ≈ 4e-16 (large
d_me is physically right — stellar g_e bounds sit ~22 dex above Planck-suppressed strength, and the
sibling RedGiants.txt shows the same scale). **The `d_e_large` label is likely misassigned here**
(a "canonical-but-large" case the >1e3 heuristic cannot distinguish); with the converter above,
extraction-side sin θ → d_me lands on the file directly, so the GT side should be treated as
canonical (return None), not excluded.

## Common Confusions

- **Where the √2 lives.** It is κ = 1/(√2·M_Pl,red), i.e. the reduced-vs-1/κ Planck-mass bookkeeping.
  The Yukawa √2 (y_e = √2·m_e/v) cancels in the physical vertex m_e/v — do not apply both.
- **Which Planck mass.** Repo convention: reduced M_Pl at 2.4e18 (notebook value). Using full
  M_Pl = 1.221e19 without the √(4π) would inflate C by 0.55 dex.
- **sin θ vs g_L declarations.** A future extraction of this paper may emit g_L (Yukawa) instead:
  that needs the *different* vetted map d_me = g_e·√2·M_Pl/m_e (= g_e × 6.64e21), not C above.
  Key strictly on the mixing-angle wording.
- **Direction guard.** sin θ ≪ 1, d_me ≫ 1 here; the y < 1e-3 guard prevents double conversion of a
  canonical-claiming or already-converted trace.

## Literature

- Bottaro, Caputo, Raffelt, Vitagliano — the paper resolved here. https://arxiv.org/abs/2303.00778
- Damour & Donoghue, PRD 82, 084033 (2010) — d_i/κ normalization (round-1 audited). https://arxiv.org/abs/1007.2792
- Hardy & Lasenby — red-giant/stellar g_e bounds; source of the RedGiants.txt cross-check. https://arxiv.org/abs/1611.05852

## Follow-ups

1. Sweep the other `d_e_large`-labeled ScalarElectron/ScalarPhoton GT entries for the same
   canonical-but-large misassignment (RedGiants.txt is already implicated by the cross-check).
2. Add the reciprocal-safety unit test mirroring round 2's Family-6 guard: `sin_theta_higgs_portal`
   must refuse values > 1e-3.

## Citation audit

**Auditor:** gpd-bibliographer, 2026-07-14. **Verdict: PASS — all 4 items.** No hallucinated or
wrong-paper citations. Methods: arXiv export API (metadata for 2303.00778, 1007.2792, 1611.05852);
claim-content verification against cached PDFs `~/.cache/aal_pdf_cache/{2303.00778,1611.05852}.pdf`
and a fresh download of 1007.2792; repo anchors re-read from the working tree.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | arXiv 2303.00778 exists; Bottaro, Caputo, Raffelt, Vitagliano; "Stellar limits on scalars from electron-nucleus bremsstrahlung"; quoted bounds real | **PASS** | arXiv metadata exact (all 4 authors, title verbatim). Abstract states g_L ≲ 4×10⁻¹⁶ (WD luminosity function), sin θ ≲ 2×10⁻¹⁰ (Higgs portal), valid m_φ ≲ 1 keV — all three exactly as the note quotes. In-paper Eq. (4.12) gives the precise value **sin θ ≲ 1.9×10⁻¹⁰** (abstract rounds to 2×10⁻¹⁰), matching the extraction sample points 1.9e-10; the paper itself writes "we use ge = (me/v) sin θ", independently confirming Parametrization (B) with no stray √2. The "(JCAP, unverified volume)" flag now resolves: DOI 10.1088/1475-7516/2023/07/071 = **JCAP 07 (2023) 071** — metadata completion, not an error. |
| 2 | Damour–Donoghue normalization κ = √(4πG_N) = 1/(√2·M_Pl,red) | **PASS** | 1007.2792 real: Damour & Donoghue, "Equivalence Principle Violations and Couplings of a Light Dilaton," Phys.Rev.D 82:084033 (2010), DOI 10.1103/PhysRevD.82.084033 — matches the note's Literature entry. Claim content re-located verbatim in the PDF this session: "κ ≡ √4πG is the inverse of the Planck mass" (below Eq. 3) and Eq. (12) L_int = κφ[(d_e/4e²)F² − … − Σ(d_mi + γ_mi d_g) m_i ψ̄ψ] — exactly the κ·d_me·m_e·φēe vertex the derivation rests on. The QSNET wording (2112.10618 Eq. 1, "κ = √(4πG_N)") carries the round-2 audit (CITATION-AUDIT item 13, verbatim-confirmed there); consistent with the DD original. Numerically 1/κ = √2·M_Pl,red = 3.444×10¹⁸ GeV ✓. |
| 3 | v = 246.22 GeV; reduced Planck mass usage | **PASS** | v = 246.22 GeV is the standard electroweak vev, v = (√2 G_F)^(−1/2) ≈ 246.2197 GeV [background knowledge; PDG-standard]; 2303.00778 itself uses v = 246 GeV — consistent. The note correctly attributes `_M_PL_GEV = 2.4e18` as repo-internal (verified at `evaluation/conventions.py:203`, with the comment block explaining the notebook-value choice over 2.418e18) and claims no external citation for it. `_CANONICAL_TOKEN` ScalarElectron→d_me and the raw-load `WhiteDwarfs` method at `PlotFuncs_ScalarVector.py:233` also verified in the working tree. |
| 4 | Red-giant g_e ≈ 7×10⁻¹⁶ loop-closure cross-check attributed to Hardy & Lasenby 1611.05852 | **PASS** | 1611.05852 real: Hardy & Lasenby, "Stellar cooling bounds on new light particles: plasma mixing effects," JHEP 02 (2017) 033 — matches the note's Literature entry. The bound is real and correctly attributed: 2303.00778 itself states verbatim "very restrictive bounds gL < 0.7×10⁻¹⁵ … based on the brightness of the tip of the red giant (RG) branch [24]" with [24] = Hardy & Lasenby — i.e. g_e ≲ 7×10⁻¹⁶, the exact value the note's RedGiants.txt back-conversion lands on (4.709e6 × m_e/(√2·2.4e18) = 7.09×10⁻¹⁶ ✓). `limit_data/ScalarElectron/RedGiants.txt` header cites 1611.05852 directly. |

**Honest gaps:** the ~7e-16 red-giant number was verified via 2303.00778's verbatim quotation of
Hardy–Lasenby (their result appears as gL < 0.7×10⁻¹⁵ with explicit attribution), plus the repo data
file's own header citation; the value was not re-read off Hardy–Lasenby's Fig. 3 axis (figure-pixel
data is not text-extractable). v = 246.22 GeV is standard-constant background knowledge, not an
audited external citation (the note makes no citation claim for it). The note's dex-residual
arithmetic and machine-encodable guard logic are derivation/code assertions outside citation-audit
scope (the numeric loop closures were spot-recomputed and agree).

**Recommended metadata completion (optional, not an error):** replace "(JCAP, unverified volume)"
with "JCAP 07 (2023) 071, DOI 10.1088/1475-7516/2023/07/071."
