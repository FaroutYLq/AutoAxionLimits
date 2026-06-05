---
concept: Coupling-convention canonicalization (scalar/dilaton d_i, axion-nucleon, axion-EDM, 1/f_a, dark-photon mixing)
date: 2026-06-04
mode: standalone
project_context: AutoAxionLimits extraction pipeline — canonicalizing extracted exclusion curves against limit_data/<Type>/*.txt
citation_status: primary physics verified from repo code + Damour-Donoghue / haloscope-convention sources; two experiment-specific scalar files (CsCav, Holometer) flagged unverified at file level
---

# Coupling-Convention Conversions for AutoAxionLimits

## Executive Summary

Per coupling family, the **canonical y-variable the repo actually stores/plots** is:

- **Scalar (ScalarPhoton / ScalarElectron):** dimensionless **Damour–Donoghue `d_e` / `d_{m_e}`** (notebook y-labels are literally `$d_e$`, `$d_{m_e}$`; axis range `g_min=1e-13 … g_max=1e10`). The alternative axis `g_{phi γ}` [GeV⁻¹] is just `d_e/(√2·M_Pl)`. **BUT the per-file storage is NOT internally consistent**: some files (SrSi, the DM-search clock/cavity limits) really are `d_e ~ 1e-5…1e-3`, while others (CsCav, Holometer, DAMNED-as-photon, HQuartzSapphire, and all fifth-force/Yukawa files) store an **experiment-native or fifth-force `α`-type quantity** that is converted to `d_e` only inside the notebook (e.g. `d_e = 500·√α`, `d_{m_e} = 4000·√α`, and `|α| = (d_{m_e}/4000)²`). The huge values up to `1e20…1e30` are **fill-wall sentinels** (the closing vertex of a `fill_between` polygon), not coupling values.
- **AxionNeutron / AxionProton:** canonical is the **dimensionless `g_{an}` / `g_{ap}`** (notebook y-label `$|g_{an}|$`, range `1e-16…1e-1`). The **files store the GeV⁻¹ derivative coupling `g_{aN}/(2 m_N) = C_N/(2 f_a)`**, and `PlotFuncs.py` multiplies by `2·m_N` in-code. **Headline: `g_{aN} (dimensionless) = 2 m_N · (stored GeV⁻¹ value)`, with `m_n = 0.93957`, `m_p = 0.93828` GeV.**
- **fa plane:** canonical is **`1/f_a` [GeV⁻¹]** (y-label `$1/f_a$ [GeV$^{-1}$]`, range `1e-20…1e-4`).
- **AxionEDM:** canonical is **`g_{anγ}` [GeV⁻²]** (the EDM/gluon operator coupling). Exact in-repo link: `g_{anγ} [GeV⁻²] = 3.7e-3 · (1/f_a [GeV⁻¹])`.
- **DarkPhoton:** canonical is the **kinetic mixing `χ` (= ε), dimensionless — NOT ε²** (FigSetup uses `chi_min/chi_max`; CAST file header says "kinetic mixing [dimensionless]", values reach `χ=1`).

The single highest-value rule for the pipeline: **a multi-decade residual against a scalar file is almost always a convention/sentinel artifact, not extraction error.** For axion-nucleon, the factor that bites is `2 m_N ≈ 1.88 GeV` (proton/neutron) between the GeV⁻¹ file and the dimensionless plot.

## Why This Matters Here

The pipeline extracts an exclusion curve (coupling vs mass) from a paper and compares it to `limit_data/<Type>/*.txt`. The *same physics* appears in different y-conventions:

- A paper may report axion-nucleon as `g_{aNN}` [GeV⁻¹] (CASPEr/NASDUCK style), while the repo plots dimensionless `g_{an}` — a fixed factor `2 m_N` apart (~0.27 dex).
- A scalar paper may report `d_e ~ 1e-3`, while a fifth-force-derived repo file stores an `α`-type number `~1e7`, with the `√`-and-prefactor conversion living only in the notebook — a **10–15 dex** apparent offset that is *not* an error.
- A dark-photon paper may report `ε²`; the repo stores `ε` (a factor-of-2-in-dex / square-root difference).

Encoding these into a convention registry lets the comparator canonicalize before computing residuals, so genuine extraction errors are not masked by, and are not confused with, convention mismatches.

## Prerequisites and Dependencies

- Reduced Planck mass `M_Pl = 2.418e18 GeV` (the notebook uses `M_pl = 2.4e18`; Damour–Donoghue's `κ ≡ √(4πG) = 1/M_Pl`).
- Natural units `ħ=c=1`; masses in eV (x-axis) and GeV (couplings); `ħc = 0.1973 GeV·fm`, and the notebook's `0.1973e-6` converts a force range λ [m] to a mass `m_φ = 0.1973e-6 / λ` [eV] (i.e. `m [eV] = ħc/λ` with `ħc ≈ 1.973e-7 eV·m`).
- Local DM density `ρ_DM ≈ 0.3–0.4 GeV/cm³` enters ONLY for DM-search (oscillating-field) limits, never for fifth-force/EP/stellar bounds. (Mirrors the repo's `sqrt(rho_DM)` correction-scope rule in CLAUDE.md.)
- Damour–Donoghue dilaton parameterization (the `d_i`).
- Axion derivative (shift-symmetric) coupling and its on-shell pseudoscalar reduction.

## Core Explanation

### 1. Scalar / dilaton `d_i` (Priority 1)

**Definition (established).** In the Damour–Donoghue framework (arXiv:1007.2792) the light scalar φ couples to SM operators with five *dimensionless* coefficients `d_e, d_g, d_{m_e}, d_{m_u}, d_{m_d}`, via terms like `(d_e/4e²) F_{μν}F^{μν}`, `(d_g β_3/2g_3) G²`, `d_{m_e} m_e ēe`, multiplied by the **dimensionless** field `φ̂ ≡ κφ = φ/M_Pl`. The phenomenological consequence is

  `α(φ) = α·(1 + d_e φ̂)`,  `m_i(φ) = m_i·(1 + d_{m_i} φ̂)`,  with `φ̂ = φ/M_Pl`.

So `d_e` is dimensionless, and current excluded values run `~1e-3` (EP tests) down to `~1e-5` and up to `~0.16–0.7` for the weakest DM-search bands (confirmed: Campbell et al., PRL 126, 071301 (2021), arXiv:2010.08107, report `|d_e| ≳ 0.16`, `|d_{m_e}−d_g| ≳ 0.7`).

**Oscillating-DM amplitude (established).** For φ as dark matter, the field oscillates with amplitude `φ_0 = √(2 ρ_DM)/m_φ`, so the *fractional* oscillation of a constant is

  `δα/α = d_e · φ_0/M_Pl = d_e · √(2 ρ_DM)/(M_Pl · m_φ)`.

This is why DM-search files (clocks/cavities/interferometers) report `d_e` directly — they have already divided out `√(2ρ_DM)/(M_Pl m_φ)`.

**What the large-valued repo files actually store (verified from repo, file-source unverified).** Reading `PlotFuncs_ScalarVector.py`, every Scalar method just `loadtxt`s the file and `FilledLimit(...)`s it with `y2=1e20` — **no in-method transform**. Yet the notebook y-axis is `$d_e$` with `g_max=1e10`. The reconciliation comes from the notebook (`Scalars.ipynb`):

- Fifth-force / ISL files are stored as `(λ [m], α)` (e.g. `ScalarNucleon/IUPUI.txt` header literally reads `lambda [m]  alpha`) and converted *in the notebook*:
  - `d_e = 500·√α` (composition-independent, `Q_e = 1/500`),
  - `d_{m_e} = 4000·√α` (`Q_{m_e} = 1/4000`),
  - and the union-ISL file uses `d = 500·α·√(1.37e37)`.
- The inverse map the notebook uses for the `|α|` plot is explicit: `|α| = (d_{m_e}/4000)²`. So **`d ∝ √α` with a per-charge prefactor (~500 EM, ~4000 electron).**
- The values up to `1e20/1e30` are **sentinels**: inspecting the data, the first and last rows of `CsCav.txt`, `Holometer.txt`, `HQuartzSapphire.txt` are `(mass, 1e30)` / `(mass, 1e20)` — the polygon-closing vertices for `fill_between`. The real interior values are e.g. CsCav `~2.7e4…3.6e8`, Holometer `~1e4…1e13`, HQuartzSapphire `~1.6…1e6`, SrSi `~1e-5…1e-3`.

**Therefore the offset spread (0.6 dex → 15+ dex) is real internal inconsistency, not noise:**

| File | interior range | what it is | offset vs true `d_e` |
|---|---|---|---|
| `ScalarPhoton/SrSi.txt` | `1e-5 … 1e-3` | genuine `d_e` (Sr-clock/Si-cavity DM search) | ~0 (≤0.6 dex) |
| `ScalarPhoton/HQuartzSapphire.txt` | `1.6 … 1e6` | experiment-native (BAW/sapphire), NOT `d_e` | several dex |
| `ScalarElectron/CsCav.txt` | `2.7e4 … 3.6e8` | experiment-native (header *claims* `d_me`, but magnitude is wrong for `d_me`) | ~10–14 dex |
| `ScalarPhoton/Holometer.txt` | `1e4 … 1e13` | experiment-native (header claims `d_e`) | ~10–15 dex |
| `ScalarNucleon/IUPUI.txt` | `1e4 … 2e17` | fifth-force `α` vs λ — needs `d = 500·√α` | 15+ dex |

**Exact conversion to canonical `d_e` (which constants enter):**

- From a **DM fractional amplitude** `δα/α`: `d_e = (δα/α) · M_Pl · m_φ / √(2 ρ_DM)`. Constants: `M_Pl`, `m_φ` (the x-axis mass), `ρ_DM`. (This is the inverse of the oscillating-DM formula; use ONLY for DM searches.)
- From a **fifth-force/EP `α`-strength** (Yukawa `V = α G m_1 m_2 e^{-r/λ}/r`): `d_e = Q_e^{-1}·√α` with `Q_e^{-1} ≈ 500` (EM) and `Q_{m_e}^{-1} ≈ 4000` (electron mass). Constants: charge `Q`, no `M_Pl`/`ρ_DM`. (This is exactly the notebook's transform; `M_Pl` does *not* enter because `α` is already the gravitational-strength ratio.)
- From the **GeV⁻¹ Compton-like coupling** `g_{φγ}` [GeV⁻¹]: `d_e = g_{φγ}·√2·M_Pl` (inverse of the `AlternativeCouplingAxis(scale=1/(√2 M_Pl))`).

**Dimensional check.** `d_e` dimensionless ✓. `δα/α` dimensionless; `M_Pl·m_φ/√(ρ_DM)` has dims `[GeV]·[GeV]/[GeV²] = 1` (with `ρ_DM` in GeV⁴) ✓. `g_{φγ}[GeV⁻¹]·M_Pl[GeV] = 1` ✓. `√α` dimensionless ✓.

### 2. Axion–nucleon (Priority 2) — factor of 2 RESOLVED

**Lagrangian (established).** The shift-symmetric derivative coupling is
  `L ⊃ (C_N / 2 f_a) (∂_μ a) N̄ γ^μ γ^5 N`.
Integrating by parts and using the nucleon Dirac equation, `∂_μ(N̄ γ^μ γ^5 N) = 2 m_N · N̄ i γ^5 N`, so on-shell this reduces to the **pseudoscalar form** `L ⊃ g_{aN} a N̄ i γ^5 N` with

  **`g_{aN} = C_N m_N / f_a`** (dimensionless).

**Two coexisting "GeV⁻¹" conventions in the literature:**
- "derivative coupling" `g_{aNN} ≡ C_N/(2 f_a)` [GeV⁻¹] — used by CASPEr-wind / NASDUCK (their plotted GeV⁻¹ number).
- some authors quote `C_N/f_a` [GeV⁻¹] (no 2), and SNO-type analyses quote yet another normalization.

**What the repo does (verified, decisive).** In `PlotFuncs.py` the `AxionNeutron` class comment states verbatim: *"often couplings are actually given as `g_an/2 m_n` (this is what is in the limit_data files) … we are using the dimensionless coupling, so will multiply by the neutron mass."* The data-file headers confirm it: `AxionNeutron/NASDUCK-SERF.txt` → `g_aNN [GeV^-1]`; `AxionProton/NASDUCK-SERF.txt` → `g_ap/2m_p [GeV^-1]`. Every plotting method does `dat[:,1] *= 2*AxionNeutron.m_n` (and `2*m_p` for protons). Notebook y-axis is `$|g_{an}|$`, range `1e-16…1e-1`.

So the **repo's operative relation is:**

  **`g_{aN}^{dimensionless} = 2 m_N · (value stored in file)`, where the stored value = `g_{aNN} = C_N/(2 f_a)` [GeV⁻¹].**
  Equivalently `g_{aN} = 2 m_N · C_N/(2 f_a) = m_N C_N / f_a` — consistent with the on-shell reduction above.

**Verdict on the pipeline's current rule.** The pipeline uses `g_aN = 2 m_N (C_N/f_a)` for both electron and nucleon. This is **correct *iff* "`C_N/f_a`" denotes the stored GeV⁻¹ value (which is really `C_N/2f_a`)** — i.e. the rule should be read as `g_aN = 2 m_N × (the paper's plotted GeV⁻¹ coupling)`. If instead the code literally multiplies the *true* `C_N/f_a` (no internal 2) by `2 m_N`, it is a **factor of 2 too large**. **Action:** verify that the quantity the pipeline calls `C_N/f_a` is the experiment's plotted `g_{aNN}` [GeV⁻¹] (= `C_N/2f_a`); if so, keep `×2 m_N`. The electron analog `g_ae = 2 m_e (C_e/f_a)` has the identical structure and is correct under the same reading.

**SNO exception (verified).** `AxionNeutron/SNO.txt` header is `g_an/m_n [GeV^-1]` and the code does `dat[:,1] *= AxionNeutron.m_n` (factor `m_n`, no 2). The repo comment: *"their notation defines their g_an as my g_an/m_n not g_an/2m_n."* So the per-file multiplier is convention-dependent — the registry must key the conversion **per file/source**, not globally.

### 3. Briefer families (Priority 3)

**(a) fa plane.** Canonical `1/f_a` [GeV⁻¹] (verified: `Axion_fa.ipynb` y-label `$1/f_a$ [GeV$^{-1}$]`, `g_min=1e-20…g_max=1e-4`; `fa/nEDM.txt` header `1/f_a [GeV^-1]`). It is the **unnormalized** inverse PQ scale (not Planck-normalized). The alternative axis is `g_{anγ}` [GeV⁻²] via `scale=3.7e-3` (see (b)). Convert from `f_a` [GeV] by simple reciprocal: `(1/f_a) = 1/f_a`.

**(b) AxionEDM.** Canonical `g_{anγ}` [GeV⁻²] (verified: `AxionEDM.ipynb` y-label `$|g_{anγ}|$ [GeV$^{-2}$]`; `AxionEDM/nEDM.txt` header `g_{aγn} [GeV^-2]`). Physics chain (gluon `aG̃G` operator → oscillating nucleon EDM `d_n(t) = g_d a(t)`):
  - `1/f_a` [GeV⁻¹] ↔ `g_{anγ}` [GeV⁻²]: **`g_{anγ} = 3.7e-3 GeV⁻¹ · (1/f_a)`** (the repo's `AlternativeCouplingAxis(scale=3.7e-3)`; the `3.7e-3 GeV⁻¹ ≈ g_d` is the QCD coefficient `≈ 2.4×10⁻¹⁶ e·cm·GeV` re-expressed, established to ~few %).
  - `d_n` [e·cm] ↔ `g_d` [GeV⁻²-equivalent]: `d_n(t) = g_d a_0 cos(m_a t)`, `g_d ≈ 2.4×10⁻¹⁶ (C_G/f_a) e·cm·GeV` per di Cortona et al. The `C_G/(f_a m_a)` [GeV⁻²] form (relevant to arXiv:2204.01454) differs from `g_{anγ}` by the **extra `1/m_a`** — i.e. it folds in the field amplitude `a_0 = √(2ρ)/m_a`; do not confuse the operator coupling (`g_{anγ}`, mass-independent) with the *response* (`∝ 1/m_a`). Flag any file whose slope tracks `1/m_a` as the `C_G/(f_a m_a)` convention.

**(c) DarkPhoton.** Canonical **kinetic mixing `χ` (= ε), dimensionless, NOT ε²** (verified: `DarkPhoton.FigSetup(chi_min, chi_max)`; `DarkPhoton/CAST.txt` header "kinetic mixing [dimensionless]", values up to `χ=1`, CAST bound `~1e-7` in χ). If a paper reports `ε²`, convert `χ = √(ε²)`. The notebook DM-overproduction lines apply *physics* factors (`χ_a = ε·√(2/3/0.13)` etc.) — these are anisotropy/DM-fraction corrections, not unit conversions, and must NOT be applied during canonicalization.

## Formal Structure / Equations

```
Scalar (DM search):     δα/α = d_e · √(2 ρ_DM) / (M_Pl · m_φ)
  → d_e = (δα/α) · M_Pl · m_φ / √(2 ρ_DM)
Scalar (fifth force/EP): d_e = Q_e^{-1} √α  (Q_e^{-1}≈500 EM, Q_{m_e}^{-1}≈4000 electron)
Scalar (GeV^-1 form):    d_e = g_{φγ}[GeV^-1] · √2 · M_Pl
Axion-N (repo):          g_{aN}^{dimless} = 2 m_N · g_{aNN}[GeV^-1],  g_{aNN}=C_N/(2 f_a)
                         (SNO file: g_{aN}=m_N · (stored g_an/m_n))
                         m_n=0.93957 GeV, m_p=0.93828 GeV
Axion-N (on-shell):      g_{aN} = C_N m_N / f_a   [from L=(C_N/2f_a)(∂a)N̄γγ5N]
fa:                      canonical 1/f_a [GeV^-1]
AxionEDM:                g_{anγ}[GeV^-2] = 3.7e-3 · (1/f_a)[GeV^-1]
                         C_G/(f_a m_a)[GeV^-2] = g_{anγ}/m_a  (response, ∝1/m_a)
DarkPhoton:              χ = ε = √(ε²)
M_Pl = 2.418e18 GeV (reduced)
```

## Project-Specific Connection (grounded in the actual repo)

Files/code inspected (paths + observed values):

- `PlotFuncs_ScalarVector.py` (lines 47–211 `ScalarPhoton`, 211+ `ScalarElectron`): methods call `FilledLimit(..., y2=1e20)` with **no per-method transform** — so file storage is taken at face value on a `$d_e$`/`$d_{m_e}$` axis.
- `Scalars.ipynb`: y-labels `$d_e$` (`g_min=1e-13, g_max=1e10`), `$d_{m_e}$`; `AlternativeCouplingAxis(scale=1/(√2·M_pl), ylabel "g_{φγ} [GeV^-1]")`; conversion cells `d_e=500·√α`, `d_{m_e}=4000·√α`, `|α|=(d_{m_e}/4000)²`; `M_pl=2.4e18`.
- Data ranges read (sentinels included): `ScalarElectron/CsCav.txt` 2.7e4–1e30 (interior ≤3.6e8); `ScalarElectron/DAMNED.txt` 0.083–1e30; `ScalarPhoton/Holometer.txt` 9.4e3–4e13; `ScalarNucleon/IUPUI.txt` 7.5e3–2.2e17 (header `lambda [m] alpha`); `ScalarPhoton/HQuartzSapphire.txt` 1.6–1e20; `ScalarPhoton/SrSi.txt` **9.5e-6–4.6e-3 (genuine d_e)**.
- `PlotFuncs.py` `AxionNeutron` (lines 2471–2778): class comment "files hold `g_an/2 m_n`", `m_n=0.93957`; every method `dat[:,1] *= 2*AxionNeutron.m_n`; SNO uses `*= m_n` (line ~2722). `AxionProton` `m_p=0.93828`.
- Headers: `AxionNeutron/NASDUCK-SERF.txt` `g_aNN [GeV^-1]`; `AxionProton/NASDUCK-SERF.txt` `g_ap/2m_p [GeV^-1]`; `AxionNeutron/SNO.txt` `g_an/m_n [GeV^-1]`; `AxionEDM/nEDM.txt` `g_{aγn} [GeV^-2]`; `fa/nEDM.txt` `1/f_a [GeV^-1]`; `DarkPhoton/CAST.txt` "kinetic mixing [dimensionless]".
- Notebooks: `AxionNeutron.ipynb` ylab `$|g_{an}|$` (1e-16…1e-1); `Axion_fa.ipynb` ylab `$1/f_a$ [GeV^-1]` + `AlternativeCouplingAxis(scale=3.7e-3, "|g_{anγ}| [GeV^-2]")`; `AxionEDM.ipynb` ylab `$|g_{anγ}|$ [GeV^-2]`; `DarkPhoton` `chi_min/chi_max`.

**Implication for the registry:** canonicalization must be keyed **per file/source**, not just per coupling family, because within ScalarPhoton/ScalarElectron the storage convention varies file-to-file (SrSi = `d_e`; CsCav/Holometer/IUPUI = native/fifth-force), and within AxionNeutron the per-file multiplier is `2 m_n` (most) vs `m_n` (SNO).

## Common Confusions and Failure Modes

- **Sentinel values as couplings.** `1e20`, `1e30`, `1e99` first/last rows are `fill_between` closing vertices. Strip the polygon-closing endpoints (the rows where y = exactly `1e20/1e30/1e99`) before computing residuals or value-range discrimination.
- **Assuming the header is the truth.** `CsCav.txt`/`Holometer.txt` headers say `d_me`/`d_e` but the magnitudes (1e4–1e13) are inconsistent with `d_e`. Use the *value range*, not the header string, to classify.
- **Applying `√(2ρ_DM)/M_Pl/m_φ` to non-DM bounds.** Fifth-force/EP/stellar `d_e` files use `d=Q^{-1}√α`; `M_Pl` and `ρ_DM` must NOT enter. Mirror the repo's DM-correction scope rule.
- **Axion-nucleon double-counting the 2.** The trap is whether the pipeline's `C_N/f_a` already includes the experiment's internal `1/2`. The safe operational rule: `g_{aN} = 2 m_N × (the GeV⁻¹ number the paper plots)`, except SNO-style sources (`× m_N`).
- **DarkPhoton ε vs ε².** Storing ε² when the repo uses χ produces a clean factor-of-two-in-dex (square) offset; detect via the value being the square of the expected χ.
- **AxionEDM operator vs response.** `g_{anγ}` (flat in `m_a`) vs `C_G/(f_a m_a)` (`∝1/m_a`): check the curve's mass-scaling, not just the unit string.

## Machine-Encodable Convention Table

| coupling_type | canonical (repo) | alternates (detection range) | conversion → canonical (formula + constants) | confidence |
|---|---|---|---|---|
| ScalarPhoton | `d_e` (dimensionless) | (i) genuine `d_e` [1e-6…1e0]; (ii) fifth-force `α`/native [>1e3, up to sentinel 1e20/1e30] | (i) identity; (ii) `d_e = 500·√α` (drop sentinel rows); GeV⁻¹ form `d_e = g_{φγ}·√2·M_Pl`, `M_Pl=2.418e18` | High (canonical), Medium (per-file native source) |
| ScalarElectron | `d_{m_e}` (dimensionless) | same two classes; CsCav interior 2.7e4…3.6e8 = native | (i) identity; (ii) `d_{m_e}=4000·√α`; `|α|=(d_{m_e}/4000)²` | High (canonical); CsCav/Holometer file-source **unverified** |
| ScalarNucleon | `d_e`/`d_g`-type (dimensionless) on `$d_e$` axis | files often `(λ[m], α)` [α: 1e-30…1, mass via `m=0.1973e-6/λ`] | `d = 500·α·√(1.37e37)` (Union-ISL) or `500·√α`; mass `m[eV]=0.1973e-6/λ[m]` | Medium (notebook-derived) |
| ScalarBaryon | feeds Scalar `d_e`/`d_{m_e}` | `(λ[m], α)` fifth-force pairs | `d_e=500·√α`, `d_{m_e}=4000·√α`; mass `=0.1973e-6/λ` | Medium |
| AxionNeutron | `g_{an}` (dimensionless) [1e-16…1e-1] | file = `g_{aNN}=C_n/2f_a` [GeV⁻¹]; SNO = `g_an/m_n` [GeV⁻¹] | `g_an = 2·m_n·(file)`, `m_n=0.93957`; **SNO: `g_an = m_n·(file)`** | High (verified in code) |
| AxionProton | `g_{ap}` (dimensionless) | file = `g_{ap}/2m_p` [GeV⁻¹] | `g_ap = 2·m_p·(file)`, `m_p=0.93828` | High (verified) |
| AxionEDM | `g_{anγ}` [GeV⁻²] | `1/f_a` [GeV⁻¹]; `C_G/(f_a m_a)` [GeV⁻², ∝1/m_a]; `d_n` [e·cm] | `g_{anγ}=3.7e-3·(1/f_a)`; `g_{anγ}=(C_G/(f_a m_a))·m_a` | High (1/f_a link verified); Medium (e·cm coeff) |
| fa | `1/f_a` [GeV⁻¹] [1e-20…1e-4] | `f_a` [GeV]; `g_{anγ}` [GeV⁻²] | `(1/f_a)=1/f_a`; `(1/f_a)=g_{anγ}/3.7e-3` | High (verified) |
| DarkPhoton | `χ`=`ε` (dimensionless) [up to 1] | `ε²` [≈ χ², much smaller] | `χ=√(ε²)`; do NOT apply DM-fraction `√(2/3/0.13)` factors | High (verified) |

Sentinel rule: for any Scalar file, discard rows where `y ∈ {1e20, 1e30, 1e99}` before range-based classification. Scalar discriminator: if (post-sentinel) `max(y) > 1e3` → native/fifth-force convention (convert), else treat as genuine `d_e`.

## Literature Guide

Foundational:
- **Damour & Donoghue, "Equivalence Principle Violations and Couplings of a Light Dilaton," Phys. Rev. D 82, 084033 (2010)** — defines the dimensionless `d_i` and `α(φ)=α(1+d_eφ̂)`, `φ̂=φ/M_Pl`. Open: https://arxiv.org/abs/1007.2792
- **di Cortona, Hardy, Pardo Vega, Villadoro, "The QCD axion, precisely," JHEP 01 (2016) 034** — axion-nucleon `c_N` and the EDM `g_d` coefficient (the `3.7e-3`/`2.4e-16 e·cm` constants). Open: https://arxiv.org/abs/1511.02867

Practical / working:
- **Campbell, McAllister, Goryachev, Ivanov & Tobar, PRL 126, 071301 (2021)** (arXiv:2010.08107; repo H/Quartz/Sapphire source) — concrete `|d_e| ≳ 0.16`, `|d_{m_e}−d_g| ≳ 0.7` exclusion magnitudes confirming `d_e` is the plotted variable and its order. Open: https://arxiv.org/abs/2010.08107 [author attribution corrected per citation audit; "Tretiak et al." is arXiv:2201.02042, the CsCav source]
- **PDG "Axions and Other Similar Particles" review** — standard `g_{aN}`/derivative-coupling definitions. Open: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-axions.pdf
- **cajohare/AxionLimits (upstream repo)** — the actual convention source for this fork; `PlotFuncs.py` axion-nucleon comments and `AlternativeCouplingAxis` scales are authoritative for "canonical." Open: https://github.com/cajohare/AxionLimits

Frontier / experiment papers cited in the questions:
- **NASDUCK-SERF, arXiv:2209.13588** — `g_{aNN}` [GeV⁻¹] axion-nucleon (matches repo file headers). https://arxiv.org/abs/2209.13588
- **SNO axion-induced dissociation, arXiv:2004.02733** — the `g_an/m_n` (no-factor-2) convention; explains the SNO-only `× m_n`. https://arxiv.org/abs/2004.02733
- **Schulthess et al., "New Limit on Axion-Like Dark Matter using Cold Neutrons," PRL 129, 191801 (2022)** (arXiv:2204.01454) — the `C_G/(f_a m_a)` `∝1/m_a` EDM-response convention (limit `C_G/f_a m_a = 2.7×10¹³ GeV⁻²`). https://arxiv.org/abs/2204.01454
- **Cs/cavity scalar-DM bounds, arXiv:2201.02042** (CsCav) and **Fermilab Holometer, arXiv:2108.04746** — sources for the two files whose stored magnitudes (1e4–1e13) are inconsistent with `d_e`; **their exact stored convention could not be verified via WebFetch (PDFs did not text-extract) and is flagged unverified.** https://arxiv.org/abs/2201.02042 , https://arxiv.org/abs/2108.04746

## Suggested Follow-up Questions

1. For CsCav (2201.02042) and Holometer (2108.04746): pull the figure axis labels from the published HTML/figures to nail whether the stored 1e4–1e13 numbers are `1/Λ` [GeV⁻¹], a frequency-domain amplitude, or an experiment-native sensitivity — currently unverified.
2. Should the registry store the *sentinel value* per family (1e20 vs 1e30 vs 1e99) so the comparator strips exactly the right closing vertices?
3. Confirm in pipeline code whether the variable named `C_N/f_a` is the experiment's plotted GeV⁻¹ coupling (then `×2 m_N` is right) or the true `C_N/f_a` (then it is 2× too large).
4. Do any AxionProton files besides NASDUCK use a non-`2m_p` multiplier (e.g. `cg=0/cg=1` SN files)? Each may need its own per-file key.
