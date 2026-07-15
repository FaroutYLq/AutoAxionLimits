---
concept: AxionEDM d_n [e·cm] oscillating-EDM amplitude → canonical g_{anγ} [GeV⁻²] (mass-dependent)
status: vetted (derivation + dimensional check + numeric spot-check + citation audit)
date: 2026-07-14
tokens_drained: 1708.06367, 2101.01241, 2208.07293  (the 3 unconvertible e·cm queue entries)
---

# Mass-dependent converter: oscillating nucleon-EDM amplitude d_n [e·cm] → g_{anγ} [GeV⁻²]

This is the **first mass-dependent converter** in the registry (Phase 2, #625).
Every prior converter is a constant factor (or a per-point *reciprocal*); this
one carries an explicit `× m_a` because the oscillating-EDM amplitude is the
axion-field *response*, and the field amplitude itself scales as `1/m_a`.

## Physics

An axion dark-matter background sources an **oscillating** nucleon EDM
(di Cortona, Hardy, Pardo Vega, Villadoro, *The QCD axion, precisely*,
JHEP 01 (2016) 034, arXiv:1511.02867):

    d_n(t) = g_d · a(t),   a(t) = a_0 cos(m_a t),   a_0 = √(2 ρ_DM) / m_a

where `g_d ≡ g_{anγ}` is the EDM **operator coupling** [GeV⁻²] — the canonical
repo quantity (`AxionEDM.ipynb` y-label `|g_{anγ}| [GeV⁻²]`; `nEDM.txt` header
`g_{aγn} [GeV^-2]`) — and `a_0` [GeV] is the local axion-field amplitude for a
DM density `ρ_DM`. Experiments that report a bound on the **oscillating EDM
amplitude** `d_n` [e·cm] (nEDM, CASPEr, JEDI storage ring) are therefore
constraining `g_{anγ}` only through the mass-dependent response:

    d_n = g_{anγ} · a_0 = g_{anγ} · √(2 ρ_DM) / m_a
    ⇒   g_{anγ} = d_n · m_a / √(2 ρ_DM)                     (natural units)

The `× m_a` is the operator-vs-response distinction the EXPLAIN doc flags
(`g_{anγ}` flat in `m_a`, the response `∝ 1/m_a`); inverting the response
re-introduces the `m_a`. This is a **plane conversion, not a DM-density
rescale** — the CLAUDE.md single-owner `sqrt(rho)` rule is untouched; `ρ_DM`
enters here as the intrinsic field-amplitude normalisation, not as a
paper→repo density ratio.

## Unit conversion (dimensional check)

`d_n` is quoted as "value · e · cm". In natural units (ħ=c=1,
Heaviside–Lorentz e = √(4πα) = 0.3028, ħc = 1.9733×10⁻¹⁴ GeV·cm ⇒
1 cm = 5.0677×10¹³ GeV⁻¹):

    d_n[GeV⁻¹] = d_n[e·cm] · e · (5.0677×10¹³ GeV⁻¹/cm)
               = d_n[e·cm] · 1.5346×10¹³          (K_ecm)

    ρ_DM = 0.4 GeV/cm³ = 0.4 · (5.0677×10¹³)⁻³ GeV⁴ = 3.074×10⁻⁴² GeV⁴
    √(2 ρ_DM) = 2.479×10⁻²¹ GeV²

    m_a[GeV] = m_a[eV] · 10⁻⁹

Combining (with m_a in **eV**):

    g_{anγ}[GeV⁻²] = C · d_n[e·cm] · m_a[eV],
    C = K_ecm · 10⁻⁹ / √(2 ρ_DM) = 6.19×10²⁴

Dimensions: [GeV⁻¹]·[GeV]/[GeV²] = GeV⁻². ✓

## Numeric spot-check (against same-paper GT curves)

The `g_{anγ}` GT files (`nEDM.txt`, `JEDI.txt`) are the **same limits from the
same papers** as the `d_n [e·cm]` extractions, so a correct conversion must
reproduce them. Converting the extracted `d_n` points:

| paper | GT file | mass overlap | median \|residual\| |
|---|---|---|---|
| 1708.06367 (nEDM) | nEDM.txt | 4e-22 … 4e-18 eV (6 pts) | **0.255 dex** |
| 2208.07293 (JEDI) | JEDI.txt | ~5e-10 eV (2 pts) | **0.129 dex** |
| 2101.01241 (CASPEr) | CASPEr-electric.txt | single-point GT, no curve overlap | (same physics) |

The residuals are **flat in mass** across 4 decades (nEDM: −0.35 … +0.48 dex,
scattered about 0, not monotone) — confirming the `× m_a` power is correct; a
wrong mass-power would trend monotonically over 4 decades. The scatter is the
vision-read noise of the extraction, not a systematic conversion error. The
constant `C` has **no free parameters** (e, ħc, ρ_DM = 0.4 all standard).

## Magnitude guard

A genuine oscillating-nucleon-EDM amplitude sits in `d_n ∈ [1e-30, 1e-18]`
e·cm (current static nEDM ~1e-26; oscillating-amplitude bounds probe a similar
band). Input medians outside this band are refused (`GUARD_REFUSED`) — the
same plausible-range discipline every round-2 converter carries.

## Citation audit

- **di Cortona et al., JHEP 01 (2016) 034, arXiv:1511.02867** — the axion–EDM
  operator `g_d`/`g_{anγ}` and its QCD coefficient (`3.7×10⁻³ GeV⁻¹ · (1/f_a)`,
  the inv_fa link). Establishes `d_n(t) = g_d a(t)`.
- **Abel et al. (nEDM), PRX 7 (2017) 041034, arXiv:1708.06367** — the
  oscillating-`d_n`-amplitude bound; its `g_{anγ}` re-expression is `nEDM.txt`.
- **JEDI storage-ring EDM, arXiv:2208.07293** — deuteron oscillating-EDM
  amplitude `d_AC`; `JEDI.txt` is its `g_{anγ}` form.
- `a_0 = √(2 ρ_DM)/m_a` — standard non-relativistic DM field amplitude
  (mirrors the EXPLAIN doc §Oscillating-DM amplitude and CLAUDE.md sqrt(rho)
  scope). ρ_DM = 0.4 GeV/cm³.
