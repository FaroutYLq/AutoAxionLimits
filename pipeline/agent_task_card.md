# AxionLimitBench task card

You are extracting an experimental exclusion limit from a particle-physics paper.
The output is judged against the curve that the maintainer of the AxionLimits
compilation digitised for the same paper, so the goal is the curve a careful
physicist would add to that compilation: the paper's headline measured limit,
in the canonical variable and units listed below, as a list of (mass, coupling)
points.

## Inputs you may use

- The paper itself: `./paper.pdf` in the current working directory.
- The paper's own arXiv e-print (LaTeX source, embedded figure files, ancillary
  data files): `https://arxiv.org/e-print/<id>` or `https://arxiv.org/src/<id>`.
- Any code you write yourself. `python3` is available with numpy, scipy,
  pymupdf (`fitz`), matplotlib, PIL, cv2 and skimage.

You must NOT consult secondary sources: no limit compilations (AxionLimits,
cajohare, DarkCast), no HEPData, INSPIRE, PDG, no other papers, no web search.
Everything you report must come from this paper and its e-print. Sessions are
audited for this.

## What to decide

1. **Is there a new measured limit?** `is_new_limit` is true only if the paper
   reports its own experimental or observational exclusion on one of the
   couplings below. Projections, sensitivity forecasts and proposals are
   `is_projection: true` and are out of scope (report them as such with no
   points). Pure theory papers and reinterpretations of other experiments'
   data are not new limits unless the paper derives a new bound itself.
2. **Which coupling?** One `coupling_type` from the table. If the paper bounds
   several, pick the one it presents as its main result.
3. **The curve.** The boundary of the excluded region, as points
   `[mass_eV, coupling]` with the coupling in the canonical variable and units
   of the table, ordered by mass, positive finite numbers only, typically
   10 to 100 points (fewer if the limit is a flat line or a single value; a
   flat limit over a stated mass range is two points). Trace the paper's
   headline curve (the observed limit, not the expected sensitivity band).
4. **Metadata**, as in the schema below.

## Canonical planes

x-axis is always the boson mass in eV. y-axis:

| coupling_type | variable | units | definition | exact `coupling_convention` label |
|---|---|---|---|---|
| AxionPhoton | g_aγγ | GeV^-1 | L = -(g_aγγ/4) a F F̃ | `g_agamma [GeV^-1]` |
| AxionElectron | g_ae | dimensionless | L = g_ae (∂_μ a / 2m_e) ē γ^μ γ_5 e, i.e. g_ae = C_e m_e / f_a | `g_ae (dimensionless)` |
| AxionNeutron | g_an | dimensionless | L = (g_an / 2m_n) ∂_μ a n̄ γ^μ γ_5 n, i.e. g_an = C_n m_n / f_a; **see the 2m_N rule below** | `g_an (dimensionless)` |
| AxionProton | g_ap | dimensionless | same normalisation with the proton; **see the 2m_N rule below** | `g_ap (dimensionless)` |
| AxionEDM | g_d (also written g_aNγ) | GeV^-2 | L = -(i/2) g_d a N̄ σ_μν γ_5 N F^μν; the oscillating nucleon EDM is d_n = g_d a | `g_d [GeV^-2]` |
| AxionMass | 1/f_a | GeV^-1 | inverse axion decay constant in the convention where the QCD axion has m_a = 5.7 μeV × (10^12 GeV / f_a); used for bounds quoted directly on f_a, e.g. CP-violating or EDM-type searches | `1/f_a [GeV^-1]` |
| DarkPhoton | χ (kinetic mixing, also ε) | dimensionless | L ⊃ -(χ/2) F_μν F'^μν; report χ, never χ² | `kinetic mixing chi (dimensionless)` |
| VectorBL | g_B-L | dimensionless | gauge coupling of the U(1)_B-L boson; report g, never α = g²/4π | `g_BL (dimensionless)` |
| ScalarPhoton | d_e | dimensionless | Damour–Donoghue dilatonic coupling, α → α(1 + d_e κ φ), κ = √(4π)/M_Pl, M_Pl = 1.22×10^19 GeV | `d_e (dimensionless)` |
| ScalarElectron | d_me | dimensionless | m_e → m_e(1 + d_me κ φ), same κ | `d_me (dimensionless)` |
| ScalarNucleon | g_N (or the paper's own dimensionless scalar–nucleon coupling) | dimensionless | L = g_N φ N̄N; fifth-force papers instead quote |α| (Yukawa strength relative to gravity) versus range λ, with m = ħc/λ = 1.97×10^-7 eV·m / λ | `coupling (dimensionless)` |
| ScalarBaryon | g_B | dimensionless | coupling of a scalar to baryon number (equivalence-principle and fifth-force tests) | `coupling (dimensionless)` |
| MonopoleDipole | g_s g_p | dimensionless | product of scalar and pseudoscalar couplings for a monopole–dipole spin-dependent force (electron–nucleon or nucleon–nucleon), versus mediator mass | `g_s g_p (dimensionless)` |
| AxionCPV | as the paper defines | dimensionless | other CP-violating axion couplings | `coupling (dimensionless)` |

Conventions of the compilation:

- **Dark-matter density: do NOT rescale.** Report dark-matter-search limits
  (haloscopes and other direct searches) exactly as the paper states them, at
  the paper's own assumed local density, and record that density in
  `dm_density_assumed` (null if the limit does not depend on it). The
  compilation stores paper-native values and applies any density rescaling
  only when it draws the plot.
- **Axion–nucleon normalisation (the 2m_N rule).** The canonical g_an and
  g_ap are the dimensionless couplings of L = (g_aN / 2m_N) ∂_μ a N̄ γ^μ γ_5 N.
  Many papers instead write L = g ∂_μ a N̄ γ^μ γ_5 N with g in GeV^-1, or the
  non-relativistic Hamiltonian H = g ∇a · σ_N; those are the same physics with
  g_aN = 2 m_N g, **not** m_N g. Example: g = 1.0×10^-5 GeV^-1 gives
  g_an = 2 × 0.939 GeV × 1.0×10^-5 GeV^-1 = 1.9×10^-5. If the paper bounds a
  combination such as the isovector |g_an − g_ap|/2, report the bound on the
  single canonical coupling with the other set to zero and say so in `notes`.
- **Scenarios.** If the paper shows several curves (e.g. polarisation
  scenarios, model variants, prior choices), report the one it presents as its
  main result and say which in `notes`.
- **Declare what you emitted.** `coupling_convention` must be EXACTLY the label
  from the last column of the table for the coupling you report, e.g.
  `"g_d [GeV^-2]"`. It is parsed by a strict matcher: do not add symbols,
  definitions, formulas or the paper's own notation to it (a declaration like
  `"g_adgamma [GeV^-2] (d_d = g a)"` is rejected as unknown even when the
  numbers are right). Put derivations, equation numbers and conversions in
  `notes`. Only if you genuinely could not convert to the canonical variable,
  emit the paper's native variable and declare it with one of these native
  labels instead: `"d_n [e cm]"` (oscillating nucleon EDM amplitude),
  `"f_a [GeV]"`, `"chi^2 (dimensionless)"`,
  `"alpha (Yukawa strength relative to gravity)"`. A truthful native
  declaration is far better than a guessed conversion.

## Output: `./result.json`

```json
{
  "paper_title": "string",
  "coupling_type": "AxionPhoton | AxionElectron | ... | null",
  "is_new_limit": true,
  "is_projection": false,
  "data_points": [[1.0e-6, 3.2e-14], [2.0e-6, 2.9e-14]],
  "data_source": "table | text | figure_vision | figure_vector | ancillary_file | none",
  "dm_density_assumed": 0.45,
  "confidence_level": 0.95,
  "coupling_convention": "g_agamma [GeV^-1]",
  "suggested_experiment_name": "ShortName_Year",
  "extraction_confidence": 0.8,
  "notes": "which figure/table/equation the curve came from, scenario chosen, conversions applied, caveats"
}
```

- `data_source`: where the numbers came from (`ancillary_file` for a data
  file shipped with the e-print, `figure_vector` for coordinates recovered from
  a vector figure, `figure_vision` for reading a raster/vector figure by eye,
  `table` / `text` for tabulated or quoted values).
- `dm_density_assumed`: the density the paper assumed in GeV/cm³, or null when
  the limit does not depend on it.
- `confidence_level`: 0.90 or 0.95 as the paper states.
- `extraction_confidence`: your calibrated probability (0 to 1) that the curve
  is within a factor of two of the paper's headline limit over its mass range.
- If there is no measured limit to report, still write the file with
  `is_new_limit: false` (or `is_projection: true`) and `data_points: []`.

Write the file, then stop. Do not print the points in your final reply.
