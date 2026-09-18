
## Production additions (AutoAxionLimits)

This session runs inside the AutoAxionLimits pipeline, not the benchmark. The
task card above still holds, with the following changes and additions.

### Sources

- Besides the paper and its e-print you MAY use the paper's own official data
  release: its HEPData record (https://www.hepdata.net, search by arXiv id), or
  a Zenodo / GitHub data release the paper itself links. A file published by
  the authors beats any digitisation: use it, set `data_source` to
  `ancillary_file`, and name the record in `notes`.
- You must still NOT consult limit compilations (AxionLimits, cajohare,
  DarkCast) or other papers' data. The compilation you are feeding is
  AxionLimits itself; copying from it is circular and is reported to the
  reviewer.

### Extra output fields (add them to `result.json`)

- `"polarization_assumption"`: for dark-photon dark-matter searches, the
  polarisation scenario the reported curve assumes, in the paper's own words
  (e.g. `"random polarisation, sqrt(2/3) factor"`); `null` otherwise.
- `"alternatives"`: a list of short strings, one per other curve or coupling in
  the paper that a curator might reasonably have chosen instead (other
  scenarios, other couplings, other panels, expected vs observed, combined vs
  per-experiment), each naming where it is (figure / table / equation) and how
  it differs from what you reported. Empty list if there is none. A human
  reviewer reads this list.
- `"headline_check"`: if the paper quotes a headline number (for example
  "g_aγγ < 3×10⁻¹¹ GeV⁻¹ at 10 μeV"), an object
  `{"quoted": <value>, "mass_eV": <mass>, "traced": <your curve's value at that
  mass>}`; `null` if the paper quotes none.

### Diagnostic overlay

- Write `./overlay.png`: the figure panel you traced (cropped or as rendered)
  with your data points drawn on top in a contrasting colour, on the panel's
  own axes, so the reviewer sees the fit at a glance. If your points came from
  a table, text or a data file, overlay them on the paper's limit figure anyway
  when one exists. Skip only if the paper has no figure of the limit.
