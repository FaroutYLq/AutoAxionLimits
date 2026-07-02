# Plan: extraction-channel upgrades (handover for the next session)

Written 2026-07-02, immediately after the post-full346 merge (PRs #650–#657; see
`PLAN_post_full346.md`'s status header). Three workstreams, all **implementable
keyless** with real keyless scoreboards; each has a small key-gated validation
tail. Context: the micro median (0.245 dex) is now BELOW the 0.32-dex run-to-run
noise floor — the payoff frontier is the failure tail, the macro average, and
replacing the noisiest channel (figure vision, 0.268 dex median) with exact or
near-exact channels.

Standing constraints (unchanged):
- GT = O'Hare repo `limit_data` files only; exclusions documented + reversible.
- Every eval/pipeline PR reports before/after impact (keyless scoreboard now;
  Haiku subset re-extraction when the key returns — #648).
- Pipeline/infra PRs never mix with science content; never push master; PRs only.
- Extractor changes do NOT move the local rescore (cached snapshots) — do not
  claim benchmark improvement from a rescore after extractor-side changes.

Suggested order: **WS3 → WS1 → WS2** (smallest first; WS1's survey informs how
much WS2 is even needed — every paper with source data stops needing CV).

---

## WS1 — arXiv source tarballs as a first-class extraction channel

**Idea:** many papers' e-print source contains the actual curve coordinates —
pgfplots `\addplot table {file.dat}` data files, `.csv`/`.txt` companions, tikz
coordinate lists, numeric table bodies. Extracting those is deterministic and
exact; it should outrank every other source.

### Design

New module `pipeline/source_data.py`:
1. **Fetch**: `https://arxiv.org/e-print/<id>` (gzip/tar), cached like the PDF
   cache (`_pdf_cache_dir` pattern), polite backoff (reuse `download_pdf`'s
   retry discipline). Free — no Anthropic API.
2. **Safe extraction**: tarfile with path-traversal guards (reject absolute
   paths / `..`), per-file + total size caps (~50 MB), depth cap. Treat ALL
   content as untrusted data (never execute; sanitize before any LLM prompt —
   same `===PAPER_CONTENT===` delimiter discipline as PDF text).
3. **Candidate scan** (deterministic):
   - files referenced by `\addplot table/... {<path>}` or `\pgfplotstableread`;
   - loose `.dat/.csv/.txt` files that parse as ≥2 numeric columns;
   - inline `coordinates {(x,y)...}` blocks and long numeric table bodies.
   Rank candidates: referenced-by-result-figure > filename tokens
   (`limit|exclusion|bound|constraint|fig<N>`) > column count == 2 >
   value ranges plausible for the coupling (reuse `VALID_RANGES`).
4. **Axis semantics without vision**: pgfplots axis options (`xlabel=`,
   `ylabel=`, `xmode=log`) and figure captions give the units DETERMINISTICALLY
   — parse them into the declared `coupling_convention` string (the #657
   contract) so the 1d registry converts (GHz→eV, λ→eV, g², Γ…).
5. **Disambiguation LLM call (cheap, key-gated, behind a flag)**: when >1
   plausible candidate, one text-only call listing file heads + captions picks
   the target and confirms axis semantics. Heuristics-only mode must work.
6. **Selector integration**: new source `"source_data"`; add to
   `transform_guard.SOURCE_TIER` at 5 (above `table`=4); flows through the
   existing `Candidate`/`quality()` gates unchanged (validity, declaration
   contract, magnitude guards all apply). `data_source` enum gains the value —
   check consumers: `report.py`/`subset_compare` source breakdown lists,
   reviewer PR body, `_POINT_LIMIT_SOURCES` (source_data is NOT a point-limit).

### Keyless scoreboard (do this FIRST — it sizes the whole workstream)

Survey script `evaluation/source_survey.py`: run fetch+scan over the 346
benchmark ids (arXiv download only), then for each paper with a candidate file
compare the candidate's data DIRECTLY against the GT curve (reuse
`compute_interpolation_metrics` + canonicalization). Report:
- hit rate (% of pool with extractable source data),
- best-candidate median residual vs the paper's current channel residual,
- how many current >1-dex papers become <0.3 dex.
This measures the channel's ceiling with zero Anthropic calls. Ship the survey
as its own PR (report committed under `evaluation/eval_runs/source_survey.md`).

### Risks / edge cases
- Source withdrawn or PDF-only → fall through to existing channels silently.
- Figures included as pre-rendered PDFs/PNGs (no data files) — expected common;
  that residue is WS2's population.
- Multi-panel data files (3+ columns, several curves per file) — column-pair
  selection needs the disambiguation call; heuristic: leftmost x + each y.
- Tarball bombs / weird encodings — guards above; never raise out of the
  channel (log + fall through).

### PR slicing
1. fetcher + scanner + safety guards + unit tests (keyless);
2. survey report on the 346 pool (keyless scoreboard);
3. selector integration + declared-convention wiring (+ flag-gated LLM
   disambiguation); Haiku subset before/after when the key lands.

---

## WS2 — Deterministic CV curve tracing with LLM curve *selection* (#606 backstop)

**Idea:** the LLM decides WHICH curve (semantics); OpenCV extracts the pixels
(precision). Kills the ~0.2–0.3 dex vision tracing noise and the spot-check
×0.1 miscalibration class (2401.18076, 2112.12116).

### Existing assets (do not rebuild)
- `pipeline/axis_ocr.py` (#570/P1): axis tick OCR + decade-axis interpretation;
  CI already installs tesseract + opencv-python-headless + pillow.
- `extract_figures_from_pdf` (figure rendering), stage-2a axis reads,
  `cv_trace` already exists as a SOURCE_TIER entry (rank 0 = "unverified").
- The vision-verify spot-check + benchmark-line calibration
  (`_calibrate_vision_data`) — reuse as the corroboration signal.

### Design

`pipeline/cv_trace.py` (new or upgraded):
1. Render candidate figure pages at high DPI (pymupdf).
2. Detect the plot axes box (largest axis-aligned rectangle via line detection);
   calibrate pixel→data using `axis_ocr` ticks (log/linear per axis).
3. Segment curve candidates by color clustering (mask per distinct
   saturated color; skeletonize; order by x; drop < N-pixel fragments).
   Output: candidate set, each with color descriptor + coarse shape stats.
4. **One cheap LLM call**: figure image + rendered candidate overlays ("curve A
   = red solid, spans …") → pick the paper's OWN result curve + declare the
   axis quantity (feeds the #657 declaration + WS3 gate lexicons: reject picks
   matching named existing experiments when `is_projection=true`).
5. Verification before promotion: CV axis calibration must agree with stage-2a
   reads (reuse R2 axis-disagreement contract); benchmark-line/spot-check
   corroboration. Verified traces get a `cv_trace_verified` source at tier 3.5
   (above `figure_vision`, below `table`) — unverified stays tier 0.

### Keyless scoreboard

`evaluation/cv_ceiling.py`: for benchmark papers with GT, run steps 1–3 on the
paper's figures and score EVERY candidate curve against GT (no selection, no
LLM). Report per paper: does ANY CV candidate beat the cached vision residual?
- % of figure-vision papers where CV extraction is feasible (axes calibrated),
- best-candidate residual distribution vs vision's 0.268 dex median,
- the 9 wrong-curve papers: is the CORRECT curve among the candidates?
This proves the ceiling and quantifies how much of the win is selection (LLM,
key-gated) vs extraction (CV, keyless).

### Risks
- Dense filled compilation plots (many overlapping translucent regions) — CV
  will fail there; that is fine, fall back to vision (gate D territory).
- Dashed/dotted monochrome curves — skeleton gaps; bridge small gaps along x,
  else drop the candidate (never emit a fragment as a curve — R4 applies).
- Recalibrate thresholds only via the existing contract constants; do not fork
  new magic numbers.

### PR slicing
1. cv_trace core + axis calibration + tests on synthetic rendered figures;
2. ceiling survey on the benchmark figures (keyless scoreboard);
3. LLM selection call + selector integration + subset validation (key).

---

## WS3 — Wrong-curve gates + replay harness (Lever 5, keyless half)

**Idea:** the 2d gates are post-hoc checks on extraction OUTPUTS, and every
input they need is already in the cached snapshots (vision notes, points,
coupling, `is_projection`) + the metadata cache (abstracts). Build the gates
now; measure them by replay; leave only the re-prompt reaction and lever-D
re-validation for the key.

### Gates (`pipeline/vision_gates.py`, pure functions)

| Gate | Trigger | Action | Known catches |
|---|---|---|---|
| A projection-target | `is_projection=true` AND vision notes name an existing experiment as the traced curve (lexicon built from `limit_data/**` file stems + common names: Eot-Wash, EP, CAST, SN1987A…) | reject candidate | 1512.06165, 1508.01798, 2309.07995 |
| B axis-vs-coupling | vision-reported y-axis quantity contradicts the declared coupling's canonical axis family (extends #657: the gate REJECTS the candidate so selection falls back, instead of only re-declaring) | reject candidate | 1708.02111 |
| C mass-regime | traced x-window ≥ ~3 dex outside the abstract-stated mass window; fire only when the abstract parse is unambiguous (regex for `eV/µeV/meV/keV/GeV` ranges; skip if none/multiple) | reject/flag | 1903.12190 (14 dex), 1808.02340 (6 dex) |
| D compilation-envelope | notes admit tracing an envelope/union/"combined with surrounding context" | demote candidate (new `Candidate` flag, same pattern as `reconstruction`) | 1008.3536, 1207.3275 |

Rejection is never silent: falls back to the next candidate or zero points +
`[VISION GATE]` note + confidence cap (mirror `[CONVENTION REVIEW]`).

### Replay harness (`evaluation/replay_gates.py`) — the keyless scoreboard

Replay the gates over all 346 cached snapshots (`evaluation/results/*.json` +
`metadata_cache.json` abstracts). Report (commit as
`evaluation/eval_runs/gate_replay.md`):
- per-gate trigger list with the note excerpt that fired it,
- catch rate on the 9 known wrong-curve papers (target ≥ 5/9 — notes-based
  gates cannot catch traces whose notes are silent),
- false-trigger rate on the ~120 good vision papers (target ≤ 2%; tune
  lexicons/thresholds down, never per-paper).

Caveat for the implementer: cached snapshot `notes` are the WINNING sample's
notes post-selection — at runtime the gate sees each candidate's own notes,
which is strictly more information. Replay is therefore a LOWER bound on catch
rate; say so in the report.

### Key-gated remainder (do NOT attempt keyless)
- re-prompt/re-route reaction (re-run vision at another panel / higher DPI);
- panel targeting by abstract window (prompt change);
- the plan-mandated re-validation of the 9 papers against
  `fix/613-leverd-curve-selection` @ f931446e BEFORE further prompt work;
- Haiku subset before/after via `subset_eval` + the gate workflow.

### PR slicing
1. gates + unit tests + replay harness + replay report (one PR, keyless);
2. runtime integration into `run_extraction_agent` candidate flow (small PR;
   include the replay numbers as its impact block);
3. key-gated validation follow-up.

---

## Cross-cutting notes for the implementing session

- Branch names: `pipeline/ws1-source-data`, `pipeline/ws2-cv-trace`,
  `pipeline/ws3-vision-gates` (+ `evaluation/...` where eval-only).
- Tests live in `evaluation/tests/` (CI job `eval_tests` runs them keyless;
  opencv/tesseract/pillow already installed there).
- The selector (`transform_guard.quality`) is the single integration point for
  all three channels — extend tiers, never fork a second ranking.
- Each new channel/candidate must carry a truthful `coupling_convention`
  declaration (#657) so the 1d registry does conversions — none of these
  channels should ever convert units itself except via `to_canonical`.
- When the key returns, the shared validation protocol for everything above is:
  Haiku subset re-extraction (`subset_eval.py`) before/after per PR +
  `evaluation/gate.py`, then the Phase-4A full-pool re-baseline.
