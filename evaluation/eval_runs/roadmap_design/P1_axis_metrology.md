# P1 — Trustworthy axis metrology (OCR'd ticks + geometric fit, cross-validated)

> Roadmap phase **P1** of issue **#566** (governs) == **#550** Phase 1, but with the
> validation #550 assumed and the integration never added.
> Design/scoping only — no pipeline code in this doc; nothing committed/filed.
> Acceptance harness: `evaluation/subset_eval.py` + `evaluation/subset_compare.py`.
> Code designed against: `AAL-integration/pipeline/extractor.py`
> (`_run_stage2a_axes`, `_attach_cv_calibration`, `_calibrate_axis`,
> `_attempt_cv_curve_trace`, the CV-trace override gate at :1031),
> `AAL-integration/pipeline/plot_calibration.py`
> (`detect_plot_region`, `detect_axis_ticks`, `build_log_transform`, `trace_curve`),
> `AAL-integration/pipeline/config.py` (`VALID_RANGES`).
> **Builds on P0** (`pipeline/transform_guard.py`): P1 produces the *corroboration*
> object P0's `guard_transform` consumes; P0's **R2** axis-override rule is what
> actually commits-or-reverts the P1 result.

---

## 1. Problem & evidence

### 1.1 Why the current axis calibration is untrustworthy

`_attach_cv_calibration` / `_calibrate_axis` (extractor.py:1256-1396) implement the
#550 "LLM-semantics / CV-metrology split" *literally*: the **LLM supplies the tick
VALUES** (`x_axis_tick_values`, `y_axis_tick_values`, from the Stage-2a prompt at
extractor.py:1189-1197), and **OpenCV supplies only the tick PIXEL positions**
(`detect_axis_ticks`, plot_calibration.py:253-341). `build_log_transform`
(plot_calibration.py:348-441) then fits `log10(value)` vs `pixel` and the axis is
overwritten whenever the resulting endpoint disagrees with the LLM range by
`> 0.5 dex` (extractor.py:1370-1372).

This pipeline has **no independent reading of the tick labels**. The pixel↔value
pairing is taken on faith:

- The CV tick *count* and the LLM value *count* are aligned by **order and common
  prefix only** (extractor.py:1343-1347: `n = min(len(pix), len(vals))`,
  `transform = build_log_transform(pix[:n], vals[:n], scale)`). If `detect_axis_ticks`
  finds a spurious tick (gridline, minor tick, frame edge) or misses one, the i-th
  pixel is paired with the wrong i-th value.
- A wrong-but-*self-consistent* subset still passes the only sanity gate
  (`r2 >= 0.95`, plot_calibration.py:427): two or three collinear mispaired points
  fit a line beautifully, so a 12-decade or 24-decade endpoint shift sails through.
- `panel_index` is the LLM's guess; if it points at the wrong panel
  (extractor.py:1300-1303) the whole bbox/tick fit is for a different plot.

The result is class **A** in the fan-out: 7 papers where CV calibration overrode a
good LLM axis by many decades, all `r2`-passing.

### 1.2 The four CV-miscalib failures P1 must catch
(`evaluation/eval_runs/per_paper_findings.md`)

| Paper | What CV wrote | Mechanism (root cause P1 targets) | Eval delta |
|---|---|---|---|
| **1506.08082** (L131-134) | `y-max 2e-6 → 1e+22 GeV⁻¹` (24 decades; unphysical coupling axis) | Mispaired y-ticks fit a line with `r2≥0.95`; nothing checks that `1e22 GeV⁻¹` is impossible or that the OCR'd top label reads `2e-6`, not `1e22`. | resid 0.043 → 1.03 |
| **1907.05475** (L136-139) | `x-min 1e-21 → 2.10e-33`, `x-max 1e-08 → 8.09e-18` (~12 decades) | Same: a self-consistent but wrong pixel↔value pairing. Spot-check then ran at 6.31e-26 eV vs the correct 3.00e-12 eV. | resid 0.073 → 1.065 |
| **2212.01139** (L156-159) | `y-min 1e-10 → 2.26e-11` (single mislocated **bottom** tick, ~4.4× = 0.64 dex) | `detect_axis_ticks` found the bottom *frame* or a minor tick as the lowest "tick"; one mislocated tick re-anchored the whole y-fit, re-routing the traced curve ~1 dex off GT. **This is the sub-1.0-dex case** — P0's R2 alone may not catch it; it needs the OCR corroboration P1 supplies. | resid 1.262 → 2.281 |
| **1907.11485** (L141-144) | `x-max 35 → 115.6`, `y-max 1e-40 → 7.07e-32`; then "Using 388 CV-traced curve points" | **Wrong panel.** The Stage-2a LLM axes belonged to panel A (`x=[3,35] GeV`, a cross-section panel); CV calibrated against that mis-chosen panel and re-traced, collapsing the coupling to a flat `3.3e-47` (31 decades off the correct `2.9e-16`). Master had ignored the bad panel and anchored to Table I. | resid 2.29 → ZO |

Two reference points bound the thresholds from the *other* side:

- **2402.12892** (L229-232, **genuine win**): CV correctly read `x-max 30 → 124.29`
  (0.62 dex) on a zoom-inset panel; the spot-check then snapped to identity
  (raw_ratio 0.83), resid 0.853 → 0.102. **P1 must NOT revert this** — proof the
  override floor must sit *above* 0.62 dex *with corroboration*.
- **1207.3275** (L209-212, **genuine win**): CV read linear-printed `[-10,0]` tick
  labels and correctly interpreted them as `1e-10..1e0` decades, recovering the
  full V-exclusion (resid 3.403 → 2.134). **P1's OCR must agree with this**, not
  fight it.

### 1.3 What P1 fixes that P0 alone does not

P0's **R2** reverts a CV axis override that disagrees with the LLM by `> 1.0 dex`
*without corroboration*. That clears the gross blow-ups (1506.08082 24 dex,
1907.05475 12 dex) **as a backstop**, but:

1. **P0 has no positive corroboration signal pre-P1.** Its only proxy is
   `r2 ≥ 0.95` + spot-check (P0 §2.2 check 3, §3.2), and `r2` is exactly the gate
   the class-A failures already pass. So P0 *reverts everything > 1.0 dex* — which
   would also revert a *legitimate* large correction. P1 supplies the real "is the
   CV axis right?" signal so large-but-correct overrides (e.g. a genuinely mislabeled
   LLM axis) can still commit.
2. **2212.01139 is sub-1.0-dex (0.64 dex)** so R2 lets it through; only an
   independent OCR read of the bottom tick label catches that the lowest detected
   "tick" is not a `1e-10` label. P1 is the *only* phase that fixes 2212.01139.
3. **1907.11485 is a wrong-panel error**, not a within-panel fit error. P1 adds a
   panel-consistency check (OCR'd labels must be physically plausible for the
   coupling type) that demotes the wrong panel before it ever calibrates.

So P1 turns axis calibration from "LLM tells us the values, CV measures pixels,
trust the fit" into **"OCR reads the labels from pixels, geometry fits them, and the
two must agree with each other AND only then are they allowed to overrule the LLM."**

---

## 2. Design

### 2.1 Core idea: a third, independent channel — OCR'd tick labels

Today there are two channels (LLM tick *values*, CV tick *pixels*). P1 adds a
**third independent channel: OCR-read tick *labels* from the figure pixels.** The
override decision becomes a **three-way agreement test**:

```
LLM tick values        ──┐
OCR'd tick labels      ──┼──►  fuse into (pixel, value) pairs, fit, cross-validate
CV tick pixel positions──┘
```

An axis override is allowed **only when OCR and geometry agree with each other**
(the measurement is self-consistent) **and they contradict the LLM** (the LLM was
the one that was wrong). If OCR and geometry *disagree*, the measurement is
untrustworthy → keep the LLM axis. If all three agree → keep the LLM axis (no
override needed). This is the gate #550 Phase 1 named ("override only when OCR +
geometry agree and disagree with the LLM") and the integration never implemented.

### 2.2 Stack & licenses (concrete)

| Component | Library | Version pin | License | Why |
|---|---|---|---|---|
| OCR engine | **Tesseract** via **pytesseract** | `pytesseract>=0.3.10`; system `tesseract>=4.1` (LSTM) | Tesseract **Apache-2.0**; pytesseract **Apache-2.0** | Pure-text OCR of *short numeric* tick labels (`10⁻¹⁵`, `0.1`, `30`). We do **not** need a chart-aware model — labels are isolated, high-contrast crops. Apache-2.0 matches the repo and P0's no-copyleft constraint. |
| Tick / plot-rect detection | **OpenCV** (already a dep) | existing pin | **Apache-2.0** | `detect_plot_region` / `detect_axis_ticks` already exist (plot_calibration.py:185-341); P1 reuses them and adds **label-crop bbox** emission. |
| Numeric parse | stdlib `re` + `unicodedata` | — | PSF | Normalize superscripts/×10ⁿ to floats (§2.4). |

**Why Tesseract over PaddleOCR.** PaddleOCR (Apache-2.0) has stronger scene-text
recall but pulls **PaddlePaddle** (large, frequently CUDA-coupled, heavier wheel
matrix) — disproportionate for reading ~6 short numeric labels per axis, and it
widens the Actions/CLI runtime. Tesseract is a single system package
(`apt-get install tesseract-ocr`), CPU-only, deterministic, and the labels are the
*easy* OCR case (printed, axis-aligned, high contrast). **Decision: Tesseract.**
PaddleOCR is the documented fallback if eval shows Tesseract recall < ~70% on the
figure subset (open question §5). Both are Apache-2.0, so the licensing decision is
free either way.

**Optional-dependency discipline.** Mirror `plot_calibration.py`'s existing pattern
(try/except import at module top, `_OCR_AVAILABLE` flag, every function degrades to
`None`). If `pytesseract`/`tesseract` is absent, P1 emits `corroborated=False` for
**every** axis override, so the result is identical to "P0 with no corroboration"
(i.e. R2 reverts all > 1.0 dex overrides). **P1 never regresses below P0.**

### 2.3 New module surface

**New file `pipeline/axis_ocr.py`** (OCR + numeric-label parsing; Apache-2.0 deps):

```python
@dataclass(frozen=True)
class TickLabel:
    pixel: float        # tick centre pixel (from detect_axis_ticks)
    value: float        # parsed numeric value from OCR
    raw_text: str       # raw OCR string (for the note / debugging)
    ocr_conf: float     # tesseract per-label confidence 0..1

def ocr_axis_labels(panel_img, bbox, axis: str,
                    tick_pixels: list[int]) -> list[TickLabel]:
    """Crop a label band beside each detected tick, OCR it, parse to float.
    Returns only labels that parse to a finite positive number with ocr_conf
    above the floor. Never raises; [] on any failure."""
```

**New function in `pipeline/plot_calibration.py`**:

```python
def calibrate_axis_ocr(panel_img, bbox, axis, scale,
                       tick_pixels, llm_values) -> AxisCalibration | None:
    """OCR the tick labels, fit pixel->value from the OCR'd (pixel,value)
    pairs, and cross-validate against (a) the geometric residual and
    (b) the LLM-supplied values. Returns an AxisCalibration carrying the
    transform AND the corroboration verdict, or None if OCR found < 2
    parseable labels (no independent channel -> fall back to current path)."""
```

```python
@dataclass(frozen=True)
class AxisCalibration:
    transform: Callable[[float], float]   # pixel -> data (validated fit)
    ocr_geom_agree: bool       # OCR labels vs geometric fit: median |Δ| <= tol
    ocr_vs_llm_dex: float      # |log10(OCR-implied endpoint) - log10(LLM)|
    n_labels: int              # parseable OCR labels used
    fit_r2: float
    note: str
    @property
    def corroborated(self) -> bool: ...   # §2.6 — the object P0.R2 consumes
```

`AxisCalibration` is the **corroboration object** P0 §2.2/§3.2 reserved a slot for:
P0's `guard_transform(... corroborated=axis_cal.corroborated ...)` consumes
`.corroborated` directly. P1 changes the *meaning* of "corroborated" from P0's weak
`r2≥0.95` proxy to "OCR and geometry agree."

### 2.4 Numeric label parsing (§2.3 `ocr_axis_labels`)

Physics tick labels come in three printed forms; the parser must handle all and
**reproduce the 1207.3275 win** (linear `[-10..0]` decade exponents):

1. **Power-of-ten with superscript**: `10⁻¹⁵`, `10^{-15}`, `10-15` → `1e-15`.
   Regex on `10` followed by a (possibly Unicode-superscript) signed integer;
   `unicodedata.normalize` + a superscript→ASCII table maps `⁻¹⁵`→`-15`.
2. **Mantissa×10ⁿ**: `2×10⁻⁶`, `3.0e-9`, `2*10^-6` → `2e-6`.
3. **Plain decimal / integer**: `0.1`, `30`, `500` → as-is.
4. **Bare exponent on a labeled-decade axis** (the 1207.3275 case): when the
   x_axis is *log* but the tick *labels* are bare integers `-10,-9,…,0` whose count
   and order match decade ticks, interpret each as `10**k`. Guard: only when
   `x_axis_scale == "log"` AND the bare integers are consecutive-ish AND the LLM
   range is decades. (This mirrors what the LLM did correctly for 1207.3275; P1
   does it in OCR so it is *measured*, not eyeballed.)

The label band is a thin rectangle offset from `bbox` along the axis: for x, below
`y1` spanning `[tick_px - Δ, tick_px + Δ]`; for y, left of `x0`. Crop size is
derived from the median tick spacing so adjacent labels don't bleed in. Each crop is
OCR'd in tesseract **PSM 7** (single text line) with a numeric+`×10⁻^.eE` char
allowlist to suppress garbage.

Discard a `TickLabel` if `ocr_conf < 0.50` or the string doesn't parse to a finite
positive number. **At least 2 surviving labels per axis** are required to build the
OCR channel; otherwise `calibrate_axis_ocr` returns `None` and the existing
LLM-value path runs unchanged (with `corroborated=False`).

### 2.5 Robust geometric fit (replaces blind common-prefix pairing)

The current `n = min(len(pix), len(vals))` prefix pairing (extractor.py:1343-1347)
is where mispairing enters. P1 fixes it two ways:

1. **Pair by position, not index.** OCR gives `(pixel, value)` pairs *directly* (the
   label is read at the tick it sits under), so there is **no index-alignment step**
   to corrupt — pixel and value come from the same physical tick. This alone removes
   the 1506.08082 / 1907.05475 mispairing.
2. **RANSAC-style robust line fit** over the OCR pairs (replacing the plain
   `lstsq` at plot_calibration.py:417): fit `log10(value)=a·pixel+b` on all
   inlier subsets, keep the consensus with the most inliers at residual
   `< 0.15 dex`, require `>= 2` inliers and `>= 60%` of labels as inliers. A single
   mislocated/misread tick (2212.01139's bottom tick) becomes an *outlier* and is
   dropped, instead of dragging the whole fit. Reuse `build_log_transform`'s
   `r2 >= 0.95` on the **inlier set** as a final guard.

### 2.6 The validation gate (`AxisCalibration.corroborated`) — concrete thresholds

`calibrate_axis_ocr` computes three quantities and sets `corroborated` per the
table. **All thresholds in dex (log10); chosen to sit between the §1.2 wins and
failures.**

| Quantity | Definition | Threshold | Evidence anchor |
|---|---|---|---|
| `ocr_geom_agree` | median over labels of `\|log10(value_i) − log10(transform(pixel_i))\|` | `<= 0.15 dex` | A self-consistent OCR+geometry fit; 0.15 dex is well below one decade and above sub-pixel jitter on a multi-decade axis. Fails on 1506.08082/1907.05475 where the *implied* labels can't all sit on one line. |
| `ocr_vs_llm_dex` | `max` over {min,max} endpoints of `\|log10(OCR-endpoint) − log10(LLM-endpoint)\|` | "contradicts LLM" if `> 0.5`; "agrees" if `<= 0.5` | The existing override trigger (extractor.py:1370) is 0.5 dex — P1 reuses it as the *agree/contradict boundary*, not as the commit trigger. |
| `endpoint_phys_ok` | each OCR'd endpoint, mapped to data, lands inside the **widened** `VALID_RANGES[ct]` (`×0.1/×10`) for the axis | hard `bool` | Rejects 1506.08082's `1e22 GeV⁻¹` (AxionPhoton coupling hi `1e-3`) and 1907.11485's `7.07e-32`/`3.3e-47` *before* they corroborate anything. |

**`corroborated` is True iff** `ocr_geom_agree` **and** `n_labels >= 2` **and**
`fit_r2 >= 0.95` **and** `endpoint_phys_ok`. The **override decision** is then
(this is what plugs into P0.R2):

```
if not OCR available / n_labels < 2:        corroborated = False  (→ P0.R2 governs alone)
elif ocr_geom_agree and ocr_vs_llm_dex<=0.5: keep LLM axis  (all three agree; no-op)  [2402.12892? no — see below]
elif ocr_geom_agree and endpoint_phys_ok:    OVERRIDE with OCR axis, corroborated=True
                                             (OCR+geom agree, contradict LLM → LLM was wrong)
else:                                        keep LLM axis, corroborated=False
                                             (OCR & geometry disagree → untrustworthy)
```

Note 2402.12892 (`x 30→124`, 0.62 dex) **is** an override-and-commit: OCR reads the
zoom-inset's real x-max tick as `124`-ish, geometry agrees, the LLM eyeballed `30`,
so `ocr_vs_llm_dex≈0.62 > 0.5` ⇒ the second branch's "contradict" arm fires ⇒
`corroborated=True` ⇒ **P0.R2 commits** (its `0.5 < d <= 1.0 ⇒ override iff
corroborated` arm). For 1506.08082/1907.05475 the OCR labels won't fit one line with
the absurd endpoints ⇒ `ocr_geom_agree=False` ⇒ `corroborated=False` ⇒ P0.R2 sees a
`>1.0 dex` *uncorroborated* override ⇒ **revert to LLM axis.**

### 2.7 Panel-consistency check (fixes 1907.11485 wrong-panel)

Before calibrating, P1 sanity-checks the LLM's `panel_index` against OCR. After
`split_panels` (extractor.py:1299) and `detect_axis_ticks`, run `ocr_axis_labels`
on the chosen panel. If the OCR'd axis endpoints, mapped through the coupling type,
land **outside widened `VALID_RANGES`** (e.g. a cross-section panel reading `1e-40
cm²` for a kinetic-mixing axis), mark the panel **implausible** and:
- if another panel's OCR'd labels are physically plausible, prefer it;
- else emit `corroborated=False` and **do not override** — the LLM-Table-I-anchored
  boundary (master's correct path for 1907.11485) survives.

This is the cheap, OCR-grounded version of "select the right panel" — it does not
need a panel classifier, just a physical-plausibility veto on OCR'd endpoints.

### 2.8 Data flow (P1 on top of P0)

```
_run_stage2a_axes
  └─ _attach_cv_calibration(axis_info, figure_paths)
       split_panels → detect_plot_region → detect_axis_ticks   [unchanged]
       └─ NEW: calibrate_axis_ocr(panel, bbox, axis, scale, tick_pixels, llm_vals)
            ├─ ocr_axis_labels  → [TickLabel(pixel,value,conf)]   (axis_ocr.py)
            ├─ robust fit (RANSAC inliers, §2.5)
            ├─ panel-consistency veto (§2.7)
            └─ → AxisCalibration{transform, corroborated, ocr_vs_llm_dex, note}
       └─ guard_transform(                                    [P0, extractor.py:1370]
              before = LLM axis endpoints,
              after  = OCR/CV axis endpoints,
              score.axis_disagree_dex = ocr_vs_llm_dex,
              corroborated = axis_cal.corroborated)           ← P0.R2 commits-or-reverts
            → commit OCR axis ONLY when corroborated (or <=0.5 dex); else KEEP LLM
_attempt_cv_curve_trace uses the COMMITTED transform           [P0 R4/R5/quality gate]
```

Every override still passes through **P0's `guard_transform`**; P1 only makes the
`corroborated` flag *true measurement* instead of a proxy. P1 changes **no commit
logic** — that lives in P0 — so the two phases compose cleanly and either can land
first (P1 degrades to P0 if OCR is unavailable).

---

## 3. Integration points (exact files / functions)

All edits in the `AAL-integration` worktree (carries #550 + #561 + P0).

### 3.1 New `pipeline/axis_ocr.py`
- `ocr_axis_labels`, `TickLabel`, the numeric-label parser (§2.4). Optional-import
  guard mirroring `plot_calibration.py:30-41` (`_OCR_AVAILABLE`).

### 3.2 `pipeline/plot_calibration.py`
- **`detect_axis_ticks` (:253-341)**: additionally return, per tick, the
  **label-crop bbox** (the band beside the tick) so `ocr_axis_labels` knows where
  to OCR. Backward-compatible: add a key, keep `{"x":[...],"y":[...]}`.
- **New `calibrate_axis_ocr` + `AxisCalibration`** (§2.3/§2.6): the robust fit
  (§2.5) replacing the blind `lstsq`/common-prefix pairing.
- `build_log_transform` (:348-441) is reused as the final inlier-set guard
  (`r2>=0.95`).

### 3.3 `pipeline/extractor.py`
- **`_calibrate_axis` (:1324-1375)**: replace the `min(len(pix),len(vals))`
  prefix-pairing + unconditional `disagree>0.5 → overwrite` (:1370-1372) with a call
  to `calibrate_axis_ocr(...)`. The function now returns
  `(transform, AxisCalibration)`; it **no longer mutates `axis_info[min_key]`
  directly** — it hands the proposed override + `corroborated` to P0's
  `guard_transform`, which decides commit-or-revert (P0 §3.2). This is the exact
  line region the 2212.01139 / 1506.08082 / 1907.05475 / 1907.11485 fixes touch.
- **`_attach_cv_calibration` (:1256-1396)**: store `AxisCalibration` on `axis_info`
  (`_cv_axis_cal_x` / `_cv_axis_cal_y`) so `_attempt_cv_curve_trace` and the
  override gate can read `corroborated`. Extend `cv_calibration_note` to record the
  OCR verdict (`ocr=2 labels, geom_agree=Y, vs_llm=0.62dex, corroborated=Y`) —
  these strings are what `per_paper_findings` greps, so make them explicit.
- **`_attempt_cv_curve_trace` (:1398-1426)** + **override gate (:1031)**: unchanged
  in P1; they consume the *committed* transform. (The trace-quality gate is P0's
  R4/R5/quality tuple.) P1 only improves the *transform* feeding them.

### 3.4 Composition with the other phases
- **P0** owns commit logic; P1 supplies `corroborated`. P0's R2 thresholds (0.5/1.0
  dex) are unchanged — P1 just makes the 0.5–1.0 dex band *trustworthily*
  corroborated and adds the sub-0.5/over-1.0 reasoning. P1 cannot regress below P0
  (OCR-absent ⇒ `corroborated=False` ⇒ identical to P0).
- **P2** (selector): a CV-trace candidate built on a *corroborated* axis gets a
  higher `corroborated` flag in P2's `quality()` tuple (P0 §2.5), so P1 strengthens
  P2's routing without P2 needing to know about OCR.
- **P3** (read determinism): P1's OCR axis read is deterministic (Tesseract is a
  pure function of pixels), so it *replaces* the nondeterministic LLM tick-value
  read for the axis — directly damping the axis half of the 1403.1290-style drift
  P3 targets. P3 still handles the *curve/coupling* read and the convention
  normalizer.
- **P4** (CI gate): adds an assert that no committed axis override has
  `corroborated=False` (P1 invariant), alongside P0's R5/no-crash asserts.

### 3.5 Packaging note
Default: in-process module (`axis_ocr.py`), matching `plot_calibration.py`. Promote
the calibrator+OCR to the standalone `plot-digitizer` **MCP** only if the Tesseract
system dep proves awkward in Actions (it installs cleanly via `apt`, so MCP is not
expected to be needed for P1; revisit at P3 if the chart-model fallback lands).

---

## 4. Acceptance criteria (measured on `subset_eval.py`)

Run on the `AAL-integration` branch **with P0 already applied** (P1 is measured *on
top of* P0, since P0 is the commit layer):

```
python -m evaluation.subset_eval extract --key union \
    --outdir evaluation/eval_runs/after_p1
python -m evaluation.subset_eval compare \
    --before evaluation/eval_runs/after_p0 \
    --after  evaluation/eval_runs/after_p1 \
    --out evaluation/eval_runs/comparison_p1.md
```

P1 is accepted iff, **vs the P0 snapshot** (no regression) **and the master
baseline** (figure improvement):

1. **The 4 class-A CV-miscalib papers recover.** `1506.08082` resid back to
   `<= 0.10 dex` (from P0's reverted-to-LLM value; ideally master's 0.043);
   `1907.05475` `<= 0.10 dex`; `2212.01139` `<= 1.30 dex` (recover the 0.64-dex
   y-min that **only OCR catches** — P0 alone leaves it at ~2.28); `1907.11485`
   back to **compared** (not zero-overlap) via the panel veto.
2. **The 2 genuine axis wins are preserved.** `2402.12892` resid `<= 0.12 dex`
   (the 0.62-dex override still commits via corroboration); `1207.3275` resid
   `<= 2.20 dex` (OCR reproduces the linear-decade-label read).
3. **figure_vision per-source row improves over master.** Median residual
   `<= 0.55 dex` (master 0.642; P0 target `<= 0.642`) **and** `<= 0.3 dex` fraction
   `>= 40%` (master 33.3%). No regression on `table`/`text` rows (P1 only touches the
   figure axis path).
4. **No new zero-overlap from P1.** `wrong_window` count attributable to axis
   override `<= ` the P0 snapshot; 0 new `unit_offset`.
5. **Every committed axis override is OCR-corroborated.** Assert over `after_p1`
   notes: 0 papers where the axis was overwritten with `corroborated=False`.
6. **OCR-absent fallback is byte-identical to P0.** Re-run a 5-paper subset with
   `tesseract` uninstalled; the resulting JSONs must equal the `after_p0` JSONs
   (P1 must never regress below P0).
7. **Determinism unchanged/better.** `--repeats 3` on the `cv_miscalib` key: mean
   coupling-scale std `<=` P0's **0.230 dex** (OCR axis read is deterministic;
   replacing the LLM tick-value read should not *add* spread and may reduce it).

Stretch (informational): figure_vision `<= 0.3 dex` fraction trending toward the
`text` row (45–48%); OCR label recall `>= 70%` on the figure subset (drives the
Tesseract-vs-PaddleOCR open question).

---

## 5. Risks & open questions

- **Tesseract recall on dense/rotated/small labels.** Y-axis labels are often
  rotated or tiny. Mitigations: PSM 7 + numeric allowlist; upscale crops 3–4×
  before OCR; rotate y-crops 90°. **Open:** if recall `< 70%` on the figure subset,
  switch to PaddleOCR (Apache-2.0, already scoped as the fallback). Decide
  empirically on the eval, not by argument.
- **2212.01139 hinges entirely on OCR catching one mislocated tick.** If OCR can't
  read that bottom label (small/faint), the RANSAC outlier-drop (§2.5) is the
  backstop (the mislocated tick becomes an inlier-set outlier), but it then leaves
  `< 2` confident labels ⇒ `corroborated=False` ⇒ P0 reverts to LLM axis ⇒ resid
  stays ~1.26 (master's value), *not* worse. So the **worst case for 2212.01139 is
  "no improvement," never a regression** — acceptable.
- **The "agree" no-op vs override boundary at 0.5 dex.** Borrowed from the existing
  trigger; if a genuine win sits at exactly ~0.5 dex it could be treated as
  "agree/no-op." None of the 5 wins do (smallest override is 2402.12892 at 0.62),
  but P4's CI gate keeps this honest as the subset grows.
- **Panel veto false-positives.** A legitimately exotic axis (e.g. AxionMass
  `1e-12..1e18`) could trip the widened-`VALID_RANGES` plausibility veto. The veto
  uses the *widened* (`×0.1/×10`) window and only *demotes* a panel when an
  alternative is plausible (§2.7), so the failure mode is "keep LLM axis," not a
  hard reject. Verify no genuine-win panel is vetoed (2402.12892's inset, 1207.3275).
- **System dependency in Actions.** `tesseract-ocr` must be installed in the
  workflow image (`apt-get install -y tesseract-ocr`). Document in
  `requirements_pipeline.txt` comments + the workflow. If this proves brittle, the
  MCP packaging (§3.5) isolates it; not expected to be needed.
- **Threshold N is small.** `0.15 dex` geom-agree and the `0.5/1.0` override bands
  are calibrated on ~6 papers. P4's CI gate is the long-term guard; P1 lands *with*
  the eval re-run, not on argument.

---

## 6. Draft sub-issue body

> **Title:** `[extractor][P1] Trustworthy axis metrology: OCR'd tick labels + geometric fit, cross-validated against the LLM axis`

**Parent:** #566 (governs). == #550 Phase 1, with the validation #550 assumed.
**Depends on:** #566-P0 (fail-safe contract) — P0 is the commit layer; P1 supplies
the corroboration signal P0's R2 consumes.

### Why
`_attach_cv_calibration`/`_calibrate_axis` (extractor.py:1256-1396) trust the
**LLM's tick VALUES** and let OpenCV measure only **pixel positions**; the pairing is
aligned by index with the only sanity check being `build_log_transform`'s
`r2>=0.95` (plot_calibration.py:427). A mispaired or mislocated tick fits a line
just as well, so CV overwrote good LLM axes by many decades on 7 papers
(`per_paper_findings.md` class A):

- **1506.08082**: `y-max 2e-6 → 1e22 GeV⁻¹` (24 dex, unphysical). resid 0.043→1.03.
- **1907.05475**: `x 1e-21→2.1e-33` (12 dex); spot-check ran at 6.3e-26 eV. resid 0.073→1.065.
- **2212.01139**: single mislocated **bottom** tick `y-min 1e-10→2.26e-11` (0.64 dex)
  re-routed the curve ~1 dex off. resid 1.262→2.281. **Sub-1.0-dex ⇒ P0.R2 can't
  catch it; only an independent OCR read can.**
- **1907.11485**: **wrong panel** — calibrated panel-A axes, collapsed coupling to
  `3.3e-47` (31 dex off). resid 2.29→ZO.

P0's R2 (revert > 1.0 dex uncorroborated) backstops the gross blow-ups but (a) has no
real corroboration signal (its proxy `r2` is the gate the failures already pass),
(b) misses the sub-1.0-dex 2212.01139, and (c) doesn't address wrong-panel. P1 adds
the missing signal.

### What
Add an **independent third channel — OCR'd tick labels** — and override the LLM axis
**only when OCR + geometry agree with each other AND contradict the LLM**.

- **Stack:** **Tesseract** via `pytesseract>=0.3.10` (Apache-2.0) for short numeric
  tick labels; **OpenCV** (existing, Apache-2.0) for plot-rect/tick detection.
  Chosen over PaddleOCR (heavier PaddlePaddle dep) — labels are the easy OCR case;
  PaddleOCR is the documented fallback if recall < 70%. Optional-import discipline:
  OCR-absent ⇒ `corroborated=False` ⇒ behaviour identical to P0 (never regresses).
- **New `pipeline/axis_ocr.py`**: `ocr_axis_labels` + numeric-label parser handling
  `10⁻¹⁵` / `2×10⁻⁶` / `0.1` / bare decade-exponents (reproduces the 1207.3275 win).
- **New `plot_calibration.calibrate_axis_ocr` + `AxisCalibration`**: OCR gives
  `(pixel,value)` pairs *directly* (no index-alignment to corrupt), fit via
  **RANSAC-style robust line** (drops a single mislocated tick — fixes 2212.01139),
  guard with `r2>=0.95` on inliers.
- **Validation gate** (`AxisCalibration.corroborated`, dex thresholds calibrated
  from evidence):
  - `ocr_geom_agree`: median `|log10(value)−log10(fit(pixel))| <= 0.15 dex`.
  - `ocr_vs_llm_dex`: agree if `<=0.5`, contradict if `>0.5` (reuses the existing
    0.5-dex trigger as the agree/contradict boundary).
  - `endpoint_phys_ok`: OCR'd endpoints inside widened `VALID_RANGES[ct]`
    (rejects 1506.08082's `1e22`, 1907.11485's `7e-32` before they corroborate).
  - **corroborated** ⇔ geom-agree ∧ `n_labels>=2` ∧ `r2>=0.95` ∧ phys-ok.
- **Panel veto** (fixes 1907.11485): OCR a panel's endpoints; if physically
  implausible for the coupling type, prefer another panel or keep the LLM/Table-I
  boundary — no panel classifier needed.
- **Composition:** P1 changes **no commit logic** — it feeds `corroborated` into
  P0's `guard_transform(... corroborated=axis_cal.corroborated)` (extractor.py:1370
  region). 2402.12892 (0.62-dex override) still commits because it's corroborated;
  1506.08082/1907.05475 revert because OCR+geometry can't agree on the absurd
  endpoints.

### Acceptance (on `evaluation/subset_eval.py`, vs P0 snapshot and master)
1. `1506.08082` resid `<=0.10`; `1907.05475` `<=0.10`; `2212.01139` `<=1.30`
   (the OCR-only fix); `1907.11485` back to **compared**.
2. Genuine wins preserved: `2402.12892` `<=0.12`; `1207.3275` `<=2.20`.
3. figure_vision row: median resid `<=0.55 dex`, `<=0.3 dex` fraction `>=40%`; no
   regression on `table`/`text`.
4. 0 new `unit_offset`; `wrong_window` `<=` P0 snapshot.
5. 0 committed axis overrides with `corroborated=False` (P1 invariant).
6. OCR-absent run is byte-identical to the P0 snapshot (never regresses below P0).
7. `--repeats 3` mean coupling-scale std `<=` P0's **0.230 dex** (OCR is
   deterministic).

### Non-goals
No new commit logic (P0). No curve tracer / selector (P2). No read-layer
determinism / convention normalizer (P3). No CI gate (P4). No chart-extraction
model (deferred to #550 Phase 3 / P3 fallback).

### Artifacts
`evaluation/eval_runs/per_paper_findings.md`,
`evaluation/eval_runs/comparison.md`,
`evaluation/eval_runs/roadmap_design/P0_failsafe_contract.md`,
`evaluation/eval_runs/roadmap_design/P1_axis_metrology.md`.
