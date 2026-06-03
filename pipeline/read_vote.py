"""N-sample read voting for extraction determinism (Follow-up B of #566).

#580 had to stop sending ``temperature`` (Opus 4.8 deprecated it), so reads no
longer decode at temperature 0 — they drift run-to-run far more than the
extraction-regression gate's noise floor (a no-op repeat pair fails G1/G4/G5).
This module denoises by running a read N times and selecting the **consensus**:

* coupling type — majority vote across the N samples;
* data points (the exclusion curve) — the **medoid** of the same-coupling-type
  curves: the one with the smallest median pairwise distance to the others, so a
  single drifted/outlier sample is rejected rather than averaged into a
  non-physical synthetic curve.

Pure + dependency-light (numpy only, already a CI dep); no anthropic. The caller
(``extractor.run_extraction_agent_voted``) runs the samples; this module just
selects among them, so it is fully unit-testable on synthetic point sets.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

try:  # numpy is a CI dep; degrade gracefully if ever absent
    import numpy as np
    _NP = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NP = False

# Number of log-spaced sample points across the overlapping mass range used to
# compare two curves.
_GRID_N = 25


def curve_distance(a_points: Sequence, b_points: Sequence) -> float:
    """Median |Δ log10(coupling)| between two curves over their overlapping mass
    range (log-log interpolation). ``inf`` if they share no mass overlap or are
    unusable. Symmetric, ``0`` for identical curves."""
    if not _NP or not a_points or not b_points:
        return float("inf")
    try:
        def _prep(pts):
            m = np.array([float(p[0]) for p in pts], dtype=float)
            g = np.array([float(p[1]) for p in pts], dtype=float)
            ok = np.isfinite(m) & np.isfinite(g) & (m > 0) & (g > 0)
            m, g = m[ok], g[ok]
            order = np.argsort(m)
            return np.log10(m[order]), np.log10(g[order])

        am, ag = _prep(a_points)
        bm, bg = _prep(b_points)
        if am.size < 2 or bm.size < 2:
            return float("inf")
        lo = max(am.min(), bm.min())
        hi = min(am.max(), bm.max())
        if not (hi > lo):
            return float("inf")
        grid = np.linspace(lo, hi, _GRID_N)
        ai = np.interp(grid, am, ag)
        bi = np.interp(grid, bm, bg)
        return float(np.median(np.abs(ai - bi)))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("curve_distance failed: %s", e)
        return float("inf")


def select_consensus(samples: list[tuple]) -> tuple[int, str]:
    """Pick the consensus sample index from N ``(coupling_type, data_points)``.

    1. **Coupling type** — majority vote over samples that named one.
    2. **Curve** — among samples whose coupling type matches the majority AND that
       have >= 2 points, return the **medoid** (min median pairwise
       :func:`curve_distance`). With no such curve, fall back to the
       majority-coupling sample with the most points, else sample 0.

    Returns ``(index, note)``; never raises on a non-empty list.
    """
    if not samples:
        return 0, "no samples"
    n = len(samples)

    cts = [ct for ct, _ in samples if ct]
    modal_ct = Counter(cts).most_common(1)[0][0] if cts else None
    ct_agree = sum(1 for ct, _ in samples if ct == modal_ct)

    cand = [i for i, (ct, pts) in enumerate(samples)
            if ct == modal_ct and pts and len(pts) >= 2]

    if not cand:
        # No multi-point curve at the modal coupling: take the modal-ct sample
        # with the most points (or sample 0).
        modal_idx = [i for i, (ct, _) in enumerate(samples) if ct == modal_ct]
        pool = modal_idx or list(range(n))
        best = max(pool, key=lambda i: len(samples[i][1] or []))
        return best, f"ct={modal_ct} ({ct_agree}/{n}); no consensus curve, took most-points"

    if len(cand) == 1:
        return cand[0], f"ct={modal_ct} ({ct_agree}/{n}); 1 curve"

    best_i, best_score = cand[0], float("inf")
    for i in cand:
        dists = [curve_distance(samples[i][1], samples[j][1]) for j in cand if j != i]
        finite = [d for d in dists if d != float("inf")]
        # median distance to the others; a sample with no finite neighbour is worst.
        score = float("inf") if not finite else float(np.median(finite)) if _NP else sum(finite) / len(finite)
        if score < best_score:
            best_i, best_score = i, score
    return best_i, (f"ct={modal_ct} ({ct_agree}/{n}); medoid {best_i} of "
                    f"{len(cand)} curves (med pairwise {best_score:.3f} dex)")
