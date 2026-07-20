"""Best-match multi-GT curve selection (#739).

A paper can carry several distinct same-coupling-type GT curves (2008.02209:
AFM / Coulomb / Plimpton-Lawton hidden-photon limits). The extraction targets
exactly ONE of them, so it must be graded against the curve it actually matches
(lowest residual), not the richest-by-points — which is arbitrary and produced a
false 3.14 dex catastrophic when the extraction matched Plimpton-Lawton at
0.05 dex.

These pin the ``_score_candidate`` + best-match selection directly, without a
full metrics run.

Run:
    python -m pytest evaluation/tests/test_best_match_multi_gt.py -v
"""

from __future__ import annotations

import numpy as np

from evaluation.evaluate import _score_candidate


class _Entry:
    """Minimal GT entry stub: only the fields _maybe_canonicalize / _score touch."""
    def __init__(self, gt, repo_file):
        self._gt = np.array(gt, dtype=float)
        self.reference_repo_file = repo_file
        self.coupling_convention = None  # None => no convention gate (raw compare)
        self.coupling_type = "DarkPhoton"

    def load_data(self):
        return self._gt


def _line(x0, x1, y, n=20):
    xs = np.logspace(np.log10(x0), np.log10(x1), n)
    return [[x, y] for x in xs]


def test_best_match_picks_the_curve_the_extraction_matches():
    # extraction sits on top of the "near" curve at y=1e-1
    ext = np.array(_line(1e-3, 1e0, 1e-1), dtype=float)
    near = _Entry(_line(1e-3, 1e0, 1.0e-1), "near.txt")   # ~0 dex
    far = _Entry(_line(1e-3, 1e0, 1.0e-4), "far.txt")     # ~3 dex, but "richest"

    candidates = [far, near]  # deliberately far (wrong) first, as richest-order might put it
    scored = [_score_candidate("x", {"data_points": ext.tolist()},
                               "DarkPhoton", ext, e, e.load_data()) for e in candidates]
    finite = [s for s in scored if s[0] == "compared" and s[2] is not None
              and np.isfinite(s[2].median_residual_dex)]
    best = min(finite, key=lambda s: s[2].median_residual_dex)
    # best match is the near curve, ~0 dex — NOT the far one
    assert best[5].reference_repo_file == "near.txt"
    assert best[2].median_residual_dex < 0.1


def test_single_candidate_is_unchanged():
    ext = np.array(_line(1e-3, 1e0, 1e-1), dtype=float)
    only = _Entry(_line(1e-3, 1e0, 1.0e-1), "only.txt")
    status, scored_via, im, ec, gc, chosen = _score_candidate(
        "x", {"data_points": ext.tolist()}, "DarkPhoton", ext, only, only.load_data())
    assert status == "compared"
    assert chosen.reference_repo_file == "only.txt"
    assert im.median_residual_dex < 0.1
