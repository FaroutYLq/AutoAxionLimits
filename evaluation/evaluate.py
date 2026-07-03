"""Main evaluation script for the extraction pipeline.

Usage:
    # Populate ground-truth data files from repo (one-time setup)
    python -m evaluation.evaluate --populate

    # Run extraction on all ground-truth papers (calls Claude API)
    python -m evaluation.evaluate --extract

    # Run extraction on a single paper
    python -m evaluation.evaluate --extract --arxiv-id 2208.03183

    # Compute metrics from cached results (no API calls)
    python -m evaluation.evaluate --metrics

    # Full pipeline: extract + metrics + report
    python -m evaluation.evaluate --extract --metrics --report

    # Generate report from cached results only
    python -m evaluation.evaluate --metrics --report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.conventions import (
    GUARD_REFUSED,
    UNCONVERTIBLE,
    canonical_convention,
    classify_reported_convention,
    file_source_convention,
    to_canonical,
)
from evaluation.ground_truth import (
    GroundTruthEntry,
    load_ground_truth,
    populate_data_from_repo,
)
from evaluation.metrics import (
    NOISE_FLOOR_RESIDUAL_DEX,
    ClassificationMetrics,
    CurveMetrics,
    InterpolationMetrics,
    SymmetricCurveMetrics,
    _expand_mass_independent,
    compute_confidence_calibration,
    compute_curve_metrics,
    compute_interpolation_metrics,
    compute_symmetric_curve_metrics,
    single_point_compare,
)
from evaluation.report import generate_report

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# arXiv metadata-fetch robustness (issue #560). The arxiv library uses an
# internal requests.Session().get() with NO timeout, so when export.arxiv.org
# is slow/throttling (429/503) or its CDN stalls, the call hangs indefinitely
# and blocks the whole evaluation run. We bound each attempt with a hard
# wall-clock deadline and cap the number of attempts.
ARXIV_FETCH_TIMEOUT_S = 30  # per-attempt hard wall-clock deadline (seconds)
ARXIV_FETCH_RETRIES = 3     # number of attempts before falling back
# Per-request socket timeout injected into the arxiv client's session, kept
# below the per-attempt wall-clock deadline so the socket-level timeout fires
# first and the worker thread dies cleanly instead of leaking.
ARXIV_SOCKET_TIMEOUT_S = 20


def _install_session_timeout(client, timeout_s: float = ARXIV_SOCKET_TIMEOUT_S) -> None:
    """Force a default per-request timeout onto the arxiv client's session.

    The arxiv library calls ``self._session.get(url, headers=...)`` with no
    ``timeout``, so ``requests`` passes ``timeout=None`` and a stalled socket
    blocks forever. Wrapping ``Session.request`` to inject a default ``timeout``
    makes the underlying call *raise* instead of hang, so the worker thread
    actually terminates. Idempotent; a no-op if the client exposes no
    ``_session`` (e.g. a test stub).
    """
    session = getattr(client, "_session", None)
    if session is None or getattr(session, "_axionlimits_timeout_installed", False):
        return
    _orig_request = session.request

    def _request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_s)
        return _orig_request(method, url, **kwargs)

    session.request = _request_with_timeout
    session._axionlimits_timeout_installed = True


def _call_with_deadline(fn, timeout_s: float, label: str = "arXiv fetch"):
    """Run blocking ``fn()`` under a hard wall-clock deadline on a daemon thread.

    Daemon threads are never joined by the interpreter at exit, so a fetch that
    is still hung when the deadline fires cannot block process shutdown (the
    failure mode a ``ThreadPoolExecutor`` worker would cause, since
    ``_python_exit`` ``join()``s non-daemon workers unconditionally with no
    timeout). A ``requests`` socket timeout is normalized to ``TimeoutError`` so
    the socket-level and wall-clock deadlines surface identically to callers.
    """
    import threading

    box: dict = {}

    def _runner() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    worker = threading.Thread(target=_runner, name="arxiv-meta-fetch", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"{label} exceeded {timeout_s}s wall-clock deadline")
    err = box.get("error")
    if err is not None:
        try:
            import requests

            socket_timeout = requests.exceptions.Timeout
        except ImportError:
            socket_timeout = ()
        if isinstance(err, socket_timeout):
            raise TimeoutError(f"{label} hit socket timeout: {err}") from err
        raise err
    return box.get("result")


def _fetch_paper_metadata(arxiv_id: str, cache_path: Path) -> tuple[str, str]:
    """Fetch real title and abstract from arXiv API. Cache results."""
    import json as _json
    if cache_path.exists():
        with open(cache_path) as f:
            cache = _json.load(f)
    else:
        cache = {}
    if arxiv_id in cache:
        return cache[arxiv_id]["title"], cache[arxiv_id]["abstract"]
    # Fetch from arXiv. The metadata API (export.arxiv.org) is aggressively
    # rate-limited; a failure here is non-fatal — we fall back to the
    # ground-truth title and an empty abstract so extraction can proceed.
    #
    # The arxiv library does not expose a request timeout (it calls
    # requests.Session().get() with no timeout internally), so a single
    # attempt could hang forever. We bound each attempt on two axes (see
    # _install_session_timeout / _call_with_deadline): a socket timeout makes
    # the blocking get() raise so the worker thread dies, and a daemon-thread
    # wall-clock deadline guarantees the attempt returns within
    # ARXIV_FETCH_TIMEOUT_S even in the pathological case — and, being a daemon
    # thread, can never block process exit. On timeout/error we back off and
    # retry up to ARXIV_FETCH_RETRIES times, then degrade gracefully to the
    # ground-truth-title fallback below. The call can never exceed
    # ~ARXIV_FETCH_RETRIES * (ARXIV_FETCH_TIMEOUT_S + backoff) seconds.
    import arxiv as _arxiv

    def _do_fetch():
        client = _arxiv.Client()
        _install_session_timeout(client)
        search = _arxiv.Search(id_list=[arxiv_id])
        return next(client.results(search), None)

    result = None
    for attempt in range(ARXIV_FETCH_RETRIES):
        try:
            result = _call_with_deadline(
                _do_fetch,
                ARXIV_FETCH_TIMEOUT_S,
                label=f"arXiv metadata fetch for {arxiv_id}",
            )
            break
        except TimeoutError:
            wait = 5 * (2 ** attempt)
            logger.warning(
                "arXiv metadata fetch for %s timed out after %ds; retry in %ds",
                arxiv_id, ARXIV_FETCH_TIMEOUT_S, wait,
            )
            time.sleep(wait)
        except Exception as e:  # HTTP 429, parse errors, transient network
            wait = 5 * (2 ** attempt)
            logger.warning("arXiv metadata fetch failed for %s (%s); retry in %ds",
                           arxiv_id, e, wait)
            time.sleep(wait)
    if result:
        cache[arxiv_id] = {"title": result.title, "abstract": result.summary}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            _json.dump(cache, f, indent=2)
        return result.title, result.summary
    logger.warning("No arXiv metadata for %s; using ground-truth title fallback", arxiv_id)
    return "", ""


def _safe_id(arxiv_id: str) -> str:
    """Filesystem-safe key for an arXiv id (old-style ids contain '/')."""
    return arxiv_id.replace("/", "_")


def _load_cached_result(arxiv_id: str) -> dict | None:
    """Load a cached extraction result, if it exists."""
    path = RESULTS_DIR / f"{_safe_id(arxiv_id)}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_result(arxiv_id: str, result: dict):
    """Cache an extraction result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{_safe_id(arxiv_id)}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved result for %s", arxiv_id)


def run_extraction(entry: GroundTruthEntry) -> dict:
    """Run the pipeline extraction on a single paper.

    Returns a dict with ExtractionResult fields + timing info.
    """
    import anthropic

    from pipeline.extractor import (
        ExtractionResult,
        download_pdf,
        run_extraction_agent_voted,
    )

    if os.environ.get("AAL_BATCH", "").lower() in ("1", "true", "yes"):
        # Message-Batches transport (50% price): same agent code, calls ride
        # a shared batching dispatcher. See pipeline/batch_client.py.
        from pipeline.batch_client import get_shared_batching_client
        client = get_shared_batching_client()
    else:
        client = anthropic.Anthropic()

    # Create a minimal paper-like object for the extractor
    class _PaperStub:
        def __init__(self, arxiv_id: str, title: str, summary: str = "", categories: list = None):
            self.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
            self.title = title
            self.summary = summary
            self.categories = categories or []

        def get_short_id(self):
            return self.arxiv_id

    real_title, abstract = _fetch_paper_metadata(
        entry.arxiv_id, RESULTS_DIR / "metadata_cache.json"
    )
    title = real_title or entry.paper_title
    paper_stub = _PaperStub(entry.arxiv_id, title, summary=abstract)
    paper_stub.arxiv_id = entry.arxiv_id

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()

        try:
            pdf_path = download_pdf(entry.arxiv_id, Path(tmpdir))
        except Exception as e:
            logger.error("PDF download failed for %s: %s", entry.arxiv_id, e)
            return {
                "arxiv_id": entry.arxiv_id,
                "error": f"PDF download failed: {e}",
                "elapsed_s": time.time() - t0,
            }

        try:
            result: ExtractionResult = run_extraction_agent_voted(
                paper_stub, pdf_path, client
            )
        except Exception as e:
            logger.error("Extraction failed for %s: %s", entry.arxiv_id, e)
            return {
                "arxiv_id": entry.arxiv_id,
                "error": f"Extraction failed: {e}",
                "elapsed_s": time.time() - t0,
            }

        elapsed = time.time() - t0

    return {
        "arxiv_id": result.arxiv_id,
        "paper_title": result.paper_title,
        "coupling_type": result.coupling_type,
        "is_new_limit": result.is_new_limit,
        "is_projection": result.is_projection,
        "data_points": result.data_points,
        "data_source": result.data_source,
        "dm_density_assumed": result.dm_density_assumed,
        "confidence_level": result.confidence_level,
        "extraction_confidence": result.extraction_confidence,
        "suggested_experiment_name": result.suggested_experiment_name,
        "coupling_convention": result.coupling_convention,  # #536/#587 — needed by the comparator canonicalizer
        "notes": result.notes,
        "num_points": len(result.data_points),
        "elapsed_s": elapsed,
    }


def _normalize_predicted_coupling(raw_ct):
    """Normalize a predicted coupling type: handle lists, apply alias normalization."""
    from pipeline.reviewer import _normalize_coupling_type

    if raw_ct is None:
        return None
    # Handle list returns — take first element
    if isinstance(raw_ct, list):
        raw_ct = raw_ct[0] if raw_ct else None
    if raw_ct is None:
        return None
    # Try normalization via reviewer aliases
    try:
        return _normalize_coupling_type(raw_ct)
    except KeyError:
        return raw_ct  # keep raw if normalization fails


# Map a limit_data/<dir>/ basename to its canonical coupling type. The data
# file's directory is the authoritative physical coupling of a GT curve —
# more reliable than GroundTruthEntry.coupling_type, which is a placeholder for
# auto-expanded entries and is occasionally wrong for multi-coupling papers
# (e.g. a DarkPhoton-labelled entry pointing at an AxionElectron data file).
try:
    from pipeline.config import COUPLING_TYPES as _COUPLING_TYPES_REG
    _DIR_TO_COUPLING = {
        Path(meta["data_dir"]).name: key for key, meta in _COUPLING_TYPES_REG.items()
    }
except Exception:  # pragma: no cover - config import is best-effort
    _DIR_TO_COUPLING = {}
_DIR_TO_COUPLING.setdefault("VectorB-L", "VectorBL")
_DIR_TO_COUPLING["fa"] = "AxionMass"  # m_a vs f_a plane is classified AxionMass


def _authoritative_coupling(entry: GroundTruthEntry) -> str:
    """Physical coupling of an entry's GT *data file*, from its repo path."""
    ref = entry.reference_repo_file
    if ref:
        parts = Path(ref).parts
        if len(parts) >= 2 and parts[0] == "limit_data":
            return _DIR_TO_COUPLING.get(parts[1], entry.coupling_type)
    return entry.coupling_type


def _usable_gt_stats(gt_data, coupling_type: str) -> tuple[int, int]:
    """(n_points, n_unique_masses) of GT points that survive boundary-closure
    filtering. Interpolation needs >= 2 distinct masses; a curve with only one
    distinct mass (a single prediction or a single-mass projection) is a point
    reference, not a comparable curve."""
    from evaluation.metrics import _COUPLING_CEILINGS, _filter_boundary
    ceil = _COUPLING_CEILINGS.get(coupling_type, 1e-2)
    f = _filter_boundary(gt_data, ceil)
    if len(f) == 0:
        return 0, 0
    return len(f), int(np.unique(f[:, 0]).size)


def _is_placeholder_entry(entry: GroundTruthEntry) -> bool:
    """True if the entry's scalar labels (is_new_limit, is_projection,
    data_source_expected, difficulty) are auto-generated placeholders rather
    than human-verified values, and therefore cannot be scored against."""
    return ("auto_expanded" in (entry.tags or [])) or entry.verified_by == "repo_upstream"


# Below this many compared papers, a per-type median is too noisy to trust;
# we still report it but flag it as small-sample.
SMALL_SAMPLE_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Convention canonicalization (#536/#587), ported from subset_compare.py in
# post-full346 Phase 1c so the full-pool scorer applies the SAME vetted
# conversions the subset comparator has used since #587. Converting only one
# side would break pairs that already agree in a shared non-canonical
# convention, so BOTH the extraction (its declared convention) and the GT
# curve (its file's source convention) are canonicalized.
# ---------------------------------------------------------------------------

def _canonicalize_curve(coupling_type, arr: np.ndarray, token):
    """Apply a vetted ``to_canonical`` conversion to an Nx2 curve (no-op if
    token is None / unknown). Returns ``(array, guard_refused)``:
    ``guard_refused`` is True when the token's magnitude guard rejected the
    values (round-2 rule — converting implausible values is worse than
    excluding the pair)."""
    if token is None or arr is None or len(arr) == 0:
        return arr, False
    pts = [(float(m), float(g)) for m, g in arr]
    out, note = to_canonical(coupling_type, pts, token)
    if note.startswith(GUARD_REFUSED):
        return arr, True
    if not out:
        return arr, False
    return np.array(out, dtype=float, ndmin=2), False


def _maybe_canonicalize(result: dict, predicted_ct: str, ext_array: np.ndarray,
                        gt_entry, gt_data: np.ndarray):
    """Apply both-sides convention canonicalization.

    Returns ``(ext_array, gt_data, unconvertible)``. ``unconvertible`` is True
    when the extraction declares a recognized but non-convertible convention
    (#604), or a recognized token whose magnitude guard refused the values —
    caller should treat as convention_mismatch. No-op (and never unconvertible)
    when the extraction does not declare a convention, so field-less old
    snapshots stay raw.
    """
    if not result.get("coupling_convention"):
        return ext_array, gt_data, False
    ext_token = classify_reported_convention(
        predicted_ct, result.get("coupling_convention"))
    if ext_token == UNCONVERTIBLE:
        return ext_array, gt_data, True
    gt_token = file_source_convention(gt_entry.reference_repo_file, predicted_ct)
    ext_c, ext_refused = _canonicalize_curve(predicted_ct, ext_array, ext_token)
    gt_c, gt_refused = _canonicalize_curve(predicted_ct, gt_data, gt_token)
    if ext_refused or gt_refused:
        return ext_array, gt_data, True
    return ext_c, gt_c, False


def _frac_within(res: np.ndarray, tau: float) -> float:
    return float(np.mean(res <= tau)) if len(res) else 0.0


def _reverse_as_effective(im: InterpolationMetrics) -> InterpolationMetrics:
    """Promote the reverse pass to the paper's effective score.

    Used when the forward pass has no interpolatable GT vertex but the
    extracted masses DO lie inside the GT range (vertex-sparse GT such as a
    2-vertex flat segment, or a single-point extraction). This cannot mask a
    wrong-mass-window failure: an extraction outside the GT range has no
    reverse residuals either.
    """
    import dataclasses
    res = im.residuals_dex_reverse
    return dataclasses.replace(
        im,
        num_interpolatable=im.num_interpolatable_reverse,
        interpolation_coverage=im.interpolation_coverage_reverse,
        residuals_dex=res,
        median_residual_dex=im.median_residual_dex_reverse,
        mean_residual_dex=im.mean_residual_dex_reverse,
        p90_residual_dex=im.p90_residual_dex_reverse,
        max_residual_dex=im.max_residual_dex_reverse,
        frac_within_0_1dex=_frac_within(res, 0.1),
        frac_within_0_3dex=_frac_within(res, 0.3),
        frac_within_0_5dex=_frac_within(res, 0.5),
        frac_within_1_0dex=_frac_within(res, 1.0),
    )


def _single_point_as_metrics(arxiv_id: str, sp, n_ext: int,
                             n_gt: int) -> InterpolationMetrics:
    """Wrap a ``single_point_compare`` result as InterpolationMetrics so
    single-point comparisons feed the same aggregates as curve comparisons."""
    med, n_matched, cov, res = sp
    return InterpolationMetrics(
        arxiv_id=arxiv_id, num_extracted=n_ext, num_ground_truth=n_gt,
        num_interpolatable=n_matched, interpolation_coverage=cov,
        residuals_dex=res,
        median_residual_dex=med,
        mean_residual_dex=float(np.mean(res)),
        p90_residual_dex=float(np.percentile(res, 90)),
        max_residual_dex=float(np.max(res)),
        frac_within_0_1dex=_frac_within(res, 0.1),
        frac_within_0_3dex=_frac_within(res, 0.3),
        frac_within_0_5dex=_frac_within(res, 0.5),
        frac_within_1_0dex=_frac_within(res, 1.0),
    )


def _bootstrap_median_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Bootstrap a (lo, hi) confidence interval for the median of ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times, takes the
    median of each resample, and returns the empirical ``ci`` percentile
    interval. Returns ``(None, None)`` for an empty input and ``(v, v)`` for a
    single value (a point with no spread). A fixed seed keeps reports
    reproducible across runs on the same cache.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return None, None
    if n == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    medians = np.median(arr[idx], axis=1)
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(medians, lo_pct)), float(np.percentile(medians, hi_pct))


def compute_all_metrics(
    entries: list[GroundTruthEntry],
    results: list[dict],
) -> dict:
    """Compute all evaluation metrics.

    Returns a dict with classification metrics, curve metrics, and calibration data.
    """
    coupling_clf = ClassificationMetrics()
    # Scalar-label metrics are scored ONLY against human-verified entries.
    # Auto-expanded / repo_upstream entries carry placeholder labels
    # (is_new_limit=True, is_projection=False, data_source="table"), so scoring
    # against them measures the placeholder, not the pipeline.
    is_limit_clf = ClassificationMetrics()
    is_projection_clf = ClassificationMetrics()
    data_source_clf = ClassificationMetrics()

    curve_metrics_list: list[CurveMetrics] = []
    interp_metrics_list: list[InterpolationMetrics] = []
    symmetric_metrics_list: list[SymmetricCurveMetrics] = []
    confidences: list[float] = []
    curve_arxiv_ids: list[str] = []

    per_paper: list[dict] = []

    # Why a paper did / didn't get a curve comparison. Honest aggregates require
    # knowing this: a paper whose extracted coupling has no matching GT curve is
    # NOT an extraction-quality failure — it is simply not comparable.
    comparison_status_counts: Counter = Counter()

    # Group all GT entries by paper; one extraction result per paper. A single
    # paper often yields several repo files (one per coupling); we must NOT
    # score one extraction against curves for couplings it never targeted.
    by_id: "OrderedDict[str, dict]" = OrderedDict()
    for entry, result in zip(entries, results):
        slot = by_id.setdefault(entry.arxiv_id, {"entries": [], "result": result})
        slot["entries"].append(entry)
        # Results are identical across a paper's entries; prefer a non-error one.
        if "error" in slot["result"] and "error" not in result:
            slot["result"] = result

    # Excluded GT entries (Phase 1a, post-full346): skipped from ALL scoring
    # (residuals AND classification — an invalid GT cannot grade either) but
    # collected here so the report lists every exclusion with its reason.
    # See evaluation/ground_truth/EXCLUSIONS.md.
    gt_exclusions: list[dict] = []

    for arxiv_id, slot in by_id.items():
        paper_entries: list[GroundTruthEntry] = slot["entries"]
        result = slot["result"]

        excluded_here = [e for e in paper_entries if e.excluded]
        for e in excluded_here:
            gt_exclusions.append({
                "arxiv_id": arxiv_id,
                "reference_repo_file": e.reference_repo_file,
                "coupling_type": e.coupling_type,
                "exclusion_reason": e.exclusion_reason,
                "exclusion_evidence": e.exclusion_evidence,
            })
        paper_entries = [e for e in paper_entries if not e.excluded]
        if not paper_entries:
            per_paper.append({
                "arxiv_id": arxiv_id,
                "status": "excluded_gt",
                "num_gt_entries_excluded": len(excluded_here),
                "exclusion_reason": excluded_here[0].exclusion_reason,
            })
            comparison_status_counts["excluded_gt"] += 1
            continue

        rep = paper_entries[0]  # representative entry for paper-level fields

        # Authoritative couplings = the couplings of the actual GT data files.
        true_couplings = {_authoritative_coupling(e) for e in paper_entries}

        paper_report: dict = {
            "arxiv_id": arxiv_id,
            "difficulty": rep.difficulty,
            "num_gt_entries": len(paper_entries),
            "true_couplings": sorted(true_couplings),
        }

        if "error" in result:
            paper_report["status"] = "extraction_failed"
            paper_report["error"] = result["error"]
            comparison_status_counts["extraction_failed"] += 1
            per_paper.append(paper_report)
            continue

        paper_report["status"] = "extracted"
        paper_report["extraction_confidence"] = result.get("extraction_confidence", 0.0)
        paper_report["data_source"] = result.get("data_source")
        paper_report["num_points_extracted"] = result.get("num_points", 0)
        paper_report["elapsed_s"] = result.get("elapsed_s", 0.0)

        # --- Coupling-type classification (against authoritative couplings) ---
        predicted_ct = _normalize_predicted_coupling(result.get("coupling_type"))
        ct_correct = predicted_ct in true_couplings if predicted_ct else False

        coupling_clf.total += 1
        if ct_correct:
            coupling_clf.correct += 1
        else:
            coupling_clf.errors.append({
                "arxiv_id": arxiv_id,
                "predicted": str(predicted_ct),
                "expected": str(sorted(true_couplings)),
            })

        # --- Scalar-label classification (human-verified entries only) ---
        verified = next((e for e in paper_entries if not _is_placeholder_entry(e)), None)
        if verified is not None:
            is_limit_clf.record(arxiv_id, result.get("is_new_limit"), verified.is_new_limit)
            is_projection_clf.record(arxiv_id, result.get("is_projection"), verified.is_projection)
            data_source_clf.record(arxiv_id, result.get("data_source"), verified.data_source_expected)

        paper_report["coupling_type_correct"] = ct_correct
        paper_report["coupling_type_predicted"] = predicted_ct
        paper_report["coupling_type_expected"] = sorted(true_couplings)

        # --- Curve comparison: ONLY against a GT curve of the same coupling ---
        extracted_points = result.get("data_points", [])
        ext_array = (np.array(extracted_points, dtype=float, ndmin=2)
                     if extracted_points else None)
        # Lever E (#587): a mass-independent (flat) extraction is recorded with
        # no usable mass (all masses <= 0) and would be dropped by boundary
        # filtering — expand it to a horizontal segment so its coupling can be
        # scored against the GT at the GT's own masses.
        if ext_array is not None:
            ext_array = _expand_mass_independent(ext_array)

        # Populated by the scoring below when comparison succeeds.
        im = None
        chosen = None
        gt_c = ext_c = None
        scored_via = None  # "forward" | "reverse" | "single_point" | "single_point_gt"

        if predicted_ct is None:
            comparison_status = "no_prediction"
        elif predicted_ct not in true_couplings:
            # The extraction targeted a coupling for which we hold no GT curve.
            comparison_status = "no_comparable_gt"
        elif ext_array is None:
            comparison_status = "no_extracted_points"
        else:
            # Convention guard (#536): a GT curve whose entry-level convention
            # label differs from the canonical one for this coupling type is
            # NOT comparable raw — the residual would be a units gap, not
            # extraction error. (The vetted per-token canonicalization below is
            # a separate, additive mechanism, mirroring subset_compare.py.)
            expected_conv, _ = canonical_convention(predicted_ct)

            # Candidate GT entries: same authoritative coupling AND usable data.
            multi_candidates = []   # n_mass >= 2 : a comparable curve
            single_candidates = []  # n_mass == 1 : a single-point (operating-mass) reference
            has_convention_mismatch = False  # same coupling, different convention
            for e in paper_entries:
                if _authoritative_coupling(e) != predicted_ct:
                    continue
                # Skip (and flag) GT curves whose convention differs from the
                # extraction's expected one. None on either side = unknown, so
                # we do not treat it as a mismatch.
                if (expected_conv is not None
                        and e.coupling_convention is not None
                        and e.coupling_convention != expected_conv):
                    has_convention_mismatch = True
                    continue
                gt = e.load_data()
                if gt is None:
                    gt = e.load_reference_data(PROJECT_ROOT)
                if gt is None:
                    continue
                n_pts, n_mass = _usable_gt_stats(gt, predicted_ct)
                if n_mass >= 2:
                    multi_candidates.append((n_mass, e, gt))
                elif n_pts >= 1:
                    single_candidates.append((1, e, gt))

            if multi_candidates:
                multi_candidates.sort(key=lambda t: -t[0])  # richest GT curve wins
                _, chosen, gt_data = multi_candidates[0]
                # Canonicalize BOTH sides (vetted conversions only; no-op for
                # snapshots that declare no convention).
                ext_c, gt_c, unconvertible = _maybe_canonicalize(
                    result, predicted_ct, ext_array, chosen, gt_data)
                if unconvertible:
                    # Declared convention is recognized but has NO vetted
                    # conversion (#604, e.g. oscillating-EDM e*cm amplitude):
                    # a convention gap, not a 15-dex "residual".
                    comparison_status = "convention_mismatch"
                    chosen = None
                else:
                    comparison_status = "compared"
                    scored_via = "forward"
                    im = compute_interpolation_metrics(
                        arxiv_id, ext_c, gt_c, coupling_type=predicted_ct,
                    )
                    if im.num_interpolatable == 0:
                        if im.num_interpolatable_reverse > 0:
                            # Vertex-sparse GT or 1-point extraction inside the
                            # GT range: promote the reverse pass (GT evaluated
                            # at the extracted masses) to the effective score.
                            scored_via = "reverse"
                            im = _reverse_as_effective(im)
                        else:
                            # Sparse single-value extraction at one of the GT
                            # curve's operating masses (nearest-mass tolerance;
                            # guarded to <= 3 distinct extracted masses so a
                            # genuine wrong-window curve failure is not masked
                            # by one lucky near-mass point).
                            sp = single_point_compare(
                                gt_c, ext_c, predicted_ct, require_sparse_ref=True)
                            if sp is not None:
                                scored_via = "single_point"
                                ext_n = im.num_extracted
                                gt_n = im.num_ground_truth
                                im = _single_point_as_metrics(arxiv_id, sp, ext_n, gt_n)
                            # else: stays "forward" with an infinite residual —
                            # a genuine zero_overlap.
            elif single_candidates:
                # No multi-point GT curve. A single-mass GT is a point
                # reference: compare the extracted value at that operating
                # mass (single-point mode, #612).
                _, chosen, gt_data = single_candidates[0]
                ext_c, gt_c, unconvertible = _maybe_canonicalize(
                    result, predicted_ct, ext_array, chosen, gt_data)
                if unconvertible:
                    comparison_status = "convention_mismatch"
                    chosen = None
                else:
                    sp = single_point_compare(ext_c, gt_c, predicted_ct)
                    if sp is not None:
                        comparison_status = "compared"
                        scored_via = "single_point_gt"
                        n_ext_u = int(np.unique(ext_c[:, 0]).size)
                        im = _single_point_as_metrics(arxiv_id, sp, n_ext_u, 1)
                    else:
                        # GT is a single-mass point the extraction never
                        # reaches — not a comparable curve.
                        comparison_status = "gt_point_reference"
                        chosen = None
            elif has_convention_mismatch:
                # The only same-coupling GT curve(s) use a different convention.
                # Excluded from residuals — this is a units gap, not error.
                comparison_status = "convention_mismatch"
            else:
                comparison_status = "gt_unusable"

        paper_report["comparison_status"] = comparison_status
        comparison_status_counts[comparison_status] += 1

        if comparison_status == "compared":
            paper_report["gt_file"] = chosen.reference_repo_file
            # Coupling type this paper was scored under — the basis for the
            # per-coupling-type breakdown and the macro-average. For a "compared"
            # paper this is exactly the predicted coupling (which is guaranteed
            # to be one of the authoritative GT couplings).
            paper_report["comparison_coupling"] = predicted_ct
            paper_report["scored_via"] = scored_via

            interp_metrics_list.append(im)
            confidences.append(result.get("extraction_confidence", 0.0))
            curve_arxiv_ids.append(arxiv_id)

            paper_report["interp_metrics"] = {
                "num_extracted": im.num_extracted,
                "num_ground_truth": im.num_ground_truth,
                "num_interpolatable": im.num_interpolatable,
                "interpolation_coverage": im.interpolation_coverage,
                "median_residual_dex": im.median_residual_dex,
                "mean_residual_dex": im.mean_residual_dex,
                "p90_residual_dex": im.p90_residual_dex,
                "max_residual_dex": im.max_residual_dex,
                "frac_within_0_1dex": im.frac_within_0_1dex,
                "frac_within_0_3dex": im.frac_within_0_3dex,
                "frac_within_0_5dex": im.frac_within_0_5dex,
                "frac_within_1_0dex": im.frac_within_1_0dex,
                # Reverse pass (GT interpolated onto extracted masses).
                "num_interpolatable_reverse": im.num_interpolatable_reverse,
                "interpolation_coverage_reverse": im.interpolation_coverage_reverse,
                "median_residual_dex_reverse": im.median_residual_dex_reverse,
                "mean_residual_dex_reverse": im.mean_residual_dex_reverse,
                "p90_residual_dex_reverse": im.p90_residual_dex_reverse,
                "max_residual_dex_reverse": im.max_residual_dex_reverse,
            }

            # Symmetric / 2-D shape metrics: area-between-curves + mass Jaccard.
            # Computed on the CANONICALIZED curves so they see the same units
            # the interpolation metric scored. Skipped for single-mass-GT
            # comparisons (no curve shape exists to score; a degenerate
            # jaccard=0 entry would pollute the aggregates).
            if scored_via != "single_point_gt":
                sm = compute_symmetric_curve_metrics(
                    arxiv_id, ext_c, gt_c, coupling_type=predicted_ct,
                )
                symmetric_metrics_list.append(sm)
                paper_report["symmetric_metrics"] = {
                    "area_between_log": sm.area_between_log,
                    "overlap_log_mass_width": sm.overlap_log_mass_width,
                    "mass_jaccard": sm.mass_jaccard,
                    "ext_log_mass_lo": sm.ext_log_mass_lo,
                    "ext_log_mass_hi": sm.ext_log_mass_hi,
                    "gt_log_mass_lo": sm.gt_log_mass_lo,
                    "gt_log_mass_hi": sm.gt_log_mass_hi,
                }

                cm = compute_curve_metrics(arxiv_id, ext_c, gt_c)
                curve_metrics_list.append(cm)
                paper_report["curve_metrics"] = {
                    "hausdorff_log": cm.hausdorff_log,
                    "coverage_at_0_5dex": cm.coverage_at_0_5dex,
                    "coverage_at_1_0dex": cm.coverage_at_1_0dex,
                    "mass_range_overlap": cm.mass_range_overlap,
                    "median_coupling_log_error": cm.median_coupling_log_error,
                    "p90_coupling_log_error": cm.p90_coupling_log_error,
                    "num_extracted": cm.num_extracted,
                    "num_ground_truth": cm.num_ground_truth,
                }
            else:
                paper_report["symmetric_metrics"] = None
                paper_report["curve_metrics"] = None
        else:
            paper_report["interp_metrics"] = None
            paper_report["curve_metrics"] = None
            paper_report["symmetric_metrics"] = None

        per_paper.append(paper_report)

    # Confidence calibration (uses interpolation metrics)
    calibration = compute_confidence_calibration(
        confidences, interp_metrics_list, curve_arxiv_ids
    )

    # Aggregate interpolation statistics (primary).
    #
    # Two distinct failure modes are kept separate:
    #   (1) zero mass-range overlap -> median residual is inf. This is a
    #       MASS-RANGE failure (extraction spans the wrong masses, often only
    #       1-2 points), NOT a coupling-value error. Folding inf into a mean
    #       would be meaningless, so these are counted, not averaged.
    #   (2) finite residual -> a genuine coupling-value comparison. Summarised
    #       with the MEDIAN across papers (robust); the mean is outlier-driven
    #       and reported only as a secondary number.
    if interp_metrics_list:
        valid = [m for m in interp_metrics_list if m.median_residual_dex < float("inf")]
        n_zero_overlap = len(interp_metrics_list) - len(valid)
        med_resids = [m.median_residual_dex for m in valid]
        aggregate_interp = {
            "n_papers": len(interp_metrics_list),
            "n_zero_overlap": n_zero_overlap,
            "n_finite": len(valid),
            "mean_interpolation_coverage": float(np.mean([m.interpolation_coverage for m in interp_metrics_list])),
            # Robust headline: median across papers of each paper's median residual.
            "median_median_residual_dex": float(np.median(med_resids)) if valid else None,
            "p25_median_residual_dex": float(np.percentile(med_resids, 25)) if valid else None,
            "p75_median_residual_dex": float(np.percentile(med_resids, 75)) if valid else None,
            # Outlier-sensitive; kept for continuity with prior reports.
            "mean_median_residual_dex": float(np.mean(med_resids)) if valid else None,
            "mean_p90_residual_dex": float(np.mean([m.p90_residual_dex for m in valid])) if valid else None,
            "mean_frac_within_0_3dex": float(np.mean([m.frac_within_0_3dex for m in valid])) if valid else None,
            "mean_frac_within_0_5dex": float(np.mean([m.frac_within_0_5dex for m in valid])) if valid else None,
        }
        # Reverse pass aggregate (GT interpolated onto extracted masses).
        # A large gap between forward and reverse residual flags extent/shape
        # mismatch even when the forward residual alone looks good.
        valid_rev = [m for m in interp_metrics_list
                     if m.median_residual_dex_reverse < float("inf")]
        rev_resids = [m.median_residual_dex_reverse for m in valid_rev]
        aggregate_interp["n_finite_reverse"] = len(valid_rev)
        aggregate_interp["mean_interpolation_coverage_reverse"] = float(
            np.mean([m.interpolation_coverage_reverse for m in interp_metrics_list])
        )
        aggregate_interp["median_median_residual_dex_reverse"] = (
            float(np.median(rev_resids)) if valid_rev else None
        )
        aggregate_interp["mean_median_residual_dex_reverse"] = (
            float(np.mean(rev_resids)) if valid_rev else None
        )
    else:
        aggregate_interp = {"n_papers": 0}

    # Aggregate symmetric / 2-D shape metrics (area-between-curves, Jaccard).
    if symmetric_metrics_list:
        areas = [m.area_between_log for m in symmetric_metrics_list
                 if m.area_between_log < float("inf")]
        jaccards = [m.mass_jaccard for m in symmetric_metrics_list]
        aggregate_symmetric = {
            "n_papers": len(symmetric_metrics_list),
            "n_finite_area": len(areas),
            "median_area_between_log": float(np.median(areas)) if areas else None,
            "mean_area_between_log": float(np.mean(areas)) if areas else None,
            "median_mass_jaccard": float(np.median(jaccards)),
            "mean_mass_jaccard": float(np.mean(jaccards)),
        }
    else:
        aggregate_symmetric = {"n_papers": 0}

    # Aggregate legacy curve statistics (secondary)
    if curve_metrics_list:
        coverages_05 = [m.coverage_at_0_5dex for m in curve_metrics_list]
        coverages_10 = [m.coverage_at_1_0dex for m in curve_metrics_list]
        med_errs = [m.median_coupling_log_error for m in curve_metrics_list
                    if m.median_coupling_log_error < float("inf")]
        mass_overlaps = [m.mass_range_overlap for m in curve_metrics_list]

        aggregate_curve = {
            "n_papers_with_curves": len(curve_metrics_list),
            "mean_coverage_0_5dex": float(np.mean(coverages_05)),
            "mean_coverage_1_0dex": float(np.mean(coverages_10)),
            "mean_median_coupling_log_error": float(np.mean(med_errs)) if med_errs else None,
            "mean_mass_range_overlap": float(np.mean(mass_overlaps)),
        }
    else:
        aggregate_curve = {"n_papers_with_curves": 0}

    # Per-difficulty breakdown. NOTE: difficulty is a placeholder label for the
    # repo-sourced pool (almost all "medium"), so this is informational only.
    difficulty_breakdown = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [p for p in per_paper if p.get("difficulty") == diff]
        if not subset:
            continue
        extracted = [p for p in subset if p.get("status") == "extracted"]
        with_interp = [p for p in extracted if p.get("interp_metrics") is not None]
        valid_interp = [p for p in with_interp
                        if p["interp_metrics"]["median_residual_dex"] < float("inf")]
        difficulty_breakdown[diff] = {
            "total": len(subset),
            "extracted": len(extracted),
            "coupling_type_accuracy": (
                sum(1 for p in extracted if p.get("coupling_type_correct")) / len(extracted)
                if extracted else 0.0
            ),
            "median_residual_dex": (
                float(np.median([p["interp_metrics"]["median_residual_dex"] for p in valid_interp]))
                if valid_interp else None
            ),
            "mean_frac_within_0_3dex": (
                float(np.mean([p["interp_metrics"]["frac_within_0_3dex"] for p in valid_interp]))
                if valid_interp else None
            ),
        }

    # Per-data-source breakdown (grouped by the pipeline's reported source).
    # Reported with the robust median to match the headline; the zero-overlap
    # count is kept separate so the vision/text signal is not muddied by
    # mass-range failures.
    source_breakdown = {}
    for source in ["table", "figure_vision", "text"]:
        subset = [p for p in per_paper if p.get("data_source") == source]
        if not subset:
            continue
        with_interp = [p for p in subset if p.get("interp_metrics") is not None]
        valid_interp = [p for p in with_interp
                        if p["interp_metrics"]["median_residual_dex"] < float("inf")]
        source_breakdown[source] = {
            "total": len(subset),
            "n_compared": len(with_interp),
            "n_zero_overlap": len(with_interp) - len(valid_interp),
            "median_residual_dex": (
                float(np.median([p["interp_metrics"]["median_residual_dex"] for p in valid_interp]))
                if valid_interp else None
            ),
            "mean_frac_within_0_3dex": (
                float(np.mean([p["interp_metrics"]["frac_within_0_3dex"] for p in valid_interp]))
                if valid_interp else None
            ),
        }

    # --- Per-coupling-type breakdown (issue #543) -----------------------------
    # The compared-paper pool is dominated by AxionPhoton, so the micro-average
    # (per-paper) headline is effectively an AxionPhoton number. We break the
    # residual down by coupling type, attach a per-type N + a bootstrap 95% CI
    # on the median residual, and flag types with N < SMALL_SAMPLE_THRESHOLD as
    # too small to trust on their own. The macro-average (equal weight per type)
    # is then reported alongside the micro-average so the headline is not
    # silently driven by the largest type.
    by_type: "OrderedDict[str, list[float]]" = OrderedDict()
    for p in per_paper:
        if p.get("comparison_status") != "compared":
            continue
        ct = p.get("comparison_coupling")
        im = p.get("interp_metrics")
        if ct is None or im is None:
            continue
        resid = im["median_residual_dex"]
        if resid is None or resid >= float("inf"):
            continue  # zero mass-range overlap → not a coupling-value comparison
        by_type.setdefault(ct, []).append(float(resid))

    per_type_breakdown = {}
    per_type_medians: list[float] = []  # one median per type → macro-average
    for ct, resids in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        n = len(resids)
        med = float(np.median(resids))
        ci_lo, ci_hi = _bootstrap_median_ci(resids)
        per_type_breakdown[ct] = {
            "n": n,
            "median_residual_dex": med,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "small_sample": n < SMALL_SAMPLE_THRESHOLD,
        }
        per_type_medians.append(med)

    # Micro-average: median across all compared papers (per-paper weight) — this
    # is the existing headline (`interpolation_aggregate.median_median_residual_dex`),
    # recomputed here over exactly the per-type pool so micro and macro share a
    # denominator and are directly comparable.
    all_type_resids = [r for resids in by_type.values() for r in resids]
    micro_median = float(np.median(all_type_resids)) if all_type_resids else None
    # Macro-average: equal weight per coupling type (mean of per-type medians).
    macro_median = float(np.mean(per_type_medians)) if per_type_medians else None
    macro_micro_gap = (
        macro_median - micro_median
        if (macro_median is not None and micro_median is not None)
        else None
    )
    per_type_aggregate = {
        "n_types": len(per_type_breakdown),
        "n_papers_compared": len(all_type_resids),
        "micro_median_residual_dex": micro_median,
        "macro_median_residual_dex": macro_median,
        "macro_minus_micro_dex": macro_micro_gap,
        "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        "per_type": per_type_breakdown,
    }

    n_papers = len(by_id)
    return {
        "n_papers": n_papers,
        "gt_exclusions": {
            "n_entries": len(gt_exclusions),
            "n_papers_fully_excluded": comparison_status_counts.get("excluded_gt", 0),
            "entries": gt_exclusions,
        },
        "per_type_aggregate": per_type_aggregate,
        "classification": {
            "coupling_type": {"accuracy": coupling_clf.accuracy, "total": coupling_clf.total, "errors": coupling_clf.errors},
            "is_new_limit": {"accuracy": is_limit_clf.accuracy, "total": is_limit_clf.total, "errors": is_limit_clf.errors},
            "is_projection": {"accuracy": is_projection_clf.accuracy, "total": is_projection_clf.total, "errors": is_projection_clf.errors},
            "data_source": {"accuracy": data_source_clf.accuracy, "total": data_source_clf.total, "errors": data_source_clf.errors},
        },
        "comparison_coverage": {
            "n_papers": n_papers,
            "n_compared": comparison_status_counts.get("compared", 0),
            "status_counts": dict(comparison_status_counts),
        },
        "interpolation_aggregate": aggregate_interp,
        "symmetric_aggregate": aggregate_symmetric,
        "curve_aggregate": aggregate_curve,
        "confidence_calibration": [asdict(b) for b in calibration],
        "difficulty_breakdown": difficulty_breakdown,
        "source_breakdown": source_breakdown,
        "per_paper": per_paper,
    }


def build_metrics_summary(all_metrics: dict) -> dict:
    """Build a small, git-trackable summary of the full metrics dict.

    The full ``metrics.json`` is gitignored (it embeds per-paper API outputs and
    is large), so runs are not diffable in review. This summary distils the
    headline numbers — per-status counts, per-type N, micro/macro residual
    averages, and the calibration overconfidence gap — into a compact, stable
    object that IS committed, so a metrics regression shows up in a PR diff.

    Pure function of ``all_metrics`` (no I/O) so it is unit-testable.
    """
    coverage = all_metrics.get("comparison_coverage", {})
    interp = all_metrics.get("interpolation_aggregate", {})
    symmetric = all_metrics.get("symmetric_aggregate", {})
    per_type = all_metrics.get("per_type_aggregate", {})
    classification = all_metrics.get("classification", {})

    # Per-type N (compared papers per coupling type).
    per_type_n = {
        ct: info.get("n", 0)
        for ct, info in (per_type.get("per_type") or {}).items()
    }

    # Calibration overconfidence gap: the most-confident bin's mean confidence
    # minus its actual accuracy. Positive => the extractor is overconfident in
    # the bin it is most sure about. Uses the top NON-EMPTY confidence bin.
    calibration = all_metrics.get("confidence_calibration", []) or []
    overconfidence_gap = None
    top_bin_confidence = None
    top_bin_accuracy = None
    for b in reversed(calibration):
        if b.get("n_papers", 0) > 0:
            top_bin_confidence = b.get("mean_confidence")
            top_bin_accuracy = b.get("actual_accuracy")
            if top_bin_confidence is not None and top_bin_accuracy is not None:
                overconfidence_gap = top_bin_confidence - top_bin_accuracy
            break

    exclusions = all_metrics.get("gt_exclusions", {})

    return {
        "n_papers": all_metrics.get("n_papers", 0),
        "gt_exclusions": {
            "n_entries": exclusions.get("n_entries", 0),
            "n_papers_fully_excluded": exclusions.get("n_papers_fully_excluded", 0),
        },
        "status_counts": coverage.get("status_counts", {}),
        "classification_accuracy": {
            field: classification.get(field, {}).get("accuracy")
            for field in ("coupling_type", "is_new_limit",
                          "is_projection", "data_source")
        },
        "interpolation": {
            "n_finite": interp.get("n_finite"),
            "n_zero_overlap": interp.get("n_zero_overlap"),
            "median_median_residual_dex": interp.get("median_median_residual_dex"),
            "mean_frac_within_0_3dex": interp.get("mean_frac_within_0_3dex"),
            "median_median_residual_dex_reverse": interp.get(
                "median_median_residual_dex_reverse"),
        },
        "symmetric": {
            "median_area_between_log": symmetric.get("median_area_between_log"),
            "median_mass_jaccard": symmetric.get("median_mass_jaccard"),
        },
        "per_type_aggregate": {
            "n_types": per_type.get("n_types"),
            "n_papers_compared": per_type.get("n_papers_compared"),
            "micro_median_residual_dex": per_type.get("micro_median_residual_dex"),
            "macro_median_residual_dex": per_type.get("macro_median_residual_dex"),
            "macro_minus_micro_dex": per_type.get("macro_minus_micro_dex"),
            "per_type_n": per_type_n,
        },
        "calibration": {
            "noise_floor_residual_dex": NOISE_FLOOR_RESIDUAL_DEX,
            "top_bin_mean_confidence": top_bin_confidence,
            "top_bin_actual_accuracy": top_bin_accuracy,
            "overconfidence_gap": overconfidence_gap,
        },
    }


def write_metrics_summary(all_metrics: dict, path: Path) -> dict:
    """Write the diffable metrics summary to ``path`` and return it."""
    summary = build_metrics_summary(all_metrics)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    return summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate the AutoAxionLimits extraction pipeline")
    parser.add_argument("--populate", action="store_true",
                        help="Populate ground-truth data/ from repo reference files")
    parser.add_argument("--extract", action="store_true",
                        help="Run extraction on ground-truth papers (calls Claude API)")
    parser.add_argument("--metrics", action="store_true",
                        help="Compute metrics from cached extraction results")
    parser.add_argument("--report", action="store_true",
                        help="Generate evaluation report (markdown + plots)")
    parser.add_argument("--gold", action="store_true",
                        help="Score cached extractions against the hand-curated "
                             "gold set and print the gold-vs-repo upstream gap "
                             "(delegates to evaluation.gold_diff; additive, does "
                             "not affect the repo-pool metrics above)")
    parser.add_argument("--arxiv-id", type=str, default=None,
                        help="Only process this arXiv ID (with --extract)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cached result exists")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for report (default: evaluation/report.md)")

    args = parser.parse_args()

    if not any([args.populate, args.extract, args.metrics, args.report, args.gold]):
        parser.print_help()
        return

    if args.gold:
        # Additive gold-set scoring; reuses cached results + metrics, separate
        # from the repo-pool aggregates. Kept as a thin delegate so the gold
        # logic lives in its own module (minimal evaluate.py footprint).
        from evaluation.gold_diff import compute_gold_diff, render_report
        diff = compute_gold_diff()
        s = diff["summary"]
        logger.info("gold-vs-repo upstream gap: %.3f dex (N=%s); "
                    "extraction-vs-gold: %.3f (N=%s); extraction-vs-repo: %.3f (N=%s)",
                    s["gold_vs_repo_median_dex"] or float("nan"), s["gold_vs_repo_n"],
                    s["ext_vs_gold_median_dex"] or float("nan"), s["ext_vs_gold_n"],
                    s["ext_vs_repo_median_dex"] or float("nan"), s["ext_vs_repo_n"])
        gold_report = Path(__file__).parent / "gold_report.md"
        gold_report.write_text(render_report(diff))
        logger.info("Gold report written to %s", gold_report)
        if not any([args.populate, args.extract, args.metrics, args.report]):
            return

    entries = load_ground_truth()
    logger.info("Loaded %d ground-truth papers", len(entries))

    if args.populate:
        n = populate_data_from_repo(PROJECT_ROOT)
        logger.info("Populated %d data files from repo", n)

    if args.extract:
        target_entries = entries
        if args.arxiv_id:
            target_entries = [e for e in entries if e.arxiv_id == args.arxiv_id]
            if not target_entries:
                logger.error("arXiv ID %s not found in ground truth", args.arxiv_id)
                return

        for entry in target_entries:
            cached = _load_cached_result(entry.arxiv_id)
            if cached and not args.force:
                logger.info("Using cached result for %s", entry.arxiv_id)
                continue

            logger.info("Extracting %s: %s", entry.arxiv_id, entry.paper_title)
            result = run_extraction(entry)
            _save_result(entry.arxiv_id, result)

            # Be nice to the API
            time.sleep(2)

    if args.metrics or args.report:
        # Load all cached results
        results = []
        valid_entries = []
        for entry in entries:
            cached = _load_cached_result(entry.arxiv_id)
            if cached is None:
                logger.warning("No cached result for %s, skipping", entry.arxiv_id)
                continue
            results.append(cached)
            valid_entries.append(entry)

        if not results:
            logger.error("No cached results found. Run --extract first.")
            return

        all_metrics = compute_all_metrics(valid_entries, results)

        # Save metrics
        metrics_path = RESULTS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info("Metrics saved to %s", metrics_path)

        # Diffable summary (committed to git; full metrics.json stays ignored).
        summary_path = RESULTS_DIR / "metrics_summary.json"
        write_metrics_summary(all_metrics, summary_path)
        logger.info("Metrics summary saved to %s", summary_path)

        if args.report:
            report_path = args.output or str(Path(__file__).parent / "report.md")
            generate_report(all_metrics, report_path)
            logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
