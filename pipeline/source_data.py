"""arXiv e-print source tarballs as a deterministic extraction channel (WS1).

Many papers' e-print source contains the actual curve coordinates — pgfplots
``\\addplot table {file.dat}`` data files, ``.csv``/``.txt`` companions, inline
``coordinates {(x,y)...}`` lists. Extracting those is deterministic and exact,
so the channel should ultimately outrank every LLM read (planned
``source_data`` tier ABOVE ``table`` in ``transform_guard.SOURCE_TIER``; the
selector wiring is a later PR — this module only fetches, scans, and ranks).

Security posture: ALL tarball content is untrusted data.

* nothing is ever executed (no TeX compilation, no shell-out);
* archive extraction rejects absolute paths, ``..`` traversal, links and
  special files, and enforces per-file / total / count / depth caps;
* text destined for any later LLM prompt must be routed through the same
  ``===PAPER_CONTENT===`` delimiter discipline as PDF text (this module only
  emits it via :func:`sanitize_snippet`, which strips control characters).

Failure discipline: the channel NEVER raises out of :func:`scan_arxiv_source`
— withdrawn sources, PDF-only submissions, tar bombs and undecodable files all
log and fall through to an empty candidate list so the existing channels run
unaffected. See ``evaluation/eval_runs/PLAN_extraction_channels.md`` (WS1).
"""

from __future__ import annotations

import gzip
import io
import logging
import math
import os
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits for untrusted-archive handling
# ---------------------------------------------------------------------------

MAX_TOTAL_BYTES = 50 * 1024 * 1024   # whole extracted tree
MAX_FILE_BYTES = 20 * 1024 * 1024    # any single member
MAX_MEMBERS = 2000
MAX_DEPTH = 8
# Only files this small are ever parsed as candidate data (a 20 MB "table" is
# a simulation dump, not a published limit curve).
MAX_DATA_FILE_BYTES = 5 * 1024 * 1024
MAX_TEX_FILE_BYTES = 5 * 1024 * 1024

_DATA_SUFFIXES = {".dat", ".csv", ".txt", ".tsv"}
_TEX_SUFFIXES = {".tex", ".tikz", ".pgf"}


class SourceDataError(Exception):
    """Internal channel error (guards tripped, malformed archive, ...)."""


# ---------------------------------------------------------------------------
# Fetch (mirrors extractor.download_pdf: persistent cache + backoff)
# ---------------------------------------------------------------------------

def _source_cache_dir() -> Optional[Path]:
    """Persistent cross-run e-print cache (default ``~/.cache/aal_source_cache``,
    overridable via ``AAL_SOURCE_CACHE``; set it empty to disable). Same
    rationale as the PDF cache: arXiv throttles bursts, and the eval pool
    re-fetches the same 346 papers across runs."""
    val = os.environ.get("AAL_SOURCE_CACHE")
    if val is None:
        return Path.home() / ".cache" / "aal_source_cache"
    if not val.strip():
        return None
    return Path(val)


def download_source(arxiv_id: str, workdir: Path, *, max_retries: int = 5,
                    base_delay: float = 5.0) -> Path:
    """Download the raw e-print blob (gzip/tar/pdf/tex — format sniffed later).

    Retry/backoff discipline copied from ``extractor.download_pdf``: retry on
    429 / 5xx / timeout / transport errors, raise immediately on other 4xx
    (404 = source withdrawn). Serves from the persistent cache when available.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    safe_id = arxiv_id.replace("/", "_")
    blob_path = workdir / f"{safe_id}.eprint"
    if blob_path.exists() and blob_path.stat().st_size > 0:
        return blob_path

    cache_dir = _source_cache_dir()
    cached = (cache_dir / f"{safe_id}.eprint") if cache_dir else None
    if cached and cached.exists() and cached.stat().st_size > 0:
        import shutil
        shutil.copyfile(cached, blob_path)
        logger.info("Using cached e-print for %s", arxiv_id)
        return blob_path

    logger.info("Downloading %s", url)
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                resp = client.get(url)
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.content
            if not data:
                raise ValueError("empty e-print response")
            blob_path.write_bytes(data)
            if cached is not None:
                try:
                    import shutil
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(blob_path, cached)
                except OSError as e:  # cache is best-effort
                    logger.debug("e-print cache write failed for %s: %s", arxiv_id, e)
            return blob_path
        except (httpx.TimeoutException, httpx.TransportError, ValueError) as e:
            last_exc = e
        except httpx.HTTPStatusError as e:
            last_exc = e
            code = e.response.status_code
            if not (code == 429 or code >= 500):
                raise
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.warning("download_source %s failed (attempt %d/%d), retrying in %.0fs: %s",
                           arxiv_id, attempt + 1, max_retries, delay, str(last_exc)[:80])
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Safe unpack
# ---------------------------------------------------------------------------

def _safe_member(member: tarfile.TarInfo) -> bool:
    """Regular file, relative path, no traversal, sane depth and size."""
    if not member.isreg():
        return False  # links, devices, dirs (dirs are implied by file paths)
    name = member.name
    if name.startswith(("/", "\\")) or ".." in Path(name).parts:
        return False
    if len(Path(name).parts) > MAX_DEPTH:
        return False
    if member.size > MAX_FILE_BYTES:
        return False
    return True


def _gunzip_capped(data: bytes, cap: int | None = None) -> bytes:
    """Decompress at most ``cap`` bytes (gzip-bomb guard). ``cap`` resolves to
    the module-level ``MAX_TOTAL_BYTES`` at call time, not definition time."""
    if cap is None:
        cap = MAX_TOTAL_BYTES
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        out = gz.read(cap + 1)
    if len(out) > cap:
        raise SourceDataError(f"gzip expands past {cap} bytes")
    return out


def extract_source(blob_path: Path, dest_dir: Path) -> list[Path]:
    """Unpack an e-print blob into ``dest_dir`` with tar-bomb/traversal guards.

    Handles the four formats arXiv serves: gzipped tar (the common case),
    gzipped single TeX file, bare tar, and PDF-only submissions (returns
    ``[]`` — there is no source to scan). Returns the extracted file paths.
    Raises :class:`SourceDataError` when a guard trips.
    """
    raw = blob_path.read_bytes()
    if not raw:
        return []
    if raw[:5] == b"%PDF-":
        return []  # PDF-only submission
    if raw[:2] == b"\x1f\x8b":
        raw = _gunzip_capped(raw)
        if raw[:5] == b"%PDF-":
            return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    except tarfile.TarError:
        # Single-file submission (a lone .tex): store under a fixed name.
        p = dest_dir / "main.tex"
        p.write_bytes(raw[:MAX_FILE_BYTES])
        return [p]

    total = 0
    with tf:
        members = tf.getmembers()
        if len(members) > MAX_MEMBERS:
            raise SourceDataError(f"archive has {len(members)} members > {MAX_MEMBERS}")
        for member in members:
            if not _safe_member(member):
                logger.debug("skipping unsafe member %r", member.name)
                continue
            total += member.size
            if total > MAX_TOTAL_BYTES:
                raise SourceDataError(f"archive expands past {MAX_TOTAL_BYTES} bytes")
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            out = dest_dir / member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with fobj:
                out.write_bytes(fobj.read(MAX_FILE_BYTES))
            extracted.append(out)
    return extracted


# ---------------------------------------------------------------------------
# Numeric-table parsing
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"^\s*(?:#|%|//)")
_SPLIT_RE = re.compile(r"[,\s;]+")


def parse_numeric_table(text: str, *, max_rows: int = 50000,
                        min_rows: int = 3) -> Optional[list[tuple[float, ...]]]:
    """Parse ``text`` as a whitespace/comma/semicolon-separated numeric table.

    Returns rows (each a tuple of floats, trimmed to the modal column count)
    when the content is table-like: >= ``min_rows`` data rows, >= 2 columns,
    and >= 90% of non-comment/non-blank lines numeric. Otherwise ``None``.
    """
    rows: list[tuple[float, ...]] = []
    n_content = 0
    for line in text.splitlines()[: max_rows * 2]:
        line = line.strip()
        if not line or _COMMENT_RE.match(line):
            continue
        n_content += 1
        parts = [p for p in _SPLIT_RE.split(line) if p]
        try:
            vals = tuple(float(p) for p in parts)
        except ValueError:
            continue
        if len(vals) >= 2 and all(math.isfinite(v) for v in vals):
            rows.append(vals)
        if len(rows) >= max_rows:
            break
    if len(rows) < min_rows or n_content == 0 or len(rows) / n_content < 0.9:
        return None
    ncols = max(set(len(r) for r in rows), key=lambda n: sum(1 for r in rows if len(r) == n))
    rows = [r[:ncols] for r in rows if len(r) >= ncols]
    if ncols < 2 or len(rows) < min_rows:
        return None
    return rows


# ---------------------------------------------------------------------------
# TeX scanning
# ---------------------------------------------------------------------------

# \addplot [opts] table [opts] {file}; \addplot table {file};
_ADDPLOT_TABLE_RE = re.compile(
    r"\\addplot\+?\s*(?:\[[^\]]*\])?\s*table\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_PGFTABLEREAD_RE = re.compile(r"\\pgfplotstableread\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
# coordinates {(1e-6,3e-11) (2e-6,4e-11) ...}
_COORDINATES_RE = re.compile(r"coordinates\s*\{((?:\s*\([^()]*\))+)\s*\}")
_COORD_PAIR_RE = re.compile(r"\(\s*([^,()\s]+)\s*,\s*([^,()\s]+)\s*\)")
_AXIS_BEGIN_RE = re.compile(r"\\begin\{(?:log*axis|axis|semilogx|semilogy|loglogaxis)\}\s*\[",
                            re.IGNORECASE)
_AXIS_OPT_RE = re.compile(
    r"\b(xlabel|ylabel|xmode|ymode)\s*=\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|[^,\]]+)")

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_snippet(text: str, limit: int = 400) -> str:
    """Strip control characters and truncate — the only sanctioned way tarball
    text leaves this module (prompt-injection defence, same discipline as the
    PDF text sanitizer)."""
    return _CONTROL_CHARS_RE.sub("", text or "")[:limit]


def _axis_hints_before(tex: str, pos: int) -> dict[str, str]:
    """Options of the nearest ``\\begin{axis}[...]`` preceding ``pos``."""
    best = None
    for m in _AXIS_BEGIN_RE.finditer(tex, 0, pos):
        best = m
    if best is None:
        return {}
    opts = tex[best.end(): min(best.end() + 2000, len(tex))]
    hints: dict[str, str] = {}
    for om in _AXIS_OPT_RE.finditer(opts):
        hints[om.group(1).lower()] = sanitize_snippet(om.group(2).strip("{} "), 120)
    return hints


@dataclass
class SourceCandidate:
    """One candidate curve-data source found in an e-print."""

    rel_path: str                 # file path in the source tree, or "<tex>#coords<i>"
    kind: str                     # "pgfplots_ref" | "loose_file" | "inline_coordinates"
    rows: list = field(repr=False, default_factory=list)  # list[tuple[float, ...]]
    n_cols: int = 0
    referenced: bool = False      # named by an \addplot table / pgfplotstableread
    axis_hints: dict = field(default_factory=dict)   # xlabel/ylabel/xmode/ymode (sanitized)
    column_labels: Optional[list] = None  # lowercased header labels, if the file has them
    score: float = 0.0

    @property
    def n_rows(self) -> int:
        return len(self.rows)


_FILENAME_TOKEN_RE = re.compile(
    r"limit|exclusion|bound|constraint|sensitivity|proj|fig\d|figure\d", re.IGNORECASE)


def _plausible_ranges(rows: list, valid_for_ct: Optional[dict]) -> bool:
    """Median (col0, col1) inside the x10-widened VALID_RANGES window."""
    if not valid_for_ct or not rows:
        return False
    xs = sorted(r[0] for r in rows if r[0] > 0)
    ys = sorted(r[1] for r in rows if len(r) > 1 and r[1] > 0)
    if not xs or not ys:
        return False
    mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
    (m_lo, m_hi), (c_lo, c_hi) = valid_for_ct["mass"], valid_for_ct["coupling"]
    return (m_lo / 10 <= mx <= m_hi * 10) and (c_lo / 10 <= my <= c_hi * 10)


def score_candidate(cand: SourceCandidate, *, valid_for_ct: Optional[dict] = None) -> float:
    """Deterministic rank: referenced-by-figure > filename tokens > 2 columns >
    plausible value ranges (per the WS1 plan)."""
    s = 0.0
    if cand.referenced:
        s += 4.0
    if _FILENAME_TOKEN_RE.search(cand.rel_path):
        s += 2.0
    if cand.n_cols == 2:
        s += 1.0
    if _plausible_ranges(cand.rows, valid_for_ct):
        s += 1.0
    return s


def _read_text(path: Path, cap: int) -> Optional[str]:
    try:
        if path.stat().st_size > cap:
            return None
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def scan_candidates(src_dir: Path, *, valid_for_ct: Optional[dict] = None,
                    max_candidates: int = 60) -> list[SourceCandidate]:
    """Deterministic candidate scan over an unpacked source tree.

    Finds (a) data files referenced by pgfplots commands, (b) loose
    ``.dat/.csv/.txt/.tsv`` files that parse as >= 2 numeric columns, and
    (c) inline ``coordinates {...}`` blocks. Returns candidates sorted by
    :func:`score_candidate`, best first, capped at ``max_candidates``.
    """
    src_dir = Path(src_dir)
    referenced: dict[str, dict] = {}   # normalized rel path -> axis hints
    inline: list[SourceCandidate] = []

    tex_files = [p for p in sorted(src_dir.rglob("*")) if p.suffix.lower() in _TEX_SUFFIXES]
    for tex_path in tex_files:
        tex = _read_text(tex_path, MAX_TEX_FILE_BYTES)
        if tex is None:
            continue
        for pat in (_ADDPLOT_TABLE_RE, _PGFTABLEREAD_RE):
            for m in pat.finditer(tex):
                ref = m.group(1).strip()
                if not ref or "\\" in ref:  # macro-built path — unresolvable
                    continue
                referenced.setdefault(ref, _axis_hints_before(tex, m.start()))
        for i, m in enumerate(_COORDINATES_RE.finditer(tex)):
            pairs = []
            for pm in _COORD_PAIR_RE.finditer(m.group(1)):
                try:
                    pairs.append((float(pm.group(1)), float(pm.group(2))))
                except ValueError:
                    pairs = []
                    break
            if len(pairs) >= 3:
                inline.append(SourceCandidate(
                    rel_path=f"{tex_path.relative_to(src_dir)}#coords{i}",
                    kind="inline_coordinates", rows=pairs, n_cols=2,
                    referenced=True,
                    axis_hints=_axis_hints_before(tex, m.start())))

    candidates: list[SourceCandidate] = list(inline)
    seen: set[str] = set()
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _DATA_SUFFIXES:
            continue
        rel = str(path.relative_to(src_dir))
        text = _read_text(path, MAX_DATA_FILE_BYTES)
        if text is None:
            continue
        rows = parse_numeric_table(text)
        if rows is None:
            continue
        # match any reference by exact rel path, basename, or suffix-less form
        ref_hit = None
        for ref in referenced:
            rp = ref.lstrip("./")
            if rel == rp or Path(rel).name == Path(rp).name or rel == rp + path.suffix:
                ref_hit = ref
                break
        candidates.append(SourceCandidate(
            rel_path=rel, kind="pgfplots_ref" if ref_hit else "loose_file",
            rows=rows, n_cols=len(rows[0]), referenced=ref_hit is not None,
            axis_hints=referenced.get(ref_hit, {}) if ref_hit else {},
            column_labels=extract_column_labels(text, len(rows[0]))))
        seen.add(rel)

    for c in candidates:
        c.score = score_candidate(c, valid_for_ct=valid_for_ct)
    candidates.sort(key=lambda c: (-c.score, c.rel_path))
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Runtime curve pick (selector integration — no oracle, no LLM)
# ---------------------------------------------------------------------------

def extract_column_labels(text: str, ncols: int) -> Optional[list]:
    """Column labels from the header line directly above the first data row.

    Two accepted forms: a delimiter-split header whose token count equals the
    column count (``mass_kev,limit,s2_roi_min,...``), and a prose comment
    header that opens with ``mass [unit]`` (``# mass [eV]  photon coupling
    ...``) — mapped to ``["mass_<unit>", "limit", ...]``. Anything else
    returns ``None``: an unrecognized header must make the caller fall back
    to structural heuristics, never guess.
    """
    lines = text.splitlines()[:200]
    first_data = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        parts = [p for p in _SPLIT_RE.split(s) if p]
        try:
            if len(parts) >= 2:
                [float(p) for p in parts]
                first_data = i
                break
        except ValueError:
            pass
    if first_data is None:
        return None
    for j in range(first_data - 1, -1, -1):
        s = lines[j].strip().lstrip("#%/ ").strip()
        if not s:
            continue
        for parts in (
            [p.strip().lower() for p in s.split(",")],
            [p.strip().lower() for p in s.split()],
        ):
            if len(parts) == ncols and parts[0]:
                try:
                    [float(p) for p in parts]
                except ValueError:
                    return parts  # non-numeric header with matching arity
                return None      # a stray numeric line — not a header
        m = re.match(r"(?:m|mass)\s*\[\s*([a-zA-Zµμ]+)\s*\]", s)
        if m:
            return [f"mass_{m.group(1).lower()}"] + ["limit"] * (ncols - 1)
        return None  # nearest non-blank line is not a recognizable header
    return None


# Mass units a file header may declare, in eV. Deliberately excludes tokens
# that are case-ambiguous once lowercased ("mev": meV vs MeV differ by 9 dex,
# "pev": peV vs PeV) — an ambiguous unit makes the candidate fall through.
_MASS_UNIT_EV: dict[str, float] = {
    "ev": 1.0, "kev": 1e3, "gev": 1e9, "tev": 1e12,
    "uev": 1e-6, "µev": 1e-6, "μev": 1e-6, "nev": 1e-9,
}

_MASS_LABEL_RE = re.compile(r"^(?:mass|m)(?:[_\s\[\(]|$)")
# any eV-suffixed token in the label ("kev", "mev", "ev", ...); the token is
# then looked up in _MASS_UNIT_EV — an eV-ish token NOT in the map (case-
# ambiguous "mev"/"pev", unknown prefixes) disqualifies the column entirely.
_MASS_UNIT_IN_LABEL_RE = re.compile(r"(?:^|[_\s\[\(])([a-zµμ]*ev)(?:[\]\)\s_]|$)")
_Y_LABEL_RE = re.compile(r"limit|coupl|bound|excl|mixing|\bg_?[a-z]{1,4}\b|chi|kappa|epsilon")


# Filename evidence that a data file belongs to THIS coupling's limit curve.
# Short tokens match whole filename parts (split on non-alphanumerics, so
# "alp" matches 5e_results_alp.csv but not "alpha"); tokens of >= 5 chars
# also match as substrings of the normalized path ("darkphoton" inside
# DarkPhoton_DM_Constraint_Summary.dat).
_CT_FILENAME_TOKENS: dict[str, tuple[str, ...]] = {
    "DarkPhoton": ("darkphoton", "hiddenphoton", "kineticmixing", "paraphoton"),
    "AxionPhoton": ("axionphoton", "gagg", "gag", "gagamma", "axion"),
    "AxionElectron": ("axionelectron", "gae", "alp", "axion"),
    "AxionNeutron": ("axionneutron", "gan", "gann", "axion"),
    "AxionProton": ("axionproton", "gap", "axion"),
    "AxionEDM": ("axionedm", "edm", "axion"),
    "ScalarPhoton": ("scalarphoton", "scalar", "dilaton"),
    "ScalarElectron": ("scalarelectron", "scalar", "dme"),
    "ScalarNucleon": ("scalarnucleon", "scalar"),
    "ScalarBaryon": ("scalarbaryon", "scalar"),
    "VectorBL": ("vectorbl", "gbl", "bminusl"),
    "VectorB-L": ("vectorbl", "gbl", "bminusl"),
}


def _filename_matches_ct(rel_path: str, coupling_type: Optional[str]) -> bool:
    tokens = _CT_FILENAME_TOKENS.get(coupling_type or "", ())
    if not tokens:
        return False
    norm = re.sub(r"[^a-z0-9]+", " ", rel_path.lower())
    parts = set(norm.split())
    joined = norm.replace(" ", "")
    for t in tokens:
        if (t in parts) or (len(t) >= 5 and t in joined):
            return True
    return False


def best_curve_candidate(candidates: list[SourceCandidate],
                         valid_for_ct: Optional[dict],
                         coupling_type: Optional[str] = None,
                         *, min_rows: int = 3,
                         ) -> Optional[tuple[list[tuple[float, float]], SourceCandidate]]:
    """Pick the (points, candidate) the runtime channel should emit, or None.

    Deliberately conservative — the channel emits only when TWO independent
    deterministic signals agree, otherwise it falls through to the other
    extraction channels:

    1. **identity units in range**: x = column 0 and the first y column whose
       (median mass, median coupling) sit inside the STRICT ``VALID_RANGES``
       window. Files needing a unit transform (GHz->eV, log10 columns, g^2)
       fail the window — no conversion is attempted here (#657: that is the
       registry's job; the ceiling survey showed real hits are overwhelmingly
       identity-unit ``anc/`` files).
    2. **the filename names the coupling** (:data:`_CT_FILENAME_TOKENS`):
       being in-range alone is NOT evidence of being the right curve — the
       heuristic check found in-range picks of a *different figure's* data
       (1907.11485 5a nuclear-recoil file, 2402.07976 decay_rate_general)
       that would have been emitted at the top tier.
    """
    if not valid_for_ct:
        return None
    (m_lo, m_hi) = valid_for_ct["mass"]
    (c_lo, c_hi) = valid_for_ct["coupling"]

    def _in_range(pts):
        if len(pts) < min_rows:
            return False
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        return (m_lo <= mx <= m_hi) and (c_lo <= my <= c_hi)

    for cand in candidates:
        if not _filename_matches_ct(cand.rel_path, coupling_type):
            continue
        rows = cand.rows
        if len(rows) < min_rows:
            continue
        labels = cand.column_labels
        if labels:
            # Header labels are authoritative: x = the mass-labeled column
            # (converted by its declared unit — the file's own header is the
            # same evidence class as the GT ingester's header conversions),
            # y = the first limit/coupling-labeled column. No label match ->
            # fall through; never guess against an explicit header.
            xi = x_scale = None
            for i, lab in enumerate(labels):
                if _MASS_LABEL_RE.match(lab):
                    um = _MASS_UNIT_IN_LABEL_RE.search(lab)
                    # no unit token -> identity; a unit token that is not in
                    # the map (ambiguous "mev"/"pev", unknown) -> skip file.
                    scale = _MASS_UNIT_EV.get(um.group(1)) if um else 1.0
                    if scale is not None:
                        xi, x_scale = i, scale
                    break  # first mass column decides
            if xi is None:
                continue
            yi = next((i for i, lab in enumerate(labels)
                       if i != xi and _Y_LABEL_RE.search(lab)), None)
            if yi is None:
                continue
            pts = [(float(r[xi]) * x_scale, float(r[yi])) for r in rows
                   if len(r) > max(xi, yi) and r[xi] > 0 and r[yi] > 0]
            if _in_range(pts):
                return pts, cand
            continue
        # No header: only an unambiguous two-column (x, y) curve file is
        # trusted, identity units, strict-range checked. Wider unlabeled
        # tables fall through — the heuristic check caught in-range picks of
        # non-mass columns (keV/eV twins, recoil energies) in such files.
        if cand.n_cols == 2:
            pts = [(float(r[0]), float(r[1])) for r in rows
                   if r[0] > 0 and r[1] > 0]
            if _in_range(pts):
                return pts, cand
    return None


# ---------------------------------------------------------------------------
# Channel entry point
# ---------------------------------------------------------------------------

def scan_arxiv_source(arxiv_id: str, workdir: Path, *,
                      valid_for_ct: Optional[dict] = None) -> list[SourceCandidate]:
    """Fetch + unpack + scan one paper's e-print. NEVER raises: withdrawn
    source, PDF-only submission, tar bombs, or any internal error log and
    return ``[]`` so the existing extraction channels run unaffected."""
    try:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        blob = download_source(arxiv_id, workdir)
        src_dir = workdir / f"{arxiv_id.replace('/', '_')}_src"
        files = extract_source(blob, src_dir)
        if not files:
            logger.info("%s: no extractable source (PDF-only or empty)", arxiv_id)
            return []
        return scan_candidates(src_dir, valid_for_ct=valid_for_ct)
    except Exception as e:
        logger.warning("source-data channel failed for %s (falling through): %s",
                       arxiv_id, str(e)[:200])
        return []
