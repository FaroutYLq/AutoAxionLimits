"""Unit tests for the WS1 arXiv source-tarball channel (pipeline/source_data.py).

No network calls: every archive is synthesized in-test. Pins the safety
guards (traversal, links, member/size caps, gzip bombs), the numeric-table
parser, the pgfplots scanner + ranking, and the never-raise channel contract.

Run:
    pytest evaluation/tests/test_source_data.py -v
"""

from __future__ import annotations

import gzip
import io
import sys
import tarfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import source_data as sd
from pipeline.source_data import (
    SourceDataError,
    extract_source,
    parse_numeric_table,
    sanitize_snippet,
    scan_arxiv_source,
    scan_candidates,
    score_candidate,
    SourceCandidate,
)


def make_tar(files: dict[str, bytes], *, gz: bool = True,
             extra_members: list[tarfile.TarInfo] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for info in extra_members or []:
            tf.addfile(info)
    raw = buf.getvalue()
    return gzip.compress(raw) if gz else raw


DATA = b"# mass coupling\n1e-6 3e-11\n2e-6 4e-11\n3e-6 5e-11\n"


# ---------------------------------------------------------------------------
# extract_source safety guards
# ---------------------------------------------------------------------------

class TestExtractSource:
    def test_normal_gzip_tar(self, tmp_path):
        blob = tmp_path / "b.eprint"
        blob.write_bytes(make_tar({"main.tex": b"\\documentclass{article}",
                                   "data/curve.dat": DATA}))
        out = extract_source(blob, tmp_path / "src")
        assert sorted(p.name for p in out) == ["curve.dat", "main.tex"]
        assert (tmp_path / "src" / "data" / "curve.dat").read_bytes() == DATA

    def test_pdf_only_submission(self, tmp_path):
        blob = tmp_path / "b.eprint"
        blob.write_bytes(b"%PDF-1.5 rest-of-pdf")
        assert extract_source(blob, tmp_path / "src") == []

    def test_gzipped_pdf(self, tmp_path):
        blob = tmp_path / "b.eprint"
        blob.write_bytes(gzip.compress(b"%PDF-1.5 rest"))
        assert extract_source(blob, tmp_path / "src") == []

    def test_gzipped_single_tex(self, tmp_path):
        blob = tmp_path / "b.eprint"
        blob.write_bytes(gzip.compress(b"\\documentclass{article} hello"))
        out = extract_source(blob, tmp_path / "src")
        assert [p.name for p in out] == ["main.tex"]

    def test_traversal_and_absolute_members_skipped(self, tmp_path):
        blob = tmp_path / "b.eprint"
        blob.write_bytes(make_tar({"../evil.txt": DATA, "/abs/evil.txt": DATA,
                                   "ok.dat": DATA}))
        out = extract_source(blob, tmp_path / "src")
        assert [p.name for p in out] == ["ok.dat"]
        assert not (tmp_path / "evil.txt").exists()

    def test_symlink_members_skipped(self, tmp_path):
        link = tarfile.TarInfo(name="link.dat")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        blob = tmp_path / "b.eprint"
        blob.write_bytes(make_tar({"ok.dat": DATA}, extra_members=[link]))
        out = extract_source(blob, tmp_path / "src")
        assert [p.name for p in out] == ["ok.dat"]

    def test_too_deep_member_skipped(self, tmp_path):
        deep = "/".join(["d"] * 12) + "/x.dat"
        blob = tmp_path / "b.eprint"
        blob.write_bytes(make_tar({deep: DATA, "ok.dat": DATA}))
        out = extract_source(blob, tmp_path / "src")
        assert [p.name for p in out] == ["ok.dat"]

    def test_member_count_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "MAX_MEMBERS", 2)
        blob = tmp_path / "b.eprint"
        blob.write_bytes(make_tar({f"f{i}.dat": DATA for i in range(3)}))
        with pytest.raises(SourceDataError):
            extract_source(blob, tmp_path / "src")

    def test_total_size_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "MAX_TOTAL_BYTES", 100)
        blob = tmp_path / "b.eprint"
        blob.write_bytes(make_tar({"a.dat": b"x" * 80, "b.dat": b"y" * 80}))
        with pytest.raises(SourceDataError):
            extract_source(blob, tmp_path / "src")

    def test_gzip_bomb_guard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sd, "MAX_TOTAL_BYTES", 1024)
        blob = tmp_path / "b.eprint"
        blob.write_bytes(gzip.compress(b"\x00" * 10_000))
        with pytest.raises(SourceDataError):
            extract_source(blob, tmp_path / "src")

    def test_empty_blob(self, tmp_path):
        blob = tmp_path / "b.eprint"
        blob.write_bytes(b"")
        assert extract_source(blob, tmp_path / "src") == []


# ---------------------------------------------------------------------------
# parse_numeric_table
# ---------------------------------------------------------------------------

class TestParseNumericTable:
    def test_two_columns_with_comments(self):
        rows = parse_numeric_table("# header\n% tex comment\n1e-6 3e-11\n"
                                   "2e-6 4e-11\n3e-6 5e-11\n")
        assert rows == [(1e-6, 3e-11), (2e-6, 4e-11), (3e-6, 5e-11)]

    def test_csv_and_semicolons(self):
        rows = parse_numeric_table("1,2\n3;4\n5 6\n")
        assert rows == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

    def test_prose_rejected(self):
        assert parse_numeric_table("This is a readme.\nNot data at all.\n") is None

    def test_mostly_numeric_required(self):
        text = "1 2\n3 4\nword salad here\nmore words\n5 6\n"
        assert parse_numeric_table(text) is None  # 3/5 numeric < 90%

    def test_single_column_rejected(self):
        assert parse_numeric_table("1\n2\n3\n4\n") is None

    def test_min_rows(self):
        assert parse_numeric_table("1 2\n3 4\n") is None

    def test_modal_column_trim(self):
        rows = parse_numeric_table("1 2 3\n4 5 6\n7 8 9\n")
        assert rows and all(len(r) == 3 for r in rows)


# ---------------------------------------------------------------------------
# scan_candidates + ranking
# ---------------------------------------------------------------------------

TEX = rb"""
\documentclass{article}
\begin{document}
\begin{figure}
\begin{tikzpicture}
\begin{axis}[xmode=log, ymode=log, xlabel={$m_a$ [eV]},
             ylabel={$g_{a\gamma}$ [GeV$^{-1}$]}]
\addplot[red] table {data/limit_curve.dat};
\addplot coordinates {(1e-6, 3e-11) (2e-6, 4e-11) (3e-6, 5e-11)};
\end{axis}
\end{tikzpicture}
\end{figure}
\end{document}
"""


class TestScanCandidates:
    @pytest.fixture()
    def tree(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "main.tex").write_bytes(TEX)
        (tmp_path / "data" / "limit_curve.dat").write_bytes(DATA)
        (tmp_path / "data" / "loose_table.txt").write_bytes(
            b"10 20 30\n40 50 60\n70 80 90\n")
        (tmp_path / "README.txt").write_bytes(b"just words here\nno data\n")
        return tmp_path

    def test_finds_and_ranks(self, tree):
        cands = scan_candidates(tree)
        paths = [c.rel_path for c in cands]
        assert "data/limit_curve.dat" in paths
        assert "data/loose_table.txt" in paths
        assert not any("README" in p for p in paths)
        # referenced + filename token beats the loose 3-column file
        assert cands[0].rel_path == "data/limit_curve.dat"
        assert cands[0].referenced and cands[0].kind == "pgfplots_ref"

    def test_axis_hints(self, tree):
        cands = scan_candidates(tree)
        best = cands[0]
        assert best.axis_hints.get("xmode") == "log"
        assert "eV" in best.axis_hints.get("xlabel", "")

    def test_inline_coordinates(self, tree):
        cands = scan_candidates(tree)
        inline = [c for c in cands if c.kind == "inline_coordinates"]
        assert len(inline) == 1
        assert inline[0].rows == [(1e-6, 3e-11), (2e-6, 4e-11), (3e-6, 5e-11)]

    def test_score_components(self):
        base = SourceCandidate(rel_path="x.dat", kind="loose_file",
                               rows=[(1.0, 2.0)] * 3, n_cols=2)
        named = SourceCandidate(rel_path="exclusion_fig3.dat", kind="loose_file",
                                rows=[(1.0, 2.0)] * 3, n_cols=2)
        ref = SourceCandidate(rel_path="x.dat", kind="pgfplots_ref",
                              rows=[(1.0, 2.0)] * 3, n_cols=2, referenced=True)
        assert score_candidate(ref) > score_candidate(named) > score_candidate(base)

    def test_plausible_range_bonus(self):
        vr = {"mass": (1e-12, 1e-2), "coupling": (1e-19, 1e-5)}
        inside = SourceCandidate(rel_path="a.dat", kind="loose_file",
                                 rows=[(1e-6, 3e-11)] * 3, n_cols=2)
        outside = SourceCandidate(rel_path="a.dat", kind="loose_file",
                                  rows=[(1e9, 3e4)] * 3, n_cols=2)
        assert score_candidate(inside, valid_for_ct=vr) == \
            score_candidate(outside, valid_for_ct=vr) + 1.0


class TestSanitize:
    def test_control_chars_stripped_and_truncated(self):
        assert sanitize_snippet("a\x00b\x1bc" + "d" * 500) == "abc" + "d" * 397
        assert sanitize_snippet(None) == ""


# ---------------------------------------------------------------------------
# channel never raises
# ---------------------------------------------------------------------------

class TestChannelContract:
    def test_download_failure_falls_through(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")
        monkeypatch.setattr(sd, "download_source", boom)
        assert scan_arxiv_source("1234.56789", tmp_path) == []

    def test_corrupt_blob_falls_through(self, tmp_path, monkeypatch):
        blob = tmp_path / "1234.56789.eprint"
        blob.write_bytes(b"\x1f\x8b corrupted gzip stream")
        monkeypatch.setattr(sd, "download_source", lambda *a, **k: blob)
        assert scan_arxiv_source("1234.56789", tmp_path) == []
