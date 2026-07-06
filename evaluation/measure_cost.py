"""Measure REAL per-paper extraction cost on the production path (Opus, N=1).

Runs ``pipeline.extractor.run_extraction_agent`` on a handful of papers,
monkeypatching ``client.messages.create`` to accumulate per-call token usage,
then prices it at published Claude Opus 4.8 rates (standard / non-batch, i.e.
what the live daily pipeline actually pays):

    input  $5 / 1M      output $25 / 1M
    cache write (1h TTL) $10 / 1M   cache read $0.50 / 1M

This exists to settle "what does one extraction actually cost" with a direct
measurement rather than a bottom-up token estimate. Point it at real arXiv IDs
via ``MEASURE_ARXIV_IDS`` (comma-separated); it downloads each PDF, runs the
single-read production path, and prints a per-paper token + cost breakdown plus
a grand total. The account's credit-balance delta is the independent check.

Usage (locally or in CI, with ANTHROPIC_API_KEY set):
    EXTRACTOR_MODEL=claude-opus-4-8 python -m evaluation.measure_cost
    MEASURE_ARXIV_IDS=2204.03818,2303.08666 python -m evaluation.measure_cost
"""
import os
import sys
import tempfile
from pathlib import Path

# Production single-read path at standard (non-batch) price.
os.environ.setdefault("EXTRACTOR_MODEL", "claude-opus-4-8")
os.environ.pop("AAL_BATCH", None)

import anthropic  # noqa: E402
from pipeline.extractor import run_extraction_agent, download_pdf  # noqa: E402

DEFAULT_IDS = [
    "2606.26253", "2406.16840", "2204.03818", "2303.08666", "2012.10764",
]
PAPERS = [s.strip() for s in os.environ.get("MEASURE_ARXIV_IDS", "").split(",") if s.strip()] \
    or DEFAULT_IDS

# Claude Opus 4.8 pricing ($/token). Cache write is 1h-TTL (2x); read is 0.1x.
P_IN, P_OUT, P_CW, P_CR = 5e-6, 25e-6, 10e-6, 0.5e-6


class _PaperStub:
    def __init__(self, arxiv_id, title, summary=""):
        self.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
        self.title = title
        self.summary = summary
        self.categories = []
        self.arxiv_id = arxiv_id

    def get_short_id(self):
        return self.arxiv_id


def _cost(rows):
    return sum(i * P_IN + o * P_OUT + cw * P_CW + cr * P_CR
              for _, _, i, o, cw, cr in rows)


def main():
    model = os.environ["EXTRACTOR_MODEL"]
    print(f"model={model}  papers={PAPERS}\n")

    client = anthropic.Anthropic()
    calls = []                      # (paper, model, in, out, cache_write, cache_read)
    cur = {"id": None}
    orig = client.messages.create

    def wrapped(*a, **k):
        r = orig(*a, **k)
        u = r.usage
        calls.append((
            cur["id"], k.get("model", "?"),
            u.input_tokens, u.output_tokens,
            getattr(u, "cache_creation_input_tokens", 0) or 0,
            getattr(u, "cache_read_input_tokens", 0) or 0,
        ))
        return r

    client.messages.create = wrapped

    hdr = f"{'paper':>12} {'calls':>5} {'in':>10} {'cache_wr':>10} {'cache_rd':>10} {'out':>8} {'cost$':>9}"
    print(hdr)
    grand = []
    for pid in PAPERS:
        cur["id"] = pid
        before = len(calls)
        try:
            with tempfile.TemporaryDirectory() as td:
                pdf = download_pdf(pid, Path(td))
                run_extraction_agent(_PaperStub(pid, title=pid, summary=""), pdf, client)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{pid:>12}  ERROR {type(e).__name__}: {str(e)[:90]}")
        rows = calls[before:]
        grand += rows
        ti = sum(r[2] for r in rows); to = sum(r[3] for r in rows)
        tw = sum(r[4] for r in rows); tr = sum(r[5] for r in rows)
        print(f"{pid:>12} {len(rows):>5} {ti:>10,} {tw:>10,} {tr:>10,} {to:>8,} {_cost(rows):>9.4f}")

    print("-" * len(hdr))
    ti = sum(r[2] for r in grand); to = sum(r[3] for r in grand)
    tw = sum(r[4] for r in grand); tr = sum(r[5] for r in grand)
    tot = _cost(grand)
    n = len(PAPERS)
    print(f"{'TOTAL':>12} {len(grand):>5} {ti:>10,} {tw:>10,} {tr:>10,} {to:>8,} {tot:>9.4f}")
    print(f"\nTOTAL cost (token math): ${tot:.4f} over {n} papers")
    print(f"MEAN per-paper:          ${tot / n:.4f}")
    print(f"tokens: input(uncached)={ti:,}  cache_write={tw:,}  cache_read={tr:,}  output={to:,}")


if __name__ == "__main__":
    sys.exit(main())
