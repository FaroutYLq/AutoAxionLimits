"""Unit tests for the Message-Batches transport shim (pipeline/batch_client.py).

No network: a fake API implements the batches endpoints. Pins the contract:
concurrent ``create`` calls coalesce into one batch with VERBATIM params
(transport-only guarantee), per-item error mapping preserves the #648
semantics (billing -> FatalAPIError aborting everything; rate-limit ->
transparent re-enqueue), and results route back to the right caller.

Run:
    pytest evaluation/tests/test_batch_client.py -v
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.batch_client import BatchingClient
from pipeline.extractor import FatalAPIError


def _msg(text):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2,
                              cache_read_input_tokens=0,
                              cache_creation_input_tokens=0))


class FakeBatchesAPI:
    """Succeeds every item by echoing its prompt; scriptable per-item errors."""

    def __init__(self, item_errors=None):
        self.submitted: list[list] = []
        self.item_errors = dict(item_errors or {})  # prompt-text -> (kind, etype, emsg)
        b = SimpleNamespace(create=self._create, retrieve=self._retrieve,
                            results=self._results)
        self.messages = SimpleNamespace(batches=b)
        self._last = None

    def _create(self, requests):
        self.submitted.append(list(requests))
        self._last = requests
        return SimpleNamespace(id=f"b{len(self.submitted)}", processing_status="ended")

    def _retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended")

    def _results(self, bid):
        out = []
        for r in self.submitted[int(bid[1:]) - 1]:
            prompt = str(r["params"]["messages"])
            hit = next((v for k, v in self.item_errors.items() if k in prompt), None)
            if hit:
                kind, etype, emsg = hit
                if self.item_errors.pop(next(k for k in self.item_errors if k in prompt), None) is not None:
                    pass  # scripted errors fire once, then succeed on retry
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
                    type=kind, error=SimpleNamespace(error=SimpleNamespace(type=etype, message=emsg)))))
            else:
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
                    type="succeeded", message=_msg(prompt))))
        return iter(out)


def _client(fake, **kw):
    kw.setdefault("flush_after_s", 0.15)
    kw.setdefault("poll_s", 0.01)
    return BatchingClient(api_client=fake, **kw)


def _call_many(bc, prompts):
    results = {}
    def go(p):
        results[p] = bc.messages.create(model="m", max_tokens=8,
                                        messages=[{"role": "user", "content": p}])
    ts = [threading.Thread(target=go, args=(p,)) for p in prompts]
    for t in ts: t.start()
    for t in ts: t.join(timeout=20)
    return results


def test_concurrent_calls_coalesce_and_route_back():
    fake = FakeBatchesAPI()
    bc = _client(fake)
    res = _call_many(bc, [f"prompt-{i}" for i in range(6)])
    assert len(fake.submitted) == 1              # one batch for the wave
    assert len(fake.submitted[0]) == 6
    for p, msg in res.items():                   # each caller got ITS result
        assert p in msg.content[0].text
    assert bc.stats["items"] == 6


def test_params_forwarded_verbatim():
    fake = FakeBatchesAPI()
    bc = _client(fake)
    kwargs = {"model": "claude-x", "max_tokens": 77, "temperature": 0.0,
              "system": "sys", "messages": [{"role": "user", "content": "hi"}]}
    bc.messages.create(**kwargs)
    sent = fake.submitted[0][0]["params"]
    for k, v in kwargs.items():
        assert sent[k] == v, k


def test_billing_error_becomes_fatal_for_everyone():
    fake = FakeBatchesAPI(item_errors={
        "poison": ("errored", "invalid_request_error",
                   "Your credit balance is too low to access the Anthropic API.")})
    bc = _client(fake)
    errs = {}
    def go(p):
        try:
            bc.messages.create(model="m", max_tokens=8,
                               messages=[{"role": "user", "content": p}])
            errs[p] = None
        except BaseException as e:
            errs[p] = e
    ts = [threading.Thread(target=go, args=(p,)) for p in ("poison", "innocent")]
    for t in ts: t.start()
    for t in ts: t.join(timeout=20)
    assert isinstance(errs["poison"], FatalAPIError)
    # subsequent calls fail fast too
    with pytest.raises(FatalAPIError):
        bc.messages.create(model="m", max_tokens=8,
                           messages=[{"role": "user", "content": "later"}])


def test_rate_limited_item_retries_transparently():
    fake = FakeBatchesAPI(item_errors={
        "flaky": ("errored", "rate_limit_error", "Too many requests")})
    bc = _client(fake)
    msg = bc.messages.create(model="m", max_tokens=8,
                             messages=[{"role": "user", "content": "flaky"}])
    assert "flaky" in msg.content[0].text        # succeeded on the retry batch
    assert len(fake.submitted) == 2


def test_non_retryable_item_error_raises_only_that_caller():
    fake = FakeBatchesAPI(item_errors={
        "bad": ("errored", "invalid_request_error", "max_tokens too large")})
    bc = _client(fake)
    res = {}
    def go(p):
        try:
            m = bc.messages.create(model="m", max_tokens=8,
                                   messages=[{"role": "user", "content": p}])
            res[p] = m
        except BaseException as e:
            res[p] = e
    ts = [threading.Thread(target=go, args=(p,)) for p in ("bad", "fine")]
    for t in ts: t.start()
    for t in ts: t.join(timeout=20)
    assert isinstance(res["bad"], RuntimeError)
    assert not isinstance(res["fine"], BaseException)
