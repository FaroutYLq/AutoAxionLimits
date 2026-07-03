"""Message-Batches transport shim: same agent code, half price.

The extraction agent is a sequential per-paper chain (classify -> stage 1 ->
stage 2a -> stage 2 -> verify -> ...), so it cannot be submitted to the
Batches API as one item. Rewriting it as stage-waves would duplicate the
orchestration and invite drift. This module takes the other path: run many
paper-agents concurrently (threads) against a client whose
``messages.create`` BLOCKS, while a dispatcher coalesces the pending calls of
ALL threads into heterogeneous Message Batches (50% price), submits, polls,
and hands each thread its own result.

Transport-only guarantee: the request params each thread passes to
``create(**kwargs)`` are forwarded VERBATIM as one batch item — same prompts,
same model, same order of stages per paper — so the extraction distribution
is unchanged; only latency and price differ.

Error mapping preserves the ``_call_with_retry`` / #648 contract:

* billing / auth item errors  -> :class:`pipeline.extractor.FatalAPIError`
* rate-limit / overloaded     -> transparent re-enqueue into the next batch
  (up to ``max_item_attempts``)
* other item errors           -> raised in the calling thread

Enable in the eval driver with ``AAL_BATCH=1`` (see
``evaluation.evaluate.run_extraction``). Not used by the daily pipeline
(latency-sensitive, low volume).
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RETRYABLE_ERROR_TYPES = {"rate_limit_error", "overloaded_error", "api_error"}
_FATAL_MARKERS = ("credit balance", "billing", "authentication", "permission")


@dataclass
class _Pending:
    kwargs: dict
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None
    attempts: int = 0


class _Messages:
    def __init__(self, outer: "BatchingClient"):
        self._outer = outer

    def create(self, **kwargs):
        return self._outer._submit_and_wait(kwargs)


class BatchingClient:
    """Drop-in ``client`` whose ``messages.create`` rides Message Batches.

    ``flush_after_s``: a batch is submitted once no NEW call has arrived for
    this long (lets a wave of concurrent threads fully assemble) or when
    ``max_batch`` items are pending. ``poll_s``: batch status poll interval.
    """

    def __init__(self, api_client=None, *, flush_after_s: float = 15.0,
                 max_batch: int = 500, poll_s: float = 20.0,
                 max_item_attempts: int = 3):
        import anthropic
        self._api = api_client or anthropic.Anthropic()
        self.messages = _Messages(self)
        self._flush_after_s = flush_after_s
        self._max_batch = max_batch
        self._poll_s = poll_s
        self._max_item_attempts = max_item_attempts
        self._lock = threading.Condition()
        self._queue: list[_Pending] = []
        self._last_arrival = 0.0
        self._seq = itertools.count()
        self._fatal: Optional[BaseException] = None
        self._stats = {"batches": 0, "items": 0, "in": 0, "out": 0,
                       "cache_read": 0, "cache_write": 0}
        self._dispatcher = threading.Thread(target=self._run, daemon=True,
                                            name="batch-dispatcher")
        self._dispatcher.start()

    # ------------------------------------------------------------- caller side
    def _submit_and_wait(self, kwargs: dict):
        p = _Pending(kwargs=kwargs)
        with self._lock:
            if self._fatal is not None:
                raise self._fatal
            self._queue.append(p)
            self._last_arrival = time.monotonic()
            self._lock.notify_all()
        p.event.wait()
        if p.error is not None:
            raise p.error
        return p.result

    # --------------------------------------------------------- dispatcher side
    def _take_batch(self) -> list[_Pending]:
        with self._lock:
            while True:
                if self._fatal is not None:
                    return []
                if self._queue:
                    quiet = time.monotonic() - self._last_arrival
                    if len(self._queue) >= self._max_batch or quiet >= self._flush_after_s:
                        batch = self._queue[: self._max_batch]
                        del self._queue[: len(batch)]
                        return batch
                    self._lock.wait(timeout=max(0.25, self._flush_after_s - quiet))
                else:
                    self._lock.wait(timeout=1.0)

    def _run(self):
        while True:
            batch = self._take_batch()
            if not batch:
                if self._fatal is not None:
                    return
                continue
            try:
                self._process(batch)
            except Exception as e:  # dispatcher must never die silently
                logger.exception("batch dispatcher error: %s", e)
                self._abort(e, batch)

    def _abort(self, exc: BaseException, batch: list[_Pending]):
        with self._lock:
            self._fatal = exc
            pending = self._queue
            self._queue = []
        for p in batch + pending:
            p.error = exc
            p.event.set()

    def _process(self, batch: list[_Pending]):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        by_id = {}
        requests = []
        for p in batch:
            cid = f"c{next(self._seq)}"
            by_id[cid] = p
            requests.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(**p.kwargs)))
        mb = self._api.messages.batches.create(requests=requests)
        logger.info("batch %s submitted: %d items", mb.id, len(requests))
        while True:
            mb = self._api.messages.batches.retrieve(mb.id)
            if mb.processing_status == "ended":
                break
            time.sleep(self._poll_s)

        retry: list[_Pending] = []
        for r in self._api.messages.batches.results(mb.id):
            p = by_id.pop(r.custom_id)
            kind = r.result.type
            if kind == "succeeded":
                msg = r.result.message
                u = msg.usage
                self._stats["items"] += 1
                self._stats["in"] += u.input_tokens
                self._stats["out"] += u.output_tokens
                self._stats["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                self._stats["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                p.result = msg
                p.event.set()
                continue
            err = getattr(r.result, "error", None)
            etype = getattr(getattr(err, "error", err), "type", "") or ""
            emsg = str(getattr(getattr(err, "error", err), "message", err))
            low = emsg.lower()
            if any(m in low for m in _FATAL_MARKERS):
                from pipeline.extractor import FatalAPIError
                exc = FatalAPIError(f"API availability error — batch item: {emsg}")
                self._abort(exc, [p] + list(by_id.values()) + retry)
                return
            p.attempts += 1
            if (kind in ("errored",) and etype in _RETRYABLE_ERROR_TYPES
                    and p.attempts < self._max_item_attempts) or kind == "expired":
                retry.append(p)
            else:
                p.error = RuntimeError(f"batch item {kind}: {etype}: {emsg[:300]}")
                p.event.set()
        # anything the results stream never mentioned (shouldn't happen)
        for p in by_id.values():
            p.error = RuntimeError("batch item missing from results stream")
            p.event.set()
        self._stats["batches"] += 1
        if retry:
            logger.info("re-enqueueing %d retryable items", len(retry))
            with self._lock:
                self._queue.extend(retry)
                self._last_arrival = time.monotonic()
                self._lock.notify_all()

    # ------------------------------------------------------------------ stats
    @property
    def stats(self) -> dict:
        return dict(self._stats)


_shared: Optional[BatchingClient] = None
_shared_lock = threading.Lock()


def get_shared_batching_client() -> BatchingClient:
    """Process-wide singleton so all paper-threads share one dispatcher."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = BatchingClient()
        return _shared
