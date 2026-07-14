"""Subscription-window exhaustion must be FATAL, never a per-stage failure.

2026-07-14 incident: a full-benchmark run hit the Max window mid-run and the
CLI reported "You've hit your session limit · resets 4:10pm (America/Chicago)"
— a marker variant `_USAGE_LIMIT_MARKERS` did not know. `_raise_for_text` fell
through to the generic RuntimeError, `_call_with_retry` treated it as a
transient stage failure, and 313/347 extractions fail-opened to EMPTY results
(the #648 failure class: an availability error is a property of the run,
never the paper). These tests pin every observed CLI limit phrasing to the
FatalAPIError path.
"""

import pytest

from pipeline.cli_client import ClaudeCLIClient
from pipeline.extractor import FatalAPIError


def _client():
    # _raise_for_text does not touch instance state beyond self, so a bare
    # object works regardless of constructor requirements.
    return object.__new__(ClaudeCLIClient)


# Every phrasing the CLI has been observed (or documented) to emit on window
# exhaustion. The 2026-07-14 incident string is verbatim.
LIMIT_TEXTS = [
    "You've hit your session limit · resets 4:10pm (America/Chicago)",
    "You've hit your limit · resets 6pm",
    "Usage limit reached — your limit resets at 3:00 PM",
    "5-hour limit reached",
    "Weekly limit reached, reset at Monday 9am",
]


@pytest.mark.parametrize("text", LIMIT_TEXTS)
def test_window_exhaustion_is_fatal(text):
    with pytest.raises(FatalAPIError):
        _client()._raise_for_text(text, 1)


def test_transient_errors_stay_transient():
    # Rate limit and overload must remain retryable anthropic errors, not
    # fatal — only genuine window exhaustion aborts the run.
    import anthropic
    with pytest.raises(anthropic.APIStatusError):
        _client()._raise_for_text("429 too many requests", 1)
    with pytest.raises(anthropic.APIStatusError):
        _client()._raise_for_text("overloaded_error", 1)


def test_unrelated_failure_not_swallowed_as_fatal():
    # A garden-variety CLI crash raises the generic RuntimeError (retried as
    # transient), NOT FatalAPIError.
    with pytest.raises(RuntimeError) as ei:
        _client()._raise_for_text("segmentation fault", 1)
    assert not isinstance(ei.value, FatalAPIError)
