"""Keep all durable pipeline state unchanged during an extraction preview.

The context covers nested extraction calls as well as the entrypoint's own
bookkeeping. Downloads and logs may still be written; production state may not.
"""

from contextvars import ContextVar
from functools import wraps
from inspect import signature


_preview = ContextVar("pipeline_preview", default=False)


def preview_run(function):
    """Scope state suppression to a call with ``dry_run=True`` (also positional)."""
    parameters = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        arguments = parameters.bind(*args, **kwargs).arguments
        token = _preview.set(_preview.get() or bool(arguments.get("dry_run", False)))
        try:
            return function(*args, **kwargs)
        finally:
            _preview.reset(token)

    return wrapped


def state_writer(function):
    """Suppress a durable state write inside a preview, including on failure."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not _preview.get():
            return function(*args, **kwargs)
        return None

    return wrapped
