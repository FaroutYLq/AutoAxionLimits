"""Headless-``claude``-CLI transport shim: same agent code, subscription billing.

The whole model-call contract in this repo is a single method —
``client.messages.create(**kwargs) -> obj`` where every call site reads only
``obj.content[0].text`` (see ``pipeline.extractor``). :class:`ClaudeCLIClient`
implements exactly that surface by shelling out to ``claude -p`` (headless
Claude Code), so the daily / weekly / backfill pipelines can run on a Claude
Code subscription (keychain OAuth) instead of an ``ANTHROPIC_API_KEY``.

Transport-only guarantee: the request params each caller passes to
``create(**kwargs)`` — model, system prompt, user message (text + base64 PNG
image blocks) — are forwarded to the CLI VERBATIM. Nothing about the prompts,
stage flow, guards, or model selection changes; only the wire between this
process and Claude does. This mirrors the guarantee ``batch_client.py`` makes
for the Message-Batches transport.

Deltas from the SDK transport, all pre-existing no-ops for the production
model, documented so parity is auditable:

* ``temperature`` is dropped. The production model ``claude-opus-4-8`` already
  rejects it and ``extractor._create`` strips it via self-heal, so the SDK
  path also runs without it — no methodology delta for the default model.
* ``max_tokens`` is dropped (the CLI has no equivalent knob). The 128–2048
  caps are truncation guards, not methodology; the prompts already demand
  short JSON and ``_parse_json_response`` is fence/prose tolerant.
* ``cache_control`` markers are dropped (prompt caching is an API-billing
  optimisation with no CLI analogue).

Error mapping preserves the ``_call_with_retry`` / #648 contract:

* auth / billing / subscription-window-exhausted -> :class:`FatalAPIError`
  (the run aborts WITHOUT marking papers — it is an environment failure).
* rate-limit / overloaded / timeout -> a genuine ``anthropic`` error object
  so ``extractor._call_with_retry``'s ``isinstance`` checks give it backoff.
* anything else -> ``RuntimeError`` (paper-scoped, like a bad batch item).

Enable via ``AAL_BACKEND=claude-cli`` (see ``pipeline.client_factory``).
Prompt-injection defence: the paper text is untrusted, so the subprocess runs
with all built-in tools disabled (``--tools ""``) except Read on vision calls
(to load the PNGs we wrote), no MCP servers, no project settings, and inside a
throwaway cwd so repo ``CLAUDE.md`` is never auto-discovered.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Environment variables scrubbed from the subprocess. The first group would
# silently flip billing back to the API (a stray exported key routes every
# call through API billing instead of the subscription); the CLAUDE_CODE_*
# group leaks from the parent session because the skills launch these
# pipelines from INSIDE a Claude Code session, and could confuse the child.
_SCRUB_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDECODE",
)
# Prefix-scrubbed: every CLAUDE_CODE_* var inherited from the parent session.
_SCRUB_ENV_PREFIXES = ("CLAUDE_CODE_",)

# Built-in tools denied on every call (belt-and-suspenders over ``--tools ""``;
# the lockdown test asserts these are present in argv).
_DISALLOWED_TOOLS = (
    "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task,Glob,Grep,TodoWrite"
)

# stderr / result-text fragments that identify an ENVIRONMENT failure (fatal).
# Superset of batch_client._FATAL_MARKERS plus subscription-auth phrasing.
_FATAL_MARKERS = (
    "credit balance",
    "billing",
    "authentication",
    "invalid api key",
    "invalid x-api-key",
    "permission",
    "not logged in",
    "/login",
    "please run /login",
    "oauth token",
    "unauthorized",
)
# Subscription-window exhaustion: not recoverable inside _call_with_retry's
# four short backoffs, so treat as fatal (abort without marking papers, #648)
# rather than a retryable rate limit.
_USAGE_LIMIT_MARKERS = (
    "usage limit reached",
    "usage limit",
    "5-hour limit",
    "weekly limit",
    "reset at",
)
_RATELIMIT_MARKERS = ("rate limit", "too many requests", "429")
_OVERLOAD_MARKERS = ("overloaded", "529", "service unavailable")


def _fatal(message: str):
    """Build a :class:`FatalAPIError` (imported lazily to avoid an import cycle
    with ``extractor``, exactly as ``batch_client`` does)."""
    from pipeline.extractor import FatalAPIError

    return FatalAPIError(message)


def _anthropic_status_error(status: int, message: str):
    """A genuine ``anthropic`` error object over a synthetic httpx response, so
    ``extractor._call_with_retry``'s ``isinstance`` checks route it to backoff.
    Same construction the test-suite uses (test_fatal_api._api_error)."""
    import anthropic
    import httpx

    req = httpx.Request("POST", "claude-cli://local/messages")
    resp = httpx.Response(status, request=req)
    body = {"error": {"message": message}}
    if status == 429:
        return anthropic.RateLimitError(message, response=resp, body=body)
    return anthropic.APIStatusError(message, response=resp, body=body)


class _Messages:
    def __init__(self, outer: "ClaudeCLIClient"):
        self._outer = outer

    def create(self, **kwargs):
        return self._outer._run(kwargs)


class ClaudeCLIClient:
    """Drop-in ``client`` whose ``messages.create`` shells out to ``claude -p``.

    ``binary``: the CLI executable (default ``claude``). ``timeout_s`` /
    ``vision_timeout_s``: subprocess wall-clock caps (vision calls carry up to
    eight PNGs and take minutes); overridable via ``AAL_CLI_TIMEOUT`` /
    ``AAL_CLI_VISION_TIMEOUT``. ``scratch_root``: parent dir for per-call temp
    dirs (default the system temp dir).
    """

    def __init__(
        self,
        binary: str = "claude",
        *,
        timeout_s: int | None = None,
        vision_timeout_s: int | None = None,
        scratch_root: str | None = None,
    ):
        self.messages = _Messages(self)
        self._binary = binary
        self._timeout_s = int(
            timeout_s if timeout_s is not None
            else os.environ.get("AAL_CLI_TIMEOUT", "1200")
        )
        self._vision_timeout_s = int(
            vision_timeout_s if vision_timeout_s is not None
            else os.environ.get("AAL_CLI_VISION_TIMEOUT", "2700")
        )
        self._scratch_root = scratch_root
        # Models whose first successful call has passed the substitution check.
        self._verified_models: set[str] = set()

    # ---------------------------------------------------------------- translate
    @staticmethod
    def _extract_text(content: Any) -> str:
        """Join the text of a ``system`` value that may be a str or a list of
        ``{"type": "text", "text": ...}`` blocks."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(parts)

    def _build_prompt(self, messages: list, workdir: str) -> tuple[str, bool]:
        """Render the single user message into a prompt string, writing any
        base64 image blocks to PNG files in ``workdir`` and referencing them by
        absolute path. Returns (prompt, has_images). Block order is preserved so
        interleaved text/figure context reaches the model in the same order the
        SDK payload had it."""
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise RuntimeError(
                "ClaudeCLIClient expects exactly one user message "
                f"(got {len(messages)} / roles "
                f"{[m.get('role') for m in messages]})"
            )
        content = messages[0].get("content")
        if isinstance(content, str):
            return content, False

        parts: list[str] = []
        img_index = 0
        has_images = False
        for block in content:
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image":
                src = block.get("source", {})
                data = src.get("data", "")
                png_path = os.path.join(workdir, f"fig_{img_index:02d}.png")
                with open(png_path, "wb") as fh:
                    fh.write(base64.standard_b64decode(data))
                parts.append(
                    f"[Figure {img_index}: read the image file at {png_path} "
                    "with the Read tool before answering.]"
                )
                img_index += 1
                has_images = True
            else:
                logger.warning("cli_client: dropping unknown block type %r", btype)
        return "\n\n".join(parts), has_images

    def _argv(self, model: str, system: str, has_images: bool) -> list[str]:
        argv = [
            self._binary,
            "-p",
            "--output-format", "json",
            "--model", model,
            "--system-prompt", system,
            # Untrusted paper text: no built-in tools at all, except Read on
            # vision calls to load the PNGs we just wrote.
            "--tools", "Read" if has_images else "",
            "--disallowedTools", _DISALLOWED_TOOLS,
            # No MCP servers, no user/project settings/hooks/skills.
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
        ]
        return argv

    def _child_env(self) -> dict:
        env = dict(os.environ)
        for key in list(env):
            if key in _SCRUB_ENV or any(
                key.startswith(p) for p in _SCRUB_ENV_PREFIXES
            ):
                env.pop(key, None)
        return env

    # ------------------------------------------------------------------- invoke
    def _run(self, kwargs: dict):
        model = kwargs.get("model")
        if not model:
            raise RuntimeError("cli_client: 'model' is required")
        system = self._extract_text(kwargs.get("system"))
        messages = kwargs.get("messages") or []
        for dropped in ("temperature", "max_tokens"):
            if dropped in kwargs:
                logger.debug("cli_client: dropping unsupported kwarg %s", dropped)

        workdir = tempfile.mkdtemp(
            dir=self._scratch_root, prefix="aal-cli-"
        )
        try:
            prompt, has_images = self._build_prompt(messages, workdir)
            argv = self._argv(model, system, has_images)
            timeout = self._vision_timeout_s if has_images else self._timeout_s
            try:
                proc = subprocess.run(
                    argv,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=workdir,
                    env=self._child_env(),
                    timeout=timeout,
                )
            except FileNotFoundError as e:
                raise _fatal(
                    f"claude CLI not found ({self._binary!r}); install it or set "
                    "AAL_BACKEND=api"
                ) from e
            except subprocess.TimeoutExpired as e:
                # Treat as overload: retryable so _call_with_retry backs off.
                raise _anthropic_status_error(
                    529, f"claude CLI timed out after {timeout}s"
                ) from e

            return self._parse(proc, model)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _parse(self, proc: subprocess.CompletedProcess, model: str):
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        data = self._loads(stdout)

        if data is None:
            # No parseable JSON: classify from whatever text we have.
            self._raise_for_text(f"{stderr}\n{stdout}", proc.returncode)

        if data.get("is_error") or data.get("subtype") not in (None, "success"):
            self._raise_for_text(
                f"{stderr}\n{data.get('result', '')}\n{data.get('subtype', '')}",
                proc.returncode,
            )

        result_text = data.get("result")
        if not isinstance(result_text, str) or not result_text.strip():
            raise RuntimeError(
                f"claude CLI returned empty result (returncode {proc.returncode})"
            )

        self._assert_model(data, model)
        return self._build_message(data, model, result_text)

    @staticmethod
    def _loads(stdout: str) -> dict | None:
        """Parse the CLI's JSON result, tolerating stray lines around it."""
        stdout = stdout.strip()
        if not stdout:
            return None
        try:
            obj = json.loads(stdout)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(stdout[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _raise_for_text(self, text: str, returncode: int):
        low = text.lower()
        if any(m in low for m in _FATAL_MARKERS):
            raise _fatal(f"claude CLI availability error: {text.strip()[:400]}")
        if any(m in low for m in _USAGE_LIMIT_MARKERS):
            raise _fatal(
                f"claude subscription usage limit reached: {text.strip()[:400]}"
            )
        if any(m in low for m in _RATELIMIT_MARKERS):
            raise _anthropic_status_error(429, text.strip()[:400])
        if any(m in low for m in _OVERLOAD_MARKERS):
            raise _anthropic_status_error(529, text.strip()[:400])
        raise RuntimeError(
            f"claude CLI call failed (returncode {returncode}): {text.strip()[:400]}"
        )

    def _assert_model(self, data: dict, model: str):
        """Silent-model-substitution guard (precedent: the silent-Opus incident,
        evaluation/benchmark/extract_driver.py). ``modelUsage`` keys are the
        models the CLI actually billed; if none matches the requested model a
        different model ran — a methodology violation, so abort the run."""
        if model in self._verified_models:
            return
        usage = data.get("modelUsage")
        if isinstance(usage, dict) and usage:
            keys = list(usage.keys())
            if not any(k.startswith(model) or model.startswith(k) for k in keys):
                raise _fatal(
                    f"claude CLI ran model(s) {keys} but {model!r} was requested "
                    "(silent substitution)"
                )
        self._verified_models.add(model)

    @staticmethod
    def _build_message(data: dict, model: str, result_text: str):
        """Assemble a real ``anthropic.types.Message`` (``.construct`` skips
        validation of the fields the CLI does not report). Call sites read only
        ``.content[0].text``; ``.usage`` is populated for the batch-stats path."""
        import anthropic
        from anthropic.types import Message, TextBlock, Usage

        u = data.get("usage") or {}
        usage = Usage.construct(
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_input_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                u.get("cache_creation_input_tokens", 0) or 0
            ),
        )
        del anthropic  # imported only to assert the dep is present
        return Message.construct(
            id=str(data.get("session_id", "cli")),
            type="message",
            role="assistant",
            model=model,
            stop_reason="end_turn",
            stop_sequence=None,
            content=[TextBlock.construct(type="text", text=result_text)],
            usage=usage,
        )
