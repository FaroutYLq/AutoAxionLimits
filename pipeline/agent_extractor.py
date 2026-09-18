"""Generic tool-using agent as the extraction stage (``AAL_EXTRACTOR=agent``).

AxionLimitBench (arXiv, 2026-09) measured a generic Claude Code agent against
this repository's staged pipeline on the same 292 papers and the same model:
the agent reached a 56% success rate at 10% tolerance against the pipeline's
19%, with 0.0% catastrophic errors against 3.8%. The mechanism is not a better
prompt: the agent fetches the paper's arXiv e-print, digitises vector figures
from their path geometry, and iterates with its own code until its trace
matches the paper's quoted numbers. This module makes that agent the
production extraction stage.

Design
------
* One headless ``claude -p`` session per paper, in a throwaway working
  directory containing only ``paper.pdf``. Seven built-in tools (Read, Write,
  Edit, Bash, Glob, Grep, WebFetch); WebSearch and the limit compilations are
  denied at the permission layer, git push / gh at the Bash layer.
* The system prompt is the AxionLimitBench task card VERBATIM
  (``agent_task_card.md``, sha256 pinned to the benchmark's ``docs/TASK.md``)
  plus a short production section (``agent_task_card_production.md``: the
  paper's own HEPData / data release is allowed, and the agent adds
  ``alternatives``, ``headline_check``, ``polarization_assumption`` and an
  ``overlay.png`` for the human reviewer). Production therefore runs the
  benchmarked system, and the card's provenance is recorded in every result.
* The agent's ``result.json`` becomes a stage result and goes through the
  SAME deterministic tail as the staged pipeline
  (:func:`pipeline.extractor.finalize_extraction`: gauge-group relabel,
  VALID_RANGES validation, R5 hard floor, convention-review screens with
  the escalation queue). Nothing downstream (reviewer, density-at-plot-time,
  PR gate) changes.
* Transport follows ``AAL_BACKEND``: ``claude-cli`` bills the subscription
  (keychain OAuth; the API key is scrubbed from the child), anything else
  passes ``ANTHROPIC_API_KEY`` through to the CLI (Actions). Availability
  errors (auth, billing, subscription window, silent model substitution) are
  :class:`FatalAPIError` per #648; a crash or timeout with no ``result.json``
  is :class:`AgentInfraError` (the dispatcher may fall back to the staged
  pipeline); an abstention is an answer.
* Every session leaves an audit trail under ``pipeline/logs/agent/<id>/``
  (stream-json transcript, result.json, overlay.png, the agent's own scripts)
  and the paths are returned in ``ExtractionResult.artifacts``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .extractor import (
    CLAUDE_MODEL,
    ExtractionResult,
    FatalAPIError,
    finalize_extraction,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TASK_CARD_PATH = HERE / "agent_task_card.md"
PRODUCTION_CARD_PATH = HERE / "agent_task_card_production.md"
# sha256 of AxionLimitBench docs/TASK.md (v0.1.1, the card every leaderboard
# run used). load_task_card() refuses to run on a drifted copy so that
# "production runs the benchmarked system" stays a checkable statement.
BENCHMARK_TASK_CARD_SHA256 = (
    "503a9033a0f9b64515fead0f9179cebeee02a6ccf2168878954aafa6e1a3ceeb"
)
DRIVER_VERSION = "1.0.0"

# Parent-session variables that would silently change billing or confuse the
# child (the pipelines are routinely launched from inside a Claude Code
# session via the skills).
SCRUB_ENV = (
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDECODE",
)
SCRUB_PREFIXES = ("CLAUDE_CODE_",)

USAGE_LIMIT_MARKERS = (
    "usage limit reached", "usage limit", "5-hour limit", "weekly limit",
    "reset at", "session limit", "hit your limit", "· resets", "limit resets",
    "spend limit", "monthly limit", "hit your", "claude.ai/settings/usage",
    "raise it at",
)
FATAL_MARKERS = (
    "credit balance", "billing", "invalid api key", "invalid x-api-key",
    "not logged in", "/login", "authentication",
)
RATE_MARKERS = (
    "rate limit", "too many requests", "429", "overloaded", "529",
    "service unavailable",
)
# Limit compilations: consulting one is reported to the reviewer (the task
# card forbids it; production does not disqualify, the benchmark did).
COMPILATION_PATTERNS = (
    "cajohare", "axionlimits", "cajohare.github.io", "darkcast",
)
NETWORK_INDICATORS = ("http://", "https://", "curl ", "wget ", "git clone",
                      "urlopen", "requests.get", "fetch(")

# Permission deny-list handed to the CLI. Production allows HEPData, Zenodo
# and GitHub (the paper's own data release) but never the compilation, never
# web search, and never a push or a gh call from inside the session.
SETTINGS = {
    "permissions": {
        "deny": [
            "WebSearch",
            "WebFetch(domain:cajohare.github.io)",
            "Bash(git push:*)",
            "Bash(gh:*)",
        ]
    }
}
ALLOWED_TOOLS = "Read,Write,Edit,Bash,Glob,Grep,WebFetch"

# result.json fields carried into the stage result (task card schema plus the
# production extras).
RESULT_FIELDS = (
    "paper_title", "coupling_type", "is_new_limit", "is_projection",
    "data_points", "data_source", "dm_density_assumed", "confidence_level",
    "extraction_confidence", "suggested_experiment_name",
    "coupling_convention", "notes",
    "polarization_assumption", "alternatives", "headline_check",
)
ARTIFACT_GLOBS = ("result.json", "overlay*.png", "*.py")


class AgentInfraError(RuntimeError):
    """The session produced no answer for a reason that is not the paper's
    (crash, timeout, unparseable output). The dispatcher may fall back."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def resolve_extractor() -> str:
    """``agent`` (default) or ``pipeline``, from ``AAL_EXTRACTOR``."""
    val = os.environ.get("AAL_EXTRACTOR", "agent").strip().lower() or "agent"
    if val not in ("agent", "pipeline"):
        raise ValueError(f"AAL_EXTRACTOR must be 'agent' or 'pipeline', got {val!r}")
    return val


def fallback_enabled() -> bool:
    return os.environ.get("AAL_EXTRACTOR_FALLBACK", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def artifacts_root() -> Path:
    return Path(os.environ.get("AAL_AGENT_LOGS") or (HERE / "logs" / "agent"))


def load_task_card() -> tuple[str, str]:
    """Return (full system-prompt text, sha256 of the benchmark core).
    Raises if the core drifted from the benchmark's card."""
    core = TASK_CARD_PATH.read_text()
    sha = hashlib.sha256(core.encode()).hexdigest()
    if sha != BENCHMARK_TASK_CARD_SHA256:
        raise RuntimeError(
            f"{TASK_CARD_PATH.name} drifted from the AxionLimitBench task card "
            f"(sha256 {sha[:12]} != {BENCHMARK_TASK_CARD_SHA256[:12]}); production "
            "must run the benchmarked card, update BENCHMARK_TASK_CARD_SHA256 "
            "deliberately if the benchmark card itself changed")
    extra = PRODUCTION_CARD_PATH.read_text() if PRODUCTION_CARD_PATH.exists() else ""
    return core + extra, sha


def transport_mode() -> str:
    """``cli`` (subscription) when AAL_BACKEND selects the CLI transport,
    else ``api`` (ANTHROPIC_API_KEY passed through to the CLI)."""
    backend = os.environ.get("AAL_BACKEND", "api").strip().lower()
    return "cli" if backend in ("claude-cli", "cli") else "api"


def child_env() -> dict:
    env = dict(os.environ)
    for k in list(env):
        if k in SCRUB_ENV or any(k.startswith(p) for p in SCRUB_PREFIXES):
            env.pop(k, None)
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if transport_mode() == "cli":
        # Subscription billing: a stray exported key would silently flip the
        # session to API billing.
        env.pop("ANTHROPIC_API_KEY", None)
    elif not env.get("ANTHROPIC_API_KEY"):
        raise FatalAPIError(
            "agent extractor: ANTHROPIC_API_KEY is not set and AAL_BACKEND is "
            "not claude-cli; nothing can bill this session")
    return env


def build_argv(model: str, task_card: str, *, binary: str,
               max_budget_usd: float | None, effort: str | None) -> list[str]:
    argv = [
        binary, "-p",
        "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--no-session-persistence",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--settings", json.dumps(SETTINGS),
        "--tools", ALLOWED_TOOLS,
        "--dangerously-skip-permissions",
        "--append-system-prompt", task_card,
    ]
    if max_budget_usd is not None:
        argv += ["--max-budget-usd", str(max_budget_usd)]
    if effort:
        argv += ["--effort", effort]
    return argv


def user_prompt(arxiv_id: str, work: Path) -> str:
    return (
        f"AutoAxionLimits extraction for arXiv:{arxiv_id}.\n\n"
        f"The paper's PDF is at ./paper.pdf in the current working directory "
        f"({work}). Follow the task card in your system prompt, including the "
        f"production additions. When you are done, write the result JSON to "
        f"exactly this absolute path: {work / 'result.json'} (not a relative "
        f"path: your shell's working directory persists between commands), "
        f"write the overlay to {work / 'overlay.png'} when the paper has a "
        f"figure of the limit, and reply with the single line DONE."
    )


# ---------------------------------------------------------------------------
# Session output handling
# ---------------------------------------------------------------------------

def parse_events(stdout: str) -> tuple[list[dict], dict | None]:
    events: list[dict] = []
    result = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(ev)
        if ev.get("type") == "result":
            result = ev
    return events, result


def classify_error_text(text: str) -> str:
    low = (text or "").lower()
    if any(m in low for m in USAGE_LIMIT_MARKERS):
        return "usage_limit"
    if any(m in low for m in FATAL_MARKERS):
        return "fatal"
    if any(m in low for m in RATE_MARKERS):
        return "rate"
    return "paper"


def scan_compilation_use(events: list[dict]) -> list[str]:
    """Compilation patterns in what the AGENT ISSUED (tool inputs that reach
    the network, and its own prose). Tool results are not scanned: papers
    cite compilations, and reading the paper is the task."""
    hits: set[str] = set()
    hosts = tuple(p for p in COMPILATION_PATTERNS if "." in p)
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for b in (ev.get("message") or {}).get("content") or []:
            if b.get("type") == "tool_use":
                blob = json.dumps(b.get("input")).lower()
                networked = (b.get("name") == "WebFetch"
                             or any(k in blob for k in NETWORK_INDICATORS))
                hits.update(p for p in (COMPILATION_PATTERNS if networked else hosts)
                            if p in blob)
            elif b.get("type") == "text":
                blob = str(b.get("text")).lower()
                hits.update(p for p in hosts if p in blob)
    return sorted(hits)


def normalise_result(raw: dict) -> tuple[dict, list[str]]:
    """Coerce the agent's result.json into a stage-result dict (the shape
    :func:`finalize_extraction` consumes); never raises."""
    problems: list[str] = []
    out: dict = {k: raw.get(k) for k in RESULT_FIELDS}
    pts: list[list[float]] = []
    for p in raw.get("data_points") or []:
        try:
            if isinstance(p, dict):
                m, g = float(p.get("mass_eV", p.get("mass"))), float(p.get("coupling"))
            else:
                m, g = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            problems.append(f"bad point {p!r}")
            continue
        if not (m > 0 and g > 0) or m != m or g != g:
            problems.append(f"non-positive/nan point {p!r}")
            continue
        pts.append([m, g])
    pts.sort(key=lambda q: q[0])
    out["data_points"] = pts
    out["is_new_limit"] = (bool(raw.get("is_new_limit"))
                           if raw.get("is_new_limit") is not None else bool(pts))
    out["is_projection"] = bool(raw.get("is_projection", False))
    if out["data_source"] is None:
        out["data_source"] = "none" if not pts else "unknown"
    for k in ("confidence_level", "extraction_confidence", "dm_density_assumed"):
        v = out.get(k)
        if v is not None:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                problems.append(f"non-numeric {k}={v!r}")
                out[k] = None
    if out["confidence_level"] is None:
        out["confidence_level"] = 0.95
    if out["extraction_confidence"] is None:
        out["extraction_confidence"] = 0.5
    for k in ("paper_title", "suggested_experiment_name", "notes",
              "coupling_convention", "polarization_assumption"):
        if out.get(k) is not None and not isinstance(out[k], str):
            out[k] = json.dumps(out[k])
    alts = out.get("alternatives")
    if alts is None:
        out["alternatives"] = []
    elif isinstance(alts, str):
        out["alternatives"] = [alts]
    elif isinstance(alts, list):
        out["alternatives"] = [a if isinstance(a, str) else json.dumps(a) for a in alts]
    else:
        problems.append(f"alternatives not a list: {alts!r}")
        out["alternatives"] = []
    if out.get("headline_check") is not None and not isinstance(out["headline_check"], dict):
        problems.append(f"headline_check not an object: {out['headline_check']!r}")
        out["headline_check"] = None
    if out.get("notes") is None:
        out["notes"] = ""
    if out.get("suggested_experiment_name") is None:
        out["suggested_experiment_name"] = "Unknown"
    return out, problems


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

@dataclass
class AgentSession:
    raw: Optional[dict]                 # parsed result.json (None = none written)
    meta: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _collect_artifacts(work: Path, dest: Path) -> dict:
    """Copy the session's small, reviewable outputs (never the e-print)."""
    dest.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for pattern in ARTIFACT_GLOBS:
        for src in sorted(work.rglob(pattern)):
            if src.stat().st_size > 20 * 1024 * 1024:
                continue
            rel = src.relative_to(work)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            key = rel.as_posix()
            out[key] = str(target)
    if (dest / "result.json").exists():
        out["result.json"] = str(dest / "result.json")
    if (dest / "overlay.png").exists():
        out["overlay.png"] = str(dest / "overlay.png")
    return out


def run_agent_session(arxiv_id: str, pdf_path: Path, *, model: str | None = None,
                      timeout_s: int | None = None,
                      max_budget_usd: float | None = None,
                      effort: str | None = None,
                      binary: str | None = None,
                      artifacts_dir: Path | None = None) -> AgentSession:
    """Run one session and return its parsed answer plus audit metadata.

    Raises :class:`FatalAPIError` on availability errors (auth / billing /
    subscription window / silent model substitution / missing CLI) and
    :class:`AgentInfraError` when the session ended without a result.json for
    a non-availability reason. Rate limits and overloads are retried with
    backoff (three attempts)."""
    model = model or CLAUDE_MODEL
    timeout_s = timeout_s or _env_int("AAL_AGENT_TIMEOUT", 1800)
    if max_budget_usd is None:
        max_budget_usd = _env_float("AAL_AGENT_MAX_BUDGET_USD", 5.0)
        if max_budget_usd <= 0:
            max_budget_usd = None
    effort = effort or os.environ.get("AAL_AGENT_EFFORT") or None
    binary = binary or os.environ.get("AAL_CLI_BINARY", "claude")
    sid = _safe_id(arxiv_id)
    artifacts_dir = artifacts_dir or (artifacts_root() / sid)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    task_card, card_sha = load_task_card()
    argv = build_argv(model, task_card, binary=binary,
                      max_budget_usd=max_budget_usd, effort=effort)
    work = artifacts_dir / "work"
    prompt = user_prompt(arxiv_id, work)
    mode = transport_mode()
    env = child_env()

    last_err = ""
    for attempt in range(3):
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        shutil.copyfile(pdf_path, work / "paper.pdf")
        t0 = time.time()
        timed_out = False
        try:
            proc = subprocess.run(argv, input=prompt, text=True, capture_output=True,
                                  cwd=work, env=env, timeout=timeout_s)
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except FileNotFoundError as e:
            raise FatalAPIError(
                f"agent extractor: claude CLI not found ({binary!r}); install it "
                "(npm install -g @anthropic-ai/claude-code) or set AAL_EXTRACTOR=pipeline"
            ) from e
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
            stderr = (e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr) or ""
            rc = -9
        elapsed = time.time() - t0
        events, result_ev = parse_events(stdout)
        events_path = artifacts_dir / "events.jsonl"
        events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        err_text = ""
        if result_ev is None:
            err_text = f"{stderr}\n{stdout[-2000:]}"
        elif result_ev.get("is_error") or result_ev.get("subtype") not in (None, "success"):
            err_text = (f"{stderr}\n{result_ev.get('result', '')}\n"
                        f"{result_ev.get('subtype', '')}")

        if err_text and not timed_out:
            kind = classify_error_text(err_text)
            if kind in ("usage_limit", "fatal"):
                # #648: an availability error is a property of the run.
                raise FatalAPIError(f"agent extractor ({mode}): {err_text.strip()[:400]}")
            if kind == "rate":
                last_err = err_text
                delay = 60 * (attempt + 1)
                logger.warning("agent session %s: rate/overload, backing off %ds",
                               arxiv_id, delay)
                time.sleep(delay)
                continue
            # budget cap / max-turn terminations still leave result.json if the
            # agent got that far; fall through and use whatever exists.

        result_path = work / "result.json"
        problems: list[str] = []
        if not result_path.exists():
            found = sorted(work.rglob("result.json"), key=lambda q: len(q.parts))
            if found:
                result_path = found[0]
                problems.append(f"result.json found at {result_path.relative_to(work)}")
        raw = None
        if result_path.exists():
            try:
                raw = json.loads(result_path.read_text())
                if not isinstance(raw, dict):
                    problems.append("result.json is not an object")
                    raw = None
            except json.JSONDecodeError as e:
                problems.append(f"result.json unparseable: {e}")
        if raw is not None and result_path != work / "result.json":
            shutil.copyfile(result_path, work / "result.json")

        usage = (result_ev or {}).get("usage") or {}
        model_usage = (result_ev or {}).get("modelUsage") or {}
        models_billed = sorted(model_usage.keys())
        if models_billed and not any(m == model or m.startswith(model) for m in models_billed):
            raise FatalAPIError(
                f"agent extractor: silent model substitution, requested {model}, "
                f"billed {models_billed}")
        compilation_hits = scan_compilation_use(events)
        meta = {
            "system": "agent_claude_code",
            "driver_version": DRIVER_VERSION,
            "task_card_sha256": card_sha,
            "model_requested": model,
            "models_billed": models_billed,
            "effort": effort or "default",
            "transport": mode,
            "timed_out": timed_out,
            "wrote_result_json": raw is not None,
            "cli_is_error": bool((result_ev or {}).get("is_error")),
            "cli_subtype": (result_ev or {}).get("subtype"),
            "cli_returncode": rc,
            "num_turns": (result_ev or {}).get("num_turns"),
            "duration_ms": (result_ev or {}).get("duration_ms"),
            "total_cost_usd": (result_ev or {}).get("total_cost_usd"),
            "usage": {k: usage.get(k) for k in (
                "input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens")},
            "compilation_consulted": compilation_hits,
            "schema_problems": problems,
            "elapsed_s": elapsed,
            "attempt": attempt,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        artifacts = _collect_artifacts(work, artifacts_dir)
        artifacts["events.jsonl"] = str(events_path)
        (artifacts_dir / "session.json").write_text(json.dumps(meta, indent=1))
        artifacts["session.json"] = str(artifacts_dir / "session.json")
        shutil.rmtree(work, ignore_errors=True)

        if raw is None:
            why = ("timed out" if timed_out else
                   (err_text.strip()[:300] if err_text else "no result.json written"))
            raise AgentInfraError(f"agent session for {arxiv_id} produced no result.json: {why}")
        return AgentSession(raw=raw, meta=meta, artifacts=artifacts, problems=problems)

    raise AgentInfraError(
        f"agent session for {arxiv_id}: rate-limited on every attempt: {last_err[:200]}")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_agent_extraction(paper, pdf_path: Path, client) -> ExtractionResult:
    """The agent as extraction stage: session -> stage result -> shared tail."""
    arxiv_id = re.sub(r"v\d+$", "", paper.get_short_id())
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    session = run_agent_session(arxiv_id, pdf_path)
    stage, problems = normalise_result(session.raw or {})
    problems = session.problems + problems
    if problems:
        stage["notes"] = (stage.get("notes") or "") + " | schema: " + "; ".join(problems)
    if session.meta.get("compilation_consulted"):
        stage["notes"] = ((stage.get("notes") or "")
                          + " | [COMPILATION CONSULTED] the agent reached "
                          + ", ".join(session.meta["compilation_consulted"])
                          + "; check the curve is the paper's own")
        stage["extraction_confidence"] = min(
            float(stage.get("extraction_confidence") or 0.0), 0.5)
    if session.meta.get("timed_out"):
        stage["notes"] = (stage.get("notes") or "") + " | agent session timed out; result.json as left"
    stage["notes"] = ((stage.get("notes") or "")
                      + f" | agent: src={stage.get('data_source')}"
                        f" turns={session.meta.get('num_turns')}"
                        f" cost=${(session.meta.get('total_cost_usd') or 0):.2f}"
                        f" card={session.meta.get('task_card_sha256', '')[:12]}")
    data_points = [(float(m), float(g)) for m, g in stage.get("data_points") or []]
    logger.info(
        "agent extraction %s: ct=%s n=%d src=%s conf=%.2f turns=%s $%.2f",
        arxiv_id, stage.get("coupling_type"), len(data_points), stage.get("data_source"),
        float(stage.get("extraction_confidence") or 0.0), session.meta.get("num_turns"),
        session.meta.get("total_cost_usd") or 0.0,
    )
    result = finalize_extraction(
        paper, arxiv_id, arxiv_url, stage, data_points,
        pre_ct=None, pre_conf=0.0, client=client,
    )
    result.alternatives = list(stage.get("alternatives") or [])
    result.headline_check = stage.get("headline_check")
    result.artifacts = dict(session.artifacts)
    return result
