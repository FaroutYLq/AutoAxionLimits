# Model backends (`AAL_BACKEND`)

Choose how model calls are authenticated and billed. Both backends use the same
pipeline prompts, stages and checks; this does not guarantee identical model
outputs. For local runs, start with [setup and usage](../docs/local-pipelines.md).

| `AAL_BACKEND` | Client | Billing | Auth |
|---|---|---|---|
| unset / `api` (default) | `anthropic.Anthropic` | `ANTHROPIC_API_KEY` (pay-per-token) | API key |
| `claude-cli` | [`ClaudeCLIClient`](cli_client.py) → `claude -p` | Claude Code **subscription** | keychain OAuth (`claude` login) |

Direct module commands and GitHub Actions default to `api`. The local skill
runner honors `AAL_BACKEND` when set and otherwise defaults to `claude-cli`.

## `claude-cli` backend

Shells out to headless `claude -p` for each model call, so the daily / weekly /
backfill pipelines can run on a Pro/Max subscription with no API key. Enable it:

```bash
# Requires the local setup, Claude login and GitHub authentication.
bash scripts/run_pipeline.sh run daily --backend claude-cli -- --dry-run --max-papers 1
```

The in-session skills wrap this with an isolated clone and state-branch
handling — prefer them for real runs: `daily-arxiv-digest`,
`weekly-preprint-check`, `backfill-extraction`. They work in Claude Code and
Codex through the same [local runner](../docs/local-pipelines.md).

### What the CLI client does

- Translates `messages.create(**kwargs)` into a `claude -p` subprocess: `model`
  → `--model`, `system` → `--system-prompt`, the single user message → stdin,
  base64 image blocks → PNG files the subprocess reads with the Read tool.
- Drops `temperature` / `max_tokens` / `cache_control` (no CLI equivalents).
  Model names are forwarded unchanged; transport parity still needs evaluation.
- **Locks the subprocess down** (the paper text is untrusted): all built-in
  tools disabled except Read on vision calls, no MCP servers, no project
  settings, a throwaway cwd so repo `CLAUDE.md` is never loaded, and the child
  env scrubbed of `ANTHROPIC_API_KEY` (would silently flip billing to the API)
  and inherited `CLAUDE_CODE_*` session vars.
- Maps failures onto the existing `_call_with_retry` / `FatalAPIError` (#648)
  contract: auth / billing / subscription-usage-limit → `FatalAPIError` (abort
  the run without marking papers); rate-limit / overload / timeout → genuine
  `anthropic` errors that get backoff. Silent model substitution (the CLI
  running a different model than requested) is fatal.

### Tuning env vars

| Var | Default | Meaning |
|---|---|---|
| `AAL_CLI_BINARY` | `claude` | CLI executable name/path |
| `AAL_CLI_TIMEOUT` | `1200` | text-call subprocess timeout (s) |
| `AAL_CLI_VISION_TIMEOUT` | `2700` | vision-call subprocess timeout (s; up to 8 PNGs) |

### Constraints & caveats

- **Execution access.** The subprocess needs network access and reads the
  subscription OAuth token from the macOS keychain; in Claude Code that means the
  launcher must run outside the shell sandbox, which the user grants. Otherwise
  the preflight ping fails with "Not logged in" (exit 2) and looks like a
  usage-window outage.
- **Subscription rate windows.** A large backfill can exhaust the usage window — throttle
  with `--max-papers` and `--resume` (window exhaustion aborts cleanly, exit 2,
  candidate re-queued).
- **Message Batches (`AAL_BATCH`) is API-only** and cannot combine with
  `claude-cli`; the eval driver raises if both are set.
- **Vision fidelity** may differ slightly from the API path (the CLI Read tool
  can recompress large PNGs). Validate with a side-by-side eval-subset parity
  run before trusting CLI-backend extractions for real limits.


# Extraction stage (`AAL_EXTRACTOR`)

Orthogonal to the transport: which *system* produces the curve.

| `AAL_EXTRACTOR` | Stage | Notes |
|---|---|---|
| unset / `agent` (default) | [`agent_extractor.py`](agent_extractor.py): one headless `claude -p` session per paper with the AxionLimitBench task card | needs the `claude` CLI on PATH (subscription login under `AAL_BACKEND=claude-cli`, or `ANTHROPIC_API_KEY` otherwise), `ghostscript` for EPS figures, and the python image stack in `requirements_pipeline.txt` |
| `pipeline` | the staged text -> vision -> select pipeline (`run_staged_extraction`) | the pre-2026-09 production path; bit-identical to before |

Both stages end in the same deterministic tail (`finalize_extraction`) and feed
the same reviewer and PR gate. Knobs: `AAL_AGENT_MAX_BUDGET_USD` (5),
`AAL_AGENT_TIMEOUT` (1800 s), `AAL_AGENT_EFFORT` (CLI default),
`AAL_AGENT_LOGS` (`pipeline/logs/agent`), `AAL_EXTRACTOR_FALLBACK` (1: an
agent crash/timeout with no `result.json` falls back to the staged pipeline;
an abstention never does). The task card core is pinned to the benchmark's
`docs/TASK.md` by sha256, so a production run is the benchmarked system; see
`CLAUDE.md` for the evidence and the contracts.


## Independent publication review

Every real daily, backfill, and changed-data preprint PR now requires a separate
agent to approve its source evidence and freshly rendered full/highlighted
images. Dry runs do not invoke this publication stage. Withdrawal/removal flags
retain their existing human-review workflow.

The reviewer starts a fresh Claude CLI session outside the checkout, with only
Read/Glob/Grep tools, no project settings or MCP servers. It receives the paper
PDF, extraction metadata, proposed data and plotting source, notebook source,
and corresponding full/highlighted PNGs. Successful tool reads of the PDF, data,
metadata and both images are required; naming an image in the final answer is
not evidence of inspection. The review card is
[publication_review_card.md](publication_review_card.md).

Scientific judgments belong to the agent: result identity, novelty, statistical
confidence, physical conventions, representative numerical agreement, correct
highlighting and visual clarity. Code enforces evidence availability, structured
and consistent decisions, and matching artifact hashes; it does not supply a
numerical threshold or substitute for those judgments.

An agent can approve, request a display repair, or require human review. Display
repairs use a separate read-only agent that proposes axes, supported styling
arguments or one label. The driver applies these bounded choices, renders again,
and starts a fresh independent reviewer. There are at most two repair cycles.
Data, confidence claims and physical transformations cannot be repaired through
this path. Unsupported repairs, scientific uncertainty, exhausted repair cycles,
missing evidence, invalid decisions and CLI failures block publication. There is
no fallback that skips agent approval.

Approved reports (including findings, inspected files, prompt/model provenance,
evidence hashes and repair history) are committed under `pipeline/reviews/` and
linked in the PR. The driver verifies the report and approved artifact hashes
before committing/publishing. Local evidence packets, full CLI transcripts and
blocked findings remain under `pipeline/logs/publication_review/`. Review failure
leaves the paper/version eligible for retry and aborts the run with a failure;
inspect `failure.json` and `review.json` there before resuming a blocked paper.
Scientific decisions still require human adjudication and PR merge review.

The CLI is required even with `AAL_EXTRACTOR=pipeline`; it uses the same billing
environment as agent extraction (`AAL_BACKEND=claude-cli` removes the API key;
otherwise the configured API credentials are inherited). Configure:

| Variable | Default | Meaning |
|---|---|---|
| `PUBLICATION_REVIEW_MODEL` | reviewer model (`claude-opus-4-8`) | independent reviewer and display-repair model |
| `AAL_PUBLICATION_REVIEW_TIMEOUT` | `600` | timeout in seconds per session |
| `AAL_PUBLICATION_REVIEW_BUDGET` | `5` | CLI dollar budget per session; at most five sessions per candidate |

The local runner passes timeout/budget through. Set a model override explicitly
with its `--env PUBLICATION_REVIEW_MODEL=...` option; inherited model overrides
are scrubbed to keep scheduled production runs on the configured default.
