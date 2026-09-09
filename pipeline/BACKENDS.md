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

- **Runs must not be sandboxed.** The subprocess needs network access and reads
  the subscription OAuth token from the macOS keychain; in Claude Code run the
  launcher with the shell sandbox disabled, or the preflight ping fails with
  "Not logged in" (exit 2) and looks like a usage-window outage.
- **Subscription rate windows.** A large backfill can exhaust the usage window — throttle
  with `--max-papers` and `--resume` (window exhaustion aborts cleanly, exit 2,
  candidate re-queued).
- **Message Batches (`AAL_BATCH`) is API-only** and cannot combine with
  `claude-cli`; the eval driver raises if both are set.
- **Vision fidelity** may differ slightly from the API path (the CLI Read tool
  can recompress large PNGs). Validate with a side-by-side eval-subset parity
  run before trusting CLI-backend extractions for real limits.
