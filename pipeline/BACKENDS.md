# Model backends (`AAL_BACKEND`)

Every pipeline entrypoint gets its model client from
[`pipeline/client_factory.py`](client_factory.py). One env var selects the
transport. **The transport is the only thing that changes** — prompts, stage
flow, guards, thresholds, and model selection are identical across backends
(the extraction distribution is unchanged; this is a transport-only swap, like
[`batch_client.py`](batch_client.py)).

| `AAL_BACKEND` | Client | Billing | Auth |
|---|---|---|---|
| unset / `api` (default) | `anthropic.Anthropic` | `ANTHROPIC_API_KEY` (pay-per-token) | API key |
| `claude-cli` | [`ClaudeCLIClient`](cli_client.py) → `claude -p` | Claude Code **subscription** | keychain OAuth (`claude` login) |

GitHub Actions leave `AAL_BACKEND` unset, so CI is bit-identical to before.

## `claude-cli` backend

Shells out to headless `claude -p` for each model call, so the daily / weekly /
backfill pipelines can run on a Pro/Max subscription with no API key. Enable it:

```bash
# prerequisites: `claude` on PATH and logged in to a subscription, plus `gh` auth
AAL_BACKEND=claude-cli python -m pipeline.orchestrator --dry-run
```

The in-session skills wrap this with worktree isolation and state-branch
handling — prefer them for real runs: `/daily-arxiv-digest`,
`/weekly-preprint-check`, `/backfill-extraction`.

### What the CLI client does

- Translates `messages.create(**kwargs)` into a `claude -p` subprocess: `model`
  → `--model`, `system` → `--system-prompt`, the single user message → stdin,
  base64 image blocks → PNG files the subprocess reads with the Read tool.
- Drops `temperature` / `max_tokens` / `cache_control` (no CLI equivalents).
  `temperature` is already a no-op for the production model `claude-opus-4-8`
  (the API path strips it too via `extractor._create`), so there is no
  methodology delta for the default model.
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

- **Runs must not be sandboxed.** The subprocess needs network access and must
  read the subscription OAuth token from the macOS keychain; a sandboxed shell
  fails with "Not logged in".
- **Subscription rate windows.** Daily/weekly volume (1–3 papers) is well within
  Max limits. A large backfill can exhaust the 5-hour/weekly window — throttle
  with `--max-papers` and `--resume` (window exhaustion aborts cleanly, exit 2,
  candidate re-queued).
- **Message Batches (`AAL_BATCH`) is API-only** and cannot combine with
  `claude-cli`; the eval driver raises if both are set.
- **Vision fidelity** may differ slightly from the API path (the CLI Read tool
  can recompress large PNGs). Validate with a side-by-side eval-subset parity
  run before trusting CLI-backend extractions for real limits.
