"""Agent extraction stage (pipeline/agent_extractor.py): lockdown, task-card
provenance, result normalisation, error classes and the dispatcher, all
without a real model. The "CLI" is a shell script that behaves like
``claude -p --output-format stream-json``."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import agent_extractor as ag
from pipeline import extractor as ex
from pipeline.extractor import FatalAPIError


# ---------------------------------------------------------------- fixtures

def _fake_cli(tmp_path: Path, name: str, body: str) -> str:
    """A fake `claude` binary. ``body`` is bash run with cwd = the session
    work dir and stdin = the user prompt."""
    p = tmp_path / name
    p.write_text("#!/bin/bash\ncat > /dev/null\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _result_event(model="claude-fable-5", is_error=False, subtype="success",
                  text="DONE", cost=1.23, turns=7):
    return json.dumps({
        "type": "result", "subtype": subtype, "is_error": is_error,
        "result": text, "num_turns": turns, "duration_ms": 1000,
        "total_cost_usd": cost,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "modelUsage": {model: {"inputTokens": 10}},
    })


GOOD_RESULT = {
    "paper_title": "A haloscope search",
    "coupling_type": "AxionPhoton",
    "is_new_limit": True,
    "is_projection": False,
    "data_points": [[2.0e-6, 3.0e-14], [1.0e-6, 2.5e-14], {"mass_eV": 3.0e-6, "coupling": 4.0e-14}],
    "data_source": "figure_vector",
    "dm_density_assumed": 0.45,
    "confidence_level": 0.9,
    "coupling_convention": "g_agamma [GeV^-1]",
    "suggested_experiment_name": "TestScope_2026",
    "extraction_confidence": 0.85,
    "notes": "traced Fig. 3 from the e-print PDF paths",
    "polarization_assumption": None,
    "alternatives": ["Fig. 3 dashed: expected sensitivity", {"where": "Fig. 4", "what": "g_ae panel"}],
    "headline_check": {"quoted": 3e-14, "mass_eV": 2e-6, "traced": 3.1e-14},
}


@pytest.fixture
def paper():
    stub = SimpleNamespace(title="A haloscope search", summary="abstract text",
                           categories=["hep-ex"], entry_id="http://arxiv.org/abs/2609.00001v2")
    stub.get_short_id = lambda: "2609.00001v2"
    return stub


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AAL_AGENT_LOGS", str(tmp_path / "logs"))
    monkeypatch.setenv("AAL_BACKEND", "claude-cli")   # subscription path: no key needed
    monkeypatch.delenv("AAL_EXTRACTOR", raising=False)
    monkeypatch.delenv("AAL_EXTRACTOR_FALLBACK", raising=False)
    monkeypatch.setattr(ag.time, "sleep", lambda s: None)
    yield


# ------------------------------------------------------------- provenance

def test_task_card_is_the_benchmark_card():
    text, sha = ag.load_task_card()
    assert sha == ag.BENCHMARK_TASK_CARD_SHA256
    assert text.startswith("# AxionLimitBench task card")
    assert "## Production additions" in text
    assert '"alternatives"' in text and "overlay.png" in text
    assert "web search" in text and "never copy a compilation" in text


def test_drifted_card_refuses(monkeypatch, tmp_path):
    bad = tmp_path / "card.md"
    bad.write_text("# not the card\n")
    monkeypatch.setattr(ag, "TASK_CARD_PATH", bad)
    with pytest.raises(RuntimeError, match="drifted"):
        ag.load_task_card()


# --------------------------------------------------------------- lockdown

def test_argv_lockdown():
    argv = ag.build_argv("claude-fable-5", "CARD", binary="claude",
                         max_budget_usd=5.0, effort=None)
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == ag.ALLOWED_TOOLS
    assert "Task" not in ag.ALLOWED_TOOLS and "WebSearch" in ag.ALLOWED_TOOLS
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv and "--no-session-persistence" in argv
    assert "--dangerously-skip-permissions" in argv
    assert argv[argv.index("--max-budget-usd") + 1] == "5.0"
    assert argv[argv.index("--append-system-prompt") + 1] == "CARD"
    settings = json.loads(argv[argv.index("--settings") + 1])
    deny = settings["permissions"]["deny"]
    assert "Bash(git push:*)" in deny and "Bash(gh:*)" in deny
    # production does not fence the web (user decision 2026-09-19): no
    # WebSearch / WebFetch / host denials at all
    assert not any(d.startswith(("WebSearch", "WebFetch")) for d in deny)


def test_child_env_cli_scrubs_key(monkeypatch):
    monkeypatch.setenv("AAL_BACKEND", "claude-cli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("CLAUDE_CODE_FOO", "1")
    monkeypatch.setenv("CLAUDECODE", "1")
    env = ag.child_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_FOO" not in env and "CLAUDECODE" not in env
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_child_env_api_passes_key(monkeypatch):
    monkeypatch.setenv("AAL_BACKEND", "api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    assert ag.child_env()["ANTHROPIC_API_KEY"] == "sk-secret"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(FatalAPIError):
        ag.child_env()


# ---------------------------------------------------------- normalisation

def test_normalise_result_coerces_points_and_extras():
    out, problems = ag.normalise_result(dict(GOOD_RESULT, data_points=GOOD_RESULT["data_points"] + [[-1, 2], "junk"]))
    assert out["data_points"] == [[1.0e-6, 2.5e-14], [2.0e-6, 3.0e-14], [3.0e-6, 4.0e-14]]
    assert len(problems) == 2
    assert out["alternatives"] == ["Fig. 3 dashed: expected sensitivity",
                                   '{"where": "Fig. 4", "what": "g_ae panel"}']
    assert out["headline_check"]["quoted"] == 3e-14
    assert out["is_new_limit"] is True and out["is_projection"] is False


def test_normalise_result_defaults():
    out, problems = ag.normalise_result({})
    assert out["data_points"] == [] and out["is_new_limit"] is False
    assert out["data_source"] == "none"
    assert out["confidence_level"] == 0.95 and out["extraction_confidence"] == 0.5
    assert out["alternatives"] == [] and out["headline_check"] is None
    assert out["notes"] == "" and out["suggested_experiment_name"] == "Unknown"
    assert problems == []


def test_classify_error_text():
    assert ag.classify_error_text("You've hit your limit · resets 4pm") == "usage_limit"
    assert ag.classify_error_text("hit your monthly spend limit · raise it at claude.ai/settings/usage") == "usage_limit"
    assert ag.classify_error_text("Invalid API key") == "fatal"
    assert ag.classify_error_text("Not logged in · /login") == "fatal"
    assert ag.classify_error_text("529 overloaded") == "rate"
    assert ag.classify_error_text("could not parse Fig 3") == "paper"


def test_scan_compilation_use_flags_actions_not_paper_text():
    events = [
        {"type": "user", "message": {"content": [{"type": "tool_result",
            "content": "see cajohare.github.io/AxionLimits for a compilation"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls AxionLimitsv2.pdf"}}]}},
    ]
    assert ag.scan_compilation_use(events) == []
    events.append({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "WebFetch",
         "input": {"url": "https://cajohare.github.io/AxionLimits/"}}]}})
    assert "cajohare" in ag.scan_compilation_use(events)


# ------------------------------------------------------------- the session

def test_session_end_to_end(tmp_path, paper, pdf, monkeypatch):
    body = (
        "test -f ./paper.pdf || exit 9\n"
        f"cat > ./result.json <<'JSON'\n{json.dumps(GOOD_RESULT)}\nJSON\n"
        "printf 'PNG' > ./overlay.png\n"
        "echo 'print(1)' > ./trace.py\n"
        f"echo '{_result_event()}'\n"
    )
    cli = _fake_cli(tmp_path, "claude-ok", body)
    monkeypatch.setenv("AAL_CLI_BINARY", cli)

    res = ex.run_extraction_agent(paper, pdf, client=object())
    assert isinstance(res, ex.ExtractionResult)
    assert res.arxiv_id == "2609.00001"
    assert res.coupling_type == "AxionPhoton"
    assert res.data_points == [(1.0e-6, 2.5e-14), (2.0e-6, 3.0e-14), (3.0e-6, 4.0e-14)]
    assert res.data_source == "figure_vector"
    assert res.dm_density_assumed == 0.45
    assert res.coupling_convention == "g_agamma [GeV^-1]"
    assert res.alternatives[0].startswith("Fig. 3 dashed")
    assert res.headline_check["traced"] == 3.1e-14
    assert "agent: src=figure_vector turns=7 cost=$1.23" in res.notes
    # audit trail
    logs = Path(os.environ["AAL_AGENT_LOGS"]) / "2609.00001"
    assert (logs / "events.jsonl").exists() and (logs / "session.json").exists()
    assert Path(res.artifacts["result.json"]).exists()
    assert Path(res.artifacts["overlay.png"]).read_bytes() == b"PNG"
    assert "trace.py" in res.artifacts
    assert not (logs / "work").exists()          # e-print/work dir not kept
    meta = json.loads((logs / "session.json").read_text())
    assert meta["task_card_sha256"] == ag.BENCHMARK_TASK_CARD_SHA256
    assert meta["transport"] == "cli" and meta["models_billed"] == ["claude-fable-5"]


def test_abstention_is_an_answer_not_a_fallback(tmp_path, paper, pdf, monkeypatch):
    abst = {"is_new_limit": False, "is_projection": True, "data_points": [],
            "coupling_type": "AxionPhoton", "notes": "projection only"}
    cli = _fake_cli(tmp_path, "claude-abstain",
                    f"cat > ./result.json <<'JSON'\n{json.dumps(abst)}\nJSON\necho '{_result_event()}'\n")
    monkeypatch.setenv("AAL_CLI_BINARY", cli)
    monkeypatch.setattr(ex, "run_staged_extraction",
                        lambda *a, **k: pytest.fail("fallback must not run on abstention"))
    res = ex.run_extraction_agent(paper, pdf, client=object())
    assert res.is_new_limit is False and res.is_projection is True and res.data_points == []


def test_no_result_json_falls_back_to_staged_pipeline(tmp_path, paper, pdf, monkeypatch):
    cli = _fake_cli(tmp_path, "claude-empty", f"echo '{_result_event(text='I gave up')}'\n")
    monkeypatch.setenv("AAL_CLI_BINARY", cli)
    called = {}

    def staged(p, path, client):
        called["yes"] = True
        return ex.ExtractionResult(
            arxiv_id="2609.00001", paper_title="t", arxiv_url="u", coupling_type="AxionPhoton",
            is_new_limit=True, is_projection=False, data_points=[(1e-6, 1e-13)],
            data_source="table", dm_density_assumed=None, polarization_assumption=None,
            confidence_level=0.9, suggested_experiment_name="X", extraction_confidence=0.7,
            notes="staged")
    monkeypatch.setattr(ex, "run_staged_extraction", staged)
    res = ex.run_extraction_agent(paper, pdf, client=object())
    assert called and "[AGENT FALLBACK]" in res.notes and "no result.json" in res.notes

    monkeypatch.setenv("AAL_EXTRACTOR_FALLBACK", "0")
    with pytest.raises(ag.AgentInfraError):
        ex.run_extraction_agent(paper, pdf, client=object())


@pytest.mark.parametrize("text", [
    "You've hit your limit · resets 4:10pm (America/Chicago)",
    "hit your monthly spend limit · raise it at claude.ai/settings/usage",
    "Not logged in. Please run /login",
    "Your credit balance is too low",
])
def test_availability_errors_are_fatal_not_fallback(tmp_path, paper, pdf, monkeypatch, text):
    cli = _fake_cli(tmp_path, "claude-limit",
                    f"echo '{_result_event(is_error=True, subtype='error', text=text)}'\n")
    monkeypatch.setenv("AAL_CLI_BINARY", cli)
    monkeypatch.setattr(ex, "run_staged_extraction",
                        lambda *a, **k: pytest.fail("availability error must not fall back"))
    with pytest.raises(FatalAPIError):
        ex.run_extraction_agent(paper, pdf, client=object())


def test_silent_model_substitution_is_fatal(tmp_path, paper, pdf, monkeypatch):
    cli = _fake_cli(tmp_path, "claude-haiku",
                    f"cat > ./result.json <<'JSON'\n{json.dumps(GOOD_RESULT)}\nJSON\n"
                    f"echo '{_result_event(model='claude-haiku-4-5-20251001')}'\n")
    monkeypatch.setenv("AAL_CLI_BINARY", cli)
    with pytest.raises(FatalAPIError, match="substitution"):
        ex.run_extraction_agent(paper, pdf, client=object())


def test_missing_cli_is_fatal(paper, pdf, monkeypatch):
    monkeypatch.setenv("AAL_CLI_BINARY", "/nonexistent/claude")
    with pytest.raises(FatalAPIError, match="not found"):
        ex.run_extraction_agent(paper, pdf, client=object())


def test_compilation_use_is_reported_not_capped(tmp_path, paper, pdf, monkeypatch):
    fetch = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "WebFetch",
         "input": {"url": "https://cajohare.github.io/AxionLimits/"}}]}})
    cli = _fake_cli(tmp_path, "claude-leak",
                    f"cat > ./result.json <<'JSON'\n{json.dumps(GOOD_RESULT)}\nJSON\n"
                    f"echo '{fetch}'\necho '{_result_event()}'\n")
    monkeypatch.setenv("AAL_CLI_BINARY", cli)
    res = ex.run_extraction_agent(paper, pdf, client=object())
    assert "[COMPILATION CONSULTED]" in res.notes
    assert res.extraction_confidence == GOOD_RESULT["extraction_confidence"]


def test_dispatch_pipeline_env_skips_agent(paper, pdf, monkeypatch):
    monkeypatch.setenv("AAL_EXTRACTOR", "pipeline")
    monkeypatch.setattr(ag, "run_agent_session",
                        lambda *a, **k: pytest.fail("agent must not run"))
    sentinel = object()
    monkeypatch.setattr(ex, "run_staged_extraction", lambda *a, **k: sentinel)
    assert ex.run_extraction_agent(paper, pdf, client=object()) is sentinel


def test_dispatch_rejects_unknown_extractor(monkeypatch):
    monkeypatch.setenv("AAL_EXTRACTOR", "magic")
    with pytest.raises(ValueError):
        ag.resolve_extractor()


# ------------------------------------------------------- shared guard tail

def test_inverse_fa_read_is_not_decade_snapped():
    """The task card's canonical AxionMass plane is 1/f_a [GeV^-1]; a correct
    read at ~6e-17 GeV^-1 (f_a = 1.6e16 GeV, 2105.13963) must pass the shared
    range guard untouched (it was snapped x1e12 by the old 1e-12 floor)."""
    pts = [(1e-20, 6.4e-17), (1e-15, 6.4e-17), (1e-11, 2.0e-17)]
    out, note = ex._validate_extracted_range(list(pts), "AxionMass")
    assert out == pts
    assert "Auto-correct" not in note


def test_collider_alp_trace_passes_reviewer_hard_range():
    """2607.07800 (Belle II ALP -> gamma gamma, 0.17-9.8 GeV, g_agamma up to
    7e-3 GeV^-1): a correct collider trace must not be rejected as a unit
    error by the reviewer's min/max window (it was, on the first unattended
    production run)."""
    from pipeline.reviewer import validate_data_ranges
    pts = [(1.75e8, 8.0e-5), (3.05e8, 9.0e-5), (9.78e9, 7.4e-3)]
    validate_data_ranges(pts, "AxionPhoton")          # must not raise
    validate_data_ranges([(5e7, 1e-3), (5e10, 1e-2)], "DarkPhoton")
    with pytest.raises(ValueError):
        validate_data_ranges([(1e-6, 10.0)], "AxionPhoton")   # a real blunder still fails
