"""Independent, tool-using science/visual review before a science PR is published.

The reviewer sees a frozen evidence packet and has read-only tools. A separate
session can propose bounded display adjustments; each adjustment is rendered
and reviewed in a fresh session. Science changes are always escalated. There is
no fail-open or deterministic substitute for an unavailable agent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from .agent_extractor import child_env, classify_error_text, parse_events
from .extractor import FatalAPIError, _parse_json_response
from .plot_regen import PlotGenerationError, generate_review_plots
from .reviewer import CLAUDE_MODEL

logger = logging.getLogger(__name__)
SCIENCE_CHECKS = ('result_identity', 'novelty', 'confidence_level', 'conventions', 'numerical_agreement')
PLOT_CHECKS = ('highlight_target', 'visibility')
CHECKS = SCIENCE_CHECKS + PLOT_CHECKS
CARD = Path(__file__).with_name('publication_review_card.md')


class PublicationReviewError(PlotGenerationError):
    """No approved review: retain evidence and leave the paper retryable."""


@dataclass
class ApprovedPublication:
    plot_files: list[str]
    highlight_files: list[str]
    report_path: str
    summary: str
    report_sha256: str

    def section(self, ref: str) -> str:
        url = f'https://github.com/FaroutYLq/AutoAxionLimits/blob/{ref}/{self.report_path}'
        return ('## Independent agent review\n\n' + self.summary + '\n\n'
                f'[Evidence, findings, and review history]({url})\n\n')


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value):
    return asdict(value) if is_dataclass(value) else dict(vars(value))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def make_packet(directory: Path, *, paper_pdf, extraction, proposal: dict,
                repo_root: Path, plots: list[str]) -> tuple[dict, list[str]]:
    """Stage actual evidence, without credentials, local settings, or prior verdicts."""
    directory.mkdir(parents=True)
    (directory / 'paper.pdf').write_bytes(paper_pdf if isinstance(paper_pdf, bytes) else Path(paper_pdf).read_bytes())
    data = _json(extraction)
    data.pop('artifacts', None)  # do not expose another agent's session as authority
    _write_json(directory / 'extraction.json', data)
    _write_json(directory / 'proposal.json', proposal)
    for rel in [proposal['data_file_path'], proposal['plotfuncs_file'], proposal['notebook_path'], *plots]:
        source = (repo_root / rel).resolve()
        if not source.is_relative_to(repo_root.resolve()):
            raise PublicationReviewError(f'Evidence path escapes checkout: {rel}')
        dest = directory / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith('.ipynb'):
            nb = json.loads(source.read_text())
            # Source/metadata are useful evidence; embedded old output is not.
            for cell in nb.get('cells', []):
                cell['outputs'] = [] if cell.get('cell_type') == 'code' else cell.get('outputs', [])
            _write_json(dest, nb)
        else:
            shutil.copyfile(source, dest)
    original = subprocess.run(['git', 'show', f"HEAD:{proposal['data_file_path']}"],
                              cwd=repo_root, capture_output=True)
    if original.returncode == 0:
        (directory / 'previous-data.txt').write_bytes(original.stdout)
    required = ['paper.pdf', 'extraction.json', 'proposal.json', proposal['data_file_path'], *plots]
    manifest = {p.relative_to(directory).as_posix(): _hash(p)
                for p in sorted(directory.rglob('*')) if p.is_file()}
    _write_json(directory / 'manifest.json', manifest)
    return manifest, required


def successful_reads(events: list[dict], directory: Path) -> set[str]:
    """Require successful Read tool results, not self-reported image inspection."""
    pending = {}
    read = set()
    for event in events:
        for block in (event.get('message') or {}).get('content') or []:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use' and block.get('name') == 'Read':
                name = (block.get('input') or {}).get('file_path', '')
                path = Path(name)
                path = path if path.is_absolute() else directory / path
                try:
                    pending[block['id']] = path.resolve().relative_to(directory.resolve()).as_posix()
                except (ValueError, KeyError):
                    pass
            if block.get('type') == 'tool_result' and not block.get('is_error', False):
                if block.get('tool_use_id') in pending:
                    read.add(pending[block['tool_use_id']])
    return read


def run_review_agent(directory: Path, *, role: str, required: list[str], feedback=None) -> tuple[dict, dict]:
    """An independent session, with native PDF/image Read and no write/shell/web tools."""
    model = os.environ.get('PUBLICATION_REVIEW_MODEL', CLAUDE_MODEL)
    binary = os.environ.get('AAL_CLI_BINARY', 'claude')
    card = CARD.read_text()
    argv = [binary, '-p', '--model', model, '--output-format', 'stream-json', '--verbose',
            '--no-session-persistence', '--setting-sources', '', '--strict-mcp-config',
            '--tools', 'Read,Glob,Grep', '--disallowedTools',
            'Bash,Write,Edit,NotebookEdit,WebFetch,WebSearch,Task,Agent',
            '--dangerously-skip-permissions', '--append-system-prompt', card]
    budget = float(os.environ.get('AAL_PUBLICATION_REVIEW_BUDGET', '5'))
    if budget > 0:
        argv += ['--max-budget-usd', str(budget)]
    prompt = (f'Your role is {role}. Evidence is in {directory}.\n'
              'Read these actual files with the Read tool, including the PDF and every PNG:\n'
              + '\n'.join(str(directory.resolve() / p) for p in required) + '\nRead the plotting source as needed.\n'
              'Return your final JSON in your answer; you cannot modify files.\n')
    if feedback is not None:
        prompt += '\nIndependent reviewer findings to address:\n' + json.dumps(feedback)
    started = time.monotonic()
    try:
        # Avoid inheriting the checkout's CLAUDE.md or another agent's local context.
        with tempfile.TemporaryDirectory(prefix='aal-publication-session-') as session_cwd:
            proc = subprocess.run(argv, input=prompt, cwd=session_cwd, env=child_env(),
                                  text=True, capture_output=True,
                                  timeout=int(os.environ.get('AAL_PUBLICATION_REVIEW_TIMEOUT', '600')))
    except subprocess.TimeoutExpired as exc:
        (directory.parent / f'{role}-events.jsonl').write_bytes(
            exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or '').encode())
        raise PublicationReviewError(f'{role} timed out; no approval issued') from exc
    except FileNotFoundError as exc:
        raise FatalAPIError(f'Publication reviewer CLI unavailable: {binary}') from exc
    events, result = parse_events(proc.stdout)
    _write_json(directory.parent / f'{role}-session.json', {
        'model': model, 'elapsed_s': time.monotonic() - started, 'returncode': proc.returncode,
        'prompt_sha256': hashlib.sha256(card.encode()).hexdigest(), 'result': result})
    (directory.parent / f'{role}-events.jsonl').write_text(proc.stdout)
    if proc.returncode != 0 or not result or result.get('is_error') or result.get('subtype') != 'success':
        detail = str((result or {}).get('result') or proc.stderr or 'missing completion')[-1500:]
        if classify_error_text(detail) in ('fatal', 'usage_limit'):
            raise FatalAPIError(f'Publication {role} unavailable: {detail}')
        raise PublicationReviewError(f'Publication {role} failed: {detail}')
    billed = list((result or {}).get('modelUsage') or {})
    if billed and not any(m == model or m.startswith(model) for m in billed):
        raise FatalAPIError(f'Publication {role}: requested {model}, billed {billed}')
    read = successful_reads(events, directory)
    missing = set(required) - read
    if missing:
        raise PublicationReviewError(f'{role} did not inspect required evidence: {sorted(missing)}')
    try:
        answer = _parse_json_response(result.get('result', ''))
    except Exception as exc:
        raise PublicationReviewError(f'{role} returned invalid JSON') from exc
    if not isinstance(answer, dict):
        raise PublicationReviewError(f'{role} returned no structured decision')
    return answer, {'model': model, 'models_billed': billed, 'inspected_files': sorted(read),
                    'prompt_sha256': hashlib.sha256(card.encode()).hexdigest()}


def validate_verdict(answer: dict) -> dict:
    """Validate completeness/consistency only; scientific judgments belong to the agent."""
    decision = answer.get('decision')
    if decision not in ('approve', 'revise_plot', 'needs_human_review'):
        raise PublicationReviewError('Reviewer decision is missing or invalid')
    if not isinstance(answer.get('summary'), str) or not answer['summary'].strip():
        raise PublicationReviewError('Reviewer must explain its decision')
    checks = answer.get('checks')
    if not isinstance(checks, dict) or not set(CHECKS).issubset(checks):
        raise PublicationReviewError('Reviewer omitted required scientific/visual judgments')
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get('status') not in ('pass', 'fail', 'uncertain') or not isinstance(check.get('evidence'), str) or not check['evidence'].strip():
            raise PublicationReviewError(f'Incomplete reviewer evidence: {name}')
    findings = answer.get('findings')
    if not isinstance(findings, list):
        raise PublicationReviewError('Reviewer findings must be a list')
    for finding in findings:
        if not isinstance(finding, dict) or finding.get('category') not in ('science', 'plot') or finding.get('severity') not in ('blocking', 'advisory') or any(
            not isinstance(finding.get(k), str) or not finding[k].strip()
            for k in ('description', 'evidence', 'requested_change')):
            raise PublicationReviewError('A finding lacks actionable evidence or requested changes')
    blocking = [f for f in findings if f['severity'] == 'blocking']
    if decision == 'approve' and (blocking or any(c['status'] != 'pass' for c in checks.values())):
        raise PublicationReviewError('Approval conflicts with unresolved review findings')
    if decision == 'revise_plot' and (not blocking or any(f['category'] == 'science' for f in blocking)
                                    or any(checks[k]['status'] != 'pass' for k in SCIENCE_CHECKS)):
        raise PublicationReviewError('Scientific uncertainty cannot be handled as an automatic plot repair')
    if decision != 'approve' and not blocking:
        raise PublicationReviewError('A blocked review must identify a concrete blocking finding')
    return answer


def review_for_publication(*, extraction, paper_pdf, proposal: dict, repo_root: Path,
                           plot_files: list[str], highlight_files: list[str]) -> ApprovedPublication:
    """Approve exact artifacts, or retain an actionable report and abort publication."""
    sid = re.sub(r'[^A-Za-z0-9_.-]', '_', str(extraction.arxiv_id))
    run_id = uuid.uuid4().hex
    audit = repo_root / 'pipeline/logs/publication_review' / sid / run_id
    history = []
    original_notebook = (repo_root / proposal['notebook_path']).read_bytes()
    original_data_hash = _hash(repo_root / proposal['data_file_path'])
    original_code_hash = _hash(repo_root / proposal['plotfuncs_file'])
    approved = False
    try:
        for attempt in range(3):  # at most two bounded display corrections
            packet = audit / str(attempt) / 'evidence'
            highlighted_pngs = [p for p in highlight_files if p.endswith('.png')]
            full_pngs = [p.replace('_highlighted.', '.') for p in highlighted_pngs]
            if not highlighted_pngs or any(p not in plot_files for p in full_pngs):
                raise PublicationReviewError('Independent review needs full and highlighted PNG pairs')
            pngs = full_pngs + highlighted_pngs
            candidate_files = [proposal['data_file_path'], proposal['plotfuncs_file'], proposal['notebook_path'],
                               *plot_files, *highlight_files]
            candidate_hashes = {p: _hash(repo_root / p) for p in dict.fromkeys(candidate_files)}
            manifest, required = make_packet(packet, paper_pdf=paper_pdf, extraction=extraction,
                                            proposal=proposal, repo_root=repo_root, plots=pngs)
            answer, session = run_review_agent(packet, role='reviewer', required=required)
            if any(_hash(packet / p) != sha for p, sha in manifest.items()):
                raise PublicationReviewError('Evidence changed during independent review')
            verdict = validate_verdict(answer)
            history.append({'attempt': attempt, 'verdict': verdict, 'session': session, 'evidence_sha256': manifest})
            _write_json(audit / 'review.json', {'history': history})
            if verdict['decision'] == 'approve':
                # Bind the report to the checkout artifacts that will be committed.
                for p, digest in candidate_hashes.items():
                    if _hash(repo_root / p) != digest:
                        raise PublicationReviewError(f'Artifact changed after review: {p}')
                relative = f'pipeline/reviews/{sid}-{run_id}.json'
                _write_json(repo_root / relative, {
                    'schema_version': 1, 'arxiv_id': extraction.arxiv_id,
                    'decision': 'approve', 'summary': verdict['summary'],
                    'artifact_sha256': candidate_hashes, 'history': history})
                approved = True
                return ApprovedPublication(plot_files, highlight_files, relative, verdict['summary'],
                                           _hash(repo_root / relative))
            if verdict['decision'] == 'needs_human_review' or attempt == 2:
                changes = '; '.join(f['requested_change'] for f in verdict['findings'] if f['severity'] == 'blocking')
                raise PublicationReviewError(f'Independent review blocked publication: {changes}. Evidence: {audit / "review.json"}')
            adjustments, repair_session = run_review_agent(packet, role='plot_repair', required=required, feedback=verdict)
            if any(_hash(packet / p) != sha for p, sha in manifest.items()):
                raise PublicationReviewError('Evidence changed during plot repair')
            from .plot_adjustments import apply_plot_adjustments
            proposal = dict(proposal)
            proposal['notebook_call'] = apply_plot_adjustments(repo_root, proposal, adjustments)
            history[-1]['repair'] = {'adjustments': adjustments, 'session': repair_session}
            _write_json(audit / 'review.json', {'history': history})
            if _hash(repo_root / proposal['data_file_path']) != original_data_hash or _hash(repo_root / proposal['plotfuncs_file']) != original_code_hash:
                raise PublicationReviewError('Display repair changed scientific data or plotting transformations')
            plot_files, highlight_files = generate_review_plots(
                proposal['notebook_path'], proposal['notebook_call'], repo_root, proposal['data_file_path'])
    except (FatalAPIError, PublicationReviewError) as exc:
        _write_json(audit / 'failure.json', {'error': str(exc), 'history': history})
        raise
    except Exception as exc:
        _write_json(audit / 'failure.json', {'error': str(exc), 'history': history})
        raise PublicationReviewError(f'Publication review failed; no approval. Evidence: {audit}: {exc}') from exc
    finally:
        if not approved:
            (repo_root / proposal['notebook_path']).write_bytes(original_notebook)


def verify_approval(approval: ApprovedPublication, repo_root: Path) -> None:
    """Recheck the approved file hashes immediately before committing/publishing."""
    if not isinstance(approval, ApprovedPublication):
        raise PublicationReviewError('An independent agent approval is required')
    if _hash(repo_root / approval.report_path) != approval.report_sha256:
        raise PublicationReviewError('Review report changed after approval')
    report = json.loads((repo_root / approval.report_path).read_text())
    if report.get('decision') != 'approve' or not report.get('artifact_sha256') or not report.get('history'):
        raise PublicationReviewError('Review report does not approve publication')
    for path, digest in report.get('artifact_sha256', {}).items():
        if _hash(repo_root / path) != digest:
            raise PublicationReviewError(f'Review is stale after artifact modification: {path}')
