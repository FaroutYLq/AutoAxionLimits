export const meta = {
  name: 'failure-digest',
  description: 'Per-paper failure diagnosis of the after_roadmap eval vs ground truth, then synthesize a digest',
  phases: [
    { title: 'Diagnose', detail: 'one agent per failing paper' },
    { title: 'Synthesize', detail: 'group root causes into a digest' },
  ],
}

// args = { failures: [{arxiv_id, status, coupling, median_resid, data_source}], after_dir, run_label }
log(`args type=${typeof args}; keys=${args && typeof args === 'object' ? Object.keys(args).join(',') : 'n/a'}; failures len=${args && args.failures ? args.failures.length : 'none'}`)
let _argsObj = args
if (typeof args === 'string') { try { _argsObj = JSON.parse(args) } catch (e) { log('args was a string but not JSON: ' + e) } }
const failures = (_argsObj && _argsObj.failures) || []
const afterDir = (_argsObj && _argsObj.after_dir) || 'evaluation/eval_runs/after_roadmap'
const runLabel = (_argsObj && _argsObj.run_label) || 'after_roadmap'

if (!failures.length) {
  log('No failures passed in args; nothing to diagnose.')
  return { diagnosed: 0 }
}

const DIAG_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['arxiv_id', 'symptom', 'root_cause', 'extractor_fault', 'fixable', 'fix_suggestion', 'confidence'],
  properties: {
    arxiv_id: { type: 'string' },
    symptom: { type: 'string', description: 'one sentence: what went wrong in the comparison' },
    root_cause: {
      type: 'string',
      enum: ['source_misrouting', 'convention_units', 'vision_trace_drift', 'text_truncation',
             'comparator_encoding', 'figure_extraction', 'gt_benchmark_issue', 'unextractable', 'other'],
    },
    extractor_fault: { type: 'boolean', description: 'true if a fixable pipeline fault, false if genuinely unextractable / GT issue' },
    fixable: { type: 'boolean' },
    fix_suggestion: { type: 'string', description: 'concrete next action (code location / approach), or why not fixable' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

log(`Diagnosing ${failures.length} failing papers from ${runLabel}`)

const diagnoses = await parallel(failures.map((f) => () =>
  agent(
    `You are diagnosing ONE failing paper from the AutoAxionLimits extraction benchmark.\n\n` +
    `Paper: arXiv ${f.arxiv_id}\n` +
    `Comparison status: ${f.status}  | coupling: ${f.coupling}  | data_source: ${f.data_source}  | median_resid(dex): ${f.median_resid}\n\n` +
    `Investigate from the repo (read-only):\n` +
    `1. The extraction snapshot: ${afterDir}/${f.arxiv_id}.json (coupling_type, data_source, coupling_convention, data_points x/y ranges).\n` +
    `2. The ground truth: the papers.json entry for ${f.arxiv_id} in evaluation/ground_truth/papers.json (coupling_type, coupling_convention, reference_repo_file) AND the actual GT curve — try evaluation/ground_truth/data/${f.arxiv_id}.txt first, else the reference_repo_file under limit_data/. Report GT x/y ranges.\n` +
    `3. Compare the two: do the mass (x) ranges overlap? do the coupling (y) scales agree, or is there a clean dex offset (convention) vs scattered disagreement (vision drift)?\n` +
    `4. If useful, read the paper abstract via the gpd-arxiv MCP tools (search/read_paper) to confirm the reported quantity/units.\n\n` +
    `Classify the ROOT CAUSE into exactly one of: source_misrouting (a sparse text point won over a figure curve / wrong source), convention_units (a fixed dex offset = units/convention gap), vision_trace_drift (figure traced but scattered/log-misread), text_truncation (result number missing from text), comparator_encoding (box/line or mass-independent scoring artifact), figure_extraction (the limit figure was never delivered to vision), gt_benchmark_issue (the GT file/label itself is wrong — wrong era/observable/convention/sentinel), unextractable (no curve exists in the paper), other.\n\n` +
    `Decide extractor_fault (fixable pipeline bug) vs not (GT issue / truly unextractable). Give a concrete fix_suggestion (code location or approach) and your confidence. Be decisive and specific; cite the actual numbers you read.`,
    { label: `diag:${f.arxiv_id}`, phase: 'Diagnose', schema: DIAG_SCHEMA, agentType: 'Explore' }
  ).then((d) => ({ ...d, status: f.status, coupling: f.coupling, median_resid: f.median_resid }))
    .catch(() => null)
))

const ok = diagnoses.filter(Boolean)
log(`Diagnosed ${ok.length}/${failures.length}; synthesizing`)

// Synthesize: one agent writes the digest markdown directly to disk.
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['written_path', 'by_root_cause', 'top_recommendations'],
  properties: {
    written_path: { type: 'string' },
    by_root_cause: { type: 'object', additionalProperties: { type: 'integer' } },
    top_recommendations: { type: 'array', items: { type: 'string' } },
  },
}

const summary = await agent(
  `You are synthesizing a per-paper failure digest for the AutoAxionLimits extraction benchmark run "${runLabel}".\n\n` +
  `Here are ${ok.length} per-paper diagnoses (JSON):\n\n${JSON.stringify(ok, null, 1)}\n\n` +
  `Write a markdown digest to evaluation/eval_runs/failure_analysis_${runLabel}.md with:\n` +
  `1. A headline: total failures, count genuine extractor-fault vs GT-issue vs unextractable.\n` +
  `2. A table grouping papers BY root_cause (count + the arxiv_ids), ranked by papers-recoverable.\n` +
  `3. The highest-yield next levers (ranked), each with the concrete fix and the papers it would recover.\n` +
  `4. A short appendix: one line per paper (arxiv_id | status | root_cause | extractor_fault | fix_suggestion).\n` +
  `Be quantitative and decisive. Use the Write tool to create the file. Return the path you wrote, the by_root_cause counts, and your top recommendations.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { diagnosed: ok.length, of: failures.length, summary }
