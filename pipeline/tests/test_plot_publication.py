"""PR #778: inherited imports and stale plots must never pass publication."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import plot_regen as plots
from pipeline.pr_creator import plot_review_section


def notebook(root, source="DarkPhoton.New(ax)\nMySaveFig(fig,'DarkPhoton')"):
    path = root / 'DarkPhoton.ipynb'
    path.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'id': 'plot', 'cell_type': 'code', 'execution_count': None, 'metadata': {},
         'outputs': [], 'source': source.splitlines(True)}]}))
    return path


def output(root, name, content='new'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_kernel_imports_run_checkout_even_with_launcher_pythonpath(tmp_path, monkeypatch):
    pytest.importorskip('nbconvert')
    pytest.importorskip('ipykernel')
    launcher = tmp_path / 'launcher'
    launcher.mkdir()
    (launcher / 'PlotFuncs.py').write_text("raise AssertionError('wrong checkout imported')")
    (tmp_path / 'PlotFuncs.py').write_text(
        "from pathlib import Path\ndef save():\n"
        "    Path('plots').mkdir(exist_ok=True)\n"
        "    Path('plots/DarkPhoton.pdf').write_text('correct checkout')\n")
    notebook(tmp_path, "from PlotFuncs import save\nsave()\ndef MySaveFig(*a): pass\nMySaveFig(None,'DarkPhoton')")
    monkeypatch.setenv('PYTHONPATH', str(launcher))
    monkeypatch.setenv('JUPYTER_RUNTIME_DIR', str(tmp_path / 'runtime'))
    ok, err = plots.execute_notebook('DarkPhoton.ipynb', tmp_path)
    assert ok, err
    assert (tmp_path / 'plots/DarkPhoton.pdf').read_text() == 'correct checkout'


@pytest.mark.parametrize('kind', ['stale', 'missing', 'empty'])
def test_success_exit_without_fresh_full_output_is_failure(tmp_path, monkeypatch, kind):
    notebook(tmp_path)
    if kind == 'stale':
        output(tmp_path, 'plots/DarkPhoton.pdf', 'old')
    def run(*args, **kwargs):
        assert kwargs['env']['PYTHONPATH'].split(os.pathsep)[0] == str(tmp_path)
        if kind == 'empty':
            output(tmp_path, 'plots/DarkPhoton.pdf', '')
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(plots.subprocess, 'run', run)
    ok, err = plots.execute_notebook('DarkPhoton.ipynb', tmp_path)
    assert not ok and 'fresh' in err


@pytest.mark.parametrize('rc, png', [(1, True), (0, False)])
def test_failed_or_partial_highlight_has_no_publishable_outputs(tmp_path, monkeypatch, rc, png):
    notebook(tmp_path)
    def run(*args, **kwargs):
        assert kwargs['env']['PYTHONPATH'].split(os.pathsep)[0] == str(tmp_path)
        output(tmp_path, 'plots/DarkPhoton_highlighted.pdf')
        if png:
            output(tmp_path, 'plots/plots_png/DarkPhoton_highlighted.png')
        return SimpleNamespace(returncode=rc, stderr='failed after save' if rc else '')
    monkeypatch.setattr(plots.subprocess, 'run', run)
    ok, err, files = plots.execute_notebook_highlighted('DarkPhoton.ipynb', 'DarkPhoton.New(ax)', tmp_path)
    assert not ok and files == []
    assert not (tmp_path / 'DarkPhoton_highlighted_tmp.ipynb').exists()


def test_full_png_cannot_be_inherited_while_highlight_is_fresh(tmp_path, monkeypatch):
    notebook(tmp_path)
    output(tmp_path, 'plots/plots_png/DarkPhoton.png', 'old')
    monkeypatch.setattr(plots, 'execute_notebook', lambda *a: (True, ''))
    monkeypatch.setattr(plots, 'execute_notebook_highlighted', lambda *a, **kw: (
        True, '', ['plots/DarkPhoton_highlighted.pdf', 'plots/plots_png/DarkPhoton_highlighted.png']))
    with pytest.raises(plots.PlotGenerationError, match='Missing fresh'):
        plots.generate_review_plots('DarkPhoton.ipynb', 'DarkPhoton.New(ax)', tmp_path, 'data.txt')


def test_section_requires_matching_full_plot_not_first_notebook_plot():
    with pytest.raises(plots.PlotGenerationError):
        plot_review_section('branch', 'New', [], [])
    with pytest.raises(plots.PlotGenerationError):
        plot_review_section('branch', 'New', ['plots/plots_png/Wrong.png'], ['plots/plots_png/Right_highlighted.png'])
    body = plot_review_section('sha', 'New', ['plots/plots_png/Right.png'], ['plots/plots_png/Right_highlighted.png'])
    assert '/sha/plots/plots_png/Right.png' in body


def test_preprint_target_uses_data_reference_not_filename(tmp_path):
    (tmp_path / 'PlotFuncs.py').write_text('class DarkPhoton:\n    def Alias(ax):\n        loadtxt("limit_data/DarkPhoton/FileName.txt")\n')
    notebook(tmp_path, "# DarkPhoton.Alias(ax)\nDarkPhoton.Alias(ax, text_on=False)\nMySaveFig(fig,'DarkPhoton')")
    assert plots.find_plot_target('limit_data/DarkPhoton/FileName.txt', tmp_path) == (
        'DarkPhoton.ipynb', 'DarkPhoton.Alias(ax, text_on=False)')
    with pytest.raises(plots.PlotGenerationError):
        plots.find_plot_target('limit_data/DarkPhoton/Unknown.txt', tmp_path)


def test_grouped_preprint_target_highlights_only_its_limit(tmp_path):
    import copy
    (tmp_path / 'PlotFuncs.py').write_text('class DarkPhoton:\n    def Group(ax):\n        DarkPhoton.Alias(ax)\n    def Alias(ax):\n        loadtxt("limit_data/DarkPhoton/FileName.txt")\n')
    notebook(tmp_path, "DarkPhoton.Group(ax)\nMySaveFig(fig,'DarkPhoton')")
    nb = json.loads((tmp_path / 'DarkPhoton.ipynb').read_text())
    original = copy.deepcopy(nb)
    name, call = plots.find_plot_target('limit_data/DarkPhoton/FileName.txt', tmp_path)
    assert call == 'DarkPhoton.Alias(ax)'
    exposed = plots._expose_grouped_target(nb, call, tmp_path)
    transformed, names = plots._build_highlight_notebook(exposed, call)
    assert names == ['DarkPhoton_highlighted']
    code = ''.join(transformed['cells'][1]['source'])
    assert '_HIGHLIGHT_ACTIVE = True\nDarkPhoton.Alias(' in code
    assert '_HIGHLIGHT_ACTIVE = True\nDarkPhoton.Group(' not in code
    assert nb == original


def test_target_prefers_dedicated_method_over_legacy_group(tmp_path):
    (tmp_path / 'PlotFuncs.py').write_text('class DarkPhoton:\n    def Group(ax):\n        loadtxt("limit_data/DarkPhoton/Test.txt")\n    def Test(ax):\n        loadtxt("limit_data/DarkPhoton/Test.txt")\n')
    notebook(tmp_path, "DarkPhoton.Group(ax)\nDarkPhoton.Test(ax)\nMySaveFig(fig,'DarkPhoton')")
    assert plots.find_plot_target('limit_data/DarkPhoton/Test.txt', tmp_path)[1] == 'DarkPhoton.Test(ax)'


def test_preprint_publish_helper_still_pushes_before_creating_pr(monkeypatch):
    from pipeline import pr_creator
    calls = []
    monkeypatch.setattr(pr_creator, '_run_git', lambda args, cwd: calls.append(('git', args)))
    monkeypatch.setattr(pr_creator, '_run_gh', lambda args, cwd: calls.append(('gh', args)) or 'url')
    assert pr_creator.create_pull_request_preprint('branch', 'title', 'body') == 'url'
    assert calls[0] == ('git', ['push', '-u', 'origin', 'branch'])
    assert calls[1][0] == 'gh'


def test_preprint_render_failure_restores_data_before_any_publication(tmp_path, monkeypatch):
    from pipeline import preprint_checker as pc, pr_creator
    path = tmp_path / 'limit_data/DarkPhoton/Test.txt'
    path.parent.mkdir(parents=True)
    path.write_text('original data')
    monkeypatch.setattr(pc, 'format_data_file', lambda *a: 'new data')
    monkeypatch.setattr(pc, 'find_plot_target', lambda *a: ('DarkPhoton.ipynb', 'DarkPhoton.Test(ax)'))
    def fail(*a):
        raise plots.PlotGenerationError('missing highlight')
    monkeypatch.setattr(pc, 'generate_review_plots', fail)
    monkeypatch.setattr(pr_creator, '_run_git', lambda *a: pytest.fail('must not publish'))
    with pytest.raises(plots.PlotGenerationError):
        pc._create_update_pr(tmp_path, 'limit_data/DarkPhoton/Test.txt', '2601.00001', 1, 2,
                             SimpleNamespace(coupling_type='DarkPhoton'), [(1, 2)], [], [], 'changed', object())
    assert path.read_text() == 'original data'


def test_weekly_plot_failure_keeps_old_version_retryable(tmp_path, monkeypatch):
    from pipeline import preprint_checker as pc
    filename = 'limit_data/DarkPhoton/Test.txt'
    state = {'files': {filename: {'known_version': 1, 'published': False}}}
    saved = []
    monkeypatch.setattr(pc, 'load_version_state', lambda: state)
    monkeypatch.setattr(pc, 'save_version_state', lambda value: saved.append(json.loads(json.dumps(value))))
    monkeypatch.setattr(pc, 'make_client', lambda **kw: object())
    monkeypatch.setattr(pc, 'scan_data_files_for_arxiv_ids', lambda root: {filename: '2601.00001'})
    monkeypatch.setattr(pc, 'batch_check_published_semantic_scholar', lambda ids: {})
    monkeypatch.setattr(pc, 'batch_get_latest_versions', lambda ids: {'2601.00001': (2, False, object())})
    monkeypatch.setattr(pc, 'is_withdrawn', lambda aid: False)
    monkeypatch.setattr(pc, 'is_published', lambda paper: False)
    monkeypatch.setattr(pc, 'download_pdf', lambda *a: Path('unused'))
    monkeypatch.setattr(pc, 'run_extraction_agent', lambda *a: SimpleNamespace(
        data_points=[(1, 2)], is_projection=False, extraction_confidence=0.9))
    monkeypatch.setattr(pc, 'apply_corrections', lambda result: ([(1, 2)], [], []))
    monkeypatch.setattr(pc, 'data_has_changed', lambda *a: True)
    monkeypatch.setattr(pc, 'summarise_changes', lambda *a: 'changed')
    def fail(**kw):
        raise plots.PlotGenerationError('renderer failed')
    monkeypatch.setattr(pc, '_create_update_pr', fail)
    with pytest.raises(SystemExit) as error:
        pc.run_weekly_check(tmp_path)
    assert error.value.code == 3
    assert saved[-1]['files'][filename]['known_version'] == 1


def test_runner_replaces_launcher_pythonpath_with_checkout(tmp_path, monkeypatch):
    from pipeline.local_runner import child_environment
    env, _, _ = child_environment({'PYTHONPATH': '/old/launcher', 'PATH': '/bin'}, tmp_path, 'claude-cli')
    assert env['PYTHONPATH'] == str(tmp_path.resolve())
