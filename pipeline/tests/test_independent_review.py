"""Publication requires real agent judgment tied to the actual source and images."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import publication_review as review
from pipeline.plot_adjustments import apply_plot_adjustments
from pipeline.plot_regen import _build_highlight_notebook


def verdict(decision='approve', science=False):
    out = {'decision': decision, 'summary': 'Compared paper figure 4 with the data and both images.',
           'checks': {name: {'status':'pass', 'evidence':'Paper p. 4, Fig. 4; plotted contour agrees at 1 eV.'}
                      for name in review.CHECKS}, 'findings': []}
    if decision != 'approve':
        out['checks']['confidence_level' if science else 'visibility']['status'] = 'fail'
        out['findings'] = [{'category':'science' if science else 'plot', 'severity':'blocking',
                           'description':'Unsupported confidence' if science else 'Label is clipped',
                           'evidence':'Paper does not state 95% CL' if science else 'Label outside the frame',
                           'requested_change':'Establish and label the actual confidence' if science else 'Move label inside axes'}]
    return out


@pytest.fixture
def candidate(tmp_path):
    data='limit_data/DarkPhoton/Test.txt'
    proposal=dict(operation='new_limit',data_file_path=data,plotfuncs_file='PlotFuncs.py',
                  notebook_path='DarkPhoton.ipynb',notebook_call='DarkPhoton.Test(ax)')
    (tmp_path/data).parent.mkdir(parents=True)
    (tmp_path/data).write_text('1 1e-3\n2 2e-3\n')
    (tmp_path/'PlotFuncs.py').write_text('class DarkPhoton:\n    def Test(ax,col="red",fs=12,lw=1,text_on=True):\n        pass\n')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[{'cell_type':'code','metadata':{},
        'outputs':[],'execution_count':None,'source':['fig,ax = DarkPhoton.FigSetup()\n','DarkPhoton.Test(ax)\n',"MySaveFig(fig,'DarkPhoton')\n"]}]}
    (tmp_path/'DarkPhoton.ipynb').write_text(json.dumps(nb))
    normal=['plots/DarkPhoton.pdf','plots/plots_png/DarkPhoton.png']
    highlight=['plots/DarkPhoton_highlighted.pdf','plots/plots_png/DarkPhoton_highlighted.png']
    for name in normal+highlight:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original '+name.encode())
    return dict(extraction=SimpleNamespace(arxiv_id='2610.02083',confidence_level=.95,notes='95% placeholder'),
                paper_pdf=b'%PDF fixture', proposal=proposal,repo_root=tmp_path,plot_files=normal,highlight_files=highlight)


@pytest.mark.parametrize('change', ['missing_check','empty_evidence','blocking_approval','scientific_repair','uncertain_approval','missing_action'])
def test_verdict_cannot_silently_approve_or_repair_scientific_doubt(change):
    out=verdict()
    if change=='missing_check': del out['checks']['novelty']
    if change=='empty_evidence': out['checks']['novelty']['evidence']=''
    if change=='blocking_approval': out=verdict('needs_human_review',True);out['decision']='approve'
    if change=='scientific_repair': out=verdict('revise_plot',True)
    if change=='uncertain_approval':out['checks']['confidence_level']['status']='uncertain'
    if change=='missing_action':out=verdict('needs_human_review',True);out['findings'][0]['requested_change']=''
    with pytest.raises(review.PublicationReviewError):review.validate_verdict(out)


def events_for(required, answer, model, failed=None):
    events=[]
    for i,path in enumerate(required):
        events += [{'type':'assistant','message':{'content':[{'type':'tool_use','name':'Read','id':str(i),'input':{'file_path':path}}]}},
                   {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':str(i),'is_error':path==failed,'content':'read'}]}}]
    events.append({'type':'result','subtype':'success','is_error':False,'result':json.dumps(answer),'modelUsage':{model:{}}})
    return '\n'.join(json.dumps(e) for e in events)


@pytest.mark.parametrize('failed', [None, 'plot.png'])
def test_readonly_agent_requires_actual_successful_image_reads(tmp_path, monkeypatch, failed):
    packet=tmp_path/'evidence';packet.mkdir()
    required=['paper.pdf','plot.png']
    monkeypatch.setenv('AAL_BACKEND','claude-cli')
    def run(argv, **kw):
        assert argv[argv.index('--tools')+1]=='Read,Glob,Grep'
        assert '--no-session-persistence' in argv and '--strict-mcp-config' in argv
        assert not Path(kw['cwd']).is_relative_to(packet)
        assert str(packet / 'paper.pdf') in kw['input']
        assert 'ANTHROPIC_API_KEY' not in kw['env']
        model=argv[argv.index('--model')+1]
        return SimpleNamespace(returncode=0,stderr='',stdout=events_for(required,verdict(),model,failed))
    monkeypatch.setattr(review.subprocess,'run',run)
    if failed:
        with pytest.raises(review.PublicationReviewError,match='did not inspect'):
            review.run_review_agent(packet,role='reviewer',required=required)
    else:
        result,meta=review.run_review_agent(packet,role='reviewer',required=required)
        assert result['decision']=='approve' and meta['inspected_files']==['paper.pdf','plot.png']


@pytest.mark.parametrize('mode', ['timeout','budget','bad_json','missing_reads'])
def test_agent_transport_has_no_fail_open(tmp_path, monkeypatch, mode):
    packet=tmp_path/'evidence';packet.mkdir();monkeypatch.setenv('AAL_BACKEND','claude-cli')
    def run(argv, **kw):
        if mode=='timeout':raise subprocess.TimeoutExpired(argv,1)
        event={'type':'result','subtype':'success','is_error':False,'result':'{}'}
        if mode=='budget':event.update(subtype='error_max_budget_usd',result=json.dumps(verdict()))
        if mode=='bad_json':event['result']='not json'
        return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(event))
    monkeypatch.setattr(review.subprocess,'run',run)
    with pytest.raises(review.PublicationReviewError):
        review.run_review_agent(packet,role='reviewer',required=['plot.png'] if mode=='missing_reads' else [])


def test_scientific_block_is_actionable_and_never_sent_to_repair(candidate, monkeypatch):
    calls=[]
    def agent(directory, **kw):calls.append(kw['role']);return verdict('needs_human_review',True),{}
    monkeypatch.setattr(review,'run_review_agent',agent)
    with pytest.raises(review.PublicationReviewError,match='Establish and label'):
        review.review_for_publication(**candidate)
    assert calls==['reviewer']
    report=next(candidate['repo_root'].glob('pipeline/logs/publication_review/**/review.json'))
    assert json.loads(report.read_text())['history'][0]['verdict']['decision']=='needs_human_review'
    assert not list(candidate['repo_root'].glob('pipeline/reviews/*.json'))


def test_display_repair_rerenders_then_obtains_fresh_independent_approval(candidate, monkeypatch):
    calls=[];renders=[]
    repair={'decision':'apply','reason':'Move label into frame','label':{'text':'Test','x':1.5,'y':.01}}
    answers=iter([verdict('revise_plot'),repair,verdict()])
    def agent(directory, **kw):calls.append(kw['role']);return next(answers),{'model':'fixture'}
    def render(*args):
        renders.append(args)
        for path in candidate['plot_files']+candidate['highlight_files']:
            (candidate['repo_root']/path).write_text('rerendered')
        return candidate['plot_files'],candidate['highlight_files']
    monkeypatch.setattr(review,'run_review_agent',agent);monkeypatch.setattr(review,'generate_review_plots',render)
    result=review.review_for_publication(**candidate)
    assert calls==['reviewer','plot_repair','reviewer'] and len(renders)==1
    report=json.loads((candidate['repo_root']/result.report_path).read_text())
    assert len(report['history'])==2 and report['history'][0]['repair']['adjustments']==repair
    assert (candidate['repo_root']/candidate['proposal']['data_file_path']).read_text()=='1 1e-3\n2 2e-3\n'
    review.verify_approval(result,candidate['repo_root'])
    (candidate['repo_root']/candidate['highlight_files'][1]).write_text('changed after review')
    with pytest.raises(review.PublicationReviewError,match='stale'):
        review.verify_approval(result,candidate['repo_root'])


def test_repair_budget_is_bounded_and_restores_notebook(candidate,monkeypatch):
    root=candidate['repo_root'];original=(root/'DarkPhoton.ipynb').read_bytes();calls=[]
    def agent(directory,**kw):
        calls.append(kw['role'])
        return (verdict('revise_plot') if kw['role']=='reviewer' else {'decision':'apply','reason':'Larger font','call_kwargs':{'fs':18}}),{}
    monkeypatch.setattr(review,'run_review_agent',agent)
    monkeypatch.setattr(review,'generate_review_plots',lambda *a:(candidate['plot_files'],candidate['highlight_files']))
    with pytest.raises(review.PublicationReviewError,match='blocked'):
        review.review_for_publication(**candidate)
    assert calls==['reviewer','plot_repair','reviewer','plot_repair','reviewer']
    assert (root/'DarkPhoton.ipynb').read_bytes()==original


def test_artifacts_changed_during_review_cannot_be_approved(candidate,monkeypatch):
    def agent(directory,**kw):
        (candidate['repo_root']/'DarkPhoton.ipynb').write_text('different notebook')
        return verdict(),{}
    monkeypatch.setattr(review,'run_review_agent',agent)
    with pytest.raises(review.PublicationReviewError,match='changed after review'):
        review.review_for_publication(**candidate)


@pytest.mark.parametrize('bad', [{'decision':'apply','reason':'Change science','data_points':[[1,2]]},
                                {'decision':'apply','reason':'Zoom','axis_limits':{'x':[2,1]}},
                                {'decision':'apply','reason':'Execute','call_kwargs':{'RescaleByMass':True}},
                                {'decision':'apply','reason':'No change'}])
def test_repair_accepts_display_choices_only(candidate,bad):
    before=(candidate['repo_root']/'DarkPhoton.ipynb').read_bytes()
    with pytest.raises(review.PlotGenerationError):apply_plot_adjustments(candidate['repo_root'],candidate['proposal'],bad)
    assert (candidate['repo_root']/'DarkPhoton.ipynb').read_bytes()==before


def test_agent_label_is_visible_in_highlight_and_axis_change_precedes_drawing(candidate):
    call=apply_plot_adjustments(candidate['repo_root'],candidate['proposal'],{
        'decision':'apply','reason':'Keep label visible','axis_limits':{'x':[.1,10]},
        'label':{'text':"Test ' label",'x':1,'y':.1},'call_kwargs':{'text_on':False}})
    nb=json.loads((candidate['repo_root']/'DarkPhoton.ipynb').read_text())
    source=''.join(nb['cells'][0]['source'])
    assert source.index('ax.set_xlim')<source.index(call)
    highlighted,_=_build_highlight_notebook(nb,call)
    source=''.join(highlighted['cells'][1]['source'])
    active=source.split('_HIGHLIGHT_ACTIVE = True\n',1)[1].split('_HIGHLIGHT_ACTIVE = False',1)[0]
    assert '# AAL publication label' in active


@pytest.mark.parametrize("status", ["pass", "fail", "uncertain"])
def test_extra_evidence_is_retained_but_cannot_bypass_approval(status):
    out=verdict();out['checks']['additional_observation']={'status':status,'evidence':'Additional source comparison'}
    if status == 'pass':
        assert review.validate_verdict(out) == out
    else:
        with pytest.raises(review.PublicationReviewError):review.validate_verdict(out)


@pytest.mark.parametrize("missing", ["full", "highlight"])
def test_no_approval_without_corresponding_images(candidate,monkeypatch,missing):
    candidate['plot_files' if missing == 'full' else 'highlight_files']=[]
    monkeypatch.setattr(review,'run_review_agent',lambda *a,**k:pytest.fail('must have both images'))
    with pytest.raises(review.PublicationReviewError,match='PNG pairs'):
        review.review_for_publication(**candidate)
