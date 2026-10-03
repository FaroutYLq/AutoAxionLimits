"""Highlight the drawn exclusion, not a new shape guessed from raw table rows."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from types import SimpleNamespace

from pipeline.plot_regen import _build_highlight_notebook


@pytest.fixture
def render(monkeypatch, tmp_path):
    # The injected notebook patches class methods; restore them after each test.
    for name in ['fill_between','fill','plot','text','axhline','axvline','arrow']:
        monkeypatch.setattr(Axes,name,getattr(Axes,name))
    monkeypatch.setattr(Figure,'text',Figure.text)
    def run(method, data):
        path=tmp_path/'raw.txt';np.savetxt(path,data)
        source="""fig,ax = plt.subplots(figsize=(4,4),dpi=100)
ax.set(xscale='log',yscale='log',xlim=(1,100),ylim=(1e-6,1))
ax.fill_between([1,100],1e-3,1,color='crimson',zorder=100)
Example.Limit(ax)
MySaveFig(fig,'test')
"""
        nb={'cells':[{'cell_type':'code','source':source.splitlines(keepends=True),'metadata':{},'outputs':[]}]}
        patched,_=_build_highlight_notebook(nb,'Example.Limit(ax)',str(path))
        ns={'plt':plt,'Example':SimpleNamespace(Limit=method),'MySaveFig':lambda *a:None}
        for cell in patched['cells']:exec(''.join(cell['source']),ns)
        ns['fig'].canvas.draw()
        return ns['fig'],ns['ax']
    yield run
    plt.close('all')


def color_at(fig,ax,x,y):
    x,y=ax.transData.transform((x,y))
    pixels=np.asarray(fig.canvas.buffer_rgba())
    return pixels[pixels.shape[0]-1-int(y),int(x),:3]


def test_two_endpoint_upper_bound_is_continuous_above_boundary(render):
    def limit(ax,col='red',lw=3):
        ax.fill_between([2,50],[2e-5,2e-5],1,color=col,zorder=1)
        ax.plot([2,50],[2e-5,2e-5],color=col,lw=lw,zorder=1)
        ax.text(5,3e-5,'New bound',color=col)
    fig,ax=render(limit,[[2,2e-5],[50,2e-5]])
    assert len(ax.collections)==2  # background + actual bound; no endpoint stripes
    assert ax.collections[1].get_zorder()>ax.collections[0].get_zorder()
    assert ax.texts[0].get_zorder()>ax.collections[1].get_zorder()
    # Above the unrelated high-z-order background edge remains red, not grey.
    for y in [1e-4,1e-2,.1]:
        r,g,b=color_at(fig,ax,10,y)
        assert r>220 and g<40 and b<40
    assert np.all(color_at(fig,ax,10,5e-6)>240)  # below upper bound stays unexcluded


def test_plotting_conversion_and_finite_band_geometry_are_preserved(render):
    raw=np.array([[2,2e-6],[10,3e-6],[50,4e-6]])
    def limit(ax,col='red',lw=3):
        converted=raw[:,1]*10
        ax.fill_between(raw[:,0],converted,converted*4,color=col)
    fig,ax=render(limit,raw)
    assert len(ax.collections)==2
    vertices=ax.collections[1].get_paths()[0].vertices
    assert vertices[:,1].min()==pytest.approx(2e-5)
    assert vertices[:,1].max()==pytest.approx(1.6e-4)
    assert np.all(color_at(fig,ax,10,1e-5)>240)  # raw values were not overlaid
    r,g,b=color_at(fig,ax,10,6e-5)
    assert r>220 and g<40 and b<40
    r,g,b=color_at(fig,ax,10,.1)
    assert abs(int(r)-int(g))<3 and abs(int(g)-int(b))<3  # above real band is background


def test_single_point_uses_transformed_artist_marker_without_fake_width(render):
    def limit(ax,col='red',lw=3):
        ax.plot([10],[2e-4],color=col,lw=lw)  # method converted the raw value
    _,ax=render(limit,[[10,2e-6]])
    assert len(ax.collections)==1  # no invented filled exclusion or raw-data spike
    assert ax.lines[0].get_marker()=='o'
    assert ax.lines[0].get_ydata()[0]==pytest.approx(2e-4)
