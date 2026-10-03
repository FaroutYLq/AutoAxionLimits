"""Apply an agent's bounded display choices without accepting executable patches."""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

from .plot_regen import PlotGenerationError


def _number(value, name, low=0, high=1e300):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low < value <= high:
        raise PlotGenerationError(f'Invalid display value for {name}: {value!r}')
    return value


def apply_plot_adjustments(repo_root: Path, proposal: dict, answer: dict) -> str:
    """Only axes, supported display kwargs, and one annotation may change."""
    if set(answer) - {'decision', 'reason', 'axis_limits', 'call_kwargs', 'label'}:
        raise PlotGenerationError('Plot repair proposed unsupported changes')
    if answer.get('decision') != 'apply' or not isinstance(answer.get('reason'), str) or not answer['reason'].strip():
        raise PlotGenerationError(f'Plot repair requires human action: {answer.get("reason", "no supported repair")}')
    axes, kwargs, label = answer.get('axis_limits', {}), answer.get('call_kwargs', {}), answer.get('label')
    if not isinstance(axes, dict) or set(axes) - {'x', 'y'} or not isinstance(kwargs, dict) or set(kwargs) - {'col', 'fs', 'lw', 'text_on'}:
        raise PlotGenerationError('Repair may change only display settings')
    if not axes and not kwargs and label is None:
        raise PlotGenerationError('Repair agent proposed no changes')
    for axis, bounds in axes.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise PlotGenerationError('Axis limits must be [minimum, maximum]')
        lo, hi = (_number(v, axis) for v in bounds)
        if lo >= hi:
            raise PlotGenerationError('Axis limits must increase')
    for key, value in kwargs.items():
        if key in ('fs', 'lw'):
            _number(value, key, high=100 if key == 'fs' else 10)
        elif key == 'text_on' and not isinstance(value, bool):
            raise PlotGenerationError('text_on must be boolean')
        elif key == 'col' and (not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}|[a-zA-Z]+', value)):
            raise PlotGenerationError('Invalid colour')
    if label is not None:
        if not isinstance(label, dict) or set(label) - {'text', 'x', 'y', 'fs'} or not isinstance(label.get('text'), str) or not 0 < len(label['text']) <= 200:
            raise PlotGenerationError('Label must supply short text and positive data coordinates')
        for name in ('x', 'y'):
            _number(label.get(name), name)
        _number(label.get('fs', 14), 'label font size', high=100)
    call = proposal['notebook_call'].strip()
    expr = ast.parse(call, mode='eval').body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute) or not isinstance(expr.func.value, ast.Name):
        raise PlotGenerationError('Cannot safely identify the target plotting method')
    tree = ast.parse((repo_root / proposal['plotfuncs_file']).read_text())
    methods = [m for c in tree.body if isinstance(c, ast.ClassDef) and c.name == expr.func.value.id
               for m in c.body if isinstance(m, ast.FunctionDef) and m.name == expr.func.attr]
    if len(methods) != 1:
        raise PlotGenerationError('Ambiguous plotting method')
    supported = {arg.arg for arg in methods[0].args.args + methods[0].args.kwonlyargs}
    if set(kwargs) - supported and methods[0].args.kwarg is None:
        raise PlotGenerationError('Requested styling is not supported by this method')
    expr.keywords = [k for k in expr.keywords if k.arg not in kwargs]
    expr.keywords.extend(ast.keyword(arg=k, value=ast.Constant(value=v)) for k, v in kwargs.items())
    new_call = ast.unparse(expr)
    path = repo_root / proposal['notebook_path']
    nb = json.loads(path.read_text())
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        lines = ''.join(cell.get('source', [])).splitlines()
        targets = [i for i, line in enumerate(lines) if line.strip() == call]
        if not targets:
            continue
        index = targets[-1]
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        lines[index] = indent + new_call
        if label is not None:
            lines = [line for line in lines if not line.endswith('# AAL publication label')]
            index = next(i for i, line in enumerate(lines) if line.strip() == new_call)
            lines.insert(index + 1, indent + f"ax.text({label['x']!r}, {label['y']!r}, {label['text']!r}, fontsize={label.get('fs', 14)!r}, color='red', zorder=2000, clip_on=True)  # AAL publication label")
        for axis, bounds in axes.items():
            tag = f'# AAL publication {axis} axis'
            lines = [line for line in lines if not line.endswith(tag)]
            setup = [i for i, line in enumerate(lines) if re.match(r'\s*fig\s*,\s*ax\s*=.*FigSetup\(.*\)\s*$', line)]
            if len(setup) != 1:
                raise PlotGenerationError('Axis repair needs one unambiguous single-line FigSetup')
            lines.insert(setup[0] + 1, f'ax.set_{axis}lim({bounds[0]!r}, {bounds[1]!r})  {tag}')
        cell['source'] = [line + '\n' for line in lines]
        path.write_text(json.dumps(nb, indent=1) + '\n')
        return new_call
    raise PlotGenerationError('Automatic styling of an indirect/grouped call needs human review')
