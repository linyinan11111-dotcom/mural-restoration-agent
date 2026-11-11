"""P1-2: a base install must declare Pydantic 2.

desktop_web imports Engine, Engine imports planner, and planner imports pydantic
at module level. requirements-dev.txt's FastAPI extra used to hide the missing
pin, so a clean `pip install -r requirements.txt` could not start the desktop shell.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines(path):
    lines = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if line and not line.startswith('-'):
            lines.append(line)
    return lines


def test_base_requirements_declare_pydantic_v2_for_desktop_startup():
    planner = (ROOT / 'mural_agent' / 'planner.py').read_text(encoding='utf-8')
    engine = (ROOT / 'mural_agent' / 'engine.py').read_text(encoding='utf-8')
    desktop = (ROOT / 'mural_agent' / 'desktop_web.py').read_text(encoding='utf-8')
    assert re.search(r'(?m)^from pydantic import ', planner)
    assert 'from .planner import' in engine
    assert 'from .engine import Engine' in desktop

    base = _requirement_lines(ROOT / 'requirements.txt')
    pins = [line for line in base if line.lower().startswith('pydantic')]
    assert pins == ['pydantic>=2,<3'], pins
    # The dev extra's FastAPI dependency must not be the only place pydantic appears.
    dev = (ROOT / 'requirements-dev.txt').read_text(encoding='utf-8')
    assert 'fastapi' in dev.lower()
    assert not any(line.lower().startswith('pydantic') for line in _requirement_lines(ROOT / 'requirements-dev.txt'))
