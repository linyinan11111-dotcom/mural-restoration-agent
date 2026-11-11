"""Keep desktop imports usable with runtime requirements alone."""
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.version import Version

from mural_agent.config import ROOT


def test_desktop_declares_pydantic_2_as_a_direct_runtime_dependency():
    requirements = [Requirement(line) for line in (ROOT / 'requirements.txt').read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
    pydantic = next((item for item in requirements if item.name.lower() == 'pydantic'), None)
    assert pydantic is not None, 'The desktop planner needs Pydantic even without FastAPI/dev dependencies.'
    assert Version('2.0') in pydantic.specifier
    assert Version('2.99') in pydantic.specifier
    assert Version('1.10') not in pydantic.specifier
    assert Version('3.0') not in pydantic.specifier


def test_desktop_import_does_not_require_the_development_api_adapter():
    # FastAPI used to supply Pydantic accidentally. Block it in a fresh process;
    # CI additionally runs this import before installing any dev dependencies.
    code = """
import importlib.abc
import sys
class NoDevAPI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'uvicorn'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoDevAPI())
import mural_agent.desktop
from mural_agent.planner import Budget
assert Budget(candidates=1).model_dump()['candidates'] == 1
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                   capture_output=True, text=True, timeout=45)
