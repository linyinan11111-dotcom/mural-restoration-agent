"""Bundle an independent CPython runtime and resolved package files, no base Conda dependency."""
from pathlib import Path, PurePosixPath
from importlib import metadata
import argparse
import json
import shutil
import sys
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.markers import default_environment
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'scripts'))
from download_models import download_file, digest

ROOT_PACKAGES = ['PySide6-Essentials', 'PySide6-Addons', 'PySide6-WebEngine', 'numpy', 'pillow', 'opencv-python-headless', 'httpx', 'pydantic',
                 'torch', 'torchvision', 'transformers', 'diffusers', 'accelerate', 'scikit-image', 'lpips']


def resolve():
    queue, packages = list(ROOT_PACKAGES), {}
    environment = default_environment()
    environment['extra'] = ''
    while queue:
        name = canonicalize_name(queue.pop())
        if name in packages:
            continue
        distribution = metadata.distribution(name)
        packages[name] = distribution
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate(environment):
                queue.append(requirement.name)
    return packages


def build(destination, include_models):
    destination.mkdir(parents=True, exist_ok=True)
    runtime = destination/'runtime'
    runtime.mkdir(exist_ok=True)
    cache = ROOT/'build/python-3.13.12-embed-amd64.zip'
    download_file('https://www.python.org/ftp/python/3.13.12/python-3.13.12-embed-amd64.zip', cache)
    with zipfile.ZipFile(cache) as archive:
        archive.extractall(runtime)
    pth = runtime/'python313._pth'
    pth.write_text('python313.zip\n.\nLib/site-packages\n..\nimport site\n', encoding='utf-8')
    site_packages = runtime/'Lib/site-packages'
    site_packages.mkdir(parents=True, exist_ok=True)
    packages = resolve()
    inventory = []
    for name, distribution in packages.items():
        print('Bundling '+name+' '+distribution.version, flush=True)
        count = 0
        for item in distribution.files or []:
            relative = PurePosixPath(str(item).replace('\\', '/'))
            if '..' in relative.parts or relative.is_absolute() or '__pycache__' in relative.parts:
                continue
            source = Path(distribution.locate_file(item))
            if not source.is_file():
                continue
            target = site_packages/Path(str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or source.stat().st_size != target.stat().st_size:
                shutil.copy2(source, target)
            count += 1
        inventory.append({'name': name, 'version': distribution.version, 'files': count,
                          'license': distribution.metadata.get('License-Expression') or distribution.metadata.get('License', '')})
    for directory in ('mural_agent', 'assets', 'scripts', 'datasets', 'ui'):
        shutil.copytree(ROOT/directory, destination/directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.part'))
    for name in ('README.md', 'VALIDATION.md', '.env.example'):
        shutil.copy2(ROOT/name, destination/name)
    shutil.copy2(ROOT/'拾壁.exe', destination/'拾壁.exe')
    if include_models:
        shutil.copytree(ROOT/'models', destination/'models', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*.part'))
    (destination/'start.cmd').write_text('@echo off\r\ncd /d "%~dp0"\r\nstart "" "%~dp0runtime\\pythonw.exe" -m mural_agent.desktop\r\n', encoding='ascii')
    manifest = {'format': 1, 'python': '3.13.12 embedded AMD64', 'python_archive_sha256': digest(cache),
                'packages': inventory, 'includes_models': include_models,
                'excluded': ['API credentials', 'user workspace', 'Conda paths']}
    (destination/'bundle-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Portable app created: '+str(destination), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'dist/Shibi-1.0')
    parser.add_argument('--include-models', action='store_true')
    args = parser.parse_args()
    build(args.output.resolve(), args.include_models)
