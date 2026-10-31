"""Small Windows EXE entry point for the locally installed native application."""
from pathlib import Path
import ctypes
import os
import subprocess
import sys


def main():
    here = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    root = next((candidate for candidate in (here, here.parent)
                 if any((candidate / p).is_file() for p in ('runtime/pythonw.exe', '.venv/Scripts/pythonw.exe')) and (candidate / 'mural_agent/desktop.py').is_file()), None)
    if root is None:
        ctypes.windll.user32.MessageBoxW(None, '请将拾壁.exe 保留在完整项目目录中。未找到本地运行环境，请先安装依赖。', '拾壁', 0x10)
        return 1
    (root / 'data').mkdir(exist_ok=True)
    # Do not pass PyInstaller's private DLL search directory to the Python/Qt child.
    ctypes.windll.kernel32.SetDllDirectoryW(None)
    with (root / 'data/desktop.log').open('a', encoding='utf-8') as log:
        interpreter = root / ('runtime/pythonw.exe' if (root/'runtime/pythonw.exe').exists() else '.venv/Scripts/pythonw.exe')
        subprocess.Popen([str(interpreter), '-m', 'mural_agent.desktop'],
                         cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

