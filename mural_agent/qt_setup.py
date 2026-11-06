"""Shared Qt application initialization for the desktop shell."""

import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase


def configure_application(app):
    fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ('msyh.ttc', 'simsun.ttc', 'segoeui.ttf'):
        path = fonts / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont('Microsoft YaHei', 10))
    app.setStyle('Fusion')
