"""Native QWidget render and real GPU workflow in an isolated offscreen test.

This does not inspect or automate an existing desktop window.
"""
from pathlib import Path
from io import BytesIO
import json
import os
import sys
import time
import zipfile
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
import numpy as np
from PIL import Image
from mural_agent.config import Settings, ROOT
from mural_agent.desktop import MainWindow, configure_application


def main():
    app = QApplication([])
    configure_application(app)
    settings = Settings.from_env()
    settings.data_dir = ROOT / 'artifacts/native-verification'
    settings.api_key = ''
    settings.api_model = ''
    window = MainWindow(settings)
    window.resize(1420, 940)
    window.show()
    app.processEvents()
    window.load_demo()
    app.processEvents()
    window.canvas.checkpoint()
    points = [(130, 90), (155, 190), (142, 250), (170, 390), (158, 530)]
    for x, y in points:
        window.canvas.stroke(QPointF(x, y))
    window.canvas.last = None
    window.start_job()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        app.processEvents()
        window.poll_job()
        if not window.running:
            break
        time.sleep(.05)
    if window.job['status'] != 'review':
        raise RuntimeError(json.dumps(window.job, ensure_ascii=True))
    window.candidates.setCurrentRow(0)
    window.choose_candidate(window.candidates.item(0))
    window.tabs.setCurrentIndex(1)
    window.review_note.setText('原生桌面离屏验收：合成纹样，真实 LaMa GPU 推理。')
    window.select_candidate()
    app.processEvents()
    folder = ROOT / 'artifacts'
    window.grab().save(str(folder / 'desktop-preview.png'))
    archive = window.engine.export(window.job['id'])
    with zipfile.ZipFile(archive) as zipped:
        original = np.asarray(Image.open(BytesIO(zipped.read('original/working.png'))))
        result = np.asarray(Image.open(BytesIO(zipped.read('result.png'))))
        mask = np.asarray(Image.open(BytesIO(zipped.read('mask.png')))) > 127
        assert np.array_equal(original[~mask], result[~mask])
    candidate = window.candidate
    report = {'interface': 'PySide6 native Qt Widgets', 'verification_mode': 'isolated offscreen application',
              'web_server_required': False, 'job_status': window.job['status'],
              'tool': candidate['tool'], 'device': candidate['versions']['device'],
              'candidate_elapsed_seconds': candidate['elapsed_seconds'], 'metrics': candidate['metrics'],
              'archive': str(archive), 'candidate_count': len(window.job['candidates'])}
    (folder / 'desktop-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    window.timer.stop()
    window.close()
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

