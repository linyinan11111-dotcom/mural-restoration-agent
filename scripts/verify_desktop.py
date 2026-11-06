"""Acceptance run for the Qt WebEngine desktop shell.

Run: .\\.venv\\Scripts\\python.exe scripts/verify_desktop.py
"""
from io import BytesIO
from pathlib import Path
import base64
import json
import os
import shutil
import sys
import time
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --no-sandbox --disable-logging --log-level=3')
os.environ.setdefault('QTWEBENGINE_DISABLE_SANDBOX', '1')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication

from mural_agent.config import Settings
from mural_agent.desktop_web import WebMainWindow
from mural_agent.qt_setup import configure_application

OUTPUT = ROOT / 'artifacts'
ACTIVE = {'queued', 'planning', 'running', 'evaluating'}


def wait_until(app, predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def stroke_mask():
    mask = Image.new('L', (960, 640), 0)
    draw = ImageDraw.Draw(mask)
    points = [(130, 90), (155, 190), (142, 250), (170, 390), (158, 530)]
    draw.line(points, fill=255, width=14, joint='curve')
    points = [(540, 80), (515, 190), (548, 300), (527, 410), (550, 555)]
    draw.line(points, fill=255, width=14, joint='curve')
    return np.asarray(mask) > 127


def main():
    OUTPUT.mkdir(exist_ok=True)
    settings = Settings.from_env()
    settings.data_dir = OUTPUT / 'native-verification-web'
    settings.api_key = ''
    settings.api_model = ''
    app = QApplication([])
    configure_application(app)
    window = WebMainWindow(settings)
    window.resize(1440, 920)
    loaded = {}
    window.view.loadFinished.connect(lambda ok: loaded.update(done=True, ok=ok))
    window.show()
    if not wait_until(app, lambda: loaded.get('done'), 60):
        raise RuntimeError('Qt WebEngine did not finish loading ui/index.html')
    if not loaded.get('ok'):
        raise RuntimeError('Qt WebEngine failed to load ui/index.html')

    bridge = window.bridge
    demo = json.loads(bridge.loadDemo())
    image_id = demo['id']
    mask = stroke_mask()
    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format='PNG')
    mask_record = json.loads(bridge.saveMask(image_id, base64.b64encode(buffer.getvalue()).decode(), '[]'))
    request = {'image_id': image_id, 'mask_id': mask_record['id'], 'backend': 'lama', 'planner': 'rules',
               'candidate_count': 2, 'instructions': '', 'mode': 'conservative', 'registration_id': None,
               'structure_control': False,
               'budget': {'candidates': 2, 'revisions': 0, 'timeout_seconds': 240, 'max_api_calls': 1},
               'quality_thresholds': {}}
    job = json.loads(bridge.createJob(json.dumps(request)))

    def finished():
        jobs = json.loads(bridge.snapshot())['jobs']
        return next(item for item in jobs if item['id'] == job['id'])['status'] not in ACTIVE

    if not wait_until(app, finished, 300):
        raise RuntimeError('LaMa job did not finish within 300s')
    job = next(item for item in json.loads(bridge.snapshot())['jobs'] if item['id'] == job['id'])
    attempts = job.get('attempts', [])
    lama = next((candidate for candidate in job.get('candidates', [])
                 if candidate['tool'] == 'lama' and candidate.get('versions', {}).get('device') == 'cuda'), None)
    if job['status'] != 'review' or lama is None:
        diagnostic = {'status': job['status'], 'error': job.get('error'),
                      'attempts': [{'step': attempt.get('step', {}).get('tool'), 'status': attempt.get('status'),
                                    'error': attempt.get('error')} for attempt in attempts]}
        raise RuntimeError('Shipped shell did not produce a LaMa CUDA candidate: ' + json.dumps(diagnostic, ensure_ascii=False))

    bridge.selectCandidate(job['id'], lama['id'], 'WebEngine 离屏验收：合成纹样，真实 LaMa CUDA。')
    archive = window.engine.export(job['id'])
    exported = OUTPUT / 'verified-web-restoration.zip'
    shutil.copy2(archive, exported)
    with zipfile.ZipFile(exported) as zipped:
        original = np.asarray(Image.open(BytesIO(zipped.read('original/working.png'))))
        result = np.asarray(Image.open(BytesIO(zipped.read('result.png'))))
        saved_mask = np.asarray(Image.open(BytesIO(zipped.read('mask.png')))) > 127
        outside = int(np.abs(original[~saved_mask].astype(np.int16) - result[~saved_mask].astype(np.int16)).max())
        manifest = json.loads(zipped.read('manifest.json'))
    if outside != 0:
        raise RuntimeError(f'unmasked pixels changed: max difference {outside}')
    if manifest['selected_candidate']['id'] != lama['id']:
        raise RuntimeError('archive does not contain the selected LaMa candidate')

    window.grab().save(str(OUTPUT / 'desktop-web-preview.png'))
    report = {'interface': 'PySide6 Qt WebEngine + QWebChannel bridge',
              'verification_mode': 'isolated offscreen application, shipped ui/ shell',
              'web_server_required': False,
              'page_loaded': True,
              'job_status': job['status'],
              'tool': lama['tool'],
              'device': lama['versions']['device'],
              'candidate_elapsed_seconds': lama['elapsed_seconds'],
              'metrics': lama['metrics'],
              'outside_max_difference': outside,
              'candidate_count': len(job['candidates']),
              'attempts': [{'tool': attempt.get('step', {}).get('tool'), 'status': attempt.get('status')} for attempt in attempts],
              'archive': str(exported),
              'preview': str(OUTPUT / 'desktop-web-preview.png')}
    (OUTPUT / 'desktop-web-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    # QtWebEngine's offscreen teardown can abort the process after a successful
    # run; flush the report and leave explicitly so the exit code reflects it.
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    raise SystemExit(main())
