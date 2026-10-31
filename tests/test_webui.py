"""Regression tests for the shipped Qt WebEngine shell and its QWebChannel bridge.

These tests deliberately drive the same files that ship (ui/index.html, ui/app.js,
mural_agent/desktop_web.py) instead of the legacy Qt Widgets window, so a broken
frontend cannot pass the suite.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --no-sandbox')
os.environ.setdefault('QTWEBENGINE_DISABLE_SANDBOX', '1')

import base64
import json
import time
import zipfile
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('PySide6.QtWebEngineWidgets')

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from mural_agent.config import ROOT, Settings
from mural_agent.desktop_web import WebMainWindow
from mural_agent.storage import new_id

pytestmark = pytest.mark.web
ACTIVE = {'queued', 'planning', 'running', 'evaluating'}


def png(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def wait_until(app, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def run_js(window, script, timeout=10):
    box = {}
    loop = QEventLoop()

    def callback(value):
        box['value'] = value
        loop.quit()

    window.view.page().runJavaScript(script, callback)
    QTimer.singleShot(int(timeout * 1000), loop.quit)
    loop.exec()
    return box.get('value')


def isolated_settings(tmp_path):
    return Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt',
                    sam_model=tmp_path / 'missing-sam', diffusion_model=tmp_path / 'missing-sd',
                    controlnet_model=tmp_path / 'missing-cn', vision_model=tmp_path / 'missing-qwen',
                    research_adapter=tmp_path / 'missing-research.json')


@pytest.fixture()
def window(qt_app, tmp_path):
    shell = WebMainWindow(isolated_settings(tmp_path))
    loaded = {}
    shell.view.loadFinished.connect(lambda ok: loaded.update(done=True, ok=ok))
    shell.show()
    assert wait_until(qt_app, lambda: loaded.get('done'), timeout=30), 'WebEngine page did not finish loading'
    assert loaded.get('ok'), 'WebEngine failed to load ui/index.html'
    try:
        yield shell
    finally:
        shell.engine.executor.shutdown(wait=False, cancel_futures=True)
        shell.close()
        qt_app.processEvents()


def test_shipped_shell_loads_and_exposes_bridge(window, qt_app):
    assert window.view.url().toString().endswith('/ui/index.html')
    snapshot = json.loads(window.bridge.snapshot())
    assert snapshot['images'] == []
    assert snapshot['models']['lama'] is False
    assert snapshot['aux_version'] >= 0
    assert run_js(window, "document.title || ''") != ''
    assert run_js(window, "Boolean(document.querySelector('#damage-list') && document.querySelector('#reference-modal') && document.querySelector('#verify-modal'))") is True
    # QtWebEngine has no default prompt implementation; the reference workflow must
    # not depend on window.prompt.
    assert 'window.prompt' not in (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')


def test_settings_button_hit_area_does_not_cover_the_page(window, qt_app):
    """A viewport-sized pseudo-element on the topbar gear intercepted clicks and
    opened the API settings dialog from anywhere on the right half of the window."""
    probe = """(() => {
      const points = [[innerWidth - 120, 300], [innerWidth - 60, innerHeight - 200],
                      [innerWidth / 2, 120], [innerWidth - 30, 60]];
      return JSON.stringify(points.map(([x, y]) => {
        const element = document.elementFromPoint(x, y);
        if (!element) return 'none';
        if (element.closest('[data-action="open-settings"]')) return 'settings';
        return (element.id || element.className || element.tagName).toString();
      }));
    })()"""
    targets = json.loads(run_js(window, probe))
    assert 'settings' not in targets, targets
    click_far_right = """(() => {
      const element = document.elementFromPoint(innerWidth - 100, innerHeight - 300);
      if (element) element.dispatchEvent(new MouseEvent('click', {bubbles: true}));
      return document.querySelector('#settings-modal').classList.contains('hidden');
    })()"""
    assert run_js(window, click_far_right) is True
    # The real button must still open the dialog.
    assert run_js(window, "document.querySelector('.topbar [data-action=\"open-settings\"]').click(); true") is True
    assert run_js(window, "document.querySelector('#settings-modal').classList.contains('hidden')") is False
    run_js(window, "document.querySelector('#settings-modal').querySelector('[data-action=\"close-settings\"]').click(); true")


def test_grayscale_mask_round_trip_keeps_only_white_pixels(window, qt_app):
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[16:32, 24:40] = 255
    encoded = base64.b64encode(png(mask)).decode()
    script = ("(() => { const canvas = document.querySelector('#mask-canvas');"
              " canvas.width = 64; canvas.height = 48;"
              f" window.__shibui.showMaskFromBase64('{encoded}'); return 'scheduled'; }})()")
    assert run_js(window, script) == 'scheduled'
    assert wait_until(qt_app, lambda: run_js(window, 'Boolean(window.__shibui.state.baseMask)') is True, timeout=10)
    exported = run_js(window, 'window.__shibui.exportMaskBase64()')
    assert isinstance(exported, str) and exported
    result = np.asarray(Image.open(BytesIO(base64.b64decode(exported))).convert('L'))
    assert result.shape == mask.shape
    assert np.array_equal(result > 127, mask > 127), 'grayscale mask round-trip changed the editable region'


def test_bridge_job_flow_produces_verified_archive(window, qt_app):
    demo = json.loads(window.bridge.loadDemo())
    image_id = demo['id']
    mask = np.zeros((640, 960), dtype=np.uint8)
    mask[280:330, 440:520] = 255
    saved = json.loads(window.bridge.saveMask(image_id, base64.b64encode(png(mask)).decode(), '[]'))
    request = {'image_id': image_id, 'mask_id': saved['id'], 'backend': 'opencv', 'planner': 'rules',
               'candidate_count': 2, 'instructions': '', 'mode': 'conservative', 'registration_id': None,
               'structure_control': False,
               'budget': {'candidates': 2, 'revisions': 0, 'timeout_seconds': 60, 'max_api_calls': 1},
               'quality_thresholds': {}}
    job = json.loads(window.bridge.createJob(json.dumps(request)))
    assert job['id']

    def finished():
        jobs = json.loads(window.bridge.snapshot())['jobs']
        return next(item for item in jobs if item['id'] == job['id'])['status'] not in ACTIVE

    assert wait_until(qt_app, finished, timeout=30)
    latest = next(item for item in json.loads(window.bridge.snapshot())['jobs'] if item['id'] == job['id'])
    assert latest['status'] == 'review', latest
    assert len(latest['candidates']) == 2
    assert all(candidate['metrics']['outside_unchanged'] for candidate in latest['candidates'])
    selected = latest['candidates'][0]
    reviewed = json.loads(window.bridge.selectCandidate(job['id'], selected['id'], 'shipped shell regression'))
    assert reviewed['selected_id'] == selected['id']

    archive = window.engine.export(job['id'])
    with zipfile.ZipFile(archive) as zipped:
        original = np.asarray(Image.open(BytesIO(zipped.read('original/working.png'))))
        result = np.asarray(Image.open(BytesIO(zipped.read('result.png'))))
        saved_mask = np.asarray(Image.open(BytesIO(zipped.read('mask.png')))) > 127
        assert np.array_equal(original[~saved_mask], result[~saved_mask])
        manifest = json.loads(zipped.read('manifest.json'))
        assert manifest['selected_candidate']['id'] == selected['id']
        assert manifest['job']['review']['note'] == 'shipped shell regression'


def test_damage_panel_renders_candidates_and_recommendation(window, qt_app):
    script = """(() => {
      const state = window.__shibui.state;
      state.image = {id: 'fake-image', width: 960, height: 640};
      state.damageItems = [{index: 1, type: 'crack', bbox: [100, 100, 200, 800], size: 'large',
        confidence: 0.9, evidence: 'demo', recommendation: {method: 'LaMa 纹理补全'}}];
      window.__shibui.renderDamage('模型输出仅为候选');
      return JSON.stringify({count: document.querySelectorAll('.damage-item').length,
                             note: document.querySelector('#damage-note').textContent});
    })()"""
    rendered = json.loads(run_js(window, script))
    assert rendered['count'] == 1
    assert 'LaMa' in rendered['note']


def test_merge_proposal_and_aux_version_contract(window, qt_app):
    demo = json.loads(window.bridge.loadDemo())
    image_id = demo['id']
    window.bridge.setCurrentImage(image_id)
    proposal_id = new_id()
    proposal_mask = np.zeros((640, 960), dtype=np.uint8)
    proposal_mask[300:340, 460:520] = 255
    Image.fromarray(proposal_mask).save(window.store.folder(proposal_id) / 'mask.png')
    window.store.put('proposal', {'id': proposal_id, 'image_id': image_id,
                                  'created_at': '2026-01-01T00:00:00+00:00',
                                  'fraction': float((proposal_mask > 0).mean()),
                                  'metadata': {'notice': 'test proposal'}, 'confirmed': False})
    current = np.zeros((640, 960), dtype=np.uint8)
    current[400:420, 600:660] = 255
    merged = json.loads(window.bridge.mergeProposal(proposal_id, base64.b64encode(png(current)).decode()))
    assert 'error' not in merged, merged
    assert merged['fraction'] > float((current > 0).mean())
    assert merged['mask_file_url']

    version = json.loads(window.bridge.snapshot())['aux_version']
    window.bridge.aux = {'status': 'done', 'result': {'notice': 'x'}}
    window.bridge.aux_version += 1
    assert json.loads(window.bridge.snapshot())['aux_version'] == version + 1
