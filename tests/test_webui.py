"""Regression tests for the shipped Qt WebEngine shell and QWebChannel bridge."""
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
    assert run_js(window, "Boolean(document.querySelector('#damage-list') && document.querySelector('#reference-modal') && document.querySelector('#verify-modal') && document.querySelector('#auto-button') && document.querySelector('#allow-generative-attempt') && document.querySelector('#damage-generative'))") is True
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


def test_auto_restore_creates_a_traceable_autonomous_job(window, qt_app, monkeypatch):
    import mural_agent.auto_restore as auto_module

    def fake_assess(engine, image_record):
        return {'items': [{'index': 1, 'type': 'paint_loss', 'bbox': [400, 300, 560, 380],
                           'size': 'medium', 'confidence': 0.9, 'evidence': 'fake assessment'}],
                'limitations': 'fake limitations'}

    def fake_refine(engine, image, item):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        height, width = mask.shape
        x0, y0, x1, y1 = item['bbox']
        mask[round(y0 * height / 1000):round(y1 * height / 1000),
             round(x0 * width / 1000):round(x1 * width / 1000)] = 255
        return mask, {'method': 'fake_refine'}

    monkeypatch.setattr(auto_module, 'assess_image', fake_assess)
    monkeypatch.setattr(auto_module, 'refine_item', fake_refine)
    # The bridge only checks that a local VLM is installed; the actual inference
    # is monkeypatched above, so a fake provenance marker keeps this test hermetic.
    vision = window.store.root / 'fake-vision'
    vision.mkdir(exist_ok=True)
    (vision / 'provenance.json').write_text('{}', encoding='utf-8')
    window.bridge.settings.vision_model = vision

    demo = json.loads(window.bridge.loadDemo())
    started = json.loads(window.bridge.autoRestore(demo['id']))
    assert started['status'] == 'running'

    assert wait_until(qt_app, lambda: json.loads(window.bridge.snapshot())['aux'].get('status') in {'done', 'error'}, 30)
    aux = json.loads(window.bridge.snapshot())['aux']
    assert aux['status'] == 'done', aux
    auto = aux['result']['auto']
    assert auto['mask']['fraction'] > 0
    assert auto['job']['planner'] == 'auto'
    assert auto['steps']

    job_id = auto['job']['id']

    def finished():
        jobs = json.loads(window.bridge.snapshot())['jobs']
        return next(entry for entry in jobs if entry['id'] == job_id)['status'] not in ACTIVE

    assert wait_until(qt_app, finished, 30)
    job = next(entry for entry in json.loads(window.bridge.snapshot())['jobs'] if entry['id'] == job_id)
    assert job['status'] == 'review', job
    assert job['auto']['mask_fraction'] > 0
    assert job['auto']['steps'][0]['tool'] in {'telea', 'ns', 'lama', 'color'}
    assert job['candidates'], 'autonomous job must produce at least one candidate'
    assert job['candidates'][0]['metrics']['outside_unchanged']


def test_auto_analyze_button_shows_defect_list_without_submitting(window, qt_app, monkeypatch):
    """The AI button now analyzes and lists defects; it must not auto-create a job."""
    import mural_agent.auto_restore as auto_module

    monkeypatch.setattr(auto_module, 'assess_image', lambda engine, record: {
        'items': [{'index': 1, 'type': 'paint_loss', 'bbox': [400, 300, 560, 380],
                   'size': 'medium', 'confidence': 0.9, 'evidence': 'fake'}],
        'limitations': 'fake'})

    def fake_refine(engine, image, item):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        height, width = mask.shape
        mask[round(300 * height / 1000):round(380 * height / 1000),
             round(400 * width / 1000):round(560 * width / 1000)] = 255
        return mask, {'method': 'fake'}

    monkeypatch.setattr(auto_module, 'refine_item', fake_refine)
    vision = window.store.root / 'fake-vision'
    vision.mkdir(exist_ok=True)
    (vision / 'provenance.json').write_text('{}', encoding='utf-8')
    window.bridge.settings.vision_model = vision

    assert run_js(window, "document.querySelector('[data-action=\"load-demo\"]').click(); true") is True
    assert wait_until(qt_app, lambda: run_js(window, 'Boolean(window.__shibui.state.image)') is True, 30)
    assert wait_until(qt_app, lambda: run_js(window, "!document.querySelector('[data-action=\"auto-restore\"]').disabled") is True, 15)
    assert run_js(window, "document.querySelector('[data-action=\"auto-restore\"]').click(); true") is True
    assert wait_until(qt_app, lambda: json.loads(window.bridge.snapshot())['aux'].get('status') in {'done', 'error'}, 30)
    aux = json.loads(window.bridge.snapshot())['aux']
    assert aux['status'] == 'done', aux
    auto = aux['result']['auto']
    assert auto['job'] is None, 'analysis must not enqueue a repair job'
    assert auto.get('awaiting_confirmation') is True
    assert auto['defects']
    assert wait_until(qt_app, lambda: run_js(window, "document.querySelectorAll('.damage-item').length") >= 1, 20)
    assert run_js(window, "document.querySelector('#results-view').classList.contains('active')") is False
    assert json.loads(window.bridge.snapshot())['jobs'] == []


def test_damage_modal_and_manual_defect(window, qt_app):
    assert run_js(window, "document.querySelector('[data-action=\"load-demo\"]').click(); true") is True
    assert wait_until(qt_app, lambda: run_js(window, 'Boolean(window.__shibui.state.image)') is True, 30)
    assert run_js(window, "document.querySelector('[data-action=\"open-damage-list\"]').click(); true") is True
    assert wait_until(qt_app, lambda: run_js(window, "!document.querySelector('#damage-modal').classList.contains('hidden')") is True, 10)
    assert run_js(window, "document.querySelector('[data-action=\"close-damage-list\"]').click(); true") is True
    assert run_js(window, "document.querySelector('#damage-modal').classList.contains('hidden')") is True
    assert run_js(window, "document.querySelector('[data-action=\"open-damage-list\"]').click(); true") is True
    assert wait_until(qt_app, lambda: run_js(window, "!document.querySelector('#damage-modal').classList.contains('hidden')") is True, 10)
    modal_width = run_js(window, "document.querySelector('.damage-modal-card').getBoundingClientRect().width")
    assert modal_width >= 650, modal_width

    image_id = run_js(window, 'window.__shibui.state.image.id')
    made = json.loads(window.bridge.addManualDefect(image_id, json.dumps({
        'bbox': [100, 120, 360, 420], 'type': 'manual', 'size': 'manual'})))
    defect = made['defect']
    assert defect['proposal_id']
    proposal = window.store.get('proposal', defect['proposal_id'])
    assert proposal['region_class'] == 'fillable_lacuna'
    assert proposal['repair_allowed'] is True
    assert (window.store.folder(defect['proposal_id']) / 'mask.png').is_file()

    script = ("""(() => { window.__shibui.state.damageItems = [""" + json.dumps(defect) +
              """]; window.__shibui.renderDamage(); """
              """document.querySelector('[data-damage="0"]').click(); return true; })()""")
    assert run_js(window, script) is True
    assert wait_until(qt_app, lambda: run_js(window, "document.querySelector('#damage-highlight').classList.contains('visible')") is True, 10)
    assert run_js(window, "document.querySelector('#damage-modal').classList.contains('hidden')") is True

    # Draw a second defect directly through the DOM pointer handlers.
    assert run_js(window, "document.querySelector('[data-action=\"open-damage-list\"]').click(); true") is True
    assert run_js(window, "document.querySelector('[data-action=\"add-manual-defect\"]').click(); true") is True
    drag = """(() => {
      const c = document.querySelector('#mask-canvas');
      const r = c.getBoundingClientRect();
      const fire = (type, fx, fy, buttons) => c.dispatchEvent(new PointerEvent(type, {
        bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse',
        clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
        button: 0, buttons }));
      fire('pointerdown', 0.55, 0.55, 1);
      fire('pointermove', 0.75, 0.75, 1);
      fire('pointerup', 0.75, 0.75, 0);
      return true;
    })()"""
    assert run_js(window, drag) is True
    assert wait_until(qt_app, lambda: run_js(window, 'window.__shibui.state.damageItems.length') == 2, 10)
    assert run_js(window, "!document.querySelector('#damage-modal').classList.contains('hidden')") is True
    assert len(window.bridge.store.list('proposal')) == 2

    assert run_js(window, "document.querySelector('[data-action=\"close-damage-list\"]').click(); true") is True
    assert run_js(window, "document.querySelector('[data-action=\"open-triple-view\"]').click(); true") is True
    assert wait_until(qt_app, lambda: run_js(window, "!document.querySelector('#triple-modal').classList.contains('hidden')") is True, 10)
    assert run_js(window, "document.querySelectorAll('.triple-panel').length") == 3
    assert wait_until(qt_app, lambda: run_js(window, "document.querySelector('#triple-original').naturalWidth > 0") is True, 10)
    # The isolated fixture has no diffusion weights; generation buttons must be
    # disabled instead of failing only after a click.
    assert run_js(window, "document.querySelector('[data-action=\"generate-large-panel\"]').disabled") is True
    assert run_js(window, "document.querySelector('[data-action=\"generate-full-panel\"]').disabled") is True
    assert run_js(window, "document.querySelector('[data-action=\"close-triple-view\"]').click(); true") is True


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


def test_swipe_comparison_keeps_both_images_on_the_same_rect(window, qt_app):
    image_id, candidate_id = new_id(), new_id()
    Image.new('RGB', (960, 640), '#c3c3c3').save(window.store.folder(image_id) / 'image.png')
    Image.new('RGB', (960, 640), '#7a7a7a').save(window.store.folder(candidate_id) / 'result.png')
    source_url = window.bridge._file_url(window.store.folder(image_id) / 'image.png')
    result_url = window.bridge._file_url(window.store.folder(candidate_id) / 'result.png')
    script = ("(() => {"
              " document.querySelector('.view-tab[data-view=\"results\"]').click();"
              " const state = window.__shibui.state;"
              f" state.image = {{id: '{image_id}', width: 960, height: 640, file_url: '{source_url}'}};"
              f" state.candidate = {{file_url: '{result_url}', difference_url: '{result_url}', metrics: {{}}}};"
              " document.querySelector('#comparison-mode').value = 'compare';"
              " window.__shibui.renderComparison();"
              " return true; })()")
    assert run_js(window, script) is True
    assert wait_until(qt_app, lambda: run_js(window, "(() => { const imgs = [...document.querySelectorAll('.comparison-frame img')]; return imgs.length === 2 && imgs.every(img => img.naturalWidth > 0); })()") is True, 20)
    assert wait_until(qt_app, lambda: run_js(window, "(() => { const frame = document.querySelector('.comparison-frame'); return frame && frame.getBoundingClientRect().width > 0; })()") is True, 10)
    run_js(window, 'window.__shibui.layoutComparison(); true')
    rects = json.loads(run_js(window, """JSON.stringify([...document.querySelectorAll('.comparison-frame img')]
      .map(img => { const r = img.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }))"""))
    assert len(rects) == 2, rects
    for left, right in zip(rects[0], rects[1]):
        assert abs(left - right) < 1.5, rects
    frame = json.loads(run_js(window, """JSON.stringify((() => {
      const r = document.querySelector('.comparison-frame').getBoundingClientRect(); return [r.width, r.height]; })())"""))
    assert abs(frame[0] / frame[1] - 960 / 640) < 0.02, frame


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


def test_ordinary_speculative_mode_shows_required_confirmations(window, qt_app):
    """P2-6: ordinary speculative mode must show risk and mask confirmation.

    Previously the risk checkbox was hidden unless enhanced mode or the
    experimental generative switch was on, so the user could not supply the
    acknowledgement execution requires.
    """
    script = """(() => {
      const probe = (professional, mode) => {
        const step = document.querySelector('#professional-step');
        step.checked = professional;
        step.dispatchEvent(new Event('change'));
        const select = document.querySelector('#repair-mode');
        select.value = mode;
        select.dispatchEvent(new Event('change'));
        const shown = (id) => document.querySelector(id).closest('label').style.display !== 'none';
        return { risk: shown('#risk-ack'), gate: shown('#mask-gate-confirmed'),
                 enhanced: shown('#allow-enhanced') };
      };
      return JSON.stringify({
        professionalSpeculative: probe(true, 'speculative'),
        standardSpeculative: probe(false, 'speculative'),
        professionalConservative: probe(true, 'conservative'),
        standardConservative: probe(false, 'conservative'),
        professionalEnhanced: probe(true, 'enhanced'),
      });
    })()"""
    shown = json.loads(run_js(window, script))
    assert shown['professionalSpeculative'] == {'risk': True, 'gate': True, 'enhanced': False}
    assert shown['standardSpeculative'] == {'risk': True, 'gate': True, 'enhanced': False}
    assert shown['professionalConservative'] == {'risk': False, 'gate': True, 'enhanced': False}
    assert shown['standardConservative'] == {'risk': False, 'gate': False, 'enhanced': False}
    assert shown['professionalEnhanced'] == {'risk': True, 'gate': True, 'enhanced': True}


def load_plain_fixture(window, qt_app):
    """Use a flat test image; these regressions never invoke a generation model."""
    from mural_agent import workspace
    record = workspace.add_image(window.store, png(np.full((240, 320, 3), 160, np.uint8)), 'ui-fixture.png')
    exposed = window.bridge._image(record)
    assert run_js(window, f'window.__shibui.loadImage({json.dumps(exposed)}); true') is True
    assert wait_until(qt_app, lambda: run_js(window, "document.querySelector('#mask-canvas').width") == 320)
    assert wait_until(qt_app, lambda: run_js(window, "document.querySelector('#mask-canvas').getBoundingClientRect().width > 0") is True)
    return record


def pointer_stroke(window, tool, start=(0.4, 0.4), end=None, size=20):
    # Exercise the shipped pointer handlers, including pointerup/replay and undo.
    script = """(() => {
      const tool = TOOL;
      const button = document.querySelector(`[data-tool="${tool}"]`);
      if (button) button.click(); else window.__shibui.state.tool = tool;
      const c = document.querySelector('#mask-canvas'), r = c.getBoundingClientRect();
      window.__shibui.state.brushSize = SIZE * r.width / c.width;
      const fire = (type, point, buttons) => c.dispatchEvent(new PointerEvent(type, {
        bubbles: true, pointerId: 1, pointerType: 'mouse', button: 0, buttons,
        clientX: r.left + r.width * point[0], clientY: r.top + r.height * point[1] }));
      fire('pointerdown', START, 1);
      if (END) fire('pointermove', END, 1);
      fire('pointerup', END || START, 0);
      return true;
    })()"""
    script = (script.replace('TOOL', json.dumps(tool)).replace('SIZE', str(size))
              .replace('START', json.dumps(start)).replace('END', json.dumps(end)))
    assert run_js(window, script) is True


def risk_pixel(window, x=128, y=96):
    return json.loads(run_js(window, f"JSON.stringify([...document.querySelector('#region-canvas').getContext('2d').getImageData({x}, {y}, 1, 1).data])"))


@pytest.mark.parametrize('tool', ['uncertain', 'uncertainty'])
def test_uncertain_strokes_use_the_uncertain_gate_even_when_overdrawn(window, qt_app, tool):
    load_plain_fixture(window, qt_app)
    pointer_stroke(window, 'brush', start=(0.65, 0.65))
    for _ in range(3):
        pointer_stroke(window, tool)
    assert risk_pixel(window)[3] > 0
    assert run_js(window, "document.querySelector('#force-override').closest('label').style.display !== 'none'") is True
    assert run_js(window, "document.querySelector('[data-action=\"run-repair\"]').disabled") is False
    # Without expert override it must stop at the uncertain gate, not at the
    # unrestorable gate. With override it must transmit the correct categories.
    run_js(window, "document.querySelector('#mask-gate-confirmed').checked = true; document.querySelector('[data-action=\"run-repair\"]').click(); true")
    assert '专家强制放行' in run_js(window, "document.querySelector('#toast').textContent")
    run_js(window, """window.__shibui.state.bridge.createJob = (request, callback) => {
      window.__capturedRequest = JSON.parse(request); callback('{"error":"fixture: no execution"}');
    };
    document.querySelector('#force-override').checked = true;
    document.querySelector('#force-override-reason').value = 'Regression fixture expert review';
    document.querySelector('[data-action="run-repair"]').click(); true""")
    assert wait_until(qt_app, lambda: run_js(window, 'Boolean(window.__capturedRequest)') is True)
    request = json.loads(run_js(window, 'JSON.stringify(window.__capturedRequest)'))
    assert request['has_uncertain'] is True
    assert request['has_unrestorable'] is False
    assert window.store.list('job') == []


def test_unrestorable_strokes_still_block_repair_after_refresh(window, qt_app):
    load_plain_fixture(window, qt_app)
    pointer_stroke(window, 'brush')
    pointer_stroke(window, 'unrestorable')
    run_js(window, "window.__shibui.refresh(); true")
    assert run_js(window, "document.querySelector('[data-action=\"run-repair\"]').disabled") is True
    assert run_js(window, "document.querySelector('#force-override').closest('label').style.display") == 'none'


def test_damage_list_has_no_ineffective_confirmation_checkbox(window, qt_app):
    load_plain_fixture(window, qt_app)
    run_js(window, """window.__shibui.state.damageItems = [
      {bbox: [100, 100, 200, 200], type: 'crack', confirmed: false},
      {bbox: [300, 300, 400, 400], type: 'paint_loss', confirmed: true}
    ]; window.__shibui.renderDamage(); true""")
    assert run_js(window, "document.querySelectorAll('[data-confirm-defect], .damage-confirm').length") == 0
    assert run_js(window, "document.querySelectorAll('[data-run-defect]:not(:disabled)').length") == 2
    assert run_js(window, "document.querySelector('[data-action=\"run-all-defects\"]').disabled") is False


@pytest.mark.parametrize('tool', ['uncertain', 'unrestorable'])
def test_eraser_removes_risk_and_mask_pixels_with_ordered_undo(window, qt_app, tool):
    load_plain_fixture(window, qt_app)
    pointer_stroke(window, 'brush', start=(0.3, 0.4), end=(0.5, 0.4))
    pointer_stroke(window, tool, start=(0.3, 0.4), end=(0.5, 0.4))
    assert risk_pixel(window)[3] > 0
    pointer_stroke(window, 'eraser', start=(0.25, 0.4), end=(0.55, 0.4), size=50)
    assert risk_pixel(window)[3] == 0
    mask = base64.b64decode(run_js(window, 'window.__shibui.exportMaskBase64()'))
    assert np.asarray(Image.open(BytesIO(mask)).convert('L')).max() == 0
    assert run_js(window, "document.querySelector('#force-override').closest('label').style.display") == 'none'

    # Later unrelated strokes/redraws must not resurrect erased red regions.
    pointer_stroke(window, 'brush', start=(0.8, 0.8))
    assert risk_pixel(window)[3] == 0
    run_js(window, "document.querySelector('[data-action=\"undo\"]').click(); true")
    assert risk_pixel(window)[3] == 0
    run_js(window, "document.querySelector('[data-action=\"undo\"]').click(); true")
    assert risk_pixel(window)[3] > 0
    mask = base64.b64decode(run_js(window, 'window.__shibui.exportMaskBase64()'))
    assert np.asarray(Image.open(BytesIO(mask)).convert('L'))[96, 128] == 255
    # A new stroke after the eraser/undo sequence is classified by its own type.
    pointer_stroke(window, 'uncertain', start=(0.3, 0.4), end=(0.5, 0.4), size=40)
    assert run_js(window, "document.querySelector('#force-override').closest('label').style.display !== 'none'") is True
    assert run_js(window, "document.querySelector('[data-action=\"run-repair\"]').disabled") is False


@pytest.mark.parametrize('count', [1, 2])
def test_defect_launch_selects_task_and_exposes_batch_results(window, qt_app, monkeypatch, count):
    image = load_plain_fixture(window, qt_app)
    # Hold all tasks queued: this tests UI state/cancellation without repairing
    # or generating a mural, and keeps the observation independent of timing.
    submitted = []
    monkeypatch.setattr(window.engine, 'submit', lambda identifier: submitted.append(identifier))
    defects = []
    for i in range(count):
        made = json.loads(window.bridge.addManualDefect(image['id'], json.dumps({
            'bbox': [100 + i*250, 100, 200 + i*250, 200], 'type': 'manual', 'size': 'small'})))
        assert 'error' not in made, made
        defects.append(made['defect'])
    run_js(window, f"window.__shibui.state.damageItems = {json.dumps(defects)}; window.__shibui.renderDamage(); document.querySelector('[data-action=\"open-damage-list\"]').click(); true")
    assert run_js(window, "document.querySelectorAll('[data-confirm-defect], .damage-confirm').length") == 0
    action = '[data-run-defect="0"]' if count == 1 else '[data-action="run-all-defects"]'
    assert run_js(window, f'document.querySelector({json.dumps(action)}).click(); true') is True
    assert wait_until(qt_app, lambda: len(submitted) == count)
    assert wait_until(qt_app, lambda: run_js(window, 'Boolean(window.__shibui.state.job)') is True)
    first = submitted[0]
    assert run_js(window, 'window.__shibui.state.job.id') == first
    assert run_js(window, "document.querySelector('#results-view').classList.contains('active')") is True
    assert run_js(window, "document.querySelector('#job-status').textContent") == '排队中'
    assert run_js(window, "document.querySelector('[data-action=\"cancel-job\"]').getBoundingClientRect().height > 0") is True
    assert run_js(window, "document.querySelectorAll('[data-batch-job]').length") == count
    assert run_js(window, "document.querySelectorAll('[data-batch-job]:disabled').length") == 0
    assert run_js(window, "document.querySelector('#damage-modal').classList.contains('hidden')") is True

    # Inject a finished fixture candidate: polling must display it automatically.
    candidate = {'id': new_id(), 'job_id': first, 'tool': 'telea', 'metrics': {}, 'versions': {'device': 'fixture'}}
    for name in ('result.png', 'difference.png'):
        Image.new('RGB', (320, 240), 'gray').save(window.store.folder(candidate['id']) / name)
    window.store.put('candidate', candidate)
    window.store.update('job', first, status='review', progress=100, candidates=[candidate])
    run_js(window, 'window.__shibui.refresh(); true')
    assert wait_until(qt_app, lambda: run_js(window, 'window.__shibui.state.candidate?.id || null') == candidate['id'])
    assert run_js(window, "document.querySelectorAll('#comparison-stage img').length") >= 1
    assert run_js(window, "document.querySelector('#job-status').textContent") == '等待复核'
    if count == 1:
        # Re-running from an already displayed result must select the new task,
        # clearing the previous candidate rather than leaving it on screen.
        run_js(window, "document.querySelector('[data-action=\"open-damage-list\"]').click(); document.querySelector('[data-run-defect=\"0\"]').click(); true")
        assert wait_until(qt_app, lambda: len(submitted) == 2)
        assert wait_until(qt_app, lambda: run_js(window, 'window.__shibui.state.job.id') == submitted[1])
        assert run_js(window, 'window.__shibui.state.candidate === null') is True
        assert run_js(window, "document.querySelector('#job-status').textContent") == '排队中'
    if count == 2:
        second = submitted[1]
        run_js(window, f'document.querySelector(\'[data-batch-job="{second}"]\').click(); true')
        assert wait_until(qt_app, lambda: run_js(window, 'window.__shibui.state.job.id') == second)
        assert wait_until(qt_app, lambda: run_js(window, 'window.__shibui.state.activeJobId') == second)
        assert run_js(window, 'window.__shibui.state.candidate === null') is True
        assert run_js(window, "document.querySelector('#job-status').textContent") == '排队中'
        run_js(window, "document.querySelector('[data-action=\"cancel-job\"]').click(); true")
        assert wait_until(qt_app, lambda: window.store.get('job', second)['cancel_requested'])
        assert window.store.get('job', first)['cancel_requested'] is False
        window.store.update('job', second, status='cancelled')
        run_js(window, 'window.__shibui.refresh(); true')
        assert wait_until(qt_app, lambda: run_js(window, "document.querySelector('#batch-summary').textContent") == '本批次 2 项 · 2 项已结束')
        run_js(window, f'document.querySelector(\'[data-batch-job="{first}"]\').click(); true')
        assert wait_until(qt_app, lambda: run_js(window, 'window.__shibui.state.candidate?.id || null') == candidate['id'])
