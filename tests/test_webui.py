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


def load_demo(window, qt_app):
    assert run_js(window, "document.querySelector('[data-action=\"load-demo\"]').click(); true") is True
    ready = ("Boolean(window.__shibui.state.image && document.querySelector('#mask-canvas').width > 0"
             " && document.querySelector('#mask-canvas').getBoundingClientRect().width > 20)")
    assert wait_until(qt_app, lambda: run_js(window, ready) is True, 30), 'demo image did not lay out the mask canvas'


def paint_region_tool(window, tool, brush=36):
    script = f"""(() => {{
      const canvas = document.querySelector('#mask-canvas');
      const rect = canvas.getBoundingClientRect();
      const x = rect.left + rect.width / 2;
      const y = rect.top + rect.height / 2;
      document.querySelector('[data-tool="{tool}"]').click();
      const slider = document.querySelector('#brush-size');
      slider.value = '{brush}';
      slider.dispatchEvent(new Event('input', {{bubbles: true}}));
      const fire = (type, buttons) => canvas.dispatchEvent(new PointerEvent(type, {{
        bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse',
        clientX: x, clientY: y, button: 0, buttons}}));
      fire('pointerdown', 1);
      fire('pointerup', 0);
      const region = document.querySelector('#region-canvas');
      const ctx = region.getContext('2d');
      const px = Math.floor(region.width / 2), py = Math.floor(region.height / 2);
      const pixel = Array.from(ctx.getImageData(px, py, 1, 1).data);
      document.querySelector('.advanced-settings').open = true;
      const force = document.querySelector('#force-override').closest('label');
      const banner = document.querySelector('#risk-banner');
      return JSON.stringify({{
        pixel, stats: window.__shibui.regionPixelStats(),
        forceDisplay: force.style.display,
        forceVisible: force.getClientRects().length > 0,
        banner: banner.textContent,
        bannerHidden: banner.classList.contains('hidden'),
      }});
    }})()"""
    return json.loads(run_js(window, script))


def test_uncertain_stroke_is_not_classified_as_unrestorable(window, qt_app):
    """P2-8: rgba(220,0,60) used to match the unrestorable red test first."""
    load_demo(window, qt_app)
    uncertain = paint_region_tool(window, 'uncertain')
    pixel = uncertain['pixel']
    assert pixel[0] >= 200 and pixel[1] < 40 and pixel[2] >= 40 and pixel[3] > 100, pixel
    assert uncertain['stats']['uncertain'] is True
    assert uncertain['stats']['unrestorable'] is False
    assert uncertain['forceDisplay'] != 'none'
    assert uncertain['forceVisible'] is True
    assert uncertain['bannerHidden'] is False
    assert '不确定' in uncertain['banner']
    assert '不可复原' not in uncertain['banner']

    assert run_js(window, "document.querySelector('[data-action=\"clear-mask\"]').click(); true") is True
    blocked = paint_region_tool(window, 'unrestorable')
    assert blocked['stats']['unrestorable'] is True
    assert blocked['stats']['uncertain'] is False, blocked
    assert blocked['forceDisplay'] == 'none'
    assert blocked['forceVisible'] is False
    assert '不可复原' in blocked['banner']


def test_eraser_removes_risk_marks_and_ordinary_mask(window, qt_app):
    """P2-9: redraw used to repaint every risk stroke and ignore eraser history."""
    load_demo(window, qt_app)
    before = paint_region_tool(window, 'uncertain', brush=28)
    assert before['pixel'][3] > 100
    assert before['stats']['uncertain'] is True
    script = """(() => {
      const canvas = document.querySelector('#mask-canvas');
      const rect = canvas.getBoundingClientRect();
      const x = rect.left + rect.width / 2;
      const y = rect.top + rect.height / 2;
      document.querySelector('[data-tool="eraser"]').click();
      const slider = document.querySelector('#brush-size');
      slider.value = '80';
      slider.dispatchEvent(new Event('input', {bubbles: true}));
      const fire = (type, buttons) => canvas.dispatchEvent(new PointerEvent(type, {
        bubbles: true, cancelable: true, pointerId: 7, pointerType: 'mouse',
        clientX: x, clientY: y, button: 0, buttons}));
      fire('pointerdown', 1);
      fire('pointerup', 0);
      const region = document.querySelector('#region-canvas');
      const pixel = Array.from(region.getContext('2d').getImageData(
        Math.floor(region.width / 2), Math.floor(region.height / 2), 1, 1).data);
      const banner = document.querySelector('#risk-banner');
      return JSON.stringify({pixel, stats: window.__shibui.regionPixelStats(),
        bannerHidden: banner.classList.contains('hidden')});
    })()"""
    erased = json.loads(run_js(window, script))
    assert erased['pixel'][3] < 40, erased['pixel']
    assert erased['stats'] == {'uncertain': False, 'unrestorable': False}
    assert erased['bannerHidden'] is True

    mask = json.loads(run_js(window, """(() => {
      const canvas = document.querySelector('#mask-canvas');
      const rect = canvas.getBoundingClientRect();
      const x = rect.left + rect.width * 0.25;
      const y = rect.top + rect.height * 0.25;
      document.querySelector('[data-tool="brush"]').click();
      const slider = document.querySelector('#brush-size');
      slider.value = '24';
      slider.dispatchEvent(new Event('input', {bubbles: true}));
      const fire = (type, px, py, buttons) => canvas.dispatchEvent(new PointerEvent(type, {
        bubbles: true, cancelable: true, pointerId: 8, pointerType: 'mouse',
        clientX: px, clientY: py, button: 0, buttons}));
      fire('pointerdown', x, y, 1);
      fire('pointerup', x, y, 0);
      const point = [(x - rect.left) * canvas.width / rect.width, (y - rect.top) * canvas.height / rect.height];
      const before = canvas.getContext('2d').getImageData(Math.floor(point[0]), Math.floor(point[1]), 1, 1).data[3];
      document.querySelector('[data-tool="eraser"]').click();
      slider.value = '80';
      slider.dispatchEvent(new Event('input', {bubbles: true}));
      fire('pointerdown', x, y, 1);
      fire('pointerup', x, y, 0);
      const after = canvas.getContext('2d').getImageData(Math.floor(point[0]), Math.floor(point[1]), 1, 1).data[3];
      return JSON.stringify({before, after});
    })()"""))
    assert mask['before'] > 100, mask
    assert mask['after'] < 40, mask


def test_defect_include_filter_limits_batch_but_not_single_run(window, qt_app):
    """P2-9: the old 确认 checkbox did not change which defect jobs were created."""
    load_demo(window, qt_app)
    image_id = run_js(window, 'window.__shibui.state.image.id')
    defects = []
    for bbox in ([120, 140, 280, 320], [420, 180, 640, 400]):
        made = json.loads(window.bridge.addManualDefect(image_id, json.dumps({
            'bbox': bbox, 'type': 'manual', 'size': 'manual'})))
        defects.append(made['defect'])
    script = """(() => {
      window.__shibui.state.damageItems = %s;
      window.__shibui.renderDamage();
      const boxes = document.querySelectorAll('[data-include-defect]');
      if (boxes.length !== 2 || boxes[0].closest('label').textContent.trim() !== '纳入批次') return 'markup';
      boxes[0].click();
      const runAll = document.querySelector('[data-action="run-all-defects"]');
      if (runAll.disabled) return 'disabled';
      runAll.click();
      return 'started';
    })()""" % json.dumps(defects)
    assert run_js(window, script) == 'started'

    def batch_jobs():
        return [job for job in json.loads(window.bridge.snapshot())['jobs']
                if job.get('defect_proposal_id')]

    def batch_settled():
        jobs = batch_jobs()
        idle = run_js(window, "!window.__shibui.state.running && document.querySelector('[data-run-defect=\"0\"]') && !document.querySelector('[data-run-defect=\"0\"]').disabled")
        return len(jobs) == 1 and jobs[0]['status'] not in ACTIVE and idle is True

    assert wait_until(qt_app, batch_settled, 40), batch_jobs()
    assert batch_jobs()[0]['defect_proposal_id'] == defects[1]['proposal_id']
    assert run_js(window, 'window.__shibui.state.job && window.__shibui.state.job.id') == batch_jobs()[0]['id']
    assert run_js(window, "document.querySelector('#job-status').textContent") != '尚未开始'

    assert run_js(window, "document.querySelector('[data-run-defect=\"0\"]').click(); 'clicked'") == 'clicked'
    assert wait_until(qt_app, lambda: len(batch_jobs()) == 2, 30), batch_jobs()
    proposals = {job['defect_proposal_id'] for job in batch_jobs()}
    assert proposals == {defects[0]['proposal_id'], defects[1]['proposal_id']}
    assert run_js(window, 'window.__shibui.state.job.defect_proposal_id') == defects[0]['proposal_id']


def test_defect_list_start_binds_the_new_jobs(window, qt_app, monkeypatch):
    """P1-3: creating defect jobs must select them instead of leaving 尚未开始."""
    import threading
    release = threading.Event()
    original = window.engine.run

    def blocking(job_id):
        assert release.wait(30)
        return original(job_id)

    monkeypatch.setattr(window.engine, 'run', blocking)
    try:
        load_demo(window, qt_app)
        image_id = run_js(window, 'window.__shibui.state.image.id')
        defects = []
        for bbox in ([100, 120, 260, 300], [400, 200, 620, 420]):
            made = json.loads(window.bridge.addManualDefect(image_id, json.dumps({
                'bbox': bbox, 'type': 'crack', 'size': 'small'})))
            defects.append(made['defect'])
        assert run_js(window, """(() => {
          window.__shibui.state.damageItems = %s;
          window.__shibui.renderDamage();
          const button = document.querySelector('[data-action="run-all-defects"]');
          if (button.disabled) return 'disabled';
          button.click();
          return 'started';
        })()""" % json.dumps(defects)) == 'started'

        def bound():
            view = run_js(window, """JSON.stringify({
              jobId: window.__shibui.state.job && window.__shibui.state.job.id,
              status: document.querySelector('#job-status').textContent,
              queue: document.querySelectorAll('#job-queue .job-queue-item').length,
              queueHidden: document.querySelector('#job-queue').classList.contains('hidden'),
              results: document.querySelector('#results-view').classList.contains('active'),
              cancelHidden: document.querySelector('[data-action="cancel-job"]').classList.contains('hidden'),
            })""")
            if not view:
                return False
            state = json.loads(view)
            return (state['jobId'] and state['status'] != '尚未开始' and state['queue'] == 2
                    and state['queueHidden'] is False and state['results'] is True
                    and state['cancelHidden'] is False)

        assert wait_until(qt_app, bound, 20), run_js(window, "document.querySelector('#job-status').textContent")
        switched = json.loads(run_js(window, """(() => {
          const current = window.__shibui.state.job.id;
          const other = [...document.querySelectorAll('#job-queue .job-queue-item')]
            .find(button => button.dataset.queueJob !== current);
          other.click();
          return JSON.stringify({from: current, to: window.__shibui.state.job.id,
            status: document.querySelector('#job-status').textContent});
        })()"""))
        assert switched['to'] and switched['to'] != switched['from']
        assert switched['status'] != '尚未开始'
        release.set()

        def results_visible():
            return run_js(window, "document.querySelectorAll('#candidate-strip .candidate-card').length") >= 1

        assert wait_until(qt_app, results_visible, 40)
        shown = run_js(window, 'window.__shibui.state.job.id')
        created = {job['id'] for job in json.loads(window.bridge.snapshot())['jobs']}
        assert shown in created
    finally:
        release.set()
