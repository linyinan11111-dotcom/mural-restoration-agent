"""Regressions for the 2026-11-11 audit items P2-6, P2-7, P2-10, and P2-11.

Each test fails on the pre-fix behavior: confirmation flags dropped before
execution, enhanced archives missing the work mask, cloud-only full generation
blocked on a local diffusion install, and ControlNet holes filled with white.
"""
from io import BytesIO
import json
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from mural_agent import workspace
from mural_agent.config import Settings
from mural_agent.desktop_web import select_full_generation_backend
from mural_agent.engine import Engine
from mural_agent.model_worker import make_inpaint_condition
from mural_agent.professional import validate_job_record_for_execution, validate_professional_create
from mural_agent.storage import Store


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def bench(tmp_path):
    settings = Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt',
                        diffusion_model=tmp_path / 'missing-sd')
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    yield store, engine
    engine.executor.shutdown(wait=True, cancel_futures=True)


def sample(store):
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(80, 180, 96, dtype=np.uint8)[None, :]
    image[..., 1] = 140
    image[..., 2] = 90
    image[20:40, 40:48] = (250, 250, 250)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[20:40, 40:48] = 255
    record = workspace.add_image(store, png(image), 'case.png')
    saved = workspace.add_mask(store, record, png(mask))
    return record, saved, image, mask


def wait(store, job):
    for _ in range(400):
        current = store.get('job', job['id'])
        if current['status'] not in {'queued', 'planning', 'running', 'evaluating'}:
            return current
        time.sleep(0.02)
    pytest.fail('job did not finish')


def patch_diffusion(monkeypatch, engine):
    original = engine.tools.availability

    def fake():
        data = dict(original())
        data['diffusion'] = True
        return data

    monkeypatch.setattr(engine.tools, 'availability', fake)
    monkeypatch.setattr(engine, 'submit', lambda job_id: None)


def test_ordinary_speculative_acknowledgement_is_kept(bench, monkeypatch):
    """P2-6: professional speculative without the experimental switch used to
    store risk_acknowledged=False and then fail in validate_job_record_for_execution.
    """
    store, engine = bench
    image, mask, *_ = sample(store)
    patch_diffusion(monkeypatch, engine)
    meta = validate_professional_create(
        mask, rationale='可见大片缺失，只作推测对照', evidence_level='speculative',
        mode='speculative', mask_gate_confirmed=True, risk_acknowledged=True)
    assert meta['risk_acknowledged'] is True
    with pytest.raises(ValueError, match='风险提示'):
        validate_professional_create(
            mask, rationale='可见大片缺失，只作推测对照', evidence_level='speculative',
            mode='speculative', mask_gate_confirmed=True, risk_acknowledged=False)
    job = workspace.create_job(
        store, engine, image, mask, 'diffusion', 'rules', 1, '推测补全说明',
        mode='speculative', workflow='professional_step',
        rationale='可见大片缺失，只作推测对照', evidence_level='speculative',
        risk_acknowledged=True, mask_gate_confirmed=True)
    assert job['risk_acknowledged'] is True
    assert job['mask_gate_confirmed'] is True
    validate_job_record_for_execution(job, mask)


def test_standard_speculative_and_enhanced_flags_match_runtime(bench, monkeypatch):
    """P2-6: standard workflow used to drop the confirmations it had just accepted."""
    store, engine = bench
    image, mask, *_ = sample(store)
    patch_diffusion(monkeypatch, engine)
    with pytest.raises(ValueError, match='风险提示'):
        workspace.create_job(
            store, engine, image, mask, 'diffusion', 'rules', 1, '推测',
            mode='speculative', workflow='standard',
            risk_acknowledged=False, mask_gate_confirmed=True)
    speculative = workspace.create_job(
        store, engine, image, mask, 'diffusion', 'rules', 1, '推测补全说明',
        mode='speculative', workflow='standard',
        risk_acknowledged=True, mask_gate_confirmed=True)
    assert speculative['risk_acknowledged'] is True
    assert speculative['mask_gate_confirmed'] is True
    assert speculative['allow_enhanced'] is False
    validate_job_record_for_execution(speculative, mask)

    enhanced = workspace.create_job(
        store, engine, image, mask, 'opencv', 'rules', 1, '',
        mode='enhanced', workflow='standard',
        allow_enhanced=True, risk_acknowledged=True, mask_gate_confirmed=True,
        budget={'candidates': 1, 'revisions': 0, 'timeout_seconds': 60})
    assert enhanced['allow_enhanced'] is True
    assert enhanced['risk_acknowledged'] is True
    assert enhanced['mask_gate_confirmed'] is True
    validate_job_record_for_execution(enhanced, mask)


def test_enhanced_archive_includes_effective_work_mask(bench):
    """P2-7: QA against the archived original mask used to see outside edits
    because the expanded work mask was never exported.
    """
    store, engine = bench
    image, mask_record, original, user_mask = sample(store)
    job = workspace.create_job(
        store, engine, image, mask_record, 'opencv', 'rules', 1, '',
        mode='enhanced', workflow='standard',
        allow_enhanced=True, risk_acknowledged=True, mask_gate_confirmed=True,
        budget={'candidates': 1, 'revisions': 0, 'timeout_seconds': 60})
    finished = wait(store, job)
    assert finished['status'] == 'review', finished.get('error')
    metrics = finished['candidates'][0]['metrics']
    assert metrics['outside_unchanged'] is True
    assert metrics['evaluated_mask'] == 'enhanced_work_mask'
    assert metrics['evaluated_mask_file'] == 'enhanced_work_mask.png'
    assert 'enhanced_work_mask' in metrics['evaluated_mask_note']
    store.update('job', finished['id'], selected_id=finished['candidates'][0]['id'])
    with zipfile.ZipFile(engine.export(finished['id'])) as archive:
        names = set(archive.namelist())
        assert 'mask.png' in names
        assert 'enhanced_work_mask.png' in names
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['mask_files'] == {
            'original': 'mask.png',
            'effective': 'enhanced_work_mask.png',
            'metrics_correspond_to': 'enhanced_work_mask',
        }
        saved_mask = np.asarray(Image.open(BytesIO(archive.read('mask.png')))) > 127
        work = np.asarray(Image.open(BytesIO(archive.read('enhanced_work_mask.png')))) > 127
        result = np.asarray(Image.open(BytesIO(archive.read('result.png'))))
        source = np.asarray(Image.open(BytesIO(archive.read('original/working.png'))))
    assert np.array_equal(saved_mask, user_mask > 0)
    assert work.sum() > saved_mask.sum()
    assert work[saved_mask].all()
    changed = np.any(result != source, axis=2)
    assert not (changed & ~work).any()
    assert np.array_equal(source, original)


def test_conservative_metrics_name_the_original_mask(bench):
    store, engine = bench
    image, mask_record, *_ = sample(store)
    job = workspace.create_job(
        store, engine, image, mask_record, 'opencv', 'rules', 1, '',
        budget={'candidates': 1, 'revisions': 0, 'timeout_seconds': 60})
    finished = wait(store, job)
    assert finished['status'] == 'review', finished.get('error')
    metrics = finished['candidates'][0]['metrics']
    assert metrics['evaluated_mask'] == 'original_mask'
    assert metrics['evaluated_mask_file'] == 'mask.png'
    assert 'mask.png' in metrics['evaluated_mask_note']
    store.update('job', finished['id'], selected_id=finished['candidates'][0]['id'])
    with zipfile.ZipFile(engine.export(finished['id'])) as archive:
        assert 'enhanced_work_mask.png' not in archive.namelist()
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['mask_files']['effective'] == 'mask.png'
        assert manifest['mask_files']['metrics_correspond_to'] == 'original_mask'


def test_full_generation_backend_is_selected_before_dependency_checks(tmp_path, monkeypatch):
    """P2-10: imagine available and local diffusion missing must be allowed to start."""
    assert select_full_generation_backend({'imagine': True, 'diffusion': False}) == 'imagine'
    assert select_full_generation_backend({'imagine': False, 'diffusion': True}) == 'diffusion'
    with pytest.raises(ValueError, match='云端'):
        select_full_generation_backend({'imagine': False, 'diffusion': False})

    pytest.importorskip('PySide6.QtWidgets')
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from mural_agent.desktop_web import Bridge

    def start(imagine, diffusion):
        settings = Settings(
            data_dir=tmp_path / f'data-{imagine}-{diffusion}',
            lama_model=tmp_path / 'missing.pt',
            diffusion_model=tmp_path / 'missing-sd',
            imagine_api_key='test-token' if imagine else '',
            imagine_base_url='https://example.invalid/v1',
            imagine_model='grok-imagine-image-2.0')
        store = Store(settings.data_dir)
        engine = Engine(settings, store)
        monkeypatch.setattr(engine, 'submit', lambda job_id: None)
        if diffusion and not imagine:
            # diffusers is not installed in this environment; pretend the local
            # backend is present so the selection itself can be observed.
            monkeypatch.setattr(engine.tools, 'availability', lambda: {
                'opencv': True, 'telea': True, 'ns': True, 'lama': False, 'reference': True,
                'diffusion': True, 'imagine': False, 'denoise': True, 'color': True,
                'deblur': True, 'research': False, 'sam2': False, 'controlnet': False})
        bridge = Bridge(settings, store, engine)
        try:
            demo = json.loads(bridge.loadDemo())
            return json.loads(bridge.generateFullImage(demo['id']))
        finally:
            engine.executor.shutdown(wait=False, cancel_futures=True)

    cloud = start(True, False)
    assert cloud.get('backend') == 'imagine', cloud
    assert not cloud.get('error')
    assert cloud['mode'] == 'speculative'

    local = start(False, True)
    assert local.get('backend') == 'diffusion', local
    assert not local.get('error')

    blocked = start(False, False)
    assert isinstance(blocked.get('error'), str)
    assert '云端' in blocked['error']
    assert '扩散' in blocked['error']


def test_controlnet_inpaint_condition_uses_negative_one_protocol():
    """P2-11: masked pixels must be -1, not the old white fill that became [1, 1, 1]."""
    image = Image.new('RGB', (8, 6), (4, 5, 6))
    image.putpixel((1, 2), (10, 20, 30))
    image.putpixel((3, 1), (255, 128, 0))
    mask = Image.new('L', (8, 6), 0)
    mask.putpixel((3, 1), 255)
    mask.putpixel((4, 1), 200)
    condition = make_inpaint_condition(image, mask)
    assert condition.shape == (1, 3, 6, 8)
    assert condition.dtype == np.float32
    assert condition.flags['C_CONTIGUOUS']
    np.testing.assert_allclose(condition[0, :, 2, 1], np.array([10, 20, 30], np.float32) / 255.0)
    assert np.all(condition[0, :, 1, 3] == -1)
    assert np.all(condition[0, :, 1, 4] == -1)
    # The previous protocol painted the hole white and handed diffusers a PIL
    # image, which normalizes those pixels to [1, 1, 1].
    assert not np.allclose(condition[0, :, 1, 3], 1.0)
    unmasked = np.ones(condition.shape[2:], dtype=bool)
    unmasked[1, 3] = False
    unmasked[1, 4] = False
    assert condition[0].min(axis=0)[unmasked].min() >= 0
    assert condition[0].max(axis=0)[unmasked].max() <= 1
    with pytest.raises(ValueError, match='same image size'):
        make_inpaint_condition(image, Image.new('L', (4, 4), 0))
