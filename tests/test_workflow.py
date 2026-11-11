from io import BytesIO
import json
import time
import zipfile
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from mural_agent.app import create_app
from mural_agent.config import Settings
from mural_agent.imaging import composite, decode_image
from mural_agent.storage import Store, new_id


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def workbench(tmp_path):
    settings = Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt')
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def upload_case(client):
    x = np.arange(96, dtype=np.uint8)
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[..., 0] = x[None, :] + 100
    image[..., 1] = 140
    image[..., 2] = 75
    image[15:48, 42:46] = [245, 230, 195]
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[15:48, 42:46] = 255
    record = client.post('/api/images', files={'file': ('case.png', png(image), 'image/png')})
    assert record.status_code == 201
    record = record.json()
    saved_mask = client.post(f'/api/images/{record["id"]}/masks', files={'file': ('mask.png', png(mask), 'image/png')})
    assert saved_mask.status_code == 201
    return record, saved_mask.json(), image, mask


def wait_job(client, identifier):
    for _ in range(300):
        job = client.get(f'/api/jobs/{identifier}').json()
        if job['status'] not in {'queued', 'planning', 'running', 'evaluating'}:
            return job
        time.sleep(.02)
    pytest.fail('Background job did not finish')


def test_full_restoration_review_and_archive(workbench):
    client, app = workbench
    image, mask_record, original, mask = upload_case(client)
    response = client.post('/api/jobs', json={'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 3})
    assert response.status_code == 202
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'review', job
    assert len(job['candidates']) == 3
    assert job['plan']['source'] == 'rules'
    for candidate in job['candidates']:
        result = np.asarray(Image.open(BytesIO(client.get(candidate['url']).content)))
        assert np.array_equal(result[mask == 0], original[mask == 0])
        assert np.any(result[mask > 0] != original[mask > 0])
        assert candidate['metrics']['outside_unchanged']
    assert client.get(f'/api/jobs/{job["id"]}/export').status_code == 400
    chosen = job['candidates'][1]
    review = client.post(f'/api/jobs/{job["id"]}/review', json={'candidate_id': chosen['id'], 'note': '保留原有色调'})
    assert review.status_code == 200
    archive = client.get(f'/api/jobs/{job["id"]}/export')
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content)) as zipped:
        assert {'result.png', 'mask.png', 'difference.png', 'manifest.json', 'original/source.bin'} <= set(zipped.namelist())
        manifest = json.loads(zipped.read('manifest.json'))
        assert manifest['selected_candidate']['id'] == chosen['id']
        assert manifest['job']['review']['note'] == '保留原有色调'
        assert manifest['mask']['sha256'] == mask_record['sha256']
        assert 'api_key' not in json.dumps(manifest)


def test_input_validation_and_record_ownership(workbench):
    client, app = workbench
    image, saved_mask, original, mask = upload_case(client)
    assert client.post('/api/images', files={'file': ('bad.png', b'not an image')}).status_code == 400
    for invalid in [np.zeros_like(mask), np.full_like(mask, 255), mask[:10]]:
        response = client.post(f'/api/images/{image["id"]}/masks', files={'file': ('mask.png', png(invalid))})
        assert response.status_code == 400
    other = client.post('/api/images', files={'file': ('other.png', png(original))}).json()
    assert client.post('/api/jobs', json={'image_id': other['id'], 'mask_id': saved_mask['id']}).status_code == 400
    for backend, planner in [('lama', 'rules'), ('opencv', 'api')]:
        assert client.post('/api/jobs', json={'image_id': image['id'], 'mask_id': saved_mask['id'],
                                            'backend': backend, 'planner': planner}).status_code == 409
    assert client.get(f'/api/assets/{image["id"]}/source.bin').status_code == 404
    assert client.get('/api/assets/../../.env').status_code == 404
    assert client.post('/api/jobs', json={'image_id': image['id'], 'mask_id': saved_mask['id'], 'candidate_count': 20}).status_code == 422


def test_compositing_discards_all_unmasked_model_changes():
    original = np.full((8, 12, 3), [30, 60, 90], dtype=np.uint8)
    prediction = np.full_like(original, 255)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:4, 3:7] = 255
    result = composite(original, prediction, mask)
    assert np.array_equal(result[mask == 0], original[mask == 0])
    assert np.all(result[mask > 0] == 255)


def test_thin_mask_composite_keeps_full_repair_strength():
    original = np.full((8, 12, 3), [30, 60, 90], dtype=np.uint8)
    prediction = np.full_like(original, 255)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[4, 3:9] = 255  # 1 px hairline crack
    result = composite(original, prediction, mask)
    assert np.array_equal(result[mask == 0], original[mask == 0])
    assert np.all(result[mask > 0] == 255)


def test_thick_mask_composite_feathers_boundary_not_interior():
    original = np.full((40, 40, 3), [20, 40, 60], dtype=np.uint8)
    prediction = np.full_like(original, 220)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[8:32, 8:32] = 255
    result = composite(original, prediction, mask)
    assert np.array_equal(result[mask == 0], original[mask == 0])
    assert np.all(result[18:22, 18:22] == 220)


def test_mixed_mask_keeps_fine_crack_when_a_thick_hole_is_present():
    """A 40 px crack stays repaired when the same mask also has a 20x20 hole.

    Feathering used to take the mask-wide maximum thickness, so the hole
    drove the crack's blend weight to zero (0/40 after a perfect prediction
    had repaired 40/40 alone). Local thickness keeps the crack, including
    when the crack touches the hole, and the hole center still takes the fill.
    """
    surround = np.array([30, 60, 90], np.uint8)
    damage = np.array([214, 206, 190], np.uint8)
    fill = np.array([12, 180, 48], np.uint8)

    def scene(mask):
        image = np.empty(mask.shape + (3,), np.uint8)
        image[:] = surround
        image[mask > 0] = damage
        prediction = np.empty_like(image)
        prediction[:] = fill
        return image, prediction

    separate = np.zeros((64, 96), np.uint8)
    separate[8, 4:44] = 255
    separate[28:48, 60:80] = 255
    original, prediction = scene(separate)
    result = composite(original, prediction, separate)
    assert np.array_equal(result[separate == 0], original[separate == 0])
    assert np.all(result[8, 4:44] == fill)
    assert np.all(result[38, 70] == fill)

    attached = np.zeros((64, 96), np.uint8)
    attached[20, 4:44] = 255
    attached[16:36, 44:64] = 255
    original, prediction = scene(attached)
    result = composite(original, prediction, attached)
    assert np.array_equal(result[attached == 0], original[attached == 0])
    assert np.all(result[20, 4:36] == fill)
    assert np.all(result[26, 54] == fill)


def test_export_blocks_rejected_and_unflagged_hypothesis(workbench):
    client, app = workbench
    image, mask_record, *_ = upload_case(client)
    response = client.post('/api/jobs', json={'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1})
    job = wait_job(client, response.json()['id'])
    chosen = job['candidates'][0]
    assert client.post(f'/api/jobs/{job["id"]}/review',
                       json={'candidate_id': chosen['id'], 'note': 'ok'}).status_code == 200
    store = app.state.store
    store.update('job', job['id'], selected_as='rejected', review={'decision': 'reject'})
    blocked = client.get(f'/api/jobs/{job["id"]}/export')
    assert blocked.status_code == 400 and '驳回' in blocked.json()['detail']
    store.update('job', job['id'], selected_as='hypothesis',
                 review={'decision': 'adopt', 'archive_as_hypothesis': False})
    blocked = client.get(f'/api/jobs/{job["id"]}/export')
    assert blocked.status_code == 400 and '假设' in blocked.json()['detail']
    store.update('job', job['id'], selected_as='hypothesis',
                 review={'decision': 'adopt', 'archive_as_hypothesis': True})
    archive = client.get(f'/api/jobs/{job["id"]}/export')
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content)) as zipped:
        manifest = json.loads(zipped.read('manifest.json'))
        assert manifest['selected_as'] == 'hypothesis'


def test_transparent_white_is_not_an_editable_mask_region():
    rgba = np.full((8, 8, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[2:4, 2:4, 3] = 255
    decoded = np.asarray(decode_image(png(rgba), mask=True))
    assert (decoded > 127).sum() == 4


def test_defect_jobs_ignore_risk_classification(workbench):
    """User explicitly opted out of risk gating for defect-list repairs."""
    client, app = workbench
    image, mask_record, original, mask = upload_case(client)
    store, engine = app.state.store, app.state.engine
    proposal_id = new_id()
    folder = store.folder(proposal_id)
    proposal_mask = np.zeros((64, 96), dtype=np.uint8)
    proposal_mask[20:40, 30:50] = 255
    Image.fromarray(proposal_mask).save(folder / 'mask.png')
    store.put('proposal', {
        'id': proposal_id, 'image_id': image['id'], 'created_at': '2026-11-09T00:00:00+00:00',
        'fraction': float((proposal_mask > 0).mean()), 'confirmed': False,
        'region_class': 'unrestorable', 'repair_allowed': False,
        'metadata': {'type': 'crack', 'risk_notice': 'blocked by heuristic',
                     'recommendation': {'backend': 'lama'}},
    })
    from mural_agent.defect_jobs import run_confirmed_defects
    result = run_confirmed_defects(store, engine, image['id'], [proposal_id])
    assert result['skipped'] == []
    assert len(result['jobs']) == 1
    job = result['jobs'][0]
    assert job['defect_risk_class'] == 'unrestorable'
    working_mask = store.get('mask', job['mask_id'])
    assert working_mask['region_class'] == 'fillable_lacuna'


def test_cancelled_queued_job_never_executes(workbench):
    client, app = workbench
    identifier = new_id()
    app.state.store.put('job', {'id': identifier, 'status': 'queued', 'cancel_requested': True, 'events': [], 'candidates': []})
    app.state.engine.run(identifier)
    job = app.state.store.get('job', identifier)
    assert job['status'] == 'cancelled'
    assert job['candidates'] == []


def test_restart_marks_inflight_task_interrupted(tmp_path):
    store = Store(tmp_path)
    identifier = new_id()
    store.put('job', {'id': identifier, 'status': 'running'})
    restarted = Store(tmp_path)
    assert restarted.get('job', identifier)['status'] == 'interrupted'


def test_failed_api_planning_records_explicit_fallback(workbench, monkeypatch):
    client, app = workbench
    image, mask, _, _ = upload_case(client)
    app.state.settings.api_model = 'test-vision'
    app.state.settings.api_key = 'test-secret-never-log'
    def fail(*args, **kwargs):
        raise RuntimeError('test-secret-never-log')
    monkeypatch.setattr('mural_agent.engine.api_plan', fail)
    response = client.post('/api/jobs', json={'image_id': image['id'], 'mask_id': mask['id'], 'planner': 'api'})
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'review'
    assert job['plan']['source'] == 'rules'
    assert any('回退' in event['message'] for event in job['events'])
    assert 'test-secret-never-log' not in json.dumps(job)

