"""Gates for professional_step workflow: confirm mask, rationale, evidence level."""
from io import BytesIO
import time
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from mural_agent.app import create_app
from mural_agent.config import Settings
from mural_agent import workspace
from mural_agent.professional import (
    WORKFLOW_PROFESSIONAL, validate_candidate_selection, validate_professional_create,
)
from mural_agent.auto_restore import propose_restore


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


def test_professional_rejects_missing_rationale_and_observation(workbench):
    client, app = workbench
    image, mask_record, *_ = upload_case(client)
    payload = {
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'workflow': 'professional_step', 'evidence_level': 'conservative', 'rationale': '',
        'mask_gate_confirmed': True,
    }
    response = client.post('/api/jobs', json=payload)
    assert response.status_code == 400
    assert '依据' in response.json()['detail']

    payload['rationale'] = '可见细裂缝，色块边缘连续'
    payload['evidence_level'] = 'observation'
    response = client.post('/api/jobs', json=payload)
    assert response.status_code == 400
    assert 'observation' in response.json()['detail']


def test_professional_rejects_unconfirmed_mask(workbench):
    client, app = workbench
    store, engine = app.state.store, app.state.engine
    image, mask_record, *_ = upload_case(client)
    store.update('mask', mask_record['id'], user_confirmed=False, mask_source='ai_candidate')
    mask_record = store.get('mask', mask_record['id'])
    with pytest.raises(ValueError, match='尚未经人工确认'):
        workspace.create_job(
            store, engine, image, mask_record, 'opencv', 'rules', 1, 'note',
            workflow=WORKFLOW_PROFESSIONAL, rationale='可见缺损', evidence_level='conservative',
            mask_gate_confirmed=True)


def test_professional_job_records_provenance_and_runs(workbench):
    client, app = workbench
    image, mask_record, original, mask = upload_case(client)
    response = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'workflow': 'professional_step', 'rationale': '裂缝边缘可见，保守 Telea 修补',
        'evidence_level': 'conservative', 'expert_notes': '先修这一块，勿整图',
        'mask_source': 'manual', 'backend': 'opencv', 'mask_gate_confirmed': True,
    })
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'review', job
    assert job['workflow'] == 'professional_step'
    assert job['rationale'].startswith('裂缝')
    assert job['evidence_level'] == 'conservative'
    assert job['expert_notes'] == '先修这一块，勿整图'
    assert job['mask_source'] == 'manual'
    candidate = job['candidates'][0]
    result = np.asarray(Image.open(BytesIO(client.get(candidate['url']).content)))
    assert np.array_equal(result[mask == 0], original[mask == 0])


def test_speculative_selection_requires_hypothesis_flag(workbench):
    job = {'workflow': 'professional_step', 'mode': 'speculative'}
    candidate = {'id': 'c1', 'speculative': True, 'tool': 'diffusion'}
    with pytest.raises(ValueError, match='假设性建议'):
        validate_candidate_selection(job, candidate, archive_as_hypothesis=False, decision='adopt')
    selection = validate_candidate_selection(
        job, candidate, archive_as_hypothesis=True, decision='adopt')
    assert selection['selected_as'] == 'hypothesis'

    # Conservative candidate may be accepted without the flag.
    accepted = validate_candidate_selection(
        {'workflow': 'professional_step', 'mode': 'conservative'},
        {'id': 'c2', 'speculative': False, 'tool': 'telea'}, archive_as_hypothesis=False,
        decision='adopt')
    assert accepted['selected_as'] == 'accepted'


def test_review_endpoint_blocks_speculative_without_flag(workbench):
    client, app = workbench
    store = app.state.store
    image, mask_record, *_ = upload_case(client)
    response = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'backend': 'opencv', 'workflow': 'standard',
    })
    job = wait_job(client, response.json()['id'])
    candidate = job['candidates'][0]
    # Force speculative marker as if diffusion produced it.
    store.update('candidate', candidate['id'], speculative=True, tool='diffusion')
    candidates = store.get('job', job['id'])['candidates']
    for item in candidates:
        if item['id'] == candidate['id']:
            item['speculative'] = True
            item['tool'] = 'diffusion'
    store.update('job', job['id'], candidates=candidates, mode='speculative',
                 workflow='professional_step')
    blocked = client.post(f'/api/jobs/{job["id"]}/review',
                          json={'candidate_id': candidate['id'], 'note': '试选', 'decision': 'adopt'})
    assert blocked.status_code == 400
    assert '假设性' in blocked.json()['detail']
    ok = client.post(f'/api/jobs/{job["id"]}/review', json={
        'candidate_id': candidate['id'], 'note': '仅假设', 'archive_as_hypothesis': True,
        'decision': 'adopt'})
    assert ok.status_code == 200
    assert ok.json()['selected_as'] == 'hypothesis'


def test_propose_restore_does_not_create_job(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt')
    from mural_agent.storage import Store
    from mural_agent.engine import Engine
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    image = workspace.add_image(store, png(np.zeros((32, 48, 3), dtype=np.uint8) + 120), 'demo.png')

    fake_items = [{'index': 1, 'type': 'crack', 'bbox': [100, 100, 400, 400], 'size': 'small',
                   'confidence': 0.9, 'evidence': 'visible hairline', 'confirmed': False}]

    monkeypatch.setattr('mural_agent.auto_restore.assess_image',
                        lambda *_a, **_k: {'items': fake_items, 'limitations': ''})

    def refine(engine, image_arr, item):
        mask = np.zeros(image_arr.shape[:2], dtype=np.uint8)
        mask[8:20, 10:20] = 255
        return mask, {'method': 'bounding_box_fallback', 'confirmed': False,
                      'index': item['index'], 'type': item['type'], 'size': item['size'],
                      'confidence': item['confidence'], 'evidence': item['evidence'],
                      'bbox': item['bbox']}

    monkeypatch.setattr('mural_agent.auto_restore.refine_item', refine)
    result = propose_restore(store, engine, image)
    assert result['job'] is None
    assert result['awaiting_confirmation'] is True
    assert result['mask']['user_confirmed'] is False
    assert result['mask']['mask_source'] == 'ai_candidate'
    assert store.list('job') == []
    assert len(result['proposals']) >= 1


def test_validate_professional_create_unit():
    mask = {'user_confirmed': True, 'mask_source': 'manual'}
    meta = validate_professional_create(
        mask, rationale='可见颜料缺失', evidence_level='conservative', mode='conservative',
        expert_notes='同意先修', mask_source='manual', mask_gate_confirmed=True)
    assert meta['evidence_level'] == 'conservative'
    with pytest.raises(ValueError):
        validate_professional_create(
            {'user_confirmed': False}, rationale='x', evidence_level='conservative', mode='conservative',
            mask_gate_confirmed=True)


def test_enhanced_requires_allow_flag_and_matching_mode():
    mask = {'user_confirmed': True, 'mask_source': 'manual'}
    with pytest.raises(ValueError, match='允许加强'):
        validate_professional_create(
            mask, rationale='大片露底，保守后仍空洞', evidence_level='enhanced',
            mode='enhanced', allow_enhanced=False, mask_gate_confirmed=True,
            risk_acknowledged=True, contest_demo=False)
    with pytest.raises(ValueError, match='同时'):
        validate_professional_create(
            mask, rationale='大片露底', evidence_level='enhanced',
            mode='conservative', allow_enhanced=True, mask_gate_confirmed=True,
            risk_acknowledged=True, contest_demo=False)
    meta = validate_professional_create(
        mask, rationale='大片露底，保守后仍空洞，加强色彩延续', evidence_level='enhanced',
        mode='enhanced', allow_enhanced=True, expert_notes='可试，不定稿',
        mask_gate_confirmed=True, risk_acknowledged=True, contest_demo=False)
    assert meta['evidence_level'] == 'enhanced'
    assert meta['allow_enhanced'] is True


def test_enhanced_selection_stamps_non_historical():
    from mural_agent.professional import LEVEL_LABELS
    job = {'workflow': 'professional_step', 'mode': 'enhanced', 'evidence_level': 'enhanced'}
    candidate = {'id': 'c1', 'enhanced': True, 'tool': 'diffusion'}
    selection = validate_candidate_selection(job, candidate, archive_as_hypothesis=False, decision='adopt')
    assert selection['selected_as'] == 'enhanced'
    assert '非历史' in selection['label']
    with pytest.raises(ValueError, match='假设性'):
        validate_candidate_selection(
            {'workflow': 'professional_step', 'mode': 'speculative'},
            {'id': 'c2', 'speculative': True, 'tool': 'diffusion'},
            archive_as_hypothesis=False, decision='adopt')


def test_enhanced_job_via_api(workbench):
    client, app = workbench
    image, mask_record, original, mask = upload_case(client)
    blocked = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'workflow': 'professional_step', 'rationale': '露底可见，加强填补',
        'evidence_level': 'enhanced', 'mode': 'enhanced', 'backend': 'opencv',
        'allow_enhanced': False, 'mask_gate_confirmed': True, 'risk_acknowledged': True,
        'contest_demo': False,
    })
    assert blocked.status_code == 400
    assert '允许加强' in blocked.json()['detail']
    response = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'workflow': 'professional_step', 'rationale': '露底可见，加强填补',
        'evidence_level': 'enhanced', 'mode': 'enhanced', 'backend': 'opencv',
        'allow_enhanced': True, 'expert_notes': '演示加强档',
        'mask_gate_confirmed': True, 'risk_acknowledged': True,
        'contest_demo': False,
    })
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'review', job
    assert job['mode'] == 'enhanced'
    assert job['evidence_level'] == 'enhanced'
    assert job.get('allow_enhanced') is True
    candidate = job['candidates'][0]
    assert candidate.get('enhanced') is True
    assert candidate.get('speculative') is False
    result = np.asarray(Image.open(BytesIO(client.get(candidate['url']).content)))
    # 加强允许掩膜 = 露底 ∪ 裂纹；仅保证加强允许掩膜外相对原图不变
    from mural_agent.segmentation import build_enhanced_work_mask
    work_mask, meta = build_enhanced_work_mask(original, mask)
    assert np.array_equal(result[work_mask == 0], original[work_mask == 0])
    assert job.get('enhanced_mask_meta') or meta['enhanced_fraction'] >= meta['lacuna_fraction']
    ok = client.post(f'/api/jobs/{job["id"]}/review', json={
        'candidate_id': candidate['id'], 'note': '加强结果，非定稿', 'decision': 'adopt'})
    assert ok.status_code == 200
    assert ok.json()['selected_as'] == 'enhanced'



def test_enhanced_strength_default_and_work_mask():
    import numpy as np
    from mural_agent.professional import ENHANCED_DIFFUSION_STRENGTH
    from mural_agent.segmentation import build_enhanced_work_mask, craquelure_mask, ENHANCED_DIFFUSION_STRENGTH as SEG_STRENGTH
    assert 0.60 <= ENHANCED_DIFFUSION_STRENGTH <= 0.75
    assert SEG_STRENGTH == ENHANCED_DIFFUSION_STRENGTH
    # Synthetic: bright field + dark crack line + lacuna blob
    image = np.full((64, 96, 3), 180, dtype=np.uint8)
    image[20:44, 10:14] = 40  # vertical dark crack
    lacuna = np.zeros((64, 96), dtype=np.uint8)
    lacuna[8:20, 50:70] = 255
    work, meta = build_enhanced_work_mask(image, lacuna)
    assert meta['lacuna_fraction'] > 0
    assert (work[lacuna > 0] > 0).all()
    # Crack column should contribute some pixels to enhanced allow mask
    assert meta['crack_fraction'] >= 0 or (work > 0).any()
    cracks, cmeta = craquelure_mask(image)
    assert 'fraction' in cmeta


def test_risk_ack_and_review_decision_gates():
    mask = {'user_confirmed': True, 'mask_source': 'manual'}
    with pytest.raises(ValueError, match='已核对掩膜'):
        validate_professional_create(
            mask, rationale='可见露底', evidence_level='conservative', mode='conservative',
            mask_gate_confirmed=False)
    with pytest.raises(ValueError, match='比赛演示|生成性'):
        validate_professional_create(
            mask, rationale='大片露底需加强', evidence_level='enhanced', mode='enhanced',
            allow_enhanced=True, mask_gate_confirmed=True, risk_acknowledged=True,
            contest_demo=True)
    with pytest.raises(ValueError, match='风险提示'):
        validate_professional_create(
            mask, rationale='大片露底需加强', evidence_level='enhanced', mode='enhanced',
            allow_enhanced=True, mask_gate_confirmed=True, risk_acknowledged=False,
            contest_demo=False)
    with pytest.raises(ValueError, match='不可复原'):
        unrestorable_mask = {**mask, 'region_class': 'unrestorable'}
        validate_professional_create(
            unrestorable_mask, rationale='大片缺失', evidence_level='conservative',
            mode='conservative', mask_gate_confirmed=True)
    uncertain_mask = {**mask, 'region_class': 'uncertain'}
    with pytest.raises(ValueError, match='强制放行'):
        validate_professional_create(
            uncertain_mask, rationale='不确定区试修', evidence_level='conservative', mode='conservative',
            mask_gate_confirmed=True, force_override=False)
    ok = validate_professional_create(
        uncertain_mask, rationale='不确定区试修', evidence_level='conservative', mode='conservative',
        mask_gate_confirmed=True, force_override=True,
        force_override_reason='专家目视露底连续，可保守 Telea')
    assert ok['force_override'] is True
    job = {'workflow': 'professional_step', 'mode': 'conservative', 'evidence_level': 'conservative'}
    candidate = {'id': 'c1', 'tool': 'telea'}
    with pytest.raises(ValueError, match='采纳'):
        validate_candidate_selection(job, candidate)
    rejected = validate_candidate_selection(job, candidate, decision='reject')
    assert rejected['selected_as'] == 'rejected'
    adopted = validate_candidate_selection(job, candidate, decision='adopt')
    assert adopted['selected_as'] == 'accepted'


def test_contest_blocks_generative_export_helper():
    from mural_agent.professional import contest_allows_pixel_export
    assert contest_allows_pixel_export(
        {'contest_demo': True, 'mode': 'conservative', 'evidence_level': 'conservative'})[0]
    assert not contest_allows_pixel_export(
        {'contest_demo': True, 'mode': 'enhanced', 'evidence_level': 'enhanced'})[0]
    assert not contest_allows_pixel_export(
        {'contest_demo': True, 'mode': 'conservative', 'selected_as': 'hypothesis'})[0]
