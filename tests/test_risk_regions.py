"""Risk region classification + professional gates (no generative deliverable)."""
from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from mural_agent.app import create_app
from mural_agent.config import Settings
from mural_agent import workspace
from mural_agent.professional import WORKFLOW_PROFESSIONAL, validate_professional_create
from mural_agent.risk_regions import (
    classify_component, classify_mask_regions, render_risk_overlay,
    apply_region_action, mask_blocks_repair, aggregate_region_class,
)


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


def upload_case(client, region_class=None):
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
    saved_mask = client.post(f'/api/images/{record["id"]}/masks',
                             files={'file': ('mask.png', png(mask), 'image/png')})
    assert saved_mask.status_code == 201
    mask_json = saved_mask.json()
    return record, mask_json, image, mask


def test_classify_component_rules():
    assert classify_component(area_frac=0.001, mount_touch=0.0, line_score=0.0)[0] == 'fillable_lacuna'
    assert classify_component(area_frac=0.05, mount_touch=0.0, line_score=0.0)[0] == 'uncertain'
    assert classify_component(area_frac=0.02, mount_touch=0.2, line_score=0.0)[0] == 'unrestorable'
    assert classify_component(area_frac=0.05, mount_touch=0.0, line_score=0.30)[0] == 'fillable_lacuna'


def test_classify_mask_and_overlay_shapes():
    rgb = np.full((80, 120, 3), 160, dtype=np.uint8)
    # olive mount strip on left border
    rgb[:, :15] = [90, 95, 70]
    lacuna = np.zeros((80, 120), dtype=np.uint8)
    lacuna[10:70, 20:100] = 255  # large hole
    risk = classify_mask_regions(rgb, lacuna)
    assert risk['region_class'] in {'uncertain', 'unrestorable', 'fillable_lacuna'}
    assert 'regions' in risk
    overlay = render_risk_overlay(rgb, risk['class_masks'])
    assert overlay.shape == rgb.shape


def test_mask_blocks_repair_and_force():
    mask = {'region_class': 'uncertain', 'user_confirmed': True}
    blocked, msg = mask_blocks_repair(mask)
    assert blocked and '不确定' in msg
    blocked, _ = mask_blocks_repair(mask, expert_force_override=True)
    assert not blocked
    mask2 = {'region_class': 'unrestorable'}
    blocked, msg = mask_blocks_repair(mask2)
    assert blocked and '不可复原' in msg


def test_apply_region_action_and_aggregate():
    regions = [
        {'id': 1, 'region_class': 'uncertain', 'status': 'proposed', 'evidence': 'x'},
        {'id': 2, 'region_class': 'fillable_lacuna', 'status': 'proposed', 'evidence': 'y'},
    ]
    updated = apply_region_action(regions, 1, 'confirm_fillable')
    assert updated[0]['region_class'] == 'fillable_lacuna'
    assert aggregate_region_class(r['region_class'] for r in updated) == 'fillable_lacuna'
    refused = apply_region_action(regions, 1, 'refuse_skip')
    assert refused[0]['status'] == 'refused'


def test_professional_create_blocks_uncertain_without_force():
    mask = {'user_confirmed': True, 'mask_source': 'manual', 'region_class': 'uncertain'}
    with pytest.raises(ValueError, match='不确定|禁止'):
        validate_professional_create(
            mask, rationale='可见露底', evidence_level='conservative', mode='conservative',
            mask_gate_confirmed=True)
    meta = validate_professional_create(
        mask, rationale='专家复核后强制小范围保守', evidence_level='conservative',
        mode='conservative', mask_gate_confirmed=True,
        force_override=True, force_override_reason='专家目视露底连续，仅保守 Telea')
    assert meta['force_override'] is True


def test_professional_create_blocks_unrestorable():
    mask = {'user_confirmed': True, 'mask_source': 'manual', 'region_class': 'unrestorable'}
    with pytest.raises(ValueError, match='不可复原|禁止'):
        validate_professional_create(
            mask, rationale='试修', evidence_level='conservative', mode='conservative',
            mask_gate_confirmed=True)


def test_api_job_blocked_for_uncertain_mask(workbench):
    client, app = workbench
    image, mask_record, *_ = upload_case(client)
    store = app.state.store
    store.update('mask', mask_record['id'], region_class='uncertain', user_confirmed=True)
    response = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'workflow': 'professional_step', 'rationale': '想试试',
        'evidence_level': 'conservative', 'backend': 'opencv',
    })
    assert response.status_code == 400
    assert '禁止' in response.json()['detail'] or '不确定' in response.json()['detail']


def test_api_job_force_override_allows_conservative(workbench):
    client, app = workbench
    image, mask_record, original, mask = upload_case(client)
    store = app.state.store
    store.update('mask', mask_record['id'], region_class='uncertain', user_confirmed=True)
    response = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'workflow': 'professional_step', 'rationale': '专家强制：仅保守纹理，不生成纹样',
        'evidence_level': 'conservative', 'backend': 'opencv',
        'mask_gate_confirmed': True,
        'force_override': True,
        'force_override_reason': '专家目视露底连续，仅保守纹理',
    })
    assert response.status_code == 202, response.text


def test_uncertain_mask_blocks_standard_workflow(workbench):
    """H1 regression: standard workflow must not skip the risk gate."""
    client, app = workbench
    image, mask_record, *_ = upload_case(client)
    store = app.state.store
    store.update('mask', mask_record['id'], region_class='uncertain', user_confirmed=True)
    response = client.post('/api/jobs', json={
        'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
        'backend': 'opencv', 'workflow': 'standard',
    })
    assert response.status_code == 400
    assert '不确定' in response.json()['detail'] or '禁止' in response.json()['detail']


def test_standard_enhanced_requires_explicit_gates(workbench):
    """H2 regression: enhanced cannot bypass opt-in flags via standard workflow."""
    client, app = workbench
    image, mask_record, *_ = upload_case(client)
    base = {'image_id': image['id'], 'mask_id': mask_record['id'], 'candidate_count': 1,
            'backend': 'opencv', 'workflow': 'standard', 'mode': 'enhanced'}
    blocked = client.post('/api/jobs', json=base)
    assert blocked.status_code == 400
    assert '加强' in blocked.json()['detail'] or '已核对' in blocked.json()['detail']
    allowed = client.post('/api/jobs', json={
        **base, 'mask_gate_confirmed': True, 'allow_enhanced': True, 'risk_acknowledged': True})
    assert allowed.status_code == 202, allowed.text


def test_rejected_and_hypothesis_results_cannot_be_exported_as_pixels():
    from mural_agent.professional import validate_selected_export
    candidate = {'id': 'c1', 'tool': 'telea'}
    with pytest.raises(ValueError, match='驳回'):
        validate_selected_export({'selected_as': 'rejected', 'review': {'decision': 'reject'}}, candidate)
    with pytest.raises(ValueError, match='假设'):
        validate_selected_export({'selected_as': 'hypothesis', 'review': {'archive_as_hypothesis': False}}, candidate)
    allowed = validate_selected_export(
        {'selected_as': 'hypothesis', 'review': {'archive_as_hypothesis': True}}, candidate)
    assert allowed['selected_as'] == 'hypothesis'


def test_workspace_create_job_unit_block(workbench):
    client, app = workbench
    store, engine = app.state.store, app.state.engine
    image, mask_record, *_ = upload_case(client)
    store.update('mask', mask_record['id'], region_class='unrestorable', user_confirmed=True)
    mask_record = store.get('mask', mask_record['id'])
    with pytest.raises(ValueError, match='不可复原|禁止'):
        workspace.create_job(
            store, engine, image, mask_record, 'opencv', 'rules', 1, 'note',
            workflow=WORKFLOW_PROFESSIONAL, rationale='不可补绘', evidence_level='conservative')
