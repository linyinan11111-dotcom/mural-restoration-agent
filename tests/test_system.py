from io import BytesIO
import json
import time
import zipfile
import cv2
import numpy as np
import pytest
from PIL import Image
from mural_agent import workspace, references, datasets
from mural_agent.config import Settings
from mural_agent.storage import Store
from mural_agent.engine import Engine
from mural_agent.planner import rule_plan
from mural_agent.imaging import composite
from mural_agent.quality import assess, paired_metrics
from mural_agent.backup import backup_workspace, restore_workspace


def png(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, 'PNG')
    return stream.getvalue()


@pytest.fixture
def system(tmp_path):
    settings = Settings(tmp_path/'store', tmp_path/'missing.pt')
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    yield store, engine
    engine.executor.shutdown(wait=True)


def inputs(store):
    rng = np.random.default_rng(41)
    image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    for i in range(80):
        cv2.circle(image, (int(rng.integers(15, 385)), int(rng.integers(15, 285))), 5, (30, 200, 120), 2)
    record = workspace.add_image(store, png(image), 'texture.png')
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[120:145, 180:205] = 255
    saved_mask = workspace.add_mask(store, record, png(mask))
    return image, mask, record, saved_mask


def wait(store, job):
    for _ in range(500):
        latest = store.get('job', job['id'])
        if latest['status'] not in {'queued', 'planning', 'running', 'evaluating'}:
            return latest
        time.sleep(.01)
    pytest.fail('Job timeout')


def test_quality_feedback_has_hard_limits_and_original_input(system, monkeypatch):
    store, engine = system
    image, mask, record, saved = inputs(store)
    seen = []
    def tool(step, original, active, job, folder, timeout):
        seen.append((original.copy(), step.tool))
        return original.copy(), {'device': 'cpu'}  # no_change triggers actual feedback.
    monkeypatch.setattr(engine.tools, 'execute', tool)
    job = workspace.create_job(store, engine, record, saved, 'opencv', 'rules', 3, '')
    result = wait(store, job)
    assert result['status'] == 'review'
    assert result['quality_outcome'] == 'needs_human_review'
    assert len(result['revisions']) == 2
    assert len(result['candidates']) == 3
    assert len(result['attempts']) == 3
    assert all(np.array_equal(a, image) for a, _ in seen)
    assert seen[0][1] != seen[1][1]
    assert result['plan']['regions'][0]['mask_id'] == saved['id']


def test_tool_failures_exit_without_infinite_retries(system, monkeypatch):
    store, engine = system
    _, _, image, mask = inputs(store)
    def fail(*args):
        raise RuntimeError('local failure')
    monkeypatch.setattr(engine.tools, 'execute', fail)
    result = wait(store, workspace.create_job(store, engine, image, mask, 'opencv', 'rules', 3, ''))
    assert result['status'] == 'failed'
    assert len(result['attempts']) == 3
    assert len(result['revisions']) == 2


def test_resume_preserves_candidate_and_budget(system):
    store, engine = system
    _, _, image, mask = inputs(store)
    result = wait(store, workspace.create_job(store, engine, image, mask, 'opencv', 'rules', 2, ''))
    first = result['candidates'][0]
    # Simulate a checkpoint after one candidate; remove the second DB row to model pre-run state.
    with store.connect() as con:
        con.execute('DELETE FROM records WHERE kind=? AND id=?', ('candidate', result['candidates'][1]['id']))
    store.update('job', result['id'], status='interrupted', candidates=[first], attempts=result['attempts'][:1])
    engine.resume(result['id']).result(timeout=10)
    recovered = store.get('job', result['id'])
    assert recovered['status'] == 'review'
    assert len(recovered['candidates']) == 2
    assert recovered['candidates'][0]['id'] == first['id']
    assert len({c['id'] for c in recovered['candidates']}) == 2


def test_resume_rejects_tampered_frozen_input(system):
    store, engine = system
    array, _, image, mask = inputs(store)
    job = wait(store, workspace.create_job(store, engine, image, mask, 'opencv', 'rules', 1, ''))
    store.update('job', job['id'], status='interrupted')
    Image.fromarray(np.zeros_like(array)).save(store.folder(image['id'])/'image.png')
    engine.resume(job['id']).result(timeout=10)
    result = store.get('job', job['id'])
    assert result['status'] == 'failed' and 'hash' in result['error']


def test_registration_requires_provenance_review_and_exact_coverage(system):
    store, engine = system
    image, mask, record, saved = inputs(store)
    metadata = {'source': 'test fixture', 'license': 'generated fixture', 'group': 'fixture', 'relation': 'same_mural'}
    ref = references.add_reference(store, png(image), 'reference', metadata)
    registration = references.register(store, record, ref, mask)
    assert registration['inliers'] >= 12 and not registration['approved']
    with pytest.raises(ValueError, match='核验'):
        workspace.create_job(store, engine, record, saved, 'reference', 'rules', 1, '', mode='evidence', registration_id=registration['id'])
    references.approve_registration(store, registration['id'], png(mask), 'Verified generated fixture')
    job = wait(store, workspace.create_job(store, engine, record, saved, 'reference', 'rules', 1, '', mode='evidence', registration_id=registration['id']))
    assert job['status'] == 'review'
    store.update('job', job['id'], selected_id=job['candidates'][0]['id'])
    with zipfile.ZipFile(engine.export(job['id'])) as archive:
        assert {'evidence/source.bin', 'evidence/aligned.png', 'evidence/approved.png'} <= set(archive.namelist())
    metadata['relation'] = 'related'
    weak = references.add_reference(store, png(image), 'weak', metadata)
    with pytest.raises(ValueError, match='同题材'):
        references.register(store, record, weak, mask)


def test_thick_mask_score_ranks_the_edited_band_not_only_the_preserved_edge():
    """Black, matching gray, and white fills of one thick hole must not tie.

    The preserved outer ring makes the edge statistic identical. The score
    has to look at the band that was actually blended, and the fill closer
    to the surrounding paint ranks above the worse fills (lower score).
    """
    surround = np.array([90, 110, 130], np.uint8)
    clean = np.empty((48, 48, 3), np.uint8)
    clean[:] = surround
    mask = np.zeros((48, 48), np.uint8)
    mask[14:34, 14:34] = 255
    damaged = clean.copy()
    damaged[mask > 0] = (16, 16, 16)
    fills = {
        'black': np.zeros_like(damaged),
        'gray': clean.copy(),
        'white': np.full_like(damaged, 255),
    }
    scores, edges, bands, maes = {}, {}, {}, {}
    for name, prediction in fills.items():
        output = composite(damaged, prediction, mask)
        metrics = assess(damaged, output, mask)
        scores[name] = metrics['score']
        edges[name] = metrics['boundary_pair_delta']
        bands[name] = metrics['edited_band_delta']
        maes[name] = paired_metrics(clean, output, mask)['mask_mae']
    assert len(set(edges.values())) == 1
    assert len(set(scores.values())) == 3
    assert scores['gray'] < scores['black']
    assert scores['gray'] < scores['white']
    assert bands['gray'] < bands['black']
    assert bands['gray'] < bands['white']
    assert maes['gray'] < maes['black']
    assert maes['gray'] < maes['white']


def test_mixed_mask_composite_and_score_distinguish_repairs():
    """Compositing and scoring together: a crack beside a hole is repaired, and two fills differ."""
    surround = np.array([40, 80, 120], np.uint8)
    damaged = np.empty((64, 96, 3), np.uint8)
    damaged[:] = surround
    mask = np.zeros((64, 96), np.uint8)
    mask[8, 4:44] = 255
    mask[28:48, 60:80] = 255
    damaged[mask > 0] = (210, 200, 180)
    good = np.empty_like(damaged)
    good[:] = surround
    bad = np.full_like(damaged, 255)
    good_out = composite(damaged, good, mask)
    bad_out = composite(damaged, bad, mask)
    assert np.all(good_out[8, 4:44] == surround)
    assert np.all(bad_out[8, 4:44] == 255)
    assert np.all(good_out[38, 70] == surround)
    assert np.array_equal(good_out[mask == 0], damaged[mask == 0])
    assert np.array_equal(bad_out[mask == 0], damaged[mask == 0])
    good_metrics = assess(damaged, good_out, mask)
    bad_metrics = assess(damaged, bad_out, mask)
    assert good_metrics['edited_band_delta'] < bad_metrics['edited_band_delta']
    assert good_metrics['score'] != bad_metrics['score']
    assert good_metrics['score'] < bad_metrics['score']


def test_engine_recommends_the_better_thick_fill(system, monkeypatch):
    store, engine = system
    surround = np.array([90, 110, 130], np.uint8)
    image = np.empty((48, 48, 3), np.uint8)
    image[:] = surround
    mask = np.zeros((48, 48), np.uint8)
    mask[14:34, 14:34] = 255
    image[mask > 0] = (16, 16, 16)
    record = workspace.add_image(store, png(image), 'hole.png')
    saved = workspace.add_mask(store, record, png(mask))
    fills = [
        np.zeros_like(image),
        np.broadcast_to(surround, image.shape).copy(),
        np.full_like(image, 255),
    ]

    def tool(step, original, active, job, folder, timeout):
        return fills.pop(0), {'device': 'cpu'}

    monkeypatch.setattr(engine.tools, 'execute', tool)
    result = wait(store, workspace.create_job(store, engine, record, saved, 'opencv', 'rules', 3, ''))
    assert result['status'] == 'review'
    scores = [candidate['metrics']['score'] for candidate in result['candidates']]
    edges = [candidate['metrics']['boundary_pair_delta'] for candidate in result['candidates']]
    assert len(set(edges)) == 1
    assert len(set(scores)) == 3
    assert scores[1] < scores[0] and scores[1] < scores[2]
    # The preserved edge still fails the boundary check for every fill.
    # Ranking has to come from the edited-band score, not from candidate order.
    assert all(not candidate['metrics']['passed'] for candidate in result['candidates'])
    assert result['recommended_id'] == result['candidates'][1]['id']


def test_paired_metrics_are_region_scoped():
    target = np.zeros((64, 80, 3), np.uint8)
    output = target.copy()
    mask = np.zeros((64, 80), np.uint8)
    mask[20:30, 30:40] = 255
    output[mask > 0] = 10
    metrics = paired_metrics(target, output, mask)
    assert metrics['mask_mae'] == 10
    assert metrics['mask_psnr'] == pytest.approx(28.1308, abs=.001)
    assert metrics['roi_ssim'] < 1
    identical = paired_metrics(target, target, mask)
    assert identical['perfect_match'] and identical['roi_ssim'] == 1


def test_split_merges_duplicate_images_across_groups():
    records = [{'id': str(i), 'sha256': 'shared' if i < 2 else str(i), 'metadata': {'group': str(i)}} for i in range(20)]
    result = datasets.grouped_split(records)
    assert result['group_count'] == 19
    assignment = {identifier: split for split, ids in result['splits'].items() for identifier in ids}
    assert assignment['0'] == assignment['1']
    assert result == datasets.grouped_split(records)


def test_backup_roundtrip_and_zip_slip_rejection(system, tmp_path):
    store, engine = system
    inputs(store)
    path = tmp_path/'backup.zip'
    backup_workspace(store, path)
    destination = restore_workspace(path, tmp_path/'restored')
    restored = Store(destination)
    assert len(restored.list('image')) == 1
    malicious = tmp_path/'bad.zip'
    with zipfile.ZipFile(malicious, 'w') as archive:
        archive.writestr('backup.json', json.dumps({'format': 1, 'sha256': {}}))
        archive.writestr('../outside.txt', 'bad')
    with pytest.raises(ValueError, match='不安全'):
        restore_workspace(malicious, tmp_path/'not-created')
    assert not (tmp_path/'outside.txt').exists()


def test_speculative_cannot_run_in_conservative_mode(system, monkeypatch):
    store, engine = system
    _, _, image, mask = inputs(store)
    monkeypatch.setattr(engine.tools, 'availability', lambda: {'diffusion': True, 'controlnet': True})
    with pytest.raises(ValueError, match='推测'):
        workspace.create_job(store, engine, image, mask, 'diffusion', 'rules', 1, '')
