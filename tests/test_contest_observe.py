"""Contest observe gate: unreviewed export is blocked, reviewed export is annotation-only."""
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mural_agent.app import create_app
from mural_agent.config import ROOT, Settings
from mural_agent.contest_pack import resolve_competition_pack
from mural_agent.engine import Engine
from mural_agent.human_gate import (
    apply_evaluate,
    builtin_pending_document,
    builtin_reviewed_document,
    load_builtin_sample,
    load_sample,
    resolve_gate_dir,
    write_observe_zip,
)
from mural_agent.professional import contest_allows_pixel_export
from mural_agent.storage import Store


@pytest.fixture
def workbench(tmp_path):
    settings = Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt')
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app


def test_shipped_pack_and_gate_samples_resolve_outside_gitignored_data():
    pack = resolve_competition_pack(ROOT, create=False)
    assert pack.name == 'demo_competition'
    assert 'datasets' in pack.parts
    assert (pack / 'source.jpg').is_file()
    assert (pack / 'overlay_risk.png').is_file()
    assert (pack / 'compare_source_overlay.png').is_file()
    regions = json.loads((pack / 'regions.json').read_text(encoding='utf-8'))
    kinds = {item.get('region_kind') or item.get('class') for item in regions['regions']}
    assert kinds == {'repairable', 'uncertainty', 'unrestorable'}
    gate_dir = resolve_gate_dir(ROOT)
    assert (gate_dir / 'sample-from-shibi-pending.json').is_file()
    assert (gate_dir / 'sample-from-shibi-reviewed.json').is_file()
    assert 'datasets' in gate_dir.parts


def test_unreviewed_gate_blocks_annotation_and_review_export(tmp_path):
    pending = apply_evaluate(load_sample('sample-from-shibi-pending', root=ROOT))
    assert pending['gate']['blocked'] is True
    assert pending['export_allowed']['annotation_layer'] is False
    assert pending['export_allowed']['review_record'] is False
    assert pending['export_allowed']['restored_image'] is False
    builtin = apply_evaluate(builtin_pending_document())
    assert builtin['gate']['blocked'] is True
    with pytest.raises(ValueError, match='门禁未放行'):
        write_observe_zip(tmp_path / 'blocked.zip', pending, ROOT / 'datasets' / 'demo_competition')
    assert not (tmp_path / 'blocked.zip').exists()


def test_reviewed_export_is_annotation_and_report_only(tmp_path):
    reviewed = apply_evaluate(load_sample('sample-from-shibi-reviewed', root=ROOT))
    assert reviewed['gate']['blocked'] is False
    assert reviewed['export_allowed']['annotation_layer'] is True
    assert reviewed['export_allowed']['review_record'] is True
    assert reviewed['export_allowed']['restored_image'] is False
    fresh = builtin_reviewed_document()
    assert fresh['gate']['blocked'] is False
    assert fresh['export_allowed']['restored_image'] is False

    pack = tmp_path / 'pack'
    pack.mkdir()
    source = ROOT / 'datasets' / 'demo_competition'
    for name in ('overlay_risk.png', 'compare_source_overlay.png', 'legend.png', 'regions.json'):
        (pack / name).write_bytes((source / name).read_bytes())
    Image.new('RGB', (8, 8), (255, 0, 0)).save(pack / 'result.png')
    (pack / 'restored_image.png').write_bytes(b'not-a-real-mural')
    dest = tmp_path / 'observe.zip'
    saved = write_observe_zip(dest, reviewed, pack)
    assert saved['restored_image'] is False
    with zipfile.ZipFile(dest) as zipped:
        names = set(zipped.namelist())
        assert 'gate_review.json' in names
        assert 'README.txt' in names
        assert 'overlay_risk.png' in names
        assert 'regions.json' in names
        blob = json.loads(zipped.read('gate_review.json'))
        assert blob['export_allowed']['restored_image'] is False
        for name in names:
            lowered = Path(name).name.lower()
            assert 'result' not in lowered
            assert 'restored' not in lowered
            assert lowered != 'raw.png'
            assert not lowered.endswith('.bin')


def test_generative_restore_stays_blocked(tmp_path):
    generative = load_builtin_sample('sample-generative-restore')
    assert generative['gate']['blocked'] is True
    assert generative['export_allowed']['restored_image'] is False
    assert any('禁生成' in reason or '生成' in reason for reason in generative['gate']['block_reasons'])
    with pytest.raises(ValueError, match='禁生成|门禁未放行|修复成图'):
        write_observe_zip(tmp_path / 'gen.zip', generative, ROOT / 'datasets' / 'demo_competition')

    flagged = builtin_reviewed_document()
    flagged['export_allowed']['restored_image'] = True
    with pytest.raises(ValueError, match='修复成图'):
        write_observe_zip(tmp_path / 'restored.zip', flagged, None)

    assert contest_allows_pixel_export({
        'contest_demo': True, 'mode': 'conservative', 'evidence_level': 'conservative'})
    assert not contest_allows_pixel_export({
        'contest_demo': True, 'mode': 'enhanced', 'evidence_level': 'enhanced'})
    assert not contest_allows_pixel_export({
        'contest_demo': True, 'mode': 'speculative', 'evidence_level': 'speculative'})
    assert not contest_allows_pixel_export({
        'contest_demo': True, 'mode': 'conservative', 'selected_as': 'hypothesis'})
    assert contest_allows_pixel_export({
        'contest_demo': False, 'mode': 'enhanced', 'evidence_level': 'enhanced'})


def test_engine_export_refuses_contest_generative_final(tmp_path):
    settings = Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt')
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    try:
        store.put('job', {
            'id': 'job-enhanced',
            'selected_id': 'cand-1',
            'candidates': [{'id': 'cand-1', 'tool': 'diffusion', 'enhanced': True}],
            'contest_demo': True,
            'mode': 'enhanced',
            'evidence_level': 'enhanced',
            'selected_as': 'enhanced',
            'image_id': 'img',
            'mask_id': 'mask',
        })
        with pytest.raises(ValueError, match='生成性|成图|标注层'):
            engine.export('job-enhanced')
    finally:
        engine.executor.shutdown(wait=False, cancel_futures=True)


def test_competition_demo_api_starts_blocked_without_a_restored_image(workbench):
    client, _app = workbench
    response = client.post('/api/competition-demo')
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload['gate']['blocked'] is True
    assert payload['export_allowed']['annotation_layer'] is False
    assert payload['export_allowed']['review_record'] is False
    assert payload['export_allowed']['restored_image'] is False
    assert payload['observe']['no_generative_mural'] is True
    overlay = client.get(payload['observe']['overlay_risk_url'])
    assert overlay.status_code == 200
    assert overlay.headers['content-type'].startswith('image/')
    Image.open(BytesIO(overlay.content))


def test_ui_keeps_contest_entry_and_annotation_only_export_copy():
    css = (ROOT / 'ui' / 'styles.css').read_text(encoding='utf-8')
    script = (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')
    html = (ROOT / 'ui' / 'index.html').read_text(encoding='utf-8')
    assert 'load-competition-demo' in html
    assert '演示：已复核可导出' in html
    assert '[data-action="load-competition-demo"] { display: inline-flex; }' in css
    assert '导出复核档案' in script
    assert 'region_kind' in script
    assert '比赛观察模式禁止保存修复成图' in script
    assert '比赛观察模式不生成修复成图' in script
