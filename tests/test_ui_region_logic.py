"""Node checks for the shared region/batch rules used by ui/app.js.

These cover the decisions that used to live inline: uncertain red was classified
as unrestorable, eraser strokes were dropped when risk marks were redrawn, the
defect checkbox did not filter a batch, and created jobs were not focused.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')

pytestmark = pytest.mark.skipif(not NODE, reason='node is required to load ui/region_logic.js')


def node_json(script):
    completed = subprocess.run([NODE, '-e', script], cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_audit_pixel_stays_uncertain_and_eraser_stays_in_history():
    payload = node_json(r"""
const logic = require('./ui/region_logic.js');
const uncertain = [220, 0, 60, 184];
const unrestorable = [180, 0, 40, 217];
const oldWouldBlock = (pixel) => {
  const [r, g, , a] = pixel;
  return a >= 40 && r > 150 && g < 50 && (r - g) > 120;
};
const edits = logic.regionEdits(
  ['region', 'mask'],
  [{tool: 'uncertain', points: [[8, 8]], size: 12}],
  [{tool: 'eraser', points: [[8, 8]], size: 80}],
  null);
const batch = logic.batchRunIndexes([
  {includeInBatch: false},
  {includeInBatch: true},
  {},
]);
const focus = logic.focusCreatedJobs([{id: 'a', status: 'queued'}, {id: 'b', status: 'queued'}]);
console.log(JSON.stringify({
  uncertain: logic.classifyRegionPixel(...uncertain),
  unrestorable: logic.classifyRegionPixel(...unrestorable),
  oldUncertainMatchedBoth: oldWouldBlock(uncertain) && oldWouldBlock(unrestorable),
  categoryUncertain: logic.classifyCategorySample(255, 0, 0, 255),
  categoryUnrestorable: logic.classifyCategorySample(0, 0, 255, 255),
  faintIgnored: logic.classifyCategorySample(255, 0, 0, 40),
  edits: edits.map(edit => edit.op),
  batch,
  focus,
}));
""")
    assert payload['oldUncertainMatchedBoth'] is True
    assert payload['uncertain'] == 'uncertain'
    assert payload['unrestorable'] == 'unrestorable'
    assert payload['categoryUncertain'] == 'uncertain'
    assert payload['categoryUnrestorable'] == 'unrestorable'
    assert payload['faintIgnored'] is None
    assert payload['edits'] == ['paint', 'erase']
    assert payload['batch'] == [1, 2]
    assert payload['focus'] == {'jobId': 'a', 'batchJobIds': ['a', 'b'], 'showQueue': True}
    assert node_json("""
const logic = require('./ui/region_logic.js');
console.log(JSON.stringify(logic.focusCreatedJobs([{id: 'only'}])));
""")['showQueue'] is False


def test_app_uses_shared_rules_for_the_audit_paths():
    source = (ROOT / 'ui' / 'app.js').read_text(encoding='utf-8')
    assert 'r > 150 && g < 50 && r - g > 120' not in source
    assert 'data-confirm-defect' not in source
    assert '纳入批次' in source
    assert 'window.MuralLogic.regionEdits' in source
    assert 'window.MuralLogic.batchRunIndexes' in source
    assert 'window.MuralLogic.focusCreatedJobs' in source
    assert "state.damageItems.map((_, index) => index)" not in source
    html = (ROOT / 'ui' / 'index.html').read_text(encoding='utf-8')
    assert 'region_logic.js' in html
    assert 'id="job-queue"' in html
    assert 'id="risk-banner"' in html
