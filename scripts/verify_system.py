"""Real model and native UI acceptance; writes evidence, no GUI automation of user windows."""
from pathlib import Path
from io import BytesIO
import argparse
import json
import os
import sys
import time
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication
from mural_agent.config import Settings
from mural_agent.desktop import MainWindow, configure_application
from mural_agent import workspace, datasets, references
from mural_agent.imaging import image_array, mask_array
from mural_agent.quality import paired_metrics


def png(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, 'PNG')
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/system-acceptance')
    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--controlnet', action='store_true')
    parser.add_argument('--lpips', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    settings.data_dir = args.output/'workspace'
    settings.api_key = ''
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    window = MainWindow(settings)
    window.resize(1480, 980)
    window.show()
    app.processEvents()
    photograph = ROOT/'datasets/public-fresco-smoke/152270.jpg'
    target = image_array(photograph)
    damaged, mask, params = datasets.synthetic_damage(target, kind='loss', seed=7, fraction=.025)
    image = workspace.add_image(window.store, png(damaged), '馆藏壁画照片＋合成缺损（测试）.png')
    saved_mask = workspace.add_mask(window.store, image, png(mask))
    window.load_image(image)
    window.canvas.import_mask(png(mask))
    reports = {'python': sys.executable, 'sys_path': sys.path, 'availability': window.engine.tools.availability(),
               'source': 'Cleveland Museum of Art 152270, CC0', 'synthetic_added_damage': params,
               'tests': {}}
    def record(name, payload):
        reports['tests'][name] = payload
        (args.output/'report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    def execute(backend, mode='conservative', registration=None, control=False, planner='rules'):
        job = workspace.create_job(window.store, window.engine, image, saved_mask, backend, planner, 1,
            'Restore only the selected loss; preserve aged mineral colors and painted lines.', mode=mode,
            registration_id=registration, structure_control=control,
            budget={'candidates': 1, 'revisions': 0, 'timeout_seconds': 300})
        end = time.monotonic()+330
        while time.monotonic() < end:
            app.processEvents()
            job = window.store.get('job', job['id'])
            if job['status'] not in {'queued', 'planning', 'running', 'evaluating'}:
                break
            time.sleep(.05)
        if job['status'] != 'review':
            record(backend+('_control' if control else ''), {'passed': False, 'job': job})
            raise RuntimeError('Acceptance failed: '+backend+' '+str(job.get('error')))
        candidate = job['candidates'][0]
        assert candidate['metrics']['outside_unchanged']
        output = image_array(window.store.folder(candidate['id'])/'result.png')
        payload = {'passed': True, 'job_id': job['id'], 'planner_source': job['plan']['source'], 'candidate': candidate,
                   'paired': paired_metrics(target, output, mask, args.lpips)}
        window.job = job
        window.poll_job()
        window.candidates.setCurrentRow(0)
        window.review_note.setText('自动化工程验收；参考和损坏均为构造测试，不是历史复原结论。')
        window.select_candidate()
        payload['archive'] = str(window.engine.export(job['id']))
        app.processEvents()
        window.tabs.setCurrentIndex(1)
        window.grab().save(str(args.output/(backend+('_control' if control else '')+'.png')))
        record(backend+('_control' if control else '')+('_vision' if planner == 'local' else ''), payload)
        return candidate
    try:
        execute('lama')
        ys, xs = np.where(mask > 0)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        proposed, meta = window.engine.sam.predict(damaged, points=[[float(xs.mean()), float(ys.mean())]], labels=[1], box=box)
        assert proposed.shape == mask.shape and np.count_nonzero(proposed) > 0
        window.show_proposal((proposed, meta))
        window.tabs.setCurrentIndex(2)
        app.processEvents()
        window.grab().save(str(args.output/'assistance.png'))
        record('sam2', {'passed': True, 'mask_pixels': int((proposed > 0).sum()), 'metadata': meta})
        # Same photo before artificial damage is a controlled registration fixture.
        metadata = {'source': 'CMA CC0 image before synthetic damage', 'license': 'CC0',
                    'group': 'cma-152270', 'relation': 'same_mural', 'note': 'Synthetic reference fixture, not a historical photograph.'}
        ref = references.add_reference(window.store, photograph.read_bytes(), '人工损坏前测试参考', metadata)
        reg = references.register(window.store, image, ref, mask)
        reg = references.approve_registration(window.store, reg['id'], png(mask), 'Controlled synthetic fixture exact correspondences')
        window.show_registration(reg)
        execute('reference', mode='evidence', registration=reg['id'])
        if args.diffusion:
            execute('diffusion', mode='speculative')
        if args.controlnet:
            execute('diffusion', mode='speculative', control=True)
        if args.vision:
            execute('lama', planner='local')
            assert reports['tests']['lama_vision']['planner_source'] == 'local_vlm', 'Local VLM fell back to rules'
        reports['completed'] = True
        (args.output/'report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
        print('System acceptance passed: '+str(args.output/'report.json'))
    finally:
        window.timer.stop()
        window.close()
        window.engine.executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
