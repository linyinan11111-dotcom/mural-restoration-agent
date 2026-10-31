"""Equal-candidate-budget fixed vs feedback evaluation; ground truth never selects candidates."""
from pathlib import Path
from io import BytesIO
import argparse
import csv
import json
import sys
import time
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mural_agent.config import Settings
from mural_agent.storage import Store
from mural_agent.engine import Engine
from mural_agent import workspace, datasets
from mural_agent.imaging import image_array
from mural_agent.quality import paired_metrics


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, 'PNG')
    return buffer.getvalue()


def run(manifest, output, limit, perceptual):
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    settings.data_dir = output/'workspace'
    settings.api_key = ''
    store = Store(settings.data_dir)
    engine = Engine(settings, store)
    rows = []
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()][:limit]
    try:
        for index, record in enumerate(records):
            target = image_array(manifest.parent/record['image'])
            for kind in ('crack', 'hole', 'loss'):
                damaged, mask, parameters = datasets.synthetic_damage(target, seed=42+index, kind=kind)
                image_record = workspace.add_image(store, png(damaged), f'{index}-{kind}.png')
                mask_record = workspace.add_mask(store, image_record, png(mask))
                for treatment, revisions in [('fixed_tools', 0), ('feedback_agent', 2)]:
                    started = time.monotonic()
                    job = workspace.create_job(store, engine, image_record, mask_record,
                        'lama' if engine.lama.available() else 'opencv', 'rules', 3, '',
                        budget={'candidates': 3, 'revisions': revisions, 'timeout_seconds': 300})
                    while True:
                        job = store.get('job', job['id'])
                        if job['status'] not in {'queued', 'planning', 'running', 'evaluating'}:
                            break
                        time.sleep(.02)
                    row = {'source': record['source'], 'group': record['group'], 'kind': kind, 'treatment': treatment,
                           'job_id': job['id'], 'status': job['status'], 'candidate_budget': 3,
                           'elapsed_seconds': round(time.monotonic()-started, 3), 'synthetic': parameters}
                    if job['status'] == 'review':
                        candidate = store.get('candidate', job['recommended_id'])
                        output_array = image_array(store.folder(candidate['id'])/'result.png')
                        row.update(tool=candidate['tool'], attempts=len(job['attempts']), revisions=len(job['revisions']),
                            engineering=candidate['metrics'], paired=paired_metrics(target, output_array, mask, perceptual))
                        store.update('job', job['id'], selected_id=candidate['id'], review={'note': 'Benchmark heuristic selection; no expert review', 'time': job['updated_at']})
                        engine.export(job['id'])
                    rows.append(row)
                    (output/'results.json').write_text(json.dumps({'protocol': 'Same frozen original and mask; three-candidate cap; selection uses engineering checks only; target used only after selection.',
                        'scope': 'Synthetic added damage on museum fresco photographs. No historical ground truth, no expert evaluation, no claims of Agent superiority.',
                        'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(f'{index+1}/{len(records)} {kind} {treatment}: {job["status"]}', flush=True)
        # Anonymous reviewer form; ratings remain empty until a human supplies them.
        with (output/'blind-review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['case_id', 'line_continuity', 'brushwork', 'tone', 'symbol_accuracy', 'evidence_support', 'reviewer', 'notes'])
            for i, row in enumerate(rows):
                writer.writerow([f'case-{i+1:04}', '', '', '', '', '', '', ''])
        (output/'blind-key.json').write_text(json.dumps({f'case-{i+1:04}': row['job_id'] for i, row in enumerate(rows)}, indent=2), encoding='utf-8')
        blind = output/'blind-images'
        blind.mkdir(exist_ok=True)
        import shutil
        for i, row in enumerate(rows):
            job = store.get('job', row['job_id'])
            if job.get('selected_id'):
                shutil.copy2(store.folder(job['selected_id'])/'result.png', blind/f'case-{i+1:04}.png')
    finally:
        engine.executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT/'datasets/public-fresco-smoke/images.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--lpips', action='store_true')
    args = parser.parse_args()
    run(args.manifest, args.output, args.limit, args.lpips)
