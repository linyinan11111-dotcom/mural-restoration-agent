"""Verify an already selected real task through the running HTTP service."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile
import httpx
import numpy as np
from PIL import Image


def main():
    with httpx.Client(base_url='http://127.0.0.1:8765', timeout=30) as client:
        response = client.get('/api/jobs')
        response.raise_for_status()
        selected = next((job for job in response.json() if job.get('selected_id')), None)
        if not selected:
            raise RuntimeError('Select a candidate in the UI before running this check.')
        response = client.get(f'/api/jobs/{selected["id"]}/export')
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            original = np.asarray(Image.open(BytesIO(archive.read('original/working.png'))))
            result = np.asarray(Image.open(BytesIO(archive.read('result.png'))))
            mask = np.asarray(Image.open(BytesIO(archive.read('mask.png')))) > 127
            assert np.array_equal(original[~mask], result[~mask])
            assert hashlib.sha256(archive.read('result.png')).hexdigest() == manifest['selected_candidate']['result_sha256']
            assert hashlib.sha256(archive.read('original/source.bin')).hexdigest() == manifest['image']['original_sha256']
            report = {'job_id': selected['id'], 'tool': manifest['selected_candidate']['tool'],
                      'device': manifest['selected_candidate']['versions']['device'],
                      'original_bytes_preserved': True, 'outside_pixels_unchanged': True,
                      'candidate_elapsed_seconds': manifest['selected_candidate']['elapsed_seconds'],
                      'archive_entries': archive.namelist()}
    folder = Path('artifacts')
    folder.mkdir(exist_ok=True)
    (folder / 'verified-restoration.zip').write_bytes(response.content)
    (folder / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

