"""Five CC0 museum fresco photographs; smoke corpus, not a professional benchmark."""
from pathlib import Path
import json
import sys
import httpx
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mural_agent.imaging import sha256


def main():
    folder = ROOT/'datasets/public-fresco-smoke'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for identifier in (138881, 152270, 152259, 108532, 150420):
            response = client.get(f'https://openaccess-api.clevelandart.org/api/artworks/{identifier}')
            response.raise_for_status()
            obj = response.json()['data']
            if obj.get('share_license_status') != 'CC0' or 'fresco' not in obj['technique'].lower():
                raise ValueError('Unexpected license or medium')
            image_url = obj['images']['web']['url']
            path = folder/f'{identifier}.jpg'
            if not path.exists():
                image = client.get(image_url)
                image.raise_for_status()
                path.write_bytes(image.content)
            (folder/f'{identifier}.metadata.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            records.append({'image': path.name, 'group': 'cma-'+str(obj['accession_number']),
                'source': f'https://www.clevelandart.org/art/{obj["accession_number"]}', 'image_url': image_url,
                'license': 'CC0 per Cleveland Museum of Art API', 'title': obj['title'],
                'category': 'real_damage', 'sha256': sha256(path),
                'notice': 'Museum fresco photograph, not an intact historical ground truth. No expert damage labels.'})
    (folder/'images.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records)+'\n', encoding='utf-8')
    print('Downloaded '+str(len(records))+' documented fresco photographs to '+str(folder))


if __name__ == '__main__':
    main()
