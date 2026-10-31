"""Download public model artifacts at a pinned revision and record SHA-256."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import httpx
import shutil

ROOT = Path(__file__).resolve().parent.parent
SPECS = {
 'sam2': ('facebook/sam2.1-hiera-tiny', 'de431c4043854a71d8101e17995dfe596bf101a5', 'sam2', 'Apache-2.0'),
 'diffusion': ('stable-diffusion-v1-5/stable-diffusion-inpainting', '8a4288a76071f7280aedbdb3253bdb9e9d5d84bb', 'sd-inpainting', 'CreativeML OpenRAIL-M'),
 'controlnet': ('lllyasviel/control_v11p_sd15_inpaint', 'c96e03a807e64135568ba8aecb66b3a306ec73bd', 'controlnet-inpaint', 'OpenRAIL; review the pinned repository model card'),
 'vision': ('Qwen/Qwen3-VL-2B-Instruct', '89644892e4d85e24eaac8bacfd4f463576704203', 'qwen-vl', 'Apache-2.0'),
}


def download_file(url, path, expected_hash=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected_hash or digest(path) == expected_hash):
        return
    temporary = path.with_suffix(path.suffix+'.part')
    if expected_hash:
        with httpx.Client(follow_redirects=True, timeout=40) as client:
            head = client.head(url)
            head.raise_for_status()
            total = int(head.headers.get('content-length', 0))
        if total > 128*1024**2 and head.headers.get('accept-ranges') == 'bytes':
            # Some regional proxies stall on multi-GB streams; bounded ranges are retryable.
            chunk_size = 16*1024**2
            with temporary.open('wb') as stream:
                stream.truncate(total)
            def part(start):
                end = min(total-1, start+chunk_size-1)
                for retry in range(4):
                    try:
                        with httpx.Client(follow_redirects=True, timeout=90) as client:
                            response = client.get(url, headers={'Range': f'bytes={start}-{end}'})
                            response.raise_for_status()
                        if response.status_code != 206 or len(response.content) != end-start+1 or not response.headers.get('content-range', '').startswith(f'bytes {start}-{end}/'):
                            raise ValueError('Invalid range response')
                        with temporary.open('r+b') as stream:
                            stream.seek(start)
                            stream.write(response.content)
                        return
                    except (httpx.HTTPError, ValueError):
                        if retry == 3:
                            raise
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(part, range(0, total, chunk_size)))
            if digest(temporary) != expected_hash:
                raise ValueError('SHA-256 mismatch: '+path.name)
            temporary.replace(path)
            label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            print('Downloaded and hash-verified '+label, flush=True)
            return
    for attempt in range(3):
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=180)) as client:
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_bytes(1024*1024):
                            stream.write(chunk)
            if expected_hash and digest(temporary) != expected_hash:
                raise ValueError('SHA-256 mismatch: '+path.name)
            temporary.replace(path)
            label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            print('Downloaded '+label, flush=True)
            return
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_model(name):
    repo, revision, folder, license_name = SPECS[name]
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        response = client.get('https://huggingface.co/api/models/'+repo, params={'blobs': 'true'})
        response.raise_for_status()
        info = response.json()
    if info.get('sha') != revision:
        raise RuntimeError(f'{repo} 当前 revision 已变化；请先人工复核后更新 SPECS。')
    files = []
    for entry in info['siblings']:
        filename = entry['rfilename']
        if filename.endswith(('.json', '.txt', '.md')) or 'LICENSE' in filename.upper():
            files.append(entry)
        elif filename.endswith('.safetensors'):
            if name == 'diffusion' and '.fp16.' not in filename:
                continue
            if name == 'controlnet' and '.fp16.' not in filename:
                continue
            files.append(entry)
    destination = ROOT / 'models' / folder
    def one(entry):
        filename = entry['rfilename']
        path = destination / filename
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Model repository returned an unsafe file path.')
        expected = entry.get('lfs', {}).get('sha256')
        download_file(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', path, expected)
        return {'file': filename, 'sha256': digest(path), 'bytes': path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(one, files))
    ledger = {'repository': repo, 'revision': revision, 'license': license_name,
              'model_card_license': info.get('cardData', {}).get('license'), 'files': manifest}
    (destination/'provenance.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verified '+name+' @ '+revision, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', choices=list(SPECS))
    parser.add_argument('--lpips', action='store_true')
    args = parser.parse_args()
    for name in args.models:
        download_model(name)
    if args.lpips:
        import torch
        cache_path = Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
        project_path = ROOT/'models/lpips/alexnet-owt-7be5be79.pth'
        if cache_path.exists() and not project_path.exists():
            project_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, project_path)
        download_file('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', project_path)
        checksum = digest(project_path)
        if not checksum.startswith('7be5be79'):
            raise ValueError('LPIPS backbone SHA-256 prefix mismatch.')
        (project_path.parent/'provenance.json').write_text(json.dumps({
            'repository': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
            'sha256': checksum, 'license': 'PyTorch pretrained weights; see source distribution terms',
            'bytes': project_path.stat().st_size}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Verified LPIPS AlexNet SHA256 '+checksum, flush=True)


if __name__ == '__main__':
    main()
