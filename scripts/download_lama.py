"""Download the trusted IOPaint-distributed TorchScript LaMa checkpoint."""
from pathlib import Path
import argparse
import hashlib
import urllib.request

URL = 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='models/big-lama.pt')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f'Checkpoint already exists: {output}')
        return
    temporary = output.with_suffix('.download')
    request = urllib.request.Request(URL, headers={'User-Agent': 'mural-agent/0.1'})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as target:
        total = int(source.headers.get('Content-Length', '0'))
        received = 0
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if received % (32 * 1024 * 1024) < 1024 * 1024:
                print(f'{received // (1024*1024)} / {total // (1024*1024)} MB', flush=True)
    if temporary.stat().st_size < 1_000_000:
        raise RuntimeError('Downloaded file is too small to be a valid model.')
    temporary.replace(output)
    print(f'Saved: {output}\nSHA256: {digest.hexdigest()}')
    print('Source: ' + URL)
    print('Recorded hash establishes the downloaded artifact identity; it is not an independent authenticity signature.')


if __name__ == '__main__':
    main()

