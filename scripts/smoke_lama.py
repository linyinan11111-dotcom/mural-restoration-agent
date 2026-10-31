"""Real model inference check, independent of any test doubles or API keys."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from mural_agent.config import Settings
from mural_agent.models import LamaBackend
from mural_agent.imaging import composite, evaluate, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.device = args.device
    backend = LamaBackend(settings)
    image = np.full((224, 320, 3), [179, 143, 98], dtype=np.uint8)
    image[70:150, :, :] = [100, 126, 108]
    image[:, 150:155] = [240, 228, 204]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[15:205, 148:157] = 255
    start = time.perf_counter()
    prediction = backend.predict(image, mask)
    result = composite(image, prediction, mask)
    metrics = evaluate(image, result, mask)
    assert metrics['outside_unchanged']
    assert metrics['changed_pixels'] > 0
    folder = Path('artifacts')
    folder.mkdir(exist_ok=True)
    Image.fromarray(image).save(folder / 'smoke-original.png')
    Image.fromarray(mask).save(folder / 'smoke-mask.png')
    Image.fromarray(result).save(folder / 'smoke-result.png')
    import torch
    record = {'device': str(backend.device), 'torch': torch.__version__, 'seconds': round(time.perf_counter()-start, 3),
              'shape': list(result.shape), 'metrics': metrics, 'weight_sha256': sha256(settings.lama_model),
              'gpu': torch.cuda.get_device_name(0) if backend.device.type == 'cuda' else None}
    (folder / f'lama-smoke-{backend.device.type}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()

