"""Dataset provenance, group-level splitting, and paired evaluation manifests."""
from pathlib import Path
import hashlib
import json
import random
from PIL import Image, ImageDraw
import numpy as np
from .storage import new_id, timestamp
from .workspace import add_image
from .imaging import sha256, image_array, mask_array, decode_image
from .quality import paired_metrics


def import_manifest(store, path):
    path = Path(path).resolve()
    entries = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    results = []
    root = path.parent.resolve()
    resolved = []
    for entry in entries:
        if any(not entry.get(k) for k in ('image', 'group', 'source', 'license', 'category')):
            raise ValueError('每行需要 image、group、source、license、category。')
        if entry['category'] not in {'real_damage', 'synthetic_paired', 'registered_paired'}:
            raise ValueError('category 必须是 real_damage、synthetic_paired 或 registered_paired。')
        files = {}
        for key in ('image', 'mask', 'target'):
            if entry.get(key):
                candidate = (root / entry[key]).resolve()
                if not candidate.is_relative_to(root) or not candidate.is_file():
                    raise ValueError('数据文件不存在或路径超出清单目录：'+str(entry[key]))
                files[key] = candidate
        if entry['category'] != 'real_damage' and not (entry.get('target') and entry.get('mask')):
            raise ValueError('配对样本必须提供 target 和 mask。')
        payload = files['image'].read_bytes()
        image = decode_image(payload)
        if entry.get('sha256') and hashlib.sha256(payload).hexdigest().lower() != str(entry['sha256']).lower():
            raise ValueError('原图 SHA-256 不匹配：'+str(entry['image']))
        if 'target' in files:
            target = decode_image(files['target'].read_bytes())
            mask = decode_image(files['mask'].read_bytes(), mask=True)
            if target.size != image.size or mask.size != image.size:
                raise ValueError('配对图像、目标和掩膜必须已经对齐且尺寸一致。')
            if not (np.asarray(mask) > 127).any():
                raise ValueError('配对掩膜不能为空。')
        resolved.append((entry, files))

    # Only write records after every manifest entry has passed validation.
    for entry, files in resolved:
        payload = files['image'].read_bytes()
        image = add_image(store, payload, Path(entry['image']).name)
        identifier = new_id()
        folder = store.folder(identifier)
        for key in ('target', 'mask'):
            if key in files:
                decode_image(files[key].read_bytes(), mask=(key == 'mask')).save(folder/(key+'.png'))
        metadata = {k: v for k, v in entry.items() if k not in {'image', 'target', 'mask'}}
        results.append(store.put('dataset', {'id': identifier, 'image_id': image['id'], 'metadata': metadata,
            'sha256': image['working_sha256'], 'paired': bool(entry.get('target')), 'created_at': timestamp()}))
    return results


def grouped_split(records, seed=42, fractions=(.7, .15, .15)):
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions)-1) > 1e-6:
        raise ValueError('集合比例须为三个非负数且和为 1。')
    # Merge groups with identical image hashes so duplicates cannot leak across splits.
    parent = {r['metadata']['group']: r['metadata']['group'] for r in records}
    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    hashes = {}
    for record in records:
        group = record['metadata']['group']
        if record['sha256'] in hashes:
            parent[root(group)] = root(hashes[record['sha256']])
        hashes[record['sha256']] = group
    groups = sorted({root(g) for g in parent})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    a, b = round(n*fractions[0]), round(n*(fractions[0]+fractions[1]))
    assignment = {g: 'train' if i < a else 'validation' if i < b else 'test' for i, g in enumerate(groups)}
    return {'seed': seed, 'fractions': fractions, 'group_count': n,
            'notice': '按壁画分组并合并完全重复图；仍需人工检查近重复照片及相邻裁片。',
            'splits': {name: [r['id'] for r in records if assignment[root(r['metadata']['group'])] == name]
                       for name in ('train', 'validation', 'test')}}


def synthetic_damage(image, seed=42, kind='crack', fraction=.08):
    if kind not in {'crack', 'hole', 'loss'} or not .001 <= fraction <= .5:
        raise ValueError('不支持的合成损坏参数。')
    randomizer = random.Random(seed)
    h, w = image.shape[:2]
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if kind == 'crack':
        for i in range(3):
            points = [(randomizer.randint(w//8, max(w//8, w*7//8)), int(h*j/7)) for j in range(8)]
            draw.line(points, fill=255, width=max(1, int(min(w, h)*fraction/6)))
    else:
        size = max(2, round((h*w*fraction)**.5))
        x, y = randomizer.randrange(max(1, w-size)), randomizer.randrange(max(1, h-size))
        if kind == 'hole':
            draw.ellipse((x, y, min(w-1, x+size), min(h-1, y+size)), fill=255)
        else:
            draw.polygon([(x, y), (x+size, y+size//4), (x+size*3//4, y+size), (x+size//5, y+size*4//5)], fill=255)
    array = np.asarray(mask).copy()
    damaged = image.copy()
    damaged[array > 0] = [224, 211, 179]
    return damaged, array, {'seed': seed, 'kind': kind, 'requested_fraction': fraction,
                           'actual_fraction': float((array > 0).mean()), 'historical_ground_truth': False}


def evaluate_pair(store, candidate, target_path, perceptual=False):
    output = image_array(store.folder(candidate['id'])/'result.png')
    target = image_array(target_path)
    mask = mask_array(store.folder(candidate['mask_id'])/'mask.png')
    metrics = paired_metrics(target, output, mask, perceptual)
    return {'candidate_id': candidate['id'], 'target_sha256': sha256(Path(target_path)),
            'created_at': timestamp(), 'metrics': metrics,
            'alignment': 'user-declared aligned comparison; not a historical authenticity test'}
