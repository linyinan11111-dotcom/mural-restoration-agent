"""One-click autonomous restoration proposal.

The local vision model inspects the whole image, proposes damage boxes, the
pipeline refines each box into a mask, unions the accepted regions, routes each
region to a conservative repair tool and starts one bounded job. Nothing here is
an authority: every box, mask and tool choice is stored on the job/archive so a
human can inspect, edit the mask and re-run, or discard the result.
"""
from io import BytesIO

import numpy as np
from PIL import Image

from .damage_assessment import assess_image, recommendation_for
from .imaging import image_array
from .segmentation import crack_mask_proposal
from .storage import new_id, timestamp
from . import workspace

MAX_ITEMS = 5
MAX_MASK_FRACTION = 0.25
MIN_CONFIDENCE = 0.35
MIN_BOX_FRACTION = 0.0005    # a 0.05% box is almost certainly noise/text, not damage
MIN_MASK_FRACTION = 0.0005   # refuse to submit a repair that touches almost nothing
STEP_LIMIT = 3
LABELS = {'crack': '裂缝', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
          'flaking': '起甲/剥落', 'stain': '污渍/变色', 'other': '其他疑似缺损'}
# Tie-breaking order when several tools receive the same weight.
TOOL_ORDER = {'lama': 0, 'telea': 1, 'ns': 2, 'color': 3, 'denoise': 4, 'deblur': 5}


def box_fraction(item):
    bbox = item.get('bbox') or []
    if len(bbox) != 4:
        return 0.0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1])) / 1_000_000


def select_items(items, min_confidence=MIN_CONFIDENCE, max_items=MAX_ITEMS):
    """Keep the most confident, plausibly sized candidates; duplicates are merged upstream."""
    selected = [item for item in items
                if float(item.get('confidence', 0)) >= min_confidence and box_fraction(item) >= MIN_BOX_FRACTION]
    selected.sort(key=lambda item: float(item.get('confidence', 0)), reverse=True)
    return selected[:max_items]


def validate_assessment(assessment):
    """Reject truncated VLM output instead of acting on half-parsed boxes."""
    limitations = str(assessment.get('limitations', ''))
    if '截断' in limitations or 'truncated' in limitations.lower():
        raise ValueError('AI 输出被截断，未自动提交修复任务；请重试，或改用手动画选。')
    return limitations


def route_items(items, available):
    """Map each item to a tool, downgrading LaMa to Telea when LaMa is unavailable."""
    routes = []
    for item in items:
        route = dict(recommendation_for(item))
        if route['tool'] == 'lama' and not available.get('lama', False):
            route = {'backend': 'telea', 'tool': 'telea',
                     'method': 'Telea 局部纹理传播（LaMa 未安装，已降级）'}
        routes.append(route)
    return routes


def choose_backend(routes):
    """Pick the backend that the majority of accepted regions are routed to."""
    scores = {}
    for route in routes:
        tool = route['tool']
        scores[tool] = scores.get(tool, 0) + (2 if tool == 'lama' else 1)
    if not scores:
        return 'telea'
    return max(scores, key=lambda tool: (scores[tool], -TOOL_ORDER.get(tool, 99)))


def build_steps(items, routes):
    """One candidate per distinct recommended tool, up to STEP_LIMIT."""
    steps = []
    seen = set()
    for item, route in zip(items, routes):
        tool = route['tool']
        if tool in seen:
            continue
        seen.add(tool)
        label = LABELS.get(item.get('type'), '疑似缺损')
        steps.append({'tool': tool,
                      'reason': f"AI 自主规划：{label}（{item.get('size', 'uncertain')}）→ {route['method']}"})
    if not steps:
        steps = [{'tool': 'telea', 'reason': 'AI 自主规划：无可路由候选，回退保守修补。'}]
    return steps[:STEP_LIMIT]


def merge_masks(masks, shape, max_fraction=MAX_MASK_FRACTION):
    """Union masks in order, skipping any that would push the edit region over the cap."""
    union = np.zeros(shape, dtype=np.uint8)
    accepted = []
    for index, mask in enumerate(masks):
        if mask is None or not (mask > 0).any():
            continue
        trial = union.copy()
        trial[mask > 0] = 255
        if (trial > 0).mean() > max_fraction:
            continue
        union = trial
        accepted.append(index)
    return union, accepted


def refine_item(engine, image, item):
    """Turn one VLM box into a mask; fall back to the box when SAM is unavailable."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [int(value) for value in item['bbox']]
    pad_x, pad_y = max(3, round(width * .005)), max(3, round(height * .005))
    box = [max(0, round(x0 * width / 1000) - pad_x), max(0, round(y0 * height / 1000) - pad_y),
           min(width, round(x1 * width / 1000) + pad_x), min(height, round(y1 * height / 1000) + pad_y)]
    if item.get('type') == 'crack':
        mask, metadata = crack_mask_proposal(image, box)
        if not (mask > 0).any():
            mask = box_mask(image.shape[:2], box)
            metadata = {'method': 'bounding_box_fallback', 'notice': '细线提取为空，已退回矩形框候选。'}
    else:
        try:
            mask, metadata = engine.sam.predict(image, box=box)
        except Exception as exc:
            mask = box_mask(image.shape[:2], box)
            metadata = {'method': 'bounding_box_fallback',
                        'notice': 'SAM 不可用或分割失败，已退回矩形框候选。',
                        'error': str(exc)[:200]}
    metadata = dict(metadata or {})
    metadata.update({key: item[key] for key in ('index', 'type', 'size', 'confidence', 'evidence', 'bbox')
                     if key in item})
    metadata['recommendation'] = recommendation_for(item)
    metadata['confirmed'] = False
    return mask, metadata


def box_mask(shape, box):
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = [int(value) for value in box]
    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    return mask


def png_bytes(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def summary_for(items, routes, fraction, backend):
    kinds = {}
    for item in items:
        label = LABELS.get(item.get('type'), '疑似缺损')
        kinds[label] = kinds.get(label, 0) + 1
    detail = '、'.join(f'{label}×{count}' for label, count in kinds.items())
    tools = ' / '.join(dict.fromkeys(route['tool'] for route in routes))
    return (f'AI 自主：识别 {len(items)} 处疑似缺损（{detail}），自动掩膜占 {fraction:.2%}，'
            f'建议 {tools}（主引擎 {backend}）。识别、掩膜与工具均为候选，须人工复核。')


def auto_restore(store, engine, image_record, min_confidence=MIN_CONFIDENCE, max_items=MAX_ITEMS):
    """Run the full autonomous proposal and create one bounded conservative job."""
    assessment = assess_image(engine, image_record)
    limitations = validate_assessment(assessment)
    selected = select_items(assessment.get('items', []), min_confidence, max_items)
    if not selected:
        raise ValueError('AI 未发现可确认的缺损候选；请手动画选，或换一张更清晰的图。')

    image = image_array(store.folder(image_record['id']) / 'image.png')
    available = engine.tools.availability()
    routes = route_items(selected, available)
    masks, metadata_list = [], []
    for item in selected:
        mask, metadata = refine_item(engine, image, item)
        masks.append(mask)
        metadata_list.append(metadata)
    union, accepted = merge_masks(masks, image.shape[:2])
    if not accepted:
        raise ValueError('AI 提议的缺损区域过大或过于分散（超过图像 25%），为避免覆盖整幅图已停止；请手动画选。')

    accepted_items = [selected[index] for index in accepted]
    accepted_routes = [routes[index] for index in accepted]
    backend = choose_backend(accepted_routes)
    if backend == 'lama' and not available.get('lama', False):
        backend = 'telea'
    steps = build_steps(accepted_items, accepted_routes)

    proposal_ids = []
    for index, item_index in enumerate(accepted):
        proposal_id = new_id()
        folder = store.folder(proposal_id)
        Image.fromarray(masks[item_index]).save(folder / 'mask.png')
        Image.fromarray(masks[item_index]).save(folder / 'proposal.png')
        store.put('proposal', {'id': proposal_id, 'image_id': image_record['id'],
                               'created_at': timestamp(), 'metadata': metadata_list[item_index],
                               'adopted': False, 'confirmed': False})
        proposal_ids.append(proposal_id)

    fraction = float((union > 0).mean())
    if fraction < MIN_MASK_FRACTION:
        raise ValueError('AI 候选掩膜过小（低于图像 0.05%），未自动提交；请改用手动画选。')
    mask_record = workspace.add_mask(store, image_record, png_bytes(union), proposal_ids)
    summary = summary_for(accepted_items, accepted_routes, fraction, backend)
    job = workspace.create_job(store, engine, image_record, mask_record, backend, 'auto', len(steps),
        summary, mode='conservative', budget={'candidates': len(steps), 'revisions': 1, 'timeout_seconds': 300},
        auto={'items': accepted_items, 'routes': accepted_routes, 'steps': steps,
              'backend': backend, 'mask_fraction': fraction,
              'limitations': limitations[:1000],
              'summary': summary})
    return {'mask': mask_record, 'job': job, 'items': accepted_items, 'routes': accepted_routes,
            'steps': steps, 'backend': backend, 'summary': summary}
