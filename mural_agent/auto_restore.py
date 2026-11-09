"""One-click autonomous restoration proposal.

The local vision model inspects the whole image, proposes damage boxes, the
pipeline refines each box into a mask, unions the accepted regions, routes each
region to a conservative repair tool and starts one bounded job. Nothing here is
an authority: every box, mask and tool choice is stored on the job/archive so a
human can inspect, edit the mask and re-run, or discard the result.

In ``professional_step`` workflow, only candidate regions/proposals are produced;
no merged mask is auto-confirmed and no repair job is enqueued.
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


def context_instructions(context):
    """Pass unverified visual context to the repair planner as cautious constraints."""
    if not isinstance(context, dict):
        return ''
    lines = ['文化背景为 Qwen 视觉假设，未经文献或专家核实；不得作为历史事实。']
    for key, label in (('visible_subject', '可见题材'),
                       ('possible_period_or_tradition', '可能年代/传统（待核）')):
        value = str(context.get(key, '')).strip()
        if value:
            lines.append(f'{label}：{value}')
    for key, label in (('visual_clues', '可见线索'),
                       ('preservation_constraints', '保护约束'),
                       ('verification_needed', '待查证')):
        values = context.get(key) or []
        if isinstance(values, list) and values:
            lines.append(f'{label}：' + '；'.join(str(value) for value in values[:5]))
    lines.append('不得擦除或臆造人物、题记、宗教符号、纹样及颜料边界；证据不足时只做最小保守修补。')
    return '\n'.join(lines)[:2000]


def _collect_proposals(store, engine, image_record, min_confidence=MIN_CONFIDENCE, max_items=MAX_ITEMS):
    """Shared assess → refine → merge path used by auto and professional propose."""
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
    proposals = []
    defects = []
    for index, item_index in enumerate(accepted):
        proposal_id = new_id()
        folder = store.folder(proposal_id)
        Image.fromarray(masks[item_index]).save(folder / 'mask.png')
        Image.fromarray(masks[item_index]).save(folder / 'proposal.png')
        meta = dict(metadata_list[item_index])
        meta['recommendation'] = dict(accepted_routes[index])
        method = str(meta.get('method', '')).lower()
        if accepted_items[index].get('type') == 'crack' and 'fallback' not in method:
            meta['mask_source'] = 'crack'
        elif 'sam' in method:
            meta['mask_source'] = 'sam'
        else:
            meta['mask_source'] = 'ai_candidate'
        meta['confirmed'] = False
        # Heuristic risk class per proposal (label and gate, do not auto-generate).
        try:
            from .risk_regions import classify_mask_regions, aggregate_region_class
            risk = classify_mask_regions(image, masks[item_index])
            meta['region_class'] = risk['region_class']
            meta['risk_regions'] = risk['regions']
            meta['risk_notice'] = risk['notice']
            meta['repair_allowed'] = not risk['blocking']
        except Exception as exc:  # noqa: BLE001 — keep proposal even if heuristic fails
            meta['region_class'] = 'uncertain'
            meta['risk_regions'] = []
            meta['risk_notice'] = f'风险分类失败，默认标不确定：{str(exc)[:120]}'
            meta['repair_allowed'] = False
        record = store.put('proposal', {'id': proposal_id, 'image_id': image_record['id'],
                                        'created_at': timestamp(), 'metadata': meta,
                                        'adopted': False, 'confirmed': False,
                                        'mask_source': meta['mask_source'],
                                        'region_class': meta.get('region_class', 'uncertain'),
                                        'repair_allowed': bool(meta.get('repair_allowed', False))})
        proposal_ids.append(proposal_id)
        proposals.append(record)
        defect = dict(accepted_items[index])
        defect['proposal_id'] = proposal_id
        defect['recommendation'] = dict(accepted_routes[index])
        defect['repair_allowed'] = bool(meta.get('repair_allowed', False))
        defect['region_class'] = meta.get('region_class', 'uncertain')
        defect['risk_notice'] = meta.get('risk_notice', '') if not meta.get('repair_allowed') else ''
        defect['confirmed'] = False
        defects.append(defect)

    paired = {(item.get('type'), tuple(item.get('bbox') or [])) for item in accepted_items}
    omitted = {}
    for item in assessment.get('items', []):
        key = (item.get('type'), tuple(item.get('bbox') or []))
        if key in paired or key in omitted:
            continue
        if key not in {(candidate.get('type'), tuple(candidate.get('bbox') or [])) for candidate in selected}:
            omitted[key] = '置信度或区域尺寸未达到自动细化门槛'
        else:
            omitted[key] = '独立掩膜未通过面积限制，需手工确认选区'
    for item in assessment.get('items', []):
        key = (item.get('type'), tuple(item.get('bbox') or []))
        if key not in omitted:
            continue
        defect = dict(item)
        defect['proposal_id'] = None
        defect['recommendation'] = route_items([item], available)[0]
        defect['repair_allowed'] = False
        defect['region_class'] = 'uncertain'
        defect['risk_notice'] = omitted.pop(key)
        defect['confirmed'] = False
        defects.append(defect)

    fraction = float((union > 0).mean())
    if fraction < MIN_MASK_FRACTION:
        raise ValueError('AI 候选掩膜过小（低于图像 0.05%），未自动提交；请改用手动画选。')
    summary = summary_for(accepted_items, accepted_routes, fraction, backend)
    cultural_context = assessment.get('cultural_context') or {}
    return {
        'assessment': assessment,
        'cultural_context': cultural_context,
        'planning_notes': context_instructions(cultural_context),
        'limitations': limitations,
        'accepted_items': accepted_items,
        'accepted_routes': accepted_routes,
        'backend': backend,
        'steps': steps,
        'proposal_ids': proposal_ids,
        'proposals': proposals,
        'defects': defects,
        'masks': masks,
        'accepted': accepted,
        'union': union,
        'fraction': fraction,
        'summary': summary,
        'available': available,
    }


def propose_restore(store, engine, image_record, min_confidence=MIN_CONFIDENCE, max_items=MAX_ITEMS):
    """Professional step path: list refined candidates for expert confirmation only."""
    collected = _collect_proposals(store, engine, image_record, min_confidence, max_items)
    # Unconfirmed preview mask for UI overlay; cannot create_job until user confirms.
    from .risk_regions import classify_mask_regions, aggregate_region_class
    image = None
    try:
        from .imaging import image_array
        image = image_array(store.folder(image_record['id']) / 'image.png')
        risk = classify_mask_regions(image, collected['union'])
        region_class = risk['region_class']
        risk_regions = risk['regions']
        risk_notice = risk['notice']
    except Exception as exc:  # noqa: BLE001
        region_class = aggregate_region_class(
            [p.get('region_class') for p in collected['proposals']])
        risk_regions = []
        risk_notice = f'整体风险分类降级：{str(exc)[:120]}'
        risk = {'blocking': region_class in {'uncertain', 'unrestorable'}, 'class_masks': {}}
    mask_record = workspace.add_mask(
        store, image_record, png_bytes(collected['union']), collected['proposal_ids'],
        mask_source='ai_candidate', user_confirmed=False,
        region_class=region_class, risk_regions=risk_regions, expert_force_override=False)
    summary = ('专业分步：仅产出候选选区与风险标红，未自动执行修复，禁止生成纹样/图像。'
               '请逐块「确认可填 / 标不确定 / 拒答跳过」后再决定是否保守填补。'
               + ' ' + collected['summary'] + ' ' + risk_notice)
    return {
        'mask': mask_record,
        'job': None,
        'items': collected['accepted_items'],
        'defects': collected['defects'],
        'routes': collected['accepted_routes'],
        'steps': collected['steps'],
        'backend': collected['backend'],
        'summary': summary,
        'cultural_context': collected['cultural_context'],
        'planning_notes': collected['planning_notes'],
        'proposals': collected['proposals'],
        'workflow': 'professional_step',
        'awaiting_confirmation': True,
        'risk': {
            'region_class': region_class,
            'regions': risk_regions,
            'blocking': bool(risk.get('blocking')),
            'notice': risk_notice,
        },
    }


def auto_restore(store, engine, image_record, min_confidence=MIN_CONFIDENCE, max_items=MAX_ITEMS,
                 workflow='standard'):
    """Run autonomous proposal; professional_step stops before job creation."""
    from .professional import WORKFLOW_PROFESSIONAL, normalize_workflow
    workflow = normalize_workflow(workflow)
    if workflow == WORKFLOW_PROFESSIONAL:
        return propose_restore(store, engine, image_record, min_confidence, max_items)

    collected = _collect_proposals(store, engine, image_record, min_confidence, max_items)
    mask_record = workspace.add_mask(
        store, image_record, png_bytes(collected['union']), collected['proposal_ids'],
        mask_source='ai_candidate', user_confirmed=True)
    summary = collected['summary']
    job = workspace.create_job(
        store, engine, image_record, mask_record, collected['backend'], 'auto',
        len(collected['steps']), summary, mode='conservative',
        budget={'candidates': len(collected['steps']), 'revisions': 1, 'timeout_seconds': 300},
        auto={'items': collected['accepted_items'], 'routes': collected['accepted_routes'],
              'steps': collected['steps'], 'backend': collected['backend'],
              'mask_fraction': collected['fraction'],
              'cultural_context': collected['cultural_context'],
              'planning_notes': collected['planning_notes'],
              'limitations': collected['limitations'][:1000], 'summary': summary},
        workflow='standard', rationale=summary, evidence_level='conservative',
        expert_notes=collected['planning_notes'], mask_source='ai_candidate')
    return {'mask': mask_record, 'job': job, 'items': collected['accepted_items'],
            'defects': collected['defects'],
            'routes': collected['accepted_routes'], 'steps': collected['steps'],
            'backend': collected['backend'], 'summary': summary,
            'cultural_context': collected['cultural_context'],
            'planning_notes': collected['planning_notes']}
