"""Application services shared by the native desktop interface and tests."""
from io import BytesIO
import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw
from .storage import new_id, timestamp
from .imaging import decode_image, sha256


def add_image(store, payload, name):
    image = decode_image(payload)
    identifier = new_id()
    folder = store.folder(identifier)
    (folder / 'source.bin').write_bytes(payload)
    image.save(folder / 'image.png')
    return store.put('image', {'id': identifier, 'name': name[:200], 'width': image.width,
                              'height': image.height, 'created_at': timestamp(),
                              'original_sha256': hashlib.sha256(payload).hexdigest(),
                              'working_sha256': sha256(folder / 'image.png'),
                              'color_note': '原始字节已保留；修复使用 EXIF 校正后的 8 位 RGB 工作副本。',
                              'url': f'/api/assets/{identifier}/image.png'})


def add_mask(store, image_record, payload, proposal_ids=None, mask_source='manual', user_confirmed=True,
             region_class='fillable_lacuna', risk_regions=None, expert_force_override=False,
             force_override_reason=''):
    from .professional import normalize_mask_source
    try:
        from .risk_regions import normalize_region_class
        klass = normalize_region_class(region_class)
    except Exception:
        klass = (region_class or 'fillable_lacuna')
    mask = decode_image(payload, mask=True)
    if mask.size != (image_record['width'], image_record['height']):
        raise ValueError('掩膜必须与工作图尺寸一致。')
    array = np.asarray(mask) > 127
    if not array.any():
        raise ValueError('请先用画笔圈选需要修复的区域。')
    if array.all():
        raise ValueError('请保留一部分未损坏区域作为参考，不能选择整幅图像。')
    identifier = new_id()
    path = store.folder(identifier) / 'mask.png'
    Image.fromarray(array.astype(np.uint8) * 255).save(path)
    source = normalize_mask_source(mask_source)
    return store.put('mask', {'id': identifier, 'image_id': image_record['id'], 'created_at': timestamp(),
                             'fraction': round(float(array.mean()), 6), 'sha256': sha256(path),
                             'proposal_ids': proposal_ids or [], 'user_confirmed': bool(user_confirmed),
                             'mask_source': source,
                             'region_class': klass,
                             'risk_regions': list(risk_regions or []),
                             'expert_force_override': bool(expert_force_override),
                             'force_override_reason': (force_override_reason or '')[:2000],
                             'url': f'/api/assets/{identifier}/mask.png'})


def create_job(store, engine, image_record, mask_record, backend, planner, count, instructions,
               mode='conservative', registration_id=None, budget=None, structure_control=False,
               quality_thresholds=None, auto=None, workflow='standard', rationale='',
               evidence_level=None, expert_notes='', mask_source=None, allow_enhanced=False,
               risk_acknowledged=False, mask_gate_confirmed=False, has_unrestorable=False,
               has_uncertain=False, force_override=False, force_override_reason='',
               contest_demo=None):
    from .planner import Budget
    from .imaging import mask_array
    from .professional import (WORKFLOW_PROFESSIONAL, normalize_workflow, normalize_mask_source,
                               validate_professional_create)
    available = engine.tools.availability()
    workflow = normalize_workflow(workflow)
    if backend == 'lama' and not engine.lama.available():
        raise ValueError('LaMa 权重或 PyTorch 未安装，请先配置模型。')
    if planner == 'api' and not engine.settings.api_ready:
        raise ValueError('请先在 API 设置中填写服务地址、视觉模型与密钥。')
    if planner == 'local' and not (engine.settings.vision_model/'provenance.json').exists():
        raise ValueError('本地视觉模型权重尚未安装。')
    if mask_record['image_id'] != image_record['id']:
        raise ValueError('掩膜不属于当前图像。')
    if planner == 'auto' and not (isinstance(auto, dict) and auto.get('steps')):
        raise ValueError('AI 自动任务缺少规划信息。')
    if backend not in {'opencv', 'telea', 'ns', 'lama', 'reference', 'diffusion', 'denoise', 'color', 'deblur', 'research'} or planner not in {'api', 'rules', 'local', 'auto'} or not 1 <= count <= 3:
        raise ValueError('不支持的修复配置。')
    if not available.get(backend):
        raise ValueError('所选工具尚未配置，请在模型与环境中查看安装状态。')
    if mode not in {'conservative', 'evidence', 'speculative', 'enhanced'}:
        raise ValueError('不支持的修复模式。')
    if (backend == 'reference') != (mode == 'evidence'):
        raise ValueError('证据辅助模式必须使用核验参考重建工具。')
    if mode == 'speculative' and backend != 'diffusion':
        raise ValueError('推测模式必须使用扩散补全。')
    if backend == 'diffusion' and mode not in {'speculative', 'enhanced'}:
        raise ValueError('扩散补全仅允许在推测或加强修复模式下使用。')
    if structure_control and (backend != 'diffusion' or not available['controlnet']):
        raise ValueError('结构控制需要已安装的 ControlNet 和扩散工具。')
    if registration_id:
        registration = store.get('registration', registration_id)
        if not registration or registration['image_id'] != image_record['id'] or not registration.get('approved'):
            raise ValueError('参考配准未核验或不属于当前壁画。')
    if mode == 'evidence':
        if not registration_id:
            raise ValueError('请先配准并核验参考资料。')
        approved = mask_array(store.folder(registration_id)/'approved.png')
        editable = mask_array(store.folder(mask_record['id'])/'mask.png')
        if ((editable > 0) & (approved == 0)).any():
            raise ValueError('修复区域超出已核验参考范围。')
    professional_meta = {}
    if workflow == WORKFLOW_PROFESSIONAL:
        # Prefer dedicated rationale; fall back to instructions for API callers.
        region_class = (mask_record.get('region_class') or '')
        if region_class == 'unrestorable':
            has_unrestorable = True
        if region_class == 'uncertain':
            has_uncertain = True
        if mask_record.get('expert_force_override'):
            force_override = True
            force_override_reason = force_override_reason or mask_record.get('force_override_reason') or ''
        professional_meta = validate_professional_create(
            mask_record,
            rationale=rationale or instructions,
            evidence_level=evidence_level,
            mode=mode,
            expert_notes=expert_notes,
            mask_source=mask_source or mask_record.get('mask_source'),
            allow_enhanced=allow_enhanced,
            risk_acknowledged=risk_acknowledged,
            mask_gate_confirmed=mask_gate_confirmed,
            has_unrestorable=has_unrestorable,
            has_uncertain=has_uncertain,
            force_override=force_override,
            force_override_reason=force_override_reason,
            contest_demo=contest_demo,
        )
    elif evidence_level:
        # Allow optional provenance fields on standard jobs without full gates.
        from .professional import normalize_evidence_level
        professional_meta = {
            'rationale': (rationale or instructions or '')[:2000],
            'evidence_level': normalize_evidence_level(evidence_level),
            'expert_notes': (expert_notes or '')[:2000],
            'mask_source': normalize_mask_source(mask_source or mask_record.get('mask_source') or 'manual'),
        }
    else:
        professional_meta = {
            'rationale': (rationale or '')[:2000],
            'evidence_level': None,
            'expert_notes': (expert_notes or '')[:2000],
            'mask_source': normalize_mask_source(mask_source or mask_record.get('mask_source') or 'manual'),
        }
    limits = Budget(**(budget or {'candidates': count})).model_dump()
    limits['candidates'] = count
    if quality_thresholds and any(k not in {'boundary_pair_delta', 'boundary_gradient'} or not 0 <= float(v) <= 442 for k, v in quality_thresholds.items()):
        raise ValueError('质量阈值不合法。')
    identifier = new_id()
    job = {'id': identifier, 'image_id': image_record['id'], 'mask_id': mask_record['id'],
           'backend': backend, 'planner': planner, 'candidate_count': count, 'instructions': instructions[:2000],
           'status': 'queued', 'progress': 0, 'created_at': timestamp(), 'updated_at': timestamp(),
           'candidates': [], 'events': [], 'plan': None, 'selected_id': None, 'error': None,
           'cancel_requested': False, 'mode': mode, 'registration_id': registration_id,
           'budget': limits, 'structure_control': structure_control, 'attempts': [], 'revisions': [],
           'elapsed_seconds': 0, 'quality_thresholds': quality_thresholds or {},
           'proposal_ids': mask_record.get('proposal_ids', []), 'auto': auto,
           'workflow': workflow,
           'rationale': professional_meta.get('rationale') or '',
           'evidence_level': professional_meta.get('evidence_level'),
           'expert_notes': professional_meta.get('expert_notes') or '',
           'mask_source': professional_meta.get('mask_source') or 'manual',
           'allow_enhanced': bool(professional_meta.get('allow_enhanced') or (mode == 'enhanced' and allow_enhanced)),
           'risk_acknowledged': bool(professional_meta.get('risk_acknowledged')),
           'mask_gate_confirmed': bool(professional_meta.get('mask_gate_confirmed')),
           'has_unrestorable': bool(professional_meta.get('has_unrestorable')),
           'has_uncertain': bool(professional_meta.get('has_uncertain')),
           'force_override': bool(professional_meta.get('force_override')),
           'force_override_reason': professional_meta.get('force_override_reason') or '',
           'contest_demo': bool(professional_meta.get('contest_demo', False)),
           'selected_as': None}
    store.put('job', job)
    engine.submit(identifier)
    return job


def sample_image():
    image = Image.new('RGB', (960, 640), '#d5bb8b')
    draw = ImageDraw.Draw(image)
    for inset in (20, 28, 50, 58):
        draw.rectangle((inset, inset, 960-inset, 640-inset), outline='#86523e', width=3)
    for cx in (230, 480, 730):
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x, y = cx + 80*math.cos(rad), 320 + 80*math.sin(rad)
            draw.ellipse((x-27, y-45, x+27, y+45), fill='#a5533f', outline='#674837', width=3)
        draw.ellipse((cx-54, 266, cx+54, 374), fill='#637c70', outline='#674837', width=4)
        draw.ellipse((cx-22, 298, cx+22, 342), fill='#dfc286', outline='#674837', width=3)
    draw.line([(130, 90), (155, 190), (142, 250), (170, 390), (158, 530)], fill='#ede0bb', width=8)
    draw.line([(540, 80), (515, 190), (548, 300), (527, 410), (550, 555)], fill='#ede0bb', width=9)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

