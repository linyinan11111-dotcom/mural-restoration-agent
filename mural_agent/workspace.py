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


def add_mask(store, image_record, payload, proposal_ids=None):
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
    return store.put('mask', {'id': identifier, 'image_id': image_record['id'], 'created_at': timestamp(),
                             'fraction': round(float(array.mean()), 6), 'sha256': sha256(path),
                             'proposal_ids': proposal_ids or [], 'user_confirmed': True,
                             'url': f'/api/assets/{identifier}/mask.png'})


def create_job(store, engine, image_record, mask_record, backend, planner, count, instructions,
               mode='conservative', registration_id=None, budget=None, structure_control=False,
               quality_thresholds=None):
    from .planner import Budget
    from .imaging import mask_array
    available = engine.tools.availability()
    if backend == 'lama' and not engine.lama.available():
        raise ValueError('LaMa 权重或 PyTorch 未安装，请先配置模型。')
    if planner == 'api' and not engine.settings.api_ready:
        raise ValueError('请先在 API 设置中填写服务地址、视觉模型与密钥。')
    if planner == 'local' and not (engine.settings.vision_model/'provenance.json').exists():
        raise ValueError('本地视觉模型权重尚未安装。')
    if mask_record['image_id'] != image_record['id']:
        raise ValueError('掩膜不属于当前图像。')
    if backend not in {'opencv', 'telea', 'ns', 'lama', 'reference', 'diffusion', 'denoise', 'color', 'deblur', 'research'} or planner not in {'api', 'rules', 'local'} or not 1 <= count <= 3:
        raise ValueError('不支持的修复配置。')
    if not available.get(backend):
        raise ValueError('所选工具尚未配置，请在模型与环境中查看安装状态。')
    if mode not in {'conservative', 'evidence', 'speculative'}:
        raise ValueError('不支持的修复模式。')
    if (backend == 'reference') != (mode == 'evidence'):
        raise ValueError('证据辅助模式必须使用核验参考重建工具。')
    if (backend == 'diffusion') != (mode == 'speculative'):
        raise ValueError('推测模式必须使用扩散补全，其他模式禁用扩散。')
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
           'proposal_ids': mask_record.get('proposal_ids', [])}
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

