"""Whitelisted tools with versioned requests, isolated optional GPU worker."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import time
import cv2
import numpy as np
from PIL import Image
from .imaging import image_array, mask_array, opencv_inpaint, sha256

TOOLS = {'telea', 'ns', 'lama', 'reference', 'diffusion', 'denoise', 'color', 'deblur', 'research'}


class ToolRegistry:
    def __init__(self, engine):
        self.engine = engine

    def availability(self):
        s = self.engine.settings
        return {'opencv': True, 'telea': True, 'ns': True, 'lama': self.engine.lama.available(), 'reference': True,
                'diffusion': (s.diffusion_model / 'model_index.json').exists() and importlib.util.find_spec('diffusers') is not None,
                'denoise': True, 'color': True, 'deblur': True,
                'research': s.research_adapter.exists(), 'sam2': self.engine.sam.available(),
                'controlnet': (s.controlnet_model / 'config.json').exists()}

    def execute(self, step, original, mask, job, attempt_folder, timeout):
        engine, store, s = self.engine, self.engine.store, self.engine.settings
        if step.tool not in TOOLS:
            raise ValueError('工具不在白名单。')
        if step.tool in {'telea', 'ns'}:
            return opencv_inpaint(original, mask, step.tool, step.radius), {'opencv': cv2.__version__, 'device': 'cpu'}
        if step.tool == 'lama':
            output = engine.lama.predict(original, mask)
            import torch
            return output, {'torch': torch.__version__, 'weight_sha256': sha256(s.lama_model),
                            'device': str(engine.lama.device), 'roi_max_side': s.lama_max_side}
        if step.tool == 'reference':
            registration = store.get('registration', job.get('registration_id', ''))
            if not registration or not registration.get('approved') or registration['image_id'] != job['image_id']:
                raise ValueError('需要本幅壁画经人工核验的参考配准。')
            approved = mask_array(store.folder(registration['id']) / 'approved.png')
            if ((mask > 0) & (approved == 0)).any():
                raise ValueError('缺损超出核验区域；请缩小修复掩膜或重新核验参考。')
            return image_array(store.folder(registration['id']) / 'aligned.png'), {
                'device': 'cpu', 'registration_id': registration['id'], 'aligned_sha256': registration['aligned_sha256']}
        if step.tool in {'denoise', 'color', 'deblur'}:
            if step.tool == 'denoise':
                output = cv2.fastNlMeansDenoisingColored(original, None, 5, 5, 7, 21)
            elif step.tool == 'color':
                # Match the selected area to the surrounding extant tone, not imagined pigment.
                ring = (cv2.dilate(mask, np.ones((21, 21), np.uint8)) > 0) & (mask == 0)
                if not ring.any():
                    raise ValueError('未找到可用的周边颜色参考。')
                correction = original[ring].mean(axis=0) - original[mask > 0].mean(axis=0)
                output = np.clip(original.astype(float) + correction * step.strength, 0, 255).astype(np.uint8)
            else:
                blur = cv2.GaussianBlur(original, (0, 0), 1.2)
                output = cv2.addWeighted(original, 1.5, blur, -.5, 0)
            return output, {'device': 'cpu', 'opencv': cv2.__version__, 'method': {'color': 'local_tone_match', 'deblur': 'unsharp_mask', 'denoise': 'nlmeans'}[step.tool]}
        if step.tool == 'diffusion' and job.get('mode') not in {'speculative', 'enhanced'}:
            raise ValueError('扩散生成仅允许在推测性补全或加强修复模式执行。')
        # Release resident models before launching a separate interpreter on a 12GB GPU.
        engine.lama.unload()
        engine.sam.unload()
        # Persist the in-memory working image/mask (enhanced may expand mask / use conservative base).
        Image.fromarray(original).save(attempt_folder / 'work_image.png')
        Image.fromarray(mask).save(attempt_folder / 'work_mask.png')
        step_payload = step.model_dump()
        if job.get('mode') == 'enhanced':
            from .segmentation import ENHANCED_DIFFUSION_STRENGTH
            from .professional import ENHANCED_DIFFUSION_PROMPT, ENHANCED_NEGATIVE_PROMPT
            step_payload['strength'] = float(step_payload.get('strength') or ENHANCED_DIFFUSION_STRENGTH)
            if step_payload['strength'] < 0.9:
                step_payload['strength'] = float(ENHANCED_DIFFUSION_STRENGTH)
            step_payload['prompt'] = step_payload.get('prompt') or ENHANCED_DIFFUSION_PROMPT
            step_payload['negative_prompt'] = ENHANCED_NEGATIVE_PROMPT
        request = {'protocol': 1, 'tool': step.tool, 'step': step_payload,
            'image': str(attempt_folder / 'work_image.png'),
            'mask': str(attempt_folder / 'work_mask.png'),
            'output': str(attempt_folder / 'raw.png'), 'device': s.device,
            'model': str(s.diffusion_model), 'controlnet': str(s.controlnet_model),
            'research_adapter': str(s.research_adapter), 'cancel_file': str(store.folder(job['id']) / 'cancel')}
        request_path = attempt_folder / 'request.json'
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        with (attempt_folder / 'worker.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([s.worker_python or sys.executable, '-m', 'mural_agent.model_worker', str(request_path)],
                stdout=log, stderr=log, creationflags=flags)
            start = time.monotonic()
            try:
                while process.poll() is None:
                    current = store.get('job', job['id'])
                    if current.get('cancel_requested'):
                        (store.folder(job['id']) / 'cancel').touch()
                        process.terminate()
                        raise RuntimeError('模型任务已取消。')
                    if time.monotonic() - start > timeout:
                        process.terminate()
                        raise RuntimeError('模型 worker 超出任务时间预算。')
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if process.returncode:
            raise RuntimeError('模型 worker 失败，诊断保存在该次尝试的 worker.log。')
        if not (attempt_folder / 'raw.png').exists():
            raise RuntimeError('模型 worker 未生成结果。')
        meta_path = attempt_folder / 'worker-result.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {'device': s.device}
        return image_array(attempt_folder / 'raw.png'), meta
