from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal
import hashlib
import math
import importlib.metadata
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .config import ROOT, Settings
from .storage import Store, new_id, timestamp
from .engine import Engine
from .imaging import decode_image, MAX_BYTES, sha256
from . import workspace
from .planner import Budget


class JobRequest(BaseModel):
    image_id: str
    mask_id: str
    backend: Literal['opencv', 'telea', 'ns', 'lama', 'reference', 'diffusion', 'denoise', 'color', 'deblur', 'research'] = 'opencv'
    planner: Literal['rules', 'api', 'local'] = 'rules'
    candidate_count: int = Field(default=2, ge=1, le=3)
    instructions: str = Field(default='', max_length=2000)
    mode: Literal['conservative', 'evidence', 'speculative'] = 'conservative'
    registration_id: str | None = None
    budget: Budget | None = None
    structure_control: bool = False


class ReviewRequest(BaseModel):
    candidate_id: str
    note: str = Field(default='', max_length=2000)


def create_app(settings=None):
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    engine = Engine(settings, store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for job in store.list('job'):
            if job['status'] in {'queued', 'planning', 'running', 'evaluating'}:
                store.update('job', job['id'], cancel_requested=True)
        engine.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='壁画修复工作台', version='0.1.0', lifespan=lifespan)
    app.state.store, app.state.engine, app.state.settings = store, engine, settings

    def require(kind, identifier):
        record = store.get(kind, identifier)
        if record is None:
            raise HTTPException(404, '记录不存在。')
        return record

    async def read_upload(upload):
        payload = await upload.read(MAX_BYTES + 1)
        await upload.close()
        return payload

    def add_image(payload, filename):
        try:
            image = decode_image(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        identifier = new_id()
        folder = store.folder(identifier)
        (folder / 'source.bin').write_bytes(payload)
        image.save(folder / 'image.png')
        return store.put('image', {'id': identifier, 'name': filename[:200], 'width': image.width,
                                  'height': image.height, 'created_at': timestamp(),
                                  'original_sha256': hashlib.sha256(payload).hexdigest(),
                                  'working_sha256': sha256(folder / 'image.png'),
                                  'color_note': '原始字节已保留；修复使用 EXIF 校正后的 8 位 RGB 工作副本。',
                                  'url': f'/api/assets/{identifier}/image.png'})

    @app.get('/api/system')
    def system():
        return {'version': '0.1.0', 'api': {'configured': settings.api_ready, 'kind': settings.api_kind,
                                         'model': settings.api_model if settings.api_ready else ''},
                'lama': engine.lama.info(), 'opencv': True, 'max_pixels': 20_000_000,
                'features': {**engine.tools.availability(),
                             'automatic_damage_detection': 'local_qwen_proposals_require_human_confirmation'}}

    @app.post('/api/images', status_code=201)
    async def upload_image(file: UploadFile = File(...)):
        return add_image(await read_upload(file), file.filename or '壁画')

    @app.get('/api/images')
    def list_images():
        return store.list('image')

    @app.post('/api/demo', status_code=201)
    def demo():
        # Procedural test artwork, not an archaeological image or an AI-generated original.
        image = Image.new('RGB', (960, 640), '#d5bb8b')
        draw = ImageDraw.Draw(image)
        for inset in (20, 28, 50, 58):
            draw.rectangle((inset, inset, 960-inset, 640-inset), outline='#86523e', width=3)
        for cx in (230, 480, 730):
            cy = 320
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x, y = cx + 80*math.cos(rad), cy + 80*math.sin(rad)
                draw.ellipse((x-27, y-45, x+27, y+45), fill='#a5533f', outline='#674837', width=3)
            draw.ellipse((cx-54, cy-54, cx+54, cy+54), fill='#637c70', outline='#674837', width=4)
            draw.ellipse((cx-22, cy-22, cx+22, cy+22), fill='#dfc286', outline='#674837', width=3)
        draw.line([(130, 90), (155, 190), (142, 250), (170, 390), (158, 530)], fill='#ede0bb', width=8)
        draw.line([(540, 80), (515, 190), (548, 300), (527, 410), (550, 555)], fill='#ede0bb', width=9)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return add_image(buffer.getvalue(), '合成纹样练习图（非文物照片）.png')

    @app.post('/api/images/{image_id}/masks', status_code=201)
    async def upload_mask(image_id: str, file: UploadFile = File(...)):
        image = require('image', image_id)
        try:
            mask = decode_image(await read_upload(file), mask=True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if mask.size != (image['width'], image['height']):
            raise HTTPException(400, '掩膜必须与工作图尺寸一致。')
        array = np.asarray(mask) > 127
        if not array.any():
            raise HTTPException(400, '请先圈选需要修复的区域。')
        if array.all():
            raise HTTPException(400, '不能选择整幅图像，请保留可参考的未损坏区域。')
        identifier = new_id()
        path = store.folder(identifier) / 'mask.png'
        Image.fromarray(array.astype(np.uint8) * 255).save(path)
        return store.put('mask', {'id': identifier, 'image_id': image_id, 'created_at': timestamp(),
                                 'fraction': round(float(array.mean()), 6), 'sha256': sha256(path),
                                 'url': f'/api/assets/{identifier}/mask.png'})

    @app.post('/api/jobs', status_code=202)
    def create_job(request: JobRequest):
        require('image', request.image_id)
        mask = require('mask', request.mask_id)
        if mask['image_id'] != request.image_id:
            raise HTTPException(400, '掩膜不属于当前图像。')
        if request.backend == 'lama' and not engine.lama.available():
            raise HTTPException(409, 'LaMa 尚未就绪，请先安装权重或选择传统修补。')
        if request.planner == 'api' and not settings.api_ready:
            raise HTTPException(409, '请先在本地 .env 中配置 API 模型与密钥并重启服务。')
        active = [j for j in store.list('job') if j['status'] in {'queued', 'planning', 'running', 'evaluating'}]
        if len(active) >= 4:
            raise HTTPException(429, '当前任务较多，请等待现有任务完成。')
        try:
            return workspace.create_job(store, engine, require('image', request.image_id), mask, request.backend,
                request.planner, request.candidate_count, request.instructions, request.mode, request.registration_id,
                request.budget.model_dump() if request.budget else None, request.structure_control)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/jobs/{job_id}/resume')
    def resume_job(job_id: str):
        require('job', job_id)
        try:
            engine.resume(job_id)
            return store.get('job', job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/jobs')
    def list_jobs():
        return store.list('job')

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str):
        return require('job', job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        job = require('job', job_id)
        if job['status'] not in {'queued', 'planning', 'running', 'evaluating'}:
            return job
        return store.update('job', job_id, cancel_requested=True)

    @app.post('/api/jobs/{job_id}/review')
    def review(job_id: str, request: ReviewRequest):
        job = require('job', job_id)
        if request.candidate_id not in [c['id'] for c in job['candidates']]:
            raise HTTPException(400, '所选候选不属于当前任务。')
        if job['status'] in {'queued', 'planning', 'running', 'evaluating'}:
            raise HTTPException(409, '请等待任务完成或取消后再选择。')
        return store.update('job', job_id, selected_id=request.candidate_id,
                            review={'note': request.note, 'time': timestamp()})

    @app.get('/api/jobs/{job_id}/export')
    def export(job_id: str):
        require('job', job_id)
        try:
            archive = engine.export(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return FileResponse(archive, media_type='application/zip', filename=f'mural-{job_id[:8]}.zip')

    @app.get('/api/assets/{identifier}/{name}')
    def asset(identifier: str, name: str):
        kinds = {'image.png': 'image', 'mask.png': 'mask', 'result.png': 'candidate', 'difference.png': 'candidate'}
        if name not in kinds:
            raise HTTPException(404, '文件不存在。')
        require(kinds[name], identifier)
        path = store.folder(identifier) / name
        if not path.is_file():
            raise HTTPException(404, '文件不存在。')
        return FileResponse(path, media_type='image/png')

    return app

