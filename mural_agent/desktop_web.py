"""Qt WebEngine desktop shell for the local HTML workbench."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

from PySide6.QtCore import QObject, QLockFile, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from . import backup, datasets, references, workspace
from .config import ROOT, Settings
from .engine import Engine
from .imaging import image_array, mask_array
from .qt_setup import configure_application
from .segmentation import damage_proposal
from .storage import Store, new_id, timestamp


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


class Bridge(QObject):
    notify = Signal(str)
    auxCompleted = Signal()

    def __init__(self, settings, store, engine):
        super().__init__()
        self.settings, self.store, self.engine = settings, store, engine
        self.aux_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mural-aux')
        self.aux = {'status': 'idle'}
        self.aux_future = None
        self.aux_version = 0
        self.current_image_id = None

    def _result(self, function):
        try:
            return _json(function())
        except Exception as exc:
            return _json({'error': str(exc)[:500]})

    def _file_url(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.store.root.resolve()):
            raise ValueError('资源路径不在工作空间内。')
        return QUrl.fromLocalFile(str(path)).toString()

    def _image(self, record):
        if not record:
            return None
        result = dict(record)
        result['file_url'] = self._file_url(self.store.folder(record['id']) / 'image.png')
        return result

    def _job(self, record):
        if not record:
            return None
        result = dict(record)
        result['mask_file_url'] = self._file_url(self.store.folder(record['mask_id']) / 'mask.png')
        result['candidates'] = [self._candidate(item) for item in record.get('candidates', [])]
        return result

    def _candidate(self, record):
        result = dict(record)
        folder = self.store.folder(record['id'])
        result['file_url'] = self._file_url(folder / 'result.png')
        result['difference_url'] = self._file_url(folder / 'difference.png')
        return result

    def _save_proposal(self, image_id, mask, metadata):
        identifier = new_id()
        folder = self.store.folder(identifier)
        Image.fromarray(mask).save(folder / 'mask.png')
        proposal = self.store.put('proposal', {'id': identifier, 'image_id': image_id, 'created_at': timestamp(),
            'fraction': float((mask > 0).mean()), 'metadata': metadata, 'confirmed': False})
        return {'proposal': proposal}

    def _snapshot(self):
        images = [self._image(item) for item in self.store.list('image')]
        if self.current_image_id is None and images:
            latest_job = next(iter(self.store.list('job')), None)
            self.current_image_id = latest_job['image_id'] if latest_job else images[0]['id']
        jobs = [self._job(item) for item in self.store.list('job')]
        availability = self.engine.tools.availability()
        return {'images': images, 'jobs': jobs, 'models': {**availability, 'vision': (self.settings.vision_model / 'provenance.json').exists(),
            'api_configured': self.settings.api_ready}, 'aux': self.aux, 'aux_version': self.aux_version,
            'api': {'kind': self.settings.api_kind, 'base_url': self.settings.api_base_url,
                    'model': self.settings.api_model, 'json_mode': self.settings.api_json_mode},
            'registration': self._registration(self.store.get('registration', self.aux.get('registration_id', '')))
                if self.aux.get('registration_id') else None}

    def _registration(self, record):
        if not record:
            return None
        result = dict(record)
        folder = self.store.folder(record['id'])
        result['overlay_url'] = self._file_url(folder / 'overlay.png')
        return result

    def _aux(self, function, label):
        if self.aux_future and not self.aux_future.done():
            raise ValueError('辅助工具仍在运行。')
        if any(job['status'] in {'queued', 'planning', 'running', 'evaluating'}
               for job in self.store.list('job')):
            raise ValueError('修复任务运行中，请完成或取消后再分析。')
        self.aux = {'status': 'running', 'label': label}
        self.aux_version += 1
        self.aux_future = self.aux_executor.submit(function)
        return {'status': 'running'}

    def poll_aux(self):
        if self.aux_future and self.aux_future.done():
            future, self.aux_future = self.aux_future, None
            try:
                self.aux = {'status': 'done', 'result': future.result()}
                value = self.aux['result']
                if value.get('proposal'):
                    url = self._file_url(self.store.folder(value['proposal']['id']) / 'mask.png')
                    value['proposal']['file_url'] = url
                    value['proposal']['mask_file_url'] = url
                if value.get('registration'):
                    self.aux['registration_id'] = value['registration']['id']
                    value['registration'] = self._registration(value['registration'])
                if value.get('references'):
                    for item in value['references']:
                        image = self.store.get('image', item['image_id'])
                        item['file_url'] = self._file_url(self.store.folder(image['id']) / 'image.png')
            except Exception as exc:
                self.aux = {'status': 'error', 'error': str(exc)[:500]}
            self.aux_version += 1
            self.auxCompleted.emit()

    @Slot(result=str)
    def snapshot(self):
        self.poll_aux()
        return self._result(self._snapshot)

    @Slot(result=str)
    def openImage(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(None, '打开壁画图像', '', '图像 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)')
            if not path: return None
            record = workspace.add_image(self.store, Path(path).read_bytes(), Path(path).name)
            self.current_image_id = record['id']
            return self._image(record)
        return self._result(action)

    @Slot(result=str)
    def loadDemo(self):
        from .workspace import sample_image
        def action():
            record = workspace.add_image(self.store, sample_image(), '合成纹样练习图（非文物照片）.png')
            self.current_image_id = record['id']
            return self._image(record)
        return self._result(action)

    @Slot(str, str, str, result=str)
    def saveMask(self, image_id, data, proposal_ids):
        def action():
            payload = base64.b64decode(data, validate=True)
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            self.current_image_id = image_id
            # Desktop save after brush/edit is always an expert-confirmed mask.
            ids = json.loads(proposal_ids) if proposal_ids else []
            source = 'ai_candidate' if ids else 'manual'
            mask = workspace.add_mask(self.store, image, payload, ids, mask_source=source, user_confirmed=True)
            return {**mask, 'file_url': self._file_url(self.store.folder(mask['id']) / 'mask.png')}
        return self._result(action)

    @Slot(str, str, result=str)
    def addManualDefect(self, image_id, payload_json):
        """Add a user-drawn rectangular defect (normalized bbox 0..1000)."""
        def action():
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            payload = json.loads(payload_json or '{}')
            try:
                bbox = [int(value) for value in payload.get('bbox', [])]
            except (TypeError, ValueError) as exc:
                raise ValueError('手动缺损框无效。') from exc
            if len(bbox) != 4:
                raise ValueError('手动缺损框无效。')
            x0, y0, x1, y1 = [max(0, min(1000, value)) for value in bbox]
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                raise ValueError('手动缺损框太小，请重新框选。')
            import numpy as np
            with Image.open(self.store.folder(image['id']) / 'image.png') as source:
                width, height = source.size
            px0, px1 = round(x0 * width / 1000), round(x1 * width / 1000)
            py0, py1 = round(y0 * height / 1000), round(y1 * height / 1000)
            px0, px1 = max(0, px0), min(width, max(px0 + 1, px1))
            py0, py1 = max(0, py0), min(height, max(py0 + 1, py1))
            array = np.zeros((height, width), dtype=np.uint8)
            array[py0:py1, px0:px1] = 255
            if not array.any():
                raise ValueError('手动缺损框无效。')
            identifier = new_id()
            folder = self.store.folder(identifier)
            Image.fromarray(array).save(folder / 'mask.png')
            Image.fromarray(array).save(folder / 'proposal.png')
            metadata = {
                'type': str(payload.get('type') or 'manual')[:40],
                'bbox': [x0, y0, x1, y1],
                'size': str(payload.get('size') or 'manual')[:20],
                'confidence': 1.0,
                'evidence': '用户手动框选的缺损区域。',
                'recommendation': {'method': 'Telea 局部纹理传播', 'backend': 'telea'},
                'region_class': 'fillable_lacuna',
                'risk_regions': [],
                'risk_notice': '手动添加，未做自动风险分类。',
                'manual': True,
            }
            proposal = self.store.put('proposal', {
                'id': identifier, 'image_id': image['id'], 'created_at': timestamp(),
                'fraction': round(float(array.mean()), 6), 'metadata': metadata,
                'adopted': False, 'confirmed': True, 'mask_source': 'manual',
                'region_class': 'fillable_lacuna', 'repair_allowed': True,
            })
            defect = {
                'index': 0, 'type': metadata['type'], 'bbox': metadata['bbox'],
                'size': metadata['size'], 'confidence': 1.0,
                'evidence': metadata['evidence'], 'recommendation': metadata['recommendation'],
                'proposal_id': identifier, 'region_class': 'fillable_lacuna',
                'repair_allowed': True, 'risk_notice': metadata['risk_notice'], 'confirmed': True,
            }
            return {'proposal': proposal, 'defect': defect}
        return self._result(action)

    @Slot(str, result=str)
    def createJob(self, request_json):
        def action():
            if self.aux_future and not self.aux_future.done():
                raise ValueError('AI 分析仍在运行，请完成后再启动修复。')
            request = json.loads(request_json)
            image = self.store.get('image', request['image_id'])
            mask = self.store.get('mask', request['mask_id'])
            if not image or not mask:
                raise ValueError('图像或掩膜不存在。')
            job = workspace.create_job(
                self.store, self.engine, image, mask, request['backend'], request['planner'],
                int(request['candidate_count']), request.get('instructions', ''),
                request.get('mode', 'conservative'), request.get('registration_id'),
                request.get('budget'), request.get('structure_control', False),
                request.get('quality_thresholds'),
                workflow=request.get('workflow', 'standard'),
                rationale=request.get('rationale', ''),
                evidence_level=request.get('evidence_level'),
                expert_notes=request.get('expert_notes', ''),
                mask_source=request.get('mask_source') or mask.get('mask_source'),
                allow_enhanced=bool(request.get('allow_enhanced')),
                risk_acknowledged=bool(request.get('risk_acknowledged')),
                mask_gate_confirmed=bool(request.get('mask_gate_confirmed')),
                has_unrestorable=bool(request.get('has_unrestorable')),
                has_uncertain=bool(request.get('has_uncertain')),
                force_override=bool(request.get('force_override')),
                force_override_reason=request.get('force_override_reason') or '',
                allow_generative_attempt=bool(request.get('allow_generative_attempt')))
            return self._job(job)
        return self._result(action)

    @Slot(str, result=str)
    def cancelJob(self, job_id):
        return self._result(lambda: self.store.update('job', job_id, cancel_requested=True))

    @Slot(str, result=str)
    def resumeJob(self, job_id):
        return self._result(lambda: (self.engine.resume(job_id), self._job(self.store.get('job', job_id)))[1])

    @Slot(str, str, str, result=str)
    def selectCandidate(self, job_id, candidate_id, note):
        def action():
            from .professional import validate_candidate_selection
            job = self.store.get('job', job_id)
            if not job or candidate_id not in [item['id'] for item in job['candidates']]:
                raise ValueError('候选不属于当前任务。')
            if job['status'] in {'queued', 'planning', 'running', 'evaluating'}:
                raise ValueError('任务完成前不能选定候选。')
            archive_as_hypothesis = False
            review_note = note
            decision = None
            if isinstance(note, str) and note.strip().startswith('{'):
                try:
                    payload = json.loads(note)
                    if isinstance(payload, dict):
                        archive_as_hypothesis = bool(payload.get('archive_as_hypothesis'))
                        review_note = payload.get('note', '')
                        decision = payload.get('decision')
                except json.JSONDecodeError:
                    pass
            candidate = next(item for item in job['candidates'] if item['id'] == candidate_id)
            selection = validate_candidate_selection(
                job, candidate, archive_as_hypothesis=archive_as_hypothesis,
                decision=decision)
            # An empty box must not erase a note restored from the archive/archive history.
            review_note = str(review_note).strip()[:2000] or (job.get('review') or {}).get('note', '')
            return self._job(self.store.update(
                'job', job_id, selected_id=candidate_id, selected_as=selection['selected_as'],
                review={'note': review_note, 'time': timestamp(),
                        'archive_as_hypothesis': selection['archive_as_hypothesis'],
                        'decision': selection.get('decision')}))
        return self._result(action)

    @Slot(str, result=str)
    def saveResult(self, candidate_id):
        def action():
            from .professional import validate_selected_export
            job = next((j for j in self.store.list('job') if any(c.get('id') == candidate_id for c in j.get('candidates') or [])), None)
            if not job:
                raise ValueError('修复候选不存在。')
            if job.get('selected_id') != candidate_id:
                raise ValueError('请先在候选对照中选择并确认该候选，再保存修复图像。')
            candidate = next(c for c in job['candidates'] if c['id'] == candidate_id)
            selection = validate_selected_export(job, candidate)
            if selection['selected_as'] in {'rejected', 'hypothesis'}:
                raise ValueError('驳回或假设性结果不能保存为修复图像；请改用「导出修复档案」并标注为假设。')
            path, _ = QFileDialog.getSaveFileName(None, '保存修复结果', 'restored.png', 'PNG 图像 (*.png)')
            if path:
                shutil.copy2(self.store.folder(candidate_id) / 'result.png', path)
                self.notify.emit('结果已保存。')
            return {'saved': bool(path)}
        return self._result(action)

    @Slot(str, result=str)
    def exportArchive(self, job_id):
        def action():
            job = self.store.get('job', job_id)
            if not job:
                raise ValueError('修复任务不存在。')
            path, _ = QFileDialog.getSaveFileName(None, '导出修复档案', f'mural-{job_id[:8]}.zip', 'ZIP 档案 (*.zip)')
            if path:
                source = self.engine.export(job_id)
                if Path(path).resolve() != source.resolve():
                    shutil.copy2(source, path)
                self.notify.emit('修复档案已导出。')
            return {'saved': bool(path)}
        return self._result(action)

    @Slot(result=str)
    def importMask(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(None, '导入掩膜', '', '图像 (*.png *.jpg *.jpeg)')
            return {'mask_data': base64.b64encode(Path(path).read_bytes()).decode()} if path else None
        return self._result(action)

    @Slot(str, result=str)
    def setCurrentImage(self, image_id):
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            self.current_image_id = image_id
            return {'image_id': image_id}
        return self._result(action)

    @Slot(str, result=str)
    def proposeDamage(self, image_id):
        def action():
            self.current_image_id = image_id
            image = image_array(self.store.folder(image_id) / 'image.png')
            array, metadata = damage_proposal(image)
            return self._save_proposal(image_id, array, metadata)
        return self._result(lambda: self._aux(action, '分析细线候选…'))

    @Slot(str, str, str, str, result=str)
    def runSam(self, image_id, points_json, labels_json, box_json):
        def action():
            self.current_image_id = image_id
            points, labels, box = json.loads(points_json), json.loads(labels_json), json.loads(box_json)
            array = image_array(self.store.folder(image_id) / 'image.png')
            mask, metadata = self.engine.sam.predict(array, points, labels, box)
            return self._save_proposal(image_id, mask, metadata)
        return self._result(lambda: self._aux(action, 'SAM 分割中…'))

    @Slot(str, result=str)
    def assessDamage(self, image_id):
        def action():
            image = self.store.get('image', image_id)
            if not image:
                raise ValueError('图像记录不存在。')
            from .damage_assessment import assess_image, recommendation_for
            result = assess_image(self.engine, image)
            items = []
            for item in result.get('items', []):
                enriched = dict(item)
                enriched['recommendation'] = recommendation_for(item)
                items.append(enriched)
            return {'items': items, 'limitations': result.get('limitations', '')}
        return self._result(lambda: self._aux(action, '本地视觉模型识别缺损…'))

    @Slot(str, str, result=str)
    def damagePreview(self, image_id, item_json):
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            bbox = [int(value) for value in json.loads(item_json).get('bbox', [])]
            if len(bbox) != 4:
                raise ValueError('候选框无效。')
            with Image.open(self.store.folder(image_id) / 'image.png') as source:
                preview = source.convert('RGB').copy()
            width, height = preview.size
            x0, y0, x1, y1 = bbox
            box = (round(x0 * width / 1000), round(y0 * height / 1000),
                   round(x1 * width / 1000), round(y1 * height / 1000))
            from PIL import ImageDraw
            ImageDraw.Draw(preview).rectangle(box, outline=(230, 70, 52), width=max(2, min(width, height) // 350))
            target = self.store.folder(image_id) / 'damage-preview.png'
            preview.save(target)
            return {'file_url': self._file_url(target)}
        return self._result(action)

    @Slot(str, str, result=str)
    def refineDamage(self, image_id, item_json):
        def action():
            if not self.store.get('image', image_id):
                raise ValueError('图像记录不存在。')
            self.current_image_id = image_id
            item = json.loads(item_json)
            bbox = [int(value) for value in item.get('bbox', [])]
            if len(bbox) != 4:
                raise ValueError('候选框无效。')
            array = image_array(self.store.folder(image_id) / 'image.png')
            height, width = array.shape[:2]
            x0, y0, x1, y1 = bbox
            pad_x, pad_y = max(3, round(width * .005)), max(3, round(height * .005))
            box = [max(0, round(x0 * width / 1000) - pad_x), max(0, round(y0 * height / 1000) - pad_y),
                   min(width, round(x1 * width / 1000) + pad_x), min(height, round(y1 * height / 1000) + pad_y)]
            if item.get('type') == 'crack':
                from .segmentation import crack_mask_proposal
                mask, metadata = crack_mask_proposal(array, box)
            else:
                mask, metadata = self.engine.sam.predict(array, box=box)
            from .damage_assessment import recommendation_for
            metadata.update({key: item[key] for key in ('index', 'type', 'size', 'confidence', 'evidence', 'bbox') if key in item})
            metadata['recommendation'] = recommendation_for(item)
            return self._save_proposal(image_id, mask, metadata)
        return self._result(lambda: self._aux(action, '细化缺损掩膜…'))

    @Slot(str, result=str)
    def adoptProposal(self, proposal_id):
        def action():
            proposal = self.store.get('proposal', proposal_id)
            if not proposal or proposal['image_id'] != self.current_image_id:
                raise ValueError('辅助选区不属于当前图像。')
            return proposal
        return self._result(action)

    @Slot(str, result=str)
    def autoRestore(self, image_id):
        def action():
            workflow = 'standard'
            identifier = image_id
            if isinstance(image_id, str) and image_id.strip().startswith('{'):
                try:
                    payload = json.loads(image_id)
                    if isinstance(payload, dict):
                        identifier = payload.get('image_id') or payload.get('id')
                        workflow = payload.get('workflow', 'standard')
                except json.JSONDecodeError:
                    pass
            image = self.store.get('image', identifier)
            if not image:
                raise ValueError('图像记录不存在。')
            if not (self.settings.vision_model / 'provenance.json').exists():
                raise ValueError('AI 分析缺损需要本地 Qwen3-VL 权重，请先安装模型。')
            self.current_image_id = identifier
            from .auto_restore import auto_restore
            from .professional import WORKFLOW_PROFESSIONAL, normalize_workflow
            workflow = normalize_workflow(workflow)
            busy = ('专业分步：仅识别候选选区供确认…'
                    if workflow == WORKFLOW_PROFESSIONAL else 'AI 自主识别缺损并生成修复候选…')
            # Rebind message for _aux by storing on closure via nonlocal pattern below.
            result = auto_restore(self.store, self.engine, image, workflow=workflow)
            mask = dict(result['mask']) if result.get('mask') else None
            if mask:
                mask['file_url'] = self._file_url(self.store.folder(mask['id']) / 'mask.png')
            job = result.get('job')
            return {'auto': {'mask': mask, 'job': self._job(job) if job else None,
                             'items': result['items'], 'steps': result['steps'],
                             'defects': result.get('defects') or [],
                             'backend': result['backend'], 'summary': result['summary'],
                             'cultural_context': result.get('cultural_context') or {},
                             'planning_notes': result.get('planning_notes') or '',
                             'workflow': result.get('workflow', workflow),
                             'awaiting_confirmation': bool(result.get('awaiting_confirmation')),
                             'proposals': result.get('proposals') or []},
                    '_busy_message': busy}
        def wrapped():
            # _aux shows a fixed message; keep professional wording via notify after.
            payload = action()
            message = payload.pop('_busy_message', None)
            if message and payload.get('auto', {}).get('awaiting_confirmation'):
                self.notify.emit(payload['auto'].get('summary') or message)
            return payload
        return self._result(lambda: self._aux(wrapped, 'AI 正在分析缺损与文化线索…'))

    @Slot(str, result=str)
    def runDefectRepairs(self, request_json):
        def action():
            request = json.loads(request_json)
            if self.aux_future and not self.aux_future.done():
                raise ValueError('AI 分析仍在运行，请完成后再启动修复。')
            from .defect_jobs import run_confirmed_defects
            result = run_confirmed_defects(
                self.store, self.engine, request.get('image_id'),
                request.get('proposal_ids'), request.get('methods'),
                generative_attempt=bool(request.get('generative_attempt')))
            result['jobs'] = [self._job(job) for job in result['jobs']]
            return result
        return self._result(action)

    @Slot(str, str, result=str)
    def mergeProposal(self, proposal_id, current_mask):
        def action():
            proposal = self.store.get('proposal', proposal_id)
            if not proposal or proposal['image_id'] != self.current_image_id:
                raise ValueError('辅助选区不属于当前图像。')
            incoming = mask_array(self.store.folder(proposal_id) / 'mask.png') > 0
            from .imaging import decode_image
            import numpy as np
            current = np.asarray(decode_image(base64.b64decode(current_mask), mask=True)) > 127
            if current.shape != incoming.shape:
                raise ValueError('当前掩膜与辅助选区尺寸不一致。')
            merged = incoming | current
            identifier = new_id()
            Image.fromarray(merged.astype('uint8') * 255).save(self.store.folder(identifier) / 'mask.png')
            result = self.store.put('proposal', {'id': identifier, 'image_id': proposal['image_id'],
                'created_at': timestamp(), 'fraction': float(merged.mean()),
                'metadata': {**proposal['metadata'], 'merged_from': proposal_id, 'confirmed': False}})
            url = self._file_url(self.store.folder(identifier) / 'mask.png')
            result['file_url'] = url
            result['mask_file_url'] = url
            return result
        return self._result(action)

    @Slot(str, result=str)
    def addReference(self, metadata_json):
        def action():
            metadata = json.loads(metadata_json)
            path, _ = QFileDialog.getOpenFileName(None, '导入参考图', '', '图像 (*.png *.jpg *.jpeg *.tif *.tiff)')
            return references.add_reference(self.store, Path(path).read_bytes(), Path(path).name, metadata) if path else None
        return self._result(action)

    @Slot(str, str, result=str)
    def searchReferences(self, query, group):
        def action():
            image_record = self.store.get('image', self.current_image_id) if self.current_image_id else None
            if not image_record:
                image_record = next(iter(self.store.list('image')), None)
            if not image_record:
                raise ValueError('请先打开一幅壁画，再检索参考资料。')
            image = image_array(self.store.folder(image_record['id']) / 'image.png')
            records = references.retrieve(self.store, image, query, group)
            for item in records:
                image_record = self.store.get('image', item['image_id'])
                item['file_url'] = self._file_url(self.store.folder(image_record['id']) / 'image.png')
            return records
        return self._result(lambda: self._aux(action, '检索参考资料…'))

    @Slot(str, str, str, result=str)
    def registerReference(self, image_id, reference_id, mask_data):
        def action():
            from PIL import Image
            image = self.store.get('image', image_id); reference = self.store.get('reference', reference_id)
            self.current_image_id = image_id
            if not image or not reference:
                raise ValueError('图像或参考资料不存在。')
            from .imaging import decode_image
            import numpy as np
            mask = np.asarray(decode_image(base64.b64decode(mask_data), mask=True))
            record = references.register(self.store, image, reference, mask)
            return {'registration': record}
        return self._result(lambda: self._aux(action, '参考图配准中…'))

    @Slot(str, str, str, result=str)
    def approveReference(self, registration_id, mask_data, note):
        def action():
            payload = base64.b64decode(mask_data)
            return references.approve_registration(self.store, registration_id, payload, note)
        return self._result(action)

    @Slot(result=str)
    def importDataset(self):
        def action():
            path, _ = QFileDialog.getOpenFileName(None, '导入数据清单', '', 'JSONL (*.jsonl *.txt)')
            if not path: return {'notice': '已取消。'}
            records = datasets.import_manifest(self.store, path)
            return {'notice': f'已导入 {len(records)} 条样本。'}
        return self._result(action)

    @Slot(result=str)
    def exportSplits(self):
        def action():
            path, _ = QFileDialog.getSaveFileName(None, '导出分组划分', 'splits.json', 'JSON (*.json)')
            if not path: return {'notice': '已取消。'}
            Path(path).write_text(_json(datasets.grouped_split(self.store.list('dataset'))), encoding='utf-8')
            return {'notice': '分组划分已导出。'}
        return self._result(action)

    @Slot(str, result=str)
    def evaluatePair(self, candidate_id):
        def action():
            from . import datasets
            path, _ = QFileDialog.getOpenFileName(None, '选择已对齐的配对目标', '', '图像 (*.png *.jpg *.jpeg *.tif *.tiff)')
            if not path: return {'cancelled': True}
            candidate = self.store.get('candidate', candidate_id)
            if not candidate: raise ValueError('候选记录不存在。')
            report = datasets.evaluate_pair(self.store, candidate, path, perceptual=True)
            QMessageBox.information(None, '配对评测', _json(report))
            return report
        return self._result(action)

    @Slot(str, result=str)
    def batchJobs(self, settings_json):
        def action():
            directory = QFileDialog.getExistingDirectory(None, '选择批量目录：每幅图需配套同名 .mask.png')
            if not directory: return {'cancelled': True}
            settings = json.loads(settings_json)
            if settings.get('mode', 'conservative') != 'conservative':
                raise ValueError('批量任务仅支持保守修补模式。')
            paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'} and '.mask.' not in p.name)
            pairs = [(path, path.with_name(path.stem + '.mask.png')) for path in paths]
            if not pairs or len(pairs) > 50 or any(not mask_path.is_file() for _, mask_path in pairs):
                raise ValueError('目录需提供 1 至 50 组图像及同名 .mask.png 文件。')
            from .imaging import decode_image
            import numpy as np
            for path, mask_path in pairs:
                image = decode_image(path.read_bytes())
                mask = decode_image(mask_path.read_bytes(), mask=True)
                pixels = np.asarray(mask) > 127
                if image.size != mask.size or not pixels.any() or pixels.all():
                    raise ValueError('无效的批量图像或掩膜：' + path.name)
            queued = []
            for path, mask_path in pairs:
                image = workspace.add_image(self.store, path.read_bytes(), path.name)
                record = workspace.add_mask(self.store, image, mask_path.read_bytes())
                job = workspace.create_job(self.store, self.engine, image, record, settings['backend'], settings['planner'],
                    settings['count'], settings.get('instructions', ''), budget={'candidates': settings['count'], 'revisions': settings['rounds'], 'timeout_seconds': settings['timeout']})
                queued.append(job['id'])
            return {'notice': f'已创建 {len(queued)} 个批量修复任务。'}
        return self._result(action)

    @Slot(result=str)
    def showModels(self):
        def action():
            QMessageBox.information(None, '模型与环境', _json(self.engine.tools.availability()))
            return {'shown': True}
        return self._result(action)

    @Slot(result=str)
    def backupWorkspace(self):
        def action():
            path, _ = QFileDialog.getSaveFileName(None, '备份工作空间', 'shibi-backup.zip', 'ZIP 档案 (*.zip)')
            return {'path': backup.backup_workspace(self.store, path)} if path else {'cancelled': True}
        return self._result(action)

    @Slot(str, result=str)
    def saveApiSettings(self, settings_json):
        def action():
            values = json.loads(settings_json)
            kind = values.get('kind')
            base = values.get('base_url', '').strip().rstrip('/')
            model = values.get('model', '').strip()
            key = values.get('key', '')
            json_mode = bool(values.get('json_mode', True))
            if kind not in {'responses', 'chat_completions'}:
                raise ValueError('不支持的 API 协议。')
            if not base.startswith(('https://', 'http://127.0.0.1', 'http://localhost')):
                raise ValueError('请填写 HTTPS 服务根地址，或本机模型服务地址。')
            fields = {'MURAL_API_KIND': kind, 'MURAL_API_BASE_URL': base,
                      'MURAL_API_MODEL': model, 'MURAL_API_JSON_MODE': str(json_mode).lower()}
            if key: fields['MURAL_API_KEY'] = key
            if any('\n' in value or '\r' in value for value in fields.values()):
                raise ValueError('配置不能包含换行。')
            env_path = ROOT / '.env'
            lines = env_path.read_text(encoding='utf-8-sig').splitlines() if env_path.exists() else []
            lines = [line for line in lines if line.split('=', 1)[0].strip() not in fields]
            lines.extend(name + '=' + value for name, value in fields.items())
            env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            self.settings.api_kind, self.settings.api_base_url = kind, base
            self.settings.api_model, self.settings.api_json_mode = model, json_mode
            if key: self.settings.api_key = key
            return {'saved': True}
        return self._result(action)


class WebMainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.data_dir)
        self.engine = Engine(settings, self.store)
        self.setWindowTitle('拾壁 · 壁画修复工作室')
        self.resize(1440, 920)
        self.setMinimumSize(980, 640)
        icon = ROOT / 'assets' / 'app.ico'
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.view = QWebEngineView(self)
        self.channel = QWebChannel(self.view.page())
        self.bridge = Bridge(settings, self.store, self.engine)
        self.channel.registerObject('bridge', self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        self.setCentralWidget(self.view)
        self.view.load(QUrl.fromLocalFile(str(ROOT / 'ui' / 'index.html')))
        self.closing = False
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(250)
        self.close_timer.timeout.connect(self._finish_close)

    def _finish_close(self):
        active = any(j['status'] in {'queued', 'planning', 'running', 'evaluating'} for j in self.store.list('job'))
        auxiliary = self.bridge.aux_future and not self.bridge.aux_future.done()
        if not active and not auxiliary:
            self.close_timer.stop()
            self.close()

    def closeEvent(self, event):
        if self.closing:
            # Second close attempt while cancellation is still pending: leave now.
            # Every finished attempt is checkpointed in SQLite and the next start
            # marks running jobs as interrupted, so work can be resumed from history.
            try:
                self.engine.executor.shutdown(wait=False, cancel_futures=True)
                self.bridge.aux_executor.shutdown(wait=False, cancel_futures=True)
            finally:
                os._exit(0)
        active = [j for j in self.store.list('job') if j['status'] in {'queued', 'planning', 'running', 'evaluating'}]
        auxiliary = self.bridge.aux_future and not self.bridge.aux_future.done()
        if active or auxiliary:
            self.closing = True
            self.bridge.notify.emit('正在取消运行中的任务，完成后自动退出；再点一次关闭可强制退出。')
            for job in active: self.store.update('job', job['id'], cancel_requested=True)
            if auxiliary and self.bridge.aux_future:
                self.bridge.aux_future.cancel()
            self.close_timer.start()
            event.ignore()
            return
        self.engine.executor.shutdown(wait=False, cancel_futures=True)
        event.accept()


def main():
    os.chdir(ROOT)
    app = QApplication(sys.argv)
    app.setApplicationName('拾壁'); app.setOrganizationName('MuralAtelier')
    configure_application(app)
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(settings.data_dir / 'desktop.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None, '拾壁', '拾壁已经在运行，请切换到已有的应用窗口。')
        return 0
    window = WebMainWindow(settings); window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
