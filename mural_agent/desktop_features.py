"""Native UI for segmentation, evidence, budgets, datasets and operations."""
from io import BytesIO
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QCheckBox, QLineEdit, QFormLayout, QDialog, QDialogButtonBox, QFileDialog,
    QListWidget, QListWidgetItem, QPlainTextEdit, QGroupBox, QScrollArea, QGridLayout, QSizePolicy)
from . import references, datasets, workspace
from .backup import backup_workspace
from .imaging import image_array, mask_array
from .segmentation import damage_proposal, crack_mask_proposal
from .damage_assessment import assess_image
from .storage import new_id, timestamp


def push(text, callback, layout=None):
    button = QPushButton(text)
    button.clicked.connect(callback)
    if layout is not None:
        layout.addWidget(button)
    return button


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, 'PNG')
    return buffer.getvalue()


class FitImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, pixmap):
        self.source = pixmap
        self.update_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()

    def update_image(self):
        if not self.source.isNull() and self.width() > 0 and self.height() > 0:
            self.setPixmap(self.source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))


class FeatureMixin:
    def build_extensions(self):
        self.aux_future = None
        self.registration = self.proposal = None
        self.proposal_ids = []
        self.damage_assessment = None
        self.damage_items = []
        self.sam_points, self.sam_labels = [], []
        self.sam_box = None
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        scroll.setWidget(page)
        root = QVBoxLayout(page)
        intro = QLabel('先让本地视觉模型列出原图疑似缺损，再按类别细化区域掩膜。检查并采用单个区域后，按类别建议修复；模型不能替代文物保护诊断。')
        intro.setWordWrap(True)
        root.addWidget(intro)
        top = QGridLayout()
        top.setHorizontalSpacing(8)
        top.setVerticalSpacing(8)
        self.assist_buttons = []
        for index, (text, callback) in enumerate([('AI 分析原图缺损', self.assess_damage), ('细线候选（启发式）', self.propose_damage),
            ('SAM 点选', lambda: self.set_tool('sam_point')), ('SAM 框选', lambda: self.set_tool('sam_box')),
            ('执行 SAM 分割', self.segment), ('清除 SAM 提示', self.clear_prompts)]):
            item = push(text, callback)
            self.assist_buttons.append(item)
            top.addWidget(item, index // 3, index % 3)
        root.addLayout(top)
        self.damage_list = QListWidget()
        self.damage_list.setMaximumHeight(180)
        self.damage_list.currentRowChanged.connect(self.highlight_damage)
        root.addWidget(self.damage_list)
        push('细化选中缺损掩膜', self.segment_assessed_damage, root)
        self.prompt_note = QLabel('点选：左键前景、右键背景；框选：拖动矩形。提示在圈选画布中操作。')
        root.addWidget(self.prompt_note)
        self.proposal_note = QLabel('尚无辅助选区；病害性质需要人工判断。')
        self.proposal_note.setWordWrap(True)
        root.addWidget(self.proposal_note)
        self.proposal_preview = FitImageLabel()
        root.addWidget(self.proposal_preview)
        row = QGridLayout()
        replace_mask = push('采用辅助选区（替换当前掩膜）', self.accept_proposal)
        merge_mask = push('与当前掩膜合并', lambda: self.accept_proposal(merge=True))
        row.addWidget(replace_mask, 0, 0)
        row.addWidget(merge_mask, 0, 1)
        row.setColumnStretch(0, 1)
        row.setColumnStretch(1, 1)
        root.addLayout(row)
        self.canvas.prompted.connect(self.receive_prompt)
        evidence = QGroupBox('参考图库与证据核验')
        evid_layout = QVBoxLayout(evidence)
        search = QHBoxLayout()
        self.ref_query = QLineEdit()
        self.ref_query.setPlaceholderText('题材 / 来源关键词')
        self.ref_group = QLineEdit()
        self.ref_group.setPlaceholderText('壁画分组（可选，精确筛选）')
        search.addWidget(self.ref_query)
        search.addWidget(self.ref_group)
        push('检索', self.refresh_references, search)
        push('导入参考', self.import_reference, search)
        evid_layout.addLayout(search)
        self.ref_list = QListWidget()
        self.ref_list.setMaximumHeight(130)
        evid_layout.addWidget(self.ref_list)
        row = QHBoxLayout()
        self.assist_buttons.append(push('配准所选参考', self.register_reference, row))
        push('核验当前掩膜覆盖区域', self.approve_reference, row)
        evid_layout.addLayout(row)
        self.reference_note = QLabel('来源与关系将随修复档案导出；视觉相似不等于历史证据。')
        self.reference_note.setWordWrap(True)
        evid_layout.addWidget(self.reference_note)
        self.reference_preview = type(self.comparison)()
        self.reference_preview.setMinimumHeight(260)
        evid_layout.addWidget(self.reference_preview)
        root.addWidget(evidence)
        config = QGroupBox('执行预算与局部恢复工具')
        form = QFormLayout(config)
        self.rounds = QSpinBox()
        self.rounds.setRange(0, 2)
        self.rounds.setValue(2)
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 3600)
        self.timeout.setValue(300)
        self.timeout.setSuffix(' 秒')
        self.controlnet = QCheckBox('启用 inpaint ControlNet 结构约束（推测模式）')
        self.seam_threshold = QSpinBox()
        self.seam_threshold.setRange(0, 442)
        self.seam_threshold.setValue(65)
        form.addRow('最多反馈修订', self.rounds)
        form.addRow('任务总时间预算', self.timeout)
        form.addRow('边界相邻像素差阈值', self.seam_threshold)
        form.addRow(self.controlnet)
        form.addRow(QLabel('候选总数上限取工作台设置；推理期间的超时在安全边界停止，扩散 worker 可终止。'))
        root.addWidget(config)
        operations = QGroupBox('数据、评测与维护')
        ops = QVBoxLayout(operations)
        row = QGridLayout()
        for index, (text, callback) in enumerate([('导入数据清单 JSONL', self.import_dataset),
            ('导出分组划分', self.export_splits), ('配对目标评测', self.evaluate_target)]):
            row.addWidget(push(text, callback), 0, index)
        ops.addLayout(row)
        row = QGridLayout()
        for index, (text, callback) in enumerate([('批量导入修复任务', self.batch_jobs),
            ('模型与环境', self.show_models), ('备份工作空间', self.backup_data)]):
            row.addWidget(push(text, callback), 0, index)
        ops.addLayout(row)
        self.dataset_note = QLabel('真实图缺少可靠目标时只做工程检查。清单格式见 datasets/example.jsonl。')
        self.dataset_note.setWordWrap(True)
        ops.addWidget(self.dataset_note)
        root.addWidget(operations)
        root.addStretch()
        self.tabs.addTab(scroll, '③ 辅助、证据与评测')

    def reset_extensions(self):
        self.registration = self.proposal = None
        self.proposal_ids = []
        self.damage_assessment = None
        self.damage_items = []
        if hasattr(self, 'damage_list'):
            self.damage_list.clear()
        self.sam_points, self.sam_labels, self.sam_box = [], [], None
        self.proposal_preview.set_image(QPixmap())
        self.reference_preview.scene().clear()
        self.reference_preview.item = None
        self.reference_note.setText('请导入或检索参考资料并配准。')

    def run_aux(self, function, callback, description):
        if self.running:
            return
        self.aux_callback = callback
        self.aux_future = self.engine.executor.submit(function)
        self.set_busy(True)
        self.statusBar().showMessage(description)

    def poll_aux(self):
        if self.aux_future is None:
            return False
        if self.aux_future.done():
            future, callback = self.aux_future, self.aux_callback
            self.aux_future = None
            self.set_busy(False)
            try:
                callback(future.result())
            except Exception as exc:
                self.alert(str(exc)[:500])
            if self.closing:
                self.close()
        return True

    def clear_prompts(self):
        self.sam_points, self.sam_labels, self.sam_box = [], [], None
        self.prompt_note.setText('SAM 提示已清除。左键前景、右键背景；可拖框。')

    def receive_prompt(self, prompt):
        if prompt['type'] == 'point':
            self.sam_points.append(prompt['point'])
            self.sam_labels.append(prompt['label'])
        else:
            self.sam_box = prompt['box']
        self.prompt_note.setText(f'已记录 {len(self.sam_points)} 个点；框：{self.sam_box or "无"}。返回辅助页执行分割。')
        self.statusBar().showMessage(self.prompt_note.text())

    def propose_damage(self):
        if not self.image or self.running:
            return
        image = image_array(self.store.folder(self.image['id'])/'image.png')
        self.run_aux(lambda: damage_proposal(image), self.show_proposal, '分析高对比细线候选…')

    def assess_damage(self):
        if not self.image or self.running:
            return
        record = self.image.copy()
        self.run_aux(lambda: assess_image(self.engine, record), self.show_damage_assessment,
                     '本地 Qwen 正在识别原图中的疑似缺损…')

    def show_damage_assessment(self, assessment):
        self.damage_assessment = assessment
        self.damage_items = assessment['items']
        self.damage_list.clear()
        labels = {'crack': '裂缝', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
                  'flaking': '起甲/剥落', 'stain': '污渍/变色', 'other': '其他疑似缺损'}
        for item in self.damage_items:
            x0, y0, x1, y1 = item['bbox']
            extent = (x1-x0)*(y1-y0)/10000
            text = (f"{item['index']:02d}  {labels[item['type']]} · {item['size']} · "
                    f"框选约 {extent:.2f}% · 视觉置信 {item['confidence']:.0%}")
            self.damage_list.addItem(QListWidgetItem(text))
        self.proposal = None
        if self.damage_items:
            self.damage_list.setCurrentRow(0)
            note = f"AI 提出 {len(self.damage_items)} 个疑似缺损候选。数量、类别和框选均需人工核对；这不是病害鉴定。"
        else:
            note = 'AI 未找到可确认的缺损候选；也可能是模型漏检，请放大原图人工检查。'
        self.proposal_note.setText(note + ' ' + assessment.get('limitations', ''))
        self.tabs.setCurrentIndex(2)

    def highlight_damage(self, row):
        if not self.image or not 0 <= row < len(self.damage_items):
            return
        item = self.damage_items[row]
        image = Image.open(self.store.folder(self.image['id'])/'image.png').convert('RGB')
        draw = ImageDraw.Draw(image)
        width, height = image.size
        x0, y0, x1, y1 = item['bbox']
        box = (round(x0*width/1000), round(y0*height/1000), round(x1*width/1000), round(y1*height/1000))
        draw.rectangle(box, outline=(230, 70, 52), width=max(2, min(width, height)//350))
        pixmap = QPixmap()
        pixmap.loadFromData(png(np.asarray(image)))
        self.proposal_preview.set_image(pixmap)
        labels = {'crack': '裂缝', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
                  'flaking': '起甲/剥落', 'stain': '污渍/变色', 'other': '其他疑似缺损'}
        methods = {'crack': 'Navier–Stokes 细线修补' if item['size'] != 'large' else 'LaMa 候选；长裂缝需人工复核',
                   'paint_loss': 'LaMa 纹理补全' if item['size'] == 'large' else 'Telea 局部纹理传播',
                   'plaster_loss': 'LaMa 纹理补全' if item['size'] == 'large' else 'Telea 局部纹理传播', 'flaking': 'Telea 局部纹理传播',
                   'stain': '局部色调匹配', 'other': '先人工确认，再选择保守修补'}
        x0, y0, x1, y1 = item['bbox']
        fraction = (x1-x0)*(y1-y0)/10000
        item['bbox_fraction'] = fraction
        if item['type'] == 'stain':
            item['recommended_backend'] = item['recommended_tool'] = 'color'
        elif item['type'] == 'crack':
            item['recommended_backend'] = item['recommended_tool'] = 'lama' if item['size'] == 'large' else 'ns'
        elif item['type'] in {'paint_loss', 'plaster_loss'} and item['size'] == 'large':
            item['recommended_backend'] = item['recommended_tool'] = 'lama'
        elif item['type'] in {'paint_loss', 'plaster_loss', 'flaking'}:
            item['recommended_backend'] = item['recommended_tool'] = 'telea'
        else:
            item['recommended_backend'] = 'opencv'
            item['recommended_tool'] = 'telea/ns'
        self.proposal_note.setText(
            f"疑似{labels[item['type']]}，AI估计{item['size']}，框选约 {fraction:.2f}% 图像面积；"
            f"建议：{methods[item['type']]}。视觉置信 {item['confidence']:.0%} 不是病害概率。"
            f"依据：{item['evidence'] or '无文字依据'}。请先检查红框是否落在真实损伤上。")

    def segment_assessed_damage(self):
        row = self.damage_list.currentRow()
        if not self.image or self.running or not 0 <= row < len(self.damage_items):
            return
        item = self.damage_items[row].copy()
        image = image_array(self.store.folder(self.image['id'])/'image.png')
        height, width = image.shape[:2]
        x0, y0, x1, y1 = item['bbox']
        pad_x, pad_y = max(3, round(width*.005)), max(3, round(height*.005))
        box = [max(0, round(x0*width/1000)-pad_x), max(0, round(y0*height/1000)-pad_y),
               min(width, round(x1*width/1000)+pad_x), min(height, round(y1*height/1000)+pad_y)]
        def complete(result):
            mask, metadata = result
            labels = {'crack': '裂缝', 'paint_loss': '颜料层缺失', 'plaster_loss': '地仗/灰泥缺失',
                      'flaking': '起甲/剥落', 'stain': '污渍/变色', 'other': '其他疑似缺损'}
            metadata.update({key: item[key] for key in ('index', 'type', 'size', 'confidence', 'evidence',
                                                        'bbox', 'recommended_backend', 'recommended_tool') if key in item})
            if item['type'] == 'crack':
                metadata['notice'] = ('Qwen 疑似裂缝框内的细线候选；' + metadata.get('notice', '') +
                                      '视觉识别和掩膜均须人工确认。')
            else:
                metadata['notice'] = (f"Qwen 疑似{labels[item['type']]} + SAM 框引导分割；"
                                      "两者均为待人工确认候选，不是病害诊断。")
            self.show_proposal((mask, metadata))
        if item['type'] == 'crack':
            self.run_aux(lambda: crack_mask_proposal(image, box), complete, '正在提取裂缝细线候选…')
        else:
            self.run_aux(lambda: self.engine.sam.predict(image, box=box), complete, 'SAM 正在细化选中缺损区域…')

    def segment(self):
        if not self.image or self.running:
            return
        image = image_array(self.store.folder(self.image['id'])/'image.png')
        points, labels, box = list(self.sam_points), list(self.sam_labels), self.sam_box
        self.run_aux(lambda: self.engine.sam.predict(image, points, labels, box), self.show_proposal, 'SAM 2 分割中…')

    def show_proposal(self, result):
        array, metadata = result
        identifier = new_id()
        Image.fromarray(array).save(self.store.folder(identifier)/'proposal.png')
        self.proposal = self.store.put('proposal', {'id': identifier, 'image_id': self.image['id'],
            'metadata': metadata, 'created_at': timestamp(), 'adopted': False})
        image = image_array(self.store.folder(self.image['id'])/'image.png')
        image[array > 0] = (.5*image[array > 0]+.5*np.array([220, 64, 48])).astype(np.uint8)
        pixmap = QPixmap()
        pixmap.loadFromData(png(image))
        self.proposal_preview.set_image(pixmap)
        self.proposal_note.setText(metadata['notice']+f' 候选面积 {(array > 0).mean():.2%}；采用前请检查红色区域。')
        self.tabs.setCurrentIndex(2)

    def accept_proposal(self, checked=False, merge=False):
        if not self.proposal or self.running:
            return
        metadata = self.proposal.get('metadata', {})
        if merge and metadata.get('recommended_backend') and (mask_array(BytesIO(self.canvas.export_mask())) > 0).any():
            self.alert('AI 按缺损类别逐项修复。请先完成当前掩膜的候选复核，再处理下一项。')
            return
        array = mask_array(self.store.folder(self.proposal['id'])/'proposal.png')
        if merge:
            current = np.asarray(Image.open(BytesIO(self.canvas.export_mask())).convert('L'))
            array = np.maximum(array, current)
        self.canvas.import_mask(png(array))
        self.proposal_ids.append(self.proposal['id'])
        self.store.update('proposal', self.proposal['id'], adopted=True, adopted_at=timestamp())
        backend = metadata.get('recommended_backend')
        if backend:
            self.mode.setCurrentIndex(self.mode.findData('conservative'))
            index = self.backend.findData(backend)
            if index >= 0 and self.backend.model().item(index).isEnabled():
                self.backend.setCurrentIndex(index)
            self.planner.setCurrentIndex(self.planner.findData('rules'))
            self.instructions.setPlainText(
                f"AI建议{metadata.get('type', 'unknown')}使用 {metadata.get('recommended_tool', backend)}；"
                "仅处理已确认掩膜，先检查候选，不代表历史原貌。")
        if metadata.get('index') and 0 <= self.damage_list.currentRow() < self.damage_list.count():
            item = self.damage_items[self.damage_list.currentRow()]
            item['segmented'] = True
            self.damage_list.currentItem().setText('已采用掩膜 · ' + self.damage_list.currentItem().text())
        self.set_tool('brush')
        self.tabs.setCurrentIndex(0)

    def import_reference(self):
        if self.running:
            return
        path, _ = QFileDialog.getOpenFileName(self, '导入参考图', '', '图像 (*.png *.jpg *.jpeg *.tif *.webp)')
        if not path:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('参考资料来源')
        form = QFormLayout(dialog)
        fields = {}
        for key, title in [('source', '来源 URL / 馆藏编号'), ('license', '许可 / 使用依据'), ('group', '壁画或洞窟分组'), ('date', '资料年代（已知才填）')]:
            fields[key] = QLineEdit()
            form.addRow(title, fields[key])
        relation = QComboBox()
        for title, value in [('同一壁画历史图', 'same_mural'), ('可信线稿 / 临摹', 'tracing'), ('同题材弱参考', 'related')]:
            relation.addItem(title, value)
        form.addRow('对应关系', relation)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            try:
                metadata = {k: field.text().strip() for k, field in fields.items()}
                metadata['relation'] = relation.currentData()
                references.add_reference(self.store, Path(path).read_bytes(), Path(path).name, metadata)
                self.refresh_references()
            except Exception as exc:
                self.alert(exc)

    def refresh_references(self):
        if self.running:
            return
        image = image_array(self.store.folder(self.image['id'])/'image.png') if self.image else np.zeros((20, 20, 3), np.uint8)
        def render(records):
            self.ref_list.clear()
            for record in records:
                item = QListWidgetItem(f'{record["name"]} / {record["metadata"]["group"]} / {record["metadata"]["relation"]} / {record["retrieval_score"]}')
                item.setData(Qt.ItemDataRole.UserRole, record['id'])
                self.ref_list.addItem(item)
        query, group = self.ref_query.text(), self.ref_group.text()
        self.run_aux(lambda: references.retrieve(self.store, image, query, group), render, '检索本地参考资料…')

    def register_reference(self):
        current = self.ref_list.currentItem()
        if not self.image or not current or self.running:
            return
        reference = self.store.get('reference', current.data(Qt.ItemDataRole.UserRole))
        mask = np.asarray(Image.open(BytesIO(self.canvas.export_mask())).convert('L'))
        self.run_aux(lambda: references.register(self.store, self.image, reference, mask), self.show_registration, 'SIFT / RANSAC 配准中…')

    def show_registration(self, registration):
        self.registration = registration
        self.reference_preview.load(self.store.folder(self.image['id'])/'image.png',
                                    self.store.folder(registration['id'])/'aligned.png', 'compare', .5)
        self.reference_note.setText(f'内点 {registration["inliers"]}，中位误差 {registration["median_error_px"]:.2f}px。'
            '请检查对应关系，并在圈选页仅保留有直接证据的缺损，再点击核验。')

    def approve_reference(self):
        if not self.registration or self.running:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('确认参考应用区域')
        form = QFormLayout(dialog)
        form.addRow(QLabel('当前手绘掩膜将冻结为人工核验区域。请确认画线、题材和参考来源对应。'))
        note = QLineEdit()
        note.setPlaceholderText('记录对应关系及核验依据（必填）')
        form.addRow('核验意见', note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            try:
                self.registration = references.approve_registration(self.store, self.registration['id'], self.canvas.export_mask(), note.text())
                self.reference_note.setText('参考已核验。证据模式只能修改本次核验区域内的像素。')
                self.mode.setCurrentIndex(1)
            except Exception as exc:
                self.alert(exc)

    def change_mode(self):
        mode = self.mode.currentData()
        target = 'reference' if mode == 'evidence' else 'diffusion' if mode == 'speculative' else 'lama' if self.engine.lama.available() else 'opencv'
        self.backend.setCurrentIndex(self.backend.findData(target))
        self.controlnet.setEnabled(mode == 'speculative')
        if mode != 'speculative':
            self.controlnet.setChecked(False)

    def resume_job(self):
        if self.job and not self.running:
            try:
                self.engine.resume(self.job['id'])
                self.set_busy(True)
            except Exception as exc:
                self.alert(exc)

    def show_json(self, title, data):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(780, 550)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        layout.addWidget(text)
        def save():
            path, _ = QFileDialog.getSaveFileName(dialog, '保存报告', 'report.json', 'JSON (*.json)')
            if path:
                Path(path).write_text(text.toPlainText(), encoding='utf-8')
        push('保存报告', save, layout)
        dialog.exec()

    def show_models(self):
        if self.running:
            return
        def inspect():
            import torch
            return {'tools': self.engine.tools.availability(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'torch': torch.__version__, 'paths': {'sam2': str(self.settings.sam_model), 'diffusion': str(self.settings.diffusion_model),
                'controlnet': str(self.settings.controlnet_model)}, 'api_configured': self.settings.api_ready,
                'installation': 'scripts/download_models.py sam2 diffusion controlnet --lpips'}
        self.run_aux(inspect, lambda data: self.show_json('模型与环境', data), '检查本地模型环境…')

    def import_dataset(self):
        if self.running:
            return
        path, _ = QFileDialog.getOpenFileName(self, '选择带来源和许可的数据清单', '', 'JSONL (*.jsonl)')
        if path:
            self.run_aux(lambda: datasets.import_manifest(self.store, path),
                         lambda records: self.dataset_note.setText(f'本次导入 {len(records)} 条，累计 {len(self.store.list("dataset"))} 条。'), '导入并验证数据…')

    def export_splits(self):
        if self.running:
            return
        try:
            self.show_json('按壁画分组划分', datasets.grouped_split(self.store.list('dataset')))
        except Exception as exc:
            self.alert(exc)

    def evaluate_target(self):
        if not self.candidate or self.running:
            self.statusBar().showMessage('请先生成并选择一个候选。')
            return
        path, _ = QFileDialog.getOpenFileName(self, '选择已对齐的可靠配对目标（非同题材参考）', '', '图像 (*.png *.jpg *.jpeg *.tif)')
        if path:
            candidate = self.candidate.copy()
            self.run_aux(lambda: datasets.evaluate_pair(self.store, candidate, path, perceptual=True),
                         lambda data: self.show_json('配对评测报告', data), '计算 mask MAE / PSNR 与局部 SSIM / LPIPS…')

    def backup_data(self):
        if self.running:
            return
        path, _ = QFileDialog.getSaveFileName(self, '备份本地工作空间（不含 API 密钥和模型）', 'shibi-backup.zip', 'ZIP (*.zip)')
        if path:
            self.run_aux(lambda: backup_workspace(self.store, path), lambda p: self.statusBar().showMessage('备份已保存：'+p), '备份工作空间…')

    def batch_jobs(self):
        if self.running:
            return
        directory = QFileDialog.getExistingDirectory(self, '选择批量任务目录：foo.png 与 foo.mask.png 配对')
        if not directory:
            return
        if self.mode.currentData() != 'conservative':
            self.alert('批量入口用于保守修补；证据与推测模式请逐幅核验。')
            return
        try:
            paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'} and '.mask.' not in p.name)
            pairs = [(p, p.with_name(p.stem+'.mask.png')) for p in paths]
            if not pairs or len(pairs) > 50 or any(not m.exists() for p, m in pairs):
                raise ValueError('需提供 1–50 组图片及同名 .mask.png 文件。')
            # Validate images and masks before submitting any task.
            from .imaging import decode_image
            for path, mask_path in pairs:
                image, mask = decode_image(path.read_bytes()), decode_image(mask_path.read_bytes(), mask=True)
                a = np.asarray(mask) > 127
                if image.size != mask.size or not a.any() or a.all():
                    raise ValueError('无效的批量图像或掩膜：'+path.name)
            self.batch_ids = []
            for path, mask_path in pairs:
                image = workspace.add_image(self.store, path.read_bytes(), path.name)
                mask = workspace.add_mask(self.store, image, mask_path.read_bytes())
                job = workspace.create_job(self.store, self.engine, image, mask, self.backend.currentData(),
                    self.planner.currentData(), self.count.currentData(), self.instructions.toPlainText(),
                    budget=self.current_budget())
                self.batch_ids.append(job['id'])
            self.job = self.store.get('job', self.batch_ids[-1])
            self.set_busy(True)
            self.statusBar().showMessage(f'已顺序排队 {len(self.batch_ids)} 个任务；取消将取消本批全部剩余任务。')
        except Exception as exc:
            self.alert(exc)

    def current_budget(self):
        return {'candidates': self.count.currentData(), 'revisions': self.rounds.value(), 'timeout_seconds': self.timeout.value(), 'max_api_calls': 1}
