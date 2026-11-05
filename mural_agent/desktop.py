"""Legacy Qt Widgets reference UI (not the shipped shell).

The shipped application is ``mural_agent.desktop_web`` (Qt WebEngine + ui/).
This module is kept for its canvas/undo regression tests and for
``configure_application`` (font/theme setup shared by both shells). It is not a
product entry point: ``main()`` already delegates to the WebEngine shell.
"""
from pathlib import Path
from datetime import datetime
import json
import os
import shutil
import sys
from PySide6.QtCore import Qt, QSize, QPointF, QRectF, QTimer, Signal, QBuffer, QIODevice, QUrl, QLockFile
from PySide6.QtGui import (QColor, QPainter, QPen, QPixmap, QImage, QIcon, QKeySequence,
                          QShortcut, QDesktopServices, QFont, QFontDatabase)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QSplitter, QFrame, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QSlider, QComboBox, QPlainTextEdit, QListWidget, QListWidgetItem, QTabWidget, QProgressBar,
    QFileDialog, QMessageBox, QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QStackedWidget,
    QAbstractItemView, QScrollArea, QButtonGroup, QStyle)
from .config import ROOT, Settings
from .storage import Store, timestamp
from .engine import Engine
from . import workspace
from .desktop_features import FeatureMixin

ACTIVE = {'queued', 'planning', 'running', 'evaluating'}
STATUS = {'queued': '排队中', 'planning': '规划中', 'running': '修补中', 'evaluating': '检查中',
          'review': '等待复核', 'failed': '执行失败', 'cancelled': '已取消', 'interrupted': '已中断'}
TOOLS = {'lama': 'LaMa 纹理补全', 'telea': '传统纹理传播', 'ns': '传统边缘传播',
         'reference': '参考证据重建', 'diffusion': '推测性扩散补全', 'denoise': '局部降噪',
         'color': '周边色调匹配', 'deblur': '局部锐化', 'research': '壁画研究模型'}


def button(text, callback=None, primary=False):
    widget = QPushButton(text)
    if callback:
        widget.clicked.connect(callback)
    if primary:
        widget.setObjectName('primary')
    return widget


def nav_button(text, callback):
    widget = QPushButton(text)
    widget.setObjectName('navButton')
    widget.setCheckable(True)
    widget.setAutoExclusive(True)
    widget.clicked.connect(callback)
    return widget


def label(text, name=''):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    return widget


def qimage_bytes(image):
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, 'PNG')
    return bytes(buffer.data())


class MaskView(QGraphicsView):
    changed = Signal()
    prompted = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor('#e2e9e6'))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 220)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.mask = QImage()
        self.overlay = None
        self.undo_stack = []
        self.mode = 'brush'
        self.radius = 18
        self.last = None
        self.box_start = None
        self.box_item = None
        self.fit_mode = True
        self.editable = True
        self.setAcceptDrops(False)

    def load(self, path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError('无法读取工作图。')
        self.scene().clear()
        self.box_item = None
        self.scene().addPixmap(pixmap)
        self.overlay = self.scene().addPixmap(QPixmap())
        self.overlay.setOpacity(.48)
        self.mask = QImage(pixmap.size(), QImage.Format.Format_ARGB32_Premultiplied)
        self.mask.fill(Qt.GlobalColor.transparent)
        self.undo_stack = []
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fit_mode = True
        self.fit_image()

    def fit_image(self):
        self.fit_mode = True
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect().adjusted(-8, -8, 8, 8), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_image()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.fit_mode = False
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            if .02 < self.transform().m11() * factor < 8:
                self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def checkpoint(self):
        self.undo_stack.append(qimage_bytes(self.mask))
        self.undo_stack = self.undo_stack[-12:]

    def redraw(self):
        if not self.overlay:
            return
        tinted = self.mask.copy()
        painter = QPainter(tinted)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor('#df503c'))
        painter.end()
        self.overlay.setPixmap(QPixmap.fromImage(tinted))
        self.changed.emit()

    def stroke(self, point):
        painter = QPainter(self.mask)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear if self.mode == 'eraser'
                                   else QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(Qt.GlobalColor.white, self.radius, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if self.last is None:
            painter.drawPoint(point)
        else:
            painter.drawLine(self.last, point)
        painter.end()
        self.last = point
        self.redraw()

    def mousePressEvent(self, event):
        if self.editable and not self.mask.isNull() and self.mode in {'sam_point', 'sam_box'}:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                if self.mode == 'sam_point' and event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
                    self.prompted.emit({'type': 'point', 'point': [point.x(), point.y()],
                                        'label': int(event.button() == Qt.MouseButton.LeftButton)})
                elif self.mode == 'sam_box' and event.button() == Qt.MouseButton.LeftButton:
                    self.box_start = point
                    if self.box_item:
                        self.scene().removeItem(self.box_item)
                    self.box_item = self.scene().addRect(QRectF(point, point), QPen(QColor('#357960'), 2))
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton and not self.mask.isNull() and self.editable:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.checkpoint()
                self.last = None
                self.stroke(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == 'sam_box' and self.box_start is not None and self.editable:
            point = self.mapToScene(event.position().toPoint())
            self.box_item.setRect(QRectF(self.box_start, point).normalized().intersected(self.sceneRect()))
            event.accept()
            return
        if event.buttons() & Qt.MouseButton.LeftButton and self.last is not None and self.editable:
            point = self.mapToScene(event.position().toPoint())
            self.stroke(point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.box_start is not None:
            rect = self.box_item.rect()
            self.box_start = None
            if rect.width() > 1 and rect.height() > 1:
                self.prompted.emit({'type': 'box', 'box': [rect.left(), rect.top(), rect.right(), rect.bottom()]})
            event.accept()
            return
        self.last = None
        super().mouseReleaseEvent(event)

    def clear(self):
        if self.mask.isNull() or not self.editable:
            return
        self.checkpoint()
        self.mask.fill(Qt.GlobalColor.transparent)
        self.redraw()

    def undo(self):
        if self.undo_stack and self.editable:
            self.mask = QImage.fromData(self.undo_stack.pop(), 'PNG')
            self.redraw()

    def export_mask(self):
        opaque = QImage(self.mask.size(), QImage.Format.Format_RGB32)
        opaque.fill(Qt.GlobalColor.black)
        painter = QPainter(opaque)
        painter.drawImage(0, 0, self.mask)
        painter.end()
        return qimage_bytes(opaque)

    def import_mask(self, payload):
        from .imaging import decode_image
        import numpy as np
        image = decode_image(payload, mask=True)
        if image.size != (self.mask.width(), self.mask.height()):
            raise ValueError('掩膜必须与工作图尺寸一致。')
        array = np.asarray(image) > 127
        rgba = np.full((*array.shape, 4), 255, dtype=np.uint8)
        rgba[..., 3] = array.astype(np.uint8) * 255
        self.checkpoint()
        self.mask = QImage(rgba.data, image.width, image.height, image.width * 4,
                           QImage.Format.Format_RGBA8888).copy().convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        self.redraw()


class ComparisonItem(QGraphicsItem):
    def __init__(self, original, result):
        super().__init__()
        self.original, self.result = original, result
        self.ratio = .5
        self.mode = 'compare'

    def boundingRect(self):
        return QRectF(self.result.rect())

    def paint(self, painter, option, widget=None):
        painter.drawPixmap(0, 0, self.result)
        if self.mode == 'compare':
            x = self.result.width() * self.ratio
            painter.save()
            painter.setClipRect(QRectF(0, 0, x, self.result.height()))
            painter.drawPixmap(0, 0, self.original)
            painter.restore()
            pen = QPen(QColor('#fffaf0'), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, 0), QPointF(x, self.result.height()))


class CompareView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(240, 210)
        self.setBackgroundBrush(QColor('#e2e9e6'))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.item = None

    def load(self, original, result, mode='compare', ratio=.5):
        self.scene().clear()
        self.item = ComparisonItem(QPixmap(str(original)), QPixmap(str(result)))
        self.item.mode, self.item.ratio = mode, ratio
        self.scene().addItem(self.item)
        self.scene().setSceneRect(self.item.boundingRect())
        self.fitInView(self.sceneRect().adjusted(-8, -8, 8, 8), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect().adjusted(-8, -8, 8, 8), Qt.AspectRatioMode.KeepAspectRatio)


class ApiDialog(QDialog):
    def __init__(self, settings, parent):
        super().__init__(parent)
        self.setWindowTitle('视觉 API 设置')
        self.resize(520, 340)
        self.settings = settings
        layout = QVBoxLayout(self)
        note = label('大模型负责分析与规划，本地模型负责修图。\nAPI 模式会向你填写的服务发送缩略图、掩膜预览与备注。', 'muted')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItem('Responses', 'responses')
        self.kind.addItem('Chat Completions 兼容服务', 'chat_completions')
        self.kind.setCurrentIndex(max(0, self.kind.findData(settings.api_kind)))
        self.base = QLineEdit(settings.api_base_url)
        self.model = QLineEdit(settings.api_model)
        self.model.setPlaceholderText('填写账户可用的视觉模型名称')
        self.key = QLineEdit(settings.api_key)
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText('密钥只保存在本机项目 .env 文件中')
        self.json_mode = QComboBox()
        self.json_mode.addItem('启用 JSON 模式', True)
        self.json_mode.addItem('兼容模式（不发送 response_format）', False)
        self.json_mode.setCurrentIndex(0 if settings.api_json_mode else 1)
        for title, widget in [('接口协议', self.kind), ('服务根地址', self.base), ('视觉模型', self.model),
                              ('API 密钥', self.key), ('输出格式', self.json_mode)]:
            form.addRow(title, widget)
        layout.addLayout(form)
        self.error = label('', 'error')
        layout.addWidget(self.error)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.save)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)

    def save(self):
        base = self.base.text().strip().rstrip('/')
        if not base.startswith(('https://', 'http://127.0.0.1', 'http://localhost')):
            self.error.setText('请填写 HTTPS 服务根地址，或本机模型服务地址。')
            return
        values = {'MURAL_API_KIND': self.kind.currentData(), 'MURAL_API_BASE_URL': base,
                  'MURAL_API_MODEL': self.model.text().strip(), 'MURAL_API_KEY': self.key.text().strip(),
                  'MURAL_API_JSON_MODE': str(self.json_mode.currentData()).lower()}
        if any('\n' in value or '\r' in value for value in values.values()):
            self.error.setText('配置不能包含换行。')
            return
        path = ROOT / '.env'
        lines = path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
        lines = [line for line in lines if line.split('=', 1)[0].strip() not in values]
        lines += [key + '=' + value for key, value in values.items()]
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        self.settings.api_kind, self.settings.api_base_url = values['MURAL_API_KIND'], base
        self.settings.api_model, self.settings.api_key = values['MURAL_API_MODEL'], values['MURAL_API_KEY']
        self.settings.api_json_mode = self.json_mode.currentData()
        self.accept()


class MainWindow(FeatureMixin, QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or Settings.from_env()
        self.store = Store(self.settings.data_dir)
        self.engine = Engine(self.settings, self.store)
        self.image = self.job = self.candidate = None
        self.running = False
        self.closing = False
        self.last_candidate_ids = []
        self.batch_ids = []
        self.setWindowTitle('拾壁 · 壁画修复工作室')
        self.resize(1440, 900)
        self.setMinimumSize(980, 640)
        icon = ROOT / 'assets/app.ico'
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self.build_ui()
        self.refresh_history()
        self.refresh_api()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_job)
        self.timer.start(650)
        QShortcut(QKeySequence.StandardKey.Open, self, self.open_image)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.canvas.undo)
        QShortcut(QKeySequence('B'), self, lambda: self.set_tool('brush'))
        QShortcut(QKeySequence('E'), self, lambda: self.set_tool('eraser'))
        self.statusBar().showMessage('本地桌面工作空间 · 原始文件单独保留 · 无需浏览器')

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        rail = QFrame()
        self.rail = rail
        rail.setObjectName('rail')
        rail.setFixedWidth(224)
        nav = QVBoxLayout(rail)
        nav.setContentsMargins(14, 20, 14, 16)
        nav.setSpacing(6)
        brand = QHBoxLayout()
        brand.addWidget(label('拾壁', 'brand'))
        brand.addWidget(label('MURAL\nRESTORATION', 'brandMeta'))
        brand.addStretch()
        nav.addLayout(brand)
        nav.addSpacing(20)
        self.new_task_button = button('+   新建修复任务', self.open_workbench, True)
        self.new_task_button.setObjectName('newTask')
        nav.addWidget(self.new_task_button)
        nav.addSpacing(20)
        nav.addWidget(label('工作空间', 'navCaption'))
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_workbench = nav_button('工作台', self.open_workbench)
        self.nav_tools = nav_button('工具与评测', self.open_tools)
        self.nav_history = nav_button('历史记录', self.open_history)
        for item in (self.nav_workbench, self.nav_tools, self.nav_history):
            self.nav_group.addButton(item)
            nav.addWidget(item)
        self.nav_workbench.setChecked(True)
        nav.addStretch()
        nav.addWidget(label('本机模型', 'navCaption'))
        available = self.engine.tools.availability()
        model_names = [('LaMa', available.get('lama')), ('SAM 2', available.get('sam2')),
                       ('Qwen-VL', (self.settings.vision_model/'provenance.json').exists()),
                       ('SD', available.get('diffusion'))]
        ready_models = ' · '.join(name for name, ready in model_names if ready)
        status = label(ready_models or '传统修补可用', 'railStatus')
        status.setWordWrap(True)
        nav.addWidget(status)
        nav.addWidget(label('GPU 模型按需加载', 'railNote'))
        nav.addSpacing(12)
        self.nav_api = button('API 设置', self.api_settings)
        self.nav_api.setObjectName('railAction')
        nav.addWidget(self.nav_api)
        nav.addWidget(label('DESKTOP · 1.0', 'railVersion'))
        root.addWidget(rail)
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(26, 18, 26, 12)
        title_row = QHBoxLayout()
        headings = QVBoxLayout()
        headings.addWidget(label('拾壁   /   数字壁画修复', 'eyebrow'))
        self.page_title = label('工作台', 'headline')
        headings.addWidget(self.page_title)
        title_row.addLayout(headings)
        title_row.addStretch()
        self.system_label = label('LaMa 可用' if self.engine.lama.available() else '传统修补可用', 'badge')
        title_row.addWidget(self.system_label)
        main_layout.addLayout(title_row)
        self.pages = QStackedWidget()
        main_layout.addWidget(self.pages, 1)
        root.addWidget(main, 1)
        self.build_workbench()
        self.build_history()
        self.build_extensions()
        self.mode.currentIndexChanged.connect(self.change_mode)

    def build_workbench(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        intro = QHBoxLayout()
        intro.addWidget(label('当前任务', 'muted'))
        intro.addStretch()
        self.demo_button = button('试用纹样练习图 ↗', self.load_demo)
        intro.addWidget(self.demo_button)
        layout.addLayout(intro)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)
        studio = QFrame()
        studio.setObjectName('card')
        studio_layout = QVBoxLayout(studio)
        studio_layout.setContentsMargins(16, 14, 16, 14)
        top = QHBoxLayout()
        self.image_title = label('导入壁画', 'section')
        top.addWidget(self.image_title, 1)
        self.open_button = button('打开图像…', self.open_image)
        top.addWidget(self.open_button)
        studio_layout.addLayout(top)
        self.tabs = QTabWidget()
        studio_layout.addWidget(self.tabs, 1)
        editor = QWidget()
        edit_layout = QVBoxLayout(editor)
        edit_layout.setContentsMargins(0, 8, 0, 0)
        toolbar = QHBoxLayout()
        self.brush = button('画笔', lambda: self.set_tool('brush'))
        self.brush.setCheckable(True)
        self.brush.setChecked(True)
        self.eraser = button('橡皮', lambda: self.set_tool('eraser'))
        self.eraser.setCheckable(True)
        toolbar.addWidget(self.brush)
        toolbar.addWidget(self.eraser)
        toolbar.addWidget(label('笔径', 'muted'))
        self.brush_size = QSlider(Qt.Orientation.Horizontal)
        self.brush_size.setRange(2, 100)
        self.brush_size.setValue(18)
        self.brush_size.setMaximumWidth(90)
        toolbar.addWidget(self.brush_size)
        self.brush_value = label('18', 'muted')
        toolbar.addWidget(self.brush_value)
        toolbar.addStretch()
        self.undo_button = button('撤销', lambda: self.canvas.undo())
        self.clear_button = button('清空', lambda: self.canvas.clear())
        self.import_button = button('导入掩膜', self.import_mask)
        toolbar.addWidget(self.undo_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.import_button)
        toolbar.addWidget(button('适应', lambda: self.canvas.fit_image()))
        edit_layout.addLayout(toolbar)
        self.editor_stack = QStackedWidget()
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch()
        glyph = label('', 'emptyGlyph')
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        glyph.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon).pixmap(46, 46))
        empty_layout.addWidget(glyph)
        for text, name in [('把一幅壁画带入工作台', 'emptyTitle'), ('打开本地图像，开始一次可追溯的修复。', 'muted')]:
            text_label = label(text, name)
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(text_label)
        empty_button = button('选择壁画…', self.open_image, True)
        empty_button.setMaximumWidth(180)
        empty_layout.addWidget(empty_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch()
        self.editor_stack.addWidget(empty)
        self.canvas = MaskView()
        self.canvas.setMinimumWidth(260)
        self.canvas.changed.connect(self.mask_changed)
        self.brush_size.valueChanged.connect(self.change_brush_size)
        self.editor_stack.addWidget(self.canvas)
        edit_layout.addWidget(self.editor_stack, 1)
        self.mask_note = label('红色区域允许编辑 · Ctrl + 滚轮缩放 · B 画笔 / E 橡皮', 'muted')
        edit_layout.addWidget(self.mask_note)
        self.tabs.addTab(editor, '① 圈选修复区域')
        self.build_results()
        split.addWidget(studio)
        inspector = QWidget()
        inspector.setObjectName('inspector')
        controls = QVBoxLayout(inspector)
        controls.setContentsMargins(18, 10, 16, 14)
        controls.setSpacing(8)
        controls.addWidget(label('02  修复方案', 'section'))
        controls.addSpacing(8)
        self.mode = QComboBox()
        self.mode.addItem('保守修补', 'conservative')
        self.mode.addItem('加强修复', 'enhanced')
        self.mode.addItem('证据辅助重建', 'evidence')
        self.mode.addItem('推测性补全', 'speculative')
        controls.addWidget(self.mode)
        controls.addWidget(label('修复引擎'))
        self.backend = QComboBox()
        self.backend.addItem('传统修补 · 小裂缝 / 小缺损', 'opencv')
        self.backend.addItem('Navier–Stokes · 细裂缝', 'ns')
        self.backend.addItem('Telea · 小缺失区域', 'telea')
        self.backend.addItem('LaMa · 上下文纹理补全', 'lama')
        for value in ('reference', 'diffusion', 'denoise', 'color', 'deblur', 'research'):
            self.backend.addItem(TOOLS[value], value)
        available = self.engine.tools.availability()
        for i in range(self.backend.count()):
            self.backend.model().item(i).setEnabled(available.get(self.backend.itemData(i), False))
        if self.engine.lama.available():
            self.backend.setCurrentIndex(self.backend.findData('lama'))
        else:
            lama_index = self.backend.findData('lama')
            self.backend.model().item(lama_index).setEnabled(False)
        controls.addWidget(self.backend)
        controls.addWidget(label('规划方式'))
        self.planner = QComboBox()
        self.planner.addItem('本地规则 · 无需密钥', 'rules')
        self.planner.addItem('视觉 API · 未配置', 'api')
        self.planner.addItem('本地 Qwen3-VL · 无需 API', 'local')
        local_ready = (self.settings.vision_model/'provenance.json').exists()
        self.planner.model().item(2).setEnabled(local_ready)
        if local_ready:
            self.planner.setCurrentIndex(2)
        controls.addWidget(self.planner)
        self.api_note = label('本地规则不会发送图片。', 'muted')
        self.api_note.setWordWrap(True)
        controls.addWidget(self.api_note)
        self.planner.currentIndexChanged.connect(self.update_api_note)
        controls.addWidget(label('候选数量'))
        self.count = QComboBox()
        for number in (1, 2, 3):
            self.count.addItem(f'{number} 个候选 · 顺序执行', number)
        self.count.setCurrentIndex(1)
        controls.addWidget(self.count)
        controls.addWidget(label('修复备注（可选）'))
        self.instructions = QPlainTextEdit()
        self.instructions.setPlaceholderText('例如：保留现有色调，只修补细裂缝。')
        self.instructions.setMaximumHeight(90)
        controls.addWidget(self.instructions)
        note = label('保留现存色调与纹理。补全结果属于数字修补建议，需结合参考资料复核。', 'muted')
        note.setWordWrap(True)
        controls.addWidget(note)
        self.run_button = button('生成修复候选  ↗', self.start_job, True)
        self.run_button.setEnabled(False)
        controls.addWidget(self.run_button)
        self.cancel_button = button('取消当前任务', self.cancel_job)
        self.cancel_button.hide()
        controls.addWidget(self.cancel_button)
        self.resume_button = button('从检查点续跑', self.resume_job)
        self.resume_button.hide()
        controls.addWidget(self.resume_button)
        controls.addSpacing(8)
        self.job_status = label('执行记录 · 尚未开始', 'section')
        controls.addWidget(self.job_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        controls.addWidget(self.progress)
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setPlaceholderText('完成圈选后，修复步骤将在这里展开。')
        controls.addWidget(self.events, 1)
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setMinimumWidth(260)
        inspector_scroll.setMaximumWidth(370)
        inspector_scroll.setWidget(inspector)
        split.addWidget(inspector_scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([930, 330])
        self.pages.addWidget(page)

    def build_results(self):
        results = QWidget()
        layout = QVBoxLayout(results)
        layout.setContentsMargins(0, 8, 0, 0)
        self.candidates = QListWidget()
        self.candidates.setViewMode(QListWidget.ViewMode.IconMode)
        self.candidates.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.candidates.setMovement(QListWidget.Movement.Static)
        self.candidates.setIconSize(QSize(132, 76))
        self.candidates.setGridSize(QSize(164, 108))
        self.candidates.setMaximumHeight(120)
        self.candidates.currentItemChanged.connect(self.choose_candidate)
        layout.addWidget(self.candidates)
        row = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItem('滑动对照', 'compare')
        self.view_mode.addItem('修复结果', 'result')
        self.view_mode.addItem('修改差异', 'difference')
        self.view_mode.currentIndexChanged.connect(self.update_comparison)
        row.addWidget(self.view_mode)
        row.addWidget(label('原图', 'muted'))
        self.compare_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_slider.setRange(0, 100)
        self.compare_slider.setValue(50)
        self.compare_slider.valueChanged.connect(self.change_comparison_ratio)
        row.addWidget(self.compare_slider, 1)
        row.addWidget(label('候选', 'muted'))
        layout.addLayout(row)
        self.comparison = CompareView()
        self.comparison.setMinimumHeight(230)
        layout.addWidget(self.comparison, 1)
        self.metrics = label('选择候选后查看像素约束与修复耗时。', 'muted')
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.review_note = QLineEdit()
        self.review_note.setMaxLength(2000)
        self.review_note.setPlaceholderText('记录选择依据（可选）')
        review_row = QHBoxLayout()
        review_row.addWidget(label('复核意见', 'muted'))
        review_row.addWidget(self.review_note, 1)
        layout.addLayout(review_row)
        actions = QHBoxLayout()
        self.select_button = button('选定此候选', self.select_candidate, True)
        self.export_button = button('导出修复档案…', self.export_archive)
        self.save_button = button('保存结果图…', self.save_result)
        for widget in (self.select_button, self.export_button, self.save_button):
            widget.setEnabled(False)
            actions.addWidget(widget)
        layout.addLayout(actions)
        self.tabs.addTab(results, '② 候选与对照')

    def build_history(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 12, 2, 2)
        layout.setSpacing(12)
        top = QHBoxLayout()
        title = QVBoxLayout()
        title.addWidget(label('工作空间  /  历史记录', 'eyebrow'))
        title.addWidget(label('历史记录', 'headline'))
        top.addLayout(title)
        top.addStretch()
        top.addWidget(button('返回工作台', self.open_workbench))
        layout.addLayout(top)
        self.history_filter = QLineEdit()
        self.history_filter.setPlaceholderText('按图像名称筛选')
        self.history_filter.textChanged.connect(self.refresh_history)
        layout.addWidget(self.history_filter)
        self.history = QListWidget()
        self.history.itemDoubleClicked.connect(self.restore_job)
        layout.addWidget(self.history, 1)
        footer = QHBoxLayout()
        footer.addWidget(label('双击记录可恢复图像、冻结掩膜、候选与复核意见。', 'muted'), 1)
        footer.addWidget(button('打开记录', lambda: self.restore_job(self.history.currentItem()), True))
        layout.addLayout(footer)
        self.pages.addWidget(page)

    def open_workbench(self):
        self.pages.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0)
        self.nav_workbench.setChecked(True)
        self.page_title.setText('工作台')

    def open_tools(self):
        self.pages.setCurrentIndex(0)
        self.tabs.setCurrentIndex(2)
        self.nav_tools.setChecked(True)
        self.page_title.setText('工具与评测')

    def alert(self, message):
        QMessageBox.warning(self, '拾壁', str(message))

    def open_image(self):
        if self.running:
            return
        path, _ = QFileDialog.getOpenFileName(self, '选择壁画图像', '', '图像 (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp)')
        if path:
            try:
                file = Path(path)
                if file.stat().st_size > 35 * 1024 * 1024:
                    raise ValueError('图片须小于 35 MB。')
                self.load_image(workspace.add_image(self.store, file.read_bytes(), file.name))
            except Exception as exc:
                self.alert(exc)

    def load_demo(self):
        if not self.running:
            self.load_image(workspace.add_image(self.store, workspace.sample_image(), '合成纹样练习图（非文物照片）.png'))
            self.statusBar().showMessage('练习图是程序绘制的纹样。请沿浅色裂缝圈选，不是历史文物照片。')

    def load_image(self, image):
        self.image = image
        self.reset_extensions()
        self.batch_ids = []
        self.job = self.candidate = None
        self.last_candidate_ids = []
        self.image_title.setText(f'{image["name"]}  /  {image["width"]} × {image["height"]}')
        self.canvas.load(self.store.folder(image['id']) / 'image.png')
        self.editor_stack.setCurrentIndex(1)
        self.candidates.clear()
        self.comparison.scene().clear()
        self.comparison.item = None
        self.events.clear()
        self.progress.setValue(0)
        self.job_status.setText('执行记录 · 尚未开始')
        self.mask_note.setText('红色区域允许编辑 · Ctrl + 滚轮缩放 · B 画笔 / E 橡皮')
        self.set_busy(False)
        self.export_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.tabs.setCurrentIndex(0)
        self.pages.setCurrentIndex(0)

    def set_tool(self, mode):
        self.canvas.mode = mode
        self.brush.setChecked(mode == 'brush')
        self.eraser.setChecked(mode == 'eraser')
        if mode in {'sam_point', 'sam_box'}:
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage('SAM 点提示：左键前景、右键背景；或拖框。随后到辅助页执行分割。')

    def change_brush_size(self, value):
        self.canvas.radius = value
        self.brush_value.setText(str(value))

    def mask_changed(self):
        self.mask_note.setText('掩膜已修改。红色区域允许修复，其他区域保持原样。')

    def import_mask(self):
        if not self.image or self.running:
            return
        path, _ = QFileDialog.getOpenFileName(self, '导入白色表示可编辑的掩膜', '', '掩膜 (*.png *.jpg *.jpeg)')
        if path:
            try:
                self.canvas.import_mask(Path(path).read_bytes())
            except Exception as exc:
                self.alert(exc)

    def set_busy(self, value):
        self.running = value
        for widget in (self.open_button, self.demo_button, self.backend, self.planner, self.count,
                       self.instructions, self.import_button, self.undo_button, self.clear_button, self.mode):
            widget.setEnabled(not value)
        if hasattr(self, 'assist_buttons'):
            for widget in (*self.assist_buttons, self.rounds, self.timeout, self.controlnet, self.seam_threshold):
                widget.setEnabled(not value)
        self.canvas.editable = not value
        self.run_button.setEnabled(not value and self.image is not None)
        self.cancel_button.setVisible(value)
        self.cancel_button.setEnabled(not bool(self.job and self.job.get('cancel_requested')))
        self.select_button.setEnabled(not value and self.candidate is not None)
        self.save_button.setEnabled(not value and self.candidate is not None)
        self.resume_button.setVisible(not value and bool(self.job and self.job['status'] in {'interrupted', 'failed', 'cancelled'}))

    def start_job(self):
        if not self.image or self.running:
            return
        try:
            mask = workspace.add_mask(self.store, self.image, self.canvas.export_mask(), self.proposal_ids)
            self.job = workspace.create_job(self.store, self.engine, self.image, mask, self.backend.currentData(),
                                            self.planner.currentData(), self.count.currentData(), self.instructions.toPlainText(),
                                            mode=self.mode.currentData(), registration_id=self.registration['id'] if self.registration else None,
                                            budget=self.current_budget(), structure_control=self.controlnet.isChecked(),
                                            quality_thresholds={'boundary_pair_delta': self.seam_threshold.value()})
            self.mask_note.setText(f'冻结掩膜占图像 {mask["fraction"]:.2%} · 所有候选从原始工作图产生')
            self.candidate = None
            self.last_candidate_ids = []
            self.candidates.clear()
            self.review_note.clear()
            self.export_button.setEnabled(False)
            self.set_busy(True)
            self.poll_job()
        except Exception as exc:
            self.alert(exc)

    def poll_job(self):
        if self.poll_aux():
            return
        if not self.job:
            return
        was_running = self.running
        self.job = self.store.get('job', self.job['id'])
        if not self.job:
            return
        self.job_status.setText('执行记录 · ' + STATUS.get(self.job['status'], self.job['status']))
        self.progress.setValue(self.job.get('progress', 0))
        lines = []
        if self.job.get('plan'):
            lines.append({'api': '视觉 API 规划：', 'local_vlm': '本地视觉模型规划：'}.get(self.job['plan']['source'], '本地规则规划：') + self.job['plan']['summary'])
        lines.extend('· ' + event['message'] for event in self.job.get('events', []))
        text = '\n\n'.join(lines)
        if self.events.toPlainText() != text:
            self.events.setPlainText(text)
            self.events.verticalScrollBar().setValue(self.events.verticalScrollBar().maximum())
        identifiers = [item['id'] for item in self.job['candidates']]
        if identifiers != self.last_candidate_ids:
            self.last_candidate_ids = identifiers
            self.render_candidates()
        active = self.job['status'] in ACTIVE
        self.set_busy(active)
        self.export_button.setEnabled(bool(self.job.get('selected_id')) and not active)
        if was_running and not active:
            self.refresh_history()
            if self.job['candidates']:
                self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage(self.job.get('error') or '候选已完成，请放大检查后选择结果。')
            if self.closing:
                self.close()

    def render_candidates(self):
        if self.image and self.job['image_id'] != self.image['id']:
            self.image = self.store.get('image', self.job['image_id'])
            self.canvas.load(self.store.folder(self.image['id'])/'image.png')
            self.canvas.import_mask((self.store.folder(self.job['mask_id'])/'mask.png').read_bytes())
            self.image_title.setText(self.image['name'])
        previous = self.candidate['id'] if self.candidate else self.job.get('selected_id')
        self.candidates.blockSignals(True)
        self.candidates.clear()
        selected_row = 0
        for index, candidate in enumerate(self.job['candidates']):
            device = 'GPU' if candidate['versions']['device'] == 'cuda' else 'CPU'
            item = QListWidgetItem(QIcon(str(self.store.folder(candidate['id']) / 'result.png')),
                                   f'{index+1:02} / {TOOLS[candidate["tool"]]}\n{candidate["elapsed_seconds"]} s / {device}')
            item.setData(Qt.ItemDataRole.UserRole, candidate['id'])
            self.candidates.addItem(item)
            if candidate['id'] == previous:
                selected_row = index
        self.candidates.blockSignals(False)
        if self.candidates.count():
            self.candidates.setCurrentRow(selected_row)

    def choose_candidate(self, current, previous=None):
        if not current or not self.job:
            return
        identifier = current.data(Qt.ItemDataRole.UserRole)
        self.candidate = next(item for item in self.job['candidates'] if item['id'] == identifier)
        self.update_comparison()
        metric = self.candidate['metrics']
        device = self.candidate['versions']['device'].upper()
        self.metrics.setText(f'掩膜外差异 {metric["outside_max_difference"]}  ·  修改 {metric["changed_pixels"]:,} 像素  ·  '
                             f'边界色差 {metric["boundary_color_gap"]}  ·  {device}\n边界色差仅作参考，不代表历史真实性。')
        if self.candidate.get('speculative'):
            self.metrics.setText('【推测性补全】缺失内容由模型生成，缺少直接历史证据。\n'+self.metrics.text())
        self.select_button.setEnabled(not self.running)
        self.save_button.setEnabled(not self.running)
        self.select_button.setText('已选定此候选' if self.job.get('selected_id') == identifier else '选定此候选')
        self.review_note.setText(self.job.get('review', {}).get('note', ''))

    def update_comparison(self):
        if not self.candidate or not self.image:
            return
        mode = self.view_mode.currentData()
        file = 'difference.png' if mode == 'difference' else 'result.png'
        self.comparison.load(self.store.folder(self.image['id']) / 'image.png',
                             self.store.folder(self.candidate['id']) / file, mode, self.compare_slider.value()/100)
        self.compare_slider.setEnabled(mode == 'compare')

    def change_comparison_ratio(self, value):
        if self.comparison.item:
            self.comparison.item.ratio = value / 100
            self.comparison.item.update()

    def select_candidate(self):
        if not self.candidate or self.running:
            return
        self.job = self.store.update('job', self.job['id'], selected_id=self.candidate['id'],
                                     review={'note': self.review_note.text(), 'time': timestamp()})
        self.select_button.setText('已选定此候选')
        self.export_button.setEnabled(True)
        self.statusBar().showMessage('选择已保存，可以导出原图、掩膜、结果与完整档案。')

    def export_archive(self):
        if not self.job or not self.job.get('selected_id'):
            return
        destination, _ = QFileDialog.getSaveFileName(self, '导出修复档案', f'mural-{self.job["id"][:8]}.zip', 'ZIP 修复档案 (*.zip)')
        if destination:
            try:
                source = self.engine.export(self.job['id'])
                if source.resolve() != Path(destination).resolve():
                    shutil.copy2(source, destination)
                self.statusBar().showMessage('修复档案已导出：' + destination)
            except Exception as exc:
                self.alert(exc)

    def save_result(self):
        if not self.candidate:
            return
        destination, _ = QFileDialog.getSaveFileName(self, '保存当前候选结果', 'restored.png', 'PNG 图像 (*.png)')
        if destination:
            try:
                source = self.store.folder(self.candidate['id']) / 'result.png'
                if source.resolve() != Path(destination).resolve():
                    shutil.copy2(source, destination)
                self.statusBar().showMessage('结果已保存：' + destination)
            except Exception as exc:
                self.alert(exc)

    def cancel_job(self):
        if self.job and self.running:
            self.store.update('job', self.job['id'], cancel_requested=True)
            for identifier in self.batch_ids:
                record = self.store.get('job', identifier)
                if record and record['status'] in ACTIVE:
                    self.store.update('job', identifier, cancel_requested=True)
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage('已请求取消，将在当前模型调用结束后停止。')

    def refresh_history(self):
        self.history.clear()
        query = self.history_filter.text().strip().casefold() if hasattr(self, 'history_filter') else ''
        for job in self.store.list('job'):
            image = self.store.get('image', job['image_id'])
            name = image['name'] if image else '壁画'
            if query and query not in name.casefold():
                continue
            date = datetime.fromisoformat(job['created_at']).astimezone().strftime('%Y-%m-%d %H:%M')
            item = QListWidgetItem(f'{name}\n{date}  /  {STATUS.get(job["status"], job["status"])}  /  {len(job["candidates"])} 个候选')
            item.setData(Qt.ItemDataRole.UserRole, job['id'])
            item.setSizeHint(QSize(600, 76))
            self.history.addItem(item)

    def open_history(self):
        self.refresh_history()
        self.pages.setCurrentIndex(1)
        self.nav_history.setChecked(True)
        self.page_title.setText('修复档案')

    def restore_job(self, item):
        if not item or self.running:
            if self.running:
                self.statusBar().showMessage('请等待当前任务完成或取消后再打开其他档案。')
            return
        job = self.store.get('job', item.data(Qt.ItemDataRole.UserRole))
        image = self.store.get('image', job['image_id'])
        if not image:
            return
        try:
            self.load_image(image)
            self.canvas.import_mask((self.store.folder(job['mask_id']) / 'mask.png').read_bytes())
            self.job = job
            index = self.mode.findData(job.get('mode', 'conservative'))
            self.mode.setCurrentIndex(max(0, index))
            self.backend.setCurrentIndex(self.backend.findData(job['backend']))
            if job.get('registration_id'):
                self.registration = self.store.get('registration', job['registration_id'])
                self.show_registration(self.registration)
            self.proposal_ids = job.get('proposal_ids', [])
            self.poll_job()
            if job['candidates']:
                self.tabs.setCurrentIndex(1)
        except Exception as exc:
            self.alert(exc)

    def api_settings(self):
        if self.running:
            self.statusBar().showMessage('请等待当前任务完成或取消后再修改 API 设置。')
            return
        dialog = ApiDialog(self.settings, self)
        if dialog.exec():
            self.refresh_api()
            self.statusBar().showMessage('API 配置已保存在本机，后续任务使用新配置。')

    def refresh_api(self):
        ready = self.settings.api_ready
        self.planner.setItemText(1, '视觉 API · ' + self.settings.api_model if ready else '视觉 API · 请先配置')
        self.planner.model().item(1).setEnabled(ready)
        self.update_api_note()

    def update_api_note(self):
        self.api_note.setText('会向已配置的服务发送缩略图、掩膜预览和备注。' if self.planner.currentData() == 'api'
                             else '本地 Qwen3-VL 分析图片，不发送数据。' if self.planner.currentData() == 'local' else '本地规则不会发送图片。')

    def closeEvent(self, event):
        if self.running:
            self.closing = True
            self.cancel_job()
            self.statusBar().showMessage('正在停止当前任务，完成后自动退出…')
            event.ignore()
        else:
            self.engine.executor.shutdown(wait=False, cancel_futures=True)
            event.accept()


STYLE = '''
QWidget { color: #292d33; font-family: "Microsoft YaHei"; font-size: 12px; }
QMainWindow, QDialog { background: #f7f8fa; }
QFrame#rail { background: #ffffff; border-right: 1px solid #e5e7eb; }
QFrame#card, QWidget#inspector { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
QLabel#brand { font-size: 22px; font-weight: 700; color: #20242a; padding: 0 3px; }
QLabel#brandMeta { color: #89909a; font-size: 8px; font-weight: 600; }
QLabel#navCaption { color: #858c96; font-size: 10px; font-weight: 600; padding: 5px 9px; }
QLabel#railStatus { color: #555e69; font-size: 10px; padding: 3px 9px; }
QLabel#railNote { color: #8a919b; font-size: 9px; padding: 0 9px; }
QLabel#railVersion { color: #9aa1aa; font-size: 9px; padding: 12px 9px 0 9px; }
QLabel#headline { font-size: 21px; font-weight: 650; margin: 1px 0; }
QLabel#eyebrow { color: #87909a; font-size: 10px; margin: 1px 0; }
QLabel#muted { color: #79818b; font-size: 11px; }
QLabel#badge { background: #f1f7f3; border: 1px solid #dcebe1; border-radius: 12px; padding: 6px 10px; color: #36724f; font-weight: 600; }
QLabel#section { font-size: 13px; font-weight: 650; padding: 4px 0; }
QLabel#emptyGlyph { color: #79818b; }
QLabel#emptyTitle { font-size: 20px; font-weight: 600; margin: 8px 0; }
QLabel#error { color: #b34437; }
QPushButton { background: #ffffff; border: 1px solid #dfe3e8; border-radius: 5px; padding: 7px 10px; min-height: 17px; }
QPushButton:hover { background: #f6f8fa; border-color: #c7cdd5; }
QPushButton:checked { background: #edf2f8; color: #2e5f9f; border-color: #edf2f8; }
QPushButton#navButton { background: transparent; border: 1px solid transparent; color: #555d67; text-align: left; padding: 9px 10px; }
QPushButton#navButton:hover { background: #f5f6f8; color: #242930; }
QPushButton#navButton:checked { background: #eef1f4; color: #252a30; font-weight: 600; }
QPushButton#newTask { text-align: left; font-weight: 600; }
QPushButton#railAction { text-align: left; color: #525b65; }
QPushButton#primary { background: #20242a; color: #ffffff; border: 1px solid #20242a; padding: 8px 12px; font-weight: 600; }
QPushButton#primary:hover { background: #383e46; }
QPushButton:disabled { color: #a8afb8; background: #f6f7f8; border-color: #eceef1; }
QPushButton#primary:disabled { background: #c8cbd0; border-color: #c8cbd0; color: #ffffff; }
QComboBox, QLineEdit, QPlainTextEdit, QSpinBox { background: #ffffff; border: 1px solid #dfe3e8; border-radius: 5px; padding: 6px 8px; selection-background-color: #dce7f5; }
QComboBox { min-height: 18px; }
QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover, QSpinBox:hover { border-color: #bfc7d1; }
QPlainTextEdit { font-size: 11px; }
QListWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; outline: none; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #f0f1f3; }
QListWidget::item:selected { background: #eef2f6; color: #292f36; border: 1px solid #dce2e9; }
QGroupBox { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 7px; margin-top: 10px; padding: 11px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #454c55; }
QTabWidget::pane { border: 1px solid #e5e7eb; background: #ffffff; border-radius: 6px; top: -1px; }
QTabBar::tab { background: transparent; border: 0; border-bottom: 1px solid #e5e7eb; padding: 8px 11px; margin-right: 4px; color: #727a84; }
QTabBar::tab:selected { background: #ffffff; color: #272d34; border-bottom: 2px solid #505c69; font-weight: 600; }
QSlider::groove:horizontal { height: 4px; background: #e3e6ea; border-radius: 2px; }
QSlider::handle:horizontal { width: 13px; height: 13px; background: #5b6571; margin: -5px 0; border-radius: 6px; }
QProgressBar { background: #eceff2; border: none; border-radius: 2px; }
QProgressBar::chunk { background: #62a37a; }
QSplitter::handle { background: transparent; width: 8px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: #d4d9df; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #f1f3f5; color: #78818b; font-size: 10px; border-top: 1px solid #e5e7eb; }
QToolTip { background: #292e35; color: #ffffff; border: 1px solid #444b55; padding: 5px 7px; }
'''


def configure_application(app):
    # Offscreen Qt on Windows may not initialize system font fallback. Explicit
    # font registration also makes Chinese rendering deterministic in previews.
    fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ('msyh.ttc', 'simsun.ttc', 'segoeui.ttf'):
        path = fonts / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont('Microsoft YaHei', 10))
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)


def main():
    from .desktop_web import main as web_main
    return web_main()


if __name__ == '__main__':
    raise SystemExit(main())

