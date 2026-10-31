import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from io import BytesIO
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QLineEdit
from mural_agent.config import Settings
from mural_agent.desktop import MainWindow, ApiDialog


def isolated_settings(tmp_path):
    # Legacy Qt Widgets shell: used by canvas regressions only. Every optional
    # model path is isolated so a developer machine with Qwen3-VL/SAM weights
    # cannot silently switch the default planner to the local VLM (which would
    # make this suite depend on 4 GB of model loading).
    return Settings(data_dir=tmp_path / 'data', lama_model=tmp_path / 'missing.pt',
                    sam_model=tmp_path / 'missing-sam', diffusion_model=tmp_path / 'missing-sd',
                    controlnet_model=tmp_path / 'missing-cn', vision_model=tmp_path / 'missing-qwen',
                    research_adapter=tmp_path / 'missing-research.json')


def test_native_canvas_and_restoration_without_webserver(tmp_path, qt_app):
    window = MainWindow(isolated_settings(tmp_path))
    assert window.planner.currentData() == 'rules'
    window.show()
    qt_app.processEvents()
    try:
        window.load_demo()
        qt_app.processEvents()
        assert window.image['width'] == 960
        canvas = window.canvas
        canvas.checkpoint()
        canvas.stroke(QPointF(130, 90))
        canvas.stroke(QPointF(155, 190))
        canvas.stroke(QPointF(142, 250))
        canvas.last = None
        mask = np.asarray(Image.open(BytesIO(canvas.export_mask())).convert('L'))
        assert (mask > 127).any()
        canvas.undo()
        empty = np.asarray(Image.open(BytesIO(canvas.export_mask())).convert('L'))
        assert not empty.any()
        canvas.import_mask(bytes_from_array(mask))
        window.start_job()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            qt_app.processEvents()
            window.poll_job()
            if not window.running:
                break
            time.sleep(.02)
        assert window.job['status'] == 'review', window.job
        assert window.candidates.count() == 2
        assert window.candidate is not None
        assert window.select_button.isEnabled()
        window.review_note.setText('原生桌面端验收')
        window.select_candidate()
        assert window.job['selected_id'] == window.candidate['id']
        archive = window.engine.export(window.job['id'])
        with zipfile.ZipFile(archive) as zipped:
            original = np.asarray(Image.open(BytesIO(zipped.read('original/working.png'))))
            result = np.asarray(Image.open(BytesIO(zipped.read('result.png'))))
            saved_mask = np.asarray(Image.open(BytesIO(zipped.read('mask.png')))) > 127
            assert np.array_equal(original[~saved_mask], result[~saved_mask])
        window.open_history()
        assert window.history.count() == 1
        window.restore_job(window.history.item(0))
        assert window.job['selected_id']
        assert window.export_button.isEnabled()
    finally:
        window.timer.stop()
        window.close()


def bytes_from_array(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, 'PNG')
    return buffer.getvalue()


def test_native_api_settings_are_local_and_password_masked(tmp_path, monkeypatch, qt_app):
    monkeypatch.setattr('mural_agent.desktop.ROOT', tmp_path)
    settings = Settings(tmp_path / 'data', tmp_path / 'model.pt')
    dialog = ApiDialog(settings, None)
    assert dialog.key.echoMode() == QLineEdit.EchoMode.Password
    dialog.model.setText('vision-test')
    dialog.key.setText('nonproduction-test-key')
    dialog.save()
    assert settings.api_ready
    assert settings.api_model == 'vision-test'
    assert 'MURAL_API_KEY=nonproduction-test-key' in (tmp_path / '.env').read_text(encoding='utf-8')
    dialog.close()
