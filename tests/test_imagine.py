"""Cloud image edit: pluggable OpenAI-compatible Images, no local SD."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from mural_agent.config import Settings
from mural_agent.imagine import ImagineClient, ImagineConfig, restore_with_imagine
from mural_agent.planner import Step
from mural_agent.professional import contest_allows_pixel_export
from mural_agent.tools import ToolRegistry


class _Response:
    def __init__(self, payload, status=200, content=b''):
        self._payload = payload
        self.status_code = status
        self.content = content
        self.text = ''

    def json(self):
        return self._payload


def _png_b64(color=(10, 180, 40), size=(6, 4)):
    image = Image.new('RGB', size, color)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _settings(tmp_path, **kwargs):
    return Settings(tmp_path / 'data', tmp_path / 'missing.pt', **kwargs)


def test_missing_key_is_explicit_and_does_not_call_the_network(tmp_path):
    settings = _settings(tmp_path, imagine_api_key='', imagine_base_url='https://images.example.test/v1',
                         imagine_model='other-vendor-model')
    called = {'n': 0}

    def transport(*_args, **_kwargs):
        called['n'] += 1
        raise AssertionError('未配置密钥时不应发出请求')

    original = np.zeros((4, 6, 3), np.uint8)
    mask = np.zeros((4, 6), np.uint8)
    mask[1:3, 2:4] = 255
    with pytest.raises(ValueError, match='API Key'):
        restore_with_imagine(settings, original, mask, '填露底', transport=transport)
    assert called['n'] == 0


def test_configured_key_posts_to_the_selected_host_and_composites(tmp_path):
    settings = _settings(
        tmp_path,
        imagine_provider='openai_json',
        imagine_base_url='https://images.example.test/v1',
        imagine_model='other-vendor-model',
        imagine_api_key='sk-test-not-real',
    )
    seen = {}

    def transport(method, url, headers=None, json=None, data=None, files=None, timeout=None):
        seen.update(method=method, url=url, headers=headers or {}, body=json, timeout=timeout)
        return _Response({'data': [{'b64_json': _png_b64()}]})

    original = np.zeros((8, 10, 3), np.uint8)
    original[..., 0] = 30
    original[..., 1] = 70
    original[..., 2] = 110
    mask = np.zeros((8, 10), np.uint8)
    mask[2:5, 3:7] = 255
    filled, meta = restore_with_imagine(settings, original, mask, 'continue pigment', transport=transport)
    assert seen['url'] == 'https://images.example.test/v1/images/edits'
    assert seen['body']['model'] == 'other-vendor-model'
    assert seen['body']['image']['url'].startswith('data:image/png;base64,')
    assert seen['headers']['Authorization'] == 'Bearer sk-test-not-real'
    assert 'sk-test-not-real' not in str(meta)
    assert meta['provider'] == 'openai_json'
    assert meta['composited_into_mask'] is True
    assert np.array_equal(filled[mask == 0], original[mask == 0])
    assert filled.shape == original.shape


def test_multipart_provider_uses_form_fields_for_another_vendor(tmp_path):
    settings = _settings(
        tmp_path,
        imagine_provider='openai',
        imagine_base_url='https://api.vendor.example/v1',
        imagine_model='gpt-image-1',
        imagine_api_key='sk-form-not-real',
    )
    seen = {}

    def transport(method, url, headers=None, json=None, data=None, files=None, timeout=None):
        seen.update(url=url, data=data, files=files, json=json)
        return _Response({'data': [{'b64_json': _png_b64(size=(4, 4))}]})

    original = np.full((4, 4, 3), 20, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[0:2, 0:2] = 255
    config = ImagineConfig.from_settings(settings)
    assert config.provider == 'openai_multipart'
    _generated, meta = ImagineClient(config, transport=transport).edit(original, mask, '')
    assert seen['url'] == 'https://api.vendor.example/v1/images/edits'
    assert seen['json'] is None
    assert seen['data']['model'] == 'gpt-image-1'
    assert 'image' in seen['files'] and 'mask' in seen['files']
    assert meta['provider'] == 'openai_multipart'
    assert 'sk-form-not-real' not in str(meta)


def test_diffusion_tool_calls_cloud_edit_instead_of_a_local_worker(tmp_path, monkeypatch):
    settings = _settings(tmp_path, imagine_api_key='sk-test-not-real', imagine_model='grok-imagine-image-2.0',
                         imagine_base_url='https://api.x.ai/v1')

    class Engine:
        lama = type('Lama', (), {'available': staticmethod(lambda: False), 'unload': staticmethod(lambda: None)})()
        sam = type('Sam', (), {'available': staticmethod(lambda: False), 'unload': staticmethod(lambda: None)})()
        store = None

        def __init__(self):
            self.settings = settings

    called = {}

    def fake_edit(self, original, mask, prompt=''):
        called['model'] = self.config.model
        called['prompt'] = prompt
        return np.array(original), {'model': self.config.model, 'provider': self.config.provider}

    monkeypatch.setattr(ImagineClient, 'edit', fake_edit)
    engine = Engine()
    assert ToolRegistry(engine).availability()['diffusion'] is True
    folder = tmp_path / 'attempt'
    folder.mkdir()
    original = np.zeros((4, 4, 3), np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[1:3, 1:3] = 255
    step = Step(tool='diffusion', reason='云端填补', prompt='keep lines')
    output, meta = ToolRegistry(engine).execute(step, original, mask, {'mode': 'enhanced', 'instructions': ''}, folder, 30)
    assert called['model'] == 'grok-imagine-image-2.0'
    assert output.shape == original.shape
    assert meta['model'] == 'grok-imagine-image-2.0'
    assert not (folder / 'request.json').exists()
    settings.imagine_api_key = ''
    assert ToolRegistry(engine).availability()['diffusion'] is False


def test_blank_key_is_not_written_over_the_stored_secret():
    from mural_agent.imagine import imagine_env_updates
    kept = imagine_env_updates({
        'provider': 'xai', 'base_url': 'https://api.x.ai/v1', 'model': 'grok-imagine-image-2.0', 'key': '',
    })
    assert 'MURAL_IMAGINE_API_KEY' not in kept
    assert kept['MURAL_IMAGINE_PROVIDER'] == 'openai_json'
    saved = imagine_env_updates({
        'provider': 'openai_json', 'base_url': 'https://images.example.test/v1',
        'model': 'other-vendor-model', 'key': 'sk-local-not-real',
    })
    assert saved['MURAL_IMAGINE_API_KEY'] == 'sk-local-not-real'
    assert saved['MURAL_IMAGINE_MODEL'] == 'other-vendor-model'


def test_contest_export_still_blocks_generative_finals():
    assert contest_allows_pixel_export({
        'contest_demo': True, 'mode': 'enhanced', 'evidence_level': 'enhanced',
        'selected_as': 'enhanced', 'candidates': [{'id': 'c', 'tool': 'diffusion', 'enhanced': True}],
        'selected_id': 'c',
    }) is False
    assert contest_allows_pixel_export({
        'contest_demo': True, 'mode': 'speculative', 'evidence_level': 'speculative',
    }) is False
