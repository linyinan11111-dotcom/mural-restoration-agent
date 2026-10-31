import json
import httpx
import pytest
from PIL import Image
from pydantic import ValidationError
from mural_agent.config import Settings
from mural_agent.planner import api_plan, rule_plan


@pytest.mark.parametrize('kind', ['responses', 'chat_completions'])
def test_api_protocol_and_local_plan_validation(tmp_path, monkeypatch, kind):
    settings = Settings(tmp_path, tmp_path / 'model.pt', api_kind=kind, api_key='private-token', api_model='vision-test')
    expected = rule_plan('opencv', 2, .02).model_dump()
    captured = []
    def handler(request):
        captured.append(request)
        if kind == 'responses':
            return httpx.Response(200, json={'output': [{'content': [{'type': 'output_text', 'text': json.dumps(expected)}]}]})
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(expected)}}]})
    original_client = httpx.Client
    monkeypatch.setattr('mural_agent.planner.httpx.Client', lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    image = Image.new('RGB', (20, 20))
    plan = api_plan(settings, image, image, 'opencv', 2, .02, '只修裂缝')
    assert plan.source == 'api'
    assert len(plan.steps) == 2
    body = json.loads(captured[0].content)
    assert body['model'] == 'vision-test'
    assert captured[0].headers['Authorization'] == 'Bearer private-token'
    if kind == 'responses':
        assert captured[0].url.path.endswith('/responses')
        assert body['store'] is False
        assert body['input'][0]['content'][1]['image_url'].startswith('data:image/jpeg;base64,')
    else:
        assert captured[0].url.path.endswith('/chat/completions')
        assert body['response_format'] == {'type': 'json_object'}


def test_api_cannot_introduce_arbitrary_tools(tmp_path, monkeypatch):
    settings = Settings(tmp_path, tmp_path / 'model.pt', api_key='private-token', api_model='vision-test')
    invalid = rule_plan('opencv', 1, .01).model_dump()
    invalid['steps'][0]['tool'] = 'execute_shell'
    original_client = httpx.Client
    handler = lambda request: httpx.Response(200, json={'output': [{'content': [{'type': 'output_text', 'text': json.dumps(invalid)}]}]})
    monkeypatch.setattr('mural_agent.planner.httpx.Client', lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    image = Image.new('RGB', (20, 20))
    with pytest.raises(ValidationError):
        api_plan(settings, image, image, 'opencv', 1, .01, '')

