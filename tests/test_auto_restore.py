"""Unit tests for the one-click autonomous restoration routing helpers."""
import numpy as np

import pytest

from mural_agent.auto_restore import (build_steps, choose_backend, merge_masks,
                                      route_items, select_items, validate_assessment)


def item(kind, size='small', confidence=0.9, index=1):
    return {'index': index, 'type': kind, 'size': size, 'confidence': confidence, 'bbox': [100, 100, 160, 700]}


def test_select_items_filters_low_confidence_and_caps_count():
    items = [item('crack', confidence=c, index=index) for index, c in enumerate([0.2, 0.9, 0.5, 0.8])]
    selected = select_items(items, min_confidence=0.35, max_items=2)
    assert [entry['index'] for entry in selected] == [1, 3]


def test_route_items_downgrades_lama_when_weights_missing():
    routes = route_items([item('paint_loss', size='large')], {'lama': False})
    assert routes[0]['tool'] == 'telea'
    assert 'LaMa 未安装' in routes[0]['method']
    routes = route_items([item('paint_loss', size='large')], {'lama': True})
    assert routes[0]['tool'] == 'lama'


def test_choose_backend_prefers_lama_when_it_is_the_majority_route():
    routes = [{'tool': 'lama'}, {'tool': 'lama'}, {'tool': 'telea'}]
    assert choose_backend(routes) == 'lama'
    assert choose_backend([{'tool': 'telea'}, {'tool': 'ns'}]) == 'telea'
    assert choose_backend([]) == 'telea'


def test_build_steps_deduplicates_tools_and_caps_at_three():
    items = [item('crack'), item('stain'), item('paint_loss'), item('flaking')]
    routes = [{'tool': 'ns', 'method': 'NS'}, {'tool': 'color', 'method': 'match'},
              {'tool': 'telea', 'method': 'Telea'}, {'tool': 'telea', 'method': 'Telea'}]
    steps = build_steps(items, routes)
    assert [step['tool'] for step in steps] == ['ns', 'color', 'telea']
    assert all('AI 自主规划' in step['reason'] for step in steps)


def test_select_items_rejects_noise_sized_boxes():
    tiny = {'index': 1, 'type': 'paint_loss', 'size': 'small', 'confidence': 0.9, 'bbox': [0, 0, 10, 10]}
    real = {'index': 2, 'type': 'crack', 'size': 'small', 'confidence': 0.9, 'bbox': [100, 100, 160, 700]}
    selected = select_items([tiny, real])
    assert [item['index'] for item in selected] == [2]


def test_truncated_vlm_output_is_not_auto_submitted():
    assert validate_assessment({'limitations': '完整输出'}) == '完整输出'
    with pytest.raises(ValueError, match='截断'):
        validate_assessment({'limitations': '模型输出被截断；仅保留已完整解析的候选，需人工检查。'})


def test_merge_masks_respects_the_total_area_cap():
    shape = (100, 100)
    small = np.zeros(shape, np.uint8); small[0:30, 0:30] = 255
    huge = np.zeros(shape, np.uint8); huge[:, :] = 255
    union, accepted = merge_masks([small, huge], shape, max_fraction=0.25)
    assert accepted == [0]
    assert 0 < (union > 0).mean() <= 0.25
