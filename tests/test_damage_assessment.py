"""Pure-function tests for the local vision-model damage proposal parser."""
from mural_agent.damage_assessment import merge_adjacent_candidates, parse_model_output, recommendation_for


def test_full_json_merges_adjacent_candidates_of_the_same_type():
    text = ('{"items": ['
            '{"type": "crack", "bbox": [100, 100, 180, 140], "size": "small", "confidence": 0.8, "evidence": "dark line"},'
            '{"type": "crack", "bbox": [185, 110, 240, 150], "size": "medium", "confidence": 0.6, "evidence": "continues"},'
            '{"type": "stain", "bbox": [600, 600, 650, 640], "size": "small", "confidence": 0.5, "evidence": "stain"}'
            '], "limitations": "visual only"}')
    parsed = parse_model_output(text)
    assert len(parsed['items']) == 2
    crack = parsed['items'][0]
    assert crack['bbox'] == [100, 100, 240, 150]
    assert crack['index'] == 1
    assert parsed['limitations'] == 'visual only'


def test_truncated_output_keeps_only_complete_candidates():
    text = ('{"items": ['
            '{"type": "crack", "bbox": [10, 20, 120, 80], "size": "small", "confidence": 0.8, "evidence": "a"},'
            '{"type": "paint_loss", "bbox": [200, 200, 260, 260], "size": "medium", "confidence": 0.7, "evidence": "b"},'
            '{"type": "flaking", "bbox": [300, 30')
    parsed = parse_model_output(text)
    assert len(parsed['items']) == 2
    assert '截断' in parsed['limitations']


def test_invalid_candidates_are_dropped_not_fatal():
    items = merge_adjacent_candidates([
        {'type': 'unknown', 'bbox': [0, 0, 10, 10]},
        {'type': 'crack', 'bbox': [1, 2, 3]},
        {'type': 'crack', 'bbox': [0, 0, 1, 1]},
        {'type': 'crack', 'bbox': [-5, -5, 20, 20], 'size': 'huge', 'confidence': 9},
        'not a dict',
    ])
    assert len(items) == 1
    assert items[0]['bbox'] == [0, 0, 20, 20]
    assert items[0]['size'] == 'uncertain'
    assert items[0]['confidence'] == 1.0


def test_duplicates_merge_and_parallel_overlapping_cracks_stay_separate():
    items = [
        {'type': 'crack', 'bbox': [127, 127, 187, 835], 'size': 'large', 'confidence': 0.9, 'evidence': 'left'},
        {'type': 'crack', 'bbox': [127, 127, 187, 835], 'size': 'large', 'confidence': 0.8, 'evidence': 'left again'},
        {'type': 'crack', 'bbox': [485, 127, 545, 865], 'size': 'large', 'confidence': 0.9, 'evidence': 'right a'},
        {'type': 'crack', 'bbox': [525, 127, 585, 865], 'size': 'large', 'confidence': 0.9, 'evidence': 'right b'},
    ]
    merged = merge_adjacent_candidates(items)
    boxes = [item['bbox'] for item in merged]
    assert boxes.count([127, 127, 187, 835]) == 1
    assert len(merged) == 3, boxes


def test_recommendation_routing_matches_the_paired_studies():
    assert recommendation_for({'type': 'crack', 'size': 'small'})['tool'] == 'ns'
    assert recommendation_for({'type': 'crack', 'size': 'large'})['tool'] == 'lama'
    assert recommendation_for({'type': 'paint_loss', 'size': 'large'})['tool'] == 'lama'
    assert recommendation_for({'type': 'paint_loss', 'size': 'small'})['tool'] == 'telea'
    assert recommendation_for({'type': 'stain', 'size': 'medium'})['tool'] == 'color'
    assert recommendation_for({'type': 'other', 'size': 'uncertain'})['backend'] == 'opencv'
