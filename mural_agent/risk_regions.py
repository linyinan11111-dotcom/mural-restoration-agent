"""Competition risk region kinds for professional_step.

REGION_KINDS
------------
repairable    — small/medium lacuna that may accept conservative texture fill
uncertainty   — large lacuna without line continuity; red hatch; default block repair
unrestorable  — inventing form against modern mount; solid red; display-only

Competition: **forbid generating mural patterns/images**. Demo export = annotation
overlay + review/refusal records only — never enhanced/speculative/restored finals.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

REGION_KINDS = ('repairable', 'uncertainty', 'unrestorable')
BLOCKING_KINDS = frozenset({'uncertainty', 'unrestorable'})
# Aliases from earlier drafts / external docs
_ALIASES = {
    'fillable_lacuna': 'repairable',
    'fillable': 'repairable',
    'uncertain': 'uncertainty',
}

REGION_LABELS = {
    'repairable': '可保守填的露底',
    'uncertainty': '高不确定（须标红）',
    'unrestorable': '不可复原主张区（禁止补绘）',
}

SMALL_FRAC = 0.008
LARGE_FRAC = 0.03
MOUNT_TOUCH_FRAC = 0.12
LINE_SCORE_OK = 0.28

# Back-compat names
REGION_CLASSES = REGION_KINDS
BLOCKING_CLASSES = BLOCKING_KINDS


def normalize_region_kind(value, default='repairable'):
    klass = (value or default or 'repairable').strip().lower()
    klass = _ALIASES.get(klass, klass)
    if klass not in REGION_KINDS:
        raise ValueError('区域类别须为 repairable / uncertainty / unrestorable。')
    return klass


# Alias
normalize_region_class = normalize_region_kind


def aggregate_region_kind(kinds: Iterable[str]) -> str:
    found = {normalize_region_kind(c) for c in kinds if c}
    if not found:
        return 'repairable'
    if 'unrestorable' in found:
        return 'unrestorable'
    if 'uncertainty' in found:
        return 'uncertainty'
    return 'repairable'


aggregate_region_class = aggregate_region_kind


def detect_mount_mask(rgb: np.ndarray) -> np.ndarray:
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    L = lab[..., 0]
    R = rgb[..., 0].astype(np.int16)
    G = rgb[..., 1].astype(np.int16)
    B = rgb[..., 2].astype(np.int16)
    olive = (
        (L >= 70) & (L <= 125) & (V >= 60) & (V <= 130)
        & (S >= 25) & (S <= 110) & (H >= 10) & (H <= 45)
        & (np.abs(R - G) <= 18) & ((R + G) / 2 > B + 5)
    )
    seedmap = (olive.astype(np.uint8) * 255)
    seedmap = cv2.morphologyEx(seedmap, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    h, w = L.shape
    reachable = np.zeros((h, w), np.uint8)
    step = 15
    seeds = []
    for x in range(0, w, step):
        seeds += [(x, 0), (x, h - 1)]
    for y in range(0, h, step):
        seeds += [(0, y), (w - 1, y)]
    for x, y in seeds:
        if seedmap[y, x] == 0:
            continue
        tmp = seedmap.copy()
        ffmask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(tmp, ffmask, (x, y), 128, loDiff=0, upDiff=0, flags=4)
        reachable |= (tmp == 128).astype(np.uint8)
    return cv2.dilate(reachable * 255, np.ones((5, 5), np.uint8), iterations=2)


def _line_continuation_score(rgb: np.ndarray, component: np.ndarray, exclude: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(component, kernel, iterations=3)
    ring = (dil > 0) & (component == 0) & (exclude == 0)
    ring_count = int(ring.sum())
    if ring_count < 20:
        return 0.0
    return float(edges[ring].astype(bool).mean())


def _mount_touch_fraction(component: np.ndarray, mount: np.ndarray) -> float:
    import cv2
    boundary = cv2.dilate(component, np.ones((3, 3), np.uint8), iterations=1)
    peri = (boundary > 0) & (component == 0)
    peri_n = int(peri.sum())
    if peri_n < 8:
        return 0.0
    return float(((peri) & (mount > 0)).sum() / peri_n)


def classify_component(*, area_frac: float, mount_touch: float, line_score: float):
    if mount_touch >= MOUNT_TOUCH_FRAC and line_score < LINE_SCORE_OK:
        return (
            'unrestorable',
            '残片外缘紧贴现代托底且周围无线稿/颜料延续；禁止生成纹样或想象补绘，仅可展示。',
        )
    if area_frac >= LARGE_FRAC and line_score < LINE_SCORE_OK:
        return (
            'uncertainty',
            '大面积露底且周边无线稿延续；高不确定，默认禁止补绘/加强/推测，须专家改判或强制放行。',
        )
    if area_frac <= SMALL_FRAC:
        return (
            'repairable',
            '小块可见露底；可候选保守纹理填补（非生成纹样），仍须人工确认。',
        )
    if line_score >= LINE_SCORE_OK:
        return (
            'repairable',
            '露底周边可见边缘/线稿延续迹象；可候选保守填补，禁止凭空生成图案。',
        )
    return (
        'uncertainty',
        '露底规模与线稿依据不足；标为不确定，默认禁止补绘。',
    )


def classify_mask_regions(rgb: np.ndarray, lacuna_mask: np.ndarray, mount_mask=None,
                          min_area: int | None = None) -> dict:
    import cv2
    rgb = np.asarray(rgb)
    lacuna = (np.asarray(lacuna_mask) > 0).astype(np.uint8)
    if mount_mask is None:
        mount = detect_mount_mask(rgb)
    else:
        mount = (np.asarray(mount_mask) > 0).astype(np.uint8) * 255
    h, w = lacuna.shape[:2]
    total = max(1, h * w)
    if min_area is None:
        min_area = max(64, int(total * 0.0003))
    lacuna = lacuna.copy()
    lacuna[mount > 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lacuna, 8)
    regions = []
    kind_masks = {k: np.zeros((h, w), np.uint8) for k in REGION_KINDS}
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, bw, bh = (int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]),
                        int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]))
        component = (labels == index).astype(np.uint8) * 255
        area_frac = area / total
        mount_touch = _mount_touch_fraction(component, mount)
        line_score = _line_continuation_score(rgb, component, mount)
        kind, evidence = classify_component(
            area_frac=area_frac, mount_touch=mount_touch, line_score=line_score)
        regions.append({
            'id': len(regions) + 1,
            'region_kind': kind,
            'region_class': kind,  # alias for older callers
            'label': REGION_LABELS[kind],
            'area_px': area,
            'area_frac': round(area_frac, 6),
            'bbox_xywh': [x, y, bw, bh],
            'mount_touch': round(mount_touch, 4),
            'line_score': round(line_score, 4),
            'evidence': evidence,
            'status': 'proposed',
            'repair_allowed': kind == 'repairable',
        })
        kind_masks[kind][component > 0] = 255
    aggregate = aggregate_region_kind(r['region_kind'] for r in regions)
    return {
        'regions': regions,
        'region_kind': aggregate,
        'region_class': aggregate,
        'class_masks': kind_masks,
        'kind_masks': kind_masks,
        'mount_fraction': float((mount > 0).mean()),
        'lacuna_fraction': float((lacuna > 0).mean()),
        'blocking': aggregate in BLOCKING_KINDS,
        'notice': (
            '风险分类为启发式演示：红斜线=uncertainty，深红实心=unrestorable；'
            '禁止生成纹样/图像。以专业人员改判为准。'
        ),
    }


def render_risk_overlay(rgb: np.ndarray, class_masks: dict,
                        alpha_repairable=0.40, alpha_uncertainty=0.50,
                        alpha_unrestorable=0.55) -> np.ndarray:
    """repairable=orange; uncertainty=red translucent; unrestorable=red hatch."""
    import cv2
    base = np.asarray(rgb).copy()
    h, w = base.shape[:2]
    overlay = base.astype(np.float32)
    repairable = np.asarray(class_masks.get('repairable', np.zeros((h, w))) > 0)
    uncertainty = np.asarray(class_masks.get('uncertainty', np.zeros((h, w))) > 0)
    unrestorable = np.asarray(class_masks.get('unrestorable', np.zeros((h, w))) > 0)
    if repairable.any():
        color = np.array([255.0, 152.0, 0.0], dtype=np.float32)  # orange
        overlay[repairable] = overlay[repairable] * (1 - alpha_repairable) + color * alpha_repairable
        out_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            (repairable.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_u8, contours, -1, (255, 140, 0), 2)
        overlay = out_u8.astype(np.float32)
    if uncertainty.any():
        color = np.array([220.0, 40.0, 40.0], dtype=np.float32)
        overlay[uncertainty] = overlay[uncertainty] * (1 - alpha_uncertainty) + color * alpha_uncertainty
        out_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            (uncertainty.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_u8, contours, -1, (200, 30, 30), 2)
        overlay = out_u8.astype(np.float32)
    if unrestorable.any():
        yy, xx = np.ogrid[:h, :w]
        hatch = ((xx + yy) % 8) < 3
        mark = unrestorable & hatch
        color = np.array([255.0, 50.0, 50.0], dtype=np.float32)
        overlay[mark] = overlay[mark] * (1 - alpha_unrestorable) + color * alpha_unrestorable
        out_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            (unrestorable.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_u8, contours, -1, (160, 0, 0), 3)
        overlay = out_u8.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _record_kind(mask_record: dict) -> str:
    raw = mask_record.get('region_kind') or mask_record.get('region_class') or 'repairable'
    try:
        return normalize_region_kind(raw)
    except ValueError:
        return 'uncertainty'


def mask_blocks_repair(mask_record: dict, *, expert_force_override: bool = False):
    if expert_force_override or mask_record.get('expert_force_override'):
        return False, ''
    kind = _record_kind(mask_record)
    if kind == 'unrestorable':
        return True, (
            '专业分步：当前选区含不可复原主张区（unrestorable）。'
            '禁止像素修复与生成纹样/图像；请改选区、标为可填，或专家强制放行并记档。'
        )
    if kind == 'uncertainty':
        return True, (
            '专业分步：当前选区含高不确定区（uncertainty，已标红）。'
            '默认禁止补绘、加强与推测生成；请确认可填、拒答跳过，或专家强制放行并记档。'
        )
    for region in mask_record.get('risk_regions') or []:
        rkind = normalize_region_kind(
            region.get('region_kind') or region.get('region_class') or 'repairable')
        status = (region.get('status') or '').strip().lower()
        if status in {'refused', 'skipped', 'fillable_confirmed', 'repairable_confirmed'}:
            continue
        if rkind in BLOCKING_KINDS and status not in {'force_approved'}:
            return True, (
                f"专业分步：区域 #{region.get('id')} 仍为 {rkind}，禁止补绘/生成。"
                '请「确认可填」「标为不确定/拒答」或专家强制放行。'
            )
    return False, ''


def apply_region_action(regions: list, region_id: int, action: str) -> list:
    action = (action or '').strip().lower()
    allowed = {'confirm_fillable', 'confirm_repairable', 'mark_uncertain',
               'mark_uncertainty', 'refuse_skip', 'force_approve'}
    if action not in allowed:
        raise ValueError(
            '动作须为 confirm_repairable / mark_uncertainty / refuse_skip / force_approve。')
    if action == 'confirm_fillable':
        action = 'confirm_repairable'
    if action == 'mark_uncertain':
        action = 'mark_uncertainty'
    updated = []
    found = False
    for region in regions:
        item = dict(region)
        if int(item.get('id', -1)) == int(region_id):
            found = True
            if action == 'confirm_repairable':
                item['region_kind'] = item['region_class'] = 'repairable'
                item['status'] = 'repairable_confirmed'
                item['repair_allowed'] = True
                item['label'] = REGION_LABELS['repairable']
                item['evidence'] = (item.get('evidence') or '') + '；专家确认可保守填（禁止生成纹样）。'
            elif action == 'mark_uncertainty':
                item['region_kind'] = item['region_class'] = 'uncertainty'
                item['status'] = 'uncertainty'
                item['repair_allowed'] = False
                item['label'] = REGION_LABELS['uncertainty']
            elif action == 'refuse_skip':
                item['status'] = 'refused'
                item['repair_allowed'] = False
                item['refusal'] = '专家拒答/跳过：不进入修复，不生成图像。'
            elif action == 'force_approve':
                item['status'] = 'force_approved'
                item['repair_allowed'] = True
                item['force_note'] = '专家强制放行（须记档）；仍禁止把生成纹样当作历史定稿。'
        updated.append(item)
    if not found:
        raise ValueError(f'未找到区域 id={region_id}。')
    return updated


def build_refusal_record(*, image_id, mask_id=None, region=None, reason='', actor='expert'):
    from .storage import timestamp
    return {
        'type': 'risk_refusal',
        'image_id': image_id,
        'mask_id': mask_id,
        'region_id': (region or {}).get('id'),
        'region_kind': (region or {}).get('region_kind') or (region or {}).get('region_class'),
        'reason': (reason or (region or {}).get('refusal') or '拒答跳过')[:2000],
        'actor': actor,
        'created_at': timestamp(),
        'policy': '禁止生成纹样/图像；拒答已记档。正赛导出仅标注层+复核记录。',
    }


def competition_export_allowed(job: dict | None = None, *, include_restored=False) -> tuple[bool, str]:
    """正赛导出：只允许标注层+复核记录；禁用 enhanced/speculative/修复成图。"""
    if include_restored:
        return False, '正赛导出禁用修复成图；仅允许 overlay_risk 与复核/拒答记录。'
    if job:
        mode = (job.get('mode') or '').strip().lower()
        level = (job.get('evidence_level') or '').strip().lower()
        if mode in {'enhanced', 'speculative'} or level in {'enhanced', 'speculative'}:
            return False, '正赛导出禁用 enhanced/speculative 产物；请导出风险标注与复核记录。'
    return True, ''
