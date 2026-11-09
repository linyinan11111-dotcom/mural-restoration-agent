"""Professional step-by-step restoration gates.

AI proposals are candidates only. Experts confirm masks, supply rationale, and
decide evidence level. Observation never executes. Speculative results stay
hypothesis-only unless the user explicitly archives them as such.

Evidence / mode ladder (least → most generative):
  observation  — note only, no pixels
  conservative — texture fill (LaMa / OpenCV); default safe path
  enhanced     — 加强修复: optional bolder fill after conservative (may use
                 diffusion on large lacunae); labeled 非历史定稿; opt-in
  evidence     — reference-aligned reconstruction only
  speculative  — most generative; hypothesis archive required to "select"
"""

WORKFLOW_STANDARD = 'standard'
WORKFLOW_PROFESSIONAL = 'professional_step'
WORKFLOWS = {WORKFLOW_STANDARD, WORKFLOW_PROFESSIONAL}

# observation = noted damage, no pixel edit allowed
# conservative / enhanced / evidence / speculative = allowed execution tiers
EVIDENCE_LEVELS = {'observation', 'conservative', 'enhanced', 'evidence', 'speculative'}
EXECUTABLE_LEVELS = {'conservative', 'enhanced', 'evidence', 'speculative'}
MASK_SOURCES = {'ai_candidate', 'manual', 'sam', 'crack'}

# Risk annotation classes (separate from repair mask_source).
# Canonical vocabulary lives in risk_regions; this module re-exports it for
# older callers and never keeps a second copy of the enum.
from .risk_regions import (
    REGION_KINDS as _RISK_REGION_KINDS,
    REGION_LABELS,
    REGION_ALIASES as _RISK_ALIASES,
    normalize_region_kind as _normalize_risk_kind,
    derive_risk_flags,
)

REGION_KINDS = set(_RISK_REGION_KINDS)
REGION_ALIASES = dict(_RISK_ALIASES)
REGION_OVERLAY = {
    'fillable_lacuna': (245, 166, 35, 150),  # amber
    'uncertain': (220, 0, 60, 170),          # red translucent
    'unrestorable': (180, 0, 40, 200),       # deep red / hatch in UI
}
REVIEW_DECISIONS = {'adopt', 'modify', 'reject'}
DECISION_LABELS = {
    'adopt': '采纳',
    'modify': '修改后采纳',
    'reject': '驳回',
}
# Human-facing labels for UI / archives
LEVEL_LABELS = {
    'observation': '观察（不执行）',
    'conservative': '保守修复',
    'enhanced': '加强修复·非历史定稿',
    'evidence': '证据辅助',
    'speculative': '推测性补全·假设性建议',
}

# Retain more observed context during enhanced inpainting.
ENHANCED_DIFFUSION_STRENGTH = 0.68
ENHANCED_DIFFUSION_PROMPT = (
    'conservative inpainting of the masked damaged area in an aged Chinese mural, '
    'continue only immediately adjacent colors, contours, and mineral pigment texture, '
    'preserve the original composition and all visible paint, lines, cracks, and patina, '
    'subtle weathered surface, no new figures, symbols, or decorative motifs'
)
ENHANCED_NEGATIVE_PROMPT = (
    'white blank patch, flat solid fill, erased linework, clean new paint, modern objects, '
    'text, watermark, neon colors, photorealistic face, invented patterns, sharp CGI, oversaturated'
)
ENHANCED_UI_HINT = '加强修复仅处理选区及附近缺损，不抹除未选中的裂纹与原画线条（仍非历史定稿）'


def normalize_workflow(value):
    workflow = (value or WORKFLOW_STANDARD).strip().lower()
    if workflow not in WORKFLOWS:
        raise ValueError('不支持的工作流；请使用 standard 或 professional_step。')
    return workflow


def normalize_evidence_level(value):
    level = (value or '').strip().lower()
    if level not in EVIDENCE_LEVELS:
        raise ValueError('证据等级须为 observation / conservative / enhanced / evidence / speculative。')
    return level


def normalize_mask_source(value, default='manual'):
    source = (value or default or 'manual').strip().lower()
    if source not in MASK_SOURCES:
        raise ValueError('掩膜来源须为 ai_candidate / manual / sam / crack。')
    return source


def require_rationale(text, *, label='依据'):
    rationale = (text or '').strip()
    if not rationale:
        raise ValueError(f'专业分步模式须填写非空{label}，不得凭空修复。')
    return rationale[:2000]


def validate_professional_create(mask_record, *, rationale, evidence_level, mode,
                                 expert_notes='', mask_source=None, allow_enhanced=False,
                                 risk_acknowledged=False, mask_gate_confirmed=False,
                                 has_unrestorable=False, has_uncertain=False,
                                 force_override=False, force_override_reason='',
                                 allow_generative_attempt=False):
    """Gate before enqueueing a professional_step job.

    Risk flags are derived from the frozen mask record; the ``has_*``
    parameters are accepted for backwards compatibility only and are ignored.
    """
    if not mask_record.get('user_confirmed'):
        raise ValueError('专业分步：掩膜尚未经人工确认，不能创建或执行修复任务。')
    if not mask_gate_confirmed:
        raise ValueError('专业分步：请勾选「已核对掩膜与依据」后再进入下一档。')
    rationale = require_rationale(rationale)
    level = normalize_evidence_level(evidence_level)
    if level == 'observation':
        raise ValueError('专业分步：observation 仅可记录观察，禁止执行像素修复。')
    source = normalize_mask_source(mask_source or mask_record.get('mask_source') or 'manual')
    if source == 'ai_candidate' and not mask_record.get('user_confirmed'):
        raise ValueError('专业分步：AI 候选掩膜须先由专家确认后再建任务。')
    flags = derive_risk_flags(mask_record)
    if mask_record.get('expert_force_override') and not force_override:
        force_override = True
        force_override_reason = force_override_reason or mask_record.get('force_override_reason') or ''
    if allow_generative_attempt:
        if mode != 'speculative':
            raise ValueError('AI 尝试性补全必须在 speculative 模式下执行。')
        if not risk_acknowledged:
            raise ValueError('AI 尝试性补全须勾选风险提示：结果是推测，不是历史原貌。')
    # Speculative is off by default: mode and level must both opt in with rationale.
    if mode == 'speculative' or level == 'speculative':
        if mode != 'speculative' or level != 'speculative':
            raise ValueError('专业分步：推测修复须同时将修复模式与证据等级设为 speculative，并写明理由。')
        require_rationale(rationale, label='推测理由')
    # Enhanced sits between conservative and speculative; requires explicit opt-in.
    if mode == 'enhanced' or level == 'enhanced':
        if mode != 'enhanced' or level != 'enhanced':
            raise ValueError('专业分步：加强修复须同时将修复模式与证据等级设为 enhanced。')
        if not allow_enhanced:
            raise ValueError('专业分步：加强修复须勾选「允许加强」，且应在保守结果基础上再执行。')
        if not risk_acknowledged:
            raise ValueError('专业分步：加强修复须勾选风险提示（文化边界 / 非历史定稿）已知悉。')
        require_rationale(rationale, label='加强修复依据')
    # The experimental generative path is an explicit expert override for large
    # losses. It still records risk flags but does not pretend to be evidence.
    if not allow_generative_attempt:
        if flags['has_unrestorable']:
            raise ValueError('专业分步：存在不可复原（unrestorable）标红区，禁止任何像素修复；仅可导出标注图与报告。')
        if flags['has_uncertain']:
            if level != 'conservative':
                raise ValueError('专业分步：不确定（uncertain）红区默认禁止加强/推测；仅可在强制放行后做保守修补。')
            if not force_override:
                raise ValueError('专业分步：不确定红区须勾选「专家强制放行」并填写理由，才可保守修补。')
            require_rationale(force_override_reason, label='强制放行理由')
    notes = (expert_notes or '').strip()[:2000]
    return {
        'rationale': rationale,
        'evidence_level': level,
        'expert_notes': notes,
        'mask_source': source,
        'allow_enhanced': bool(allow_enhanced) if level == 'enhanced' else False,
        'risk_acknowledged': bool(risk_acknowledged) if (level == 'enhanced' or allow_generative_attempt) else False,
        'mask_gate_confirmed': True,
        'has_unrestorable': bool(flags['has_unrestorable']),
        'has_uncertain': bool(flags['has_uncertain']),
        'force_override': bool(force_override) if flags['has_uncertain'] else False,
        'force_override_reason': (force_override_reason or '').strip()[:2000] if flags['has_uncertain'] else '',
        'allow_generative_attempt': bool(allow_generative_attempt),
    }


def enforce_mode_and_risk(*, mode, mask_record, mask_gate_confirmed=False, allow_enhanced=False,
                          risk_acknowledged=False, force_override=False,
                          force_override_reason='', allow_generative_attempt=False):
    """Universal gates for both standard and professional_step workflows.

    These are enforced at job creation *and* again before execution, so a
    standard-workflow job cannot bypass enhanced/speculative/risk gates.
    """
    flags = derive_risk_flags(mask_record)
    if allow_generative_attempt:
        if mode != 'speculative':
            raise ValueError('AI 尝试性补全必须在 speculative 模式下执行。')
        if not mask_gate_confirmed:
            raise ValueError('专业分步：请勾选「已核对掩膜与依据」后再进入下一档。')
        if not risk_acknowledged:
            raise ValueError('AI 尝试性补全须勾选风险提示：结果是推测，不是历史原貌。')
        return flags
    if flags['has_unrestorable']:
        raise ValueError('专业分步：存在不可复原（unrestorable）标红区，禁止任何像素修复；仅可导出标注图与报告。')
    if flags['has_uncertain']:
        if mode != 'conservative':
            raise ValueError('专业分步：不确定（uncertain）红区默认禁止加强/推测；仅可在强制放行后做保守修补。')
        if not force_override:
            raise ValueError('专业分步：不确定红区须勾选「专家强制放行」并填写理由，才可保守修补。')
        require_rationale(force_override_reason, label='强制放行理由')
    if mode == 'enhanced':
        if not mask_gate_confirmed:
            raise ValueError('专业分步：请勾选「已核对掩膜与依据」后再进入下一档。')
        if not allow_enhanced:
            raise ValueError('专业分步：加强修复须勾选「允许加强」，且应在保守结果基础上再执行。')
        if not risk_acknowledged:
            raise ValueError('专业分步：加强修复须勾选风险提示（文化边界 / 非历史定稿）已知悉。')
    if mode == 'speculative':
        if not mask_gate_confirmed:
            raise ValueError('专业分步：请勾选「已核对掩膜与依据」后再进入下一档。')
        if not risk_acknowledged:
            raise ValueError('专业分步：推测性补全须勾选风险提示已知悉。')
    return flags


def validate_job_record_for_execution(job, mask_record):
    """Defense-in-depth check at engine.run.

    Runs for every workflow: standard jobs must not bypass enhanced/speculative
    or risk gates either. professional_step additionally re-checks rationale and
    evidence-level consistency.
    """
    enforce_mode_and_risk(
        mode=job.get('mode', 'conservative'),
        mask_record=mask_record,
        mask_gate_confirmed=bool(job.get('mask_gate_confirmed')),
        allow_enhanced=bool(job.get('allow_enhanced')),
        risk_acknowledged=bool(job.get('risk_acknowledged')),
        force_override=bool(job.get('force_override')),
        force_override_reason=job.get('force_override_reason') or '',
        allow_generative_attempt=bool(job.get('allow_generative_attempt')),
    )
    if normalize_workflow(job.get('workflow')) != WORKFLOW_PROFESSIONAL:
        return
    validate_professional_create(
        mask_record,
        rationale=job.get('rationale') or job.get('instructions'),
        evidence_level=job.get('evidence_level'),
        mode=job.get('mode', 'conservative'),
        expert_notes=job.get('expert_notes', ''),
        mask_source=job.get('mask_source'),
        allow_enhanced=bool(job.get('allow_enhanced')),
        risk_acknowledged=bool(job.get('risk_acknowledged')),
        mask_gate_confirmed=bool(job.get('mask_gate_confirmed', True)),
        force_override=bool(job.get('force_override')),
        force_override_reason=job.get('force_override_reason') or '',
        allow_generative_attempt=bool(job.get('allow_generative_attempt')),
    )


def is_enhanced_candidate(job, candidate):
    return (
        bool(candidate.get('enhanced'))
        or job.get('mode') == 'enhanced'
        or job.get('evidence_level') == 'enhanced'
    )


def validate_candidate_selection(job, candidate, *, archive_as_hypothesis=False, decision=None):
    """Block one-click final selection of speculative/diffusion outputs.

    Enhanced results may be selected but are stamped selected_as='enhanced'
    (加强修复·非历史定稿) — they never replace a conservative archival path.

    Review hard gate: professional_step requires an explicit review decision
    (adopt / modify / reject). Reject stops the ladder without accepting pixels.
    """
    workflow = normalize_workflow(job.get('workflow') or WORKFLOW_STANDARD)
    if workflow == WORKFLOW_PROFESSIONAL or decision:
        choice = validate_review_decision(decision, require=True)
        if choice == 'reject':
            return {
                'selected_as': 'rejected',
                'archive_as_hypothesis': False,
                'decision': 'reject',
                'label': DECISION_LABELS['reject'],
            }
    else:
        choice = None
    enhanced = is_enhanced_candidate(job, candidate) and job.get('mode') != 'speculative'
    speculative = (
        bool(candidate.get('speculative'))
        or job.get('mode') == 'speculative'
        or (candidate.get('tool') == 'diffusion' and not enhanced)
    )
    if enhanced:
        return {
            'selected_as': 'enhanced',
            'archive_as_hypothesis': False,
            'decision': choice or 'adopt',
            'label': LEVEL_LABELS['enhanced'],
        }
    if not speculative:
        return {
            'selected_as': 'accepted' if choice != 'modify' else 'modified',
            'archive_as_hypothesis': False,
            'decision': choice or 'adopt',
        }
    if archive_as_hypothesis:
        return {
            'selected_as': 'hypothesis',
            'archive_as_hypothesis': True,
            'decision': choice or 'adopt',
        }
    if normalize_workflow(job.get('workflow')) == WORKFLOW_PROFESSIONAL or speculative:
        raise ValueError('推测/扩散结果不能直接定为选定候选；请勾选「仅作假设性建议归档」后再保存。')
    return {
        'selected_as': 'accepted',
        'archive_as_hypothesis': False,
        'decision': choice or 'adopt',
    }


def normalize_region_kind(value, default='fillable_lacuna'):
    return _normalize_risk_kind(value, default)


def validate_selected_export(job, candidate):
    """Validate that the selected candidate may leave the app as a pixel result."""
    selected_as = (job.get('selected_as') or '').strip().lower()
    review = job.get('review') or {}
    if selected_as == 'rejected' or (review.get('decision') or '').strip().lower() == 'reject':
        raise ValueError('该候选已被驳回，禁止导出或保存修复成图。')
    if selected_as == 'hypothesis' and not review.get('archive_as_hypothesis'):
        raise ValueError('该候选是假设性结果，须以「假设性建议归档」方式保存。')
    return {'selected_as': selected_as or 'accepted',
            'archive_as_hypothesis': bool(review.get('archive_as_hypothesis')),
            'decision': review.get('decision')}


def validate_review_decision(decision, *, require=True):
    """Hard gate: adopt / modify / reject before leaving the candidate step."""
    value = (decision or '').strip().lower()
    if not value:
        if require:
            raise ValueError('专业分步：须选择采纳 / 修改后采纳 / 驳回，未选择不得进入下一档或导出终稿。')
        return None
    if value not in REVIEW_DECISIONS:
        raise ValueError('复核决定须为 adopt / modify / reject。')
    return value
