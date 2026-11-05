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

# Contest annotation classes (separate from repair mask_source).
# fillable_lacuna = 可见露底，可走保守；uncertain = 红半透明，默认禁修；
# unrestorable = 红斜线，只展示禁止任何像素修复。
# Aliases repairable/uncertainty kept for older UI strokes.
REGION_KINDS = {'fillable_lacuna', 'uncertain', 'unrestorable', 'repairable', 'uncertainty'}
REGION_ALIASES = {
    'repairable': 'fillable_lacuna',
    'uncertainty': 'uncertain',
    'fillable_lacuna': 'fillable_lacuna',
    'uncertain': 'uncertain',
    'unrestorable': 'unrestorable',
}
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
# Contest demo: no mural generation as deliverable. Enhanced/speculative stay lab-only.
CONTEST_DEMO = True
CONTEST_EXECUTABLE = {'conservative'}  # demo mainline; observation never executes
CONTEST_LAB_ONLY = {'enhanced', 'speculative', 'evidence'}

# Demo copy (场景 A 保守 / 场景 B 加强) — UI reads these via bridge or hardcodes same strings.
DEMO_COPY = {
    'footer': '本流程辅助保护观察与方案讨论，不替代文物保护专业判断。',
    'A': {
        'step': 'AI 仅标疑似露底/残损区域，供对照；是否属实以人工确认为准。',
        'risk': '保守草案只做可据补全的露底填色，不补纹样、不补题记。未确认掩膜前，结果仅作观察参考。',
        'gate': '请勾选已核对掩膜与依据，再生成保守草案。',
    },
    'B': {
        'step': '加强结果含风格推断与纹样补全，属假设性预览，不能作为修复定稿。',
        'risk': '高不确定/无依据区域已标红，禁止当作史实还原。涉及人物、题记、宗教符号时，须专家复核后再外传。',
        'refuse': '依据不足时 AI 应停止补绘，并提示：建议仅保留原状记录与文献对照。',
        'gate': '须人工选择：采纳 / 修改 / 驳回；未选择不得进入下一档或导出终稿。',
    },
}

# Human-facing labels for UI / archives
LEVEL_LABELS = {
    'observation': '观察（不执行）',
    'conservative': '保守修复',
    'enhanced': '加强修复·非历史定稿',
    'evidence': '证据辅助',
    'speculative': '推测性补全·假设性建议',
}

# Default SD inpaint strength for 加强修复 (product band 0.9–0.95).
ENHANCED_DIFFUSION_STRENGTH = 0.92
ENHANCED_DIFFUSION_PROMPT = (
    'aged Chinese temple mural fragment, mineral pigments, soft weathered paint surface, '
    'fill lacunae and eliminate visible craquelure and crack networks so the paint reads continuous, '
    'match surrounding texture hue and grain, seamless inpainting, no new iconography'
)
ENHANCED_NEGATIVE_PROMPT = (
    'visible cracks, craquelure, fissures, fracture lines, modern objects, text, watermark, '
    'neon colors, photorealistic face, invented patterns, sharp CGI, oversaturated'
)
ENHANCED_UI_HINT = '加强修复会处理露底并弱化/消除可见龟裂纹（仍非历史定稿）'


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
                                 contest_demo=None):
    """Gate before enqueueing a professional_step job."""
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
    contest = CONTEST_DEMO if contest_demo is None else bool(contest_demo)
    if has_unrestorable:
        raise ValueError('专业分步：存在不可复原（unrestorable）标红区，禁止任何像素修复；仅可导出标注图与报告。')
    if has_uncertain:
        if level != 'conservative':
            raise ValueError('专业分步：不确定（uncertain）红区默认禁止加强/推测；仅可在强制放行后做保守修补。')
        if not force_override:
            raise ValueError('专业分步：不确定红区须勾选「专家强制放行」并填写理由，才可保守修补。')
        require_rationale(force_override_reason, label='强制放行理由')
    if contest and level in CONTEST_LAB_ONLY:
        raise ValueError('比赛演示主线禁止生成性补绘（加强/推测/证据重建）；请改用保守观察+标注图+报告。')
    if contest and mode in CONTEST_LAB_ONLY:
        raise ValueError('比赛演示主线禁用加强/推测修复模式；此类结果仅内部实验，不进演示终稿。')
    notes = (expert_notes or '').strip()[:2000]
    return {
        'rationale': rationale,
        'evidence_level': level,
        'expert_notes': notes,
        'mask_source': source,
        'allow_enhanced': bool(allow_enhanced) if level == 'enhanced' else False,
        'risk_acknowledged': bool(risk_acknowledged) if level == 'enhanced' else False,
        'mask_gate_confirmed': True,
        'has_unrestorable': bool(has_unrestorable),
        'has_uncertain': bool(has_uncertain),
        'force_override': bool(force_override) if has_uncertain else False,
        'force_override_reason': (force_override_reason or '').strip()[:2000] if has_uncertain else '',
        'contest_demo': contest,
    }


def validate_job_record_for_execution(job, mask_record):
    """Defense-in-depth check at engine.run for professional_step jobs."""
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
        has_unrestorable=bool(job.get('has_unrestorable')),
        has_uncertain=bool(job.get('has_uncertain')),
        force_override=bool(job.get('force_override')),
        force_override_reason=job.get('force_override_reason') or '',
        contest_demo=job.get('contest_demo'),
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

    Contest hard gate: professional_step requires an explicit review decision
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
    kind = (value or default or 'fillable_lacuna').strip().lower()
    if kind not in REGION_KINDS:
        raise ValueError('区域类型须为 fillable_lacuna / uncertain / unrestorable。')
    return REGION_ALIASES[kind]


def contest_allows_pixel_export(job):
    """Demo deliverable is annotation + report; generative finals are blocked."""
    if not (job.get('contest_demo', CONTEST_DEMO)):
        return True
    mode = job.get('mode') or 'conservative'
    level = job.get('evidence_level') or mode
    if mode in CONTEST_LAB_ONLY or level in CONTEST_LAB_ONLY:
        return False
    if job.get('selected_as') in {'enhanced', 'hypothesis'}:
        return False
    return True


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


def refuse_insufficient_evidence():
    """Standard refuse payload when AI should stop generative fill."""
    return {
        'refused': True,
        'selected_as': 'refused',
        'message': DEMO_COPY['B']['refuse'],
    }
