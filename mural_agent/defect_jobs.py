"""Create one independently reviewable conservative job per confirmed defect.

Defect-list repairs are intentionally not filtered by the heuristic risk
classes: the user asked to attempt every listed defect and simply re-run when a
result is bad. Original risk metadata is still recorded on the job.
"""
from . import workspace

ACTIVE_JOB_STATES = {'queued', 'planning', 'running', 'evaluating'}
METHODS = {
    'opencv': 'OpenCV 局部修补',
    'telea': 'Telea 纹理传播',
    'ns': 'Navier–Stokes 细线修补',
    'lama': 'LaMa AI 纹理补全',
    'color': '局部色调匹配',
    'diffusion': 'AI 生成式补全（实验）',
}


def run_confirmed_defects(store, engine, image_id, proposal_ids, selected_methods=None,
                          generative_attempt=False):
    image = store.get('image', image_id)
    if not image:
        raise ValueError('图像记录不存在。')
    proposal_ids = list(dict.fromkeys(proposal_ids or []))
    if not proposal_ids:
        raise ValueError('没有已确认的缺损项。')

    selected_methods = selected_methods or {}
    availability = engine.tools.availability()
    jobs, skipped = [], []
    for proposal_id in proposal_ids:
        proposal = store.get('proposal', proposal_id)
        if not proposal or proposal.get('image_id') != image_id:
            skipped.append({'proposal_id': proposal_id, 'reason': '缺损记录不存在或不属于当前图像'})
            continue

        previous_id = proposal.get('repair_job_id')
        previous = store.get('job', previous_id) if previous_id else None
        if previous and previous.get('status') in ACTIVE_JOB_STATES:
            skipped.append({'proposal_id': proposal_id, 'reason': '该缺损项已有修复任务正在运行'})
            continue

        metadata = proposal.get('metadata') or {}
        from .risk_regions import normalize_region_kind
        risk_class = normalize_region_kind(
            proposal.get('region_class') or metadata.get('region_class') or 'uncertain')

        if generative_attempt:
            backend = 'diffusion'
        else:
            backend = selected_methods.get(proposal_id) or (metadata.get('recommendation') or {}).get('backend') or 'telea'
        if backend not in METHODS:
            backend = 'telea'
        # Do not skip unavailable LaMa: fall back to a local classical method so
        # every confirmed defect can still be attempted.
        if backend == 'lama' and not availability.get('lama'):
            backend = 'telea'
        if backend == 'diffusion' and not availability.get('diffusion'):
            raise ValueError('AI 尝试性补全需要已安装扩散模型（models/sd-inpainting）。')

        item_type = str(metadata.get('type') or '疑似缺损')[:80]
        method = METHODS.get(backend, 'AI 生成式补全（实验）')
        manual = proposal_id in selected_methods
        source = '按人工选择' if manual else '按 AI 建议'
        rationale = (
            f'人工确认的{item_type}候选；{source}采用{method}。'
            f'原风险分类：{risk_class}（本路径按用户要求不拦截）。'
            + (' 这是 AI 尝试性补全，结果为推测，不是历史原貌。' if generative_attempt else ' 结果仅供修复人员复核。')
        )
        # Defect-list repairs bypass the risk gate by creating an explicit
        # fillable working mask; the original class stays on the job record.
        mask = workspace.add_mask(
            store, image, (store.folder(proposal_id) / 'mask.png').read_bytes(),
            [proposal_id], mask_source=proposal.get('mask_source') or 'ai_candidate',
            user_confirmed=True, region_class='fillable_lacuna', risk_regions=[])
        job = workspace.create_job(
            store, engine, image, mask, backend, 'rules', 1,
            rationale + ' ' + str(metadata.get('evidence') or ''),
            mode='speculative' if generative_attempt else 'conservative',
            budget={'candidates': 1, 'revisions': 0, 'timeout_seconds': 300},
            workflow='professional_step', rationale=rationale,
            evidence_level='speculative' if generative_attempt else 'conservative',
            expert_notes=str(metadata.get('risk_notice') or ''),
            mask_source=proposal.get('mask_source') or 'ai_candidate',
            mask_gate_confirmed=True,
            risk_acknowledged=bool(generative_attempt),
            allow_generative_attempt=bool(generative_attempt))
        job = store.update('job', job['id'], defect_proposal_id=proposal_id,
                           defect_type=item_type, recommended_method=method,
                           defect_risk_class=risk_class,
                           defect_risk_notice=str(metadata.get('risk_notice') or ''),
                           allow_generative_attempt=bool(generative_attempt))
        history = list(proposal.get('repair_job_history') or [])
        if previous_id and previous_id not in history:
            history.append(previous_id)
        store.update('proposal', proposal_id, confirmed=True,
                     repair_job_id=job['id'], repair_job_history=history)
        jobs.append(job)
    return {'jobs': jobs, 'skipped': skipped}
