from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import platform
import time
import zipfile
import importlib.metadata
import numpy as np
from PIL import Image
from .imaging import image_array, mask_array, composite, make_difference, sha256
from .quality import assess
from .models import LamaBackend
from .segmentation import SamSegmenter
from .tools import ToolRegistry
from .planner import rule_plan, auto_plan, api_plan, Plan, Step, Budget, Region
from .storage import Store, timestamp, new_id
from .local_planner import local_plan


class Cancelled(Exception):
    pass


class Engine:
    def __init__(self, settings, store: Store):
        self.settings, self.store = settings, store
        self.lama = LamaBackend(settings)
        self.sam = SamSegmenter(settings)
        self.tools = ToolRegistry(self)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mural-worker')
        self.pending = set()
        self.pending_lock = Lock()

    def submit(self, job_id):
        with self.pending_lock:
            if job_id in self.pending:
                raise ValueError('任务已在执行队列。')
            self.pending.add(job_id)
        def execute():
            try:
                self.run(job_id)
            finally:
                with self.pending_lock:
                    self.pending.discard(job_id)
        return self.executor.submit(execute)

    def resume(self, job_id):
        job = self.store.get('job', job_id)
        if not job or job['status'] not in {'interrupted', 'failed', 'cancelled'}:
            raise ValueError('只有中断、失败或取消的任务可续跑。')
        # The worker thread removes its id from `pending` only after run() returns,
        # so a terminal status can be observed a few milliseconds before the slot
        # is released. Wait briefly instead of failing a legitimate resume.
        for _ in range(100):
            with self.pending_lock:
                if job_id not in self.pending:
                    break
            time.sleep(0.02)
        with self.pending_lock:
            if job_id in self.pending:
                raise ValueError('任务尚未停止。')
        self.store.update('job', job_id, status='queued', cancel_requested=False, error=None)
        (self.store.folder(job_id) / 'cancel').unlink(missing_ok=True)
        self.store.event(job_id, '从持久化检查点续跑，沿用已用预算和已完成候选。')
        return self.submit(job_id)

    def check_cancel(self, job_id):
        job = self.store.get('job', job_id)
        if job.get('cancel_requested') or job['status'] == 'cancelled':
            raise Cancelled()

    def validate_inputs(self, job, image, mask_record):
        image_path = self.store.folder(image['id']) / 'image.png'
        mask_path = self.store.folder(mask_record['id']) / 'mask.png'
        if sha256(image_path) != image['working_sha256'] or sha256(mask_path) != mask_record['sha256']:
            raise ValueError('冻结输入 hash 不一致，拒绝续跑或执行。')
        return image_array(image_path), mask_array(mask_path)

    def make_plan(self, job, original, mask):
        fraction = float((mask > 0).mean())
        if job.get('planner') == 'auto':
            plan = auto_plan(job, fraction)
        else:
            plan = rule_plan(job['backend'], job['candidate_count'], fraction)
        if job['planner'] in {'api', 'local'}:
            overlay = original.copy()
            active = mask > 0
            overlay[active] = (.5*overlay[active] + .5*np.array([220, 64, 48])).astype(np.uint8)
            self.store.event(job['id'], '调用本地视觉模型规划，不发送图片。' if job['planner'] == 'local' else '调用视觉 API 规划：发送缩略图、掩膜预览和备注。')
            try:
                if job['planner'] == 'local':
                    plan = local_plan(self, job, original, overlay)
                else:
                    plan = api_plan(self.settings, Image.fromarray(original), Image.fromarray(overlay),
                                    job['backend'], job['candidate_count'], fraction, job['instructions'])
            except Exception:
                self.store.event(job['id'], ('本地视觉模型' if job['planner'] == 'local' else 'API')+'规划失败或返回格式不合要求，已明确回退到本地规则规划。')
        if job.get('mode', 'conservative') == 'conservative':
            from .planner import diversify_repair_methods
            include_reference = False
            registration = self.store.get('registration', job.get('registration_id', ''))
            if registration and registration.get('approved') and registration.get('image_id') == job['image_id']:
                reference = self.store.get('reference', registration.get('reference_id', ''))
                if reference and reference.get('metadata', {}).get('relation') in {'same_mural', 'tracing'}:
                    approved = mask_array(self.store.folder(registration['id']) / 'approved.png')
                    import cv2
                    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
                    largest_fraction = (int(component_stats[1:, cv2.CC_STAT_AREA].max()) / mask.size
                                        if component_count > 1 else 0.)
                    include_reference = (largest_fraction >= .01 and
                                         not ((mask > 0) & (approved == 0)).any())
            plan = diversify_repair_methods(plan, job['backend'], job['candidate_count'],
                                             self.tools.availability(), include_reference=include_reference)
            if include_reference and any(step.tool == 'reference' for step in plan.steps):
                self.store.event(job['id'], '已核验同壁画/临摹参考覆盖当前缺损；将生成参考转移对照候选。')
            if len(plan.steps) < job['candidate_count']:
                job['candidate_count'] = len(plan.steps)
                job['budget']['candidates'] = len(plan.steps)
                self.store.update('job', job['id'], candidate_count=len(plan.steps), budget=job['budget'])
                self.store.event(job['id'], f'当前可用的独立修复方法只有 {len(plan.steps)} 种，候选数量已相应调整。')
        # Runtime policy is authoritative; the model cannot alter budget/evidence/mode.
        plan.mode = job.get('mode', 'conservative')
        plan.regions = [Region(mask_id=job['mask_id'], protected_objects=job.get('protected_objects', []))]
        plan.evidence = [job['registration_id']] if job.get('registration_id') else []
        plan.budget = Budget(**job.get('budget', {'candidates': job['candidate_count']}))
        for step in plan.steps:
            if step.tool == 'imagine':
                step.prompt = job['instructions'] or step.prompt
                step.full_generation = bool(job.get('full_generation'))
                continue
            if step.tool == 'diffusion':
                step.prompt = job['instructions'] or step.prompt
                step.structure_control = job.get('structure_control', False)
                if job.get('full_generation'):
                    step.full_generation = True
                    step.strength = 0.95
                    step.structure_control = bool(self.tools.availability().get('controlnet'))
                    if not (job.get('instructions') or '').strip():
                        from .professional import FULL_GENERATION_PROMPT
                        step.prompt = FULL_GENERATION_PROMPT
                if job.get('mode') == 'enhanced':
                    from .segmentation import ENHANCED_DIFFUSION_STRENGTH
                    from .professional import ENHANCED_DIFFUSION_PROMPT, ENHANCED_NEGATIVE_PROMPT
                    step.strength = float(ENHANCED_DIFFUSION_STRENGTH)
                    if not (job.get('instructions') or '').strip():
                        step.prompt = ENHANCED_DIFFUSION_PROMPT
                    # negative_prompt is not on Step model; stash on reason for worker via prompt only.
                    # model_worker reads step.prompt; enhanced demo passes negative separately.
                    if 'eliminate visible craquelure' not in (step.prompt or ''):
                        step.prompt = ((step.prompt or '') + ' ' + ENHANCED_DIFFUSION_PROMPT).strip()[:2000]
            elif job.get('mode') == 'enhanced' and step.tool in {'telea', 'ns'}:
                # Bolder traditional fill for crack-killing when SD unavailable.
                step.radius = max(step.radius, 7)
        return plan

    def revise(self, job, step, metrics):
        if job['backend'] in {'telea', 'ns'}:
            return step.model_copy(update={'radius': min(9, step.radius+1),
                                           'reason': '保持缺损类别对应的修补算法，微调局部邻域半径。'})
        if step.tool in {'lama', 'telea', 'ns'}:
            choices = ['lama', 'telea', 'ns'] if job['backend'] == 'lama' else ['telea', 'ns']
            alternate = choices[(choices.index(step.tool)+1) % len(choices)]
            return Step(tool=alternate, radius=min(9, step.radius+2), reason='根据边界接缝或无修改反馈更换传播方式。')
        if step.tool == 'diffusion':
            return step.model_copy(update={'seed': step.seed+1, 'strength': max(.3, step.strength-.15),
                                           'reason': '根据质量反馈降低重绘强度并更换种子。'})
        return step.model_copy(update={'strength': max(.1, step.strength-.2),
                                      'reason': '根据质量反馈降低局部处理强度，保持原始输入。'})

    def run(self, job_id):
        session_start = time.monotonic()
        previous_elapsed = 0
        try:
            self.check_cancel(job_id)
            job = self.store.get('job', job_id)
            previous_elapsed = float(job.get('elapsed_seconds', 0))
            self.store.update('job', job_id, started_at=timestamp())
            image = self.store.get('image', job['image_id'])
            mask_record = self.store.get('mask', job['mask_id'])
            original, mask = self.validate_inputs(job, image, mask_record)
            from .professional import validate_job_record_for_execution
            validate_job_record_for_execution(job, mask_record)
            self.store.event(job_id, '已校验原图与冻结掩膜 hash；候选始终从该输入版本产生。')
            enhanced_meta = None
            base_for_tools = original
            if job.get('mode') == 'enhanced':
                from .segmentation import build_enhanced_work_mask
                from .imaging import opencv_inpaint  # composite already imported at module scope
                work_mask, enhanced_meta = build_enhanced_work_mask(original, mask)
                Image.fromarray(work_mask).save(self.store.folder(job_id) / 'enhanced_work_mask.png')
                (self.store.folder(job_id) / 'enhanced_mask_meta.json').write_text(
                    __import__('json').dumps(enhanced_meta, ensure_ascii=False, indent=2), encoding='utf-8')
                # Conservative base first (LaMa / Telea), then stronger tools see this base.
                try:
                    if self.lama.available():
                        cons_pred = self.lama.predict(original, mask)
                        base_for_tools = composite(original, cons_pred, mask)
                        self.store.event(job_id, '加强修复：已用 LaMa 生成保守底图，再对选区附近缺损加强。')
                    else:
                        cons_pred = opencv_inpaint(original, mask, 'telea', 5)
                        base_for_tools = composite(original, cons_pred, mask)
                        self.store.event(job_id, '加强修复：已用 Telea 生成保守底图，再对选区附近缺损加强。')
                    Image.fromarray(base_for_tools).save(self.store.folder(job_id) / 'enhanced_conservative_base.png')
                except Exception as exc:
                    self.store.event(job_id, f'加强修复：保守底图失败，回退原图加强：{str(exc)[:160]}')
                    base_for_tools = original
                mask = work_mask
                self.store.update('job', job_id, enhanced_mask_meta=enhanced_meta)
                frac = enhanced_meta.get('enhanced_fraction', 0)
                lac = enhanced_meta.get('lacuna_fraction', 0)
                crk = enhanced_meta.get('crack_fraction', 0)
                self.store.event(
                    job_id,
                    f'加强允许掩膜占比 {frac:.2%}（选中缺损 {lac:.2%} + 附近裂纹 {crk:.2%}）。')
            if job.get('plan'):
                plan = Plan.model_validate(job['plan'])
            else:
                self.store.update('job', job_id, status='planning', progress=8)
                plan = self.make_plan(job, original, mask)
                self.store.update('job', job_id, plan=plan.model_dump())
            budget = plan.budget
            # Reconcile candidate writes committed immediately before a crash.
            candidates = {c['id']: c for c in job['candidates']}
            for candidate in self.store.list('candidate'):
                if candidate['job_id'] == job_id:
                    candidates[candidate['id']] = candidate
            candidates = sorted(candidates.values(), key=lambda c: c['created_at'])
            attempts = job.get('attempts', [])
            for attempt in attempts:
                if attempt['status'] == 'executing':
                    attempt['status'] = 'interrupted'
            revisions = job.get('revisions', [])
            next_step = job.get('next_step')
            self.store.update('job', job_id, candidates=candidates, attempts=attempts, status='running', progress=18)
            exit_reason = 'candidate_limit'
            while len(candidates) < min(job['candidate_count'], budget.candidates):
                self.check_cancel(job_id)
                remaining = budget.timeout_seconds - previous_elapsed - (time.monotonic()-session_start)
                if remaining <= 0:
                    exit_reason = 'time_budget'
                    break
                if len(attempts) >= budget.candidates + budget.revisions:
                    exit_reason = 'attempt_budget'
                    break
                step = Step.model_validate(next_step) if next_step else plan.steps[min(len(candidates), len(plan.steps)-1)]
                next_step = None
                identifier = new_id()
                attempt = {'id': identifier, 'step': step.model_dump(), 'status': 'executing', 'started_at': timestamp(),
                           'input_image_id': image['id'], 'mask_id': mask_record['id']}
                attempts.append(attempt)
                self.store.update('job', job_id, attempts=attempts, next_step=None, status='running')
                self.store.event(job_id, f'尝试 {len(attempts)}：{step.tool}；{step.reason}')
                started = time.monotonic()
                try:
                    prediction, versions = self.tools.execute(step, base_for_tools, mask, job, self.store.folder(identifier), remaining)
                    self.check_cancel(job_id)
                    if prediction.shape != original.shape or not np.isfinite(prediction).all():
                        raise ValueError('工具输出尺寸异常或存在非有限像素。')
                    Image.fromarray(prediction.astype(np.uint8)).save(self.store.folder(identifier)/'raw.png')
                    output = composite(original, prediction, mask)
                    self.store.update('job', job_id, status='evaluating')
                    metrics = assess(original, output, mask, job.get('quality_thresholds'))
                    if not metrics['outside_unchanged']:
                        raise RuntimeError('候选未通过掩膜外一致性检查。')
                    folder = self.store.folder(identifier)
                    Image.fromarray(output).save(folder/'result.png')
                    Image.fromarray(make_difference(original, output)).save(folder/'difference.png')
                    candidate = {'id': identifier, 'job_id': job_id, 'image_id': image['id'], 'mask_id': mask_record['id'],
                        'tool': step.tool, 'radius': step.radius, 'parameters': step.model_dump(),
                        'parent_version': image['id'],
                           'speculative': job.get('mode') == 'speculative',
                           'enhanced': job.get('mode') == 'enhanced',
                        'metrics': metrics, 'elapsed_seconds': round(time.monotonic()-started, 2), 'versions': versions,
                        'created_at': timestamp(), 'raw_sha256': sha256(folder/'raw.png'), 'result_sha256': sha256(folder/'result.png'),
                        'url': f'/api/assets/{identifier}/result.png', 'difference_url': f'/api/assets/{identifier}/difference.png'}
                    self.store.put('candidate', candidate)
                    candidates.append(candidate)
                    attempt.update(status='completed', candidate_id=identifier, metrics=metrics)
                    if not metrics['passed'] and len(revisions) < budget.revisions and len(candidates) < budget.candidates:
                        revised = self.revise(job, step, metrics)
                        next_step = revised.model_dump()
                        revisions.append({'after_attempt': identifier, 'issues': metrics['issues'], 'next_step': next_step, 'time': timestamp()})
                        self.store.event(job_id, '质量反馈触发修订：'+', '.join(metrics['issues'])+'；'+revised.reason)
                except Cancelled:
                    attempt['status'] = 'cancelled'
                    raise
                except Exception as exc:
                    self.check_cancel(job_id)
                    message = str(exc)[:400] if isinstance(exc, (ValueError, RuntimeError)) else f'{type(exc).__name__}: {exc}'[:400]
                    attempt.update(status='failed', error=message)
                    self.store.event(job_id, message)
                    if len(revisions) >= budget.revisions:
                        exit_reason = 'retry_budget'
                        break
                    revised = self.revise(job, step, {'issues': ['tool_failure']})
                    next_step = revised.model_dump()
                    revisions.append({'after_attempt': identifier, 'issues': ['tool_failure'], 'next_step': next_step, 'time': timestamp()})
                finally:
                    attempt['elapsed_seconds'] = round(time.monotonic()-started, 3)
                    self.store.update('job', job_id, attempts=attempts, candidates=candidates, revisions=revisions,
                        next_step=next_step, elapsed_seconds=previous_elapsed+time.monotonic()-session_start,
                        progress=18+round(len(candidates)/budget.candidates*75))
            self.check_cancel(job_id)
            if not candidates:
                self.store.update('job', job_id, status='failed', error='预算内未产生候选，请检查模型、参考资料或输入。', exit_reason=exit_reason)
                return
            passed = [c for c in candidates if c['metrics'].get('passed', False)]
            recommended = min(passed or candidates, key=lambda c: c['metrics'].get('score', c['metrics']['boundary_color_gap']))['id']
            self.store.update('job', job_id, status='review', progress=100, recommended_id=recommended,
                exit_reason=exit_reason, quality_outcome='checks_passed' if passed else 'needs_human_review')
            self.store.event(job_id, '已完成有限预算闭环；保留全部候选与失败轨迹，等待人工复核。')
        except Cancelled:
            self.store.update('job', job_id, status='cancelled')
            self.store.event(job_id, '任务已取消，可从已保存的检查点续跑。')
        except Exception as exc:
            message = str(exc)[:400] if isinstance(exc, (ValueError, RuntimeError)) else '执行失败，请检查本地依赖与运行日志。'
            self.store.update('job', job_id, status='failed', error=message)
            self.store.event(job_id, message)
        finally:
            self.store.update('job', job_id, elapsed_seconds=previous_elapsed+time.monotonic()-session_start)

    def export(self, job_id):
        job = self.store.get('job', job_id)
        if not job.get('selected_id'):
            raise ValueError('请先选择一个候选结果。')
        from .professional import contest_allows_pixel_export
        if not contest_allows_pixel_export(job):
            raise ValueError('比赛演示禁止导出生成性修复成图；仅允许标注层与复核记录。')
        if job['selected_id'] not in [c['id'] for c in job['candidates']]:
            raise ValueError('选中结果不属于当前任务。')
        candidate = self.store.get('candidate', job['selected_id'])
        from .professional import validate_selected_export
        selection = validate_selected_export(job, candidate)
        image = self.store.get('image', job['image_id'])
        mask = self.store.get('mask', job['mask_id'])
        self.validate_inputs(job, image, mask)
        dependencies = {}
        for name in ('torch', 'torchvision', 'PySide6-Essentials', 'opencv-python-headless', 'transformers', 'diffusers', 'numpy', 'pillow'):
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest = {'format_version': 2, 'app_version': '1.0.0', 'exported_at': timestamp(), 'image': image,
            'mask': mask, 'job': job, 'selected_candidate': candidate, 'dependencies': dependencies,
            'python': platform.python_version(), 'working_color_space': '8-bit RGB EXIF-corrected working copy',
            'random_seed': candidate.get('parameters', {}).get('seed'),
            'selected_as': selection['selected_as'],
            'review': job.get('review'),
            'notice': ('假设性建议：该档案不代表历史原貌，不能作为修复定稿。'
                       if selection['selected_as'] == 'hypothesis' else
                       '数字修复候选；无证据区域不能视为历史原貌。工程评分不是历史真实性概率。')}
        folder = self.store.folder(job_id)
        archive, temporary = folder/'restoration.zip', folder/'restoration.tmp'
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as output:
            def add(path, name):
                if path.is_file():
                    output.write(path, name)
            add(self.store.folder(image['id'])/'source.bin', 'original/source.bin')
            add(self.store.folder(image['id'])/'image.png', 'original/working.png')
            add(self.store.folder(mask['id'])/'mask.png', 'mask.png')
            add(self.store.folder(candidate['id'])/'result.png', 'result.png')
            add(self.store.folder(candidate['id'])/'difference.png', 'difference.png')
            for item in job['candidates']:
                for name in ('raw.png', 'result.png', 'difference.png'):
                    add(self.store.folder(item['id'])/name, f'candidates/{item["id"]}/{name}')
            if job.get('registration_id'):
                registration = self.store.get('registration', job['registration_id'])
                reference = self.store.get('reference', registration['reference_id'])
                manifest['registration'], manifest['reference'] = registration, reference
                add(self.store.folder(reference['image_id'])/'source.bin', 'evidence/source.bin')
                for name in ('aligned.png', 'coverage.png', 'approved.png', 'overlay.png'):
                    add(self.store.folder(registration['id'])/name, 'evidence/'+name)
            for proposal_id in job.get('proposal_ids', []):
                proposal = self.store.get('proposal', proposal_id)
                if proposal:
                    add(self.store.folder(proposal_id)/'proposal.png', f'proposals/{proposal_id}.png')
                    output.writestr(f'proposals/{proposal_id}.json', json.dumps(proposal, ensure_ascii=False, indent=2))
            output.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        temporary.replace(archive)
        return archive
