(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const state = { bridge: null, image: null, job: null, candidate: null, keepEmpty: false, maskStrokes: [], currentStroke: null,
    baseMask: null, maskDirty: false, tool: 'brush', brushSize: 28, zoom: 1, promptPoints: [], promptLabels: [], promptBox: null,
    proposal: null, references: [], selectedReference: null, registration: null, lastSnapshot: null, toastTimer: null };
  const canvas = $('#mask-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const image = $('#source-image');
  const imageWrap = $('#canvas-wrap');

  const invoke = (method, ...args) => new Promise((resolve, reject) => {
    if (!state.bridge || !state.bridge[method]) return reject(new Error('桌面服务尚未连接。'));
    state.bridge[method](...args, result => {
      try {
        const parsed = typeof result === 'string' ? JSON.parse(result) : result;
        if (parsed?.error) reject(new Error(parsed.error)); else resolve(parsed);
      } catch (error) { reject(error); }
    });
  });
  const toast = (text) => {
    const node = $('#toast'); node.textContent = text; node.classList.remove('hidden');
    clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => node.classList.add('hidden'), 3000);
  };
  const setBusy = (value) => {
    document.body.classList.toggle('busy', value);
    $('[data-action="run-repair"]').disabled = value || !state.image || !maskHasPixels();
  };
  const maskHasPixels = () => {
    if (!canvas.width || !canvas.height) return false;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
    return false;
  };
  const notifyError = (error) => toast(error?.message || String(error));
  const fileUrl = (value) => value || '';

  function showPage(page) {
    $$('.page').forEach(item => {
      const active = item.id === `page-${page}`;
      item.classList.toggle('active', active);
      item.classList.toggle('hidden', !active);
    });
    $$('.nav-item[data-page]').forEach(item => item.classList.toggle('active', item.dataset.page === page));
    const title = { workbench: '修复工作台', tools: '工具与评测', history: '历史记录' }[page];
    $('#page-title').textContent = title;
    if (page === 'history') renderHistory();
  }

  function showView(view) {
    $$('.view-tab').forEach(item => item.classList.toggle('active', item.dataset.view === view));
    $('#edit-view').classList.toggle('active', view === 'edit');
    $('#results-view').classList.toggle('active', view === 'results');
  }

  function setTool(tool) {
    state.tool = tool;
    $$('.tool-button').forEach(item => item.classList.toggle('active', item.dataset.tool === tool));
    canvas.classList.toggle('sam-point', tool === 'sam-point');
    canvas.classList.toggle('sam-box', tool === 'sam-box');
    canvas.classList.toggle('eraser', tool === 'eraser');
    $('#mask-status').textContent = tool === 'sam-point' ? '左键添加前景点，右键添加背景点' :
      tool === 'sam-box' ? '在图像上拖动框选目标' : '红色表示允许编辑';
  }

  function updateCanvasSize() {
    if (!state.image || !image.naturalWidth) return;
    imageWrap.style.transform = `scale(${state.zoom})`;
    requestAnimationFrame(() => {
      canvas.style.width = `${image.clientWidth}px`;
      canvas.style.height = `${image.clientHeight}px`;
      $('#mask-status').textContent = `掩膜 ${Math.round(maskFraction() * 10000) / 100}% · ${state.image.width} × ${state.image.height}`;
    });
  }

  function maskFraction() {
    if (!canvas.width || !canvas.height) return 0;
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) count++;
    return count / (canvas.width * canvas.height);
  }

  function paintStroke(stroke) {
    if (!stroke.points.length) return;
    ctx.save();
    ctx.globalCompositeOperation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#d84d45'; ctx.fillStyle = '#d84d45'; ctx.globalAlpha = 1;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = stroke.size;
    if (stroke.points.length === 1) {
      ctx.beginPath(); ctx.arc(stroke.points[0][0], stroke.points[0][1], stroke.size / 2, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.beginPath(); ctx.moveTo(...stroke.points[0]);
      stroke.points.slice(1).forEach(point => ctx.lineTo(...point)); ctx.stroke();
    }
    ctx.restore();
  }

  function redrawMask() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.baseMask) ctx.drawImage(state.baseMask, 0, 0, canvas.width, canvas.height);
    state.maskStrokes.forEach(paintStroke);
    if (state.currentStroke) paintStroke(state.currentStroke);
    state.maskDirty = true;
    $('#mask-fraction').textContent = `${Math.round(maskFraction() * 10000) / 100}%`;
    setBusy(false);
  }

  function pointAt(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height];
  }

  function loadImage(record) {
    if (!record) {
      state.keepEmpty = true;
      state.image = state.job = state.candidate = state.proposal = null;
      state.maskStrokes = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      image.removeAttribute('src'); imageWrap.classList.add('hidden'); $('#empty-state').classList.remove('hidden');
      $('#image-name').textContent = '导入壁画图像'; $('#image-meta').textContent = '选择本地图片开始';
      $('#candidate-count').textContent = '0'; $('#mask-fraction').textContent = '';
      $('[data-action="run-repair"]').disabled = true;
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      $('#event-log').innerHTML = '<div class="empty-log">完成掩膜后生成候选。</div>';
      return;
    }
    state.keepEmpty = false;
    state.image = record; state.job = null; state.candidate = null; state.proposal = null;
    state.maskStrokes = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
    state.zoom = 1; state.promptPoints = []; state.promptLabels = []; state.promptBox = null;
    $('#image-name').textContent = record.name;
    $('#image-meta').textContent = `${record.width} × ${record.height} · 原始文件已归档`;
    $('#empty-state').classList.add('hidden'); imageWrap.classList.remove('hidden');
    image.onload = () => {
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      updateCanvasSize();
      setBusy(false);
    };
    image.onerror = () => toast('图像无法载入。');
    image.src = fileUrl(record.file_url);
    $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
    $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
    $('#candidate-count').textContent = '0';
    $('#event-log').innerHTML = '<div class="empty-log">完成掩膜后生成候选。</div>';
    $('#mask-fraction').textContent = '';
    $('[data-action="save-result"]').disabled = true;
    $('[data-action="select-candidate"]').disabled = true;
    $('[data-action="export-archive"]').disabled = true;
    state.proposalIds = [];
    showView('edit'); showPage('workbench');
  }

  function installMask(url) {
    const maskImage = new Image();
    maskImage.onload = () => {
      state.baseMask = maskImage; state.maskStrokes = []; state.maskDirty = false; redrawMask(); state.maskDirty = false;
    };
    maskImage.onerror = () => toast('掩膜读取失败。');
    maskImage.src = url;
  }

  function exportMaskBase64() {
    const source = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const output = document.createElement('canvas'); output.width = canvas.width; output.height = canvas.height;
    const target = output.getContext('2d'); const data = target.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < source.data.length; i += 4) {
      const value = source.data[i + 3] > 0 ? 255 : 0;
      data.data[i] = data.data[i + 1] = data.data[i + 2] = value; data.data[i + 3] = 255;
    }
    target.putImageData(data, 0, 0);
    return output.toDataURL('image/png').split(',')[1];
  }

  function showMaskFromBase64(data) {
    const maskImage = new Image();
    maskImage.onload = () => {
      state.baseMask = maskImage; state.maskStrokes = []; redrawMask(); state.maskDirty = false;
    };
    maskImage.src = `data:image/png;base64,${data}`;
  }

  async function openImage() {
    try { const record = await invoke('openImage'); if (record?.id) loadImage(record); }
    catch (error) { notifyError(error); }
  }

  async function createJob() {
    if (!state.image || !maskHasPixels()) return toast('请先圈选需要修复的区域。');
    setBusy(true);
    try {
      const mask = await invoke('saveMask', state.image.id, exportMaskBase64(), JSON.stringify(state.proposalIds || []));
      const request = { image_id: state.image.id, mask_id: mask.id,
        backend: $('#repair-backend').value, planner: $('#planner').value,
        candidate_count: Number($('#candidate-number').value), instructions: $('#instructions').value,
        mode: $('#repair-mode').value, registration_id: state.registration?.id || null,
        structure_control: $('#controlnet').checked,
        budget: { candidates: Number($('#candidate-number').value), revisions: Number($('#revisions').value), timeout_seconds: Number($('#timeout').value), max_api_calls: 1 },
        quality_thresholds: { boundary_pair_delta: 65 } };
      state.job = await invoke('createJob', JSON.stringify(request));
      state.candidate = null; state.maskDirty = false; showView('results'); toast('修复任务已加入工作队列。');
      await refresh();
    } catch (error) { setBusy(false); notifyError(error); }
  }

  async function refresh() {
    if (!state.bridge) return;
    try {
      const snapshot = await invoke('snapshot'); state.lastSnapshot = snapshot;
      renderModels(snapshot.models || {});
      if (!state.image && !state.keepEmpty && snapshot.images?.length) {
        const latestJob = snapshot.jobs?.[0];
        const record = snapshot.images.find(item => item.id === latestJob?.image_id) || snapshot.images[0];
        loadImage(record);
      }
      if (state.job) state.job = snapshot.jobs.find(item => item.id === state.job.id) || state.job;
      else if (!state.keepEmpty && snapshot.jobs?.length) state.job = snapshot.jobs.find(item => item.image_id === state.image?.id) || null;
      renderJob(state.job);
      renderHistory(snapshot.jobs || []);
      if (snapshot.aux) renderAux(snapshot.aux);
      if (snapshot.registration) state.registration = snapshot.registration;
    } catch (error) { console.warn(error); }
  }

  function renderModels(models) {
    const labels = [['LaMa', models.lama], ['SAM 2', models.sam2], ['Qwen-VL', models.vision], ['SD', models.diffusion]];
    $('#model-list').innerHTML = labels.map(([name, available]) => `<span class="model-tag ${available ? '' : 'off'}">${name}</span>`).join('');
    const ready = models.lama ? 'LaMa 可用' : '传统修补可用';
    $('#engine-status').innerHTML = `<i></i>${ready}`;
    const backend = $('#repair-backend'); const prior = backend.value;
    const items = [['opencv', '传统修补 · Telea / NS', true], ['lama', 'LaMa · 纹理补全', models.lama],
      ['reference', '参考证据重建', models.reference], ['diffusion', '扩散推测补全', models.diffusion],
      ['denoise', '局部降噪', true], ['color', '周边色调匹配', true], ['deblur', '局部锐化', true], ['research', '壁画研究模型', models.research]];
    backend.innerHTML = items.filter(([, , available]) => available).map(([value, title]) => `<option value="${value}">${title}</option>`).join('');
    backend.value = items.some(([value, , available]) => value === prior && available) ? prior : (models.lama ? 'lama' : 'opencv');
    $('#planner option[value="local"]').disabled = !models.vision;
    $('#planner option[value="api"]').disabled = !models.api_configured;
  }

  function renderJob(job) {
    if (!job) return;
    const running = ['queued', 'planning', 'running', 'evaluating'].includes(job.status);
    $('#job-status').textContent = ({ queued: '排队中', planning: '规划中', running: '修复中', evaluating: '质量检查', review: '等待复核', interrupted: '已中断', failed: '执行失败', cancelled: '已取消' })[job.status] || job.status;
    $('#job-progress').style.width = `${Math.max(0, Math.min(100, job.progress || 0))}%`;
    $('[data-action="run-repair"]').disabled = running || !state.image || !maskHasPixels();
    $('[data-action="run-repair"]').classList.toggle('hidden', running);
    $('[data-action="cancel-job"]').classList.toggle('hidden', !running);
    $('#resume-button').classList.toggle('hidden', running || !['interrupted', 'failed', 'cancelled'].includes(job.status));
    $('#candidate-count').textContent = job.candidates?.length || 0;
    const lines = [];
    if (job.plan) lines.push(`<div class="event-line"><span class="event-time">修复计划</span>${escapeHtml(job.plan.summary || '')}</div>`);
    (job.events || []).slice(-40).forEach(event => lines.push(`<div class="event-line"><span class="event-time">${escapeHtml((event.time || '').slice(11, 19))}</span>${escapeHtml(event.message)}</div>`));
    if (job.error) lines.push(`<div class="event-line"><span class="event-time">错误</span>${escapeHtml(job.error)}</div>`);
    $('#event-log').innerHTML = lines.join('') || '<div class="empty-log">生成任务后显示规划、工具调用与检查结果。</div>';
    if (running) $('#event-log').scrollTop = $('#event-log').scrollHeight;
    renderCandidates(job);
  }

  function renderCandidates(job) {
    const candidates = job.candidates || [];
    if (!candidates.length) return;
    const selected = state.candidate?.id || job.selected_id || candidates[0].id;
    if (!state.candidate) state.candidate = candidates.find(item => item.id === selected) || candidates[0];
    $('#candidate-strip').innerHTML = candidates.map((candidate, index) => `<button class="candidate-card ${candidate.id === state.candidate?.id ? 'active' : ''}" data-candidate="${candidate.id}"><img src="${fileUrl(candidate.file_url)}"><span>${index + 1}. ${escapeHtml(candidate.tool)}</span></button>`).join('');
    $$('.candidate-card').forEach(item => item.addEventListener('click', () => {
      state.candidate = candidates.find(candidate => candidate.id === item.dataset.candidate); renderCandidates(job); renderComparison();
    }));
    renderComparison();
  }

  function renderComparison() {
    const candidate = state.candidate; if (!candidate || !state.image) return;
    const stage = $('#comparison-stage'); const mode = $('#comparison-mode').value;
    const sourceUrl = fileUrl(state.image.file_url);
    const url = mode === 'difference' ? fileUrl(candidate.difference_url) : fileUrl(candidate.file_url);
    stage.innerHTML = `<img class="comparison-image" src="${url}">`;
    if (mode === 'compare') {
      const layer = document.createElement('div'); layer.className = 'comparison-original';
      const original = document.createElement('img'); original.src = sourceUrl; layer.appendChild(original); stage.appendChild(layer);
      const apply = () => {
        const ratio = Number($('#compare-range').value);
        layer.style.width = `${ratio}%`; original.style.width = `${10000 / Math.max(1, ratio)}%`;
      };
      $('#compare-range').oninput = apply; apply();
    } else $('#compare-range').disabled = true;
    $('#compare-range').disabled = mode !== 'compare';
    const metrics = candidate.metrics || {};
    $('#candidate-device').textContent = `${candidate.versions?.device || ''} · ${candidate.elapsed_seconds || 0}s`;
    $('#candidate-metrics').textContent = `掩膜外最大差异 ${metrics.outside_max_difference ?? '-'} · 边界色差 ${metrics.boundary_color_gap ?? '-'} · ${candidate.speculative ? '推测性内容需单独复核' : '工程指标不代表历史真实性'}`;
    $('[data-action="select-candidate"]').disabled = ['queued', 'planning', 'running', 'evaluating'].includes(state.job?.status);
    $('[data-action="select-candidate"]').textContent = jobSelected(candidate.id) ? '已选定候选' : '选定候选';
    $('[data-action="save-result"]').disabled = false;
    $('[data-action="export-archive"]').disabled = !state.job?.selected_id;
  }

  const jobSelected = id => state.job?.selected_id === id;

  function renderHistory(jobs = state.lastSnapshot?.jobs || []) {
    const query = ($('#history-filter').value || '').trim().toLocaleLowerCase();
    const images = state.lastSnapshot?.images || [];
    const items = jobs.filter(job => {
      const img = images.find(record => record.id === job.image_id);
      return !query || (img?.name || '').toLocaleLowerCase().includes(query);
    });
    $('#history-list').innerHTML = items.map(job => {
      const img = images.find(record => record.id === job.image_id);
      const status = ({ review: '等待复核', failed: '失败', cancelled: '已取消', interrupted: '已中断' })[job.status] || job.status;
      return `<button class="history-item" data-job="${job.id}"><span><span class="history-name">${escapeHtml(img?.name || '壁画任务')}</span><span class="history-detail">${(job.created_at || '').slice(0, 16).replace('T', ' ')} · ${(job.candidates || []).length} 个候选</span></span><span class="history-state">${escapeHtml(status)}</span></button>`;
    }).join('') || '<div class="empty-inline">暂无修复记录</div>';
    $$('.history-item').forEach(item => item.addEventListener('click', async () => {
      try {
        const job = items.find(record => record.id === item.dataset.job);
        state.job = job; state.candidate = null;
        const record = images.find(imageRecord => imageRecord.id === job.image_id);
        loadImage(record); state.job = job;
        installMask(job.mask_file_url);
        $('#repair-mode').value = job.mode || 'conservative'; $('#repair-backend').value = job.backend;
        $('#planner').value = job.planner; $('#instructions').value = job.instructions || '';
        showPage('workbench'); showView(job.candidates?.length ? 'results' : 'edit'); await refresh();
      } catch (error) { notifyError(error); }
    }));
  }

  function renderAux(aux) {
    if (aux.status === 'running') $('#sam-status').textContent = aux.label || '处理中';
    else if (aux.status === 'error') { $('#sam-status').textContent = '出错'; $('#proposal-note').textContent = aux.error || '工具执行失败。'; }
    else if (aux.status === 'done' && aux.result?.proposal) {
      state.proposal = aux.result.proposal;
      $('#proposal-image').src = fileUrl(aux.result.proposal.file_url); $('#proposal-image').classList.remove('hidden'); $('#proposal-empty').classList.add('hidden');
      $('#proposal-note').textContent = `${aux.result.proposal.metadata?.notice || '辅助选区待人工确认。'} 候选面积 ${((aux.result.proposal.fraction || 0) * 100).toFixed(2)}%。`;
      $('[data-action="use-proposal"]').disabled = false; $('[data-action="merge-proposal"]').disabled = false;
      $('#sam-status').textContent = '已生成待确认候选'; showPage('tools');
    } else if (aux.status === 'done' && aux.result?.references) renderReferences(aux.result.references);
    else if (aux.status === 'done' && aux.result?.registration) {
      state.registration = aux.result.registration; $('#registration-note').textContent = `内点 ${state.registration.inliers} · 中位误差 ${state.registration.median_error_px.toFixed(2)} px · 请人工核验覆盖区域。`;
      $('#registration-preview').innerHTML = `<img src="${fileUrl(state.registration.overlay_url)}">`;
      $('[data-action="approve-reference"]').disabled = false;
    } else if (aux.status === 'done' && aux.result?.notice) toast(aux.result.notice);
  }

  function renderReferences(records) {
    state.references = records || [];
    $('#reference-list').innerHTML = state.references.map(item => `<button class="reference-item ${item.id === state.selectedReference ? 'selected' : ''}" data-reference="${item.id}">${escapeHtml(item.name)}<small>${escapeHtml(item.metadata?.group || '')} · ${escapeHtml(item.metadata?.relation || '')} · 相似度 ${item.retrieval_score}</small></button>`).join('') || '<div class="empty-inline">没有匹配参考图</div>';
    $$('.reference-item').forEach(item => item.addEventListener('click', () => {
      state.selectedReference = item.dataset.reference; renderReferences(state.references);
      $('[data-action="register-reference"]').disabled = !state.image;
    }));
  }

  function escapeHtml(text) { return String(text ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }

  async function handleAction(action) {
    try {
      switch (action) {
        case 'new-task': showPage('workbench'); loadImage(null); state.image = null; $('#empty-state').classList.remove('hidden'); imageWrap.classList.add('hidden'); break;
        case 'open-image': await openImage(); break;
        case 'load-demo': { const record = await invoke('loadDemo'); loadImage(record); break; }
        case 'undo': if (state.maskStrokes.length) { state.maskStrokes.pop(); redrawMask(); } break;
        case 'clear-mask': state.maskStrokes = []; state.baseMask = null; redrawMask(); break;
        case 'import-mask': { const result = await invoke('importMask'); if (result?.mask_data) showMaskFromBase64(result.mask_data); break; }
        case 'zoom-in': state.zoom = Math.min(3, state.zoom * 1.15); updateCanvasSize(); break;
        case 'zoom-out': state.zoom = Math.max(.5, state.zoom / 1.15); updateCanvasSize(); break;
        case 'zoom-fit': state.zoom = 1; updateCanvasSize(); break;
        case 'run-repair': await createJob(); break;
        case 'cancel-job': if (state.job) await invoke('cancelJob', state.job.id); await refresh(); break;
        case 'resume-job': if (state.job) await invoke('resumeJob', state.job.id); await refresh(); break;
        case 'select-candidate': if (state.job && state.candidate) { state.job = await invoke('selectCandidate', state.job.id, state.candidate.id, $('#review-note').value); toast('候选选择已保存。'); renderComparison(); await refresh(); } break;
        case 'save-result': if (state.candidate) await invoke('saveResult', state.candidate.id); break;
        case 'export-archive': if (state.job) await invoke('exportArchive', state.job.id); break;
        case 'open-settings': openSettings(); break;
        case 'close-settings': $('#settings-modal').classList.add('hidden'); break;
        case 'save-settings': await saveSettings(); break;
        case 'propose-damage': if (state.image) await invoke('proposeDamage', state.image.id); showPage('tools'); break;
        case 'sam-point': setTool('sam-point'); state.promptPoints = []; state.promptLabels = []; showPage('workbench'); showView('edit'); toast('左键添加前景点，右键添加背景点。'); break;
        case 'sam-box': setTool('sam-box'); state.promptBox = null; showPage('workbench'); showView('edit'); toast('在图像上拖动框选目标。'); break;
        case 'run-sam': if (state.image) await invoke('runSam', state.image.id, JSON.stringify(state.promptPoints), JSON.stringify(state.promptLabels), JSON.stringify(state.promptBox)); break;
        case 'clear-prompts': state.promptPoints = []; state.promptLabels = []; state.promptBox = null; setTool('brush'); toast('提示已清除。'); break;
        case 'use-proposal': if (state.proposal) { await invoke('adoptProposal', state.proposal.id); installMask(state.proposal.mask_file_url); state.proposalIds = [state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'merge-proposal': if (state.proposal) { const current = exportMaskBase64(); state.proposal = await invoke('mergeProposal', state.proposal.id, current); installMask(state.proposal.mask_file_url); state.proposalIds = [...(state.proposalIds || []), state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'add-reference': await addReference(); break;
        case 'search-references': await invoke('searchReferences', $('#reference-query').value, $('#reference-group').value); break;
        case 'register-reference': if (state.selectedReference && state.image) await invoke('registerReference', state.image.id, state.selectedReference, exportMaskBase64()); break;
        case 'approve-reference': if (state.registration && state.image) { const note = window.prompt('记录对应关系及核验依据'); if (note) await invoke('approveReference', state.registration.id, exportMaskBase64(), note); } break;
        case 'import-dataset': await invoke('importDataset'); break;
        case 'export-splits': await invoke('exportSplits'); break;
        case 'evaluate-pair': if (state.candidate) await invoke('evaluatePair', state.candidate.id); else toast('请先选择候选。'); break;
        case 'batch-jobs': await invoke('batchJobs', JSON.stringify(readSettings())); break;
        case 'show-models': await invoke('showModels'); break;
        case 'backup-workspace': await invoke('backupWorkspace'); break;
      }
    } catch (error) { notifyError(error); }
  }

  function openSettings() {
    $('#api-kind').value = state.lastSnapshot?.api?.kind || 'responses';
    $('#api-base').value = state.lastSnapshot?.api?.base_url || 'https://api.openai.com/v1';
    $('#api-model').value = state.lastSnapshot?.api?.model || '';
    $('#api-key').value = '';
    $('#api-json').checked = state.lastSnapshot?.api?.json_mode !== false;
    $('#api-error').textContent = '';
    $('#settings-modal').classList.remove('hidden');
  }

  async function saveSettings() {
    const form = { kind: $('#api-kind').value, base_url: $('#api-base').value.trim(), model: $('#api-model').value.trim(), key: $('#api-key').value, json_mode: $('#api-json').checked };
    try { await invoke('saveApiSettings', JSON.stringify(form)); $('#settings-modal').classList.add('hidden'); toast('API 配置已保存。'); await refresh(); }
    catch (error) { $('#api-error').textContent = error.message; }
  }

  async function addReference() {
    const source = window.prompt('来源 URL 或馆藏编号'); if (!source) return;
    const license = window.prompt('许可或使用依据'); if (!license) return;
    const group = window.prompt('壁画 / 洞窟分组'); if (!group) return;
    const relation = window.prompt('对应关系：same_mural / tracing / related', 'same_mural');
    await invoke('addReference', JSON.stringify({ source, license, group, relation: relation || 'same_mural' }));
  }

  function readSettings() {
    return { backend: $('#repair-backend').value, planner: $('#planner').value, count: Number($('#candidate-number').value),
      instructions: $('#instructions').value, rounds: Number($('#revisions').value), timeout: Number($('#timeout').value) };
  }

  document.addEventListener('click', event => {
    const page = event.target.closest('[data-page]'); if (page) { showPage(page.dataset.page); return; }
    const view = event.target.closest('[data-view]'); if (view) { showView(view.dataset.view); return; }
    const action = event.target.closest('[data-action]'); if (action) { handleAction(action.dataset.action); return; }
    const tool = event.target.closest('[data-tool]'); if (tool) setTool(tool.dataset.tool);
  });
  $('#brush-size').addEventListener('input', event => { state.brushSize = Number(event.target.value); $('#brush-value').value = String(state.brushSize); });
  $('#repair-mode').addEventListener('change', event => {
    const backend = $('#repair-backend');
    if (event.target.value === 'evidence') backend.value = 'reference';
    else if (event.target.value === 'speculative') backend.value = 'diffusion';
    else backend.value = state.lastSnapshot?.models?.lama ? 'lama' : 'opencv';
    $('#controlnet').disabled = event.target.value !== 'speculative' || !state.lastSnapshot?.models?.controlnet;
  });
  $('#planner').addEventListener('change', event => {
    $('#planner-note').textContent = event.target.value === 'api' ? '会向已配置服务发送图像缩略图、掩膜预览和备注。' : event.target.value === 'local' ? '本地 Qwen3-VL 分析图像，不发送数据。' : '本地规则不会发送图像。';
  });
  $('#comparison-mode').addEventListener('change', renderComparison);
  $('#history-filter').addEventListener('input', () => renderHistory());
  $('#source-image').addEventListener('load', updateCanvasSize);
  new ResizeObserver(updateCanvasSize).observe($('#image-stage'));

  canvas.addEventListener('pointerdown', event => {
    if (!state.image || !image.naturalWidth) return;
    const point = pointAt(event); canvas.setPointerCapture(event.pointerId);
    if (state.tool === 'sam-point') {
      state.promptPoints.push(point); state.promptLabels.push(event.button === 2 ? 0 : 1); toast('已记录分割提示点。'); return;
    }
    if (state.tool === 'sam-box') { state.promptBox = [point[0], point[1], point[0], point[1]]; return; }
    state.currentStroke = { tool: state.tool, size: state.brushSize * canvas.width / canvas.clientWidth, points: [point] };
    paintStroke(state.currentStroke); state.maskDirty = true; setBusy(false);
  });
  canvas.addEventListener('pointermove', event => {
    if (state.currentStroke) { state.currentStroke.points.push(pointAt(event)); ctx.clearRect(0, 0, canvas.width, canvas.height); if (state.baseMask) ctx.drawImage(state.baseMask, 0, 0, canvas.width, canvas.height); state.maskStrokes.forEach(paintStroke); paintStroke(state.currentStroke); }
    if (state.tool === 'sam-box' && state.promptBox && event.buttons) state.promptBox.splice(2, 2, ...pointAt(event));
  });
  canvas.addEventListener('pointerup', () => { if (state.currentStroke) { state.maskStrokes.push(state.currentStroke); state.currentStroke = null; redrawMask(); } });
  canvas.addEventListener('contextmenu', event => event.preventDefault());
  $('#image-stage').addEventListener('wheel', event => {
    if (event.ctrlKey) { event.preventDefault(); state.zoom = Math.max(.5, Math.min(3, state.zoom * (event.deltaY < 0 ? 1.1 : .9))); updateCanvasSize(); }
  }, { passive: false });
  document.addEventListener('keydown', event => {
    if (event.target.matches('input,textarea,select')) return;
    if (event.key.toLowerCase() === 'b') setTool('brush');
    if (event.key.toLowerCase() === 'e') setTool('eraser');
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); handleAction('undo'); }
  });

  new QWebChannel(qt.webChannelTransport, channel => {
    state.bridge = channel.objects.bridge;
    state.bridge.notify.connect(message => toast(message));
    state.bridge.auxCompleted.connect(() => refresh());
    refresh(); setInterval(refresh, 900);
  });
})();
