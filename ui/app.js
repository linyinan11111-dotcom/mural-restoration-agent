(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const ACTIVE = ['queued', 'planning', 'running', 'evaluating'];
  const state = { bridge: null, image: null, job: null, candidate: null, keepEmpty: false, maskStrokes: [], currentStroke: null,
    baseMask: null, maskDirty: false, tool: 'brush', brushSize: 28, zoom: 1, promptPoints: [], promptLabels: [], promptBox: null,
    proposal: null, references: [], selectedReference: null, registration: null, lastSnapshot: null, toastTimer: null,
    auxVersion: -1, damageItems: [], selectedDamage: null, activeJobId: null, running: false };
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
    clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => node.classList.add('hidden'), 3200);
  };
  const setBusy = (value) => {
    state.running = value;
    document.body.classList.toggle('busy', value);
    $('[data-action="run-repair"]').disabled = value || !state.image || !maskHasPixels();
    $$('[data-action="open-image"], [data-action="load-demo"]').forEach(button => { button.disabled = value; });
    $$('.history-item').forEach(item => { item.disabled = value; });
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
    setBusy(state.running);
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
      state.registration = null; state.references = []; state.selectedReference = null;
      state.damageItems = []; state.selectedDamage = null;
      state.maskStrokes = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      image.removeAttribute('src'); imageWrap.classList.add('hidden'); $('#empty-state').classList.remove('hidden');
      $('#image-name').textContent = '导入壁画图像'; $('#image-meta').textContent = '选择本地图片开始';
      $('#candidate-count').textContent = '0'; $('#mask-fraction').textContent = '';
      $('[data-action="run-repair"]').disabled = true;
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      renderDamage();
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      $('#event-log').innerHTML = '<div class="empty-log">完成掩膜后生成候选。</div>';
      return Promise.resolve();
    }
    state.keepEmpty = false;
    state.image = record; state.job = null; state.candidate = null; state.proposal = null;
    state.registration = null; state.references = []; state.selectedReference = null;
    state.damageItems = []; state.selectedDamage = null;
    state.maskStrokes = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
    state.zoom = 1; state.promptPoints = []; state.promptLabels = []; state.promptBox = null;
    $('#image-name').textContent = record.name;
    $('#image-meta').textContent = `${record.width} × ${record.height} · 原始文件已归档`;
    $('#empty-state').classList.add('hidden'); imageWrap.classList.remove('hidden');
    renderDamage();
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
    invoke('setCurrentImage', record.id).catch(() => {});
    return new Promise(resolve => {
      image.onload = () => {
        canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        updateCanvasSize();
        setBusy(state.running);
        resolve();
      };
      image.onerror = () => { toast('图像无法载入。'); resolve(); };
      image.src = fileUrl(record.file_url);
    });
  }

  // Backend masks are 8-bit grayscale PNGs (white = editable). Canvas masks are
  // alpha-composited red overlays, so a plain drawImage() would make every pixel
  // opaque and turn the editable area into the whole image. Rebuild the alpha
  // channel from the luminance before it becomes baseMask.
  function maskLayerFrom(source, width, height) {
    const layer = document.createElement('canvas');
    layer.width = width; layer.height = height;
    const layerContext = layer.getContext('2d', { willReadFrequently: true });
    layerContext.drawImage(source, 0, 0, width, height);
    const sourceData = layerContext.getImageData(0, 0, width, height);
    const target = layerContext.createImageData(width, height);
    for (let i = 0; i < sourceData.data.length; i += 4) {
      const alpha = sourceData.data[i + 3];
      const luminance = (sourceData.data[i] + sourceData.data[i + 1] + sourceData.data[i + 2]) / 3;
      const editable = alpha > 127 && luminance > 127;
      target.data[i] = 216; target.data[i + 1] = 77; target.data[i + 2] = 69;
      target.data[i + 3] = editable ? 255 : 0;
    }
    layerContext.putImageData(target, 0, 0);
    return layer;
  }

  function installMaskFromSource(source) {
    const width = canvas.width || source.naturalWidth || source.width;
    const height = canvas.height || source.naturalHeight || source.height;
    if (!width || !height) { toast('掩膜载入时画布尺寸未知，请重新载入图像。'); return; }
    state.baseMask = maskLayerFrom(source, width, height);
    state.maskStrokes = []; state.currentStroke = null; state.maskDirty = false;
    redrawMask(); state.maskDirty = false;
  }

  function installMask(url) {
    const maskImage = new Image();
    maskImage.onload = () => installMaskFromSource(maskImage);
    maskImage.onerror = () => toast('掩膜读取失败。');
    maskImage.src = url;
  }

  function showMaskFromBase64(data) {
    const maskImage = new Image();
    maskImage.onload = () => installMaskFromSource(maskImage);
    maskImage.onerror = () => toast('掩膜读取失败。');
    maskImage.src = `data:image/png;base64,${data}`;
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

  async function openImage() {
    if (state.running) return toast('任务运行中，完成或取消后再打开新图像。');
    try { const record = await invoke('openImage'); if (record?.id) await loadImage(record); }
    catch (error) { notifyError(error); }
  }

  async function createJob() {
    if (!state.image || !maskHasPixels()) return toast('请先圈选需要修复的区域。');
    if (state.running) return toast('已有任务在执行，请等待完成或取消。');
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
      state.candidate = null; state.maskDirty = false;
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      showView('results'); toast('修复任务已加入工作队列。');
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
        await loadImage(record);
      }
      const activeJob = (snapshot.jobs || []).find(job => ACTIVE.includes(job.status));
      state.activeJobId = activeJob?.id || null;
      if (state.job) state.job = snapshot.jobs.find(item => item.id === state.job.id) || state.job;
      else if (!state.keepEmpty && snapshot.jobs?.length) state.job = snapshot.jobs.find(item => item.image_id === state.image?.id) || null;
      setBusy(Boolean(activeJob));
      renderJob(state.job);
      renderHistory(snapshot.jobs || []);
      if (snapshot.aux_version !== state.auxVersion) {
        state.auxVersion = snapshot.aux_version;
        renderAux(snapshot.aux);
      }
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
      ['telea', 'Telea · 细裂缝', true], ['ns', 'Navier–Stokes · 细裂缝', true],
      ['reference', '参考证据重建', models.reference], ['diffusion', '扩散推测补全', models.diffusion],
      ['denoise', '局部降噪', true], ['color', '周边色调匹配', true], ['deblur', '局部锐化', true],
      ['research', '壁画研究模型', models.research]];
    backend.innerHTML = items.filter(([, , available]) => available).map(([value, title]) => `<option value="${value}">${title}</option>`).join('');
    backend.value = items.some(([value, , available]) => value === prior && available) ? prior : (models.lama ? 'lama' : 'opencv');
    $('#planner option[value="local"]').disabled = !models.vision;
    $('#planner option[value="api"]').disabled = !models.api_configured;
  }

  function renderJob(job) {
    if (!job) return;
    const running = ACTIVE.includes(job.status);
    $('#job-status').textContent = ({ queued: '排队中', planning: '规划中', running: '修复中', evaluating: '质量检查', review: '等待复核', interrupted: '已中断', failed: '执行失败', cancelled: '已取消' })[job.status] || job.status;
    $('#job-progress').style.width = `${Math.max(0, Math.min(100, job.progress || 0))}%`;
    $('[data-action="run-repair"]').disabled = running || !state.image || !maskHasPixels();
    $('[data-action="run-repair"]').classList.toggle('hidden', running);
    $('[data-action="cancel-job"]').classList.toggle('hidden', !state.activeJobId);
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
      if (job.selected_id === state.candidate?.id && job.review?.note && !$('#review-note').value) $('#review-note').value = job.review.note;
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
    $('[data-action="select-candidate"]').disabled = ACTIVE.includes(state.job?.status);
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
      return `<button class="history-item" data-job="${job.id}" ${state.running ? 'disabled' : ''}><span><span class="history-name">${escapeHtml(img?.name || '壁画任务')}</span><span class="history-detail">${(job.created_at || '').slice(0, 16).replace('T', ' ')} · ${(job.candidates || []).length} 个候选</span></span><span class="history-state">${escapeHtml(status)}</span></button>`;
    }).join('') || '<div class="empty-inline">暂无修复记录</div>';
    $$('.history-item').forEach(item => item.addEventListener('click', async () => {
      if (state.running) { toast('任务运行中，完成或取消后再打开历史任务。'); return; }
      try {
        const job = items.find(record => record.id === item.dataset.job);
        const record = images.find(imageRecord => imageRecord.id === job.image_id);
        await loadImage(record);
        state.job = job; state.candidate = null;
        installMask(job.mask_file_url);
        $('#repair-mode').value = job.mode || 'conservative'; $('#repair-backend').value = job.backend;
        $('#planner').value = job.planner; $('#instructions').value = job.instructions || '';
        $('#review-note').value = job.review?.note || '';
        showPage('workbench'); showView(job.candidates?.length ? 'results' : 'edit'); await refresh();
      } catch (error) { notifyError(error); }
    }));
  }

  function renderAux(aux) {
    if (!aux || aux.status === 'idle') return;
    if (aux.status === 'running') {
      $('#sam-status').textContent = aux.label || '处理中';
      if (/缺损/.test(aux.label || '')) $('#damage-status').textContent = '识别中';
      return;
    }
    if (aux.status === 'error') {
      $('#sam-status').textContent = '出错';
      $('#proposal-note').textContent = aux.error || '工具执行失败。';
      toast(aux.error || '辅助工具执行失败。');
      return;
    }
    const result = aux.result || {};
    if (result.items) {
      state.damageItems = result.items;
      state.selectedDamage = null;
      renderDamage(result.limitations || '');
      $('#damage-status').textContent = result.items.length ? `待确认 ${result.items.length} 项` : '未发现候选';
      toast(result.items.length ? `AI 提出 ${result.items.length} 个疑似缺损，请人工核验。` : 'AI 未发现候选，请人工检查原图。');
      return;
    }
    if (result.proposal) {
      state.proposal = result.proposal;
      if (result.proposal.file_url) {
        $('#proposal-image').src = `${fileUrl(result.proposal.file_url)}?v=${Date.now()}`;
        $('#proposal-image').classList.remove('hidden'); $('#proposal-empty').classList.add('hidden');
      }
      $('#proposal-note').textContent = `${result.proposal.metadata?.notice || '辅助选区待人工确认。'} 候选面积 ${((result.proposal.fraction || 0) * 100).toFixed(2)}%。`;
      $('[data-action="use-proposal"]').disabled = false; $('[data-action="merge-proposal"]').disabled = false;
      $('#sam-status').textContent = '已生成待确认候选'; showPage('tools');
      return;
    }
    if (result.references) { renderReferences(result.references); return; }
    if (result.registration) {
      state.registration = result.registration;
      $('#registration-note').textContent = `内点 ${state.registration.inliers} · 中位误差 ${state.registration.median_error_px.toFixed(2)} px · 请人工核验覆盖区域。`;
      $('#registration-preview').innerHTML = `<img src="${fileUrl(state.registration.overlay_url)}">`;
      $('[data-action="approve-reference"]').disabled = false;
      return;
    }
    if (result.notice) toast(result.notice);
  }

  const DAMAGE_LABELS = { crack: '裂缝', paint_loss: '颜料层缺失', plaster_loss: '地仗/灰泥缺失', flaking: '起甲/剥落', stain: '污渍/变色', other: '其他疑似缺损' };

  function damageRecommendation(item) {
    if (item.recommendation?.method) return item.recommendation.method;
    const size = item.size || 'uncertain';
    if (item.type === 'stain') return '局部色调匹配';
    if (item.type === 'crack') return size === 'large' ? 'LaMa 候选；长裂缝需人工复核' : 'Navier–Stokes 细线修补';
    if (['paint_loss', 'plaster_loss'].includes(item.type) && size === 'large') return 'LaMa 纹理补全';
    if (['paint_loss', 'plaster_loss', 'flaking'].includes(item.type)) return 'Telea 局部纹理传播';
    return '先人工确认，再选择保守修补';
  }

  function renderDamage(limitations = '') {
    const list = $('#damage-list');
    if (!state.image || !state.damageItems.length) {
      list.innerHTML = '<div class="empty-inline">尚未分析，或没有可确认的候选。</div>';
      $('#damage-note').textContent = 'AI 识别、类别与掩膜提取都只是候选，需人工检查；不能替代病害鉴定。' + (limitations ? ' ' + limitations : '');
      $('#damage-image').classList.add('hidden'); $('#damage-empty').classList.remove('hidden');
      $('[data-action="refine-damage"]').disabled = true;
      $('#damage-status').textContent = '待命';
      return;
    }
    list.innerHTML = state.damageItems.map((item, index) => {
      const [x0, y0, x1, y1] = item.bbox || [0, 0, 0, 0];
      const extent = Math.max(0, (x1 - x0) * (y1 - y0)) / 100;
      return `<button class="damage-item ${state.damageItems[index] === state.selectedDamage ? 'selected' : ''}" data-damage="${index}">` +
        `<b>${String(index + 1).padStart(2, '0')} ${escapeHtml(DAMAGE_LABELS[item.type] || '其他疑似缺损')}</b>` +
        `<small>${escapeHtml(item.size || 'uncertain')} · 框选约 ${extent.toFixed(2)}% · 视觉置信 ${Math.round((item.confidence || 0) * 100)}%</small></button>`;
    }).join('');
    $$('.damage-item').forEach(button => button.addEventListener('click', () => selectDamage(Number(button.dataset.damage))));
    $('#damage-note').textContent = (limitations || '') || 'AI 识别、类别与掩膜提取都只是候选，需人工检查；不能替代病害鉴定。';
    if (state.selectedDamage === null) selectDamage(0);
    $('[data-action="refine-damage"]').disabled = !state.image || state.running;
  }

  async function selectDamage(index) {
    const item = state.damageItems[index];
    if (!item || !state.image) return;
    state.selectedDamage = item;
    $$('.damage-item').forEach((button, position) => button.classList.toggle('selected', position === index));
    const [x0, y0, x1, y1] = item.bbox || [0, 0, 0, 0];
    const fraction = Math.max(0, (x1 - x0) * (y1 - y0)) / 100;
    $('#damage-note').textContent = `疑似${DAMAGE_LABELS[item.type] || '缺损'}，AI 估计 ${item.size}，框选约 ${fraction.toFixed(2)}% 图像面积；建议：${damageRecommendation(item)}。视觉置信 ${Math.round((item.confidence || 0) * 100)}% 不是病害概率。依据：${item.evidence || '无文字依据'}。请先检查红框是否落在真实损伤上。`;
    $('[data-action="refine-damage"]').disabled = state.running;
    try {
      const preview = await invoke('damagePreview', state.image.id, JSON.stringify(item));
      if (preview?.file_url) {
        $('#damage-image').src = `${fileUrl(preview.file_url)}?v=${Date.now()}`;
        $('#damage-image').classList.remove('hidden'); $('#damage-empty').classList.add('hidden');
      }
    } catch (error) { console.warn(error); }
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

  function showModal(id) { $(`#${id}`).classList.remove('hidden'); }
  function hideModal(id) { $(`#${id}`).classList.add('hidden'); }

  async function saveReferenceForm() {
    const payload = { source: $('#ref-source').value.trim(), license: $('#ref-license').value.trim(),
      group: $('#ref-group').value.trim(), relation: $('#ref-relation').value };
    if (!payload.source || !payload.license || !payload.group) {
      $('#ref-error').textContent = '来源、许可依据与壁画分组均需填写。';
      return;
    }
    $('#ref-error').textContent = '';
    try {
      const result = await invoke('addReference', JSON.stringify(payload));
      if (!result) return hideModal('reference-modal');
      hideModal('reference-modal'); toast('参考资料已导入。'); await refresh();
    } catch (error) { $('#ref-error').textContent = error.message; }
  }

  async function saveVerifyNote() {
    if (!state.registration) return hideModal('verify-modal');
    const note = $('#verify-note').value.trim();
    if (!note) { $('#verify-error').textContent = '请填写对应关系与核验依据。'; return; }
    try {
      await invoke('approveReference', state.registration.id, exportMaskBase64(), note);
      hideModal('verify-modal'); toast('应用区域已核验并记录。'); await refresh();
    } catch (error) { $('#verify-error').textContent = error.message; }
  }

  async function handleAction(action) {
    try {
      switch (action) {
        case 'new-task':
          if (state.running) return toast('任务运行中，完成或取消后再新建任务。');
          await loadImage(null); state.image = null; $('#empty-state').classList.remove('hidden'); imageWrap.classList.add('hidden'); break;
        case 'open-image': await openImage(); break;
        case 'load-demo': { if (state.running) return toast('任务运行中，完成或取消后再载入图像。'); const record = await invoke('loadDemo'); await loadImage(record); break; }
        case 'undo': if (state.maskStrokes.length) { state.maskStrokes.pop(); redrawMask(); } break;
        case 'clear-mask': state.maskStrokes = []; state.currentStroke = null; state.baseMask = null; redrawMask(); break;
        case 'import-mask': { const result = await invoke('importMask'); if (result?.mask_data) showMaskFromBase64(result.mask_data); break; }
        case 'zoom-in': state.zoom = Math.min(3, state.zoom * 1.15); updateCanvasSize(); break;
        case 'zoom-out': state.zoom = Math.max(.5, state.zoom / 1.15); updateCanvasSize(); break;
        case 'zoom-fit': state.zoom = 1; updateCanvasSize(); break;
        case 'run-repair': await createJob(); break;
        case 'cancel-job': { const identifier = state.activeJobId || state.job?.id; if (identifier) { await invoke('cancelJob', identifier); } await refresh(); break; }
        case 'resume-job': if (state.job) await invoke('resumeJob', state.job.id); await refresh(); break;
        case 'select-candidate': if (state.job && state.candidate) { state.job = await invoke('selectCandidate', state.job.id, state.candidate.id, $('#review-note').value); toast('候选选择已保存。'); renderComparison(); await refresh(); } break;
        case 'save-result': if (state.candidate) await invoke('saveResult', state.candidate.id); break;
        case 'export-archive': if (state.job) await invoke('exportArchive', state.job.id); break;
        case 'open-settings': openSettings(); break;
        case 'close-settings': hideModal('settings-modal'); break;
        case 'save-settings': await saveSettings(); break;
        case 'open-reference': showModal('reference-modal'); $('#ref-error').textContent = ''; break;
        case 'close-reference': hideModal('reference-modal'); break;
        case 'save-reference': await saveReferenceForm(); break;
        case 'open-verify': if (state.registration) { showModal('verify-modal'); $('#verify-error').textContent = ''; } break;
        case 'close-verify': hideModal('verify-modal'); break;
        case 'save-verify': await saveVerifyNote(); break;
        case 'assess-damage': if (!state.image) return toast('请先打开一幅壁画。'); if (state.running) return toast('任务运行中，完成或取消后再运行 AI 识别。'); await invoke('assessDamage', state.image.id); $('#damage-status').textContent = '识别中'; break;
        case 'refine-damage': if (!state.image || !state.selectedDamage) return toast('请先分析并选中一个疑似缺损。'); if (state.running) return toast('任务运行中，完成或取消后再细化选区。'); await invoke('refineDamage', state.image.id, JSON.stringify(state.selectedDamage)); $('#sam-status').textContent = '正在细化选区'; break;
        case 'propose-damage': if (state.image) await invoke('proposeDamage', state.image.id); showPage('tools'); break;
        case 'sam-point': setTool('sam-point'); state.promptPoints = []; state.promptLabels = []; showPage('workbench'); showView('edit'); toast('左键添加前景点，右键添加背景点。'); break;
        case 'sam-box': setTool('sam-box'); state.promptBox = null; showPage('workbench'); showView('edit'); toast('在图像上拖动框选目标。'); break;
        case 'run-sam': if (state.image) await invoke('runSam', state.image.id, JSON.stringify(state.promptPoints), JSON.stringify(state.promptLabels), JSON.stringify(state.promptBox)); break;
        case 'clear-prompts': state.promptPoints = []; state.promptLabels = []; state.promptBox = null; setTool('brush'); toast('提示已清除。'); break;
        case 'use-proposal': if (state.proposal) { await invoke('adoptProposal', state.proposal.id); installMask(state.proposal.mask_file_url); state.proposalIds = [state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'merge-proposal': if (state.proposal) { const current = exportMaskBase64(); state.proposal = await invoke('mergeProposal', state.proposal.id, current); installMask(state.proposal.mask_file_url); state.proposalIds = [...(state.proposalIds || []), state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'add-reference': showModal('reference-modal'); $('#ref-error').textContent = ''; break;
        case 'search-references': await invoke('searchReferences', $('#reference-query').value, $('#reference-group').value); break;
        case 'register-reference': if (state.selectedReference && state.image) await invoke('registerReference', state.image.id, state.selectedReference, exportMaskBase64()); break;
        case 'approve-reference': if (state.registration && state.image) { showModal('verify-modal'); $('#verify-error').textContent = ''; } break;
        case 'import-dataset': { const result = await invoke('importDataset'); if (result?.notice) toast(result.notice); break; }
        case 'export-splits': { const result = await invoke('exportSplits'); if (result?.notice) toast(result.notice); break; }
        case 'evaluate-pair': if (state.candidate) await invoke('evaluatePair', state.candidate.id); else toast('请先选择候选。'); break;
        case 'batch-jobs': { const result = await invoke('batchJobs', JSON.stringify(readSettings())); if (result?.notice) { toast(result.notice); await refresh(); } break; }
        case 'show-models': await invoke('showModels'); break;
        case 'backup-workspace': { const result = await invoke('backupWorkspace'); if (result?.path) toast('工作空间已备份。'); else if (result?.cancelled) toast('已取消备份。'); break; }
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
    showModal('settings-modal');
  }

  async function saveSettings() {
    const form = { kind: $('#api-kind').value, base_url: $('#api-base').value.trim(), model: $('#api-model').value.trim(), key: $('#api-key').value, json_mode: $('#api-json').checked };
    try { await invoke('saveApiSettings', JSON.stringify(form)); hideModal('settings-modal'); toast('API 配置已保存。'); await refresh(); }
    catch (error) { $('#api-error').textContent = error.message; }
  }

  function addReference() { showModal('reference-modal'); $('#ref-error').textContent = ''; }

  function readSettings() {
    return { backend: $('#repair-backend').value, planner: $('#planner').value, count: Number($('#candidate-number').value),
      instructions: $('#instructions').value, rounds: Number($('#revisions').value), timeout: Number($('#timeout').value),
      mode: $('#repair-mode').value };
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
    if (!state.image || !image.naturalWidth || state.running) return;
    const point = pointAt(event); canvas.setPointerCapture(event.pointerId);
    if (state.tool === 'sam-point') {
      state.promptPoints.push(point); state.promptLabels.push(event.button === 2 ? 0 : 1); toast('已记录分割提示点。'); return;
    }
    if (state.tool === 'sam-box') { state.promptBox = [point[0], point[1], point[0], point[1]]; return; }
    if (event.button !== 0) return;
    state.currentStroke = { tool: state.tool, size: state.brushSize * canvas.width / canvas.clientWidth, points: [point] };
    paintStroke(state.currentStroke); state.maskDirty = true; setBusy(state.running);
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
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') { event.preventDefault(); openImage(); return; }
    if (event.key.toLowerCase() === 'b') setTool('brush');
    if (event.key.toLowerCase() === 'e') setTool('eraser');
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); handleAction('undo'); }
    if (event.key === 'Escape') { ['settings-modal', 'reference-modal', 'verify-modal'].forEach(hideModal); }
  });

  new QWebChannel(qt.webChannelTransport, channel => {
    state.bridge = channel.objects.bridge;
    state.bridge.notify.connect(message => toast(message));
    state.bridge.auxCompleted.connect(() => refresh());
    refresh(); setInterval(refresh, 900);
  });

  // Test seam: the acceptance tests drive the same mask round-trip and job flow
  // through the shipped frontend instead of duplicating the logic in Python.
  window.__shibui = { state, loadImage, installMask, showMaskFromBase64, exportMaskBase64, refresh, renderDamage };
})();
