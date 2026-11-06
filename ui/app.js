(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const ACTIVE = ['queued', 'planning', 'running', 'evaluating'];
  const state = { bridge: null, image: null, job: null, candidate: null, keepEmpty: false, maskStrokes: [], regionStrokes: [], strokeOrder: [], currentStroke: null,
    baseMask: null, maskDirty: false, tool: 'brush', brushSize: 28, zoom: 1, promptPoints: [], promptLabels: [], promptBox: null,
    proposal: null, references: [], selectedReference: null, registration: null, lastSnapshot: null, toastTimer: null,
    auxVersion: -1, damageItems: [], selectedDamage: null, activeJobId: null, running: false, autoBusy: false, observe: null, gateReview: null };
  const canvas = $('#mask-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const regionCanvas = $('#region-canvas');
  const regionCtx = regionCanvas ? regionCanvas.getContext('2d', { willReadFrequently: true }) : null;
  const image = $('#source-image');
  const imageWrap = $('#canvas-wrap');
  const REGION_COLORS = {
    uncertain: 'rgba(220,0,60,0.72)',
    uncertainty: 'rgba(220,0,60,0.72)', // alias
    unrestorable: 'rgba(180,0,40,0.85)',
  };

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
    $$('[data-action="open-image"], [data-action="load-demo"], [data-action="load-competition-demo"]').forEach(button => { button.disabled = value; });
    $$('.history-item').forEach(item => { item.disabled = value; });
  };
  const maskHasPixels = () => {
    if (!canvas.width || !canvas.height) return false;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
    return false;
  };
  const notifyError = (error) => toast(error?.message || String(error));
  const AUTO_HINT = '由本地 Qwen 自动识别缺损、生成掩膜并规划保守修复；结果仍需人工复核。';
  const AUTO_HINT_PRO = 'AI 会列出疑似缺损。核对选区、选择修补方案，再逐项或批量运行。';
  const DEMO_COPY = {
    footer: '辅助观察与方案讨论，不替代专业判断；本流程不含图像生成修复。',
    A: {
      step: 'AI 仅做病害/露底观察标注与文献摘录，不生成、不补绘壁画图像。',
      risk: '红区禁绘；AI 标注非病害鉴定结论。',
      gate: '人工确认标注与出处后，仅可导出标注层、复核记录与说明文；禁止导出修复成图。',
      export: '仅标注层 + 复核记录 + 说明，不含修复成图。',
    },
    B: {
      step: '不确定区标红 + 拒答停绘 + 人工写处置建议；不输出补绘成图。',
      risk: '任何补纹样、凭空出画面的请求一律拒答；不确定区只标红，不填色成图。',
      refuse: '依据不足 / 涉及补绘生成 → 已拒答，仅保留原状记录。',
      gate: '须人工选择：采纳 / 修改 / 驳回；未选择不得进入下一档或导出终稿。',
      export: '仅标注层 + 复核记录 + 说明，不含修复成图。',
    },
  };
  const isProfessional = () => Boolean($('#professional-step')?.checked);
  const currentWorkflow = () => (isProfessional() ? 'professional_step' : 'standard');
  const isRegionTool = (tool) => tool === 'uncertain' || tool === 'uncertainty' || tool === 'unrestorable';
  const regionPixelStats = () => {
    const stats = { uncertain: false, unrestorable: false };
    if (!regionCanvas || !regionCtx || !regionCanvas.width) return stats;
    const data = regionCtx.getImageData(0, 0, regionCanvas.width, regionCanvas.height).data;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3]; if (a < 40) continue;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      // unrestorable: deeper red, lower brightness
      if (r > 150 && g < 50 && r - g > 120) stats.unrestorable = true;
      else if (r > 160 && g < 90) stats.uncertain = true;
    }
    return stats;
  };
  const hasUnrestorableRegions = () => regionPixelStats().unrestorable;
  const hasUncertainRegions = () => regionPixelStats().uncertain;
  const selectedDecision = () => {
    const node = document.querySelector('input[name="review-decision"]:checked');
    return node ? node.value : '';
  };

  const isObserveMode = () => Boolean(state.observe);
  const gateExport = () => (state.gateReview?.export_allowed || state.observe?.export_allowed || {});
  const gateState = () => (state.gateReview?.gate || state.observe?.gate || {});
  const syncGateExportButtons = () => {
    const observe = isObserveMode();
    const gate = gateState();
    const exp = gateExport();
    const blocked = Boolean(gate.blocked);
    const canAnno = observe ? (!blocked && Boolean(exp.annotation_layer)) : true;
    const canReview = observe ? (!blocked && Boolean(exp.review_record)) : Boolean(state.job?.selected_id);
    const overlayBtn = $('[data-action="export-overlay"]');
    const archiveBtn = $('[data-action="export-archive"]');
    const confirmBtn = $('[data-action="select-candidate"]');
    const sampleBtn = $('[data-action="load-gate-reviewed-sample"]');
    if (overlayBtn) overlayBtn.disabled = observe ? !canAnno : !state.image;
    if (archiveBtn) archiveBtn.disabled = observe ? !canReview : !state.job?.selected_id;
    if (confirmBtn) {
      if (observe) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认复核决定';
      }
    }
    if (sampleBtn) sampleBtn.classList.toggle('hidden', !observe);
    const checklist = $('#gate-checklist');
    if (checklist) checklist.classList.toggle('hidden', !observe);
    const abstainInput = document.querySelector('input[name="review-decision"][value="abstain"]');
    const abstainLabel = abstainInput ? abstainInput.closest('label') : null;
    if (abstainLabel) abstainLabel.classList.toggle('hidden', !observe);
    const banner = $('#gate-status-banner');
    if (banner) {
      const text = state.gateReview?.banner || state.observe?.banner || '';
      if (observe && text) {
        banner.classList.remove('hidden');
        banner.classList.toggle('blocked', blocked);
        banner.classList.toggle('clear', !blocked);
        banner.textContent = text;
      } else if (!observe) {
        banner.classList.add('hidden');
        banner.textContent = '';
      }
    }
  };
  const readGateChecklist = () => ({
    repairable_checked: Boolean($('#check-repairable')?.checked),
    uncertainty_checked: Boolean($('#check-uncertainty')?.checked),
    unrestorable_checked: Boolean($('#check-unrestorable')?.checked),
    literature_checked: Boolean($('#check-literature')?.checked),
  });
  const applyGatePayload = (payload) => {
    if (!payload) return;
    state.gateReview = payload;
    if (state.observe) {
      state.observe.gate = payload.gate;
      state.observe.export_allowed = payload.export_allowed;
      state.observe.banner = payload.banner;
    }
    if (payload.checklist) {
      if ($('#check-repairable')) $('#check-repairable').checked = Boolean(payload.checklist.repairable_checked);
      if ($('#check-uncertainty')) $('#check-uncertainty').checked = Boolean(payload.checklist.uncertainty_checked);
      if ($('#check-unrestorable')) $('#check-unrestorable').checked = Boolean(payload.checklist.unrestorable_checked);
      if ($('#check-literature')) $('#check-literature').checked = Boolean(payload.checklist.literature_checked);
    }
    syncGateExportButtons();
  };
  const setAutoStatus = (text) => {
    const fallback = isProfessional() ? AUTO_HINT_PRO : AUTO_HINT;
    $('#auto-status').textContent = text || fallback;
  };
  const syncProfessionalUi = () => {
    const on = isProfessional();
    const autoBtn = $('[data-action="auto-restore"]');
    if (autoBtn) autoBtn.textContent = 'AI 分析缺损与文化线索';
    setAutoStatus();
    const note = $('#professional-note');
    if (note) note.style.opacity = on ? '1' : '.72';
    const mode = $('#repair-mode')?.value || 'conservative';
    const scene = (mode === 'enhanced' || mode === 'speculative') ? 'B' : 'A';
    const hint = $('#demo-step-hint');
    const banner = $('#demo-risk-banner');
    if (hint) {
      hint.classList.toggle('hidden', !on);
      hint.textContent = on ? `${DEMO_COPY[scene].step} ${DEMO_COPY[scene].gate}` : '';
    }
    if (banner) {
      banner.classList.toggle('hidden', !on);
      banner.textContent = on ? DEMO_COPY[scene].risk : '';
    }
    const riskAck = $('#risk-ack')?.closest('label');
    if (riskAck) riskAck.style.display = on && mode === 'enhanced' ? '' : 'none';
    const allowEnhanced = $('#allow-enhanced')?.closest('label');
    if (allowEnhanced) allowEnhanced.style.display = on && mode === 'enhanced' ? '' : 'none';
    const maskGate = $('#mask-gate-confirmed')?.closest('label');
    if (maskGate) maskGate.style.display = on ? '' : 'none';
    const force = $('#force-override')?.closest('label');
    const showForce = on && hasUncertainRegions();
    if (force) force.style.display = showForce ? '' : 'none';
    const reason = $('#force-override-reason');
    if (reason) reason.classList.toggle('hidden', !(showForce && $('#force-override')?.checked));
    // Disable run when unrestorable present
    const blocked = on && hasUnrestorableRegions();
    const runBtn = $('[data-action="run-repair"]');
    if (runBtn && blocked) runBtn.disabled = true;
    syncGateExportButtons();
  };
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
    if (view === 'results') requestAnimationFrame(layoutComparison);
  }

  function setTool(tool) {
    state.tool = tool;
    $$('.tool-button').forEach(item => item.classList.toggle('active', item.dataset.tool === tool));
    canvas.classList.toggle('sam-point', tool === 'sam-point');
    canvas.classList.toggle('sam-box', tool === 'sam-box');
    canvas.classList.toggle('eraser', tool === 'eraser');
    $('#mask-status').textContent = tool === 'sam-point' ? '左键添加前景点，右键添加背景点' :
      tool === 'sam-box' ? '在图像上拖动框选目标' :
      tool === 'uncertain' || tool === 'uncertainty' ? '红色：不确定区（默认禁修，须强制放行才可保守）' :
      tool === 'unrestorable' ? '深红：不可复原区（禁止任何像素修复）' :
      tool === 'eraser' ? '橡皮：同时清除可修掩膜与标红区' :
      '琥珀/红：可修露底选区（仅保守）';
  }

  function updateCanvasSize() {
    if (!state.image || !image.naturalWidth) return;
    imageWrap.style.transform = `scale(${state.zoom})`;
    requestAnimationFrame(() => {
      canvas.style.width = `${image.clientWidth}px`;
      canvas.style.height = `${image.clientHeight}px`;
      if (regionCanvas) {
        regionCanvas.style.width = `${image.clientWidth}px`;
        regionCanvas.style.height = `${image.clientHeight}px`;
      }
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
    if (isRegionTool(stroke.tool)) return paintRegionStroke(stroke);
    ctx.save();
    ctx.globalCompositeOperation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#f5a623'; ctx.fillStyle = '#f5a623'; ctx.globalAlpha = 1;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = stroke.size;
    if (stroke.points.length === 1) {
      ctx.beginPath(); ctx.arc(stroke.points[0][0], stroke.points[0][1], stroke.size / 2, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.beginPath(); ctx.moveTo(...stroke.points[0]);
      stroke.points.slice(1).forEach(point => ctx.lineTo(...point)); ctx.stroke();
    }
    ctx.restore();
    if (stroke.tool === 'eraser' && regionCtx) {
      regionCtx.save();
      regionCtx.globalCompositeOperation = 'destination-out';
      regionCtx.lineCap = 'round'; regionCtx.lineJoin = 'round'; regionCtx.lineWidth = stroke.size;
      regionCtx.strokeStyle = 'rgba(0,0,0,1)'; regionCtx.fillStyle = 'rgba(0,0,0,1)';
      if (stroke.points.length === 1) {
        regionCtx.beginPath(); regionCtx.arc(stroke.points[0][0], stroke.points[0][1], stroke.size / 2, 0, Math.PI * 2); regionCtx.fill();
      } else {
        regionCtx.beginPath(); regionCtx.moveTo(...stroke.points[0]);
        stroke.points.slice(1).forEach(point => regionCtx.lineTo(...point)); regionCtx.stroke();
      }
      regionCtx.restore();
    }
  }

  function paintRegionStroke(stroke) {
    if (!regionCtx || !stroke.points.length) return;
    regionCtx.save();
    regionCtx.globalCompositeOperation = 'source-over';
    const color = REGION_COLORS[stroke.tool] || REGION_COLORS.uncertainty;
    regionCtx.strokeStyle = color; regionCtx.fillStyle = color;
    regionCtx.lineCap = 'round'; regionCtx.lineJoin = 'round'; regionCtx.lineWidth = stroke.size;
    if (stroke.points.length === 1) {
      regionCtx.beginPath(); regionCtx.arc(stroke.points[0][0], stroke.points[0][1], stroke.size / 2, 0, Math.PI * 2); regionCtx.fill();
    } else {
      regionCtx.beginPath(); regionCtx.moveTo(...stroke.points[0]);
      stroke.points.slice(1).forEach(point => regionCtx.lineTo(...point)); regionCtx.stroke();
    }
    regionCtx.restore();
  }

  function redrawRegions() {
    if (!regionCtx) return;
    regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
    state.regionStrokes.forEach(paintRegionStroke);
    if (state.currentStroke && isRegionTool(state.currentStroke.tool)) paintRegionStroke(state.currentStroke);
  }

  function redrawMask() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.baseMask) ctx.drawImage(state.baseMask, 0, 0, canvas.width, canvas.height);
    state.maskStrokes.filter(s => !isRegionTool(s.tool)).forEach(paintStroke);
    if (state.currentStroke && !isRegionTool(state.currentStroke.tool)) paintStroke(state.currentStroke);
    redrawRegions();
    state.maskDirty = true;
    $('#mask-fraction').textContent = `${Math.round(maskFraction() * 10000) / 100}%`;
    setBusy(state.running);
  }

  function pointAt(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height];
  }


  const CLASS_CHIP = {
    repairable: 'chip-repairable',
    fillable_lacuna: 'chip-repairable',
    uncertainty: 'chip-uncertainty',
    uncertain: 'chip-uncertainty',
    unrestorable: 'chip-unrestorable',
  };
  function clearObservePack() {
    state.observe = null;
    state.gateReview = null;
    const panel = $('#observe-preview');
    if (panel) panel.classList.add('hidden');
    ['observe-overlay', 'observe-compare', 'observe-legend'].forEach(id => {
      const img = $('#' + id); if (img) { img.removeAttribute('src'); img.classList.add('hidden'); }
    });
    const list = $('#observe-regions'); if (list) list.innerHTML = '';
    ['check-repairable','check-uncertainty','check-unrestorable','check-literature'].forEach(id => {
      const node = $('#' + id); if (node) node.checked = false;
    });
    const gbanner = $('#gate-status-banner');
    if (gbanner) { gbanner.classList.add('hidden'); gbanner.textContent = ''; }
    syncGateExportButtons();
  }
  function renderObservePack(observe) {
    state.observe = observe || null;
    const panel = $('#observe-preview');
    if (!panel) return;
    if (!observe) { clearObservePack(); return; }
    panel.classList.remove('hidden');
    const setImg = (id, url) => {
      const img = $('#' + id);
      if (!img) return;
      if (url) { img.src = fileUrl(url); img.classList.remove('hidden'); }
      else { img.removeAttribute('src'); img.classList.add('hidden'); }
    };
    setImg('observe-overlay', observe.overlay_risk_url);
    setImg('observe-compare', observe.compare_url);
    setImg('observe-legend', observe.legend_url);
    const note = $('#observe-export-note');
    if (note) note.textContent = observe.export_note || DEMO_COPY.A.export;
    const banner = $('#demo-risk-banner');
    if (banner) {
      banner.classList.remove('hidden');
      banner.textContent = observe.risk_note || DEMO_COPY.A.risk;
    }
    const footer = $('#demo-footer');
    if (footer) footer.textContent = DEMO_COPY.footer;
    const list = $('#observe-regions');
    if (!list) return;
    const regions = Array.isArray(observe.regions) ? observe.regions : [];
    if (!regions.length) {
      list.innerHTML = '<div class="empty-inline">无区域条目</div>';
      if (observe.gate || observe.export_allowed || observe.banner) {
        applyGatePayload({
          gate: observe.gate,
          export_allowed: observe.export_allowed,
          banner: observe.banner,
          checklist: observe.checklist,
          candidates: observe.candidates,
        });
      } else {
        syncGateExportButtons();
      }
      showView('results');
      return;
    }
    list.innerHTML = regions.map(r => {
      const klass = r.class || r.class_alias || 'uncertainty';
      const chip = CLASS_CHIP[klass] || 'chip-uncertainty';
      const allowed = r.repair_allowed ? '可候选保守填（须人工）' : '禁绘 / 不填色';
      const evidence = (r.evidence || '').replace(/</g, '&lt;');
      const label = (r.label || klass).replace(/</g, '&lt;');
      return `<article class="observe-region ${chip}" data-region-id="${r.id}">
        <header><span class="region-chip">${label}</span><span class="region-id">#${r.id}</span><span class="region-allow">${allowed}</span></header>
        <p>${evidence}</p>
      </article>`;
    }).join('');
    // Prefer results view so judges see overlay strip immediately
    showView('results');
    if (observe.gate || observe.export_allowed || observe.banner) {
      applyGatePayload({
        gate: observe.gate,
        export_allowed: observe.export_allowed,
        banner: observe.banner,
        checklist: observe.checklist,
        candidates: observe.candidates,
      });
    } else {
      syncGateExportButtons();
    }
  }

  function loadImage(record) {
    if (!record) {
      state.keepEmpty = true;
      state.image = state.job = state.candidate = state.proposal = null;
      state.registration = null; state.references = []; state.selectedReference = null;
      state.damageItems = []; state.selectedDamage = null;
      $('#mask-gate-confirmed').checked = false;
      $('#culture-summary').classList.add('hidden');
      state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
      state.proposalIds = [];
      clearObservePack();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (regionCtx) regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height);
      image.removeAttribute('src'); imageWrap.classList.add('hidden'); $('#empty-state').classList.remove('hidden');
      $('#image-name').textContent = '导入壁画图像'; $('#image-meta').textContent = '选择本地图片开始';
      $('#candidate-count').textContent = '0'; $('#mask-fraction').textContent = '';
      $('#job-status').textContent = '尚未开始'; $('#job-progress').style.width = '0%';
      $('#job-percent').textContent = '0%';
      $('#progress-estimate').textContent = '等待任务';
      $('[data-action="run-repair"]').disabled = true;
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      renderDamage();
      state.autoBusy = false; setAutoStatus(AUTO_HINT);
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      $('#event-log').innerHTML = '<div class="empty-log">完成掩膜后生成候选。</div>';
      return Promise.resolve();
    }
    state.keepEmpty = false;
    clearObservePack();
    state.image = record; state.job = null; state.candidate = null; state.proposal = null;
    state.registration = null; state.references = []; state.selectedReference = null;
    state.damageItems = []; state.selectedDamage = null;
    $('#mask-gate-confirmed').checked = false;
    $('#culture-summary').classList.add('hidden');
    state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.baseMask = null; state.maskDirty = false;
    if (regionCtx) regionCtx.clearRect(0, 0, regionCanvas.width || 0, regionCanvas.height || 0);
    state.zoom = 1; state.promptPoints = []; state.promptLabels = []; state.promptBox = null;
    $('#image-name').textContent = record.name;
    $('#image-meta').textContent = `${record.width} × ${record.height} · 原始文件已归档`;
    $('#empty-state').classList.add('hidden'); imageWrap.classList.remove('hidden');
    state.autoBusy = false; setAutoStatus(AUTO_HINT);
    renderDamage();
    $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
    $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
    $('#candidate-count').textContent = '0';
    $('#job-status').textContent = '尚未开始'; $('#job-progress').style.width = '0%';
    $('#job-percent').textContent = '0%';
    $('#progress-estimate').textContent = '等待任务';
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
        if (regionCanvas) { regionCanvas.width = image.naturalWidth; regionCanvas.height = image.naturalHeight; regionCtx.clearRect(0, 0, regionCanvas.width, regionCanvas.height); }
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
    $('#mask-gate-confirmed').checked = false;
    state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.maskDirty = false;
    redrawMask(); state.maskDirty = false;
  }

  function installMask(url) {
    const expectedImageId = state.image?.id;
    return new Promise((resolve, reject) => {
      const maskImage = new Image();
      maskImage.onload = () => {
        try {
          if (state.image?.id === expectedImageId) installMaskFromSource(maskImage);
          resolve();
        }
        catch (error) { reject(error); }
      };
      maskImage.onerror = () => { toast('掩膜读取失败。'); reject(new Error('掩膜读取失败。')); };
      maskImage.src = url;
    });
  }

  function showMaskFromBase64(data) {
    const expectedImageId = state.image?.id;
    return new Promise((resolve, reject) => {
      const maskImage = new Image();
      maskImage.onload = () => {
        try {
          if (state.image?.id === expectedImageId) installMaskFromSource(maskImage);
          resolve();
        }
        catch (error) { reject(error); }
      };
      maskImage.onerror = () => { toast('掩膜读取失败。'); reject(new Error('掩膜读取失败。')); };
      maskImage.src = `data:image/png;base64,${data}`;
    });
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
    if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再打开新图像。');
    try { const record = await invoke('openImage'); if (record?.id) await loadImage(record); }
    catch (error) { notifyError(error); }
  }

  async function createJob() {
    if (!state.image || !maskHasPixels()) return toast('请先圈选需要修复的区域。');
    if (state.autoBusy) return toast('AI 分析仍在运行，请完成后再启动修复。');
    if (state.running) return toast('已有任务在执行，请等待完成或取消。');
    if (isProfessional()) {
      if (!$('#mask-gate-confirmed')?.checked) return toast('专业分步：请勾选「已核对掩膜与依据」后再进入下一档。');
      if ($('#evidence-level').value === 'observation') return toast('专业分步：observation 只可记录观察，不能执行修复。');
      if (hasUnrestorableRegions()) return toast('存在不可复原红区：禁止任何像素修复，请只导出标注图与报告。');
      if (hasUncertainRegions()) {
        if ($('#evidence-level').value !== 'conservative' || $('#repair-mode').value !== 'conservative') {
          return toast('不确定红区仅可在强制放行后做保守修补。');
        }
        if (!$('#force-override')?.checked) return toast('不确定红区须勾选「专家强制放行」并填写理由。');
        if (!($('#force-override-reason')?.value || '').trim()) return toast('请填写强制放行理由。');
      }
      if ($('#repair-mode').value === 'speculative' || $('#evidence-level').value === 'speculative') {
        if ($('#repair-mode').value !== 'speculative' || $('#evidence-level').value !== 'speculative') {
          return toast('专业分步：推测须同时将修复模式与证据等级设为 speculative。');
        }
      }
      if ($('#repair-mode').value === 'enhanced' || $('#evidence-level').value === 'enhanced') {
        if ($('#repair-mode').value !== 'enhanced' || $('#evidence-level').value !== 'enhanced') {
          return toast('专业分步：加强修复须同时将修复模式与证据等级设为 enhanced。');
        }
        if (!$('#allow-enhanced')?.checked) {
          return toast('专业分步：加强修复须勾选「允许加强修复」。');
        }
        if (!$('#risk-ack')?.checked) {
          return toast('专业分步：加强修复须勾选风险提示（文化边界）已知悉。');
        }
      }
    }
    setBusy(true);
    try {
      const rationale = ($('#rationale')?.value || '').trim() ||
        `依据：仅处理经人工确认的当前选区，掩膜外像素保持不变。${$('#repair-mode').value === 'speculative' ? ' 推测结果仅作假设性建议，需人工复核。' : ' AI 补全仅为待复核候选，不代表历史原貌。'}`;
      const mask = await invoke('saveMask', state.image.id, exportMaskBase64(), JSON.stringify(state.proposalIds || []));
      const request = { image_id: state.image.id, mask_id: mask.id,
        backend: $('#repair-backend').value, planner: $('#planner').value,
        candidate_count: Number($('#candidate-number').value), instructions: $('#instructions').value,
        mode: $('#repair-mode').value, registration_id: state.registration?.id || null,
        structure_control: $('#controlnet').checked,
        budget: { candidates: Number($('#candidate-number').value), revisions: Number($('#revisions').value), timeout_seconds: Number($('#timeout').value), max_api_calls: 1 },
        quality_thresholds: { boundary_pair_delta: 65 },
        workflow: currentWorkflow(),
        rationale,
        evidence_level: $('#evidence-level')?.value || null,
        expert_notes: ($('#expert-notes')?.value || '').trim(),
        mask_source: (state.proposalIds || []).length ? 'ai_candidate' : 'manual',
        allow_enhanced: Boolean($('#allow-enhanced')?.checked),
        risk_acknowledged: Boolean($('#risk-ack')?.checked),
        mask_gate_confirmed: Boolean($('#mask-gate-confirmed')?.checked),
        has_unrestorable: hasUnrestorableRegions(),
        has_uncertain: hasUncertainRegions(),
        force_override: Boolean($('#force-override')?.checked),
        force_override_reason: ($('#force-override-reason')?.value || '').trim(),
        contest_demo: false };
      state.job = await invoke('createJob', JSON.stringify(request));
      state.candidate = null; state.maskDirty = false;
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      showView('results'); toast(isProfessional() ? '专业分步任务已加入队列（单块执行）。' : '修复任务已加入工作队列。');
      await refresh();
    } catch (error) { setBusy(false); notifyError(error); }
  }

  async function refresh() {
    if (!state.bridge) return;
    try {
      const hadImage = Boolean(state.image);
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
      else if (!hadImage && !state.keepEmpty && snapshot.jobs?.length) state.job = snapshot.jobs.find(item => item.image_id === state.image?.id) || null;
      setBusy(Boolean(activeJob));
      $('[data-action="auto-restore"]').disabled = Boolean(activeJob) || state.autoBusy || !state.image;
      renderJob(state.job);
      renderHistory(snapshot.jobs || []);
      if (state.damageItems.length) renderDamage();
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
    $('#job-percent').textContent = `${Math.max(0, Math.min(100, Math.round(job.progress || 0)))}%`;
    const startedAt = Date.parse(job.started_at || '');
    const liveElapsed = running && Number.isFinite(startedAt) ? Math.max(0, (Date.now() - startedAt) / 1000) : 0;
    const elapsed = Math.max(0, Math.round((job.elapsed_seconds || 0) + liveElapsed));
    let progressCopy = '等待任务';
    if (running && job.progress > 18 && job.progress < 100) {
      const fraction = Math.min(.98, Math.max(.02, (job.progress - 18) / 75));
      const remaining = Math.max(1, Math.round(elapsed * (1 - fraction) / fraction));
      progressCopy = `预计还需 ${remaining} 秒 · 已用 ${elapsed} 秒`;
    } else if (running) {
      const limit = Math.max(0, Number(job.budget?.timeout_seconds) || 0);
      progressCopy = limit
        ? `正在准备模型与修复方案 · 已用 ${elapsed} 秒 · 时间上限 ${limit} 秒`
        : `正在准备模型与修复方案 · 已用 ${elapsed} 秒`;
    } else if (job.status === 'review') {
      progressCopy = `已完成 · 共用 ${Math.round(job.elapsed_seconds || elapsed)} 秒`;
    } else if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
      progressCopy = `任务${job.status === 'failed' ? '失败' : job.status === 'cancelled' ? '已取消' : '已中断'} · 已用 ${elapsed} 秒`;
    }
    $('#progress-estimate').textContent = progressCopy;
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
    if (!candidates.length) {
      state.candidate = null;
      $('#candidate-strip').innerHTML = '<div class="empty-inline">此任务还没有修复候选</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">任务完成后，候选结果会显示在这里</div>';
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      return;
    }
    const selected = state.candidate?.id || job.selected_id || candidates[0].id;
    if (!state.candidate) state.candidate = candidates.find(item => item.id === selected) || candidates[0];
    const methodNames = { lama: 'LaMa 纹理补全', telea: 'Telea 快速行进', ns: 'Navier–Stokes 边缘传播',
      reference: '已核验壁画参考', diffusion: '扩散补全', denoise: '局部降噪', color: '局部色调匹配', deblur: '局部锐化', research: '研究模型' };
    $('#candidate-strip').innerHTML = candidates.map((candidate, index) => `<button class="candidate-card ${candidate.id === state.candidate?.id ? 'active' : ''}" data-candidate="${candidate.id}"><img src="${fileUrl(candidate.file_url)}"><span>${index + 1}. ${escapeHtml(methodNames[candidate.tool] || candidate.tool)}</span></button>`).join('');
    $$('.candidate-card').forEach(item => item.addEventListener('click', () => {
      state.candidate = candidates.find(candidate => candidate.id === item.dataset.candidate); renderCandidates(job); renderComparison();
      if (job.selected_id === state.candidate?.id && job.review?.note && !$('#review-note').value) $('#review-note').value = job.review.note;
    }));
    renderComparison();
  }

  function applySwipe() {
    const layer = $('.comparison-original');
    if (!layer) return;
    const ratio = Math.min(100, Math.max(0, Number($('#compare-range').value) || 0));
    layer.style.width = `${ratio}%`;
    const inner = layer.querySelector('img');
    if (inner) inner.style.width = ratio > 0 ? `${10000 / ratio}%` : '0';
  }

  function layoutComparison() {
    const frame = $('.comparison-frame');
    if (!frame || !state.image) return;
    const stage = $('#comparison-stage');
    const width = state.image.width, height = state.image.height;
    if (!width || !height) return;
    const availableWidth = stage.clientWidth - 2;
    const availableHeight = stage.clientHeight - 2;
    if (availableWidth <= 0 || availableHeight <= 0) return;
    const scale = Math.min(availableWidth / width, availableHeight / height);
    frame.style.width = `${Math.max(1, Math.round(width * scale))}px`;
    frame.style.height = `${Math.max(1, Math.round(height * scale))}px`;
    applySwipe();
  }

  function renderComparison() {
    const candidate = state.candidate; if (!candidate || !state.image) return;
    const stage = $('#comparison-stage'); const mode = $('#comparison-mode').value;
    const sourceUrl = fileUrl(state.image.file_url);
    const url = mode === 'difference' ? fileUrl(candidate.difference_url) : fileUrl(candidate.file_url);
    // One frame sized to the working image keeps the candidate and the original
    // on exactly the same rectangle; previously the overlay was sized to the
    // stage height while the candidate was letterboxed, so the swipe did not line up.
    stage.innerHTML = '';
    const frame = document.createElement('div');
    frame.className = 'comparison-frame';
    const base = document.createElement('img');
    base.className = 'comparison-image';
    base.src = url;
    frame.appendChild(base);
    if (mode === 'compare') {
      const layer = document.createElement('div');
      layer.className = 'comparison-original';
      const original = document.createElement('img');
      original.src = sourceUrl;
      layer.appendChild(original);
      frame.appendChild(layer);
      $('#compare-range').oninput = applySwipe;
    }
    stage.appendChild(frame);
    $('#compare-range').disabled = mode !== 'compare';
    layoutComparison();
    const metrics = candidate.metrics || {};
    $('#candidate-device').textContent = `${candidate.versions?.device || ''} · ${candidate.elapsed_seconds || 0}s`;
    $('#candidate-metrics').textContent = `掩膜外最大差异 ${metrics.outside_max_difference ?? '-'} · 边界色差 ${metrics.boundary_color_gap ?? '-'} · ${candidate.speculative ? '推测性内容需单独复核' : (candidate.enhanced || state.job?.mode === 'enhanced') ? '加强修复·选区附近纹理补全·非历史定稿' : '工程指标不代表历史真实性'}`;
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
      if (state.running || state.autoBusy) { toast('任务或 AI 分析进行中，完成后再打开历史任务。'); return; }
      try {
        const job = items.find(record => record.id === item.dataset.job);
        const record = images.find(imageRecord => imageRecord.id === job.image_id);
        await loadImage(record);
        state.job = job; state.candidate = null;
        await installMask(job.mask_file_url);
        $('#repair-mode').value = job.mode || 'conservative'; $('#repair-backend').value = job.backend;
        $('#planner').value = ['rules', 'local', 'api'].includes(job.planner) ? job.planner : 'rules';
        $('#instructions').value = job.instructions || '';
        $('#review-note').value = job.review?.note || '';
        state.autoBusy = false;
        setAutoStatus(job.auto?.summary || AUTO_HINT);
        showPage('workbench'); showView(job.candidates?.length ? 'results' : 'edit'); await refresh();
      } catch (error) { notifyError(error); }
    }));
  }

  async function renderAux(aux) {
    if (!aux || aux.status === 'idle') return;
    if (aux.status === 'running') {
      $('#sam-status').textContent = aux.label || '处理中';
      if (/缺损/.test(aux.label || '')) $('#damage-status').textContent = '识别中';
      if ((aux.label || '').includes('AI 正在分析')) setAutoStatus(aux.label);
      return;
    }
    if (aux.status === 'error') {
      $('#sam-status').textContent = '出错';
      $('#proposal-note').textContent = aux.error || '工具执行失败。';
      if (state.autoBusy) {
        state.autoBusy = false;
        setAutoStatus('AI 缺损分析失败：' + (aux.error || '未知错误'));
      }
      toast(aux.error || '辅助工具执行失败。');
      return;
    }
    const result = aux.result || {};
    if (result.auto) {
      state.autoBusy = false;
      const auto = result.auto;
      state.damageItems = auto.defects || [];
      state.selectedDamage = null;
      state.job = null;
      state.candidate = null;
      state.proposalIds = [];
      renderDamage(auto.summary || '');
      $('#damage-status').textContent = state.damageItems.length ? `识别 ${state.damageItems.length} 项` : '无候选';
      setAutoStatus(auto.summary || 'AI 分析完成，请人工核对缺损清单。');
      $('#repair-mode').value = 'conservative';
      if (auto.backend) $('#repair-backend').value = auto.backend;
      $('#planner').value = state.lastSnapshot?.models?.vision ? 'local' : 'rules';
      $('#rationale').value = auto.summary || 'AI 提出的缺损候选，已人工检查选区。';
      $('#instructions').value = auto.planning_notes || '文化背景未核实；只做最小保守修补，不臆造壁画内容。';
      renderCulturalContext(auto.cultural_context || {});
      $('#candidate-strip').innerHTML = '<div class="empty-inline">修复候选会显示在这里</div>';
      $('#comparison-stage').innerHTML = '<div class="empty-inline">选择候选查看结果</div>';
      $('#candidate-count').textContent = '0';
      $('#job-status').textContent = '尚未开始';
      $('#job-progress').style.width = '0%';
      $('#job-percent').textContent = '0%';
      $('#progress-estimate').textContent = '等待任务';
      $('[data-action="save-result"]').disabled = true;
      $('[data-action="select-candidate"]').disabled = true;
      $('[data-action="export-archive"]').disabled = true;
      showPage('workbench'); showView('edit');
      toast('AI 已生成待确认选区与文化观察；请核对后再生成修复候选。');
      await refresh();
      return;
    }
    if (result.items) {
      state.autoBusy = false;
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

  function renderCulturalContext(context) {
    const panel = $('#culture-summary');
    if (!panel) return;
    const lines = [];
    if (context.visible_subject) lines.push(`<p><b>可见题材</b> ${escapeHtml(context.visible_subject)}</p>`);
    if (context.possible_period_or_tradition) lines.push(`<p><b>可能年代 / 传统</b> ${escapeHtml(context.possible_period_or_tradition)} <small>待核实</small></p>`);
    if (context.visual_clues?.length) lines.push(`<p><b>图像线索</b> ${escapeHtml(context.visual_clues.join('；'))}</p>`);
    if (context.preservation_constraints?.length) lines.push(`<p><b>保护约束</b> ${escapeHtml(context.preservation_constraints.join('；'))}</p>`);
    if (context.verification_needed?.length) lines.push(`<p><b>待查来源</b> ${escapeHtml(context.verification_needed.join('；'))}</p>`);
    panel.innerHTML = `<strong>文化背景观察 · 未经考证</strong>${lines.join('') || '<p>AI 未能可靠判断，请结合馆藏记录或专家意见核实。</p>'}<small>修复规划会参考这些观察线索，但不会将模型猜测当成史实。</small>`;
    panel.classList.remove('hidden');
  }

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
      list.innerHTML = '<div class="empty-inline">分析后逐项显示缺损和修复建议</div>';
      $('#damage-note').textContent = 'AI 识别、类别与掩膜提取都只是候选，需人工检查；不能替代病害鉴定。' + (limitations ? ' ' + limitations : '');
      $('#damage-image').classList.add('hidden'); $('#damage-empty').classList.remove('hidden');
      $('[data-action="run-all-defects"]').disabled = true;
      return;
    }
    const jobs = state.lastSnapshot?.jobs || [];
    list.innerHTML = state.damageItems.map((item, index) => {
      const [x0, y0, x1, y1] = item.bbox || [0, 0, 0, 0];
      const extent = Math.max(0, (x1 - x0) * (y1 - y0)) / 100;
      const job = jobs.find(record => record.defect_proposal_id === item.proposal_id);
      const status = job ? `任务${['review', 'completed'].includes(job.status) ? '已完成' : job.status === 'failed' ? '失败' : '进行中'}` : '';
      const method = item.recommendation?.method || damageRecommendation(item);
      const selectedMethod = item.selectedMethod || 'recommended';
      const blocked = !item.proposal_id || !item.repair_allowed || item.region_class !== 'repairable';
      const alreadyQueued = job && ACTIVE.includes(job.status);
      const busy = state.running || state.autoBusy;
      const blocker = item.risk_notice || (blocked ? '请先人工复核风险分类；当前不能一键修复。' : '');
      return `<div class="damage-item ${item === state.selectedDamage ? 'selected' : ''}" data-damage-row="${index}">` +
        `<div class="damage-item-main"><button class="damage-select" data-damage="${index}">${String(index + 1).padStart(2, '0')} ${escapeHtml(DAMAGE_LABELS[item.type] || '其他疑似缺损')}</button>` +
        `<small>${escapeHtml(item.size || 'uncertain')} · 框选约 ${extent.toFixed(2)}% · 视觉置信 ${Math.round((item.confidence || 0) * 100)}%</small>` +
        `<small>AI 建议：${escapeHtml(method)}</small><label class="damage-method">修补方案 <select data-method-defect="${index}" ${blocked || alreadyQueued || busy ? 'disabled' : ''}>` +
        `<option value="recommended" ${selectedMethod === 'recommended' ? 'selected' : ''}>采用 AI 建议</option>` +
        `<option value="lama" ${selectedMethod === 'lama' ? 'selected' : ''} ${state.lastSnapshot?.models?.lama ? '' : 'disabled'}>LaMa AI 纹理补全${state.lastSnapshot?.models?.lama ? '' : '（未安装）'}</option>` +
        `<option value="telea" ${selectedMethod === 'telea' ? 'selected' : ''}>Telea 局部纹理传播</option>` +
        `<option value="ns" ${selectedMethod === 'ns' ? 'selected' : ''}>Navier–Stokes 细线修补</option>` +
        `<option value="color" ${selectedMethod === 'color' ? 'selected' : ''}>局部色调匹配</option></select></label>` +
        `${blocker ? `<small>${escapeHtml(blocker)}</small>` : ''}${status ? `<small>${status}</small>` : ''}</div>` +
        `<div class="damage-item-actions"><label class="damage-confirm"><input type="checkbox" data-confirm-defect="${index}" ${item.confirmed ? 'checked' : ''} ${blocked || alreadyQueued || busy ? 'disabled' : ''}>确认</label>` +
        `<button class="button" data-run-defect="${index}" ${blocked || alreadyQueued || busy ? 'disabled' : ''}>运行修复</button></div></div>`;
    }).join('');
    $$('[data-damage]').forEach(button => button.addEventListener('click', () => selectDamage(Number(button.dataset.damage))));
    $$('[data-confirm-defect]').forEach(input => input.addEventListener('change', () => {
      state.damageItems[Number(input.dataset.confirmDefect)].confirmed = input.checked;
      renderDamage(limitations);
    }));
    $$('[data-method-defect]').forEach(select => select.addEventListener('change', () => {
      state.damageItems[Number(select.dataset.methodDefect)].selectedMethod = select.value;
    }));
    $$('[data-run-defect]').forEach(button => button.addEventListener('click', () => runDefectRepairs([Number(button.dataset.runDefect)]).catch(notifyError)));
    $('#damage-note').textContent = (limitations || '') || 'AI 识别、类别与掩膜提取都只是候选，需人工检查；不能替代病害鉴定。';
    if (state.selectedDamage === null) selectDamage(0);
    $('[data-action="run-all-defects"]').disabled = state.running || state.autoBusy || !state.damageItems.some(item => item.confirmed && item.proposal_id && item.repair_allowed && item.region_class === 'repairable' && !jobs.some(job => job.defect_proposal_id === item.proposal_id && ACTIVE.includes(job.status)));
  }

  async function runDefectRepairs(indexes) {
    if (state.running || state.autoBusy) return toast('请等待当前分析或修复任务完成。');
    const selected = indexes.map(index => state.damageItems[index]).filter(item => item && item.confirmed && item.proposal_id && item.repair_allowed && item.region_class === 'repairable');
    if (!selected.length) return toast('请先确认至少一个可保守修复的缺损项。');
    const result = await invoke('runDefectRepairs', JSON.stringify({
      image_id: state.image.id, proposal_ids: selected.map(item => item.proposal_id), confirmed: true,
      methods: Object.fromEntries(selected.filter(item => item.selectedMethod && item.selectedMethod !== 'recommended')
        .map(item => [item.proposal_id, item.selectedMethod])),
    }));
    const count = result.jobs?.length || 0;
    if (!count) {
      const reason = result.skipped?.[0]?.reason || '当前没有可运行的缺损项。';
      toast(reason);
      return;
    }
    const skipped = result.skipped?.length ? `；跳过 ${result.skipped.length} 项：${result.skipped[0].reason}` : '';
    toast(`已启动 ${count} 个修复任务${skipped}。`);
    await refresh();
    renderDamage();
  }

  async function selectDamage(index) {
    const item = state.damageItems[index];
    if (!item || !state.image) return;
    state.selectedDamage = item;
    $$('.damage-item').forEach((button, position) => button.classList.toggle('selected', position === index));
    const [x0, y0, x1, y1] = item.bbox || [0, 0, 0, 0];
    const fraction = Math.max(0, (x1 - x0) * (y1 - y0)) / 100;
    $('#damage-note').textContent = `疑似${DAMAGE_LABELS[item.type] || '缺损'}，AI 估计 ${item.size}，框选约 ${fraction.toFixed(2)}% 图像面积；建议：${damageRecommendation(item)}。视觉置信 ${Math.round((item.confidence || 0) * 100)}% 不是病害概率。依据：${item.evidence || '无文字依据'}。请先检查红框是否落在真实损伤上。`;
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
          if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再新建任务。');
          await loadImage(null); state.image = null; $('#empty-state').classList.remove('hidden'); imageWrap.classList.add('hidden'); break;
        case 'open-image': await openImage(); break;
        case 'load-demo': { if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再载入图像。'); const record = await invoke('loadDemo'); await loadImage(record); break; }
        case 'load-competition-demo': {
          if (state.running || state.autoBusy) return toast('任务或 AI 分析进行中，完成后再载入图像。');
          const pack = await invoke('loadCompetitionDemo');
          await loadImage(pack);
          if (pack?.observe) {
            pack.observe.gate = pack.gate || pack.observe.gate;
            pack.observe.export_allowed = pack.export_allowed || pack.observe.export_allowed;
            pack.observe.banner = pack.banner || pack.observe.banner;
          }
          renderObservePack(pack?.observe || null);
          if (pack?.gate_review || pack?.gate) {
            applyGatePayload({
              gate: pack.gate,
              export_allowed: pack.export_allowed,
              banner: pack.banner,
              gate_review: pack.gate_review,
              candidates: pack.gate_review?.candidates,
            });
            state.gateReview = {
              ...(state.gateReview || {}),
              gate: pack.gate,
              export_allowed: pack.export_allowed,
              banner: pack.banner,
              gate_review: pack.gate_review,
              candidates: pack.gate_review?.candidates,
            };
            syncGateExportButtons();
          }
          toast('已载入比赛观察包：原图 + 三色风险标注；门禁待人工复核（默认阻断导出）。');
          break;
        }
        case 'undo': {
          if (state.currentStroke) break;
          const last = state.strokeOrder.pop();
          if (last === 'region') state.regionStrokes.pop();
          else if (last === 'mask') state.maskStrokes.pop();
          if (last) redrawMask();
          break;
        }
        case 'clear-mask': state.maskStrokes = []; state.regionStrokes = []; state.strokeOrder = []; state.currentStroke = null; state.baseMask = null; redrawMask(); break;
        case 'import-mask': { const result = await invoke('importMask'); if (result?.mask_data) await showMaskFromBase64(result.mask_data); break; }
        case 'zoom-in': state.zoom = Math.min(3, state.zoom * 1.15); updateCanvasSize(); break;
        case 'zoom-out': state.zoom = Math.max(.5, state.zoom / 1.15); updateCanvasSize(); break;
        case 'zoom-fit': state.zoom = 1; updateCanvasSize(); break;
        case 'run-repair': await createJob(); break;
        case 'auto-restore': {
          if (!state.image) return toast('请先打开一幅壁画。');
          if (state.running || state.autoBusy) return toast('已有任务或 AI 分析在执行，请稍候。');
          state.autoBusy = true;
          state.job = null; state.candidate = null;
          $('#candidate-strip').innerHTML = '<div class="empty-inline">AI 分析完成后显示修复候选</div>';
          $('#comparison-stage').innerHTML = '<div class="empty-inline">AI 分析进行中</div>';
          $('#candidate-count').textContent = '0';
          $('[data-action="save-result"]').disabled = true;
          $('[data-action="select-candidate"]').disabled = true;
          $('[data-action="export-archive"]').disabled = true;
          $('[data-action="run-repair"]').disabled = true;
          $('[data-action="auto-restore"]').disabled = true;
          renderDamage();
          setAutoStatus('AI 正在识别原图损伤、整理文化线索并生成待确认选区…');
          try {
            const payload = JSON.stringify({ image_id: state.image.id, workflow: 'professional_step' });
            await invoke('autoRestore', payload);
            toast('AI 正在分析，完成后会显示逐项缺损清单。');
          } catch (error) {
            state.autoBusy = false;
            setAutoStatus('AI 缺损分析失败：' + (error.message || error));
            notifyError(error);
            await refresh();
          }
          break;
        }
        case 'run-all-defects': await runDefectRepairs(state.damageItems.map((_, index) => index)); break;
        case 'cancel-job': { const identifier = state.activeJobId || state.job?.id; if (identifier) { await invoke('cancelJob', identifier); } await refresh(); break; }
        case 'resume-job': if (state.job) await invoke('resumeJob', state.job.id); await refresh(); break;
        case 'select-candidate': {
          if (isObserveMode() && state.image) {
            const decision = selectedDecision();
            if (!decision) return toast('请选择：采纳 / 修改后采纳 / 驳回 / 拒答。');
            const checklist = readGateChecklist();
            const payload = await invoke('applyGateReview', state.image.id, JSON.stringify({
              decision,
              note: $('#review-note').value,
              checklist,
              confirm: true,
            }));
            applyGatePayload(payload);
            toast(payload?.gate?.blocked
              ? ('门禁仍阻断：' + ((payload.gate.block_reasons || []).slice(0, 2).join('；') || '请检查勾选与决策'))
              : '复核已确认，可导出标注层 + 复核记录（禁止修复成图）。');
            break;
          }
          if (state.job && state.candidate) {
          const decision = selectedDecision();
          if (isProfessional() && !decision) return toast(DEMO_COPY.B.gate);
          const notePayload = JSON.stringify({
            note: $('#review-note').value,
            archive_as_hypothesis: Boolean($('#archive-hypothesis')?.checked),
            decision: decision || null,
          });
          state.job = await invoke('selectCandidate', state.job.id, state.candidate.id, notePayload);
          toast(state.job?.selected_as === 'rejected' ? '已驳回该候选，不进入下一档。'
            : state.job?.selected_as === 'hypothesis' ? '已按假设性建议归档。'
            : state.job?.selected_as === 'enhanced' ? '已选定加强修复结果（非历史定稿）。'
            : state.job?.selected_as === 'modified' ? '已按修改后结果采纳。'
            : '候选选择已保存。');
          renderComparison(); await refresh();
          }
        } break;
        case 'export-overlay': {
          if (!state.image) return toast('请先打开一幅壁画。');
          if (isObserveMode()) {
            const gate = gateState();
            const exp = gateExport();
            if (gate.blocked || !exp.annotation_layer) {
              return toast('门禁未放行：不能导出标注层。请先完成三色核对与复核决定。');
            }
            if (exp.restored_image) return toast(DEMO_COPY.B.refuse);
            try {
              await invoke('exportObserveAnnotation', state.image.id);
            } catch (error) { notifyError(error); }
            break;
          }
          const out = document.createElement('canvas');
          out.width = canvas.width; out.height = canvas.height;
          const octx = out.getContext('2d');
          octx.drawImage(image, 0, 0, out.width, out.height);
          octx.drawImage(canvas, 0, 0);
          if (regionCanvas) octx.drawImage(regionCanvas, 0, 0);
          octx.fillStyle = 'rgba(0,0,0,0.55)';
          octx.fillRect(0, out.height - 36, out.width, 36);
          octx.fillStyle = '#fff'; octx.font = '14px sans-serif';
          octx.fillText('琥珀=可修露底 · 红=不确定 · 深红=不可复原｜结果须人工复核', 12, out.height - 14);
          const link = document.createElement('a');
          link.download = `mural-overlay-${Date.now()}.png`;
          link.href = out.toDataURL('image/png');
          link.click();
          toast('已导出标红对照图。');
          break;
        }
        case 'save-result': if (state.candidate) {
          try { await invoke('saveResult', state.candidate.id); } catch (error) { notifyError(error); }
        } break;
        case 'export-archive': {
          if (isObserveMode() && state.image) {
            const gate = gateState();
            const exp = gateExport();
            if (gate.blocked || !exp.review_record) {
              return toast('门禁未放行：不能导出复核档案。');
            }
            if (exp.restored_image) return toast(DEMO_COPY.B.refuse);
            try { await invoke('exportObserveReview', state.image.id); } catch (error) { notifyError(error); }
            break;
          }
          if (state.job) await invoke('exportArchive', state.job.id);
          break;
        }
        case 'load-gate-reviewed-sample': {
          if (!isObserveMode() || !state.image) return toast('请先载入比赛观察包。');
          try {
            const payload = await invoke('loadGateSample', 'sample-from-shibi-reviewed');
            // Sample already has decisions; mark checklist checked for demo click-through.
            if ($('#check-repairable')) $('#check-repairable').checked = true;
            if ($('#check-uncertainty')) $('#check-uncertainty').checked = true;
            if ($('#check-unrestorable')) $('#check-unrestorable').checked = true;
            if ($('#check-literature')) $('#check-literature').checked = true;
            applyGatePayload({
              ...payload,
              checklist: {
                repairable_checked: true,
                uncertainty_checked: true,
                unrestorable_checked: true,
                literature_checked: true,
              },
            });
            toast('演示样例：已复核可导出（标注层+复核记录；restored_image=false）。');
          } catch (error) { notifyError(error); }
          break;
        }
        case 'open-settings': openSettings(); break;
        case 'close-settings': hideModal('settings-modal'); break;
        case 'save-settings': await saveSettings(); break;
        case 'open-reference': showModal('reference-modal'); $('#ref-error').textContent = ''; break;
        case 'close-reference': hideModal('reference-modal'); break;
        case 'save-reference': await saveReferenceForm(); break;
        case 'open-verify': if (state.registration) { showModal('verify-modal'); $('#verify-error').textContent = ''; } break;
        case 'close-verify': hideModal('verify-modal'); break;
        case 'save-verify': await saveVerifyNote(); break;
        case 'propose-damage': if (state.image) await invoke('proposeDamage', state.image.id); showPage('tools'); break;
        case 'sam-point': setTool('sam-point'); state.promptPoints = []; state.promptLabels = []; showPage('workbench'); showView('edit'); toast('左键添加前景点，右键添加背景点。'); break;
        case 'sam-box': setTool('sam-box'); state.promptBox = null; showPage('workbench'); showView('edit'); toast('在图像上拖动框选目标。'); break;
        case 'run-sam': if (state.image) await invoke('runSam', state.image.id, JSON.stringify(state.promptPoints), JSON.stringify(state.promptLabels), JSON.stringify(state.promptBox)); break;
        case 'clear-prompts': state.promptPoints = []; state.promptLabels = []; state.promptBox = null; setTool('brush'); toast('提示已清除。'); break;
        case 'use-proposal': if (state.proposal) { await invoke('adoptProposal', state.proposal.id); await installMask(state.proposal.mask_file_url); state.proposalIds = [state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
        case 'merge-proposal': if (state.proposal) { const current = exportMaskBase64(); state.proposal = await invoke('mergeProposal', state.proposal.id, current); await installMask(state.proposal.mask_file_url); state.proposalIds = [...(state.proposalIds || []), state.proposal.id]; showPage('workbench'); showView('edit'); setTool('brush'); } break;
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
    else if (event.target.value === 'speculative' || event.target.value === 'enhanced') {
      backend.value = state.lastSnapshot?.models?.diffusion ? 'diffusion' : (state.lastSnapshot?.models?.lama ? 'lama' : 'opencv');
    }
    else backend.value = state.lastSnapshot?.models?.lama ? 'lama' : 'opencv';
    $('#controlnet').disabled = event.target.value !== 'speculative' || !state.lastSnapshot?.models?.controlnet;
    if (event.target.value === 'speculative' && $('#evidence-level')) $('#evidence-level').value = 'speculative';
    if (event.target.value === 'enhanced' && $('#evidence-level')) $('#evidence-level').value = 'enhanced';
    if (event.target.value === 'conservative' && $('#evidence-level')) $('#evidence-level').value = 'conservative';
  });
  $('#professional-step')?.addEventListener('change', syncProfessionalUi);
  $('#repair-mode')?.addEventListener('change', syncProfessionalUi);
  $('#evidence-level')?.addEventListener('change', syncProfessionalUi);
  $('#force-override')?.addEventListener('change', syncProfessionalUi);
  syncProfessionalUi();
  $('#professional-step').checked = true;
  $('#planner').addEventListener('change', event => {
    $('#planner-note').textContent = event.target.value === 'api' ? '会向已配置服务发送图像缩略图、掩膜预览和备注。' : event.target.value === 'local' ? '本地 Qwen3-VL 分析图像，不发送数据。' : '本地规则不会发送图像。';
  });
  $('#comparison-mode').addEventListener('change', renderComparison);
  $('#history-filter').addEventListener('input', () => renderHistory());
  $('#source-image').addEventListener('load', updateCanvasSize);
  new ResizeObserver(updateCanvasSize).observe($('#image-stage'));
  new ResizeObserver(() => layoutComparison()).observe($('#comparison-stage'));

  canvas.addEventListener('pointerdown', event => {
    if (!state.image || !image.naturalWidth || state.running) return;
    const point = pointAt(event); canvas.setPointerCapture(event.pointerId);
    if (state.tool === 'sam-point') {
      state.promptPoints.push(point); state.promptLabels.push(event.button === 2 ? 0 : 1); toast('已记录分割提示点。'); return;
    }
    if (state.tool === 'sam-box') { state.promptBox = [point[0], point[1], point[0], point[1]]; return; }
    if (event.button !== 0) return;
    state.currentStroke = { tool: state.tool, size: state.brushSize * canvas.width / canvas.clientWidth, points: [point] };
    $('#mask-gate-confirmed').checked = false;
    paintStroke(state.currentStroke); state.maskDirty = true; setBusy(state.running);
  });
  canvas.addEventListener('pointermove', event => {
    if (state.currentStroke) {
      state.currentStroke.points.push(pointAt(event));
      redrawMask();
    }
    if (state.tool === 'sam-box' && state.promptBox && event.buttons) state.promptBox.splice(2, 2, ...pointAt(event));
  });
  canvas.addEventListener('pointerup', () => {
    if (!state.currentStroke) return;
    if (isRegionTool(state.currentStroke.tool)) {
      state.regionStrokes.push(state.currentStroke); state.strokeOrder.push('region');
    } else {
      state.maskStrokes.push(state.currentStroke); state.strokeOrder.push('mask');
    }
    state.currentStroke = null;
    redrawMask();
  });
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
  window.__shibui = { state, loadImage, installMask, showMaskFromBase64, exportMaskBase64, refresh, renderDamage,
    renderComparison, layoutComparison };
})();
