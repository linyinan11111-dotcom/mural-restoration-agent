/* Shared, DOM-free rules for risk-mark classification and defect-batch focus.
   The desktop page and the regression tests both use this file so the color
   and batch decisions cannot drift back into the ambiguous inline checks. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.MuralLogic = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const REGION_TOOLS = { uncertain: true, uncertainty: true, unrestorable: true };

  // Visible stroke colors, straight alpha as Qt WebEngine reported them:
  //   uncertain    rgba(220,0,60,0.72) → [220, 0, 60, 184]
  //   unrestorable rgba(180,0,40,0.85) → [180, 0, 40, 217]
  // The previous test (r>150 && g<50 && r-g>120) matched both reds, so a
  // normal uncertain stroke was reported as unrestorable.
  function classifyRegionPixel(r, g, b, a) {
    if (a < 40) return null;
    if (r >= 200 && g < 90 && (r - g) > 100) return 'uncertain';
    if (r > 140 && r < 200 && g < 80 && b < 90 && (r - g) > 100) return 'unrestorable';
    return null;
  }

  function aggregateRegionPixels(data) {
    const stats = { uncertain: false, unrestorable: false };
    for (let i = 0; i < data.length; i += 4) {
      const kind = classifyRegionPixel(data[i], data[i + 1], data[i + 2], data[i + 3]);
      if (kind) stats[kind] = true;
    }
    return stats;
  }

  // Explicit category buffer: uncertain is opaque red, unrestorable is opaque
  // blue. These channels do not overlap, including on antialiased edges that
  // stay above the alpha cutoff.
  function classifyCategorySample(r, g, b, a) {
    if (a < 128) return null;
    if (b > 200 && r < 80) return 'unrestorable';
    if (r > 200 && b < 80) return 'uncertain';
    return null;
  }

  function aggregateCategory(data) {
    const stats = { uncertain: false, unrestorable: false };
    for (let i = 0; i < data.length; i += 4) {
      const kind = classifyCategorySample(data[i], data[i + 1], data[i + 2], data[i + 3]);
      if (kind) stats[kind] = true;
    }
    return stats;
  }

  function orderedStrokes(strokeOrder, regionStrokes, maskStrokes) {
    const ordered = [];
    let regionIndex = 0;
    let maskIndex = 0;
    const regions = regionStrokes || [];
    const masks = maskStrokes || [];
    for (const kind of strokeOrder || []) {
      if (kind === 'region' && regionIndex < regions.length) ordered.push(regions[regionIndex++]);
      else if (kind === 'mask' && maskIndex < masks.length) ordered.push(masks[maskIndex++]);
    }
    return ordered;
  }

  // Risk marks and the eraser share one chronological history. Replaying only
  // the region strokes paints the red marks back on top of every erase.
  function regionEdits(strokeOrder, regionStrokes, maskStrokes, currentStroke) {
    const edits = [];
    const push = (stroke) => {
      if (!stroke || !stroke.points || !stroke.points.length) return;
      if (REGION_TOOLS[stroke.tool]) edits.push({ op: 'paint', stroke });
      else if (stroke.tool === 'eraser') edits.push({ op: 'erase', stroke });
    };
    orderedStrokes(strokeOrder, regionStrokes, maskStrokes).forEach(push);
    push(currentStroke);
    return edits;
  }

  // "纳入批次" filters 运行全部. Omitted/true means included; only an explicit
  // false opts an item out. Per-item 运行修复 does not use this list.
  function batchRunIndexes(items) {
    const indexes = [];
    (items || []).forEach((item, index) => {
      if (item && item.includeInBatch !== false) indexes.push(index);
    });
    return indexes;
  }

  function focusCreatedJobs(jobs) {
    const created = (jobs || []).filter(job => job && job.id);
    if (!created.length) return null;
    return {
      jobId: created[0].id,
      batchJobIds: created.map(job => job.id),
      showQueue: created.length > 1,
    };
  }

  return {
    REGION_TOOLS,
    classifyRegionPixel,
    aggregateRegionPixels,
    classifyCategorySample,
    aggregateCategory,
    orderedStrokes,
    regionEdits,
    batchRunIndexes,
    focusCreatedJobs,
  };
});
