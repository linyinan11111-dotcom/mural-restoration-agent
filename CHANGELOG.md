# CHANGELOG

## 2026-10-31 优化（发货前端修复 + 验收对齐）

### 修复：发货 WebEngine 前端的阻断性缺陷

- **灰度掩膜导入/恢复会把整图变成可编辑区**（`ui/app.js`）：后端掩膜是 8-bit 灰度 PNG，旧实现直接 `drawImage` 后用 alpha 判断可编辑像素，等于全图选中，导致导入掩膜、采用辅助选区、合并选区、历史恢复全部被后端拒绝。现在先按 `alpha > 127 且亮度 > 127` 重建 alpha 层再绘制。新增“灰度掩膜往返”回归测试。
- **选区合并必然 `NameError`**（`mural_agent/desktop_web.py`）：`mergeProposal` 使用 `Image.fromarray` 但模块顶部未导入 `PIL.Image`。已补导入并加回归测试。
- **参考导入/核验依赖 `window.prompt`**（`ui/app.js`、`ui/index.html`）：QWebEngine 默认不提供输入对话框，流程会静默失败或阻塞。改为 HTML 模态框（来源/许可/分组/对应关系，以及核验说明）。
- **辅助结果后每 0.9 s 被强制拉回工具页**：`Bridge.aux` 永不清空且前端每次刷新都重新渲染/跳页。新增单调递增的 `aux_version`，前端只在版本变化时消费一次结果。
- **历史恢复时掩膜可能被图片 `onload` 清掉**：`loadImage` 改为返回 Promise，历史恢复先等图像加载再铺掩膜；同时恢复复核意见、清空上一幅图的 registration/references/proposal 状态。
- **新建任务后旧候选卡残留**：创建任务/换图时清空候选条与对照区。
- **任务运行中仍可换图/恢复历史**：运行期禁用打开图像、载入练习图与历史任务，取消按钮改为基于全局活动任务。
- **关闭任务窗口无法可靠退出**：取消中会提示；再次点击关闭可强制退出（已完成尝试均已落库，重启后可续跑）。
- **复核意见被空输入覆盖**：`selectCandidate` 在输入为空时保留已有复核意见。
- **删除 Ctrl+O 失效**：补回 `Ctrl+O` 打开图像；README 中“滚动条平移”的不实描述已修正。
- **AI 缺损识别只接在旧 Qt Widgets 页**：已移植到发货 WebEngine 前端（“工具/评测”页：分析原图缺损 → 选中条目 → 细化选中区域），旧页面与发货页面共用 `damage_assessment.recommendation_for` 的修复建议映射。

### 修复：算法与存储

- **ControlNet 结构控制必定失败**：`model_worker.py` 加载 ControlNet 时缺少 `variant='fp16'` / `use_safetensors=True`，与仓库中只有 `*.fp16.safetensors` 的布局不匹配。已补参数。
- **细裂缝在推理缩小时可能整体消失**：LaMa 与 SD 路径现在按缩放比例膨胀推理掩膜（合成仍使用原始冻结掩膜），并在掩膜缩空时明确报错，而不是静默产出“无修改”候选。
- **SQLite 增加 WAL、kind 索引，事件保留最近 200 条**：降低 900 ms 快照轮询与后台写入的阻塞，避免任务事件无限增长。
- **断点续跑竞态**：`Engine.resume` 在任务状态已终态但工作线程尚未释放槽位时短暂等待，而不是误报“任务尚未停止”。

### 测试

- 新增 `tests/test_webui.py`：4 项发货壳回归（页面加载与桥接快照、灰度掩膜往返、桥接建任务→复核→ZIP 导出、`mergeProposal` 与 `aux_version` 契约）。
- 新增 `tests/test_damage_assessment.py`：缺损解析（完整/截断/非法候选）、相邻合并、重复框去重、并行重叠裂缝不误并、修复建议路由。
- `tests/conftest.py` 统一离屏 Qt 设置；`test_desktop.py` 隔离全部模型路径并断言旧界面默认规划器为 `rules`（此前会因本机装有 Qwen3-VL 权重而卡在 planning 超时）。
- 当前 `pytest -q`：**31 passed**（约 8 秒）。
- 真实 Qwen3-VL 缺损识别经桥接冒烟通过：合成纹样练习图，含模型加载约 41 秒，8 条原始候选去重/合并后剩 3 条并生成红框预览（`artifacts/damage-smoke/`）。测试同时暴露并修复了 VLM 重复候选与并行裂缝被误合并的问题。

### 验收

- `scripts/verify_desktop.py` 重写为**发货 WebEngine 外壳**验收：离屏加载 `ui/index.html`，通过 QWebChannel 桥完成真实 LaMa CUDA 修复，并断言 `tool=lama`、`device=cuda`、`outside_max_difference=0`；报告 `artifacts/desktop-web-verification.json`，预览 `artifacts/desktop-web-preview.png`。
- README/VALIDATION 与实现对齐：文件表、快捷键、测试入口、权重体积（8.34 GB）、provenance 缺口、便携包需重建、旧 Widgets 界面的定位。

### 文档与仓库

- 新增本 CHANGELOG 与基线/修复两次 git 提交；`.gitignore` 保持对模型、数据、验收产物和 `.env` 的忽略。
- `requirements.txt` 收敛为桌面运行依赖（新增 `scikit-image`），FastAPI/uvicorn 测试适配器依赖移入 `requirements-dev.txt`。

### 已知未完成（下一步）

- 真实 Qwen3-VL 缺损识别推理验收（耗时、误检/漏检、建议合理性）；SD/ControlNet 实际生成验收；参考配准跨图重建质量验证。
- SD 路径仍是整图 512px、无 ROI；LPIPS 每次重建网络且跑 CPU。
- 数据集分组划分尚未接入 `benchmark.py`；配对指标区域口径（mask vs bbox+16）待统一。
- 旧 Qt Widgets 界面（`desktop.py` / `desktop_features.py`）仍保留用于画布回归，尚未删除；`scripts/verify_system.py` 仍基于旧界面。
- 领域验证（专家盲评、固定分组评测集、领域模型对照）仍缺失。
