# CHANGELOG

## 2026-11-09 — 三分屏对照与全图 AI 生成

- 新增“三分屏对照”：原图 / AI 修补大面积 / 全 AI 生成三栏弹窗，可分别生成并反复查看。
- 新增 `generateFullImage`：自动创建全图掩膜，用原图边缘做 ControlNet 引导执行全图扩散生成；结果标记为 speculative。
- 新增 `generateLargeRepair`：把当前缺损候选掩膜取并集后走实验扩散，生成“大面积修补”栏。
- 全图/大面积生成均需 SD 权重，结果只能按假设归档；全图生成为全新画面，不是历史复原。
- 用真实 SD 权重跑通全图生成：任务 `review`、候选 `speculative`、掩膜外差异 0；三栏 UI 回归通过，测试 70 项通过。

## 2026-11-09 — AI 尝试性补全（实验）

- 高级设置新增「允许 AI 尝试性补全大面积缺损」：自动切到 speculative + diffusion，允许大面积及 uncertain/unrestorable 掩膜；结果必须按假设归档，不能当定稿。
- 缺损清单新增「AI 生成式补全（实验）」：单项/全部可直接用扩散生成，无掩膜项自动补矩形掩膜；同样按假设归档。
- 增加 `allow_generative_attempt` 服务端校验：显式开关可越过风险分类，但保留掩膜核对、风险提示和假设归档要求；任务记录原风险与开关。
- 用真实 SD 权重端到端跑通一次缺陷生成：任务 `review`、候选 `speculative`、掩膜外差异 0；测试 70 项通过。
- 文档补充实验路径、限制与操作方式。

## 2026-11-09 — 缺损清单弹窗、手动添加、取消缺陷修复风险限制

- 缺损清单改为弹窗：AI 分析后自动弹出，图像上方与右侧均有“缺损清单”按钮可反复打开。
- 点击清单条目会在主图上用红框标出对应区域；没有可用掩膜的条目运行时自动按框生成矩形掩膜。
- 支持“手动添加缺损区域”：在图像上拖框即可加入清单，并立即参与修复。
- 缺损清单修复路径**取消风险分类与确认限制**：`uncertain` / `unrestorable` 也会尝试修复，原风险记录保存在任务 `defect_risk_class` 中；LaMa 未安装时自动回退 Telea。
- 保留手动画布/专业分步的门禁与导出校验，缺损清单是明确的“全部可试修”入口。
- 新增 `addManualDefect` 桥接与回归测试；测试 69 项通过。
- 实际点击验收中修复：弹窗宽度被通用 `.modal-card` 样式覆盖成 440px；任务结束后清单按钮禁用状态不刷新；批量运行中个别无掩膜项失败会中止整批（现在逐项跳过）；弹窗字号过小/“修补方案”标签换行。

## 2026-11-09 — 取消比赛观察模式

- 移除比赛观察包入口与观察预览区、三色观察清单、门禁横幅、演示样例按钮和导出观察标注/复核记录的快捷操作。
- 删除 `human_gate.py`、`scripts/run_demo_competition_risk.py` 以及比赛 demo 的 bridge 方法（`loadCompetitionDemo` / `applyGateReview` / `loadGateSample` / `exportObserveAnnotation` / `exportObserveReview`）和 `/api/competition-demo` 端点。
- 移除 `contest_demo` 字段与比赛导出专用门禁；保留专业分步的风险门禁、复核决定与导出校验。
- 复核决定（采纳 / 修改后采纳 / 驳回）保留在候选对照页，专业分步流程不受影响。
- 全量测试 67 项通过（较取消前少一项比赛专用测试）；文档同步更新。

## 2026-11-09 — 门禁、导出与验证基线修复（P0）

### 风险与门禁

- **统一风险枚举**：`fillable_lacuna / uncertain / unrestorable` 成为唯一规范词表，旧名 `repairable / uncertainty` 自动归一；`risk_regions`、`professional`、UI 与文档共用同一套常量。
- **服务端推导风险标志**：`derive_risk_flags(mask_record)` 从冻结掩膜记录推导 `has_uncertain / has_unrestorable`，创建工作区任务时不再信任前端传入的布尔值。
- **修复两条门禁绕过**（已用脚本复现并验证修复）：
  - professional_step 中 `region_class='uncertainty'` 且未传 `has_uncertain` 时曾可直接建任务；现在会被拒绝并要求专家强制放行 + 理由。
  - `workflow='standard'` 曾可整体绕过 enhanced/speculative/risk 门禁；现在两种工作流都在建任务时和 `engine.run` 执行前双重校验，`enhanced` 必须显式勾选掩膜核对 / 允许加强 / 风险知悉，`speculative` 必须显式确认风险。
  - `unrestorable` 区域任何情况下禁止像素修复。
- **比赛观察导出曾加门禁**（该模式已于同日取消，见上方条目）。

### 导出与保存

- `validate_selected_export()` 统一校验复核决定：驳回的候选不能导出 ZIP；假设性结果只能导出档案（`manifest.selected_as = hypothesis`），不能保存为普通修复 PNG。
- `engine.export` 与桌面端 `saveResult` 都使用该校验；manifest 新增 `selected_as / review` 字段。

### 合成与候选

- **修复羽化削弱薄掩膜**：`composite()` 对 <4 px 的薄掩膜直接写回完整预测；较厚掩膜只在 1–3 px 边界环带过渡、内部保持完整预测；掩膜外像素仍保证 0 差异。
- **候选数量不再静默减少**：优先使用不同算法，可用方法不足时用不同半径补足用户请求的 1–3 个候选；移除 engine 中“重复方法即提前结束”的分支。

### 测试与验证

- 更新 15 个过期测试，新增 H1/H2 门禁绕过回归、导出决定校验、薄/厚掩膜合成测试；当时 `pytest -q`：**68 passed**（取消比赛模式后为 67）。
- 新增 GitHub Actions：非 Web 测试 + `node --check` + `compileall`。
- 发货 WebEngine 外壳验收重跑通过：LaMa + CUDA、掩膜外差异 0、退出码 0。

### 文档与交付

- 重写 `VALIDATION.md` 为当前实测基线；修正使用说明的重复章节编号、过期测试数量、风险/加强表述与内部引用。
- 新增 MIT `LICENSE`（代码；模型权重与示例图各自许可）；新增本 CHANGELOG。
- 明确本机 `dist/Shibi-1.0` 为旧构建。

## 2026-10-31 — 早期迭代

历史记录保留在 git 提交中（`fe47956` baseline → `9bda39c` 稳定版 → `44d05c3` 专业流程 → `7fae2cf` 移除旧 UI）。
