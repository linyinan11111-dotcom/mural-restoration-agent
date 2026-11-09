# CHANGELOG

## 2026-11-09 — 门禁、导出与验证基线修复（P0）

### 风险与门禁

- **统一风险枚举**：`fillable_lacuna / uncertain / unrestorable` 成为唯一规范词表，旧名 `repairable / uncertainty` 自动归一；`risk_regions`、`professional`、UI 与文档共用同一套常量。
- **服务端推导风险标志**：`derive_risk_flags(mask_record)` 从冻结掩膜记录推导 `has_uncertain / has_unrestorable`，创建工作区任务时不再信任前端传入的布尔值。
- **修复两条门禁绕过**（已用脚本复现并验证修复）：
  - professional_step 中 `region_class='uncertainty'` 且未传 `has_uncertain` 时曾可直接建任务；现在会被拒绝并要求专家强制放行 + 理由。
  - `workflow='standard'` 曾可整体绕过 enhanced/speculative/risk 门禁；现在两种工作流都在建任务时和 `engine.run` 执行前双重校验，`enhanced` 必须显式勾选掩膜核对 / 允许加强 / 风险知悉，`speculative` 必须显式确认风险。
  - `unrestorable` 区域任何情况下禁止像素修复。
- **恢复比赛观察门禁**：比赛模式不允许导出 enhanced/speculative/hypothesis/rejected 的修复成图，只能导出标注层与复核记录。

### 导出与保存

- `validate_selected_export()` 统一校验复核决定：驳回的候选不能导出 ZIP；假设性结果只能导出档案（`manifest.selected_as = hypothesis`），不能保存为普通修复 PNG。
- `engine.export` 与桌面端 `saveResult` 都使用该校验；manifest 新增 `selected_as / review / contest_demo` 字段。

### 合成与候选

- **修复羽化削弱薄掩膜**：`composite()` 对 <4 px 的薄掩膜直接写回完整预测；较厚掩膜只在 1–3 px 边界环带过渡、内部保持完整预测；掩膜外像素仍保证 0 差异。
- **候选数量不再静默减少**：优先使用不同算法，可用方法不足时用不同半径补足用户请求的 1–3 个候选；移除 engine 中“重复方法即提前结束”的分支。

### 测试与验证

- 更新 15 个过期测试，新增 H1/H2 门禁绕过回归、导出决定校验、薄/厚掩膜合成测试；当前 `pytest -q`：**68 passed**。
- 新增 GitHub Actions：非 Web 测试 + `node --check` + `compileall`。
- 发货 WebEngine 外壳验收重跑通过：LaMa + CUDA、掩膜外差异 0、退出码 0。

### 文档与交付

- 重写 `VALIDATION.md` 为当前实测基线；修正使用说明的重复章节编号、过期测试数量、风险/加强/比赛表述与内部引用。
- 新增 MIT `LICENSE`（代码；模型权重与示例图各自许可）；新增本 CHANGELOG。
- 明确比赛观察包/human-gate 样例位于本地 `data/`、不随仓库提供；明确本机 `dist/Shibi-1.0` 为旧构建。

## 2026-10-31 — 早期迭代

历史记录保留在 git 提交中（`fe47956` baseline → `9bda39c` 稳定版 → `44d05c3` 专业流程 → `7fae2cf` 移除旧 UI）。
