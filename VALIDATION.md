# 验证状态

更新时间：2026-11-09（门禁/导出/合成修复后）

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最近一次完整运行：**68 passed, 1 warning**（约 16 秒，exit 0）。覆盖：

- 业务/API/规划器/存储/备份；
- 风险分类与专业门禁（含两条门禁绕过回归：uncertain 掩膜、standard 工作流 enhanced）；
- 复核决定约束导出（驳回 / 假设性结果）；
- 薄掩膜与厚掩膜合成（掩膜外 0 差异、薄掩膜完整预测）；
- 发货 Qt WebEngine 外壳（页面加载、QWebChannel、掩膜往返、建任务→复核→导出、AI 分析按钮、滑动对齐）。

静态检查：

```powershell
node --check ui/app.js
.\.venv\Scripts\python.exe -m compileall -q mural_agent
git diff --check
```

## 发货壳验收

`scripts/verify_desktop.py` 在隔离离屏窗口加载 `ui/index.html`，通过 QWebChannel 完成真实 LaMa CUDA 修复并断言工具/设备/掩膜外差异：

- 结果：exit 0，`page_loaded=true`，job=`review`，`tool=lama`，`device=cuda`，`outside_max_difference=0`；
- 报告：`artifacts/desktop-web-verification.json`；档案：`artifacts/verified-web-restoration.zip`；预览：`artifacts/desktop-web-preview.png`。

## 本轮修复的安全断言（已用隔离脚本复现）

- 旧行为：professional_step + mask `region_class='uncertainty'` + 未传 `has_uncertain` → 可建任务；本轮修复后 → 拒绝（要求专家强制放行 + 理由）。
- 旧行为：`workflow='standard'` + `mode='enhanced'` + 无任何勾选 → 可建任务；本轮修复后 → 拒绝（须勾选掩膜核对/允许加强/风险知悉）。
- 旧行为：驳回或未标记的假设性结果仍可导出/保存；本轮修复后 → 拒绝。
- 合成：<4 px 薄掩膜按完整预测写回；厚掩膜只在边界环带羽化；掩膜外像素仍为 0 差异。

## 尚未验证 / 已知限制

- 真实壁画上的缺损识别准确率、风险分类准确率、修复质量与历史可信度没有专家标注数据验证。
- 视觉 API 需要用户自己的密钥与服务；没有记录真实付费调用。
- `human_gate` 观察包、逐项/批量修复的桌面交互路径仍缺少自动化覆盖；比赛观察包与 human-gate 样例位于本地 `data/`（不随仓库提供），fresh clone 后需要自备素材。
- 本次未做交互式人工桌面点击验收（以离屏 WebEngine 外壳与 API/桥接回归代替）。
- 本机 `dist/Shibi-1.0` 是 2026-10-31 的旧构建，包含已删除的旧 UI 代码，不能作为当前交付物；需要重新构建或删除。
- 工程像素指标与合成缺损测试不能证明补全内容符合壁画历史原貌。应用输出仅供数字修复实验与人工复核。

## 可复现入口

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m "not web"
.\.venv\Scripts\python.exe scripts/verify_desktop.py
```

CI：`.github/workflows/tests.yml` 运行非 WebEngine 测试、`node --check` 与 `compileall`。
