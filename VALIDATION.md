# 拾壁系统验收记录

更新时间：2026-10-31（本次优化后）。应用为 Windows 桌面程序，发货界面使用 Qt WebEngine 加载随包 HTML/CSS/JS，通过 QWebChannel 调用本机引擎；不启动浏览器或本地 Web 服务。

## 当前实测基线

- **测试**：`pytest -q` → **32 passed**（约 9 秒）。覆盖业务/API/规划器、旧版 Qt Widgets 画布回归，以及**发货 WebEngine 外壳**：页面加载、QWebChannel 快照、灰度掩膜往返（防整图选中）、桥接建任务→复核→导出 ZIP、`mergeProposal`、`aux_version` 契约、AI 缺损面板渲染。可复现命令：`.\.venv\Scripts\python.exe -m pytest -q`。
- **发货壳验收**：`scripts/verify_desktop.py` 在隔离离屏窗口中加载 `ui/index.html`，通过桥接完成真实 LaMa CUDA 修复。本次结果：页面加载成功、job=review、候选 `tool=lama`、`device=cuda`、耗时 3.65 s、`outside_max_difference=0`、候选 2 个。报告：`artifacts/desktop-web-verification.json`；档案：`artifacts/verified-web-restoration.zip`；预览：`artifacts/desktop-web-preview.png`。
- **环境**：Python 3.13.12；PyTorch 2.11.0+cu128，`cuda_available=True`；GPU RTX 5070 Ti Laptop 11.94 GiB；PySide6 6.12、OpenCV 4.14、Transformers 4.57.6、Diffusers 0.36.0、LPIPS 可导入；`pip check` 无冲突。
- **权重**：`models/` 5 组权重共 8.34 GB；`big-lama.pt` 实测 SHA-256 与 README 记录一致；SAM2/Qwen3-VL/SD Inpainting/ControlNet 的 `provenance.json` 含固定 revision、许可与逐文件 SHA-256；LPIPS 账本只有 SHA-256/许可，`big-lama` 无 provenance（README 记录 hash）。
- **历史工件**：`artifacts/system-acceptance/report.json` 记录了 LaMa CUDA 3.75 s、SAM 2 实际推理 IoU 0.9866、参考配准通过；`artifacts/mural-study/` 有 48 例配对实验与 24 例细裂缝实验及逐例数据。

## 与旧验收记录的关系

- 旧版 `scripts/verify_desktop.py` 与 `tests/test_desktop.py` 只覆盖旧 Qt Widgets 界面，不能代表发货前端；本次已把验收改为发货 WebEngine 外壳，旧界面仅保留画布回归。
- 旧报告 `artifacts/desktop-verification.json` 在 2026-10-31 16:07 被旧脚本重跑覆盖为 `telea/cpu`（LaMa 仅作为第二候选出现），与 README 旧表述不一致；这正是本次把验收脚本改为“断言工具=LaMa、设备=CUDA”的原因。
- `dist/Shibi-1.0` 为 12:07 构建，不包含本次修复；交付前需重新运行 `scripts/build_portable.py`。

## 已实现但尚未完成真实模型验收

- **Qwen3-VL 缺损识别与本地规划**：已接入发货 UI（“工具/评测”页），子进程隔离、JSON 容错解析与去重/合并逻辑已有单元测试。**真实推理已跑通一次**：在合成纹样练习图上经桥接调用 Qwen3-VL 2B，含模型加载约 41 秒，返回 8 条原始候选（模型重复同一裂缝多次），经去重/合并后得到 3 条候选并成功生成红框预览（`artifacts/damage-smoke/`）。它不是真实壁画，漏检/误检率与建议质量仍未评估。
- **Stable Diffusion / ControlNet**：ControlNet 加载缺 fp16 变体的问题已修并可在本地实例化，但尚未实际生成候选；SD 仍为整图 512px 推理，未实现 ROI。
- **参考检索与跨图配准**：SIFT/RANSAC 与人工核验门禁有测试，真实跨图重建质量未验证。
- **配对评测（LPIPS）、数据集分组、批量任务、空间备份**：代码存在，未做完整交互验收。
- **真实视觉 API**：没有配置计费密钥，未进行实际 API 调用验收。

## 数据和专业边界

- 附带 5 张克利夫兰艺术博物馆开放访问壁画图像（CC0）。它们是公开图像示例，不是无损原貌、成对真值、专家病害标签或代表性固定评测集。
- 修复效果实证来自合成掩膜（48 + 24 例），测的是“能否补回被遮挡的像素”，不是历史真实性或专业修复质量。
- 尚未完成 CMAMRNet/PGRDiff 等领域模型的许可证核查、可复现推理或质量对照，因此没有将其标成已接入模型。
- 工程分数只检查图像边界、有限预算和掩膜外像素等性质；没有可信配对目标和专家评审时，任何分数都不能证明补全内容符合历史原貌。

## 可复现入口

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_desktop.py
.\.venv\Scripts\python.exe -m pytest -m "not web"     # 无 QtWebEngine 的机器
.\.venv\Scripts\python.exe scripts/download_models.py sam2 vision diffusion controlnet --lpips
.\.venv\Scripts\python.exe scripts/build_portable.py --output dist/Shibi-1.0 --include-models
```

本次优化的具体改动与剩余问题见 [CHANGELOG.md](CHANGELOG.md)；优化前的独立评估见 [PROJECT_REVIEW.md](PROJECT_REVIEW.md)。
