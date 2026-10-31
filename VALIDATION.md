# 拾壁系统验收记录

更新时间：2026-10-31。应用为 Windows 桌面程序，界面使用 Qt WebEngine 加载随包 HTML/CSS/JS，通过 QWebChannel 调用本机引擎；不启动浏览器或本地 Web 服务。

## 已有实测基线

- 首版 Qt Widgets 工作台通过 12 项自动化测试；测试内容覆盖画布、掩膜、后台修复、候选选择、档案恢复、像素保护、异常输入、取消、重启恢复和规划回退。这些测试仍覆盖共享引擎与存储，不代表新 WebEngine 前端的全流程验收。
- 新桌面外壳已在隔离数据目录中启动；实测 Qt WebEngine 本地页面、QWebChannel 快照、本地 960 × 640 图像加载，以及工作台/工具页切换。未触碰正式工作空间数据。
- 独立离屏验收完成了 960 × 640 合成纹样的 LaMa CUDA 修复、传统修补对照、候选选择与 ZIP 导出。RTX 5070 Ti Laptop、PyTorch 2.11.0+cu128 可用；那次合成图推理约 3.3 秒，包含模型加载，不代表实际壁画的固定速度。
- 该基线的掩膜外最大像素差为 0。它验证工程合成约束，不验证历史真实性或专业修复质量。
- `.venv` 中 CUDA PyTorch、Transformers、Diffusers、LPIPS 等模型依赖已安装；主应用启动器使用本地环境。
- SAM 2.1 tiny、Qwen3-VL 2B、Stable Diffusion Inpainting fp16、ControlNet inpaint fp16、LPIPS AlexNet 权重已下载。模型仓库固定提交和 SHA-256 写在各自 `models/*/provenance.json` 中；ControlNet 只下载 fp16 safetensors。
- 完整 Windows 便携目录已生成于 `dist/Shibi-1.0`。其 models 子目录 44 个文件，约 8.34 GB；另含 Python 3.13.12、PyTorch 2.11.0+cu128、Qt 6.12、模型运行库与依赖清单。
- 使用便携目录自带 Python 成功导入 `mural_agent.desktop`、PyTorch CUDA、PySide6、OpenCV、Transformers 和 Diffusers；确认包内 Python 能看到 CUDA。此项是包依赖检查，不是各模型的实际推理验收。

## 已实现但尚未完成本机验收

- SAM 2.1 点/框辅助分割与细线候选分析。输出明确标记为待人工确认，不是病害分类结果。
- 同壁画参考检索、SIFT/RANSAC 配准、人工限定应用区域及证据档案。
- 保守、证据辅助、推测三种模式；质量检查、最多两轮修订、任务预算、失败轨迹与断点续跑。
- Qwen3-VL 离线规划、Stable Diffusion Inpainting、ControlNet、LPIPS 配对评测、数据集导入与分组划分、批处理、空间备份、Windows 便携构建。
- 上述新增能力仍需实际模型推理和交互验收，不能仅凭权重下载、依赖导入或代码存在视为通过。

## 数据和专业边界

- 附带 5 张克利夫兰艺术博物馆开放访问壁画图像，来源条目注明 CC0。它们是公开图像示例，不是无损原貌、成对真值、专家病害标签或代表性固定评测集。
- 尚未完成 CMAMRNet/PGRDiff 等领域模型的许可证核查、可复现推理或质量对照，因此没有将其标成已接入模型。
- 没有配置真实视觉 API 密钥，未进行实际付费 API 调用。
- 工程分数只检查图像边界、有限预算和掩膜外像素等性质。没有可信配对目标和专家评审时，任何分数都不能证明修复内容符合历史原貌。

## 可复现入口

```powershell
.\.venv\Scripts\python.exe scripts/verify_desktop.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/download_models.py sam2 vision diffusion controlnet --lpips
.\.venv\Scripts\python.exe scripts/build_portable.py --output dist/Shibi-1.0 --include-models
```

首版基线产物（来自此前验收）：`artifacts/desktop-preview.png`、`artifacts/desktop-verification.json`、`artifacts/lama-smoke-cuda.json`、`artifacts/native-verification/`。离屏预览不是用户当前桌面窗口的屏幕截图。
