**拾壁 · 壁画修复工作室（Windows 原生桌面版）**

独立的 Windows 桌面应用，窗口由 PySide6 / Qt WebEngine 承载本地 HTML、CSS、JavaScript 界面，并通过 QWebChannel 调用本机修复引擎。不会启动浏览器标签页或 HTTP 网页服务；文件读写仍由桌面窗口和本机工作空间处理。

已实现：图像导入、画笔与橡皮、撤销、冻结掩膜、LaMa GPU 补全、传统算法对照、有限候选、原图滑动对照、修改差异、候选复核、历史恢复、断点续跑、结果保存和 ZIP 档案导出。**AI 一键恢复**：不用手动画掩膜，本地 Qwen3-VL 自主识别疑似缺损、生成并合并掩膜、按类别/尺寸选择保守修复工具，自动提交有限预算任务；所有识别框、掩膜来源与工具路线都写入任务记录，结果仍需人工复核、可改掩膜重跑。“工具与评测”页提供本地 Qwen3-VL 原图缺损候选识别、按缺损类别生成 SAM 或细线提取掩膜、类别和尺寸对应的修复方法建议、细线启发式候选、参考资料检索与配准核验、保守/证据/推测三种模式、有限质量反馈修订、数据清单导入、按壁画分组划分、配对评测、批量任务及工作空间备份。缺损识别、分割和类型建议都是候选，需人工确认；SAM 的分割置信度和视觉模型自评不等于病害概率。视觉 API 与本地 Qwen3-VL 都是可选规划器；不配置它们也可由本地规则完成修复。

详细的操作步骤、界面说明和常见问题见 **[使用说明.md](使用说明.md)**。

**打开应用**

当前机器直接双击项目目录中的 **拾壁.exe**，或双击 `start.cmd`。也可以执行：

```powershell
cd 'D:\lyn\壁画修复agent'
.\.venv\Scripts\python.exe -m mural_agent.desktop
```

`拾壁.exe` 是本机安装版本的入口，调用同目录 `.venv` 中的 GUI Python；需保留项目代码、运行环境和模型目录。它不是可单独复制到其他电脑的完整便携包。此结构避免把数 GB 的 CUDA 运行库塞入单文件，方便迭代和更新模型。

全新环境安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_lama.py
.\scripts\install_gpu.ps1
.\start.cmd
```

`requirements.txt` 只含桌面运行与核心图像依赖；模型/GPU 依赖在 `requirements-models.txt`，测试与 FastAPI 开发适配器依赖在 `requirements-dev.txt`。当前机器依赖和 LaMa 权重已经安装，项目 `.venv` 使用 PyTorch 2.11.0+cu128；系统 Conda 环境原有 PyTorch 保持不变。SAM 2.1、Qwen3-VL 2B、Stable Diffusion Inpainting、ControlNet inpaint 和 LPIPS AlexNet 权重已下载到本地 `models/`（约 8.34 GB）。四个 Hugging Face 模型目录带 `provenance.json`（repository、固定 revision、许可、逐文件 SHA-256）；`big-lama.pt` 的固定 SHA-256 记录在本文件下方，LPIPS 账本只有 hash/许可、没有 revision。

> 环境提示：本机 `.venv` 是 `python -m venv --system-site-packages` 创建的，`numpy` 实际来自系统 Conda。新机器请用干净 venv 并按 requirements 安装；两套环境不完全等价，依赖升级前建议先跑 `pytest -q`。

**操作流程**

1. 点击“打开图像”，或使用合成纹样练习图。
2. 用画笔圈选缺损；红色表示允许编辑。可使用橡皮、撤销、导入掩膜和适应画布。
3. 如需辅助：“工具与评测”页先运行“分析原图缺损”（本地 Qwen3-VL 提出疑似缺损框、类别与建议），选中条目后“细化选中区域”（裂缝走细线提取，其他类别走 SAM 框提示）；也可直接使用细线启发式候选或 SAM 点/框提示。采用候选后仍须人工修订并确认。证据模式需导入参考资料、配准并人工核验可应用区域。
4. 选择保守、证据辅助或推测模式，再选择可用修复工具、规划器和 1–3 个候选。推测模式允许扩散模型生成假设性结果；它不代表历史原貌。也可以直接点右侧 **“AI 一键恢复”**：本地 Qwen3-VL 自动识别缺损、生成掩膜并按类别选择保守工具，自动生成候选；识别框、掩膜和工具路线都会记录，仍需人工复核。
5. 点击“生成修复候选”，后台顺序执行，界面保持可操作。
6. 在“候选与对照”中选择候选，使用滑块对照、查看修复图或差异图。
7. 填写选择依据，点击“选定此候选”，保存 PNG 或导出 ZIP 档案。
8. 点击左侧“历史记录”，单击任务恢复原图、掩膜、候选和复核记录。

快捷键：`Ctrl+O` 打开、`Ctrl+Z` 撤销、`B` 画笔、`E` 橡皮、`Ctrl+滚轮` 或工具栏按钮缩放，点击“适应”恢复全图视野（当前版本不提供滚动条平移）。任务运行中会禁止打开新图/新建任务；关闭窗口时先请求取消，取消中再点一次关闭可强制退出。

练习图为程序绘制的合成纹样，不是历史文物照片。辅助选区不是病害分类器，SAM 2 的分割分数和视觉模型置信度都不是病害概率。工作区附带少量克利夫兰艺术博物馆开放访问的壁画图像作为来源登记示例；它们不是完整原貌、配对真值或专家标注数据。

**配置大模型 API**

打开应用左侧“API 设置”，填写接口协议、服务根地址、视觉模型名称和密钥。密钥输入框遮盖显示，配置保存在项目 `.env` 文件中并被 Git 忽略。保存后后续任务立即生效，不需要改代码。

支持两种协议：

| 协议 | 根地址示例 | 实际接口 |
|---|---|---|
| Responses | `https://api.openai.com/v1` | `/responses` |
| Chat Completions 兼容服务 | 提供商的 `/v1` 根地址 | `/chat/completions` |

选择支持图像输入的模型。BASE_URL 不包含最终接口路径。如果兼容服务不支持 `response_format`，在设置中选择兼容模式。

API 模式发送最长边不超过 1024 的原图缩略图、红色掩膜预览和备注，不发送原始文件。服务端式规划校验仍由本地代码执行：只允许白名单工具与最多 3 个候选，不能执行任意代码。API 出错或返回格式不合要求时，记录明确的规则回退。

没有配置真实 API 密钥，因此目前未进行实际计费 API 验收；两种请求格式和异常回退已通过模拟 HTTP 测试。可参考 [OpenAI 官方图像输入格式](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses)。

**模型与 GPU**

LaMa 权重来源：[Sanster/models 发布资源](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)，约 196 MB，保存于 `models/big-lama.pt`。

当前下载的 SHA-256：`344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`。hash 记录文件身份，不是独立真实性签名。

本机 RTX 5070 Ti Laptop 约 12 GB 显存，已经通过真实 CUDA 推理验证。`MURAL_DEVICE=auto` 根据 PyTorch 支持情况选择 GPU 或 CPU；`MURAL_DEVICE=cuda` 要求 GPU 可用，否则明确报错。设备实际使用情况写在候选记录中。

**可选模型安装**

新机器可在项目根目录运行以下命令安装模型依赖并下载固定 revision 的权重（当前开发机已安装）。模型文件保存在 `models/`（被 Git 忽略）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe scripts/download_models.py sam2 vision diffusion controlnet --lpips
```

SAM 2 与本地 Qwen3-VL 供选区辅助、缺损识别和离线规划使用；Stable Diffusion Inpainting 仅用于推测模式，ControlNet 可选（控制加载要求 `model.fp16.safetensors` 布局，已在 `model_worker.py` 中按 fp16 变体加载）。权重约需 8.34 GB，另需 LPIPS 骨干权重和运行缓存。Stable Diffusion / ControlNet 受各自模型卡许可约束；下载和使用前应阅读模型目录中的许可文件。扩散推理在本机通过 512px 输入和 CPU 卸载限制资源，适合生成候选，不适合作为历史证据。

CMAMRNet、PGRDiff 等壁画研究模型没有自动下载或伪造适配器。只有在代码、权重和数据许可核清，并实现其真实推理接口后，才能通过 `models/research-adapter.json` 接入。

大图使用带上下文的 ROI，默认推理最长边 1024；合成时只改动冻结 mask 内部。薄掩膜在缩放到推理分辨率前会按比例膨胀以避免细裂缝在缩小过程中消失，合成仍只使用原始冻结掩膜。结果与经过 EXIF 校正的 8 位 RGB 工作图比较，mask 外像素差必须为 0。原始上传文件字节独立保留。

这类图像补全不等于确认历史原貌。边界色差只是对照参考；真实缺损没有原始真值时，不能凭 PSNR/SSIM 证明真实性。

**项目文件**

```text
拾壁.exe                    本机版应用入口（PyInstaller 启动器）
start.cmd / start.ps1        开发启动入口
desktop_launcher.py         EXE 入口源码
mural_agent/
  desktop_web.py            发货的 Qt WebEngine 外壳与 QWebChannel 桥
  desktop.py                旧版 Qt Widgets 界面（保留画布回归与 configure_application，不是发货界面）
  desktop_features.py       旧版 Qt Widgets 扩展页（仅旧界面使用）
  damage_assessment.py      本地 Qwen3-VL 缺损候选识别与修复方法建议
  auto_restore.py           一键 AI 自主恢复：识别 → 掩膜 → 选法 → 建任务
  workspace.py              本地应用服务（图像/掩膜/任务）
  engine.py                 异步执行、质量闭环与档案导出
  models.py                 LaMa / CUDA 适配与薄掩膜保护
  planner.py / local_planner.py   规则、视觉 API、本地 Qwen 规划
  segmentation.py           细线启发式、裂缝提取、SAM 2.1
  references.py             参考检索与 SIFT/RANSAC 配准
  quality.py                工程检查与配对指标
  storage.py / backup.py    SQLite 记录（WAL）与安全备份
  imaging.py                解码、合成、像素约束和传统修补
  tools.py / model_worker.py      白名单工具与扩散/研究模型子进程
  app.py                    FastAPI 开发回归适配器（不随产品启动）
ui/                         发货前端（index.html / app.js / styles.css）
models/                     权重（不提交 Git）
data/                       原图、冻结掩膜、候选和任务数据库
assets/app.ico              应用图标
图片/                       8 张已注明许可的单幅壁画照片（可直接打开；附来源与清单）
tests/                      业务、API、规划器、旧版画布与发货壳回归测试
scripts/                    下载、验收、研究实验与便携构建脚本
artifacts/                  本机生成的验收报告与预览（不提交 Git）
CHANGELOG.md                本次优化记录与已知未完成项
PROJECT_REVIEW.md           优化前的独立项目评估报告
```

ZIP 档案包含原始文件字节、工作图、mask、选中结果、差异图和 manifest.json；记录图像/模型/权重 hash、版本、参数、设备、耗时、规划轨迹和人工意见。

取消为协作式：等待当前 API 或模型调用结束后停止。关闭有运行任务的窗口时，先请求取消并提示；取消中再点一次关闭可强制退出。再次启动时会标记上次未完成任务，保留已有候选。

生成不依赖项目虚拟环境的 Windows 便携目录：

```powershell
.\.venv\Scripts\python.exe scripts/build_portable.py --output dist/Shibi-1.0 --include-models
```

`dist/Shibi-1.0` 是 2026-10-31 12:07 构建的旧版本，**不包含本次前端/测试/算法修复**，交付前需重新运行上面的命令。将整个目录交付给另一台兼容的 Windows x64 电脑，运行其中的 `拾壁.exe` 或 `start.cmd`。便携包不会包含 `.env` 中的 API 密钥和 `data/` 用户档案；CUDA 是否可用仍取决于目标电脑驱动和显卡。

**测试与重新构建入口**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q            # 42 项，约 13 秒
.\.venv\Scripts\python.exe -m pytest -m "not web"  # 无 QtWebEngine 的机器
.\.venv\Scripts\python.exe scripts/verify_desktop.py
.\.venv\Scripts\python.exe scripts/build_launcher.py
```

`verify_desktop.py` 现在验收**发货的 Qt WebEngine 外壳**：离屏加载 `ui/index.html`，通过 QWebChannel 桥建图/存掩膜/建任务，断言候选来自 LaMa、设备为 CUDA、mask 外像素差为 0，并导出 ZIP。报告为 `artifacts/desktop-web-verification.json`，档案为 `artifacts/verified-web-restoration.zip`，预览为 `artifacts/desktop-web-preview.png`。旧 Qt Widgets 界面的画布逻辑仍由 `tests/test_desktop.py` 做回归，但不再作为产品验收依据。

尚需领域工作：取得可许可的真实壁画资料并由修复专家标注；建立按原壁画分组且无泄漏的固定评测集；复现和比较壁画专用研究模型；针对目标壁画验证辅助分割、配准和扩散候选的实际收益；做一次真实的 Qwen3-VL 缺损识别与结构化评审。通用指标与工程检查不能证明补全内容符合历史原貌。详细设计保留在同目录《壁画修复Agent_GitHub调研与完整制作流程.md》，其中初始 Web 交互方案已由当前原生桌面实现取代。
