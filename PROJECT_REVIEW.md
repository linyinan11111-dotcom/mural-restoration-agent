# 《拾壁 · 壁画修复工作室》项目评估报告

> **状态说明：** 本文写于 2026-10-31 优化之前，记录的是修复前的评估结论。本次已按报告中的 P0/P1 建议修复了发货前端的阻断缺陷、测试隔离、验收口径、ControlNet 加载与薄掩膜保护等问题；修复清单与仍待完成项见 [CHANGELOG.md](CHANGELOG.md)，最新实测基线见 [VALIDATION.md](VALIDATION.md)。

**评估日期：** 2026-10-31  
**评估对象：** `D:\lyn\壁画修复agent`（Windows 桌面版，Python 3.13 + PySide6/Qt WebEngine + 本地修复引擎）  
**评估方式：** 3 名独立评审员分别负责「可运行性/验收实证」「桌面与前端」「算法与数据管线」，主线复核关键结论；全程只读，仅运行现有测试/验收脚本与轻量函数级验证，未联网、未改源码。

**规模：** Python 5,213 行（37 文件）+ 前端 769 行（HTML/CSS/JS，3 文件）；模型权重 8.34 GB；便携包 27,949 文件 / 13.94 GB。

---

## 一、总体结论

这是一个**完成度明显高于普通"课程作业/AI 演示"的单人级领域工具原型**：工程骨架真实、权重与来源可追溯、算法链路不是桩、对自身局限的表述罕见地诚实，还做了两组有统计意义的配对实验（48 例 + 24 例）。

但它目前**还不是可交付产品**，更不是可写进文物保护流程的工具。核心原因是：

1. **实际发货的 WebEngine 前端存在多个一用即坏的阻断性缺陷**（掩膜导入即整图选中、参考核验流程冻结、选区合并必然报错、辅助结果后界面每 0.9 秒被强制跳页），而现有自动化测试和验收脚本**测的是已经被弃用的旧 Qt Widgets 界面**；
2. 本机测试基线**不绿**（21 项中 1 项确定性失败，且失败与环境里装了 Qwen3-VL 权重直接相关）；
3. 验收报告**不自证**：`verify_desktop.py` 不断言"修复工具必须是 LaMa、设备必须是 CUDA"，重跑后报告已从 `lama/cuda` 漂移成 `telea/cpu`；
4. 文档与实现存在多处漂移，仓库**没有任何 git 提交**，无法审计或回滚。

**综合评分：6.5 / 10** — 作为可演示原型与工程/伦理严谨性范例属上乘；作为可交付产品或研究基线尚不合格。

| 维度 | 分数 | 简评 |
|---|---:|---|
| 问题价值与定位 | 8.0 | 真实痛点，边界划得清（数字修复 ≠ 历史原貌） |
| 功能覆盖 / 产品完整度 | 6.5 | 功能面很广，但辅助/证据/合并等流程实际不可用 |
| 架构与工程质量 | 7.0 | 分层清晰、进程隔离、约束在运行时强制；但两套 UI、逻辑重复 |
| 代码质量 | 6.0 | 模块小而清晰，但死代码、GUI 线程重活、前端若干低级 bug |
| 验证与可复现性 | 6.5 | provenance/配对实验很强；测试红、验收口径错位 |
| 文档一致性 | 5.5 | README 文件表/快捷键/验收描述已过期，VALIDATION 落后于新工件 |
| 可交付性 | 5.5 | Exe/便携包真实，但未签名、前端缺陷、无迁移/回归保障 |

---

## 二、项目定位与完成度

- **它是什么：** 一个本地优先的壁画数字修复工作台。导入图像 → 手绘/辅助圈选缺损 → 冻结掩膜 → LaMa/传统算法/参考证据/扩散生成 1–3 个候选 → 滑杆对照、差异图、质量检查 → 人工选定 → 导出带完整 provenance 的 ZIP 档案。
- **它确实做到了什么（有实测证据）：** LaMa CUDA 推理、传统 Telea/NS、SAM 2 点/框分割、SIFT/RANSAC 参考配准与覆盖范围门禁、LPIPS 配对指标、按同哈希合并去重的分组划分、断点续跑、协作式取消、ZIP 档案导出、SQLite 记录、便携包与 EXE 启动器。
- **它明确没做到什么（项目自己也承认）：** 没有专家标注的真实壁画配对数据；SAM2/Qwen3-VL/Stable Diffusion/ControlNet 没有完成本机推理验收；CMAMRNet/PGRDiff 等领域模型未接入；没有真实付费 API 验收；通用指标不能证明历史原貌。

这个"做到/没做到"的分界写得比大多数同类项目清楚，是它最值得肯定的部分。

---

## 三、真正的亮点（建议保留并写进简历/答辩）

1. **"mask 外像素必须为 0 差异"是运行时强制，而非文档口号**：`imaging.py:52-58` 用赋值而非色彩空间转换保证未编辑像素原样；`engine.py:165-169` 在落盘前断言 `outside_unchanged`，不通过直接抛错。实证工件 `artifacts/lama-smoke-cuda.json`、`system-acceptance/report.json` 均为 0。
2. **规划器白名单是真的强制**：`planner.py:11-19` 的 `Step` 用 `Literal` + `extra='forbid'` 约束；`engine.py:87-95` 用运行时策略覆盖 `mode/regions/evidence/budget`，模型无法改变预算与模式；API/本地模型输出还要过 pydantic 与工具白名单（`planner.py:125-128`）——这比"让 LLM 自由调用工具"安全得多。
3. **模型供应链可追溯**：`download_models.py:13-18` 固定 HF revision，逐文件校验 LFS SHA-256 并写 `provenance.json`（repository/revision/license/files）；`big-lama.pt` 实测 SHA-256 与 README 公布值完全一致；range 分块下载还校验了长度与 `Content-Range`。
4. **数据/归档安全做得细**：`backup.py:37-62` 恢复时拒绝绝对路径、`..`、反斜杠、盘符与清单外文件，并有 zip-slip 回归测试；`storage.py:69-75` 只允许 32 位 hex 作为目录名；导出 ZIP 前重新校验冻结输入 hash（`engine.py:64-69`）。
5. **量化自评不是摆拍**：`artifacts/mural-study/` 用 8 张 CC0 博物馆壁画 × 3 种缺损 × 2 档强度 = 48 个配对案例比较 LaMa vs Telea（MAE 21.90 vs 22.59、PSNR 19.53 vs 19.36）；`small-cracks/` 又补了 24 例细裂缝（NS 最好，MAE 9.79 vs LaMa 14.06），并明确写出"这不代表真实壁画修复排序，不能作施工依据"。这种自我证伪的写法值得保留。
6. **安全基线对本地应用足够**：CSP 禁止内联脚本与远程连接（`ui/index.html:6`）、`LocalContentCanAccessRemoteUrls=False`、QWebChannel 暴露面窄、`_file_url` 限制在工作空间内、JS 侧无 `eval` 且输出基本都过 `escapeHtml`、API key 不在快照中回传。
7. **诚实性**：README/VALIDATION 反复声明"合成掩膜不是历史真值""边界色差只是启发式""API 未实际计费验收""新能力尚未验收"。在同类项目里，这种克制是稀缺品质。

---

## 四、阻断级问题（P0：先修再谈交付）

### P0-1 WebEngine 版掩膜导入/恢复会把整幅图变成可编辑区（最严重）
`ui/app.js:160-167,181-187` 把后端生成的黑白灰度掩膜（L 模式 PNG）直接 `drawImage` 到画布；`ui/app.js:169-179` 的导出却按 **alpha 通道** 判断可编辑像素。灰度图不透明 → 画布 alpha 全为 255 → 导出即"整图选中"。评审员已用 200×120 掩膜实测 alpha 覆盖率 100%。
**影响：** 导入掩膜、采用辅助选区、合并选区、历史恢复掩膜全部退化为整图选中，随后被 `workspace.py:32-33` 以"不能选择整幅图像"拒绝 → 这四类流程实际不可用。旧版 `desktop.py:220-232` 做了 `>127` 阈值转换，是正确的参照实现。

### P0-2 选区合并必然 NameError
`desktop_web.py:276` 调用 `Image.fromarray(...)`，但该函数内只 import 了 `decode_image` 与 `numpy`，模块顶部也没有 `PIL.Image`。实测直接返回 `{"error":"name 'Image' is not defined"}`。
**影响：** "与当前掩膜合并"功能 100% 失败。

### P0-3 参考导入/核验用 `window.prompt`，在 QWebEngine 里不可用
`ui/app.js:389,417-421` 用 `window.prompt` 收集来源、许可、分组与核验说明；`WebMainWindow`（`desktop_web.py:437-452`）没有实现 `QWebEnginePage.javaScriptPrompt` 回调。Qt 默认实现不提供输入对话框（离屏实测 7 秒不返回、JS 停住；实际窗口则表现为无输入或静默失败）。
**影响：** "导入参考""核验应用区域"两个证据流程无法完成，而证据模式又是 README 五个核心卖点之一。应改为 HTML 内联表单/模态框，或实现 page 对话框回调。

### P0-4 辅助选区生成后，界面每 0.9 秒被强制拉回工具页
`Bridge.aux` 永不清空（`desktop_web.py:76-88`），`snapshot` 每次都返回它；`ui/app.js:333-338` 在 `aux.status==='done'` 且含 proposal 时无条件 `showPage('tools')`，而刷新定时器是 900ms（`ui/app.js:481`）。
**影响：** 只要生成过一次辅助选区，用户一回到工作台就被弹走，且图片反复重载；工作台基本不可用。

### P0-5 ControlNet 结构控制必定失败（推测模式）
`model_worker.py:28-29` 单独加载 ControlNet 时没传 `variant='fp16'` / `use_safetensors=True`，而仓库里只有 `diffusion_pytorch_model.fp16.safetensors`；本地已复现 `OSError`。但 `tools.py:26` 与 `ui/app.js:441` 只看 `config.json` 存在就允许勾选。
**影响：** "推测模式 + 结构控制"会先白跑 2 轮修订再失败。补上两个参数即可加载（评审员实测 0.06 秒加载成功）。

### P0-6 关闭/取消流程没有可靠的退出路径，且会留下孤儿任务
`desktop_web.py:469-485` 关闭时只置 `cancel_requested` 每 250ms 轮询；模型调用卡住则窗口永远关不掉；第二次点 X 会 `shutdown(wait=False)` 后 accept，但正在跑的 worker/线程不会停（Python 退出还会 join 非 daemon 线程）。同时 `ui/app.js:189-210,316-327` 在任务运行中仍允许换图/恢复历史，`state.job` 被清空后正在跑的任务在 UI 上失去取消入口（`desktop_web.py` 的 job 仍在跑）。
**影响：** 卡死、假关闭、孤儿任务，正是 README:111 宣称"先请求取消再自动退出"的反面。

---

## 五、测试与验收可信度（P0/P1）

1. **本机测试基线不绿。** 实测 `.\.venv\Scripts\python.exe -m pytest -q` → **1 failed / 20 passed / 21 collected**（14.18s），失败项 `tests/test_desktop.py::test_native_canvas_and_restoration_without_webserver`（10 秒 deadline 内任务停在 `planning`，单测复跑 10.92s 仍失败）。
   **根因明确：** 测试只隔离了 `data_dir` 与 `lama_model`（`tests/test_desktop.py:17`），没有隔离 `vision_model`；而 `desktop.py:552-556` 在本机存在 `models/qwen-vl/provenance.json` 时把旧版窗口默认规划器设为 `local`。于是测试每次都去子进程加载 Qwen3-VL 2B（数十秒），10 秒内必然停在 `planning`。**这个测试的通过与否取决于机器上是否下载了权重，属于非 hermetic 测试。**
2. **验收脚本与发货 UI 错位。** `scripts/verify_desktop.py:19` 与被测的 `tests/test_desktop.py:12` 用的都是 `mural_agent.desktop` 的旧 Qt Widgets 主窗口；真正发货的是 `desktop.py:1082-1084 → desktop_web.py` 的 WebEngine 壳（`ui/`）。也就是说：**现有自动化验收 0 覆盖发货前端**，而发货前端恰好集齐了 P0-1～P0-4 四个缺陷。`VALIDATION.md:7` 已承认"不代表新 WebEngine 前端的全流程验收"，但 README 没有同步。
3. **验收报告不自证、会漂移。** `verify_desktop.py` 只记录"被选中的候选"，不断言工具/设备；重跑后 `artifacts/desktop-verification.json` 从旧值 `lama / cuda / 3.81s` 变成 `telea / cpu / 0.11s`（LaMa CUDA 只作为第二候选出现在 job 明细里）。README:130 与 VALIDATION:9 据此声称"真实 LaMa GPU 修复"，在重跑证据下已不成立。应改为：断言 `candidate.tool == 'lama'`、`versions.device == 'cuda'`、`candidate_count >= 2`，否则脚本以非零退出。
4. **文档漂移清单：**

| 位置 | 文档声称 | 实测/代码事实 |
|---|---|---|
| README:87-107 | 文件表把 `desktop.py` 描述为"原生 Qt 界面…" | 发货路径是 `desktop_web.py` + `ui/`；文件表漏列两者，`web/` 是空目录 |
| README:41 | `Ctrl+O` 打开、缩放后滚动条平移 | WebEngine 版无 Ctrl+O，`#image-stage` 为 `overflow:hidden`，无平移 |
| README:125 | `pytest -q` 作为可复现入口（暗示全绿） | 当前 1 failed；VALIDATION:7 的"12 项测试"是历史值，现为 21 项且不通过 |
| README:130 / VALIDATION:9 | verify_desktop 做"真实 LaMa GPU 修复" | 脚本不固定 planner/backend，重跑已漂移为 telea/cpu |
| README:28 | 每个模型目录都有 revision/license/SHA-256 | `big-lama.pt` 无 `provenance.json`；`lpips/provenance.json` 无 revision |
| README:79 | 权重约 7.5 GB | 实测 8.34 GB（decimal） |
| VALIDATION:16-22 | SAM2/参考配准/Qwen 规划"尚未完成本机验收" | `artifacts/system-acceptance/report.json` 已记录 SAM2（IoU 0.9866）与 reference 通过；本次 Qwen3-VL 也真实产出了合法 JSON 规划 → 文档反而低估了自己 |
| README:3 | "不会启动 HTTP 网页服务" | 成立；但 `requirements.txt` 仍含 fastapi/uvicorn，`app.py` 与 `workspace.py` 有重复实现 |
| README:16 | 依赖通过 `python -m venv .venv` 安装 | 本机 `.venv` 实为 `--system-site-packages`，`numpy` 实际来自 Conda（`C:\Users\lin\miniconda3\...`），环境不可原样复现 |

5. **仓库没有版本历史。** `git log` → "master does not have any commits yet"，19 项全部 untracked。一个强调 provenance、可复现、验收存证的项目，却没有提交历史，这是流程层面的硬伤（也直接导致无法定位"哪次改动引入 P0-1"）。

---

## 六、算法/数据层问题（P1）

1. **薄掩膜在推理缩小阶段可能整体消失。** `models.py:65-73` 用 NEAREST 把 mask 缩到 `lama_max_side`（默认 1024），实测 4000 宽图上的 1 px 竖线 mask 缩后为 0 像素；SD 路径（`model_worker.py:41-44`）同样，且把整图缩到 512、没有 ROI。后果：细裂缝类输入出现"模型没看到缺损"，候选退化为原图，`no_change` 触发修订仍无解。建议：按缩放比例膨胀推理掩膜（≥ `ceil(1/scale)+1`）、断言 `mask_small.any()`、SD 改用 ROI。
2. **没有 OOM 处理与权重校验。** `models.py:54-85` 单次 fp32 CUDA 推理，OOM 只被 `engine.py:190-200` 当普通"工具失败"降级，不清显存、不降分辨率；`tools.py:37-38` 记录权重 SHA-256 但不与固定值比对，`download_lama.py` 也不校验 README 公布的 hash，`available()` 仅查文件存在 → 半下载/被替换的权重可静默使用。
3. **规划回退不透明。** `engine.py:85-86` 吞掉所有异常只写一句通用事件，不记录原因；`plan.source` 仍是 `rules`，无法区分"规则规划"与"模型失败回退"；`planner.py:124` 不剥离 ```json 代码围栏（Responses/兼容服务很常见），会静默降级；无 429/5xx 重试。建议记录异常类型、置 `source='rules_fallback'`、剥离围栏、重试一次。
4. **分组划分没有接入任何评测。** `datasets.py` 的 union-find 去重分组是真实现（实测无跨组泄漏），但 `benchmark.py:37` 直接取整个 manifest，从不划分 train/val/test，近重复/相邻裁片也未处理（代码自述）。"无泄漏划分"目前只是 UI 里的一个导出按钮。
5. **配对指标口径不一致。** `quality.py:51-58` MAE/PSNR 只在 mask 内，SSIM/LPIPS 在 bbox+16 px（含大量未改动像素），合成评测中 SSIM 会被稀释；LPIPS 每次调用都重建网络且跑 CPU。
6. **依赖声明缺口。** `requirements.txt` 不含 torch/transformers/diffusers/scikit-image/lpips；`quality.py:55` 无条件 `import skimage`，只按 README 的快速安装会 ImportError；本地 VLM 可用性只查 `provenance.json`（`workspace.py:53`），缺 transformers 时静默回退规则。
7. **参考检索偏薄。** `references.py:27-44` 的"同壁画检索"只是全局 HSV 直方图 + 关键词加分，"同壁画"依赖人工填写的 group 文本；SIFT/RANSAC 配准与覆盖门禁（47-98）是真实且严格的，差异不在配准而在检索。
8. **存储与性能。** 单表 JSON、无 schema 版本/迁移/kind 索引（`storage.py:24-25`），`update` 为读改写、`event` 无上限，`list` 全量反序列化；前端每 900ms 全量快照并全量重渲染（`desktop_web.py:76-88` + `ui/app.js:212-229`），任务/事件一多就卡顿、图片闪烁。SQL 参数化与 32-hex 目录白名单做得对，问题在规模。

---

## 七、前端与安全层其余问题（P1/P2）

- **GUI 线程上的长任务**：`batchJobs`（最多 50 组解码+建任务）、`evaluatePair`（CPU LPIPS）、`addReference`/`importMask`/`openImage`（读文件+解码）都在 GUI 线程同步执行（`desktop_web.py:350-391,286-292,219-224,131-139`），与 README"界面保持可操作"矛盾；且 WebEngine 版缺少旧版的 35 MB / 2000 万像素前置校验。
- **撤销语义退化**：`ui/app.js:364-365,384-385` 只回放笔迹，清空/导入/采用选区不可撤销，与旧版整图快照式撤销不一致。
- **状态串台**：`loadImage` 不重置 `state.registration/references/proposalIds`（`ui/app.js:115-158`），换图后可能把旧 registration 传给新任务被后端拒绝；历史恢复不走桥接口，`Bridge.current_image_id` 不更新，`adoptProposal/mergeProposal` 会把合法选区判为"不属于当前图像"；恢复时 `loadImage` 的异步 `onload` 可能清掉先到达的 mask；复核意见不回填、再选候选会用空串覆盖（`ui/app.js:316-327,373`）。
- **旧任务的候选残留**：新任务创建时不清空候选条，`renderCandidates` 在空列表时直接 return，旧卡片可被点击。
- **批量任务 guard 失效**：`readSettings()` 不含 `mode`，`batchJobs` 里 `settings.get('mode')` 恒为 `conservative`，非保守模式批量任务会在创建一半记录后才报错。
- **WebEngine 加固不足**：`LocalContentCanAccessFileUrls=True` 且没有 `acceptNavigationRequest`/新窗口策略/off-record profile；CSP 缺 `object-src`/`base-uri`/`form-action`。当前内容全本地可信，风险有限，但一旦出现 XSS 就会被放大。
- **可用性细节**：`window.prompt` 之外，大量原生对话框以 `None` 为 parent（Windows 上可能被主窗口遮挡）；`busy` CSS 类不存在；SAM/橡皮无光标反馈、SAM 框选无矩形可视化；`Ctrl+O`/平移缺失；mask 仅靠红色区分，色觉障碍用户不可辨；`data/desktop.log` 只追加不轮转。
- **死代码与重复**：`desktop.py`（1089 行旧 Widgets 界面）+ `desktop_features.py`（403 行）只为测试/验收存在，发货壳却为复用 `configure_application` 导入整个旧模块；`app.py`（FastAPI）与 `workspace.py` 重复实现 `add_image/add_mask/demo`；`web/` 空目录；`requirements.txt` 仍含已不使用的 fastapi/uvicorn。
- **交付物**：`拾壁.exe` 与 `dist` 同 hash、PE/AMD64 有效；但**未签名、无版本信息**；`dist` 内 `scripts/` 缺 `public_fresco_study.py`/`small_crack_study.py`（应用运行不需要，但让"可复现研究"在便携包里断链）。

---

## 八、建议路线（按优先级）

**P0（1–2 天，先让"声称的东西真的能用/测试真的绿"）**
1. 修 P0-1～P0-4：灰度掩膜按 `>127` 转 alpha、补 `PIL.Image`、把 `prompt` 换成 HTML 模态、`aux` 消费后清空或按结果 id 渲染。
2. 让测试重新绿：`tests/test_desktop.py` 的 fixture 把 `vision_model` 指向不存在路径（或固定 `planner='rules'`），并给 `pytest` 一个真正确定性的基线；同时把 `test_desktop` 标注为 "legacy Widgets"。
3. 给 **发货的 WebEngine 壳** 补一个最小离屏回归：加载 `ui/index.html` → 等待 `loadFinished` → 调 bridge 建图/存掩膜/建任务 → 轮询到 review → 导出 ZIP，断言 mask 外像素为 0。
4. `verify_desktop.py` 增加 `candidate.tool/device` 断言与失败退出码；同步修正 README:130、VALIDATION:9 的表述。
5. `git init` + 分阶段提交（至少把当前状态打 tag），从此每次验收产物与代码版本绑定。

**P1（约 1 周，功能可信度）**
6. ControlNet `variant='fp16'` 修复 + 可用性检测改为真实可加载校验；`big-lama` 补 `provenance.json` 与启动时 hash 校验；`download_lama.py` 校验 README 公布的 hash。
7. 薄掩膜保护 + LaMa OOM 降级（降分辨率/清缓存/明确报错）；SD 走 ROI。
8. 关闭/取消：增加"正在取消"状态、超时强制退出入口；任务运行时禁止换图/开新任务；辅助工具改用独立 executor。
9. 依赖分层与锁定：torch/transformers 等进 `requirements-models.txt` 已在做，但 `skimage` 需进主依赖或条件导入；`.venv` 不要 `--system-site-packages`，用锁定版本重建一次并记录。
10. 前端状态与 UI 修复：换图清理 registration/proposal/候选、历史恢复走 bridge、复核意见回填、撤销覆盖导入/清空、GUI 线程长任务挪到 aux。

**P2（持续改进）**
11. 存储增量 API + kind 索引 + schema 版本；前端按 job 版本号增量渲染。
12. `grouped_split` 接入 `benchmark.py`，并加感知哈希/相邻裁片去重；用固定划分产出可复现的 baseline 表。
13. 参考检索升级为感知相似度（CLIP/ORB + 元数据），把"同壁画"从人工文本变成可计算约束。
14. 抽共享 `configure_application`/掩膜/设置服务，删除或明确标注 legacy UI；清理 `web/`、`app.py` 依赖；exe 加版本信息/签名；日志轮转。

---

## 九、适用性判断

- **适合：** 教学/答辩/作品集展示；数字修复方向的方法预研；"约束式 AI 修复"的工程范式参考（运行时不变式、provenance、白名单规划器、诚实边界）。
- **谨慎：** 作为团队协作的代码基座——先修 P0 与加 git 历史；否则两套 UI 与重复实现会持续产生漂移。
- **不适合：** 任何真实文物修复决策、历史原貌推断、面向公众的"AI 复原"结论。项目自己也是这么说的，评估方同样不建议越界。

---

## 十、本次评估的证据与副作用

- 已运行：`pytest -q`（21 项，1 failed）、`scripts/verify_desktop.py`（成功但报告漂移）、环境/权重/便携包/数据集哈希核查、SAM2/LPIPS/分组划分函数级验证、WebEngine 离屏加载探针。
- `artifacts/` 下新增/更新了评审日志与探针（`review-pytest.log`、`review-verify-desktop.log`、`review_web_probe.py`、`review-probe/`、`review-web/`），并重新生成了 `desktop-verification.json` 与 `desktop-preview.png`（新值为 `telea/cpu`，旧值 `lama/cuda 3.81s` 仍见 `VALIDATION.md:9` 与 `native-full-verification.log`）。未修改任何源码、未触碰 `data/` 用户数据、未联网下载。
