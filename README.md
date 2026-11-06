# 拾壁：壁画数字修复工作室

拾壁是一款 Windows 桌面应用，用于检查壁画图像中的疑似缺损、比较数字修补方案并保存可追溯的修复记录。它使用 PySide6 和 Qt WebEngine 显示本地界面，图像、模型推理和工作区数据均在本机处理；不会启动浏览器网页服务。

## 功能

- 导入壁画图像，并用画笔或辅助分割工具选择待处理区域。
- 使用本地视觉模型分析疑似缺损、类别、范围和可见文化线索；分析结果需要人工核验。
- 为每条已确认缺损选择建议方法，或单独指定 LaMa、Telea、Navier–Stokes、局部色调匹配，再逐项或批量运行。
- 对照原图、修复候选和差异图，复核后保存图像或导出 ZIP 档案。
- 保存历史任务，支持查看、续跑、参考资料配准和工作空间备份。

## 安装

要求 Windows 10/11 x64 和 Python 3.10 或更高版本。克隆仓库后，在项目目录运行：

```powershell
.\start.cmd
```

也可以双击项目目录中的 `start.cmd`。首次启动时，脚本会创建 `.venv` 并安装桌面运行依赖。手动安装和启动：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.cmd
```

基础图像处理可使用 CPU。LaMa、缺损识别、SAM 和扩散模型需要各自的权重；模型权重不随 Git 仓库分发。LaMa 权重可单独下载：

```powershell
.\.venv\Scripts\python.exe scripts/download_lama.py
```

本地视觉模型和可选分割、扩散模型的安装方式见 [使用说明](使用说明.md#可选模型安装) 与 [模型下载脚本](scripts/download_models.py)。NVIDIA GPU 用户可按本机驱动环境安装 CUDA 版 PyTorch，参考 `scripts/install_gpu.ps1` 和 [requirements-models.txt](requirements-models.txt)。

## 使用

1. 打开应用并载入一幅壁画图像。
2. 点击“AI 分析缺损与文化线索”。本地缺损识别需要安装 Qwen3-VL 权重。
3. 检查每个候选的位置、类别和风险，确认可修复项并选择修补方法。
4. 逐项运行，或运行全部已确认项目。
5. 对比结果与差异，复核后保存图像或导出档案。

高级设置和 API 密钥不是默认流程的必需项。缺损识别和文化观察是模型建议，不是病害鉴定；大面积语义缺失可能需要参考资料或专家判断。数字补全不能证明壁画的历史原貌，也不应用作真实文物施工依据。

完整操作说明、常见问题和边界说明见 [使用说明](使用说明.md)。工程验证信息见 [VALIDATION.md](VALIDATION.md)。

## API 规划（可选）

应用可调用支持图像输入的视觉 API 生成修复规划。配置入口位于应用的“API 设置”；密钥保存在本地 `.env` 文件中，该文件不会提交到 Git。API 规划是可选项，本地规则规划不需要密钥。图像发送范围与接口格式说明见使用说明中的 [API 设置](使用说明.md#10-api-设置怎么填)。

## 开发

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_desktop.py
```

测试包含本地业务逻辑、API 开发适配器和 Qt WebEngine 桌面桥接。完整依赖清单见 [requirements.txt](requirements.txt)、[requirements-models.txt](requirements-models.txt) 和 [requirements-dev.txt](requirements-dev.txt)。

## 目录

```text
mural_agent/       修复引擎、桌面桥接、图像处理与工作空间
ui/                随桌面应用加载的 HTML、CSS、JavaScript 界面
图片/              带来源与许可说明的示例壁画图像
datasets/          示例数据清单与公开图像测试集
scripts/           启动构建、模型下载和验证工具
tests/             自动化回归测试
assets/            桌面应用图标
```

任务数据库、导入图像、修复结果和模型权重保存在本地 `data/` 与 `models/` 目录中，不包含在仓库内。

## 许可与图像来源

仓库未提供代码许可证；公开可见不代表获得再分发或商用许可。示例壁画图像各自的来源和许可见 [`图片/来源与许可.md`](图片/来源与许可.md) 及 `datasets/public-fresco-smoke/` 中的元数据。不同图像和模型权重可能有不同许可，使用前请核对其来源说明和模型卡。
