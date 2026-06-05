# qusezhao

胶片负片自动去色罩桌面程序。项目目标是把彩色负片扫描图自动转换为接近 SP3000、Noritsu、Negative Lab Pro、NegRGB 等工作流的正片效果，同时保留类似 Lightroom 的手动调色能力。

当前仓库提交代码与立项文档；本地下载的数据集、训练输出、模型权重、实验记录和私人样片默认不提交到 GitHub。

## 当前状态

- 规则算法：橙色片基估计、负片转正、自动色阶、白平衡、色彩和锐化调节。
- 智能自动：尝试多组预设并根据图像统计选择稳定结果。
- AI 管线：包含成对数据清单、PyTorch 小模型、训练脚本和推理入口。
- GUI：支持批量队列、预览对比、自动/手动模式、AI checkpoint、profile 校准和批处理日志。
- CLI：支持单张转换、文件夹批量转换、参考样图校准和诊断 JSON。

## 项目结构

```text
.
├─ docs/                         # 立项文档、需求、技术方案、计划书、测试计划
├─ src/film_mask_automation/     # 程序源码
│  ├─ cli.py                     # 命令行入口
│  ├─ processor.py               # 规则去色罩核心算法
│  ├─ smart.py                   # 智能自动参数选择
│  ├─ profile.py                 # 负片+目标样图色彩 profile
│  ├─ gui/                       # PySide6 桌面 GUI
│  └─ ml/                        # PyTorch 训练和推理模块
├─ tools/                        # 数据集、训练清单、评估和实验脚本
├─ tests/                        # 自动测试
├─ pyproject.toml
└─ 启动图形界面.bat
```

## 安装

建议使用 Python 3.10 或更高版本。

```powershell
cd C:\Users\19831\Desktop\胶片自动去色罩自动化程序-立项文档
python -m pip install -e .
```

如果要训练 AI 模型，还需要安装可选依赖：

```powershell
python -m pip install -e ".[ml,dev]"
```

## 启动 GUI

双击项目根目录的：

```text
启动图形界面.bat
```

也可以运行：

```powershell
python -m film_mask_automation.gui.app
```

GUI 支持：

- 批量导入负片文件夹
- 原图/输出图预览对比
- 智能自动去色罩
- 手动调整色罩、黑白点、曝光、亮度、伽马、对比度、饱和度、色温、色调、RGB 通道和锐化
- AI 模型 checkpoint 选择
- 负片 + 目标样图 profile 校准
- 批处理输出和日志

## 命令行使用

单张转换：

```powershell
python -m film_mask_automation.cli convert input_negative.jpg output_positive.jpg --smart-auto
```

如果负片边缘有清楚的未曝光片基，可以使用边框估计：

```powershell
python -m film_mask_automation.cli convert input_negative.jpg output_positive.jpg --mask-source border
```

手动指定片基色：

```powershell
python -m film_mask_automation.cli convert input_negative.jpg output_positive.jpg --mask-source manual --mask-rgb 230,145,75
```

批量处理：

```powershell
python -m film_mask_automation.cli batch input_folder output_folder --smart-auto --diagnostics-json output_folder\diagnostics.json
```

## 用参考样图校准

如果有同一张照片的“负片图 + 正常去色罩目标图”，可以生成 profile：

```powershell
python -m film_mask_automation.cli calibrate input_negative.jpg reference_positive.jpg profiles\film_profile.json --preview-output preview.jpg --smart-auto
```

之后复用这个 profile：

```powershell
python -m film_mask_automation.cli convert input_negative.jpg output_positive.jpg --profile profiles\film_profile.json
```

## AI 模型训练

训练数据应该是成对图片：

- 输入：真实负片扫描图，保留橙色片基和反相颜色。
- 目标：同一张照片去色罩后的正片结果。

项目包含公开样例下载、BlueNeg 原始 DNG 渲染、训练清单生成、实验记录和结果打包脚本。数据量较大，且有版权和体积限制，所以不随 Git 仓库提交。

常用流程：

```powershell
python tools\expand_blueneg_raw_rendered.py --count 60
python tools\orient_blueneg_rendered.py
python tools\build_mixed_true_negative_dataset.py
python -m film_mask_automation.ml.train datasets\training_manifest.csv models\film_mask_tiny_local.pt --epochs 2 --steps-per-epoch 8 --batch-size 2 --crop-size 128 --base-channels 16 --device cpu
python tools\create_experiment_run.py
```

说明：当前小模型只用于验证训练和集成流程，不代表最终画质。最终模型需要更多真实负片/正片对、按胶片类型和扫描仪来源分层测试，并持续记录每次实验输出。

注意：AI 模型需要 PyTorch。命令行使用 AI 时优先运行 `.venv-ml\Scripts\python.exe`；根目录的 `启动图形界面.bat` 会优先使用 `.venv-ml`，这样 GUI 才能直接调用最新模型权重。

## 可下载训练包

仓库包含 `release_assets/`，用于让别人 clone 后直接使用当前训练集、测试集和模型权重：

- `release_assets/dataset/manifest.csv`
- `release_assets/dataset/train_manifest.csv`
- `release_assets/dataset/test_manifest.csv`
- `release_assets/models/film_mask_tiny_mixed_true_negative.pt`
- `release_assets/experiments/latest/03_model_outputs.jpg`

二进制图片和模型权重通过 Git LFS 管理。首次下载建议安装 Git LFS 后执行：

```powershell
git lfs install
git clone https://github.com/Xxx-147/qusezhao.git
cd qusezhao
git lfs pull
```

## 测试

```powershell
python -m pytest
```

## 同步到 GitHub

项目已连接到：

```text
https://github.com/Xxx-147/qusezhao.git
```

每次阶段性更新后可以运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\git_sync.ps1 "chore: describe this update"
```

也可以双击根目录的：

```text
自动同步到GitHub.bat
```

这个脚本会先运行测试，再提交非忽略文件，最后推送到 GitHub。数据集、模型、实验输出和私人图片已经被 `.gitignore` 排除，不会随同步上传。

如果想让项目在本机持续自动同步，可以打开：

```text
启动自动GitHub同步监听.bat
```

它会每 60 秒检查一次 Git 状态；发现非忽略变更后，会调用同一个测试、提交、推送流程。窗口关闭后监听停止。

## 文档

- [项目总览](docs/00_项目总览.md)
- [需求规格说明书](docs/01_需求规格说明书.md)
- [技术方案](docs/02_技术方案.md)
- [详细实施计划书](docs/03_详细实施计划书.md)
- [测试计划](docs/04_测试计划.md)
- [公开样例与行业参数研究](docs/05_公开样例与行业参数研究.md)
- [AI 模型训练计划](docs/06_AI模型训练计划.md)
- [项目进度记录](docs/07_项目进度记录.md)
