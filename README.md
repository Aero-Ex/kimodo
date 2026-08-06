# Kimodo Windows / Low-VRAM Compatibility Fork

> [!IMPORTANT]
> 本仓库是 NVIDIA Kimodo 的**非官方衍生分支**，不隶属于 NVIDIA，也不代表 NVIDIA 官方发布、认可或支持。
>
> 项目来源关系：
>
> - 官方上游：[nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo)
> - 基础分支：[Aero-Ex/kimodo](https://github.com/Aero-Ex/kimodo)
> - 当前分支：[Ayn1631/kimodo](https://github.com/Ayn1631/kimodo)

本分支面向 Windows 与低显存设备，保留 Kimodo 原有的动作生成能力，并补充本地 NF4 LLM2Vec 文本编码器、CPU 文本编码、显存卸载和 Windows 客户端构建兼容。

## 本分支的主要修改

- 支持通过 `.env` 指定本地 LLM2Vec 文本编码器目录。
- 支持将文本编码器强制加载到 CPU，以降低显存占用。
- 改进 CUDA、MPS、CPU 和多 GPU 场景下的设备判断。
- 改进文本编码器的加载、卸载、迁移和重新加载流程。
- 改进低显存模式下的内存与显存清理。
- 为 Windows 修复 `kimodo-viser` 构建时的 `npx.cmd` 解析问题。
- 提供适用于 Windows CMD 的安装流程。

## 与官方 Kimodo 的关系

Kimodo 是 NVIDIA 发布的可控人体与机器人运动扩散模型，可通过文本提示和运动学约束生成动作。

本仓库只维护 Windows、低显存和本地文本编码器相关的兼容改动。Kimodo 模型结构、官方模型权重、数据集、论文与品牌归其各自权利人所有。

## 环境要求

推荐环境：

- Windows 10 或 Windows 11，64 位。
- Python 3.12，预编译的 `motion_correction` wheel 仅适配 CPython 3.12 x64。
- NVIDIA 显卡与正常工作的 NVIDIA 驱动。
- Git。
- 可访问 GitHub、Hugging Face 和 PyTorch 官方下载源的网络。
- 低显存设备建议启用 CPU 文本编码和 `--offload`。

检查 NVIDIA 驱动：

```cmd
nvidia-smi
```

`nvidia-smi` 中的 `CUDA Version` 表示驱动支持的最高 CUDA 版本，不表示本机已经安装了同版本 CUDA Toolkit，也不要求 PyTorch 必须安装完全相同的 CUDA 构建。

## Windows CMD 安装

以下命令默认在 **CMD** 中执行。PowerShell 的环境变量与虚拟环境激活语法不同。

### 1. 克隆仓库

```cmd
git clone https://github.com/Ayn1631/kimodo.git
cd kimodo
```

### 2. 创建 Python 3.12 虚拟环境

```cmd
python -m pip install -U uv
uv venv .venv --python 3.12 --seed
call .venv\Scripts\activate.bat
python -m pip install -U pip setuptools wheel
```

确认当前解释器和 pip 都属于该虚拟环境：

```cmd
where python
python -m pip -V
```

输出路径应位于当前仓库的 `.venv` 目录中。

### 3. 安装 CUDA 版 PyTorch

请在 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 中选择：

- OS：Windows
- Package：Pip
- Language：Python
- Compute Platform：与当前 NVIDIA 驱动兼容的 CUDA 版本

然后使用页面生成的命令安装。

示例，使用 PyTorch 2.6.0 与 CUDA 12.6：

```cmd
python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

验证：

```cmd
python -c "import torch; print('Torch:', torch.__version__); print('CUDA Runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

### 4. 下载 NF4 LLM2Vec 文本编码器

```cmd
python -m pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aero-Ex/KIMODO-Meta3_llm2vec_NF4', local_dir='./models/KIMODO-Meta3_llm2vec_NF4')"
```

下载后确认关键文件存在：

```cmd
dir ".\models\KIMODO-Meta3_llm2vec_NF4\config.json"
dir ".\models\KIMODO-Meta3_llm2vec_NF4\model.safetensors"
```

如果本地目录不存在，Transformers 可能会把 Windows 路径误当成 Hugging Face 仓库 ID，并报 `HFValidationError`。

### 5. 安装 Windows 兼容版 kimodo-viser

```cmd
git clone https://github.com/nv-tlabs/kimodo-viser.git
python -c "import shutil; shutil.copy2(r'replace\_client_autobuild.py', r'kimodo-viser\src\viser\_client_autobuild.py')"
python -m pip install -e .\kimodo-viser
```

替换文件的主要作用是在 Windows 上显式调用 `npx.cmd`，而不是 Unix 风格的 `npx` 脚本。

### 6. 安装 Kimodo 主项目

先跳过源码编译版 MotionCorrection：

```cmd
set SKIP_MOTION_CORRECTION_IN_SETUP=1 && python -m pip install -e .
```

安装额外运行依赖：

```cmd
python -m pip install -U bitsandbytes python-dotenv
python -m pip install -U transformers==5.1.0
```

### 7. 安装预编译 MotionCorrection

```cmd
python -m pip install "https://github.com/Aero-Ex/kimodo/releases/download/v1.0.0/motion_correction-1.0.0-cp312-cp312-win_amd64.whl"
```

该 wheel 需要：

- CPython 3.12。
- Windows x64。
- 与当前环境兼容的 PyTorch。

如果出现 `not a supported wheel on this platform`，先检查：

```cmd
python --version
python -c "import sys, platform, struct; print(sys.executable); print(platform.machine()); print(struct.calcsize('P') * 8)"
```

### 8. 创建配置文件

```cmd
copy /Y ".env.example" ".env"
```

编辑 `.env`：

```env
CPU_Load=True
LLM2Vec_dir=D:/绝对路径/kimodo/models/KIMODO-Meta3_llm2vec_NF4
```

建议在 Windows 路径中使用 `/`，或使用完整原始字符串语义对应的路径，避免反斜杠转义与复制错误。

参数说明：

- `CPU_Load=True`：将文本编码器加载到 CPU，显著降低显存占用，但文本编码会更慢。
- `CPU_Load=False`：优先使用 MPS 或 CUDA。
- `LLM2Vec_dir`：本地 NF4 LLM2Vec 模型目录，目录中必须存在 `config.json` 等模型文件。

### 9. 启动 Demo

普通启动：

```cmd
python -m kimodo.demo
```

低显存启动：

```cmd
python -m kimodo.demo --offload
```

## PowerShell 对照

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

设置一次性环境变量并安装：

```powershell
$env:SKIP_MOTION_CORRECTION_IN_SETUP="1"; python -m pip install -e .
```

复制配置：

```powershell
Copy-Item ".env.example" ".env" -Force
```

## 常见问题

### `pip` 安装到了 Anaconda 或其他环境

不要裸用 `pip` 或 `pip3`，始终使用：

```cmd
python -m pip install 包名
```

检查：

```cmd
where python
where pip
python -m pip -V
```

真正决定安装位置的是 `python -m pip` 中的 `python`。

### 本地模型路径被当成 Hugging Face 仓库 ID

典型报错：

```text
HFValidationError: Repo id must use alphanumeric chars...
```

检查：

```cmd
python -c "from pathlib import Path; p=Path(r'D:\你的路径\KIMODO-Meta3_llm2vec_NF4'); print(p.is_dir()); print((p/'config.json').is_file())"
```

两个结果都必须为 `True`。

### `npx` 无法执行

Windows 中 npm 通常提供的是：

```text
npx.cmd
```

而不是可直接执行的 Unix `npx` 脚本。本分支通过 `replace/_client_autobuild.py` 修复该问题。

### 显存不足

NF4 文本编码器仍可能占用约 5 GB 级别的显存。4 GB 显卡建议：

```env
CPU_Load=True
```

并使用：

```cmd
python -m kimodo.demo --offload
```

Docker 和虚拟环境只能隔离依赖，不能增加物理显存。

## 修改声明

本仓库相对上游的主要修改文件包括：

- `.gitignore`
- `README.md`
- `kimodo/demo/__init__.py`
- `kimodo/demo/app.py`
- `kimodo/model/llm2vec/llm2vec.py`
- `kimodo/model/llm2vec/llm2vec_wrapper.py`
- `replace/_client_autobuild.py`
- `pyproject.toml`

被修改的上游源文件应保留原始版权与许可证声明，并包含醒目的修改说明。

## 许可证

仓库代码沿用 Apache License 2.0。完整条款见 [`LICENSE`](LICENSE)。

Apache 2.0 允许使用、修改、分发和商业集成，但分发衍生版本时需要：

- 向接收者提供 Apache 2.0 许可证。
- 保留适用的版权、专利、商标与归属声明。
- 在修改过的文件中醒目标明该文件已被修改。
- 保留上游项目提供且仍适用的 `NOTICE` 内容，如果存在。

第三方代码归属与许可证见 [`ATTRIBUTIONS.MD`](ATTRIBUTIONS.MD)。

> [!WARNING]
> 仓库代码的 Apache 2.0 许可证不自动覆盖模型权重、数据集、Meta Llama、Aero-Ex NF4 文本编码器、Kimodo 模型检查点或其他第三方资产。使用和再分发前必须分别检查各自许可条款。

## 商标与非官方声明

Apache 2.0 不授予 NVIDIA、Kimodo 或其他权利人的商标使用权。

本仓库使用项目名称仅用于说明兼容关系与来源，不表示 NVIDIA、Aero-Ex、McGill NLP 或任何其他上游作者对本分支进行认可、担保或支持。

## 致谢

感谢以下项目及其贡献者：

- NVIDIA Kimodo。
- Aero-Ex 的 Windows 与低显存适配工作。
- McGill NLP 的 LLM2Vec。
- NVIDIA kimodo-viser。
- 仓库 `ATTRIBUTIONS.MD` 中列出的其他第三方项目。

## 贡献

提交改动时，请：

1. 保留原始许可证与版权头。
2. 对修改过的上游文件添加醒目的修改声明。
3. 不要把代码许可证错误地描述为模型或数据许可证。
4. 不要将本分支描述为 NVIDIA 官方实现或官方发行版。
