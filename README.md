# Kimodo 安装

> [!IMPORTANT]
> 本仓库是 NVIDIA Kimodo 的**非官方衍生分支**，不隶属于 NVIDIA，也不代表 NVIDIA 官方发布、认可、担保或支持。
>
> 项目来源：
>
> - 官方上游：[`nv-tlabs/kimodo`](https://github.com/nv-tlabs/kimodo)
> - 基础分支：[`Aero-Ex/kimodo`](https://github.com/Aero-Ex/kimodo)
> - 当前分支：[`Ayn1631/kimodo`](https://github.com/Ayn1631/kimodo)

本分支主要用于补充 Windows、低显存、本地 NF4 LLM2Vec 文本编码器和 `kimodo-viser` 构建兼容性。

---

## 使用说明

以下“主流程”保留原有步骤、命令、顺序和参数，不进行改写。

补充内容仅包括：

- 每一步的作用说明。
- 安装后的验证方法。
- 已遇到的常见错误与原因。
- 不改变主流程的排查命令。

本教程中的主流程以 **Windows CMD** 为准。

---

## 步骤

### 1. 创建虚拟环境

```cmd
python -m pip install -U uv  # venv无法指定python版本...
uv venv venv --python 3.12
.\venv\Scripts\activate
python -m ensurepip --upgrade --default-pip
```

#### 说明

- `python -m pip install -U uv`：通过当前 Python 安装或升级 `uv`。
- `uv venv venv --python 3.12`：创建名为 `venv` 的 Python 3.12 虚拟环境。
- `.\venv\Scripts\activate`：在 Windows CMD 中激活虚拟环境。
- `python -m ensurepip --upgrade --default-pip`：为虚拟环境安装并补齐 `pip`、`pip3` 和对应版本的 pip 启动入口。

`python -m venv` 本身不能通过类似下面的参数指定 Python 版本：

```cmd
python -m venv venv -p python=3.12
```

因此这里使用 `uv` 创建指定 Python 3.12 的虚拟环境。

#### 验证

```cmd
where python
python --version
python -m pip -V
```

预期：

- `where python` 第一项指向 `venv\Scripts\python.exe`。
- `python --version` 输出 `Python 3.12.x`。
- `python -m pip -V` 的路径位于 `venv\Lib\site-packages`。

#### 常见错误

##### `python -3.12 -m venv venv`

错误：

```text
Unknown option: -3
```

原因：`-3.12` 是 `py` 启动器的参数，不是 `python.exe` 的参数。

##### `python -m venv venv -p python=3.12`

错误：

```text
venv: error: unrecognized arguments: -p python=3.12
```

原因：标准库 `venv` 没有 `-p` 参数。

##### `pip` 指向 Anaconda

检查：

```cmd
where pip
where python
python -m pip -V
```

如果裸 `pip` 指向 Anaconda，但 `python -m pip -V` 指向当前 `venv`，安装时优先使用：

```cmd
python -m pip
```

它会强制使用当前 Python 对应的 pip。

##### 虚拟环境只有 `pip3.exe`

检查：

```cmd
dir venv\Scripts\pip*
```

若只有：

```text
pip3.exe
pip3.12.exe
```

主流程中的以下命令会尝试补齐 `pip.exe`：

```cmd
python -m ensurepip --upgrade --default-pip
```

##### 同时显示 `(venv) (base)`

说明 venv 与 Conda base 同时激活。检查实际解释器：

```cmd
where python
python -c "import sys; print(sys.executable)"
```

第一项与 `sys.executable` 应指向当前 `venv`。

---

### 2. 下载pytorch

```cmd
# 登录pytorch [`https://pytorch.org/`] 选择下载合适的torch版本
# 国内源不可用!!!
# 需要魔法!
# CUDA 12.x：驱动至少 525
# CUDA 13.x：驱动至少 580
nvidia-smi
```

#### 说明

`nvidia-smi` 用于检查：

- NVIDIA 显卡是否被驱动正常识别。
- 当前显卡驱动版本。
- 驱动支持的最高 CUDA 版本。
- 显存占用和运行进程。

例如：

```text
Driver Version: 591.74
CUDA Version: 13.1
```

这里的 `CUDA Version: 13.1` 表示驱动最高支持 CUDA 13.1，不表示已经安装 CUDA Toolkit 13.1，也不要求必须下载完全相同版本的 PyTorch。

普通使用 PyTorch CUDA Wheel 时：

- 必须有兼容的 NVIDIA 显卡驱动。
- 通常不需要单独安装完整 CUDA Toolkit。
- PyTorch CUDA Wheel 自带运行所需的 CUDA Runtime。
- 只有编译 CUDA 扩展时通常才需要 `nvcc` 和 CUDA Toolkit。

#### 验证 PyTorch

安装完成后执行：

```cmd
python -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('可用:',torch.cuda.is_available()); print('显卡:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用')"
```

正常情况下：

```text
可用: True
```

并显示 NVIDIA 显卡名称。

#### 常见错误

##### `torchvision` 显示 `+cpu`

例如：

```text
torchvision 0.21.0+cpu
```

表示安装的是 CPU 版本 torchvision。

应确保 `torch`、`torchvision` 和 `torchaudio`：

- 版本互相匹配。
- 来自同一个 PyTorch CUDA 索引。
- 都安装在当前虚拟环境中。

##### `Requirement already satisfied` 指向 Anaconda

例如：

```text
D:\Mypower\Anaconda3\Lib\site-packages
```

说明安装命令使用了其他环境的 pip。

检查：

```cmd
where python
python -m pip -V
```

安装时使用当前环境的：

```cmd
python -m pip
```

##### NumPy 依赖冲突

例如：

```text
numpy 2.4.4 is incompatible
```

通常说明包被安装到了已有大量旧依赖的环境中，而不是干净的项目虚拟环境。

先确认：

```cmd
python -m pip -V
```

路径必须位于当前 `venv`。

##### 没有 `nvidia-smi`

可能原因：

- 没有 NVIDIA 显卡。
- NVIDIA 驱动未安装。
- 驱动安装损坏。
- `nvidia-smi.exe` 未加入 PATH。

检查显卡：

```cmd
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,Status"
```

---

### 3. 拉取文本编码器模型

```cmd
python -m pip install --upgrade huggingface_hub

# This will download the model to a folder named './KIMODO-Meta3_llm2vec_NF4'
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aero-Ex/KIMODO-Meta3_llm2vec_NF4', local_dir='./KIMODO-Meta3_llm2vec_NF4')"
```

#### 说明

该模型是：

```text
Aero-Ex/KIMODO-Meta3_llm2vec_NF4
```

它是基于 Llama 3 8B Instruct 的 LLM2Vec NF4 文本编码器，用于：

```text
文本 → 4096 维语义向量 → Kimodo 动作条件
```

它不是普通对话模型，不能直接用于生成聊天回复。

下载完成后，模型会位于当前目录下：

```text
KIMODO-Meta3_llm2vec_NF4
```

#### 验证

```cmd
dir ".\KIMODO-Meta3_llm2vec_NF4"
dir ".\KIMODO-Meta3_llm2vec_NF4\config.json"
dir ".\KIMODO-Meta3_llm2vec_NF4\model.safetensors"
```

也可以执行：

```cmd
python -c "from pathlib import Path; p=Path(r'.\KIMODO-Meta3_llm2vec_NF4'); print('目录:',p.is_dir()); print('config:',(p/'config.json').is_file()); print('权重:',(p/'model.safetensors').is_file())"
```

#### 常见错误

##### 是否需要 Hugging Face 密钥

该仓库公开时可以匿名下载，不需要密钥。

若出现 `401`、`403` 或访问限制，再执行：

```cmd
hf auth login
```

##### 本地路径被识别为 Hugging Face Repo ID

错误类似：

```text
HFValidationError: Repo id must use alphanumeric chars...
```

这通常不代表 Windows 路径格式本身有问题，而是程序检查本地路径时发现目录不存在，于是把该字符串继续当作 Hugging Face 仓库 ID。

检查 `.env` 中的路径是否真实存在，并确认目录中有：

```text
config.json
model.safetensors
tokenizer.json
```

##### 显存不足

NF4 模型仍可能需要约 5 GB 级别显存。4 GB 显卡建议让文本编码器使用 CPU，并在启动 Demo 时使用卸载模式。

---

### 4. 拉取仓库

```cmd
git clone https://github.com/Ayn1631/kimodo.git
cd kimodo

git clone https://github.com/nv-tlabs/kimodo-viser.git
python -c "import shutil; shutil.copy2(r'replace\_client_autobuild.py', r'kimodo-viser\src\viser\_client_autobuild.py')"
pip install -e kimodo-viser
set SKIP_MOTION_CORRECTION_IN_SETUP=1 && pip install -e .
```

#### 说明

这一部分依次完成：

1. 下载当前 Kimodo 分支。
2. 进入 Kimodo 项目目录。
3. 下载官方 `kimodo-viser`。
4. 使用 `replace\_client_autobuild.py` 覆盖 `kimodo-viser` 中对应文件。
5. 以 editable 模式安装 `kimodo-viser`。
6. 跳过 MotionCorrection 源码构建并安装 Kimodo 主项目。

替换 `_client_autobuild.py` 的原因是：

- Windows 下实际可执行入口通常是 `npx.cmd`。
- Unix 风格的 `npx` 文件不能被 Windows 直接作为原生可执行文件运行。
- PowerShell 或 CMD 输入 `npx` 时会通过 PATH 和 PATHEXT 自动找到 `npx.cmd`，但 Python 使用明确路径时不会自动完成同样的解析。

#### 验证

```cmd
python -c "import kimodo; print(kimodo.__file__)"
python -c "import viser; print(viser.__file__)"
```

路径应指向当前项目目录或当前虚拟环境。

#### 常见错误

##### PowerShell 不支持 `&&`

错误：

```text
标记“&&”不是此版本中的有效语句分隔符
```

主流程是 CMD 命令。若当前提示符以 `PS` 开头，说明正在使用 PowerShell。

##### PowerShell 中 `copy /Y` 报错

PowerShell 中的 `copy` 实际是 `Copy-Item` 别名，不支持 CMD 的 `/Y` 参数。

本主流程中的复制已经通过 Python `shutil.copy2()` 完成，不需要改动。

##### `cp` 不是内部或外部命令

`cp` 是 Linux/macOS Shell 命令，不是 Windows CMD 命令。

##### `../venv/Scripts/activate` 无法执行

这是 Linux 风格路径和命令。Windows CMD 使用 `\`，并执行 Windows 激活脚本。

##### 找不到 `npx`

检查：

```cmd
where npx
where npm
node -p "process.arch"
```

Windows 环境下通常应找到：

```text
npx.cmd
npm.cmd
```

##### `pip install -e kimodo-viser` 安装到错误环境

检查：

```cmd
where python
where pip
python -m pip -V
```

如果裸 `pip` 指向其他环境，应先修复虚拟环境中的 pip 启动器。主流程保持不变。

##### MotionCorrection 安装时要求 CMake

主流程使用：

```cmd
set SKIP_MOTION_CORRECTION_IN_SETUP=1 && pip install -e .
```

因此安装 Kimodo 主项目时会跳过 MotionCorrection，之后通过预编译 wheel 单独安装。

---

### 5. 下载预编译版motion_correction

```cmd
pip install https://github.com/Aero-Ex/kimodo/releases/download/v1.0.0/motion_correction-1.0.0-cp312-cp312-win_amd64.whl
```

#### 说明

Wheel 文件名：

```text
motion_correction-1.0.0-cp312-cp312-win_amd64.whl
```

含义：

- `cp312`：CPython 3.12。
- 第二个 `cp312`：CPython 3.12 ABI。
- `win_amd64`：Windows 64 位平台。

该预编译 wheel 不能安装到：

- Python 3.11。
- Python 3.13。
- 32 位 Python。
- Linux 容器。
- macOS。

#### 验证

```cmd
python -c "import motion_correction; print(motion_correction.__file__)"
```

#### 常见错误

##### `not a supported wheel on this platform`

检查：

```cmd
python --version
python -c "import sys,platform,struct; print(sys.executable); print(platform.machine()); print(struct.calcsize('P')*8)"
```

需要：

```text
Python 3.12.x
AMD64 / x86_64
64
```

##### 从源码安装时提示 CMake

错误：

```text
RuntimeError: CMake must be installed to build this package
```

这是源码构建路径的错误。当前主流程使用预编译 wheel，不需要修改主流程为源码编译。

##### 运行时提示未安装 MotionCorrection

错误：

```text
Motion correction is required for this postprocessing path but the motion_correction package is not installed.
```

检查导入：

```cmd
python -c "import motion_correction; print(motion_correction.__file__)"
```

并确认安装 wheel 的 pip 属于当前虚拟环境。

---

### 6. 下载其余依赖

```cmd
pip install bitsandbytes
pip install -U transformers==5.1.0
pip install dotenv
```

#### 说明

- `bitsandbytes`：用于加载 NF4 4-bit 量化模型。
- `transformers==5.1.0`：提供模型、Tokenizer 和 Hugging Face 加载接口。
- `dotenv`：用于读取 `.env` 配置。

#### 验证

```cmd
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import dotenv; print(dotenv.__file__)"
```

#### 常见错误

##### bitsandbytes 无法识别 GPU

先验证 PyTorch：

```cmd
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果 PyTorch 本身是 CPU 版，bitsandbytes 也无法正常使用 NVIDIA CUDA。

##### 依赖冲突

如果输出大量 NumPy、Pillow、protobuf 或 SciPy 冲突，检查安装位置：

```cmd
python -m pip -V
```

路径应属于当前 `venv`，不应指向 Anaconda base。

##### pip 缓存权限错误

错误类似：

```text
Permission denied: C:\Users\...\pip\cache\wheels\...
```

可先检查是否有其他 Python、pip、IDE 或杀毒软件正在占用缓存文件。

排查时可使用：

```cmd
python -m pip cache dir
python -m pip cache purge
```

---

### 7. 修改配置文件

```cmd
copy /Y ".venv.example" ".env"
# 修改.env文件里的参数和地址
```

#### 说明

该命令将示例配置复制为运行时配置：

```text
.venv.example → .env
```

`.env` 中需要根据实际模型位置设置参数，例如：

```env
CPU_Load=True
LLM2Vec_dir=D:\Test\KIMODO-Meta3_llm2vec_NF4
```

说明：

- `CPU_Load=True`：在 CPU 上加载 LLM2Vec，减少显存占用。
- `CPU_Load=False`：优先使用可用的 GPU 设备。
- `LLM2Vec_dir`：本地 NF4 LLM2Vec 模型目录。

#### 验证 `.env`

```cmd
type .env
```

验证 Python 是否能读取：

```cmd
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('CPU_Load=',os.getenv('CPU_Load')); print('LLM2Vec_dir=',os.getenv('LLM2Vec_dir'))"
```

验证模型路径：

```cmd
python -c "from dotenv import load_dotenv; from pathlib import Path; import os; load_dotenv(); p=Path(os.environ['LLM2Vec_dir']); print('目录:',p.is_dir()); print('config:',(p/'config.json').is_file())"
```

#### 常见错误

##### 把配置写进 `.venv`

`.venv` 或 `venv` 通常是虚拟环境目录，不是 `.env` 配置文件。

正确配置文件名是：

```text
.env
```

##### `.env` 变量没有被读取

只有代码调用 `load_dotenv()` 并通过 `os.getenv()` 或 `os.environ` 读取时，`.env` 才会生效。

检查：

```cmd
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('LLM2Vec_dir'))"
```

##### 仍然读取默认模型目录

如果日志仍显示：

```text
项目目录\models\KIMODO-Meta3_llm2vec_NF4
```

说明：

- `.env` 没有加载。
- 变量名与代码读取的变量名不一致。
- 启动目录不是 `.env` 所在目录。
- `.env` 中配置的路径不存在。
- 代码仍使用默认值。

##### `HFValidationError`

如果模型路径不存在，Transformers 会把本地 Windows 路径误当作 Hugging Face Repo ID。

必须确保：

```cmd
dir "%LLM2Vec_dir%\config.json"
```

能够找到文件。

---

### 9. 启动demo

```cmd
python -m kimodo.demo
```

#### 说明

该命令会：

1. 初始化 Kimodo Demo。
2. 加载默认动作生成模型。
3. 尝试连接文本编码器服务。
4. 如果服务不可达，则回退到本地 LLM2Vec 文本编码器。
5. 启动 Web Demo。

#### 验证

正常日志可能包含：

```text
Using device: cuda:0
Model kimodo-soma-rp loaded successfully
```

若文本编码器服务未启动，可能出现：

```text
Text encoder service is unreachable, falling back to local LLM2Vec encoder.
```

这是回退提示，不一定是致命错误。

#### 常见错误

##### Triton 未安装

提示：

```text
triton not found; flop counting will not work for triton kernels
```

该提示主要表示无法统计 Triton 内核 FLOP，通常不阻止模型运行。

##### 文本编码器服务无法连接

提示：

```text
Could not fetch config for http://127.0.0.1:9550/
```

程序会尝试回退到本地 LLM2Vec 编码器。只要本地模型目录配置正确，可以继续运行。

##### 本地模型路径报 `HFValidationError`

检查：

```cmd
python -c "from dotenv import load_dotenv; from pathlib import Path; import os; load_dotenv(); p=Path(os.environ['LLM2Vec_dir']); print(p); print(p.is_dir()); print((p/'config.json').is_file())"
```

路径与 `config.json` 都必须存在。

##### 显存不足

日志可能出现：

```text
VRAM tight
Offloading others
```

RTX 3050 4 GB 运行 NF4 文本编码器时仍可能显存不足。

可以使用：

```cmd
python -m kimodo.demo --offload
```

这是一条额外的低显存启动测试命令，不替换原主流程中的：

```cmd
python -m kimodo.demo
```

同时可以在 `.env` 中使用：

```env
CPU_Load=True
```

##### 内存不足

模型卸载到系统内存后，可能导致 RAM 占用明显上升。16 GB 内存环境应关闭浏览器大量标签、IDE、其他模型进程和不必要的后台程序。

---

## 许可证与来源

本仓库中的代码沿用 [Apache License 2.0](LICENSE)。

本仓库是 NVIDIA Kimodo 的非官方衍生分支，基于 Aero-Ex 的相关修改继续开发。

使用和分发本仓库代码时，应：

- 保留 Apache License 2.0 许可证。
- 保留适用的版权和归属声明。
- 保留第三方许可证。
- 在修改过的上游文件中醒目标明文件已修改。
- 如果上游包含适用的 `NOTICE`，应继续保留。

第三方归属信息见：

```text
ATTRIBUTIONS.MD
```

> [!WARNING]
> 仓库代码的 Apache License 2.0 不自动适用于模型权重、数据集、Meta Llama、Aero-Ex NF4 文本编码器、Kimodo 模型检查点或其他第三方资产。使用与再分发前，应分别检查各资源的许可证。

## 商标与非官方声明

Apache License 2.0 不授予 NVIDIA、Kimodo、Aero-Ex、McGill NLP 或其他权利人的商标使用权。

本仓库提及这些名称，仅用于说明来源、兼容关系和依赖关系，不表示任何上游组织或作者对本分支进行认可、担保或支持。

## 致谢

感谢：

- NVIDIA Kimodo。
- Aero-Ex 的 Windows 与低显存适配工作。
- McGill NLP 的 LLM2Vec。
- NVIDIA `kimodo-viser`。
- `ATTRIBUTIONS.MD` 中列出的其他第三方项目。
