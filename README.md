# Kimodo安装

## 步骤

1. 创建虚拟环境
```cmd
python -3.12 -m venv venv
.\venv\Scripts\activate
```

2. 下载pytorch  
```cmd
# 登录pytorch [`https://pytorch.org/`] 选择下载合适的torch版本
# 国内源不可用!!!
# 需要魔法!
# CUDA 12.x：驱动至少 525
# CUDA 13.x：驱动至少 580
nvidia-smi
```


3. 拉取文本编码器模型
```cmd
pip install --upgrade huggingface_hub

# This will download the model to a folder named './KIMODO-Meta3_llm2vec_NF4'
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aero-Ex/KIMODO-Meta3_llm2vec_NF4', local_dir='./KIMODO-Meta3_llm2vec_NF4')"
```


4. 拉取仓库
```cmd
git clone https://github.com/Ayn1631/kimodo.git
cd kimodo

git clone https://github.com/nv-tlabs/kimodo-viser.git
python -c "import shutil; shutil.copy2(r'replace\_client_autobuild.py', r'kimodo-viser\src\viser\_client_autobuild.py')"
pip install -e kimodo-viser
set SKIP_MOTION_CORRECTION_IN_SETUP=1 && pip install -e .
```

5. 下载预编译版motion_correction
```bash
pip install https://github.com/Aero-Ex/kimodo/releases/download/v1.0.0/motion_correction-1.0.0-cp312-cp312-win_amd64.whl
```

6. 下载其余依赖
```bash
pip install bitsandbytes
pip install -U transformers==5.1.0
```

7. 修改配置文件
`kimodo/kimodo/model/llm2vec/llm2vec_wrapper.py` 


```python
cp .env.example .env
# 修改.env文件里的参数和地址
```


9. 启动demo
```bash
python -m kimodo.demo
```