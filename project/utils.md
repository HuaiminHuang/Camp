# 🧩 一、pip 镜像配置

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

👉 它会在 `~/.config/pip/pip.conf` 或 `~/.pip/pip.conf` 里生成：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

---

## 📦 常用国内 pip 镜像源

| 镜像名     | URL                                                                                                        |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| 清华      | [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)                       |
| 阿里云     | [https://mirrors.aliyun.com/pypi/simple](https://mirrors.aliyun.com/pypi/simple)                           |
| 中科大     | [https://pypi.mirrors.ustc.edu.cn/simple](https://pypi.mirrors.ustc.edu.cn/simple)                         |
| 华为云     | [https://repo.huaweicloud.com/repository/pypi/simple](https://repo.huaweicloud.com/repository/pypi/simple) |
| 豆瓣（不稳定） | [https://pypi.douban.com/simple](https://pypi.douban.com/simple)                                           |

👉 建议优先使用清华（稳定且更新快）

---

## 🧠 验证是否成功

```bash
pip config list
```

输出应类似：

```
global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
```

然后试试安装包：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

速度通常会提升到数 MB/s。

---

# 🧰 二、conda 镜像配置（重点）

conda 默认走 `repo.anaconda.com`，在国内几乎无法访问，需要改源。

### 1️⃣ 修改 `~/.condarc` 文件

```bash
vim ~/.condarc
```

填入以下内容（推荐清华）：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

### 2️⃣ 清除缓存并更新索引

```bash
conda clean -i
conda update conda
```

---

## 📦 常见国内 Conda 镜像源

| 源   | URL                                                                                            |
| --- | ---------------------------------------------------------------------------------------------- |
| 清华  | [https://mirrors.tuna.tsinghua.edu.cn/anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda) |
| 中科大 | [https://mirrors.ustc.edu.cn/anaconda](https://mirrors.ustc.edu.cn/anaconda)                   |
| 阿里云 | [https://mirrors.aliyun.com/anaconda](https://mirrors.aliyun.com/anaconda)                     |
| 华为云 | [https://repo.huaweicloud.com/anaconda](https://repo.huaweicloud.com/anaconda)                 |

---

# 🚀 三、针对 PyTorch、Transformers、Hugging Face 下载慢的优化

## (1) Torch 镜像源（国内）

例如安装 GPU 版 PyTorch：

```bash
# 官方：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 国内清华镜像（推荐）：
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

或者使用清华镜像同步的 PyTorch 仓库：

```bash
pip install torch -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

---

## (2) Hugging Face 下载加速

AutoDL 环境常常下载 `model.safetensors` 很慢，可用以下方式：

1. 配置环境变量：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

   或者：

   ```bash
   export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
   ```

2. 使用清华镜像：

   ```bash
   git lfs install
   git clone https://hf-mirror.com/Qwen/Qwen2-7B
   ```

---

# ⚙️ 四、前瞻性建议（深度学习场景）

1. **在 AutoDL 初始化脚本中自动配置镜像源**
   例如在 `~/.bashrc` 末尾加上：

   ```bash
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   export HF_ENDPOINT=https://hf-mirror.com
   export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
   ```

2. **对 Conda 环境封装**

   * 可以创建一个 `env_setup.sh` 自动配置镜像和环境
   * 或使用 `environment.yml` + 本地镜像同步仓库

3. **大模型下载优化**

   * 对于模型如 `Qwen`, `LLaMA`, `Baichuan` 建议提前在 AutoDL 上传至 `/root/autodl-tmp/models/`，避免多次重复下载。

---

# ✅ 总结

| 类型              | 配置命令 / 文件                                                                  | 推荐镜像      |
| --------------- | -------------------------------------------------------------------------- | --------- |
| **pip**         | `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` | 清华        |
| **conda**       | `~/.condarc`                                                               | 清华        |
| **HuggingFace** | `export HF_ENDPOINT=https://hf-mirror.com`                                 | HF-Mirror |
| **PyTorch**     | `pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple`            | 清华        |

---
