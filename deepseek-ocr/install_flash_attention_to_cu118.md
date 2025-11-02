# **B 方案**（保持 cu118 的 PyTorch，不动 torch/vllm，只降系统 CUDA 到 11.8）

---

## 🚧 场景确认

你当前的环境情况是这样的：

```
/usr/local/cuda -> /usr/local/cuda-12.4
torch.version.cuda = 11.8
```

问题在于：
`flash-attn` 在编译阶段调用系统的 `nvcc`（12.4 版）
→ 与 torch 的 ABI (11.8) 不一致
→ 报错或卡死。

---

## 🔧 B 方案目标

> 让系统的默认 CUDA Toolkit 指向 **11.8**，
> 从而让所有需要编译的 C++/CUDA 扩展（如 flash-attn）
> 都使用 nvcc 11.8 进行构建。

---

## 🧭 步骤详解

### ① 查看现有 CUDA 版本路径

```bash
ls -l /usr/local/
```

你会看到类似：

```
cuda -> /etc/alternatives/cuda
cuda-12.4
```

有时 autodl 容器只装了一个版本，所以我们要补装 11.8。

---

### ② 安装 CUDA 11.8 Toolkit（并非驱动）

执行：

```bash
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

> ⚠️ 注意：
>
> * 不要装 `cuda`（那是 meta package，会升级驱动）
> * 我们只装 `cuda-toolkit-11-8`，包含编译工具 nvcc + headers + libs

---

### ③ 检查安装结果

```bash
ls /usr/local/
```

应能看到：

```
cuda-11.8
cuda-12.4
cuda -> /usr/local/cuda-12.4
```

此时 11.8 已安装，但还没切换。

---

### ④ 切换系统 CUDA 默认版本

将 `/usr/local/cuda` 链接到 11.8：

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda
```

---

### ⑤ 验证切换成功

```bash
nvcc -V
```

输出应包含：

```
Cuda compilation tools, release 11.8, V11.8.x
```

---

### ⑥ 重新编译 flash-attn

现在重新执行：

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

此时会自动使用 11.8 的 nvcc，与 torch cu118 匹配 ✅

---

## 🔍 验证最终环境

在 Python 中验证：

```python
import torch
print(torch.version.cuda)
!nvcc -V
```

应显示：

```
torch.version.cuda = 11.8
Cuda compilation tools, release 11.8
```

说明环境完全一致。

---

## ⚙️ 附加：恢复到 12.4（如果需要）

若将来你要装 cu124 版本，只需执行：

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda
```

再运行：

```bash
nvcc -V
# -> CUDA 12.4
```

即可恢复。

---

这个错误：

```
E: Unable to locate package cuda-toolkit-11-8
```

说明你的系统 **apt 源中没有 NVIDIA 的 CUDA 软件仓库**。
默认的 Ubuntu 源是不包含 `cuda-toolkit-11-8` 的，需要先手动添加 **NVIDIA 官方 apt 仓库**。

---

## ✅ 解决方案：手动添加 CUDA 11.8 官方源

以下步骤通用于 Ubuntu 20.04 / 22.04（autodl 默认多为 22.04）：

---

### 🧭 ① 检查系统版本

```bash
lsb_release -a
```

输出类似：

```
Distributor ID: Ubuntu
Release: 22.04
```

我们假设是 `22.04`，如果是 `20.04`，我会注明差异。

---

### ⚙️ ② 添加 NVIDIA CUDA 官方仓库

执行以下命令（建议复制整块直接粘贴）：

```bash
# 添加 NVIDIA 官方 GPG key
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# 添加 CUDA 11.8 仓库到 sources.list.d
sudo bash -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda-ubuntu2204.list'

# 更新索引
sudo apt update
```

> 如果你的系统是 Ubuntu 20.04，把上面命令中的 `ubuntu2204` 改为 `ubuntu2004`。

---

### 🚀 ③ 安装 CUDA Toolkit 11.8

```bash
sudo apt install -y cuda-toolkit-11-8
```

安装完成后，确认：

```bash
ls /usr/local/
# 应看到 cuda-11.8 文件夹
```

---

### 🔁 ④ 切换默认 CUDA 指向 11.8

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda
```

验证：

```bash
nvcc -V
```

输出应为：

```
Cuda compilation tools, release 11.8, V11.8.89
```

---

### ✅ ⑤ 重新安装 flash-attn

现在重新执行：

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

就会使用正确版本的 `nvcc`，与 torch (cu118) 匹配。

---

### 💡 附加说明

如果你在 **autodl 平台** 的镜像里没有 `sudo` 权限，可用以下替代方案：

```bash
apt-get update
apt-get install -y cuda-toolkit-11-8
```

（在 root 权限下，`sudo` 可省略）

---

要我帮你写一段 **自动检测 Ubuntu 版本并配置 CUDA 11.8 源的脚本** 吗？
执行一次即可自动完成上述操作。
