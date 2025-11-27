# WSL2 安装与配置指南

本文档提供一套从零开始，在全新 Windows 机器上启用、配置并使用 Ubuntu (WSL2) 环境的完整步骤。按照指引操作，即可在 Windows 系统中无缝运行完整的 Linux 环境。

---

## 1. 核心安装步骤

### 1.1 一键安装 (强烈推荐)

此方法最为快捷，会自动完成所有必要步骤。

以 **管理员身份** 打开 PowerShell，然后执行以下命令：
```powershell
wsl --install
```
该命令会自动执行三项操作：
1.  启用 WSL (Windows Subsystem for Linux) 和虚拟机平台功能。
2.  安装并设置 WSL2 为默认版本。
3.  安装默认的 Linux 发行版：**Ubuntu**。

安装完成后，根据提示重启计算机。

### 1.2 手动安装 (备选方案)

如果一键安装失败，可以尝试手动分步安装。

**第一步：启用所需功能**
以 **管理员身份** 打开 PowerShell，依次执行以下两条命令：
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
执行完毕后，重启计算机。

**第二步：安装 Ubuntu**
可以从 Microsoft Store 免费安装：
1.  打开 **Microsoft Store** 应用。
2.  搜索 "Ubuntu"。
3.  根据需要选择版本（如 `Ubuntu 22.04 LTS` 或 `Ubuntu 24.04 LTS`），点击“获取”进行安装。

### 1.3 初始化 Ubuntu
安装完成后，从“开始”菜单找到并点击 **"Ubuntu"** 图标。第一次启动时，系统会自动进行初始化，并要求你：
1.  创建一个 Linux 用户名。
2.  设置对应的密码。

完成此步骤后，你将进入 Ubuntu 的命令行终端，安装正式完成。

---

## 2. 基础配置与使用

### 2.1 设置默认 WSL 版本为 WSL2

为确保最佳性能，建议将 WSL2 设置为默认版本。
```powershell
wsl --set-default-version 2
```
你可以随时查看已安装的 Linux 发行版及其 WSL 版本：
```powershell
wsl -l -v
```
如果发现你的 Ubuntu 仍是 WSL1 版本，可以使用以下命令手动将其转换为 WSL2：
```powershell
wsl --set-version Ubuntu 2
```
*(请将命令中的 "Ubuntu" 替换为 `wsl -l -v` 列表中显示的确切名称)*

### 2.2 访问与切换环境

- **从 Windows 进入 Ubuntu**：在 PowerShell 或“开始”菜单中，直接运行 `wsl` 或 `ubuntu`。
- **从 Ubuntu 返回 Windows**：在 Ubuntu 终端中，输入 `exit`。

### 2.3 文件互通

WSL2 提供了便捷的双向文件访问能力。

- **在 Ubuntu 中访问 Windows 文件**：
  Windows 的各个盘符被挂载在 `/mnt/` 目录下。例如，访问 C 盘桌面：
  ```bash
  cd /mnt/c/Users/你的Windows用户名/Desktop
  ```

- **在 Windows 中访问 Ubuntu 文件**：
  打开 Windows 的“文件资源管理器”，在地址栏输入 `\\wsl$` 并回车，即可看到所有已安装的 Linux 发行版文件系统。例如，Ubuntu 的根目录通常位于：
  ```
  \\wsl$\Ubuntu\
  ```

### 2.4 安装基础开发工具
进入 Ubuntu 后，建议首先更新软件包并安装常用开发工具：
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential curl git zip unzip -y
```

---

## 3. 常见问题与解决方案

### 3.1 问题：在 WSL 中无法使用 Windows 代理

一个常见问题是在 WSL 终端中无法连接网络，即使 Windows 上已经开启了代理软件（如 Clash）。

#### 3.1.1 根本原因

WSL 的网络环境与 Windows 相对独立。**WSL 的 `localhost` (或 `127.0.0.1`) 指向其自身，而不是 Windows 的 `localhost`**。因此，直接在 WSL 中使用 `localhost:7890` 这样的地址是无法访问到 Windows 代理服务的。

#### 3.1.2 解决方案

要让 WSL 使用 Windows 的代理，需要完成以下两步：

**第一步：在代理软件中开启“允许局域网连接” (Allow LAN)**

此选项会将代理的监听地址从 `127.0.0.1` (仅本机) 更改为 `0.0.0.0` (接受所有局域网设备的连接)。这是让 WSL 能够访问到代理服务的前提。

**第二步：在 WSL 中配置代理指向 Windows 的 IP**

1.  **获取 Windows 的网关 IP**
    在 **WSL 终端** 中执行以下命令，找到 Windows 的 IP 地址：
    ```bash
    ip route | grep default
    ```
    输出类似于 `default via 172.30.80.1 dev eth0`，其中的 `172.30.80.1` 就是我们需要的网关 IP。

2.  **配置代理环境变量 (推荐永久方案)**
    编辑 `~/.bashrc` 文件，使其在每次启动终端时自动配置代理。
    ```bash
    nano ~/.bashrc
    ```
    在文件末尾添加以下内容 (假设代理端口为 `7890`，如果不同请自行修改)：
    ```bash
    # Proxy configuration for WSL to use Windows gateway
    GATEWAY=$(ip route | awk '/default/ {print $3}')
    export http_proxy="http://$GATEWAY:7890"
    export https_proxy="http://$GATEWAY:7890"
    export ALL_PROXY="socks5://$GATEWAY:7890"
    ```
    保存文件后，执行 `source ~/.bashrc` 使其立即生效。

3.  **为 `apt` 单独配置代理 (如果需要)**
    如果 `apt` 命令仍然无法通过代理，可以为其创建专门的配置文件：
    ```bash
    sudo nano /etc/apt/apt.conf.d/proxy.conf
    ```
    写入以下内容 (请将 `172.30.80.1` 替换为你的实际网关 IP)：
    ```
    Acquire {
      HTTP::Proxy "http://172.30.80.1:7890";
      HTTPS::Proxy "http://172.30.80.1:7890";
    }
    ```
    保存并退出即可。

完成以上配置后，你的 WSL 环境应该就能正常通过 Windows 代理访问网络了。