# 如何配置 Docker 作为 Gemini CLI 的沙箱环境

本文档的核心目标是指导你如何正确安装和配置 Docker，以成功启用 Gemini CLI 的沙箱（Sandbox）功能。沙箱通过容器化技术提供了一个安全、隔离的环境来执行命令，能有效防止对你本地文件系统的意外修改。

---

## 1. 理解 `Missing sandbox command 'docker'` 错误

当你看到这个错误时，意味着 Gemini CLI 已经配置为使用 Docker 作为沙箱 (`"sandbox": "docker"`)，但它在你的系统中无法找到或执行 `docker` 命令。

要解决这个问题，你需要确保 Docker 已经正确安装，并且其路径已经添加到了系统的环境变量 `PATH` 中。

---

## 2. 配置 Docker 沙箱的详细步骤

请按照以下步骤来确保 Docker 环境配置无误。

### 步骤一：验证 Docker 环境

在进行任何修改之前，首先检查你的系统是否已经安装了 Docker。打开 PowerShell 或 CMD，运行以下命令：

```powershell
docker --version
```

*   **如果成功**：命令会返回 Docker 的版本号，例如 `Docker version 28.5.1, build e180ab8`。这说明 Docker 已安装并且 `PATH` 配置正确。你可以直接跳到 **步骤四**。
*   **如果失败**：提示“无法将‘docker’项识别为 cmdlet...”，则说明 Docker 未安装或 `PATH` 未配置。请继续执行 **步骤二**。

### 步骤二：安装 Docker Desktop

1.  **下载 Docker Desktop**:
    访问 [Docker Desktop for Windows 官方网站](https://www.docker.com/products/docker-desktop/) 下载安装程序。

2.  **安装 Docker**:
    运行安装程序。在安装过程中，请务必确保勾选了以下两个关键选项：
    *   **Install required Windows components for WSL 2** (推荐)
    *   **Add shortcut to desktop**
    *   安装程序通常会自动处理 `PATH` 添加，无需手动勾选。

3.  **启动 Docker Desktop**:
    安装完成后，从桌面快捷方式或开始菜单启动 Docker Desktop。首次启动可能需要一些时间来完成初始化设置。请确保 Docker Desktop 右下角的图标显示为绿色，表示 Docker 引擎正在运行。

### 步骤三：验证 Docker 命令和环境变量

1.  **重启终端**:
    **关闭所有已打开的命令行窗口**，然后重新打开一个新的 PowerShell 或 CMD 窗口。这一步至关重要，因为它能确保新的环境变量生效。

2.  **再次验证 `docker` 命令**:
    在新终端中，再次运行：
    ```powershell
    docker --version
    ```
    此时应该能成功看到版本号。

3.  **(可选) 检查 `PATH` 环境变量**:
    如果上一步仍然失败，你可以手动检查 `PATH`。在 PowerShell 中运行：
    ```powershell
    $env:Path.Split(';')
    ```
    检查输出列表中是否包含类似 `C:\Program Files\Docker\Docker\resources\bin` 的路径。如果没有，你需要手动将其添加到系统环境变量中。

### 步骤四：确认 Gemini CLI 配置

确保 Gemini CLI 的配置指向 Docker。通常这是默认设置，但你可以检查 `config.json` 文件（位于 `C:\Users\<你的用户名>\.config\gemini\config.json`）来确认：

```json
"tools": {
  "sandbox": "docker"
}
```
如果该值不是 `"docker"`，请将其修改为此。

完成以上所有步骤后，再次运行 `gemini` 命令，它应该就能成功找到 Docker 并利用其沙箱功能，不再报错。

---

## 3. 备选方案：临时禁用沙箱

如果你暂时不想配置 Docker，只是想快速使用 Gemini CLI，可以临时禁用沙箱功能。

在 PowerShell 中运行以下命令即可永久禁用沙箱：

```powershell
setx GEMINI_SANDBOX none
```

**注意**：这只是一个备用方案。我们推荐使用 Docker 沙箱以获得更安全的操作体验。