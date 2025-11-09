# Gemini CLI (Node.js) 权威安装与避坑指南

本文档是针对官方 `@google/gemini-cli` (Node.js 版本) 的一份全面安装、配置和故障排除指南。它将重点解决在 Windows 环境下因 Node.js 版本、网络代理和 Google Cloud 配置引发的常见问题。

---

## 目录

1.  [核心概念：`@google/gemini-cli`](#-一核心概念google-gemini-cli)
2.  [安装与版本问题](#-二安装与版本问题)
    *   [**天坑一：`EBADENGINE` 错误与 Node.js 版本**](#-天坑一ebadengine-错误与-nodejs-版本)
3.  [核心步骤：授权与配置](#-三核心步骤授权与配置)
    *   [标准登录流程](#标准登录流程)
    *   [**天坑二：网络超时与代理配置**](#-天坑二网络超时与代理配置)
    *   [**天坑三：缺少 Google Cloud 项目 ID**](#-天坑三缺少-google-cloud-项目-id)
4.  [故障排除清单 (Checklist)](#-四故障排除清单-checklist)

---

## 🔹 一、核心概念：`@google/gemini-cli`

`@google/gemini-cli` 是一个基于 Node.js 的开源命令行工具，让你可以在终端中直接与 Gemini 模型交互。

**请注意：** 本指南专注于此 Node.js 版本，它通过 `npm` 安装，而非 Python 的 `pip`。

---

## 🔹 二、安装与版本问题

根据官方文档，安装 Gemini CLI 的前提是 **Node.js 版本 >= 20**。这是导致许多用户安装失败的第一个，也是最常见的一个坑。

### ⭐ **天坑一：`EBADENGINE` 错误与 Node.js 版本**

*   **表现**：在 `npm install` 过程中，出现大量 `npm WARN EBADENGINE Unsupported engine` 警告，提示需要的 Node.js 版本与当前版本不符。
*   **根本原因**：你当前的 Node.js 版本低于 `v20`。
*   **解决方案：使用 `nvm-windows` 管理 Node.js 版本 (强烈推荐)**

    `nvm-windows` 是一个让你能在同一台电脑上轻松安装和切换多个 Node.js 版本的工具。

    1.  **下载与安装**：
        *   前往 [nvm-windows releases 页面](https://github.com/coreybutler/nvm-windows/releases)。
        *   下载最新的 `nvm-setup.exe` 并运行安装。

    2.  **安装并切换 Node.js 版本**：
        *   安装完成后，**重启你的终端**。
        *   执行以下命令来安装并使用符合要求的 Node.js 版本（例如 `20.18.1`）：
        ```powershell
        # 安装 Node.js v20.18.1
        nvm install 20.18.1

        # 切换到该版本
        nvm use 20.18.1
        ```

    3.  **验证版本**：
        ```powershell
        node -v
        # 输出应为 v20.18.1 或更高
        ```

    4.  **安装 Gemini CLI**：
        在正确的 Node.js 版本下，全局安装 Gemini CLI：
        ```powershell
        npm install -g @google/gemini-cli
        ```

---

## 🔹 三、核心步骤：授权与配置

安装成功后，首次运行 `gemini` 命令会启动授权流程。

### 标准登录流程

1.  在终端输入 `gemini`。
2.  CLI 会提示你选择登录方式，选择 "Login with Google"。
3.  它会自动打开浏览器，并跳转到 Google 登录页面。
4.  授权后，页面会尝试回调并通知 CLI，完成登录。

这个流程中的网络和配置问题是导致授权失败的主要原因。

### ⭐ **天坑二：网络超时与代理配置**

*   **表现**：执行 `gemini` 后，浏览器无法打开 Google 页面，或 CLI 长时间等待后报错“超时 (Timeout)”。
*   **根本原因**：和 Python、Git 一样，Node.js 进程默认**不会**使用系统或浏览器的代理。你需要为它显式配置代理环境变量。
*   **解决方案：设置代理环境变量**

    假设你的代理工具（如 Clash, V2Ray）在本地 `7890` 端口上提供了 HTTP 代理。

    **临时设置 (仅当前终端窗口有效):**
    ```powershell
    $env:HTTP_PROXY="http://127.0.0.1:7890"
    $env:HTTPS_PROXY="http://127.0.0.1:7890"
    ```

    **永久设置 (推荐):**
    ```powershell
    # 这会将变量写入系统，新打开的终端都会生效
    setx HTTP_PROXY "http://127.0.0.1:7890"
    setx HTTPS_PROXY "http://127.0.0.1:7890"
    ```
    设置后，**必须重启终端**才能生效。配置好代理后，再重新运行 `gemini`。

### ⭐ **天坑三：缺少 Google Cloud 项目 ID**

*   **表现**：使用公司账户或付费版 Gemini Code Assist 许可证登录时，授权后可能提示需要设置项目 ID。
*   **根本原因**：企业级或付费功能需要将 API 调用与一个具体的 Google Cloud Platform (GCP) 项目关联，以便进行计费和配额管理。
*   **解决方案：设置项目 ID 环境变量**

    1.  **找到你的项目 ID**：
        *   访问 [Google Cloud Console](https://console.cloud.google.com/)。
        *   在页面左上角的项目选择器中，选择你的项目。
        *   记下仪表盘上显示的 **项目 ID** (Project ID)，例如 `my-gemini-project-123456`。

    2.  **设置环境变量**：
        ```powershell
        # 推荐使用 setx 进行永久设置
        setx GOOGLE_CLOUD_PROJECT "my-gemini-project-123456"
        ```
        同样，**重启终端**后生效。

---

## 🔹 四、故障排除清单 (Checklist)

如果登录仍然失败，请按以下步骤逐一排查：

1.  **Node.js 版本正确吗？**
    *   运行 `node -v`，确保版本号是 `v20.x` 或更高。如果不是，请使用 `nvm use` 切换。

2.  **代理环境变量设置了吗？**
    *   运行 `echo $env:HTTPS_PROXY`，检查输出是否为 `http://127.0.0.1:7890` (或你的代理地址)。如果没有，请返回上一步设置。

3.  **代理本身工作正常吗？**
    *   在设置了代理的终端里，运行 `curl https://accounts.google.com`。如果能返回一大堆 HTML 代码，说明代理在终端中已生效。如果超时，说明代理配置或工具本身有问题。

4.  **GCP 项目 ID 设置了吗？ (如果需要)**
    *   运行 `echo $env:GOOGLE_CLOUD_PROJECT`，检查是否已设置为你的项目 ID。

5.  **尝试不同的终端**
    *   如果你正在使用 VS Code 的内嵌终端，尝试切换到独立的 Windows PowerShell 或命令提示符 (CMD) 窗口再运行 `gemini`。

6.  **检查防火墙**
    *   临时关闭 Windows Defender 防火墙或你的第三方安全软件，以排除它们拦截了登录回调的可能性。