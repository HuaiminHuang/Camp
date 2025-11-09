# 设置 VS Code 作为 Gemini CLI 的文件编辑器

本文档将指导你如何将 Visual Studio Code (VS Code) 设置为 Gemini CLI 的默认文件编辑器。

## Gemini CLI 支持的编辑器

Gemini CLI 原生支持多种编辑器，从下面的界面可以看到，VS Code 是其中之一：

```
> Select Editor                                              Editor Preference
● 1. None
  2. Cursor (Not installed)
  3. Emacs (Not installed)
  4. Neovim (Not installed)
  5. Vim (Not installed)
  6. VS Code
  7. VSCodium (Not installed)
  8. Windsurf (Not installed)
  9. Zed (Not installed)
```

要成功配置 VS Code，请遵循以下步骤。

---

### 步骤 1：检查 VS Code 安装与 `code` 命令

首先，确保你的系统已经安装了 VS Code，并且 `code` 命令可以在命令行中正常使用。

*   **Windows 用户**:
    1.  在安装 VS Code 时，请务必勾选 **“添加到 PATH”** 的选项。
    2.  打开 PowerShell 或 CMD，输入以下命令进行验证：
        ```powershell
        code --version
        ```
    3.  如果命令返回了 VS Code 的版本号，说明 `code` 命令可用。如果提示“命令未找到”，则需要手动配置环境变量（见步骤 3）。

*   **Linux / macOS 用户**:
    *   安装 VS Code 后，确保 `code` 命令在你的 Shell 中可用。
    *   对于 macOS，可以在 VS Code 中按 `Cmd+Shift+P`，输入 `shell command`，然后选择 **"Install 'code' command in PATH"** 来安装。

> **注意**: Gemini CLI 的编辑器功能不依赖任何 VS Code 扩展。它只是调用 VS Code 程序来打开文本文件。

---

### 步骤 2：分析 VS Code 显示“Not installed”的原因

如果在 Gemini CLI 中，VS Code 旁边显示 “Not installed”，这通常意味着 CLI 在系统的 `PATH` 环境变量中找不到 `code` 命令。

*   **核心原因**: CLI 通过在命令行中执行 `code` 来检测 VS Code 是否安装。
*   **常见情况**:
    1.  **PATH 未配置**: 安装 VS Code 时忘记勾选 “添加到 PATH”。
    2.  **环境不一致**: 在 PowerShell 中 `code` 命令可用，但在其他终端（如 Git Bash, WSL）中不可用。
    3.  **沙箱限制**: 如果你在容器或沙箱（Sandbox）模式下运行 Gemini CLI，它可能无法访问宿主机的 `PATH`。

---

### 步骤 3：在 Windows 中检查并配置环境变量

你需要确保包含 `code.cmd` 的目录已经被添加到系统的 `PATH` 环境变量中。

1.  **找到 VS Code 的 `bin` 目录**
    通常，这个目录位于：
    ```
    C:\Users\<你的用户名>\AppData\Local\Programs\Microsoft VS Code\bin
    ```
    请将 `<你的用户名>` 替换为你的实际 Windows 用户名。

2.  **在 PowerShell 中检查 `Path` 变量**
    为了方便查看，可以使用以下命令将 `Path` 变量的每个条目分行显示：
    ```powershell
    $env:Path.Split(';')
    ```
    检查输出列表中是否包含上述 VS Code 的 `bin` 目录。

3.  **手动添加 `Path` (如果缺失)**
    *   按 `Win + R`，输入 `sysdm.cpl` 并回车。
    *   在“系统属性”窗口中，进入“高级”选项卡，点击“环境变量...”。
    *   在“系统变量”区域，找到并选中 `Path`，然后点击“编辑...”。
    *   点击“新建”，然后粘贴你的 VS Code `bin` 目录路径。
    *   点击所有窗口的“确定”来保存更改。
    *   **关键一步**: **关闭并重新打开** 所有的命令行窗口（包括 Gemini CLI），以使新的环境变量生效。

4.  **再次验证**
    在新打开的 PowerShell 窗口中再次运行 `code --version`，确认命令可以成功执行。

---

### 步骤 4：在 Gemini CLI 中设置编辑器

当 `code` 命令在你的终端中可用后，就可以在 Gemini CLI 中进行设置了。

1.  **使用 `/editor` 命令**
    在 Gemini CLI 中，直接输入 `/editor` 命令并回车。在弹出的菜单中，用方向键选择 "VS Code"，然后按 Enter 确认。

2.  **手动修改配置文件 (备用方案)**
    你也可以直接编辑 Gemini CLI 的配置文件 `config.json`。找到并修改 `preferredEditor` 字段：
    ```json
    {
        ...,
        "preferredEditor": "vscode"
    }
    ```
    如果修改后未立即生效，请尝试重启 Gemini CLI。

设置成功后，当你再次打开 `/editor` 菜单时，应该会看到 **"Your preferred editor is: VS Code."** 的提示。