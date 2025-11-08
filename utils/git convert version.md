在 Git 中，`git clone` 命令默认会克隆仓库的**所有历史记录**，并检出**最新的版本**（通常是 `main` 或 `master` 分支的 HEAD）。

要克隆这个仓库的**旧版本**（比如某个特定的 Tag、Commit 或 Branch），你可以遵循以下步骤。最常见且推荐的做法是先克隆整个仓库，**然后再切换**到你想要的旧版本。

-----

### **方法一：先完整克隆，然后切换到旧版本（推荐）**

这是最标准和推荐的方法，因为你保留了完整的历史记录，方便以后切换版本。

1.  **克隆整个仓库：**
    ```bash
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    ```
2.  **进入仓库目录：**
    ```bash
    cd lm-evaluation-harness
    ```
3.  **查找你想要的旧版本标识符**（例如，Tag 或 Commit Hash）：
      * 你可以查看仓库的 Releases 页面，或者使用 `git tag` 命令列出所有标签：
        ```bash
        git tag
        ```
        - 输出示例
        ```bash
        v0.0.1
        ......
        v0.4.7
        v0.4.8
        v0.4.9
        v0.4.9.1
        ```
      > 如果遇到没有返回tag，使用`git fetch --tags`更新tags

      * 或者使用 `git log` 查看提交历史并找到你想回退的 **Commit Hash**。
4.  **切换到指定的旧版本：**
      * **如果知道 Tag 名称**（例如 `v0.4.7`）：
        ```bash
        git checkout v0.4.7
        ```
        - 输出示例
        ```bash 
        Updating files: 100% (10801/10801), done.
        Note: switching to 'v0.4.7'.

        You are in 'detached HEAD' state. You can look around, make experimental
        changes and commit them, and you can discard any commits you make in this
        state without impacting any branches by switching back to a branch.
        ```
      * **如果知道 Commit Hash**（例如 `a1b2c3d4e5f67890...`）：
        ```bash
        git checkout a1b2c3d4e5f67890
        ```
      * **如果想切换到某个旧分支**（例如 `old-feature-branch`）：
        ```bash
        git checkout old-feature-branch
        ```

-----

### **方法二：浅克隆并立即检出指定版本（不推荐给初学者）**

如果你**只想要**某个旧版本的代码文件，而**不需要**完整的历史记录，可以使用这个方法来节省下载时间和磁盘空间。

  * **使用 `--depth 1` 和 `--branch` 选项：**

    ```bash
    git clone --depth 1 --branch <版本标识符> https://github.com/EleutherAI/lm-evaluation-harness.git
    ```

      * 将 `<版本标识符>` 替换为你想要的 **Tag** 或 **Branch** 名称。
      * **注意：** 如果是 **Commit Hash**，这个方法通常**不可行**，或者操作复杂。对于 Commit Hash，**强烈建议使用方法一**。

-----

要检查你 **Git 仓库** 当前检出的是哪个版本，最可靠的方法是使用 `git status` 和 `git describe` 命令。

-----

## 🛠️ 检查 Git 仓库版本

### 1\. 使用 `git status`（检查 HEAD 位置）

这个命令会告诉你 **HEAD** 当前指向哪里，以及工作目录是否干净。

```bash
git status
```

  * **如果当前版本是 `v0.4.7` 并且是分离头指针状态：**
    你会看到类似下面的输出：

    ```
    HEAD detached at v0.4.7
    nothing to commit, working tree clean
    ```

    这表明你当前的代码正是 `v0.4.7` 版本。

  * **如果当前版本是 `v0.4.9.1`（或更高）并且在一个分支上：**
    你会看到类似下面的输出：

    ```
    On branch main (或 master)
    Your branch is up to date with 'origin/main'.
    nothing to commit, working tree clean
    ```

    在这种情况下，你需要用 `git rev-parse HEAD` 或下面的 `git describe` 来获取精确的版本。

### 2\. 使用 `git describe`（获取最近的 Tag）

这个命令会给出当前提交最接近的 Tag 名称。

```bash
git describe --tags --always
```

  * **如果输出是 `v0.4.7`：** 那么你当前就在这个 Tag 上。
  * **如果输出是 `v0.4.9.1`：** 那么你当前就在这个 Tag 上。
  * **如果输出是 `v0.4.7-x-gCOMMIT_HASH`：** 比如 `v0.4.7-10-ga1b2c3d`，这意味着你是在 `v0.4.7` 版本之后又进行了 10 次提交，但你还没有打上新的 Tag。

-----

如果你想切换回最新的版本，你可以运行：

```bash
git checkout v0.4.9.1
# 或切换回主分支
git checkout main
```