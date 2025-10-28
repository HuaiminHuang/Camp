# Linux 常用命令行操作总结

本文档总结了在 Linux 系统下，特别是 AutoDL / 容器环境中常用的命令行操作，包括文件操作、环境管理、日志查看、进程管理、动态库管理以及 MongoDB 启动与调试。

---

## 1. 文件与目录操作

| 命令 | 说明 | 示例 |
|------|------|------|
| `ls` | 列出目录文件 | `ls -l`, `ls -a` |
| `ll` | 列出详细文件信息 | `ll /usr/lib/x86_64-linux-gnu/` |
| `cd` | 切换目录 | `cd ~/autodl-tmp/RAG` |
| `pwd` | 显示当前目录 | `pwd` |
| `mkdir` | 创建目录 | `mkdir -p log` |
| `cp` | 拷贝文件 | `cp a.txt b.txt` |
| `mv` | 移动或重命名文件 | `mv old.txt new.txt` |
| `rm` | 删除文件 | `rm file.txt`, `rm -rf dir/` |
| `cat` | 查看文件内容 | `cat config.ini` |
| `more/less` | 分页查看文件 | `less README.md` |
| `tail` | 查看文件末尾，实时跟踪 | `tail -f log/qwen3-7b.log` |

---

## 2. 环境管理（Conda / Python / Pip）

| 命令 | 说明 | 示例 |
|------|------|------|
| `conda activate <env>` | 激活 Conda 环境 | `conda activate rag` |
| `conda deactivate` | 退出 Conda 环境 | `conda deactivate` |
| `source <file>` | 执行 shell 脚本，导入环境变量 | `source config.ini` |
| `pip install -r requirements.txt` | 安装 Python 依赖 | `pip install -r requirements.txt` |
| `pip list` | 查看已安装的 Python 包 | `pip list` |
| `python --version` | 查看 Python 版本 | `python --version` |

> ⚠️ 注意：`source config.ini` 只会在当前 shell 导入环境变量，并不等同于 `conda activate`。

---

## 3. 进程管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `ps aux` | 查看当前所有进程 | `ps aux | grep mongod` |
| `top/htop` | 实时查看系统进程与资源占用 | `top` |
| `kill <pid>` | 杀掉指定进程 | `kill 2221` |
| `nohup <cmd> &` | 后台启动进程，并输出到文件 | `nohup python src/server/semantic_chunk.py > log/semantic_chunk.log 2>&1 &` |

---

## 4. 日志查看与调试

| 命令 | 说明 | 示例 |
|------|------|------|
| `tail -f <file>` | 实时查看文件末尾日志 | `tail -f log/qwen3-7b.log` |
| `less +F <file>` | 类似 tail -f，可以翻页 | `less +F log/semantic_chunk.log` |
| `grep <pattern> <file>` | 搜索日志中指定内容 | `grep ERROR log/qwen3-7b.log` |

> ⚠️ 注意：`tail` 只能用于文件，不能直接 `tail log/`（目录）。

---

## 5. 系统库检查与动态库管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `ldd <binary>` | 查看可执行文件依赖的动态库 | `ldd ~/RAG/mongodb-7.0.20/bin/mongod` |
| `ln -s <target> <link>` | 创建软链接 | `sudo ln -s /usr/lib/x86_64-linux-gnu/libcrypto.so.3 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.1` |
| `dpkg -i <file.deb>` | 安装 Debian 包 | `sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb` |
| `apt-get install <package>` | 安装系统包 | `sudo apt-get install wget` |
| `ldconfig` | 刷新动态链接库缓存 | `sudo ldconfig` |

> ⚠️ 在容器环境中，不建议覆盖系统库，推荐使用 **局部解包 + LD_LIBRARY_PATH** 的方式启动需要特定库的程序。

--- 
# Linux 进阶命令行操作指南

本文档汇总了 Linux 进阶命令和技巧，适合开发、运维和数据科学工作中快速定位问题、分析性能、管理系统和调试服务。

---

## 1️⃣ 系统信息与性能监控

| 命令 | 说明 | 示例 |
|------|------|------|
| `top` | 实时查看系统进程、CPU、内存占用 | `top` |
| `htop` | `top` 的增强版，可交互排序和杀进程 | `htop` |
| `vmstat` | 查看系统 CPU、内存、IO 状态 | `vmstat 1` |
| `iostat` | 查看磁盘 IO 性能 | `iostat -x 1` |
| `free` | 查看内存使用情况 | `free -h` |
| `uptime` | 查看系统运行时间及负载 | `uptime` |
| `dmesg` | 查看内核日志 | `dmesg | tail -n 50` |
| `uname -a` | 查看内核版本 | `uname -a` |

---

## 2️⃣ 磁盘与文件系统管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `df -h` | 查看磁盘使用情况 | `df -h` |
| `du -sh <dir>` | 查看目录占用空间 | `du -sh ~/autodl-tmp` |
| `lsblk` | 查看磁盘分区 | `lsblk` |
| `mount` / `umount` | 挂载 / 卸载文件系统 | `mount /dev/sdb1 /mnt` |
| `find` | 搜索文件 | `find / -name "*.log"` |
| `xargs` | 将命令输出作为参数传递 | `find . -name "*.log" | xargs rm -f` |

---

## 3️⃣ 网络管理与调试

| 命令 | 说明 | 示例 |
|------|------|------|
| `ping` | 测试网络连通性 | `ping 8.8.8.8` |
| `traceroute` | 路由追踪 | `traceroute www.google.com` |
| `netstat -tulnp` | 查看网络端口占用 | `netstat -tulnp` |
| `ss -tulwn` | 更现代的端口/连接查看 | `ss -tulwn` |
| `curl` | HTTP 请求与接口调试 | `curl -X POST http://127.0.0.1:8000/v1/score -d '{"input":"测试"}' -H "Content-Type: application/json"` |
| `wget` | 下载文件 | `wget http://example.com/file.zip` |

---

## 4️⃣ 日志分析与文本处理

| 命令 | 说明 | 示例 |
|------|------|------|
| `tail -f <file>` | 实时查看日志 | `tail -f /var/log/syslog` |
| `less +F <file>` | 分页 + 实时跟踪 | `less +F log/server.log` |
| `grep` | 搜索文本 | `grep ERROR log/server.log` |
| `awk` | 文本列处理 | `awk '{print $1, $5}' log/server.log` |
| `sed` | 文本替换/处理 | `sed -i 's/old/new/g' file.txt` |
| `cut` | 按列提取 | `cut -d',' -f2 data.csv` |
| `sort | uniq -c` | 排序并统计重复行 | `cat data.log | sort | uniq -c | sort -nr` |

---

## 5️⃣ 进程与作业管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `ps aux` | 查看所有进程 | `ps aux | grep mongod` |
| `jobs` | 查看后台作业 | `jobs` |
| `fg` / `bg` | 前台 / 后台作业切换 | `fg %1` |
| `kill <pid>` | 杀掉进程 | `kill 2221` |
| `kill -9 <pid>` | 强制杀掉进程 | `kill -9 2221` |
| `nohup <cmd> &` | 后台运行并忽略挂起 | `nohup python server.py > server.log 2>&1 &` |

---

## 6️⃣ 系统调试与依赖管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `ldd <binary>` | 查看可执行文件依赖动态库 | `ldd ~/RAG/mongodb-7.0.20/bin/mongod` |
| `strace` | 跟踪系统调用 | `strace -p 2182` |
| `lsof` | 查看文件或端口占用 | `lsof -i :27017` |
| `dpkg -i <file.deb>` | 安装 Debian 包 | `sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb` |
| `apt-get install <pkg>` | 安装系统包 | `sudo apt-get install htop` |
| `ldconfig` | 刷新动态库缓存 | `sudo ldconfig` |

---

## 7️⃣ Shell 编程与自动化技巧

| 命令/技巧 | 说明 | 示例 |
|------------|------|------|
| `&&` | 条件执行 | `mkdir log && cd log` |
| `||` | 条件执行（前命令失败执行） | `command1 || command2` |
| `>` / `>>` | 输出重定向 | `echo "hello" > file.txt` |
| `|` | 管道 | `cat file.txt | grep ERROR` |
| `$()` | 命令替换 | `echo $(date)` |
| `export VAR=value` | 设置环境变量 | `export PATH=/usr/local/bin:$PATH` |
| `source <file>` | 执行脚本，导入变量 | `source config.sh` |
| `crontab -e` | 设置定时任务 | `0 3 * * * /home/user/backup.sh` |
| `watch` | 定期执行命令 | `watch -n 2 "df -h"` |

---

## 8️⃣ 容器与进阶服务管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `docker ps` | 查看运行容器 | `docker ps -a` |
| `docker logs -f <container>` | 实时跟踪容器日志 | `docker logs -f rag_container` |
| `docker exec -it <container> /bin/bash` | 进入容器 | `docker exec -it rag_container /bin/bash` |
| `systemctl status <service>` | 查看服务状态 | `systemctl status mongod` |
| `systemctl restart <service>` | 重启服务 | `systemctl restart mongod` |

---

## 9️⃣ 小技巧与快捷组合

```bash
# 查看占用端口的进程
lsof -i :27017

# 统计日志中错误出现次数
grep ERROR log/*.log | wc -l

# 查找大文件
find / -type f -size +100M

# 实时查看日志并过滤
tail -f log/server.log | grep ERROR

# 后台启动服务并记录日志
nohup python server.py > log/server.log 2>&1 &
