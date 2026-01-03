# CLIP 模型学习与实践项目

本项目旨在深入学习和实践 OpenAI 的 CLIP（Contrastive Language-Image Pre-training）模型。经过整理，项目结构更加清晰，包含了对 CLIP 原理的研究报告、代码实现示例、在 CIFAR-100 数据集上的零样本预测与微调实验，以及完整的环境配置。

> **重要说明**：`CLIP/` 目录为 OpenAI 官方仓库的克隆（[openai/CLIP](https://github.com/openai/CLIP)），本 README 主要介绍在此基础上添加的学习与实验材料。

## 📁 项目结构（整理后）

```
.
├── README.md                           # 本文件
├── enviroment.yaml                     # Conda 环境配置文件（包含全部依赖）
├── CLIP/                               # OpenAI 官方 CLIP 仓库（子模块，无需修改）
├── docs/                               # 研究文档与资料
│   ├── CLIP_Research_Report.md         # CLIP 模型技术研究报告（详细解析）
│   ├── clip.mmd                        # CLIP 原始论文（Learning Transferable Visual Models...）的 Markdown 版本
│   ├── GraphToClip.md                  # CLIP 三阶段工作流程的 Mermaid 流程图
│   └── images/                         # 研究报告和论文中引用的图片
│       ├── 1_0.jpg                     # CLIP 整体框架图
│       ├── 2_0.jpg                     # 训练效率对比图
│       └── ...（共19张图片）
├── scripts/                            # 实验代码
│   ├── quick_start.py                  # CLIP 快速使用示例（需从 scripts 目录运行）
│   ├── fine_tuning_logistic_regression.py  # 使用逻辑回归在 CIFAR-100 上微调 CLIP
│   ├── zero_shot_prediction_for_cifar100.py # CIFAR-100 零样本预测（可视化 Top-5 结果）
│   └── test_acc.py                     # 全量评估 CLIP 在 CIFAR-100 上的 Top-1/Top-3/Top-5 准确率
└── outputs/                            # 实验输出与日志
    ├── acc.log                         # test_acc.py 运行的日志（记录准确率）
    ├── CIFAR100.log                    # zero_shot_prediction_for_cifar100.py 运行的日志
    └── res/                            # 可视化结果图片
        ├── batch_res_0.png
        ├── batch_res_1.png
        ├── batch_res_2.png
        ├── batch_res_3.png
        └── clip_prediction_result_3637.png
```

## 🚀 快速开始

### 环境配置

#### 使用 Conda 环境（推荐）

项目已提供完整的 `enviroment.yaml` 文件，可直接创建环境：

```bash
conda env create -f enviroment.yaml
conda activate ClipModel
```

#### 手动安装

若需单独安装，可参考以下步骤：

1. 安装 PyTorch（>=1.7.1）与 torchvision（请根据 CUDA 版本调整）：
   ```bash
   conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
   ```

2. 安装 CLIP 官方包及其依赖：
   ```bash
   pip install ftfy regex tqdm
   pip install git+https://github.com/openai/CLIP.git
   ```

3. 安装其他实验所需的包：
   ```bash
   pip install scikit-learn matplotlib tqdm
   ```

### 运行示例

**注意**：所有脚本均位于 `scripts/` 目录下，运行时请确保在该目录内或使用正确路径。

#### 1. 快速开始示例

```bash
cd scripts
python quick_start.py
```

该脚本加载 ViT-B/32 模型，对 `../CLIP/CLIP.png` 图片进行预测，输出三个候选文本（"a diagram", "a dog", "a cat"）的匹配概率。

#### 2. 零样本预测（CIFAR-100）

```bash
cd scripts
python zero_shot_prediction_for_cifar100.py
```

该脚本使用 CLIP 对 CIFAR-100 测试集进行零样本预测，并保存前 4 张图片的预测可视化结果到 `../outputs/res/` 目录。

#### 3. 全量评估

```bash
cd scripts
python test_acc.py
```

评估 CLIP（ViT-L/14）在 CIFAR-100 测试集上的 Top-1、Top-3、Top-5 准确率。结果参考 `../outputs/acc.log`。

#### 4. 微调实验

```bash
cd scripts
python fine_tuning_logistic_regression.py
```

使用逻辑回归在 CIFAR-100 训练集上微调 CLIP 的图像特征，并在测试集上评估（预期准确率约 86.7%）。

## 📚 研究资料

- **`docs/CLIP_Research_Report.md`**：详细的技术研究报告，涵盖 CLIP 的核心思想、模型架构、对比学习机制、零样本推理、训练策略、关键创新点、实验结果以及社会影响等内容。
- **`docs/clip.mmd`**：CLIP 原始论文《Learning Transferable Visual Models From Natural Language Supervision》的 Markdown 版本，便于本地查阅。
- **`docs/GraphToClip.md`**：使用 Mermaid 绘制的 CLIP 三阶段工作流程图（对比预训练 → 创建文本分类器 → 零样本预测），直观展示模型的工作流程。

## 📊 实验结果

### 零样本性能（ViT-L/14 on CIFAR-100）
- **Top-1 Accuracy**: 73.22%（参见 `outputs/acc.log`）
- **Top-3 Accuracy**: 89.xx%（运行 `scripts/test_acc.py` 可获取）
- **Top-5 Accuracy**: 93.xx%（运行 `scripts/test_acc.py` 可获取）

### 微调后性能（Logistic Regression）
- **Accuracy**: ~86.74%（参见 `scripts/fine_tuning_logistic_regression.py` 输出）

## ⚙️ 技术细节

### 脚本路径说明
由于项目结构整理，脚本中的路径已做相应调整：
- `quick_start.py`：图片路径指向 `../CLIP/CLIP.png`
- `zero_shot_prediction_for_cifar100.py`：输出目录指向 `../outputs/res/`
- 其他脚本无路径依赖

### 数据存储
- CIFAR-100 数据集会自动下载到 `~/.cache/` 目录
- 实验输出（日志、图片）保存到 `outputs/` 目录
- 研究文档和图片统一存放在 `docs/` 目录

## 🔧 自定义与扩展

### 修改实验参数
- 在 `scripts/` 下的各个 Python 文件中，可以调整模型类型（如 'ViT-B/32', 'ViT-L/14'）、批次大小等参数。
- 零样本预测中的 prompt 模板可修改，观察对准确率的影响。

### 适配其他数据集
代码可轻松适配到其他图像分类数据集（如 ImageNet、CIFAR-10 等），主要修改数据加载部分。

### 特征可视化
可扩展代码，使用 t-SNE 或 PCA 对图像/文本特征进行降维可视化，观察跨模态对齐效果。

## 📝 注意事项

1. **CLIP 子模块**：`CLIP/` 目录为官方仓库克隆，仅供参考和使用，不建议直接修改。
2. **GPU 显存**：使用较大模型（如 ViT-L/14）时，请根据显存调整 batch size（脚本中已设置为 32-64）。
3. **环境兼容性**：`enviroment.yaml` 是在 Linux 环境下导出的，若在 Windows 或 macOS 上使用，可能需要调整部分依赖。
4. **运行目录**：所有脚本设计为在 `scripts/` 目录下运行，确保相对路径正确。

## 🧠 后续学习建议

1. **深入理解原理**：通过 `docs/CLIP_Research_Report.md` 深入理解 CLIP 的设计思想与技术细节。
2. **Prompt 工程**：尝试在零样本预测中修改 prompt 模板（如 `"a photo of a {label}"`），观察对准确率的影响。
3. **多模态应用**：探索 CLIP 在图像检索、文本到图像生成、视频理解等任务中的应用。
4. **扩展模型**：了解 OpenCLIP、Chinese-CLIP 等社区改进版本，尝试更大规模的多模态预训练。
5. **理论分析**：研究对比学习理论、跨模态表示对齐的数学基础。

## 📖 参考文献

- OpenAI CLIP 官方仓库：https://github.com/openai/CLIP
- 论文：Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. arXiv:2103.00020.
- 博客：https://openai.com/blog/clip/
- OpenCLIP：https://github.com/mlfoundations/open_clip

## 📄 许可证

本学习项目中的代码和文档遵循 MIT 许可证。`CLIP/` 子模块遵循其原始许可证（详见 `CLIP/LICENSE`）。

---

*本项目已整理为清晰的结构，方便学习和研究使用。如有问题或建议，欢迎反馈。*