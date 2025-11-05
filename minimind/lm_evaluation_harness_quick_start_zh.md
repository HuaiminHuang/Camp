# lm-evaluation-harness 快速上手指南

本文档旨在帮助您快速安装并使用 `lm-evaluation-harness` 框架，并简要介绍其中常用的中文评测基准。

## 1. 核心概念

`lm-evaluation-harness` 是一个用于在大量标准学术基准上评估语言模型 (LLM) 的统一框架。它支持多种模型类型（如 Hugging Face, vLLM）和数百个评测任务。

## 2. 安装

首先，克隆官方仓库并安装核心依赖。

```bash
# 1. 克隆仓库
git clone https://github.com/EleutherAI/lm-evaluation-harness

# 2. 进入目录
cd lm-evaluation-harness

# 3. 安装框架
pip install -e .
```

## 3. 基本用法

评估一个模型的基本命令格式如下：

```bash
lm_eval --model <模型类型> \
    --model_args <模型参数> \
    --tasks <任务名称> \
    --device <设备> \
    --batch_size <批次大小>
```

**示例：评估一个 Hugging Face Hub 上的模型**

假设您想在 `hellaswag` 任务上评估 `EleutherAI/pythia-160m` 模型：

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

*   `--model hf`: 指定模型类型为 Hugging Face `transformers`。
*   `--model_args pretrained=...`: 指定模型的具体路径或 Hub 名称。
*   `--tasks hellaswag`: 指定要运行的评测任务。
*   **列出所有可用任务**: `lm_eval --tasks list`

## 4. 常用中文评测基准详解

通常，这些基准都是通过**多项选择题**的形式来评估模型的。其核心思想是，向模型提出一个问题和几个选项，然后判断模型给出的答案是否与正确答案一致。

*   **通用评测机制 (General Mechanism):**
    1.  **Prompt 构建**: 将问题和选项组合成一个标准的输入格式（Prompt）。
    2.  **模型推理**: 将构建好的 Prompt 输入给语言模型，让它生成答案。
    3.  **答案提取**: 从模型的输出中解析出它选择的选项（例如，"A", "B", "C" 或 "D"）。
    4.  **准确率计算**: 对比模型的答案和标准答案，计算准确率（Accuracy）。

*   **通用计算方法 (General Calculation):**
    最核心的指标是**准确率 (Accuracy)**。
    `Accuracy = (回答正确的题目数量) / (总题目数量)`
    在多分类任务中，通常使用 `acc_norm`（归一化准确率），它会处理一些格式问题，确保答案被正确匹配。

---

### C-Eval

*   **指标含义**: **C-Eval** 是一个全面的**中文基础模型评估套件**。它涵盖了从中学到大学专业水平的多个学科，旨在评估语言模型在中文知识和推理能力方面的表现。
*   **评测原理**:
    *   **知识广度与深度**: C-Eval 的题目覆盖了人文、社科、理工等四大类共 52 个不同学科，能够全面考察模型的知识储备。
    *   **难度分级**: 题目分为不同难度级别（例如：中学、大学、专业），可以测试模型在不同认知层次上的能力。
    *   **零样本/少样本学习**: 通常在零样本（Zero-shot）或五样本（5-shot）的设置下进行评测。
        *   **Zero-shot**: 直接给模型题目让它回答，最能考验模型的原始能力。
        *   **5-shot**: 在提问前，先给模型 5 个同类问题的“问题+答案”作为示例，考验模型的学习和模仿能力。

### CMMLU

*   **指标含义**: **CMMLU** (Chinese Massive Multitask Language Understanding) 是一个专门用于评估语言模型在**中文语境下的知识和推理能力**的基准，可以看作是著名英文评测 MMLU 的中文版。
*   **评测原理**:
    *   **多任务理解**: 包含从基础学科到高级专业领域的 67 个主题，涵盖了需要计算、推理和专业知识的各种任务。
    *   **文化适应性**: 题目设计考虑了中国的教育体系、文化背景和常用表达，能更准确地反映模型对中文世界的理解。
    *   **评估重点**: 主要评估模型在处理中文语言任务时的知识应用和问题解决能力。与 C-Eval 类似，它也常在零样本和少样本（通常是 5-shot）设置下进行。

### AClue

*   **指标含义**: **AClue** (Alignment CLUE) 是一个关注语言模型**对齐能力**的中文评测基准。它不仅仅评估模型的知识，更侧重于评估模型是否能理解并遵守人类的指令、价值观和偏好。
*   **评测原理**:
    *   **对齐 (Alignment)**: 这是 AClue 的核心。它想知道模型是否能生成有用的、诚实的、无害的回答。
    *   **评测维度**:
        *   **基础能力**: 包括对话、知识问答、代码生成等。
        *   **高级能力**: 考察模型是否具有创造性、能否进行逻辑推理。
        *   **价值观对齐**: 评估模型在处理涉及伦理、偏见和安全性的问题时，能否给出符合社会规范和人类价值观的回答。
    *   **评测方法**: AClue 的评测方式更多样，除了选择题，还可能包括对模型生成的开放式回答进行打分，有时甚至需要人工评估或使用更强的模型（如 GPT-4）作为“裁判”来打分。

### TMMLU

*   **指标含义**: **TMMLU** (Traditional Chinese Massive Multitask Language Understanding) 是 MMLU 的**繁体中文版本**。
*   **评测原理**:
    *   **语言和文化焦点**: 专门为评估模型在**繁体中文**环境下的表现而设计。虽然和简体中文（CMMLU）在很多知识上是共通的，但它包含了台湾、香港等地区特有的用词、文化和教育背景。
    *   **适用场景**: 如果一个模型的目标用户是繁体中文使用者，那么 TMMLU 是一个比 CMMLU 或 C-Eval 更具针对性的评测基准。
    *   **评测机制**: 与 CMMLU 和 C-Eval 相同，主要采用多项选择题的形式，通过计算准确率来评估模型在各个学科上的知识水平。

## 5. 中文评测示例

假设您想使用一个本地模型（路径为 `/path/to/your/model`）在 **CMMLU** 基准上进行 5-shot 评测：

```bash
lm_eval --model hf \
    --model_args pretrained=/path/to/your/model \
    --tasks cmmlu \
    --num_fewshot 5 \
    --batch_size 4 \
    --device cuda:0
```
*   `--tasks cmmlu`: 运行 CMMLU 评测。框架会自动加载其下的所有子任务。
*   `--num_fewshot 5`: 指定进行 5-shot 评测。如果为 0，则进行 zero-shot 评测。

## 6. 查看结果

*   使用 `--output_path <文件名.json>` 将结果保存到指定文件。
*   使用 `--log_samples` 可以记录模型的具体输入和输出，便于分析。

```bash
lm_eval --model hf \
    --model_args pretrained=/path/to/your/model \
    --tasks cmmlu \
    --num_fewshot 5 \
    --output_path results.json \
    --log_samples
```

这会生成 `results.json` (包含各科目的准确率) 和一个包含详细样本的文件夹。

