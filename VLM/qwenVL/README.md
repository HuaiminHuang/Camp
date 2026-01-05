# Qwen2.5-VL 学习与实验项目

本项目旨在深入学习和理解 Qwen2.5-VL（通义千问2.5视觉语言模型）的技术细节、架构设计及其应用。项目包含官方代码、简化实现、技术报告、学习笔记以及API测试示例。

## 📁 项目结构说明

```
.
├── README.md                           # 项目说明文档
├── .env                                # 环境变量配置（DashScope API Key）
├── docs/                               # 文档资料
│   ├── qwen2_5VL.mmd                   # Qwen2.5-VL 学习笔记（思维导图）
│   └── Qwen2.5-VL_Technical_Report.md  # Qwen2.5-VL 技术报告（官方）
├── official/                           # 官方实现代码（HuggingFace Transformers）
│   ├── __init__.py
│   ├── configuration_qwen2_5_vl.py
│   ├── masking_utils.py
│   ├── modeling_qwen2_5_vl.py
│   ├── modular_qwen2_5_vl.py
│   ├── processing_qwen2_5_vl.py
│   └── simplified_qwen_model.py
├── src/                                # 简化版模型实现（用于学习）
│   ├── annotated_vision_process.py     # 视觉预处理模块（带注释）
│   ├── language_and_head_modules.py    # 语言与头模块
│   ├── vision_modules.py               # 视觉模块
│   ├── run_model_test.py               # 模型运行测试
│   └── __pycache__/
├── supplementary/                      # 补充学习资料（Qwen3-VL相关）
│   └── qwen3_vl/
│       ├── MRoPE.ipynb                 # MRoPE 机制实验笔记
│       ├── qwen3VL.mmd                 # Qwen3-VL 学习笔记
│       ├── qwen3vl报告解读.md          # Qwen3-VL 报告解读
│       └── images/                     # 示例图片
├── tests/                              # 测试脚本与资源
│   ├── qwenVL_test.py                  # DashScope API 测试脚本
│   └── test_file/                      # 测试用图片与视频
│       ├── MHWs.jpg
│       ├── mila.png
│       └── Monster_Hunter_Wilds_the_fourth_updating.mp4
├── assets/                             # 静态资源
│   └── images/                         # 项目图片资源
│       ├── 0_0.jpg
│       └── 2_0.jpg
├── logs/                               # 日志目录
│   └── test.log                        # 测试日志
└── Qwen3-VL/                           # （忽略）官方 Qwen3-VL 仓库（git clone）
```

## 🎯 各部分详解

### 1. 官方代码 (`official/`)
此目录包含从 HuggingFace Transformers 库中提取的 Qwen2.5-VL 官方实现代码，可用于直接加载预训练模型进行推理或微调。文件说明：
- `configuration_qwen2_5_vl.py`：模型配置类
- `modeling_qwen2_5_vl.py`：核心模型架构
- `processing_qwen2_5_vl.py`：图像/视频预处理
- `masking_utils.py`：掩码生成工具
- `modular_qwen2_5_vl.py`：模块化组件
- `simplified_qwen_model.py`：简化模型接口

### 2. 简化实现 (`src/`)
为深入理解模型架构，本项目提供了一个简化版本的实现，包含以下模块：
- `vision_modules.py`：视觉编码器（ViT）实现，包含窗口注意力、动态分辨率等核心机制
- `language_and_head_modules.py`：语言模型与多模态融合头
- `annotated_vision_process.py`：对官方 `vision_process.py` 的详细注释，解释动态分辨率、动态帧率采样等预处理逻辑
- `run_model_test.py`：本地模型测试脚本（需下载预训练权重）

### 3. 文档资料 (`docs/`)
- **Qwen2.5-VL 技术报告**：官方发布的技术报告，详细阐述模型架构、训练策略、数据构造与实验结果。
- **学习笔记**：对 Qwen2.5-VL 关键技术的总结与思考，包括动态分辨率、MRoPE、文档解析等。

### 4. 补充资料 (`supplementary/`)
虽然本项目重点为 Qwen2.5-VL，但此目录存放了 Qwen3-VL 的相关资料，便于对比学习与知识延伸：
- `MRoPE.ipynb`：对 Multimodal Rotary Position Embedding 的机制实验
- `qwen3vl报告解读.md`：对 Qwen3-VL 技术报告的解读与总结

### 5. 测试脚本 (`tests/`)
- `qwenVL_test.py`：基于 DashScope API 的测试脚本，支持图像描述、视频理解等功能。
  - 需要配置 `DASHSCOPE_API_KEY` 到 `.env` 文件
  - 支持本地图片/视频文件上传与分析
- `test_file/`：测试用的图片与视频文件。

### 6. 静态资源 (`assets/`)
存放项目所需的图片资源，主要来自技术报告中的示意图。

### 7. 日志 (`logs/`)
运行测试脚本时产生的日志文件。

### 8. 忽略目录 (`Qwen3-VL/`)
此目录为从官方仓库 git clone 的 Qwen3-VL 完整项目，与本项目主要内容无关，仅作参考，请忽略。

## 🚀 快速开始

### 环境准备
1. 确保已安装 Python 3.8+。
2. 安装依赖包：
   ```bash
   pip install dashscope openai python-dotenv torch torchvision pillow requests
   ```
3. 复制 `.env.example`（若存在）为 `.env`，并填入你的 DashScope API Key：
   ```
   DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 运行 API 测试
进入项目根目录，执行：
```bash
cd tests
python qwenVL_test.py
```
脚本默认调用 `local_video()` 函数，分析 `test_file/Monster_Hunter_Wilds_the_fourth_updating.mp4` 视频内容。你可以修改 `if __name__ == "__main__":` 部分以测试其他功能。

### 使用官方代码加载模型
```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

### 学习简化实现
阅读 `src/` 目录下的代码，特别是 `annotated_vision_process.py` 中的详细注释，可以帮助理解 Qwen2.5-VL 的动态分辨率、动态帧率采样等核心预处理逻辑。

## 📚 学习路线建议
1. **入门**：阅读 `docs/Qwen2.5-VL_Technical_Report.md` 了解模型整体设计。
2. **代码实践**：运行 `tests/qwenVL_test.py` 体验模型能力。
3. **深入架构**：阅读 `src/annotated_vision_process.py` 和 `src/vision_modules.py` 理解视觉编码器。
4. **对比学习**：参考 `supplementary/qwen3_vl/` 中的资料，了解 Qwen3-VL 的改进点。
5. **官方集成**：查看 `official/` 中的代码，学习如何在 HuggingFace 生态中使用模型。

## 🔧 注意事项
- 本项目重点为 **Qwen2.5-VL**，`Qwen3-VL` 目录仅为补充资料，请勿混淆。
- 使用 DashScope API 需要相应的权限与配额，请确保账户可用。
- 本地运行模型需要较大的 GPU 内存，建议使用 API 测试或降级模型尺寸。

## 📄 许可证
本项目中的代码与文档仅供学习研究使用。官方代码遵循其原始许可证（Apache 2.0），其他资料遵循 CC BY-NC-SA 4.0 许可证。

## 🙏 致谢
- 感谢阿里巴巴 Qwen 团队开源优秀的视觉语言模型。
- 感谢 HuggingFace 社区提供的模型库与工具链。

---
*最后更新：2026年1月5日*