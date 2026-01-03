#!/bin/bash

# ViT训练脚本 - 使用Hugging Face模型
# 用途：在花卉数据集上微调ViT模型

# ============================================
# 配置部分 - 根据你的需求修改
# ============================================

# 训练配置
NUM_CLASSES=5              # 类别数（5种花）
EPOCHS=10                  # 训练轮数
BATCH_SIZE=16              # 批次大小（根据GPU显存调整）
LR=0.001                   # 初始学习率
LRF=0.01                   # 最终学习率比例

# 数据集路径
DATA_PATH="dataset/flower_photos"

# 模型配置
MODEL_PATH="google/vit-base-patch16-224-in21k" #本地模型路径
FREEZE_LAYERS=true         # 是否冻结backbone（true: 只训练分类头，false: 全参数微调）
DEVICE="cuda:0"            # 设备（cuda:0, cpu）

# ============================================
# 训练模式选择
# ============================================

echo "=========================================="
echo "ViT训练配置"
echo "=========================================="
echo "类别数: $NUM_CLASSES"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LR"
echo "数据集: $DATA_PATH"
echo "冻结backbone: $FREEZE_LAYERS"
echo "设备: $DEVICE"
echo "=========================================="

# 检查数据集是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据集路径不存在: $DATA_PATH"
    echo "请确保数据集已正确放置"
    exit 1
fi

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 数据集路径不存在: $DATA_PATH"
    echo "请确保数据集已正确放置"
    exit 1
fi

# 检查CUDA是否可用
if [ "$DEVICE" == "cuda:0" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用，请使用DEVICE=cpu'" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "警告: CUDA不可用，自动切换到CPU"
        DEVICE="cpu"
    fi
fi

# 创建weights目录
mkdir -p weights

# 运行训练
python train.py \
    --model_path  $MODEL_PATH\
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --lrf $LRF \
    --data-path $DATA_PATH \
    --freeze-layers $FREEZE_LAYERS \
    --device $DEVICE

echo "=========================================="
echo "训练完成！"
echo "模型保存在: weights/"
echo "=========================================="