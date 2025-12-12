import torch
import json
import random
from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator
from pretrain import LLM, Config, LLMDataset

def generate_mock_data(num_samples=100, output_file="mock_data.jsonl"):
    """生成100条伪数据用于训练演示"""
    mock_texts = [
        "今天天气很好，适合出门散步。",
        "人工智能技术正在快速发展。",
        "机器学习是计算机科学的重要分支。",
        "深度学习需要大量的数据支持。",
        "自然语言处理是AI的核心技术之一。",
        "计算机视觉应用广泛。",
        "强化学习在游戏领域表现出色。",
        "大语言模型改变了人机交互方式。",
        "Transformer架构是现代AI的基础。",
        "注意力机制是深度学习的重要突破。",
        "线性注意力机制提高了计算效率。",
        "MoE架构可以扩展模型规模。",
        "训练大模型需要大量计算资源。",
        "GPU加速是深度学习的关键。",
        "分布式训练可以处理更大模型。",
        "数据质量决定了模型性能。",
        "预训练+微调是常见的训练策略。",
        "模型压缩技术使大模型更实用。",
        "量化技术减少了模型存储需求。",
        "剪枝可以提高模型推理速度。",
        "知识蒸馏可以将大模型知识转移到小模型。",
        "多模态学习结合了不同类型的数据。",
        "图神经网络处理结构化数据。",
        "时间序列分析预测未来趋势。",
        "异常检测在安全领域很重要。",
        "推荐系统改善了用户体验。",
        "搜索技术让信息获取更高效。",
        "语音识别技术日趋成熟。",
        "机器翻译打破了语言障碍。",
        "自动驾驶技术正在逐步实现。",
        "机器人技术改变了制造业。",
        "医疗AI辅助医生诊断疾病。",
        "金融科技提高了金融服务效率。",
        "教育个性化技术改善学习效果。",
        "智慧城市利用AI优化城市管理。",
        "农业智能化提高了农作物产量。",
        "AI伦理问题需要认真考虑。",
        "算法公平性是AI发展的重要议题。",
        "隐私保护在AI时代尤为重要。",
        "可解释AI增加了模型透明度。",
        "联邦学习保护了用户数据隐私。",
        "对抗训练提高了模型鲁棒性。",
        "零样本学习扩展了模型应用范围。",
        "少样本学习降低了数据需求。",
        "自监督学习利用未标记数据。",
        "对比学习学习数据表示。",
        "生成模型可以创造新的内容。",
        "GANs可以生成逼真的图像。",
        "扩散模型在图像生成中表现出色。",
        "VAEs学习数据的潜在表示。",
        "强化学习通过奖励机制学习。",
        "模仿学习从专家演示中学习。",
        "元学习让模型学会如何学习。",
        "持续学习适应新数据而不遗忘。",
        "迁移学习利用已有知识解决新问题。",
        "多任务学习同时处理多个任务。",
        "课程学习从简单到复杂逐步学习。",
        "主动学习选择最有价值的样本标注。",
        "增量学习适应新数据而不重新训练。",
        "在线学习实时适应数据变化。",
        "终身学习持续积累知识。",
        "神经架构搜索自动设计网络结构。",
        "AutoML让机器学习更易用。",
        "边缘计算将AI部署到设备端。",
        "云计算提供了强大的计算能力。",
        "量子计算可能带来AI革命。",
        "神经网络模拟人脑工作原理。",
        "卷积神经网络擅长处理图像。",
        "循环神经网络适合处理序列数据。",
        "LSTM解决了长期依赖问题。",
        "GRU是LSTM的简化版本。",
        "注意力机制让模型关注重要信息。",
        "位置编码帮助模型理解序列顺序。",
        "层归一化稳定了训练过程。",
        "残差连接解决了深度网络退化问题。",
        "激活函数增加了网络非线性。",
        "Dropout防止模型过拟合。",
        "正则化技术提高模型泛化能力。",
        "优化算法决定了模型学习效果。",
        "学习率调度影响训练稳定性。",
        "批量归一化加速了模型收敛。",
        "数据增广提高了模型鲁棒性。",
        "特征工程改善了模型输入质量。",
        "模型集成提高了预测准确性。",
        "交叉验证评估模型性能。",
        "超参数调优找到最佳配置。",
        "早停防止模型过拟合。",
        "模型选择找到最适合的算法。",
        "特征选择减少了数据维度。",
        "降维技术简化了数据表示。",
        "聚类分析发现数据内在结构。",
        "分类预测离散标签。",
        "回归预测连续值。",
        "异常检测识别异常模式。",
        "关联规则发现数据间关系。",
        "时间序列预测未来值。",
        "文本分析从文本中提取信息。",
        "情感分析判断文本情感倾向。",
        "实体识别找出文本中的命名实体。",
        "关系抽取识别实体间关系。",
        "主题建模发现文本主题。",
        "问答系统回答用户问题。",
        "对话系统进行人机对话。",
        "摘要生成提取文本要点。",
        "文本续写预测后续内容。",
        "风格迁移改变文本风格。",
        "文本校正修正文本错误。",
        "机器翻译实现跨语言交流。",
        "语音合成将文字转换为语音。",
        "语音识别将语音转换为文字。",
        "说话人识别识别说话人身份。",
        "情感识别判断语音情感。",
        "音乐生成创作新的音乐。",
        "图像分类识别图像内容。",
        "目标检测定位图像中的对象。",
        "图像分割划分图像区域。",
        "人脸识别验证个人身份。",
        "图像生成创造新的图像。",
        "风格迁移改变图像风格。",
        "图像超分辨率提高图像质量。",
        "图像去噪去除图像噪声。",
        "图像修复恢复损坏的图像。",
        "视频分析理解视频内容。",
        "行为识别识别人体动作。",
        "场景理解分析环境信息。",
        "3D重建从2D图像创建3D模型。",
        "虚拟现实创造沉浸式体验。",
        "增强现实将虚拟信息叠加到现实世界。"
    ]
    
    # 确保有足够的样本
    while len(mock_texts) < num_samples:
        mock_texts.extend(mock_texts)
    
    selected_texts = random.sample(mock_texts, num_samples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in selected_texts:
            data = {"text": text}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"已生成 {num_samples} 条伪数据，保存到 {output_file}")

def main():
    # 生成伪数据
    generate_mock_data(num_samples=100)
    
    # 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    except:
        print("未找到本地tokenizer，使用默认tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
    
    # 配置模型
    config = Config(
        hidden_size=256,
        num_attention_heads=8,  # 减小模型规模以便演示
        num_key_value_heads=2,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        flash_attn=True,
        max_seq_len=256,  # 减小序列长度以便演示
        intermediate_size=512,
        vocab_size=tokenizer.vocab_size,
        n_layers=4,  # 减少层数以便演示
        softmax_attn_index=[1, 3],
        dropout=0.1,
        expert_num=4,  # 减少专家数量以便演示
        topk=2,
        output_router_logits=True,
        aux_loss_coef=0.01
    )
    
    # 创建模型
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # 准备训练参数
    data_collator = DefaultDataCollator()
    args = TrainingArguments(
        output_dir='./demo_result', 
        num_train_epochs=3,  # 训练2个epoch以便演示
        do_train=True, 
        per_device_train_batch_size=8,  # 减小批量大小以便演示
        gradient_accumulation_steps=2,
        logging_steps=1,
        report_to='tensorboard',  # 启用tensorboard记录
        save_total_limit=2, 
        save_steps=50,
        learning_rate=5e-4,  # 提高学习率以便演示
        lr_scheduler_type='cosine',
        dataloader_num_workers=0,  # 设置为0以便演示
        dataloader_pin_memory=False,
        save_safetensors=False,
        overwrite_output_dir=True  # 覆盖输出目录以便重复运行
    )          
    
    # 创建数据集
    dataset = LLMDataset('./mock_data.jsonl', tokenizer=tokenizer, max_seq_len=256)
    
    # 创建训练器
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=dataset, 
        tokenizer=tokenizer, 
        data_collator=data_collator
    )
    
    # 开始训练
    print("开始训练演示...")
    trainer.train(resume_from_checkpoint=False)
    
    # 保存模型
    trainer.save_model('./demo_saves')
    trainer.save_state()
    print("训练演示完成，模型已保存到 ./demo_saves")

if __name__ == '__main__':
    main()