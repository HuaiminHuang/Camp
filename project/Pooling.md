
# 🧠 一、什么是 Pooling 层？

Transformer 输出的是 token 级别的隐向量：

$$H = [h_1, h_2, \dots, h_n] \in \mathbb{R}^{n \times d}$$

要得到整个句子的语义表示，就必须把这些 token 向量汇聚为一个固定维度向量：

$$z = \text{Pool}(H)$$


常见的 Pooling 方式：

| 类型                    | 定义                    | 公式                             | 特征            |
| --------------------- | --------------------- | ------------------------------ | ------------- |
| **[CLS] Pooling**     | 取 `[CLS]` token 的隐藏状态 | ( $z = h_{\text{[CLS]}}$ )       | 用于分类任务，强依赖预训练 |
| **Mean Pooling**      | 所有 token 向量求平均        | ( $z = \frac{1}{n} \sum_i h_i$ ) | 表征平滑、语义一致     |
| **Max Pooling**       | 对每维取最大值               | ( $z_j = \max_i h_{i,j}$ )       | 捕捉最强激活特征，较稀疏  |
| **Attention Pooling** | 加权平均                  | ( $z = \sum_i \alpha_i h_i$ )    | 自适应关注核心 token |

---

# 🧩 二、为什么 Embedding 模型多用 **Mean Pooling**

## ✅ 1️⃣ Mean Pooling 的语义平衡性最好

Transformer 的输出 token 向量是上下文相关的分布式表征，不同 token 承载不同语义。
平均可以看作在语义空间中做“质心计算”，即：

> 得到一个**语义中心向量**，代表整句话的总体语义。

效果：

* 对于语义检索（cosine similarity 计算），平均向量的平滑特性 → 向量空间更线性、更稳定；
* 避免单个 token（如 [CLS] 或 “!”）主导句子语义。

**公式直观理解：**
$$\text{Mean}(h_1,\dots,h_n) = \frac{1}{n}\sum h_i$$
相当于在语义空间里取几何中心，更利于余弦相似度衡量。

---

## ✅ 2️⃣ 相比 [CLS]，Mean Pooling 对预训练依赖更低

在像 **BERT / RoBERTa** 这类模型中：

* [CLS] 向量在预训练时主要用于 NSP（下一句预测）或分类任务；
* 但并未显式训练成句子级语义向量；
* 因此直接取 [CLS] 作为 embedding 通常表现差。

而 Mean Pooling：

* 聚合了所有 token 的信息；
* 不依赖特定 token 的功能性训练；
* 在迁移学习中更稳健。

例如：

> Sentence-BERT / SimCSE / BGE / E5 等 embedding 模型均使用 Mean Pooling，且在 STS-B、MTEB 上远超 [CLS]。

---

## ✅ 3️⃣ 相比 Max Pooling，Mean Pooling 梯度更稳定

Max Pooling 在训练时：

* 仅最大激活的 token 参与反向传播；
* 导致梯度稀疏；
* 训练波动大，embedding 空间不连续。

Mean Pooling 则：

* 每个 token 都参与梯度传播；
* 学习信号分布均匀；
* 优化更稳定、收敛更快。

在对比学习（InfoNCE、TripletLoss）中，这个特性尤其重要，因为：

> 模型要在大规模文本对中稳定地学习语义距离关系。

---

## ✅ 4️⃣ 与相似度计算（cosine）协同更自然

Embedding 模型的最终相似度通常用 **余弦相似度**：
$$\text{sim}(q, d) = \frac{q \cdot d}{|q||d|}$$
Mean Pooling 得到的向量更接近高斯分布中心，归一化后向量范数分布更均匀，
→ 余弦距离更稳定，空间几何更线性。

而 Max Pooling 产生稀疏、偏斜的分布，会导致向量范数差异较大，
→ 影响相似度的一致性。

---

## ✅ 5️⃣ 实证结果支持：Mean Pooling 一般性能最优

在公开评测（MTEB, STS, NLI）中：

| Pooling 类型        | STS-B Spearman | NLI Accuracy | 备注              |
| ----------------- | -------------- | ------------ | --------------- |
| [CLS]             | 65~70          | 不稳定          | 强依赖预训练任务        |
| Max Pooling       | 72~76          | 中等           | 稀疏表征            |
| **Mean Pooling**  | **78~84**      | **最稳定**      | 主流 embedding 默认 |
| Attention Pooling | ≈Mean          | 成本略高         |                 |

几乎所有顶级 embedding 模型（E5, GTE, BGE, Qwen3-embedding, OpenAI text-embedding-3-large）都采用 **Mean Pooling + LayerNorm**。

---

# ⚙️ 三、Qwen3 Embedding 的具体实现

在 Qwen3 的 embedding pipeline 中：

```python
hidden_states = encoder(input_ids, attention_mask)
# attention_mask 用于避免 padding token 参与平均
embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
embeddings = F.normalize(embeddings, p=2, dim=1)
```

这里正是标准的 **Masked Mean Pooling + L2 Normalization**。

* Mask 让 padding 不影响平均；
* L2 归一化让向量适配 cosine 相似度。

---

# 四、[CLS] Pooling 的原理与动机

### 1️⃣ [CLS] 是什么？

在 BERT 架构中，输入的第一个 token 是一个特殊标记 `[CLS]`（classification token）。
BERT 在预训练阶段（尤其是 NSP 任务中）被显式地教导去让 `[CLS]` 向量代表整个句子的语义摘要。

输入序列示例：

```
[CLS] The weather is nice today. [SEP]
```

Transformer 的输出：

```
H = [h_CLS, h_1, h_2, h_3, ..., h_SEP]
```

### 2️⃣ Pooling 过程

**[CLS] pooling** 就是直接取输出矩阵中第一个向量：
[
z = h_{\text{CLS}}
]

这个向量已经在预训练任务中被优化为“全局语义代表”，
因此可以直接用于分类或句子表示任务。

### 3️⃣ 特性

| 特征          | 说明                                                                      |
| :---------- | :---------------------------------------------------------------------- |
| **可学习**     | `[CLS]` 的语义是通过预训练任务学出来的，而非简单数学平均                                        |
| **全局上下文融合** | 每一层 Transformer 的 self-attention 都允许 `[CLS]` attend 所有 token，因此它能综合整句信息 |
| **信息流方向**   | 信息不断聚合到 `[CLS]` 这个 token 上（它像一个“语义收集器”）                                 |

---

### 🧮 、数学角度理解信息流

设输入为：

$$X = [x_{\text{CLS}}, x_1, x_2, ..., x_L]$$

经过 Transformer 层：

$$H^{(l)} = \text{MHA}(H^{(l-1)}) + \text{FFN}(H^{(l-1)})$$

在每一层中：

$$h_{\text{CLS}}^{(l)} = \sum_{i=1}^L \alpha_{\text{CLS},i}^{(l)} W_V h_i^{(l-1)}$$


可以看到：

* `[CLS]` 向量通过注意力权重 ( \alpha_{\text{CLS},i} ) 聚合了整个序列的信息；
* 每一层都增强 `[CLS]` 的全局感知能力；
* 最终层的 `[CLS]` 就成了整个输入的语义中心。

于是我们只需：

$$s(q,d) = w^T h_{\text{CLS}}^{(L)} + b$$

即可作为最终的打分结果。

---

### 🧩从任务需求角度看 [CLS] Pooling 的合理性

| 特征                     | 说明                                        |
| :--------------------- | :---------------------------------------- |
| **输入是 Query–Doc pair** | 需要一个联合表示，而非独立句向量                          |
| **输出是单一得分**            | 不需要多维 embedding，只需 scalar                 |
| **模型已预训练 [CLS] 表征语义**  | BERT/RoBERTa 的预训练目标正好契合                   |
| **全交互 Attention**      | `[CLS]` 可以收集所有交互信息                        |
| **对齐 ranking 任务结构**    | Pairwise / Listwise Loss 直接作用于 `[CLS]` 输出 |

因此在架构与目标函数的契合度上，`[CLS]` pooling 几乎是“天然匹配”的方案。

---

### 🧠CLS 在实际 Reranker 中的体现

| 模型                             | Pooling                   | 输出层           | 损失函数                |
| :----------------------------- | :------------------------ | :------------ | :------------------ |
| **MonoBERT**                   | [CLS]                     | Linear(768→1) | BCE                 |
| **MonoT5**                     | Encoder [CLS] 输出到 Decoder | Cross-Entropy |                     |
| **Qwen-Reranker**              | [CLS]                     | Linear        | Pairwise / Listwise |
| **RankGPT / DeBERTa-Reranker** | [CLS]                     | Linear / MLP  | Pairwise Ranking    |
| **RocketQA Cross-Encoder**     | [CLS]                     | Linear        | Margin Ranking Loss |

这些模型都遵循相同范式：

> `[CLS] → Linear → Score → Loss`

---

## ✅ 总结一句话

> Reranker 使用 `[CLS]` pooling 的根本原因是：
>
> * 输入是「Query+Document」联合序列；
> * `[CLS]` 在每层 attention 中都聚合了全局交互语义；
> * 任务目标是输出一个相关性分数；
> * 预训练阶段模型已学会将 `[CLS]` 作为全局语义汇聚向量。

换言之：

> **[CLS] pooling 是一种“任务感知的、可学习的全局语义聚合机制”，**
> 它让 Reranker 能直接把复杂的 query–doc 交互浓缩成一个打分表示。
