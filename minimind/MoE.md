Google Switch Transformer 公式：
$$L_{aux} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot p_i$$
（您的代码中 $E$ 被并入了常数项，但本质相同）

来详细解释 MoEGate 代码中两种方法是如何**计算** $f_i$ (专家 $i$ **实际被选中的频率**) 和 $p_i$ (专家 $i$ **的平均门控概率**) 的，从而产生“序列级别”和“全局级别”的差异。

---

## 🔬 从公式看代码实现：$f_i$ 与 $p_i$ 的计算

### 1. 全局平均平衡（`self.seq_aux=False`）

**核心思想：** **直接**套用 Google Switch Transformer 的公式，所有 $B \times T$ 个 token 视为一个整体来计算 $f_i$ 和 $p_i$。

| 公式项 | 代码计算 | 解释 |
| :--- | :--- | :--- |
| **$p_i$** (专家 $i$ 的平均门控概率) | `Pi = scores.mean(0)` | $\mathbf{scores}$ (形状 $[B \times T, E]$) 是每个 token 对所有专家的 $\text{softmax}$ 概率。取 $\text{mean}(0)$ 就是对所有 $B \times T$ 个 token 求平均，得到一个 $[E,]$ 向量，**代表所有 token 的平均期望概率分布**。 |
| **$f_i$** (专家 $i$ 实际被选中的频率) | `ce = mask_ce.float().mean(0)` | $\mathbf{mask\_ce}$ (形状 $[B \times T \times top\_k, E]$) 是被选中的 $top\_k$ 专家集合的 $\text{one-hot}$ 表示。取 $\text{mean}(0)$ 就是计算 **$B \times T \times top\_k$ 次选择中，每个专家被选中的比例**。这就是 $f_i$ 的全局统计值。 |
| **$L_{aux}$** (辅助损失) | `aux_loss = (Pi * fi).sum() * self.alpha` | $\mathbf{fi}$ 等价于 $f_i \times E$ (将 $f_i$ 归一化，使其均匀分布时 $f_i \approx 1/E$，则 $fi \approx 1$)。最后通过 $\sum (p_i \cdot f_i \cdot E)$ 得到损失。|

* **全局平衡**：

$$L_{aux}^{Global} \propto \sum_{i=1}^{E} \left( \frac{1}{B \cdot T \cdot K} \sum_{j=1}^{B} \sum_{t=1}^{T} \sum_{k=1}^{K} \mathbb{I}(Expert_{j,t,k} = i) \right) \cdot \left( \frac{1}{B \cdot T} \sum_{j=1}^{B} \sum_{t=1}^{T} p_{j,t,i} \right)$$

* $f_i$ ($\frac{1}{B \cdot T \cdot K} \sum \mathbb{I}$) 是 **全局** 专家访问频率。
* $p_i$ ($\frac{1}{B \cdot T} \sum p$) 是 **全局** 专家平均门控概率。

**总结：** 全局平衡是最直接的公式实现，它计算 **Batch-Level** 的 $p_i$ 和 $f_i$，来确保整个 Batch 的专家使用是均匀的。

---

### 2. 序列级别平衡（`self.seq_aux=True`）

**核心思想：** 不在全局计算 $L_{aux}$，而是在**每个样本 (序列) $j$ 内部**计算一个局部损失 $L_{aux}^j$，最终的损失是这些局部损失的平均值 $\frac{1}{B} \sum_{j=1}^{B} L_{aux}^j$。

$$L_{aux}^j = \frac{1}{T \cdot top\_k} \sum_{i=1}^{E} \left( \sum_{t=1}^{T} \mathbb{I}(Expert_{j,t} = i) \right) \cdot \left( \frac{1}{T} \sum_{t=1}^{T} p_{j,t,i} \right)$$
其中 $p_{j,t,i}$ 是序列 $j$ 中 token $t$ 选择专家 $i$ 的概率。

| 公式项 | 代码计算 | 形状 | 解释 |
| :--- | :--- | :--- | :--- |
| **局部 $p_i$** (序列 $j$ 的平均门控概率) | `scores_for_seq_aux.mean(dim=1)` | $[B, E]$ | $\mathbf{scores\_for\_seq\_aux}$ (形状 $[B, T, E]$) 对 $T$ 维度求均值。得到 **每个样本** 的平均期望概率分布 $\left(\frac{1}{T} \sum_{t=1}^{T} p_{j,t,i}\right)$。 |
| **局部 $f_i$** (序列 $j$ 的归一化访问计数) | `ce` (在 `scatter_add_` 和 `div_` 之后) | $[B, E]$ | $\mathbf{ce}$ 统计了 **每个样本** 中每个专家被选中的次数 $\left(\sum_{t=1}^{T} \mathbb{I}(Expert_{j,t} = i)\right)$。然后除以 $T \cdot top\_k / E$，使得 $f_i$ 被归一化到 $\approx 1$ (等价于 $f_i \times E$)。 |
| **局部 $L_{aux}^j$** | `(ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1)` | $[B]$ | 向量 $\mathbf{ce}$ 和 **局部 $p_i$** 相乘，再对专家维度 $\text{sum}(dim=1)$，得到**每个样本的局部损失** $L_{aux}^j$。 |
| **$L_{aux}$** (总辅助损失) | `... .mean() * self.alpha` | $[1]$ | 对所有样本的局部损失 $\mathbf{mean()}$，求得最终的 **Batch 平均序列损失**。 |

序列平衡计算的是 Batch 中序列损失的平均值：

$$L_{aux}^{Seq} \propto \frac{1}{B} \sum_{j=1}^{B} \left[ \sum_{i=1}^{E} \left( \frac{1}{T \cdot K} \sum_{t=1}^{T} \sum_{k=1}^{K} \mathbb{I}(Expert_{j,t,k} = i) \right) \cdot \left( \frac{1}{T} \sum_{t=1}^{T} p_{j,t,i} \right) \right]$$

内层 $\sum_{i=1}^{E} \left( ... \right)$：这是在序列 $j$ 内部计算一个局部 $\sum f_i \cdot p_i$ 损失。

$f_{i}^j$: 序列 $j$ 内部的专家访问频率。

$p_{i}^j$: 序列 $j$ 内部的专家平均门控概率。

外层 $\frac{1}{B} \sum_{j=1}^{B}$：将所有 $B$ 个独立的序列损失求平均，得到最终的 $L_{aux}$。

**总结：** 序列级平衡通过在 $\mathbf{B}$ 维度上保留计算，计算了 $B$ 个独立的 $p_i$ 和 $f_i$ 向量，并对它们求和后再取平均。这迫使**每个单独的输入序列**都必须均匀地使用所有专家，而不是仅仅在整个 Batch 上看起来均匀。

---

## 🚀 结论

### 💡 关键理解差异点

计算 $\frac{1}{B \cdot T} \sum_{j=1}^{B} \sum_{t=1}^{T} L_{token\_level}$，即**对所有 token 求平均损失**。
* **序列平衡**：计算 $\frac{1}{B} \sum_{j=1}^{B} L_{seq\_level}^j$，即**对所有序列 (样本) 求平均损失**。

序列级别平衡的约束更强，因为它要求在更小的粒度（一个序列）上实现均衡，因此能更有效地防止局部专家崩溃。
