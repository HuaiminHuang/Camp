```python
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    expert_cache = torch.zeros_like(x) # [b*s, h]
    idxs = flat_expert_indices.argsort()
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    token_idxs = idxs // self.config.num_experts_per_tok
    # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
    # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
    # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
    # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        if start_idx == end_idx:
            continue
        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idx]
        expert_out = expert(expert_tokens).to(expert_cache.dtype)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

    return expert_cache
```

设置如下：

* **Token 总数 $N$**: 5 个（即 `bsz * seq_len = 5`）
* **专家总数 $E$**: 3 个（专家 0, 1, 2）
* **Top-K ($K$)**: 2 个（每个 token 选择 2 个专家）

---

### 示例输入数据

由于 $N=5, K=2$，我们共有 $5 \times 2 = 10$ 个 (token, expert) 对。

1.  **展平后的专家索引 (`flat_expert_indices`)**：
    假设这 10 个 (token, expert) 对按原始 token 顺序展平后，选择的专家索引如下：
    $$
    [2, 0, \quad 0, 1, \quad 2, 0, \quad 1, 2, \quad 0, 1]
    $$
    （来自：token 0 选 [2, 0]，token 1 选 [0, 1]，token 2 选 [2, 0]，token 3 选 [1, 2]，token 4 选 [0, 1]）

2.  **排序 (`idxs = flat_expert_indices.argsort()`)**：
    我们对上面的索引进行排序，找出哪个位置的索引是 0，哪个是 1，哪个是 2。
    $$
    \text{idxs} = [1, 3, 5, 9, 2, 6, 7, 4, 8, 0]
    $$
    * `flat_expert_indices[1] = 0` (属于专家 0)
    * `flat_expert_indices[3] = 1` (属于专家 1)
    * ...以此类推...

3.  **统计和累积 (`tokens_per_expert = flat_expert_indices.bincount().cumsum(0)`)**:
    * 专家 0 出现 4 次：位置 $1, 3, 5, 9$
    * 专家 1 出现 3 次：位置 $2, 6, 7$
    * 专家 2 出现 3 次：位置 $4, 8, 0$
    $$
    \text{bincount} = [4, 3, 3] \quad (\text{专家 } 0, 1, 2)
    $$
    $$
    \text{tokens\_per\_expert} = [4, 7, 10] \quad (\text{累积和})
    $$

4.  **映射回原始 Token 索引 (`token_idxs = idxs // K`)**:
    $K=2$。我们将排序后的索引 `idxs` 除以 2 (向下取整)，得到**原始 token 索引**。
    $$
    \text{token\_idxs} = \lfloor [1, 3, 5, 9, 2, 6, 7, 4, 8, 0] / 2 \rfloor
    $$
    $$
    \text{token\_idxs} = [0, 1, 2, 4, 1, 3, 3, 2, 4, 0]
    $$
    * 例如，`idxs[0]=1` 对应第 1 个位置的 (token, expert) 对，它是由原始 token $1/2 = 0$ 产生的。

---

### 循环演示 (`for i, end_idx in enumerate(tokens_per_expert):`)

#### 🚀 第一次循环：专家 0 (`i=0`)

| 变量 | 值 | 解释 |
| :--- | :--- | :--- |
| `i` | 0 | 专家索引 |
| `start_idx` | 0 | 专家的起始索引 |
| `end_idx` | 4 | `tokens_per_expert[0]` |
| `expert` | `self.experts[0]` | 专家 0 |
| `exp_token_idx` | `token_idxs[0:4]` $\rightarrow$ **`[0, 1, 2, 4]`** | 索引 0, 1, 2, 4 这 4 个 token 要送给专家 0 |
| `expert_tokens` | `x[[0, 1, 2, 4]]` | 从原始输入 `x` 中收集这 4 个 token |
| `weights_idxs` | `idxs[0:4]` $\rightarrow$ **`[1, 3, 5, 9]`** | 这 4 个 token 对应的门控权重索引 |
| **操作** | 1. `expert_out = expert(expert_tokens)` | 专家 0 计算输出 |
| | 2. `expert_out *= flat_expert_weights[weights_idxs]` | 乘上对应的 4 个权重 |
| | 3. `expert_cache.scatter_add_(0, [0, 1, 2, 4], expert_out)` | 将结果累加到 `expert_cache` 的第 0, 1, 2, 4 行 |

---

#### 🚀 第二次循环：专家 1 (`i=1`)

| 变量 | 值 | 解释 |
| :--- | :--- | :--- |
| `i` | 1 | 专家索引 |
| `start_idx` | 4 | `tokens_per_expert[0]` |
| `end_idx` | 7 | `tokens_per_expert[1]` |
| `expert` | `self.experts[1]` | 专家 1 |
| `exp_token_idx` | `token_idxs[4:7]` $\rightarrow$ **`[1, 3, 3]`** | 索引 1, 3, 3 这 3 个 token 要送给专家 1 |
| `expert_tokens` | `x[[1, 3, 3]]` | 从原始输入 `x` 中收集这 3 个 token (注意：token 3 被收集了两次) |
| `weights_idxs` | `idxs[4:7]` $\rightarrow$ **`[2, 6, 7]`** | 这 3 个 token 对应的门控权重索引 |
| **操作** | 1. `expert_out = expert(expert_tokens)` | 专家 1 计算输出 |
| | 2. `expert_out *= flat_expert_weights[weights_idxs]` | 乘上对应的 3 个权重 |
| | 3. `expert_cache.scatter_add_(0, [1, 3, 3], expert_out)` | 将结果累加到 `expert_cache` 的第 1 行和第 3 行（第 3 行会被累加两次） |

---

#### 🚀 第三次循环：专家 2 (`i=2`)

| 变量 | 值 | 解释 |
| :--- | :--- | :--- |
| `i` | 2 | 专家索引 |
| `start_idx` | 7 | `tokens_per_expert[1]` |
| `end_idx` | 10 | `tokens_per_expert[2]` |
| `expert` | `self.experts[2]` | 专家 2 |
| `exp_token_idx` | `token_idxs[7:10]` $\rightarrow$ **`[2, 4, 0]`** | 索引 2, 4, 0 这 3 个 token 要送给专家 2 |
| `expert_tokens` | `x[[2, 4, 0]]` | 从原始输入 `x` 中收集这 3 个 token |
| `weights_idxs` | `idxs[7:10]` $\rightarrow$ **`[4, 8, 0]`** | 这 3 个 token 对应的门控权重索引 |
| **操作** | 1. `expert_out = expert(expert_tokens)` | 专家 2 计算输出 |
| | 2. `expert_out *= flat_expert_weights[weights_idxs]` | 乘上对应的 3 个权重 |
| | 3. `expert_cache.scatter_add_(0, [2, 4, 0], expert_out)` | 将结果累加到 `expert_cache` 的第 2, 4, 0 行 |

---

### 最终结果（以 Token 0 为例）

Token 0 (原始索引 0) 被专家 0 和专家 2 处理。

| Token 索引 | 被处理的 (Token, Expert) 对的索引 | 对应的专家 |
| :---: | :---: | :---: |
| 0 | `token_idxs[0] = 0` (来自 `idxs[0]=1`) | 专家 0 |
| 0 | `token_idxs[9] = 0` (来自 `idxs[9]=0`) | 专家 2 |

最终 `expert_cache[0]` 的值将是：
$$
\text{Expert}_0(\text{Token}_0) \cdot w_{0,\text{token}0}^{(0)} + \text{Expert}_2(\text{Token}_0) \cdot w_{2,\text{token}0}^{(0)}
$$
其中 $w$ 是对应的门控权重。

这个循环成功实现了高效的**批量计算**，每个专家只需执行一次前向传播，处理所有分配给它的 token，然后通过 `scatter_add_` 机制将结果聚合起来。
