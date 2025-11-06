# 块对角旋转矩阵与工程实现的等价性

我们要证明：**块对角的“每对二维旋转”矩阵**

$\begin{align}
R = \operatorname{diag}\big(R_0,\, R_1,\, \dots,\, R_{M-1}\big), \qquad 
R_m = \begin{pmatrix} \cos_m & -\sin_m \\ \sin_m & \cos_m \end{pmatrix}
\end{align}$

**通过恰当的重排（置换）可以写成一个 2×2 块矩阵**

$\begin{align}
\begin{pmatrix}
\mathrm{C} & -\mathrm{S} \\
\mathrm{S} & \mathrm{C}
\end{pmatrix},
\end{align}$

其中 $\mathrm{C} = \operatorname{diag}(\cos_0, \cos_1, \dots, \cos_{M-1})$ 与 $\mathrm{S} = \operatorname{diag}(\sin_0, \sin_1, \dots, \sin_{M-1})$。我们将分步证明并建立与代码中 `rotate_half` 的对应关系。

---

## 1. 设定与重排说明

记每个 head 的维数 D = 2M。把向量 $x \in \mathbb{R}^{2M}$ 以 **按 pair 展开（interleaved）** 的方式写作：


$\begin{align}
x = 
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_{2M-2} \\ x_{2M-1}
\end{bmatrix} = \begin{bmatrix}
x_{2\cdot0} \\ x_{2\cdot0+1} \\ x_{2\cdot1} \\ x_{2\cdot1+1} \\ \vdots
\end{bmatrix}
\end{align}$


每对 $(x_{2m}, x_{2m+1})$ 受 $R_m$ 旋转。

---

现在定义把 `pairs-interleaved` 形式重排成 **两段串联形式**（先把所有偶数索引放一起，再把所有奇数索引放一起）：
$
P : \mathbb{R}^{2M} \to \mathbb{R}^{2M},
$

$\begin{align}
P x = 
\begin{bmatrix}
x_0 \\ x_2 \\ \vdots \\ x_{2M-2} \\ x_1 \\ x_3 \\ \vdots \\ x_{2M-1}\\
\end{bmatrix} =\begin{bmatrix}u\\ v\end{bmatrix}
\end{align}$


其中 $u = (x_0, x_2, \dots, x_{2M-2})^\top \in \mathbb{R}^{M}$、$v = (x_1, x_3, \dots, x_{2M-1})^\top \in \mathbb{R}^{M}$。

矩阵 $P$ 是一个置换矩阵（正交，$P^{-1} = P^\top$）。

---

## 2. 在重排后的坐标下的旋转矩阵形式

把块对角矩阵 $R = \operatorname{diag}(R_0, \dots, R_{M-1}$) 作用在 $x$ 上：

$\begin{align}
x' = R x.
\end{align}$

两边左乘 $P$，得到重排后的向量：

$\begin{align}
P x' = P R x.
\end{align}$

注意 $P x' = (P R P^\top) (P x)$，因为 $P P^\top = I$。设
$
\begin{align}
\widetilde{R} := P R P^\top, \qquad
\widetilde{x} := P x = \begin{bmatrix} u \\ v \end{bmatrix}.
\end{align}
$
那么

$\begin{align}
\widetilde{x}' = \widetilde{R} \, \widetilde{x}.
\end{align}$


我们要证明 $\widetilde{R}$ 恰等于 2×2 块矩阵
$\begin{align}
\widetilde{R} =
\begin{pmatrix}
\mathrm{C} & -\mathrm{S} \\
\mathrm{S} & \mathrm{C}
\end{pmatrix},
\end{align}$

其中 $\mathrm{C} = \operatorname{diag}(\cos_0, \dots, \cos_{M-1})$、$\mathrm{S} = \operatorname{diag}(\sin_0, \dots, \sin_{M-1})$。

---

## 3. 验证每个 block 的项（逐对展开）

考虑第 m 对（对应原来 $R_m$）对 x 的作用：

$\begin{align}
\begin{bmatrix} x'_{2m} \\ x'_{2m+1} \end{bmatrix}= \begin{bmatrix} \cos_m & -\sin_m \\ \sin_m & \cos_m \end{bmatrix} \begin{bmatrix} x_{2m} \\ x_{2m+1} \end{bmatrix}.
\end{align}$

在重排后，偶数索引元素 $x_{2m}$ 会出现在向量 u 的第 m 行；奇数索引元素 $x_{2m+1}$ 会出现在 v 的第 m 行。因此第 m 对的变换写成两个分量：

$\begin{align}
\text{(偶数部分)} &\quad x'_{2m} = \cos_m\, x_{2m} - \sin_m\, x_{2m+1}, \\
\text{(奇数部分)} &\quad x'_{2m+1} = \sin_m\, x_{2m} + \cos_m\, x_{2m+1}.
\end{align}$


把所有 $m = 0, \dots, M-1$ 串联，可写成向量形式：

$\begin{align}
\begin{pmatrix} u' \\ v' \end{pmatrix} =
\begin{pmatrix}
\mathrm{C} & -\mathrm{S} \\
\mathrm{S} & \mathrm{C}
\end{pmatrix}
\begin{pmatrix} u \\ v \end{pmatrix},
\end{align}$

其中第 m 行对应上面的两个式子（$\mathrm{C}_{mm} = \cos_m,\ \mathrm{S}_{mm} = \sin_m$）。这正是我们希望得到的 2×2 块矩阵形式。

换言之，块对角的 \(R\) 在按 pair 重排（即 \(P\)）后的表示就是

$\begin{align}
\widetilde{R} = P R P^\top = \begin{pmatrix}
\operatorname{diag}(\cos_m) & -\operatorname{diag}(\sin_m) \\
\operatorname{diag}(\sin_m) & \phantom{-}\operatorname{diag}(\cos_m)
\end{pmatrix}.
\end{align}$

---

## 4. 由此得到代码形式与 `rotate_half` 的对应关系

回顾代码实现（简化写法）：

$\begin{align}
q_{\text{embed}} = (q \odot \cos) + (\operatorname{rotate\_half}(q) \odot \sin). 
\end{align}$


令按前半/后半的重排 $P q = \begin{bmatrix} u \\ v \end{bmatrix}$。一项一项展开：

$\begin{align}
\begin{pmatrix} u' \\ v' \end{pmatrix} = \begin{pmatrix}
\mathrm{C} & -\mathrm{S} \\
\mathrm{S} & \mathrm{C}
\end{pmatrix}， 
\begin{pmatrix} u \\ v \end{pmatrix} = 
\begin{pmatrix}
u \odot \cos - v \odot \sin \\
u \odot \sin + v \odot \cos
\end{pmatrix}.
\end{align}$

把结果再用 $P^\top$（即把两段交织回原来的 interleaved 顺序）重排回去，会得到形式

$$q_{\text{embed}} = q \odot \cos + \operatorname{rotate\_half}(q) \odot \sin,$$

其中 `rotate_half` 的具体实现（把后半放到前、前半放到后，并对前半一段取负）正是实现上面从 \($\begin{pmatrix} u \\ v \end{pmatrix}$\) 回到 interleaved 顺序并匹配旋转矩阵展开各项所需要的操作。

因此三者互相等价：

- 原始的块对角旋转 $R = \operatorname{diag}(R_0, \dots, R_{M-1})$
- 重排后矩阵 $\begin{pmatrix} \mathrm{C} & -\mathrm{S} \\ \mathrm{S} & \mathrm{C} \end{pmatrix}$
- 以及向量化实现 $(q \odot \cos) + (\operatorname{rotate\_half}(q) \odot \sin)$

---

## 5. 例子\(D=6\)，\(M=3\)）

令 $x = [x_0, x_1, x_2, x_3]^\top$，有
$$
R = \operatorname{diag}\left(
\begin{bmatrix} \cos_0 & -\sin_0 \\ \sin_0 & \cos_0 \end{bmatrix},
\begin{bmatrix} \cos_1 & -\sin_1 \\ \sin_1 & \cos_1 \end{bmatrix},
\begin{bmatrix} \cos_2 & -\sin_2 \\ \sin_2 & \cos_2 \end{bmatrix}
\right).$$

写出 R（按 interleaved 索引）：

$$R =
\begin{pmatrix}
\cos_0 & -\sin_0 & 0 & 0 & 0 & 0 \\
\sin_0 & \cos_0 & 0 & 0 & 0 & 0\\
0 & 0 & \cos_1 & -\sin_1 & 0 & 0\\
0 & 0 & \sin_1 & \cos_1 & 0 & 0\\
0 & 0 & 0 & 0 & \cos_2 & -\sin_2 \\
0 & 0 & 0 & 0 & \sin_2 & \cos_2 \\
\end{pmatrix}.$$


取置换矩阵

$$
P = 
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0\\
0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1\\
\end{pmatrix}, Px = \begin{pmatrix} x_0 \\ x_2 \\ x_1 \\ x_3 \\ x_4 \\ x_5\end{pmatrix} = \begin{pmatrix} u \\ v \end{pmatrix}.
$$

计算 $P R P^\top$ 会得到

$$
\begin{pmatrix}
\cos_0 & 0 & 0 & -\sin_0 & 0 & 0  \\
0 & \cos_1 & 0 & 0 & -\sin_1 & 0 \\
0 & 0 & \cos_2  & 0 &0 &-\sin_2  & \\
\sin_0 & 0 & 0 & \cos_0 & 0 & 0  \\
0 & \sin_1 & 0 & 0 & \cos_1 & 0 \\
0 & 0 & \sin_2  & 0 &0 &\cos_2  & \\
\end{pmatrix} =
\begin{pmatrix}
\mathrm{C} & -\mathrm{S} \\
\mathrm{S} & \mathrm{C}
\end{pmatrix},
$$
正如前面所述。

