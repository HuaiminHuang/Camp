# **二叉搜索树（BST, Binary Search Tree）**

## 🌳 一、什么是二叉搜索树（Binary Search Tree, BST）

**定义：**

> 对于任意一个二叉树节点 `node`：
>
> * 它左子树中所有节点的值都 **小于** `node.val`
> * 它右子树中所有节点的值都 **大于** `node.val`
> * 左右子树也同样满足这个性质（递归定义）

**示意图：**

```
        8
       / \
      4   10
     / \    \
    2   6    20
```

对应关系：

* 左子树都比 8 小（2,4,6）
* 右子树都比 8 大（10,20）

---

## 🔍 二、搜索流程（`search()`）

```python
def search(self, num: int) -> TreeNode | None:
    cur = self._root
    while cur is not None:
        if cur.val < num:
            cur = cur.right     # 要找的值比当前大 → 去右子树
        elif cur.val > num:
            cur = cur.left      # 要找的值比当前小 → 去左子树
        else:
            break               # 找到了
    return cur
```

### ✅ 原理理解

搜索过程就像在有序数组里二分查找一样：

* 每次比较 `num` 和 `cur.val`
* 根据大小关系决定去左还是右
* 因为 BST 左小右大，所以能在 **O(h)** 时间内定位目标（`h`为树高）

### 🧠 例子

查找 num = 6

```
        8
       / \
      4   10
     / \
    2   6
```

比较过程：

```
8 > 6 → 去左子树
4 < 6 → 去右子树
找到 6 → 返回节点
```

---

## 🌱 三、插入流程（`insert()`）

```python
def insert(self, num: int):
    if self._root is None:
        self._root = TreeNode(num)
        return

    cur, pre = self._root, None
    while cur is not None:
        if cur.val == num:     # 不允许重复
            return
        pre = cur
        if cur.val < num:
            cur = cur.right    # 去右子树
        else:
            cur = cur.left     # 去左子树

    node = TreeNode(num)
    if pre.val < num:
        pre.right = node
    else:
        pre.left = node
```

### ✅ 插入逻辑

1. 从根节点开始查找合适的插入位置；
2. 按照“左小右大”规则一路向下；
3. 当 `cur` 为 `None` 时，说明到达叶节点外层；
4. 根据最后的父节点 `pre`，判断新节点插到左还是右；
5. 若已有相同值，则不插入（去重策略）。

### 🧠 例子

插入 num = 5 到：

```
        8
       / \
      4   10
     / \
    2   6
```

过程：

```
8 > 5 → 去左
4 < 5 → 去右
6 > 5 → 去左（cur = None，pre = 6）
→ 把 5 插到 6 的左子树
```

结果：

```
        8
       / \
      4   10
     / \
    2   6
       /
      5
```

### 插入和查找时间复杂度

* 平均情况（树较平衡）：`O(log n)`
* 最坏情况（退化为链表）：`O(n)`

---

## 🧭 对比表

| 操作 | 思路          | 时间复杂度 | 关键步骤            |
| -- | ----------- | ----- | --------------- |
| 搜索 | 比大小 → 左/右移动 | O(h)  | 一路比较直到 None 或匹配 |
| 插入 | 找空位 → 接上新节点 | O(h)  | 维护父指针 pre       |
| 删除 | 复杂（3种情况）    | O(h)  | 调整结构，保持 BST 性质  |

---
非常好，这个问题切入了 **二叉搜索树（BST）删除操作的核心难点**。
相比插入和查找，删除节点是 BST 中最复杂的部分，因为需要**在不破坏“左小右大”性质的前提下调整结构**。
我们一步一步解析这段 `remove()` 的逻辑。

---

🌳BST 的关键性质

> 对于任意节点 `cur`：
>
> * 左子树中所有节点值 < `cur.val`
> * 右子树中所有节点值 > `cur.val`

删除节点时，必须保持这个性质仍然成立。

---

## 🧩 四、删除节点的三种情况

删除操作要区分节点的“子节点数量”：

| 情况          | 说明    | 示例图       | 操作                            |
| ----------- | ----- | --------- | ----------------------------- |
| 1️⃣ 无子节点    | 叶节点   | `A`       | 直接删除                          |
| 2️⃣ 只有一个子节点 | 只有左或右 | `A→B`     | 用子节点替换自己                      |
| 3️⃣ 有两个子节点  | 左右都有  | `A` 有左右孩子 | 找**中序后继节点**（右子树最小值）替换，再删除后继节点 |

你的代码正好对应这三种情况。

---

## 🧠 五、代码逻辑拆解

### 🧩 Step 1. 查找目标节点

```python
cur, pre = self._root, None
while cur is not None:
    if cur.val == num:
        break
    pre = cur
    if cur.val < num:
        cur = cur.right
    else:
        cur = cur.left
if cur is None:
    return
```

* `cur`：当前节点（正在搜索的节点）
* `pre`：`cur` 的父节点
* 这个过程与 `search()` 一样，找到要删除的节点或确定不存在。

---

### 🧩 Step 2. 删除节点（分情况讨论）

#### ✅ 情况 1/2：`cur` 没有子节点 或 只有一个子节点

```python
if cur.left is None or cur.right is None:
    child = cur.left or cur.right
    if cur != self._root:
        if pre.left == cur:
            pre.left = child
        else:
            pre.right = child
    else:
        self._root = child
```

#### ✨ 解释：

* 如果 `cur.left` 或 `cur.right` 其中之一是 `None`：

  * `child = cur.left or cur.right` 会得到非空的那个（或者 `None`）
* 然后：

  * 如果 `cur` 不是根节点：让父节点 `pre` 指向 `child`
  * 如果 `cur` 是根节点：直接修改 `_root = child`

📘 例子：

```
    8
   / \
  4   10
     /
    9
```

删除 `10`：

* `cur=10`，`pre=8`
* `cur.left=9`，`cur.right=None`
* `child=9`
* 结果：`pre.right = 9`

结果树：

```
    8
   / \
  4   9
```

---

#### ✅ 情况 3：`cur` 有两个子节点

```python
else:
    tmp: TreeNode = cur.right
    while tmp.left is not None:
        tmp = tmp.left
    self.remove(tmp.val)
    cur.val = tmp.val
```

#### ✨ 解释：

* 当左右子节点都存在时，我们不能直接删 `cur`，否则会破坏 BST 性质。
* 我们要找到一个“可以替代它的值”：

  * **右子树中最小的节点**（中序后继）
  * 或者**左子树中最大的节点**（中序前驱）
* 这段代码选了前者（中序后继）：

---

##### 📘 举例

树结构：

```
        8
       / \
      4   10
         /  \
        9   12
```

删除节点 `10`。

过程：

1. 找右子树中最小节点：`tmp = 9`
2. 删除 `tmp`（递归调用 `self.remove(9)`）
3. 把 `cur.val` 改成 `9`

最终树：

```
        8
       / \
      4   9
           \
            12
```

BST 性质依然成立 ✅

---

## 🧩 六、为什么用“中序后继”替代？

中序遍历的顺序是 “左 → 根 → 右”。
若我们删除一个节点 `cur`，

* 它的**中序前驱**是左子树中最大值；
* 它的**中序后继**是右子树中最小值。

这两个节点都是“离它最近”的合法替代者。
选择其中任意一个，都能保持整个 BST 的有序性。

---

## ⚙️ 七、算法复杂度分析

| 操作     | 时间复杂度    | 原因                |
| ------ | -------- | ----------------- |
| 查找目标节点 | O(h)     | 走树高次              |
| 找中序后继  | O(h)     | 向下走一条路径           |
| 删除后继节点 | O(h)     | 递归一次删除            |
| **总体** | **O(h)** | h 是树高（平衡时约 log n） |

---
## TIPS

## 🌳 一、删除节点的本质问题

当一个节点有 **两个子节点** 时：

* 我们不能直接删除它，否则这棵树会断开，破坏 BST 的有序结构。
* 所以要**找到一个合适的“替身”值**来放在它的位置上。

这个“替身”必须满足：

> 替换后仍然保持「左子树 < 当前节点 < 右子树」。

于是，有两种安全的选择：

1. **右子树中最小的节点**（中序后继）
2. **左子树中最大的节点**（中序前驱）

---

## 🧭 二、什么是中序前驱与中序后继

中序遍历顺序：

```
左子树 → 当前节点 → 右子树
```

那么：

* **当前节点的中序前驱**：中序遍历里，当前节点之前的那个节点
* **当前节点的中序后继**：中序遍历里，当前节点之后的那个节点

在 BST 中有一个非常重要的性质：

* **中序前驱 = 左子树的最大值**
* **中序后继 = 右子树的最小值**

---

## 🧩 三、举例说明

我们构造一个简单的 BST：

```
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13
```

---

### ✅ 情况 1：删除节点 `8`（根节点）

* 左子树：`[1,3,4,6,7]`
* 右子树：`[10,13,14]`

#### 方法 A：找“中序后继”

右子树中最小的节点 → `10` 子树里最左的节点 → **10**

```
右子树中最小的节点是 10
```

替换：把 8 换成 10，然后递归删除节点 10（它在右子树中，容易删）

新树：

```
        10
       /  \
      3    14
     / \   /
    1  6  13
      / \
     4  7
```

---

#### 方法 B：找“中序前驱”

左子树中最大的节点 → `6` 子树里最右的节点 → **7**

```
左子树中最大的节点是 7
```

替换：把 8 换成 7，然后递归删除节点 7。

新树：

```
        7
       / \
      3   10
     / \    \
    1  6    14
       /    /
      4    13
```

两种方法都合法，只是替代方向不同。

---

### ✅ 情况 2：删除节点 `3`

```
        8
       / \
      3   10
     / \    
    1   6    
       / \   
      4   7  
```

左子树最大值：6 的最右节点 → **7**
右子树最小值：6 的最左节点 → **4**

可以选择：

* 用 `4`（右子树最小）替代；
* 或用 `1`（左子树最大）替代。


使用 "终继后续" (右子树最小)：
```python
    def remove_with_successor(self, num: int):
        """删除节点（使用中序后继）"""
        self._root = self._remove(self._root, num)

    def _remove(self, node: TreeNode, num: int) -> TreeNode | None:
        if not node:
            return None

        # 1️⃣ 寻找目标节点
        if num < node.val:
            node.left = self._remove(node.left, num)
        elif num > node.val:
            node.right = self._remove(node.right, num)
        else:
            # 2️⃣ 找到目标节点，分三种情况
            # 情况1：无子节点
            if not node.left and not node.right:
                return None
            # 情况2：只有一侧子节点
            elif not node.left:
                return node.right
            elif not node.right:
                return node.left
            # 情况3：左右子节点都存在
            else:
                # 找右子树的最小节点（中序后继）
                successor = self._min_node(node.right)
                # 用后继的值覆盖当前节点
                node.val = successor.val
                # 删除右子树中该后继节点
                node.right = self._remove(node.right, successor.val)

        return node

    def _min_node(self, node: TreeNode) -> TreeNode:
        """右子树最小值"""
        while node.left:
            node = node.left
        return node
```

- "终极前驱" (左子树最大): 
```python
    def remove_with_predecessor(self, num: int):
        """删除节点（使用中序前驱）"""
        self._root = self._remove_predecessor(self._root, num)

    def _remove_predecessor(self, node: TreeNode, num: int) -> TreeNode | None:
        if not node:
            return None

        # 1️⃣ 寻找目标节点
        if num < node.val:
            node.left = self._remove_predecessor(node.left, num)
        elif num > node.val:
            node.right = self._remove_predecessor(node.right, num)
        else:
            # 2️⃣ 删除逻辑
            if not node.left and not node.right:
                return None
            elif not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # 找左子树最大值（中序前驱）
                predecessor = self._max_node(node.left)
                # 用前驱的值覆盖当前节点
                node.val = predecessor.val
                # 删除左子树中该前驱节点
                node.left = self._remove_predecessor(node.left, predecessor.val)

        return node

    def _max_node(self, node: TreeNode) -> TreeNode:
        """左子树最大值"""
        while node.right:
            node = node.right
        return node

```


