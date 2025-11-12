# 9.1 图

“图（graph）”是一种非线性数据结构，由顶点（vertex）与边（edge）组成。
可抽象表示为集合 (G = {V, E})，例如：

$
\begin{aligned}
V &= {1,2,3,4,5} \\
E &= {(1,2), (1,3), (1,5), (2,3), (2,4), (2,5), (4,5)}
\end{aligned}
$

相比于链表（线性结构）或树（分层结构），图具有更高的自由度。
![alt text](./img/linkedlist_tree_graph.png)

---

## 9.1.1 图的常见类型与术语

### 类型

* **无向图（Undirected Graph）**：边没有方向，表示为“顶点之间的双向连接”，如微信好友关系。
* **有向图（Directed Graph）**：边有方向，(A $\to$ B) 与 (B $\to$ A) 是不同的关系，如微博中的“关注”关系。
* **连通图（Connected Graph）**：从某一顶点出发，可以到达图中的任意其他顶点。
* **非连通图（Disconnected Graph）**：存在至少一个顶点，从其出发无法到达某些顶点。
* **有权图（Weighted Graph）**：每条边附带一个权重（cost/weight），如社交系统中玩家间“亲密度”关系。

### 术语

* **邻接（Adjacency）**：两个顶点之间若存在边相连，则称它们互为邻接。
* **路径（Path）**：从顶点 A 到顶点 B 所经过的边构成的顶点序列。
* **度（Degree）**：顶点拥有的边数。对于有向图：

  * 入度（in-degree）：指向该顶点的边数
  * 出度（out-degree）：该顶点指向别人的边数

---

## 9.1.2 图的表示

以下以无向图为例介绍两种经典表示方式：

### 邻接矩阵（Adjacency Matrix）

设图中有 (n) 个顶点，则用一个 (n \times n) 矩阵 (M) 表示：

* 行、列分别对应顶点
* (M[i,j] = 1) 表示顶点 (i) 与顶点 (j) 之间有边； 0 表示无边
* 对无向图，矩阵关于主对角线对称
* 可扩展为有权图，将 1/0 换成具体权重

![alt text](./img/adjacency_matrix.png)

优缺点：

* **优**：可以 (O(1)) 时间判断任意两顶点是否有边
* **缺**：空间复杂度为 (O(n^2))，当边远少于 (n^2) 时浪费严重

### 邻接表（Adjacency List）

用 (n) 个链表（或数组/容器）表示：第 i 个表存储与顶点 i 相连的所有邻接顶点。
优缺点：

* **优**：仅存储实际存在的边，节省空间
* **缺**：要判断两顶点是否有边，或遍历所有邻接顶点，时间可能更高

优化思路：当链表过长时，可用平衡树（AVL／红黑树）或哈希表提升查找效率。

![alt text](./img/adjacency_list.png)

---

## 9.1.3 图的常见应用

许多现实系统可抽象为图模型，对应算法问题如下（见表 9-1）：

| 现实系统     | 顶点 | 边       | 图计算问题  |
| -------- | -- | ------- | ------ |
| 社交网络     | 用户 | 好友关系    | 潜在好友推荐 |
| 地铁线路     | 站点 | 站点间连通   | 最短路线推荐 |
| 太阳系／天体系统 | 星体 | 万有引力／轨道 | 行星轨道计算 |

---

## 小结

* 图是顶点与边组成的、比树结构更一般的网络结构。
* 图按是否有方向、是否连通、是否有权，可分多种类型。
* 图可用“邻接矩阵”“邻接表”等方式表示，各有优劣。
* 很多现实问题都能建模为图，再通过图算法求解。


# 图的基本操作

```python 
拆解代码：
class GraphAdjMat:
    """基于邻接矩阵实现的无向图类"""

    def __init__(self, vertices: list[int], edges: list[list[int]]):
        """构造方法"""
        # 顶点列表，元素代表“顶点值”，索引代表“顶点索引”
        self.vertices: list[int] = []
        # 邻接矩阵，行列索引对应“顶点索引”
        self.adj_mat: list[list[int]] = []
        # 添加顶点
        for val in vertices:
            self.add_vertex(val)
        # 添加边
        # 请注意，edges 元素代表顶点索引，即对应 vertices 元素索引
        for e in edges:
            self.add_edge(e[0], e[1])

    def size(self) -> int:
        """获取顶点数量"""
        return len(self.vertices)

    def add_vertex(self, val: int):
        """添加顶点"""
        n = self.size()
        # 向顶点列表中添加新顶点的值
        self.vertices.append(val)
        # 在邻接矩阵中添加一行
        new_row = [0] * n
        self.adj_mat.append(new_row)
        # 在邻接矩阵中添加一列
        for row in self.adj_mat:
            row.append(0)

    def remove_vertex(self, index: int):
        """删除顶点"""
        if index >= self.size():
            raise IndexError()
        # 在顶点列表中移除索引 index 的顶点
        self.vertices.pop(index)
        # 在邻接矩阵中删除索引 index 的行
        self.adj_mat.pop(index)
        # 在邻接矩阵中删除索引 index 的列
        for row in self.adj_mat:
            row.pop(index)

    def add_edge(self, i: int, j: int):
        """添加边"""
        # 参数 i, j 对应 vertices 元素索引
        # 索引越界与相等处理
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        # 在无向图中，邻接矩阵关于主对角线对称，即满足 (i, j) == (j, i)
        self.adj_mat[i][j] = 1
        self.adj_mat[j][i] = 1

    def remove_edge(self, i: int, j: int):
        """删除边"""
        # 参数 i, j 对应 vertices 元素索引
        # 索引越界与相等处理
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        self.adj_mat[i][j] = 0
        self.adj_mat[j][i] = 0

    def print(self):
        """打印邻接矩阵"""
        print("顶点列表 =", self.vertices)
        print("邻接矩阵 =")
        print_matrix(self.adj_mat)
```

## 一、图的基本概念

在图论中：

* **顶点（vertex）**表示图中的点；
* **边（edge）**表示两个顶点之间的连接关系；
* **无向图**的边是双向的，比如 A-B 表示 A 与 B 互相连接；
* **邻接矩阵（adjacency matrix）**是一个 $n \times n$ 的二维矩阵，用于表示顶点之间的连接关系：

  * 若顶点 `i` 与顶点 `j` 相连，则 `adj[i][j] = 1`；
  * 否则为 `0`。

---

## 二、构建图的入口：`__init__`

```python
def __init__(self, vertices: list[int], edges: list[list[int]]):
```

构造函数输入两部分数据：

1. `vertices`：顶点值列表（如 `[1, 2, 3]`）
2. `edges`：边的顶点索引列表（如 `[[0,1], [1,2]]`，表示第0号点与第1号点相连，第1号点与第2号点相连）

整个初始化过程做了两件事：

```python
for val in vertices:
    self.add_vertex(val)   # 逐个添加顶点（构造邻接矩阵的空框架）
for e in edges:
    self.add_edge(e[0], e[1])   # 逐个添加边（填充邻接矩阵）
```

---

## 三、添加顶点：`add_vertex()`

```python
def add_vertex(self, val: int):
    n = self.size()            # 当前已有的顶点数量
    self.vertices.append(val)  # 加入新的顶点值
    new_row = [0] * n          # 为新的顶点创建一行，长度为旧顶点数量（初始全为0）
    self.adj_mat.append(new_row)
    for row in self.adj_mat:   # 给每一行都添加一列（因为矩阵要保持方形）
        row.append(0)
```

举例：
假设初始空图 → 添加顶点依次为 1, 2, 3：

| 操作   | 顶点列表      | 邻接矩阵                        |
| ---- | --------- | --------------------------- |
| 添加 1 | `[1]`     | `[[0]]`                     |
| 添加 2 | `[1,2]`   | `[[0,0],[0,0]]`             |
| 添加 3 | `[1,2,3]` | `[[0,0,0],[0,0,0],[0,0,0]]` |

---

## 四、添加边：`add_edge(i, j)`

```python
def add_edge(self, i: int, j: int):
    if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
        raise IndexError()
    self.adj_mat[i][j] = 1
    self.adj_mat[j][i] = 1   # 无向图对称
```

举例：

```python
vertices = [1, 2, 3]
edges = [[0,1], [1,2]]
```

* 初始化顶点后矩阵为：

  ```
  [ [0,0,0],
    [0,0,0],
    [0,0,0] ]
  ```
* 添加边 `[0,1]` → 顶点1与顶点2相连：

  ```
  [ [0,1,0],
    [1,0,0],
    [0,0,0] ]
  ```
* 添加边 `[1,2]` → 顶点2与顶点3相连：

  ```
  [ [0,1,0],
    [1,0,1],
    [0,1,0] ]
  ```

---

## 五、最终图结构示例

若运行：

```python
g = GraphAdjMat([1, 2, 3], [[0,1],[1,2]])
g.print()
```

输出：

```
顶点列表 = [1, 2, 3]
邻接矩阵 =
[ [0, 1, 0],
  [1, 0, 1],
  [0, 1, 0] ]
```

图的结构如下：

```
1 —— 2 —— 3
```
## 链表
```python
class GraphAdjList:
    """基于邻接表实现的无向图类"""

    def __init__(self, edges: list[list[Vertex]]):
        """构造方法"""
        # 邻接表，key：顶点，value：该顶点的所有邻接顶点
        self.adj_list = dict[Vertex, list[Vertex]]()
        # 添加所有顶点和边
        for edge in edges:
            self.add_vertex(edge[0])
            self.add_vertex(edge[1])
            self.add_edge(edge[0], edge[1])

    def size(self) -> int:
        """获取顶点数量"""
        return len(self.adj_list)

    def add_edge(self, vet1: Vertex, vet2: Vertex):
        """添加边"""
        if vet1 not in self.adj_list or vet2 not in self.adj_list or vet1 == vet2:
            raise ValueError()
        # 添加边 vet1 - vet2
        self.adj_list[vet1].append(vet2)
        self.adj_list[vet2].append(vet1)

    def remove_edge(self, vet1: Vertex, vet2: Vertex):
        """删除边"""
        if vet1 not in self.adj_list or vet2 not in self.adj_list or vet1 == vet2:
            raise ValueError()
        # 删除边 vet1 - vet2
        self.adj_list[vet1].remove(vet2)
        self.adj_list[vet2].remove(vet1)

    def add_vertex(self, vet: Vertex):
        """添加顶点"""
        if vet in self.adj_list:
            return
        # 在邻接表中添加一个新链表
        self.adj_list[vet] = []

    def remove_vertex(self, vet: Vertex):
        """删除顶点"""
        if vet not in self.adj_list:
            raise ValueError()
        # 在邻接表中删除顶点 vet 对应的链表
        self.adj_list.pop(vet)
        # 遍历其他顶点的链表，删除所有包含 vet 的边
        for vertex in self.adj_list:
            if vet in self.adj_list[vertex]:
                self.adj_list[vertex].remove(vet)

    def print(self):
        """打印邻接表"""
        print("邻接表 =")
        for vertex in self.adj_list:
            tmp = [v.val for v in self.adj_list[vertex]]
            print(f"{vertex.val}: {tmp},")
整个链表构造的方式呢？
```

---

## 一、基本概念：邻接表 vs 邻接矩阵

| 对比项    | 邻接矩阵 (`GraphAdjMat`) | 邻接表 (`GraphAdjList`) |
| ------ | -------------------- | -------------------- |
| 存储方式   | 二维矩阵                 | 哈希表（字典）              |
| 空间复杂度  | O(n²)                | O(n + m)，更节省空间       |
| 适合的图   | 稠密图                  | 稀疏图                  |
| 访问连接关系 | 常数时间                 | 遍历链表时间               |

邻接表更接近链式结构：
每个顶点都维护一个“链表”（或列表），记录它所连接的其他顶点。

---

## 二、核心数据结构

```python
self.adj_list = dict[Vertex, list[Vertex]]()
```

这表示：

* `key` 是一个顶点对象（`Vertex`）；
* `value` 是一个列表（可以理解为“链表”），保存与该顶点相邻的所有顶点。

例如，图结构：

```
A —— B —— C
```

其邻接表可能是：

```python
{
    A: [B],
    B: [A, C],
    C: [B]
}
```

---

## 三、构造函数 `__init__`

```python
def __init__(self, edges: list[list[Vertex]]):
    self.adj_list = dict[Vertex, list[Vertex]]()
    for edge in edges:
        self.add_vertex(edge[0])
        self.add_vertex(edge[1])
        self.add_edge(edge[0], edge[1])
```

也就是说，输入一组边，图自动构建出：

1. 所有顶点；
2. 每个顶点的相邻顶点列表。

---

### 举个例子

假设你创建：

```python
a, b, c = Vertex('A'), Vertex('B'), Vertex('C')
edges = [[a, b], [b, c]]
g = GraphAdjList(edges)
```

构造过程分三步：

1. **初始化空邻接表**

```python
adj_list = {}
```

2. **第一条边 `[a, b]`**

* 添加顶点 `a`、`b` →
  `adj_list = { a: [], b: [] }`
* 添加边 →
  `a` 的邻居列表中加入 `b`，`b` 的邻居列表中加入 `a`

此时：

```
a: [b]
b: [a]
```

3. **第二条边 `[b, c]`**

* 添加顶点 `c` →
  `adj_list = { a: [b], b: [a], c: [] }`
* 添加边 →
  `b` 的邻居加入 `c`，`c` 的邻居加入 `b`

最终：

```
a: [b]
b: [a, c]
c: [b]
```

---

## 四、`add_edge` 和 “链表连接”的本质

```python
def add_edge(self, vet1: Vertex, vet2: Vertex):
    self.adj_list[vet1].append(vet2)
    self.adj_list[vet2].append(vet1)
```

这是邻接表构造的核心：

* 它**并没有真的创建 Python 的链表结构（如 `ListNode`）**；
* 而是用列表 `list[Vertex]` 来模拟“链式连接关系”；
* 逻辑上，每个顶点对应一个“邻居链表”。

因此：

* `self.adj_list[vet1]` 就像“顶点 vet1 的链表头”；
* 其中的每个元素就是指向的“下一个邻接点”。

---

## 五、删除操作的联动

删除顶点时：

```python
def remove_vertex(self, vet: Vertex):
    self.adj_list.pop(vet)
    for vertex in self.adj_list:
        if vet in self.adj_list[vertex]:
            self.adj_list[vertex].remove(vet)
```

这相当于：

1. 删除该顶点的链表；
2. 从所有其他顶点的链表中，去掉它的引用（断开所有边）。

这也体现出邻接表的**灵活性**：删除局部连接时，无需重建整个矩阵。

---

