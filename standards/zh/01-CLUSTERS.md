# 01-团簇

方格子团簇枚举、最近邻图分类与多格点算符支持模式。

## 1) 团簇定义 (MUST)

MUST:
- 团簇是二维方格子 $\mathbb{Z}^2$ 上的连通格点集。
- 连通性只由最近邻相邻关系确定
  （Manhattan 距离 $= 1$）。

Code form:
```python
cluster = [(x0,y0), (x1,y1), ..., (x_{N-1}, y_{N-1})]
```

## 2) 规范形式与枚举 (MUST)

MUST:
- 规范形式通过对所有 8 个 C4v 点群操作
  （`id, rot90, rot180, rot270, mirror_x, mirror_y, mirror_diag, mirror_anti`）
  变换团簇，将每个结果平移使 `min(x) = min(y) = 0`，对格点列表排序，
  取字典序最小者。
- 通过从种子格点出发的 DFS 生长枚举团簇，利用规范形式去重。

Code form:
```python
canonical_form(cluster, is_canon=True)  # C4v + 平移
generate_clusters(N)                    # DFS 枚举
```

Validation:
- 两个团簇代表相同物理几何，当且仅当它们具有相同的规范形式。

## 3) 空穴数 (MUST)

MUST:
- "空穴"指团簇中完全包含的 $2 \times 2$ 正方形格块。
- 团簇先按空穴数排序，然后再进行图分类。

Code form:
```python
count_holes(cluster)  # 计数所有 {(x,y),(x+1,y),(x,y+1),(x+1,y+1)} ⊂ cluster 的 (x,y)
```

## 4) 邻接矩阵与加权图 (MUST)

MUST:
- 邻接矩阵 $A_{ij}$ 只编码最近邻连通性：
  - `1` = NN 键
  - `0` = 无键
- 图 $G(C)$ 由此矩阵构建，边具有 `weight=1` 属性。

Code form:
```python
adj = generate_adjacency_matrix(sites)
G   = graph_from_adj(adj)  # nx.Graph，带 weight 属性
```

## 5) 同构分类 (MUST)

MUST:
- 两个团簇等价，当且仅当它们的图同构
  （通过 `nx.GraphMatcher` 加 `numerical_edge_match('weight', 0)` 检查）。
- 分类过程：先按空穴数分组，再在每组内按图同构分类。
- 每个等价类中，第一个团簇为**代表**。
  其余团簇通过同构映射重排序，使其邻接矩阵与代表完全一致。

Code form:
```python
classify_isomorphic_clusters(clusters)
# 返回：分组列表，每组 = 格点重排序后的团簇列表
```

Validation:
- 同一类中所有团簇在重排序后必须具有相同的邻接矩阵。
- 代表求解一次哈密顿量；其他成员复用本征系统。

## 6) 键生成 (MUST)

MUST:
- `Cluster.bonds` 存储所有唯一最近邻格点对 `(i, j)`，
  Manhattan 距离为 `1` 且 `i < j`。这是 Hubbard 跳跃矩阵使用的键列表。
- `Cluster.generate_bonds(N=2, is_connected=False)` 返回拟合算符组：
  每个两格点算符对应一个单元素组 `[[i, j]]`。
- `Cluster.generate_bonds(N=4 or 6, is_connected=True)` 返回连通多格点
  自旋算符配对的单元素组。

Code form:
```python
nn_bonds = cluster.bonds
bond_groups = (
    cluster.generate_bonds(N=2, is_connected=False)
    + cluster.generate_bonds(N=4, is_connected=True)
    + cluster.generate_bonds(N=6, is_connected=True)
)
```

## 7) 两格点键分类 (MUST)

MUST:
- `find_bonds_twosites(cluster)` 按规范键向量
  $(\max(|dx|, |dy|), \min(|dx|, |dy|))$ 对所有格点对 `(i, j)` 分组，
  按欧氏距离排序。
- 这为自旋耦合拟合产生键类别（如所有 NN 键、所有 NNN 键等）。

Code form:
```python
bond_types = find_bonds_twosites(cluster)
# 返回：dict_values，每个列表包含一种规范键向量对应的 (i,j) 对
```

## 8) 多格点算符模式 (MUST)

MUST:
- 四格点模式：团簇所有连通的四格点子集。
- 六格点模式：所有连通的六格点子集。
- 正方形格块模式：所有 $2 \times 2$ 正方形，用于环交换算符。
- 配对数、规范排序和输出约定定义在 `08-OPERATOR_OUTPUT.md` 中。

Code form:
```python
find_bonds_foursites(cluster)   # 四格点连通子集 × 3 种配对（规范排序）
find_bonds_sixsites(cluster)    # 六格点连通子集 × 15 种配对（规范排序）
find_squares(cluster)           # 完整的 2×2 格块，以索引元组表示
```

## 9) 可扩展性 (MUST)

MUST:
- 已提交的基线格子族为 `square`（方格子）。
- 扩展到其他格子族（三角、蜂窝等）需要：
  1. 新的 `get_neighbors` 函数定义连通性。
  2. 新的 `canonical_form` 尊重格子对称群。
  3. 邻接矩阵中新的键类型定义。
- 不得为适应扩展而修改现有方格子标准。
