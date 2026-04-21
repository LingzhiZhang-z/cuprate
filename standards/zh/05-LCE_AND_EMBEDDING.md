# 05-LCE 与嵌入

链接团簇展开（子团簇减法）与超胞嵌入。

## 1) LCE 目标 (MUST)

MUST:
- 对每个团簇 $C$，原始自旋耦合 $O(C)$ 包含所有子团簇的贡献。
- 净（连通、不可约）贡献通过减去所有真连通子团簇获得：

Math:
$$
W(C) = O(C) - \sum_{\substack{C' \subsetneq C \\ C'\text{ connected}}} W(C').
$$

- 自底向上计算：先处理最小团簇，再逐步处理更大的。

## 2) 子图枚举 (MUST)

MUST:
- 对 $N$ 个格点的团簇，枚举大小从 $2$ 到 $N-1$ 的所有连通子图。
- 连通性基于 NN 邻接（曼哈顿距离 $= 1$）。
- 每个子图通过加权图同构匹配到已有的团簇计算结果。

Code form:
```python
subgraphs, indices = get_connected_subgraphs(cluster, min_size=2)
match = find_cluster_match(subgraph, clusters)
# match = (nsites, hole, class_idx, variant_idx, mapping)
```

## 3) 算符减法 (MUST)

MUST:
- 算符列表结构为：
  `[常数项, [二格点项], [四格点项], [六格点项], ...]`
  其中每项为 `[格点索引..., 系数]`。
- 减法通过同构映射将子团簇格点索引映射到父团簇索引，
  然后逐项减去系数。

Code form:
```python
operator_minus(parent_operators, subcluster_operators, index_mapping)
```

- 算符匹配使用对格点对元组排序的规范键函数：

Code form:
```python
k2s(i, j)          = sorted([i, j])
k4s(i, j, k, l)    = sorted([(sorted([i,j]), sorted([k,l]))])
k6s(i,j,k,l,m,n)   = sorted([(sorted([i,j]), sorted([k,l]), sorted([m,n]))])
```

Validation:
- LCE 后，单格点团簇的净贡献为零（无键）。
- 对 $C' \subseteq C$ 的所有 $W(C')$ 求和必须恢复 $O(C)$。

## 4) 超胞嵌入 (MUST)

MUST:
- LCE 净耦合放置到大小为 $N_{\text{cell}} \times N_{\text{cell}}$ 的周期方格子超胞上。
- 对每个团簇，生成所有对称性不等价的取向（通过 C4v 最多 8 个），
  然后每个取向在周期边界条件下平移到所有 $N_{\text{cell}}^2$ 个位置。
- 超胞中的格点索引：`index = x * N_cell + y`，周期边界：`x_pbc = x % N_cell`。

Code form:
```python
unique_clusters = unique_cluster_transformed(cluster)  # 最多 8 个取向
indices_pbc = embed_cluster_to_squarecell_pbc(unique_cluster, N_cell)
embed_operator(couplings_pbc, coupling_net, indices_pbc)
```

## 5) 累积耦合结构 (MUST)

MUST:
- 超胞耦合数据结构为：
  - `couplings[0]`：常数项（标量）。
  - `couplings[1]`：二格点耦合矩阵（$N_{\text{cell}}^2 \times N_{\text{cell}}^2$，复数）。
  - `couplings[2]`：四格点耦合（以 `k4s` 元组为键的字典）。
  - `couplings[3]`：六格点耦合（以 `k6s` 元组为键的字典）。
  - `couplings[4]`：八格点耦合（以 `k8s` 元组为键的字典）。
- 二格点耦合进行对称化：`[i,j]` 和 `[j,i]` 均被累加。

## 6) 输出分析 (MUST)

MUST:
- 二格点耦合按规范键向量分组并按幅值排序。
- 四格点耦合通过 `normalize_four_sites` 分为 5 种拓扑类型
  （正方形、T 形、线形变体）。
- 结果同时以人可读文本和机器可读 JSON 产物写出。
