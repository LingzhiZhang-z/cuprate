# 05-LCE 与 Embed

链接团簇展开（子团簇减法）与 embed 输出。

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
- 每个子图通过 NN 图同构匹配到已有的团簇计算结果。
- LCE 输入必须包含连续的 `N=2..Nmax` 原始自旋耦合结果。
- LCE 输入由与 embed 共享的 `SEED_SET` 文本文件选择。命令行仍显式
  提供 `ROOT`、`N`、`U` 和 `T`；seed 文件只列出使用哪些 main
  `results.json`。
- 每个非空且非注释的 seed 行都是相对 `ROOT` 的路径。
- main seeds 的 `run_params.N` 必须刚好覆盖 `2..Nmax`，且
  `run_params.U/T` 必须等于 CLI 的 `U/T`。
- `MODE`、`workflow`、`twoSz`、`twoS` 和 `SCOPE` 可以在 seeds 之间不同。
  它们是 provenance 字段，不是 LCE 全局 selector。

Code form:
```python
subgraphs, indices = get_connected_subgraphs(cluster, min_size=2)
match = find_cluster_match(subgraph, clusters)
# match = (nsites, hole, class_idx, variant_idx, mapping)
```

## 3) 算符减法 (MUST)

MUST:
- LCE 从 `08-OPERATOR_OUTPUT.md` 中定义的 `results.json` schema 读取
  fitted operators；不读取文本文件或 projection NPZ 文件。
- 算符数据包含一个常数项和带 key 的自旋耦合项。
- 减法通过同构映射将子团簇格点索引映射到父团簇索引，
  然后逐项减去系数。
- 算符身份由规范 `key` 字段决定，而不是显示用的 `J*`、`K*` 或 `L*`
  标签。

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

## 4) LCE 输出 (MUST)

MUST:
- LCE 入口写出 `lce_results.json`、`weights/` 下的逐团簇 weight 文件，
  以及最小 `lce_summary.txt`。
- `lce_results.json` 的 `result_kind = "lce_spin_couplings"`。
- `lce_results.json` 是 manifest。它记录 `SEED_SET` provenance，并为
  每个具体团簇指向一个 weight 文件。
- Weight 文件位于 `weights/N_<N>/` 下，命名为
  `hole{h}_class{c}_idx{v}.json`。`N_<N>` 层级是必须的，避免不同 `N`
  中同名 cluster 互相覆盖。
- 每个 weight 文件的 `result_kind = "lce_cluster_weight"`。
- 每个 weight 文件记录 `sites`、`indices`、使用与原始自旋耦合相同
  operator group schema 的净算符 `W(C)`，以及 reconstruction 诊断。

Manifest form:
```json
{
  "schema_version": 2,
  "result_kind": "lce_spin_couplings",
  "seed_set": "block",
  "seed_token": "seed_block",
  "seed_set_file": "seed_sets/block.txt",
  "seed_set_sha256": "...",
  "source_inputs": [
    {
      "N": 2,
      "results_json": "block_main/.../results.json",
      "run_params": {
        "MODE": "Sz",
        "workflow": "occ",
        "twoSz": 0,
        "twoS": null,
        "SCOPE": "nonnegative"
      }
    }
  ],
  "run_params": {
    "N_min": 2,
    "N_max": 4,
    "U": 1.0,
    "T": 0.24,
    "seed_token": "seed_block"
  },
  "weights": [
    {
      "N": 4,
      "hole": 0,
      "class_idx": 1,
      "cluster_idx": 0,
      "weight_file": "weights/N_4/hole0_class1_idx0.json"
    }
  ]
}
```

Weight-file form:
```json
{
  "schema_version": 1,
  "result_kind": "lce_cluster_weight",
  "N": 4,
  "hole": 0,
  "class_idx": 1,
  "cluster_idx": 0,
  "sites": [[0, 0], [1, 0], [0, 1], [1, 1]],
  "indices": [0, 1, 2, 3],
  "operators": {"constant_term": {"real": 0.0, "imag": 0.0}, "groups": []},
  "raw_summary": {},
  "diagnostics": {
    "subcluster_count": 0,
    "reconstruction_error": 0.0,
    "net_term_count": 0,
    "max_abs_net_coefficient": 0.0
  }
}
```

Validation:
- 对每个团簇，所有连通子团簇净贡献 $W(C')$（$C' \subseteq C$）求和
  必须在 `ATOL["loose"]` 内恢复原始 fitted operators $O(C)$。
- 对 `N=2`，净算符等于原始算符。

## 5) Embed 输入与输出 (MUST)

MUST:
- embed 阶段必须读取 `lce_results.json` manifest 以及它引用的
  weight 文件，而不是原始 `results.json`。
- embed 阶段使用与 LCE 相同的 `ROOT`、`N`、`U`、`T` 和 `SEED_SET`
  输入。它通过 `seed_<stem>` 目录定位对应的 LCE manifest。
- 输出位于
  `ROOT/block_embed/N_{Nmax}_nelec_{nelec}_U_{U:.4f}_t_{T:.4f}/seed_<stem>/`
  下。
- `embed_results.json` 是 `result_kind = "embedded_spin_couplings"` 的 manifest。
- `embed_summary.txt` 记录源 LCE 文件、seed-set provenance 和输出数量。
- `two_site.txt` 包含 `Nmax` 内所有候选二格点键向量；未出现在累积 LCE 输出中的
  向量写为 `None`。
- 多格点 cluster 文件命名为 `N{N}_hole{h}_class{c}_idx{i}.txt`。
- 生产代码使用 `embed`，不是 `periodize`。

Code form:
```text
ROOT/block_embed/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/two_site.txt
ROOT/block_embed/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/clusters/N4_hole0_class0_idx0.txt
```

## 6) Embed 匹配算法 (MUST)

MUST:
- 数值算法是 target-driven。
- C4v 只用于生成 parent cluster 的 distinct orientations。
- target 是 concrete 的。匹配只允许平移；不要旋转、镜像、canonicalize，
  也不要除以 target multiplicity。
- 不计算或输出 `m(candidate)`。
- 不做 L2 orbit averaging，也不建立 `Wtilde` 数值路径。
- embed 不再做一次 Möbius subtraction。

Code form:
```text
for each output target P:
    value(P) = 0
    seen(P) = false
    for each LCE weight W(C):
        for each distinct C4v orientation g(C):
            for each term t in W(C) with arity(P):
                if g(t) matches P by translation only:
                    value(P) += coeff(t)
                    seen(P) = true
```

Validation:
- Plaquette distinct orientations = 1。
- Four-site line distinct orientations = 2。
- L-shape distinct orientations = 8。
- Synthetic plaquette 中每条 NN two-site term coefficient 设为 `1.0` 时，
  `two_site.txt[(1,0)] = 2.0`。
- Synthetic L-tetromino 中每条 NN two-site term coefficient 设为 `1.0` 时，
  `two_site.txt[(1,0)] = 12.0`。

## 7) Embed 二格点候选顺序 (MUST)

MUST:
- 二格点候选使用规范向量 `(dx, dy)`，满足 `dx >= dy >= 0` 且 `dx > 0`。
- 候选向量满足 `dx + dy + 1 <= Nmax`。
- 排序按 Manhattan shell、距离平方、再按向量字典序。
- 这样增大 `Nmax` 时只会在末尾增加新的候选 shell，不重排前面的条目。

Code form:
```python
sorted(candidates, key=lambda v: (v[0] + v[1], v[0]**2 + v[1]**2, v[0], v[1]))
```

## 8) Embed 多格点文本文件 (MUST)

MUST:
- 四格点和六格点 embed 文件列出 cluster sites、site indices、
  所有 perfect pairings 以及每个 pairing 的 coefficient。
- 缺失的 pairings 写为 `None`。
- 文件写在 `clusters/` 下。
- 可以添加简单 text graph section，但 v1 不强制。
