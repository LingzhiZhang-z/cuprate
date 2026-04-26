# 08-算符输出

自旋耦合算符定义、多格点算符的规范配对排序、结果输出格式。

## 1) 算符类型 (MUST)

MUST:
- 自旋耦合拟合将 $H_{\text{eff}}$ 分解为自旋算符的线性组合，
  按涉及的格点数分组：

| 元数 | 算符形式 | 标签前缀 |
|------|----------|----------|
| 1 | 恒等算符 $I$（常数 $E_0$） | — |
| 2 | $\mathbf{S}_i \cdot \mathbf{S}_j$ | J |
| 4 | $(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)$ | K |
| 6 | $(\mathbf{S}_i \cdot \mathbf{S}_j)(\mathbf{S}_k \cdot \mathbf{S}_l)(\mathbf{S}_m \cdot \mathbf{S}_n)$ | L |

- 两格点算符按**键向量**分类（见 `01-CLUSTERS.md` §7），
  按欧氏距离递增标记为 `J1`、`J2`、`J3`、...
- 多格点算符（元数 $\geq 4$）由所有 $2n$ 个格点的连通子集生成
  （见 `01-CLUSTERS.md` §8），并在输出层做规范化。

## 2) 多格点配对数 (MUST)

MUST:
- 对于 $2n$ 个连通格点组，将其分成 $n$ 个 Heisenberg 对的方式数为：

Math:
$$
(2n-1)!! = \frac{(2n)!}{2^n \, n!}
$$

| 格点数 | 对数 | 配对数 |
|--------|------|--------|
| 4 | 2 | 3 |
| 6 | 3 | 15 |
| 8 | 4 | 105 |

## 3) 规范多格点输出顺序 (MUST)

MUST:
- 多格点项的顺序在**序列化输出层**定义，内部枚举顺序不是契约的一部分。
- 每个多格点项按以下规则规范化：
  1. 每一对先写成 `(min(i,j), max(i,j))`。
  2. 每一对分配一个描述符
     `(d^2, dx_canon, dy_canon, i, j)`，其中 `(dx_canon, dy_canon)` 为规范键方向，
     满足 `dx > 0`，或 `dx == 0` 且 `dy > 0`。
  3. 同一项内部的所有 pair 按该描述符排序。
- 各项先按支撑集合分组，也就是参与格点的有序集合。组顺序取
  支撑格点坐标 `(x, y, site_index)` 的字典序。
- 同一支撑集合内，各项再按 pair 描述符元组的字典序排序。

- 这保证了：
  - 四格点和六格点项输出稳定；
  - 更局域的配对排在前面；
  - 文本和 JSON 共用同一套 K/L 顺序。

- LCE 和 embed 仍然必须通过规范 key（`k4s`、`k6s`、`k8s`）匹配算符，
  不能依赖 K/L 标签位置。
- 规范 key 在每个 term 上序列化为排序后的 pair list：
  `[[i,j]]`、`[[i,j],[k,l]]` 或 `[[i,j],[k,l],[m,n]]`。

Code form:
```python
def canonical_pair_descriptor(cluster, i, j):
    i, j = sorted((i, j))
    dx = cluster[j][0] - cluster[i][0]
    dy = cluster[j][1] - cluster[i][1]
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx*dx + dy*dy, dx, dy, i, j)
```

Validation:
- 同一算例重复运行时，多格点组顺序和组内项顺序都必须保持不变。

## 4) 系数结构 (MUST)

MUST:
- 拟合为每个算符项产生一个系数。系数向量的排列顺序为：
  1. 常数项 $c_0$。
  2. 两格点项，按键向量分组（J1, J2, J3, ...），每条键单独一个系数。
  3. 四格点项，按连通格点组分组，每种配对按规范排序。
  4. 六格点项，同样结构。
- 每个系数为复数。对于厄米的 $H_{\text{eff}}$，虚部必须可忽略；
  这在代码内校验，不在输出格式中处理。

## 5) 人类可读输出 (MUST)

MUST:
- 人类可读文本文件只是检查用 sidecar。LCE/embed 和 adiabatic seed 加载必须继续
  读取 JSON/NPZ 机器输出。
- Exchange 文本文件名：`exchanges/hole{h}_class{c}_exchange.txt`。
- Cluster 文本文件名：`clusters/hole{h}_class{c}_clusters.txt`。
- LCE weight 文本文件名：`weights/hole{h}_class{c}_idx{v}.txt`。
- Embed 文本文件为 `two_site.txt` 以及可选的 `clusters/` 下文件。
- Exchange 文本结构：
  ```
  Family: hole={h} class={c} representative={v}
  N={N}
  Sites: 0:(x0,y0)  1:(x1,y1) ...

  Projection:
    method={workflow} artifact={projection_npz}
    block={block} twoSz={twoSz|all} twoS={twoS|all} spin_dim={d} selected={d}
      selected_indices=...

  Fit:
    R2={r_squared} relative_error={rel_err} residual={residual}
    |T11-I|={t11m1_norm} overlap={overlap}
    constant={c0}

  Couplings:
    J1  vector=(dx,dy)
      (S0.S1) sites=0-1 coords=(x0,y0)-(x1,y1) coefficient={coefficient}
    K1  support=0,1,2,3 coords=0:(x0,y0) ...
      (S0.S1)(S2.S3) sites=0-1 2-3 coords=... coefficient={coefficient}
  ```
- Cluster 文本结构：
  ```
  Family: hole={h} class={c} representative={v}
  N={N}
  Representative sites: 0:(x0,y0)  1:(x1,y1) ...

  Clusters:
    cluster {cluster_idx}:
      {operator_index} -> ({x},{y})
  ```
- LCE weight 文本结构：
  ```
  LCE weight: N={N} hole={h} class={c} cluster={v}
  Sites: 0:(x0,y0)  1:(x1,y1) ...

  Diagnostics:
    subclusters={count}
    reconstruction_error={error}

  Raw summary:
    term_count={count}
    constant={raw_c0}

  Net couplings:
    constant={net_c0}

  Couplings:
    J1  vector=(dx,dy)
      (S0.S1) sites=0-1 coords=(x0,y0)-(x1,y1) coefficient={coefficient}
  ```
- 格式规则：
  - 摘要行放在最前面，打开文件即可看到关键指标。
  - 当虚部可忽略时，系数打印为实数浮点数。
  - 团簇中不存在的键方向不输出（无占位行）。
  - 多格点组按规范组顺序标为 `K1`、`K2`、... 和 `L1`、`L2`、...。
  - 多格点配对按规范排序（§3）列出，
    使用显式算符记号 `(Si.Sj)(Sk.Sl)`。
  - 小量使用科学记号（`1.23e-7` 而非 `0.0000001230`）。
  - `projection_analysis` 的文本输出也复用同一套算符组顺序，只是不打印耦合系数。

## 6) 机器可读输出 (MUST)

MUST:
- 文件名：main workflow 输出目录中的 `results.json`：
  `ROOT/block_main/N_{N}_nelec_{nelec}_U_{U:.4f}_t_{T:.4f}/mode_*/workflow_*/results.json`。
- `results.json` 是 manifest。它记录 run parameters，并为每个
  `(hole, class_idx)` family 指向一个 exchange 文件和一个 cluster-geometry 文件。
- 当前生产 `cuprate.main` 输出表示给定 `N` 的完整 family 集合。
- 未来 partial-family 优化必须先在 manifest 中显式标记不完整性，LCE 才能
  被允许读取它。预期的未来形状是：
  ```json
  {
    "complete_family_set": false,
    "family_selection": {
      "mode": "explicit",
      "families": [[0, 0], [0, 1]]
    }
  }
  ```
- Exchange 文件位于 `exchanges/`，命名为 `hole{h}_class{c}_exchange.json`。
- Cluster-geometry 文件位于 `clusters/`，命名为 `hole{h}_class{c}_clusters.json`。
- 不再写 per-cluster sidecar JSON。
- 每个 exchange 文件中必须包含 `projection`，因为它是 selected eigenstate
  indices 和逐 block projection diagnostics 的持久位置。
- Manifest 结构：
  ```json
  {
    "schema_version": 5,
    "result_kind": "spin_couplings",
    "run_params": {
      "U": 1.0,
      "T": 0.24,
      "N": 4,
      "nelec": 4,
      "MODE": "full",
      "twoSz": null,
      "twoS": null,
      "SCOPE": "nonnegative",
      "workflow": "greedy",
      "parameter_token": "N_4_nelec_4_U_1.0000_t_0.2400",
      "mode_token": "mode_full",
      "workflow_token": "workflow_greedy"
    },
    "families": [
      {
        "hole": 0,
        "class_idx": 0,
        "exchange_file": "exchanges/hole0_class0_exchange.json",
        "clusters_file": "clusters/hole0_class0_clusters.json"
      }
    ]
  }
  ```
- Exchange 文件结构：
  ```json
  {
    "schema_version": 5,
    "result_kind": "spin_coupling_exchange",
    "N": 4,
    "hole": 0,
    "class_idx": 0,
    "representative_cluster_idx": 0,
    "representative_sites": [[0,0], [1,0], [0,1], [1,1]],
    "projection": {
      "method": "greedy",
      "artifact": "artifacts/hole0_class0_projection.npz",
      "blocks": [
        {
          "block": "full",
          "twoSz": null,
          "twoS": null,
          "selected_indices": [0, 1, 2, 3],
          "t11_minus_1_norm": 0.0,
          "overlap": null,
          "selection_info": {}
        }
      ]
    },
    "operators": {
      "constant_term": {"real": ..., "imag": ...},
      "groups": [...]
    },
    "fit": {
      "relative_error": ...,
      "residual": ...,
      "r_squared": ...,
      "t11_minus_1_norm": ...,
      "overlap": ...
    },
    "metadata": {
      "rank": 0,
      "computation_time_s": 1.23
    }
  }
  ```
- Cluster-geometry 文件结构：
  ```json
  {
    "schema_version": 5,
    "result_kind": "cluster_family_geometry",
    "N": 4,
    "hole": 0,
    "class_idx": 0,
    "representative_cluster_idx": 0,
    "representative_sites": [[0,0], [1,0], [0,1], [1,1]],
    "clusters": [
      {
        "cluster_idx": 0,
        "sites": [[0,0], [1,0], [0,1], [1,1]],
        "indices": [0, 1, 2, 3]
      }
    ]
  }
  ```
- `operators.groups` 中每个条目必须具有一致的结构，不论元数：
  ```json
  {
    "arity": 2,
    "vector": [1, 0],
    "label": "J1",
    "terms": [
      {
        "sites": [0, 1],
        "key": [[0, 1]],
        "coefficient": {"real": ..., "imag": ...}
      }
    ]
  }
  ```
  对于多格点组（arity 4, 6, ...），`vector` 为 `null`，`label` 遵循
  §1 中的前缀约定（`K1`、`K2`、...；`L1`、`L2`、...）。
- 多格点组还记录 `support`，即参与格点的有序集合。
- 每个 term 必须包含 `key`。LCE/embed 必须用 `key` 判断算符身份；
  `sites` 和 `label` 只是序列化/显示辅助。
- Family exchange 元数据（`rank`、`computation_time_s`）放在 `metadata`
  子对象中，不混入顶层。
- 在 cluster-geometry 文件中，`indices[k]` 是 `sites[k]` 对应的
  family/operator site index。当前代表元重排后的 cluster enumeration 通常写
  `[0, 1, ..., N-1]`。
- `projection.blocks` 中的每个条目记录该 block 已求解本征向量框架中的
  selected eigenvector column indices。
- `run_params.SCOPE` 是必需字段，并且所有被 LCE 和 embed workflow 消费的
  main 输入都必须具有相同的 `SCOPE`。
- 对于 `workflow=adiabatic`，`run_params.adiabatic_seed` 记录 seed
  `results.json` 路径、seed schema version、seed run parameters 和 seed workflow。
  `projection.blocks` 中每个条目还记录它的 seed block label 和 seed selected indices。
- Projection artifact 必须包含足够的数据来作为后续 adiabatic run 的 seed：
  block labels、basis states、Fock-coordinate eigenvectors、selected indices、
  `H_eff` 和 `T11` metrics。
- `selection_info` 存储方法特定诊断。它可以包含 `greedy_multi` 的紧凑摘要字段；
  详细 trial logs 也可以由 selector 写成 JSONL，但最终 `results.json` 是持久输出。
- `projection_analysis` 结果使用同样的 family manifest 结构，包含 `projection`，
  但不包含 `operators` 或 `fit`。
- LCE workchains 必须从 `results.json` manifest 以及其引用的 exchange
  和 cluster-geometry JSON 文件读取，而不是从文本文件读取。
- LCE 输出写出 `lce_results.json` manifest 以及它引用的
  `weights/hole{h}_class{c}_idx{v}.json` 文件。每个 weight 文件使用同一套
  `operators` schema 记录 net couplings。
- 每个 LCE weight JSON 可以有一个同 stem 的文本 sidecar。该文本文件不是下游阶段的输入。
- Embed workchains 必须读取 LCE manifest 以及它引用的 weight 文件。
