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
- 文件名：`hole{h}_class{c}_cluster{v}_results.txt`。
- 结构：
  ```
  === Summary ===
  Hole: {h}  Class: {c}  Cluster: {v}
  N={N}  Sites: (x0,y0) (x1,y1) ...
  R²: {r_squared}  |T11-I|: {t11m1_norm}  Overlap: {overlap}

  === Two-site couplings ===
  J1  vector (dx,dy):
      Sites i-j  (xi,yi)-(xj,yj):  {coefficient}
      ...

  === Four-site couplings ===
  Group K1: sites {a, b, c, d}
      (Sa·Sb)(Sc·Sd):  {coefficient}
      (Sa·Sc)(Sb·Sd):  {coefficient}
      (Sa·Sd)(Sb·Sc):  {coefficient}

  === Six-site couplings ===
  Group L1: sites {a, b, c, d, e, f}
      (Sa·Sb)(Sc·Sd)(Se·Sf):  {coefficient}
      ...

  === Fit quality ===
  Constant term:   {c0}
  Relative error:  {rel_err}
  Residual:        {residual}
  R²:              {r_squared}
  |T11-I|:         {t11m1_norm}
  Overlap:         {overlap}
  ```
- 格式规则：
  - 摘要块放在最前面，打开文件即可看到关键指标。
  - 系数为实数浮点数；虚部在代码内校验，不输出到文件。
  - 团簇中不存在的键方向不输出（无占位行）。
  - 多格点组按规范组顺序标为 `K1`、`K2`、... 和 `L1`、`L2`、...。
  - 多格点配对按规范排序（§3）列出，
    使用显式算符记号 `(Si·Sj)(Sk·Sl)`。
  - 小量使用科学记号（`1.23e-7` 而非 `0.0000001230`）。
  - `projection_analysis` 的文本输出也复用同一套算符组顺序，只是不打印耦合系数。

## 6) 机器可读输出 (MUST)

MUST:
- 逐团簇 sidecar JSON 文件写为 `hole{h}_class{c}_cluster{v}_results.json`。
  每个 sidecar 包含与汇总结果中对应 entry 相同的 payload 形状。
- 文件名：运行输出目录中的 `results.json`。
- 包含该运行所有团簇的结果，汇总在单个文件中。
- `projection` 对 spin-coupling 结果是必需的，因为它是 selected eigenstate
  indices 和逐 block projection diagnostics 的持久位置。
- 结构：
  ```json
  {
    "schema_version": 2,
    "result_kind": "spin_couplings",
    "run_params": {"U": 1.0, "t": 0.24, "N": 4, "MODE": "full", "workflow": "greedy"},
    "clusters": [
      {
        "hole": 0,
        "class_idx": 0,
        "cluster_idx": 0,
        "sites": [[0,0], [1,0], [0,1], [1,1]],
        "projection": {
          "method": "greedy",
          "blocks": [
            {
              "block": "twoSz_all_twoS_all",
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
    ]
  }
  ```
- `operators.groups` 中每个条目必须具有一致的结构，不论元数：
  ```json
  {
    "arity": 2,
    "vector": [1, 0],
    "label": "J1",
    "terms": [{"sites": [0, 1], "coefficient": {"real": ..., "imag": ...}}]
  }
  ```
  对于多格点组（arity 4, 6, ...），`vector` 为 `null`，`label` 遵循
  §1 中的前缀约定（`K1`、`K2`、...；`L1`、`L2`、...）。
- 每个团簇的元数据（`rank`、`computation_time_s`）放在 `metadata` 子对象中，
  不混入顶层。
- `projection.blocks` 中的每个条目记录该 block 已求解本征向量框架中的
  selected eigenvector column indices。
- `selection_info` 存储方法特定诊断。它可以包含 `greedy_multi` 的紧凑摘要字段；
  详细 trial logs 也可以由 selector 写成 JSONL，但最终 `results.json` 是持久输出。
- `projection_analysis` 结果使用同样的汇总结构，包含 `projection`，
  但不包含 `operators` 或 `fit`。
- 未来 LCE/embed workchains 必须从这个汇总 JSON 读取，而不是从文本文件读取。
