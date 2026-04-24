# 06-运行时

当前 `HubbardModel` + `Block` 核心的 active workchain、cache 与 I/O 契约。

## 1) Active 运行时边界 (MUST)

MUST:
- 当前 active 源码树没有生产级 `cuprate.main`、`cuprate.lce` 或
  `cuprate.embed` 入口点。
- 运行时代码必须基于 active 模块重建：
  `clusters.py`、`states.py`、`sectors.py`、`manifold.py`、`hubbard.py`
  和 `mpi.py`。
- `src/cuprate/back/` 仅作为参考材料。不要添加会执行或保留旧 workchain
  结构的兼容路径。

## 2) 单团簇生命周期 (MUST)

MUST:
- 单团簇计算遵循以下分阶段生命周期：

Code form:
```python
model = HubbardModel(cluster, U, t)
model.set_symmetry(mode, twoSz=twoSz, twoS=twoS)
model.build_hamiltonians()
model.solve(cache_mode=cache_mode, cache_dir=cache_dir)
model.project(method=workflow, **select_kwargs)
model.fit(bond_groups=bond_groups)
```

- 可选重构必须显式调用：

Code form:
```python
model.merge_by_s2()  # 合并每个 twoSz 内的 twoS 扇区
model.merge_by_sz()  # 将 fixed-twoSz 块合并成一个 full 块
```

- 公共规范模式为 `full`、`fixed_sz`、`block_sz_full`、
  `fixed_sz_s2` 和 `block_sz_s2_full`。
- 内部 mode 常量和 cache bucket 由 `hubbard.py` 拥有。

## 3) 本征系统 Cache (MUST)

MUST:
- solve cache 只存储已求解的 blocks：basis states、本征值、本征向量、
  Hamiltonian，以及可选 `basis_transform`。
- solve cache 不存储 selected indices、`H_eff`、`T11` 或 fit 结果。
- `HubbardModel.solve()` 精确支持四种 cache mode：

| cache_mode | 行为 |
|------------|------|
| `none` | 在内存中求解每个当前 block；不使用磁盘 cache |
| `load` | 从磁盘加载每个当前 block；缺任何 block 都失败 |
| `save` | 求解每个当前 block，然后保存每个 block |
| `partial` | 加载已有 block，并求解/保存缺失 block |

- `load`、`save` 和 `partial` 需要 `cache_dir`；`none` 禁止传入 `cache_dir`。
- `HubbardModel.save(cache_dir)` 和 `Block.save(cache)` 必须写出同一种已求解 block 格式。

Code form:
```python
cache = Path(cache_dir) / model.label() / cluster.label() / model._bucket
block.save(cache)
block = Block.load(cache, twoSz, twoS)
```

## 4) Workchain 计算模型 (MUST)

MUST:
- 团簇枚举使用 `ClusterSets(N)`。
- ED、projection 和 downfolding 只对每个同构 family 的代表元计算一次，
  family 由 `(hole, class_idx)` 标识。
- 同一 family 中其他 `cluster_idx` 成员复用代表元的本征系统/projection 数据。
- fitting 和 reporting 仍可对每个 `cluster_idx` 输出，并使用该 member cluster
  自己的 operator groups。
- 除非用户显式禁用代表元复用，workchain 不得对每个同构 member 独立求解。

## 5) Workflow 选择 (MUST)

MUST:
- 规范 workflow key 是 `workflow`。
- 支持的 workflow 值为 `occ`、`energy`、`greedy`、`greedy_multi` 和 `adiabatic`。
- 旧名称如 `single`、`multi`、`multi_restart` 和 `adiabatic_restart`
  仅为 reference-data 名称，不是生产 workflow 值。
- `adiabatic` 种子由 `DELTA` 定位：在相同 cache/output root 约定下，
  前一点为 `t_previous = t - DELTA`。
- adiabatic 种子由前一 block 的本征向量和前一 selected indices 组成。
  种子必须来自前一点的 projection 输出，而不能只来自 solve cache。
- 缺少 adiabatic 种子数据是错误。不得回退到同参数 baseline 运行。

## 6) 派生输出归属 (MUST)

MUST:
- 派生 projection 输出至少包含 selected indices、逐 block 选择诊断、
  `H_eff` 和 `T11` 指标。
- Fit 输出包含 coefficients 和 fit metrics。
- workchain/result-output 层拥有派生输出的写入职责。
- `Block` 拥有局部计算和可选 selection JSONL logging，但不拥有持久化结果 schema。
- 规范 JSON 形状定义在 `08-OPERATOR_OUTPUT.md`。

## 7) 未来 CLI 参数 (MUST)

MUST:
- CLI 参数一旦重新引入，就通过 `KEY=VALUE` 传递。
- CLI 键不区分大小写。
- 规范 key 包括：
  - `N`、`U`、`T`：物理参数。
  - `MODE`：五种 public mode 之一。
  - `twoSz`、`twoS`：`MODE` 需要时的固定扇区标签。
  - `workflow`：`occ`、`energy`、`greedy`、`greedy_multi`、`adiabatic` 之一。
  - `CACHE_MODE`：`none`、`load`、`save`、`partial` 之一。
  - `CACHE_DIR`：solve cache 根目录。
  - `DELTA`：adiabatic 步长。
- 标准不允许生产 key `TYPE`、`SZ`、`S`、`S2`、`SZ_IDX`、`S_IDX`、
  `SELECT` 或 `MATCH_SPIN_SECTORS`。
