# 06-运行时

当前 `HubbardModel` + `Block` 核心的 active workchain、cache 与 I/O 契约。

## 1) Active 运行时边界 (MUST)

MUST:
- 当前 active 源码树已有生产级 `cuprate.main`，用于生成原始自旋耦合；
  也已有生产级 `cuprate.lce`，用于 linked-cluster subtraction。
- 当前 active 源码树已有生产级 `cuprate.embed`，用于把 LCE weights 嵌入到
  target coupling 输出。
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
model.set_symmetry(mode, twoSz=twoSz, twoS=twoS, scope=scope)
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

- 公共规范 CLI 模式为 `full`、`Sz` 和 `SzS2`。
- 固定 `twoSz` 和 `twoS` 数值是独立可选 selector 参数。
- `SCOPE` 控制 all-`twoSz` 的 `MODE=Sz` 和 `MODE=SzS2` 运行中的扇区范围；
  默认 `SCOPE=nonnegative` 构造 `twoSz >= 0`，`SCOPE=pm` 构造正负
  `twoSz`。

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
- runtime/workchain 调用方默认 `CACHE_MODE` 为 `save`。
- `HubbardModel.save(cache_dir)` 和 `Block.save(cache)` 必须写出同一种已求解 block 格式。

Code form:
```python
cache = Path(cache_dir) / cluster.label()
block.save(cache)
block = Block.load(cache, twoSz, twoS)
```

- 运行时路径构造集中在 `cuprate.paths`。
- 本征系统数据目录为：
  - `DATA`：`MODE=full`。
  - `DATA_twoSz`：`MODE=Sz`，默认 scope 和 `SCOPE=pm` 共享。
  - `DATA_twoSz_twoS`：`MODE=SzS2`，默认 scope 和 `SCOPE=pm` 共享。
- `SCOPE=pm` 不创建单独的本征系统 cache 目录。它复用按 block key
  区分的同一 cache，并可通过 `CACHE_MODE=partial` 补齐缺失 block。

Code form:
```text
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/DATA_twoSz/hole0_class0_idx0/twoSz_0_data.npz
```

## 4) Workchain 计算模型 (MUST)

MUST:
- 团簇枚举使用 `ClusterSets(N)`。
- ED、projection、downfolding、fitting 和 operator output 都只对每个同构
  family 的代表元计算一次，family 由 `(hole, class_idx)` 标识。
- 同一 family 中其他 `cluster_idx` 成员复用代表元的本征系统、projection 数据
  和已拟合的 exchange coefficients。
- 每个 `cluster_idx` 输出只保存将 family exchange 数据放到该 member 上所需的
  geometry metadata（`sites` 和 `indices`）。
- 除非用户显式禁用代表元复用，workchain 不得对每个同构 member 独立求解。

## 5) Workflow 选择 (MUST)

MUST:
- 规范 workflow key 是 `workflow`。
- 支持的 workflow 值为 `occ`、`energy`、`greedy`、`greedy_multi` 和 `adiabatic`。
- 旧名称如 `single`、`multi`、`multi_restart` 和 `adiabatic_restart`
  仅为 reference-data 名称，不是生产 workflow 值。
- `adiabatic` 种子是显式输入。调用方必须传入 `SEED_RESULTS`，即一个已经完成的
  `results.json` 路径。
- 种子的选择方案从 seed 文件的 `run_params.workflow` 读取。
- adiabatic 种子由前一 block 的本征向量和前一 selected indices 组成，
  并通过该 seed 文件的 projection artifact 加载。种子必须来自 projection 输出，
  而不能只来自 solve cache。
- 缺少 adiabatic 种子数据是错误。不得回退到同参数 baseline 运行。

## 6) 派生输出归属 (MUST)

MUST:
- 派生 projection 输出至少包含 selected indices、逐 block 选择诊断、
  `H_eff` 和 `T11` 指标。
- Fit 输出包含 family-level coefficients 和 fit metrics。
- workchain/result-output 层拥有派生输出的写入职责。
- `cuprate.io` 拥有 main、LCE 和 embed 输出的持久 payload 构造、
  schema-version 常量、JSON/NPZ 写出，以及人类可读文本 sidecar 写出职责。
- `Block` 拥有局部计算和可选 selection JSONL logging，但不拥有持久化结果 schema。
- 规范 JSON 形状定义在 `08-OPERATOR_OUTPUT.md`。

## 7) 运行时 CLI 参数 (MUST)

MUST:
- CLI 参数通过 `KEY=VALUE` 传递。
- `cuprate.cli` 拥有共享的 `KEY=VALUE` 解析、main-stage 默认值、
  main-stage `MODE`/`twoSz`/`twoS` 校验，以及 `cuprate.lce` 和
  `cuprate.embed` 的 seed-set 解析。
- CLI 键不区分大小写。
- 规范 key 包括：
  - `N`、`U`、`T`：物理参数。
  - `MODE`：`full`、`Sz` 或 `SzS2` 之一；默认 `full`。
  - `twoSz`、`twoS`：可选固定 block selector。`twoS` 要求
    `MODE=SzS2` 且固定 `twoSz`。
  - `SCOPE`：`nonnegative` 或 `pm` 之一；默认 `nonnegative`，只作用于
    all-`twoSz` 的 `MODE=Sz` / `MODE=SzS2` 运行。
  - `workflow`：`occ`、`energy`、`greedy`、`greedy_multi`、`adiabatic` 之一；
    默认 `occ`。
  - `CACHE_MODE`：`none`、`load`、`save`、`partial` 之一；默认 `save`。
  - `ROOT`：所有 `block_main`、`block_lce` 和 `block_embed` 输出的根目录；
    默认 `results`。
  - `SEED_RESULTS`：前一 `results.json`，仅 `workflow=adiabatic` 时必须提供。
  - `SEED_SET`：`ROOT` 下的 LCE/embed 文本文件；每个非注释行都是相对
    `ROOT` 的 main `results.json` 路径。
- `CACHE_DIR`、`OUTPUT_DIR` 和 `INPUTS` 不是生产 CLI key。
- `cuprate.lce` 只接受 `ROOT`、`N`、`U`、`T` 和 `SEED_SET`；在这个阶段中
  `N` 表示 `Nmax`。
- `cuprate.lce` 从 `SEED_SET` 读取 main results，校验它们覆盖
  `N=2..Nmax`，并校验它们的 `U/T` 等于 CLI 值。
- `cuprate.lce` 将 `lce_results.json` 写成 manifest，并把具体团簇
  weight JSON 文件和仅供检查的人类可读文本 sidecar 写到 `weights/N_<N>/` 下。
- `cuprate.embed` 只接受 `ROOT`、`N`、`U`、`T` 和同一个 `SEED_SET`；它必须
  从匹配的 `seed_<stem>` 目录读取对应的 `lce_results.json` manifest 以及它
  引用的 weight 文件。
- 标准不允许生产 key `TYPE`、`SZ`、`S`、`S2`、`SZ_IDX`、`S_IDX`、
  `BLOCKS`、`SELECT` 或 `MATCH_SPIN_SECTORS`。

## 8) 运行时目录契约 (MUST)

MUST:
- 三个运行时阶段只通过 stage 前缀区分：
  - `block_main`
  - `block_lce`
  - `block_embed`
- 通用 parameter directory token 是
  `N_{N}_nelec_{nelec}_U_{U:.4f}_t_{T:.4f}`。
- Main workflow 输出位于 `mode_* / workflow_*` 下。
- 默认 `SCOPE=nonnegative` 使用短 all-`twoSz` 路径 token：
  `mode_twoSz` 和 `mode_twoSz_twoS`。
- 显式 `SCOPE=pm` 使用 `mode_twoSz_pm` 和 `mode_twoSz_pm_twoS`。
- LCE 和 embed 输出位于 `seed_<stem>` 下，其中 `<stem>` 是 `SEED_SET`
  文件 stem。

Code form:
```text
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_full/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_0/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_pm/workflow_occ/results.json
ROOT/block_main/N_6_nelec_6_U_1.0000_t_0.0200/mode_twoSz_0_twoS_2/workflow_occ/results.json
ROOT/seed_sets/block.txt
ROOT/block_lce/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/lce_results.json
ROOT/block_lce/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/weights/N_2/hole0_class0_idx0.json
ROOT/block_lce/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/weights/N_2/hole0_class0_idx0.txt
ROOT/block_embed/N_6_nelec_6_U_1.0000_t_0.0200/seed_block/
```

## 9) 未来部分 family 运行 (MAY)

MAY:
- 第一版生产 `cuprate.main` workchain 对给定 `N` 计算完整的
  `(hole, class_idx)` family 集合。
- 未来优化可以加入显式 family 选择参数，例如
  `FAMILIES=0:0,0:1,1:0`。
- 选择单位是 family `(hole, class_idx)`，不是单个 `cluster_idx`，
  除非显式重新设计代表元复用机制。
- Family 选择必须先作用在全局 family 任务列表上，然后再做 MPI rank
  分配。不得在各个 rank 已经切片之后独立过滤。

Code form:
```python
families = _enumerate_families(N)
families = _filter_families(families, selected_families)
for family in families[rank::size]:
    process_family(family)
```

- 部分 family 输出必须在 `results.json` 中显式标记；除非未来显式请求
  debug/partial 模式，下游 LCE 必须拒绝 partial main outputs。
- 对于 `workflow=adiabatic`，seed `results.json` 只需要包含当前 partial run
  将要处理的 selected families。
