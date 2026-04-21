# 06-运行时

MPI 任务分配、I/O 约定与路径布局。

## 1) MPI 执行模型 (MUST)

MUST:
- 代码使用 `mpi4py` 的 `MPI.COMM_WORLD`。
- Rank 0 为广播和打印的根节点。
- 任务分配：轮询（round-robin），多余任务分配给后序 rank。

Code form:
```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# distribute_work: items_per_rank = total // size, 余数分给最后的 rank
```

## 2) 执行流水线 (MUST)

MUST:
- 三个入口点按顺序执行（不在同一进程中）：
  1. `python -m cuprate.main` — 团簇枚举、哈密顿量求解、选择、降维、拟合。
  2. `python -m cuprate.lce` — 读取步骤 1 的结果，执行子团簇减法。
  3. `python -m cuprate.embed` — 读取步骤 2 的结果，��入超胞。
- 仅 `cuprate.main` 使用 MPI。LCE 和 embed 为单进程。

## 3) 目录布局 (MUST)

MUST:
- `cuprate.main` 写入：
  ```
  ./Block/{base_dir}/{run_dir}/
  ```
  其中 `base_dir = U{U:.4f}_t{t:.4f}`，
  `run_dir` 遵循规范语法：
  ```
  N{N}[ _twoSz_<value> | _twoSz_all ][ _twoS_<value> | _twoS_all ][_match_spin_sectors][_{workflow}][_restart]
  ```
  各模式对应：
  - `full` -> `N{N}`
  - `fixed_sz` -> `N{N}_twoSz_<value>`
  - `fixed_sz_s2` -> `N{N}_twoSz_<value>_twoS_<value>`
  - `block_sz_full` -> `N{N}_twoSz_all`
  - `block_sz_s2_full` -> `N{N}_twoSz_all_twoS_all`
- `twoSz` / `twoS` 路径槽中仅允许 `all` 和 `n` 作为非数字标记。
- 负的固定值用 `n` 替代负号，如 `twoSz_n1` 表示 $2S_z = -1$。

- `cuprate.lce` 从 `data_transfer/Block/Block_{base_dir}/` 读取，写入 `data_transfer/LCE/LCE_{base_dir}/`。

- `cuprate.embed` 从 `data_transfer/LCE/LCE_{base_dir}/` 读取，写入 `data_transfer/Embed/Embed_{base_dir}/`。

## 4) 文件命名 (MUST)

MUST:
- 每团簇结果：`hole{h}_class{c}_cluster{v}_results.txt` 和 `.json`。
- 本征系统：`hole{h}_class{c}_eigvals.npy`、`hole{h}_class{c}_eigvecs.npy`。
- 派生数据：`_Heff.npy`、`_T11m1.npy`、`_t11_selected_indices.npy`、
  `_double_occupation_expectation.npy`、`_states.npy`、`_S2_diagonal.npy`。

## 5) 结果输出 (MUST)

MUST:
- 算符定义、规范排序和输出文件格式（`.txt` 和 `.json`）
  定义在 `08-OPERATOR_OUTPUT.md` 中。

## 6) 重启协议 (MUST)

MUST:
- 当 `restart=True` 时，代码从磁盘读取预计算的本征系统而非重新对角化。
- 仅重新执行选择 + 降��� + 拟合步骤。
- 重启目录包含每个团簇的 `_eigvals.npy` 和 `_eigvecs.npy`。
- `adiabatic` 工作流从**前一**参数点读取本征系统和选定索引，
  前一参数由 `t_previous = t - delta` 计算。
- 在绝热扫描的首个参数点，如果前一参数点的绝热结果不存在，
  则用同参数的基线运行（`workflow=None`）作为绝热初始种子。

## 7) CLI 参数 (MUST)

MUST:
- 所有参数通过命令行 `KEY=VALUE` 传递。
- CLI 键不区分大小写。
- 关键参数：
  - `N`、`U`、`T`：物理参数。
  - `MODE`：`full`、`fixed_sz`、`block_sz_full`、`fixed_sz_s2`、`block_sz_s2_full` 之一。
  - `MODE=all` 作为 `MODE=full` 的输入别名接受。
  - `twoSz`、`twoS`：固定扇区运行的物理对称性标签。
    实现可以接受这些键的任意大小写形式。
  - `workflow`：工作流（`occ`、`energy`、`single`、`multi`、`adiabatic`）。
    实现可以接受该键的任意大小写形式。
  - `SELECT`：选择模式（`block` 或默认）。
  - `RESTART`：`true`/`false`。
  - `DELTA`：绝热步长。
  - `NCELL`、`NCUT`：嵌入参数。
  - `MATCH_SPIN_SECTORS`：`true`/`false`。
- 标准不允许 `TYPE`、`SZ`、`S`、`S2`、`SZ_IDX` 或 `S_IDX`。
- `twoSz` 和 `twoS` 必须满足 `00-CONVENTIONS.md` 中定义的物理约束。
