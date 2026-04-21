# 测试问题清单 (2026-04-12)

所有问题均早于本次代码清理存在，与清理无关。

## 一、测试文件 import 不存在的符号（3 个文件完全无法加载）

### 1. `tests/test_embed_lce_cli.py`

引用了从未存在于当前代码库的符号：

```python
from cuprate.clusters import bond_analysis_spin          # 不存在
from cuprate.embed import parse_arguments                 # 不存在
from cuprate.io import read_coords_file_Block, read_operators  # 不存在
from cuprate.lce import parse_arguments, write_operators  # 不存在
from cuprate.main import cluster_save_results             # 不存在
```

**处理方式**：整个文件需要重写或删除。这些符号对应的是重构前的旧 API。

### 2. `tests/test_mode_routing.py`

```python
from cuprate.main import parse_arguments  # 不存在
```

测试内容使用旧 CLI 参数 `SZ_IDX`、`S_IDX`（已被 `twoSz`、`twoS` 取代）。

**处理方式**：用当前 `parse_main_cli_args` + `resolve_mode_spec` 重写。

### 3. `tests/test_sz0_output_and_projection.py`

```python
from cuprate.clusters import bond_analysis_spin, canonical_bond_type  # 不存在
from cuprate.hubbard import Hubbard_SingleBand                        # 旧类名，现为 HubbardModel
from cuprate.io import load_array_compat                              # 不存在
from cuprate.main import cluster_process_work_item                    # 已移至 cuprate.main.solver
```

**处理方式**：更新 import 路径并适配当前 API，或删除。

---

## 二、测试引用了未实现的函数（1 个文件，5/5 测试失败）

### 4. `tests/test_shared_periphery.py`

5 个测试中有 5 个失败：

| 测试 | 缺失符号 |
|------|----------|
| `test_parse_bool_arg_and_iter_cli_assignments` | `iter_cli_assignments` — `io.py` 中不存在 |
| `test_build_legacy_result_dirs` | `build_legacy_result_dirs` — `io.py` 中不存在 |
| `test_find_existing_cluster_report_prefers_primary...` | `find_existing_cluster_report` — `io.py` 中不存在 |
| `test_find_existing_run_dir_uses_fallback...` | `find_existing_run_dir` — `io.py` 中不存在 |
| `test_find_existing_cluster_report_raises...` | `find_existing_cluster_report` — 同上 |

这些函数从未被实现。测试本身是面向一个未完成的设计写的。

**处理方式**：要么实现这些函数，要么删除这些测试。

---

## 三、测试逻辑失败（1 个文件，2/2 测试失败）

### 5. `tests/test_sector_matched_selection.py`

两个测试都断言 `MATCH_SPIN_SECTORS=true` 时选出的态的自旋扇区计数 **精确等于** 理论值。实际输出的 Counter 与预期不符：`(0.0, 1.0)` 扇区选出 4 个态而非预期的 3 个，`(0.0, 0.0)` 选出 1 个而非 2 个。

这属于物理逻辑 bug——`_prepare_selection_blocks` 或 `select_eigenstates` 在 `match_spin_sectors` 模式下的选态行为与预期不一致。

**处理方式**：需要调试 `hubbard.py:_prepare_selection_blocks` → `downfolding.select_eigenstates` 的选态路径，确认是测试预期有误还是选态逻辑有 bug。

---

## 汇总

| 类别 | 文件数 | 失败测试数 |
|------|--------|-----------|
| import 不存在的符号（无法加载） | 3 | 全部（收集阶段报错） |
| 引用未实现函数 | 1 | 5 |
| 逻辑断言失败 | 1 | 2 |
| **合计** | **5** | **10** |
