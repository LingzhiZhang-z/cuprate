# Cuprate 代码结构简化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 9 个子包、26 个有效文件简化为 5 个顶层模块 + 3 个入口包，共 9 个文件。

**Architecture:** 所有基础设施（路径、CLI、JSON序列化、modes、params）合并到 `io.py`。物理模块（hubbard、spin、clusters）提升到顶层。三个入口（main、lce、embed）各自合并为单一 `__main__.py`。删除 plots/、monitor、txt 解析器。

**Tech Stack:** Python 3.9+, numpy, scipy, networkx, mpi4py

---

## Target Structure

```
cuprate/
  __init__.py
  mpi.py              # 不动（53行）
  hubbard.py           # ← models/hubbard.py + core/states.py + core/math_utils.py
  spin.py              # ← operators/spin.py（吸收 sort_s2）
  clusters.py          # ← clusters/square.py（改名）
  io.py                # ← shared/* + main/modes.py + main/params.py + lce共享函数
  main/
    __init__.py
    __main__.py        # ← main/__main__.py + cli.py + solver.py + reporter.py + workflow.py + space.py
  lce/
    __init__.py
    __main__.py        # ← lce/__main__.py + cli.py + utils.py（删txt解析器）
  embed/
    __init__.py
    __main__.py        # ← embed/__main__.py + cli.py + utils.py
```

## Deletion List

完成后删除的文件/目录：
- `src/cuprate/core/` （整个目录）
- `src/cuprate/shared/` （整个目录）
- `src/cuprate/models/` （整个目录）
- `src/cuprate/operators/` （整个目录）
- `src/cuprate/clusters/` （整个目录）
- `src/cuprate/plots/` （整个目录）
- `src/cuprate/main/cli.py`
- `src/cuprate/main/modes.py`
- `src/cuprate/main/params.py`
- `src/cuprate/main/monitor.py`
- `src/cuprate/main/solver.py`
- `src/cuprate/main/reporter.py`
- `src/cuprate/main/workflow.py`
- `src/cuprate/main/space.py`
- `src/cuprate/lce/cli.py`
- `src/cuprate/lce/utils.py`
- `src/cuprate/embed/cli.py`
- `src/cuprate/embed/utils.py`

## Function Migration Map

### → `cuprate/io.py`

| 来源 | 函数/类 |
|------|---------|
| `shared/cli.py` | `parse_bool_arg`, `iter_cli_assignments` |
| `main/modes.py` | 全部：`MODE_*` 常量, `RESULT_KIND_*`, `ModeSpec`, `resolve_mode_spec`, `nonnegative_sz_values`, `s_values_for_sz` |
| `main/params.py` | `Params` dataclass |
| `shared/io.py` | `PathSpec`, `build_path_spec`, `setup_work_environment`, `filter_work_items`, `setup_work_environment_previous`, `check_and_print_adiabatic_info`, `setup_params`, `read_key`, `write_file`, `load_array_compat`, `read_previous` |
| `shared/results.py` | 全部：`write_result_artifact`, `load_result_artifact`, `serialize_*`, `deserialize_*`, `build_*_artifact`, `artifact_to_operator_list`, `RESULT_SCHEMA_VERSION` |
| `shared/legacy_paths.py` | 全部：`build_legacy_result_dirs`, `find_existing_cluster_report`, `find_existing_run_dir`, `cluster_result_report_path` |
| `shared/bonds.py` | `find_bond_vector`, `write_cluster_points`, `get_all_possible_vectors` |
| `core/states.py` | `tune_sz`, `savefile`, `loadfile` |
| `lce/utils.py` | `k2s`, `k4s`, `k6s`, `k8s`, `k6s_all`, `create_dict`, `read_coords_file_Block`, `read_operators`, `read_fit_metric`, `parse_filename_Block`, `read_cluster_file_Block`, `_artifact_cluster_from_file`, `create_graph_from_cluster`, `find_cluster_match` |

### → `cuprate/hubbard.py`

| 来源 | 函数/类 |
|------|---------|
| `models/hubbard.py` | `Hubbard_SingleBand`, `compute_t11m1_norm` |
| `core/states.py` | `sum_elec`, `total_mag`, `find_index`, `is_half_filled`, `sign_fermi`, `sign_state`, `judge_state_same`, `judge_state_diff`, `Model_States_All`, `Model_States_Nele`, `Model_States_Nele_Sz`, `Model_States_Spin`, `Model_State_Sort`, `calc_double_occupation`, `calc_double_occupation_matrix`, `sort_states` |
| `core/math_utils.py` | `objective_function`, `derivative_objective_function`, `norm_matrix` |

### → `cuprate/spin.py`

| 来源 | 函数/类 |
|------|---------|
| `operators/spin.py` | 全部原有函数 |
| `core/states.py` | `sort_s2`（移入，原来是从 states 导入的） |

### → `cuprate/clusters.py`

| 来源 | 函数/类 |
|------|---------|
| `clusters/square.py` | 全部（原样搬移，改名） |

### → `cuprate/main/__main__.py`

| 来源 | 函数/类 |
|------|---------|
| `main/__main__.py` | `main()` + `if __name__` 块 |
| `main/cli.py` | `parse_arguments` |
| `main/solver.py` | `cluster_process_work_item`, `cluster_process`, `find_restart_path`, `_save_*` |
| `main/reporter.py` | 全部（txt 写入保留，txt 读取不涉及此模块） |
| `main/workflow.py` | `prepare_clusters_and_tasks`, `distribute_work` |
| `main/space.py` | `analyze_space`, `comb_safe`（此处的） |

### → `cuprate/lce/__main__.py`

| 来源 | 函数/类 |
|------|---------|
| `lce/__main__.py` | `main()` + `if __name__` 块 |
| `lce/cli.py` | `parse_arguments` |
| `lce/utils.py` | `setup_work_environment`, `get_connected_subgraphs`, `operator_minus`, `find_operator`, `write_operators`, `write_operators_LCE` |

**删除（不搬移）：** `parse_coupling_params`, `parse_coupling_params2`, `read_operators2`, `read_cluster_file`, `read_coords_file`, `parse_filename`, `print_subgraph`

### → `cuprate/embed/__main__.py`

| 来源 | 函数/类 |
|------|---------|
| `embed/__main__.py` | `main()` + `if __name__` 块 |
| `embed/cli.py` | `parse_arguments` |
| `embed/utils.py` | `setup_work_environment`, `coord_to_index`, `index_to_coord`, `generate_square_cell`, `unique_cluster_transformed`, `embed_cluster_to_squarecell_pbc`, `embed_cluster_to_squarecell_obc`, `embed_operator`, `bond_analysis_foursites`, `bond_analysis_sixsites`, `coupling_twosites_vector_from_matrix`, `coupling_foursites_vector_from_dict`, `coupling_sixsites_vector_from_dict`, `normalize_four_sites`, `normalize_six_sites`, `save_data_embed`, `write_couplings_embed`, `print_grid` |

**删除（不搬移）：** `coupling_foursites_vector_from_dict2`, `coupling_sixsites_vector_from_dict_simple`

## Import Rewrite Rules

所有内部导入和测试导入按以下规则改写：

| 旧 | 新 |
|----|-----|
| `from cuprate.models.hubbard import ...` | `from cuprate.hubbard import ...` |
| `from cuprate.operators.spin import ...` | `from cuprate.spin import ...` |
| `from cuprate.clusters.square import ...` | `from cuprate.clusters import ...` |
| `from cuprate.core.states import ...` | `from cuprate.hubbard import ...` 或 `from cuprate.io import tune_sz` |
| `from cuprate.core.math_utils import ...` | `from cuprate.hubbard import ...` |
| `from cuprate.shared.io import ...` | `from cuprate.io import ...` |
| `from cuprate.shared.results import ...` | `from cuprate.io import ...` |
| `from cuprate.shared.legacy_paths import ...` | `from cuprate.io import ...` |
| `from cuprate.shared.bonds import ...` | `from cuprate.io import ...` |
| `from cuprate.shared.cli import ...` | `from cuprate.io import ...` |
| `from cuprate.main.modes import ...` | `from cuprate.io import ...` |
| `from cuprate.main.params import ...` | `from cuprate.io import ...` |
| `from cuprate.main.cli import ...` | `from cuprate.main import ...` |
| `from cuprate.main.solver import ...` | `from cuprate.main import ...` |
| `from cuprate.main.reporter import ...` | `from cuprate.main import ...` |
| `from cuprate.main.workflow import ...` | `from cuprate.main import ...` |
| `from cuprate.main.monitor import ...` | 删除 |
| `from cuprate.lce.cli import ...` | `from cuprate.lce import ...` |
| `from cuprate.lce.utils import ...` | `from cuprate.io import ...` 或 `from cuprate.lce import ...` |
| `from cuprate.embed.cli import ...` | `from cuprate.embed import ...` |
| `from cuprate.embed.utils import ...` | `from cuprate.embed import ...` |

---

## Tasks

### Task 1: 创建 `cuprate/io.py`

**Files:**
- Create: `src/cuprate/io.py`

将所有基础设施代码合并到一个文件。按以下顺序组织：

- [ ] **Step 1: 合并所有源文件内容到 `io.py`**

文件组织结构（用注释分节）：

```python
# ============================================================
# CLI Helpers
# ============================================================
# ← shared/cli.py: parse_bool_arg, iter_cli_assignments

# ============================================================
# Modes & Params
# ============================================================
# ← main/modes.py: MODE_* 常量, RESULT_KIND_*, ModeSpec, resolve_mode_spec,
#    nonnegative_sz_values, s_values_for_sz, _normalize_half_integer,
#    _value_from_index, _index_from_value
# ← main/params.py: Params dataclass

# ============================================================
# Path Management
# ============================================================
# ← shared/io.py: PathSpec, build_path_spec, setup_work_environment,
#    filter_work_items, setup_work_environment_previous,
#    check_and_print_adiabatic_info, setup_params
# ← shared/legacy_paths.py: build_legacy_result_dirs,
#    cluster_result_report_path, find_existing_cluster_report,
#    find_existing_run_dir

# ============================================================
# Results Serialization (JSON)
# ============================================================
# ← shared/results.py: RESULT_SCHEMA_VERSION, write_result_artifact,
#    load_result_artifact, serialize_complex, deserialize_complex,
#    serialize_cluster, deserialize_cluster, _serialize_terms,
#    build_spin_coupling_artifact_from_coeffs,
#    build_spin_coupling_artifact_from_operator_list,
#    build_projection_analysis_artifact, artifact_to_operator_list

# ============================================================
# Bond Utilities
# ============================================================
# ← shared/bonds.py: find_bond_vector, write_cluster_points,
#    get_all_possible_vectors

# ============================================================
# File I/O Utilities
# ============================================================
# ← shared/io.py: read_key, write_file, load_array_compat, read_previous
# ← core/states.py: tune_sz, savefile, loadfile

# ============================================================
# Operator Key Functions
# ============================================================
# ← lce/utils.py: k2s, k4s, k6s, k8s, k6s_all

# ============================================================
# Data Reading (JSON-only, txt parsers deleted)
# ============================================================
# ← lce/utils.py: create_dict, parse_filename_Block,
#    read_cluster_file_Block, _artifact_cluster_from_file,
#    read_coords_file_Block, read_fit_metric,
#    create_graph_from_cluster, find_cluster_match
# ← 简化 read_operators: 只走 JSON 路径，删除 txt fallback
```

关键修改：
1. `build_path_spec` 原来 `from cuprate.main.modes import resolve_mode_spec` → 现在同文件内直接调用
2. `shared/results.py` 原来 `from cuprate.main.modes import RESULT_KIND_*` → 同文件内直接引用
3. `shared/results.py` 原来 `from cuprate.shared.bonds import ...` → 同文件内直接调用
4. `shared/legacy_paths.py` 原来 `from cuprate.shared.results import result_json_path` → 同文件内直接调用
5. `read_operators` 简化为：
```python
def read_operators(file_path: str):
    payload = load_result_artifact(file_path)
    if payload is not None:
        return artifact_to_operator_list(payload)
    raise FileNotFoundError(f"No JSON artifact found for {file_path}")
```
6. `find_cluster_match` 原来 `from cuprate.clusters.square import ...` → 改为 `from cuprate.clusters import ...`
7. 所有 `from cuprate.shared.*` / `from cuprate.main.modes` / `from cuprate.main.params` 的内部引用消除（同文件）

外部导入保留：
```python
import os, sys, glob, shutil, json, numpy as np, networkx as nx
from dataclasses import dataclass, field
from typing import Optional, Any
from collections.abc import Iterable, Iterator
from itertools import combinations
```

- [ ] **Step 2: 验证 io.py 可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.io import Params, PathSpec, resolve_mode_spec, read_operators; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/io.py
git commit -m "refactor: create unified io.py with all infrastructure code"
```

---

### Task 2: 创建 `cuprate/hubbard.py`

**Files:**
- Create: `src/cuprate/hubbard.py`

- [ ] **Step 1: 合并 models/hubbard.py + core/states.py + core/math_utils.py**

文件组织结构：

```python
# ============================================================
# Fock Space States (← core/states.py)
# ============================================================
# sum_elec, total_mag, find_index, is_half_filled, sign_fermi,
# sign_state, judge_state_same, judge_state_diff,
# Model_States_All, Model_States_Nele, Model_States_Nele_Sz,
# Model_States_Spin, Model_State_Sort, sort_states,
# calc_double_occupation, calc_double_occupation_matrix
#
# 注意：tune_sz, savefile, loadfile, sort_s2 不放这里
#   - tune_sz → io.py
#   - savefile, loadfile → io.py
#   - sort_s2 → spin.py
#   - test_sort_s2() 删除（不应在模块级别运行测试）

# ============================================================
# Math Utilities (← core/math_utils.py)
# ============================================================
# objective_function, derivative_objective_function, norm_matrix
#
# 删除 Chop, Matrix_Out（调试用，未被引用）

# ============================================================
# Hubbard Model (← models/hubbard.py)
# ============================================================
# compute_t11m1_norm, Hubbard_SingleBand
```

关键修改：
1. 删除所有 `from cuprate.core.states import ...` 和 `from cuprate.core.math_utils import ...`（同文件内直接调用）
2. `from cuprate.operators.spin import ...` → `from cuprate.spin import ...`
3. `from cuprate.main.modes import nonnegative_sz_values` → `from cuprate.io import nonnegative_sz_values`
4. `from cuprate.shared.io import setup_work_environment_previous, load_array_compat` → `from cuprate.io import setup_work_environment_previous, load_array_compat`

- [ ] **Step 2: 验证可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.hubbard import Hubbard_SingleBand, total_mag; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/hubbard.py
git commit -m "refactor: create unified hubbard.py with states and math utils"
```

---

### Task 3: 创建 `cuprate/spin.py`

**Files:**
- Create: `src/cuprate/spin.py`

- [ ] **Step 1: 搬移 operators/spin.py 并吸收 sort_s2**

从 `operators/spin.py` 全量复制，修改：
1. `from cuprate.core.states import sort_s2, calc_double_occupation, sort_states, total_mag` → 删除此导入
2. 从 `core/states.py` 复制 `sort_s2` 函数到此文件开头
3. `calc_double_occupation`, `sort_states`, `total_mag` 改为 `from cuprate.hubbard import calc_double_occupation, sort_states, total_mag`

- [ ] **Step 2: 验证可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.spin import compute_S2_matrix, sort_s2; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/spin.py
git commit -m "refactor: create top-level spin.py with sort_s2"
```

---

### Task 4: 创建 `cuprate/clusters.py`

**Files:**
- Create: `src/cuprate/clusters.py`

- [ ] **Step 1: 复制 clusters/square.py 到顶层并改名**

原样复制。此文件没有 cuprate 内部导入，无需修改导入。

- [ ] **Step 2: 验证可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.clusters import Clusters_Square, bond_analysis_spin; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/clusters.py
git commit -m "refactor: move clusters/square.py to top-level clusters.py"
```

---

### Task 5: 创建新的 `cuprate/main/__main__.py`

**Files:**
- Create: `src/cuprate/main/__main__.py` （覆盖原文件）

- [ ] **Step 1: 合并 main/ 下所有文件**

合并以下源文件到单一 `__main__.py`：
- `main/space.py` → `analyze_space()`, `comb_safe()`
- `main/workflow.py` → `prepare_clusters_and_tasks()`, `distribute_work()`
- `main/reporter.py` → 全部 txt 写入函数 + `cluster_save_results()`, `cluster_save_projection_results()`
- `main/solver.py` → `cluster_process()`, `cluster_process_work_item()`, `find_restart_path()`, `_save_*`
- `main/cli.py` → `parse_arguments()`
- `main/__main__.py` → `main()`, `if __name__` 块

导入重写：
```python
import os, sys, time, numpy as np
from datetime import datetime
from math import comb

from cuprate.mpi import comm, rank, size, is_root, barrier, broadcast
from cuprate.io import (
    Params, PathSpec, build_path_spec, resolve_mode_spec,
    setup_work_environment, filter_work_items, check_and_print_adiabatic_info,
    write_file, write_result_artifact, load_array_compat,
    build_spin_coupling_artifact_from_coeffs, build_projection_analysis_artifact,
    find_bond_vector, write_cluster_points, get_all_possible_vectors,
    iter_cli_assignments, parse_bool_arg, tune_sz,
    MODE_FULL, MODE_BLOCK_SZ_FULL, MODE_BLOCK_SZS2_FULL,
    RESULT_KIND_SPIN_COUPLINGS, RESULT_KIND_PROJECTION_ANALYSIS,
)
from cuprate.hubbard import Hubbard_SingleBand
from cuprate.clusters import Clusters_Square, bond_analysis_spin, canonical_bond_type
```

删除所有 `from cuprate.main.xxx import ...` 的内部引用（同文件）。
删除 `from cuprate.main.monitor import Monitor` 及所有 monitor 相关代码。

- [ ] **Step 2: 验证可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.main import parse_arguments, cluster_process_work_item; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/main/__main__.py
git commit -m "refactor: merge main/ into single __main__.py"
```

---

### Task 6: 创建新的 `cuprate/lce/__main__.py`

**Files:**
- Create: `src/cuprate/lce/__main__.py` （覆盖原文件）

- [ ] **Step 1: 合并 lce/ 下所有文件**

合并：
- `lce/cli.py` → `parse_arguments()`
- `lce/utils.py` → 仅保留 LCE 特有函数：`setup_work_environment`, `get_connected_subgraphs`, `operator_minus`, `find_operator`, `write_operators`, `write_operators_LCE`
- `lce/__main__.py` → `main()`, `if __name__` 块

**删除不搬移：**
- `parse_coupling_params` (60行)
- `parse_coupling_params2` (60行)
- `read_operators2`
- `read_cluster_file`, `read_coords_file`, `parse_filename`
- `print_subgraph`

导入重写：
```python
import os, sys, time
from datetime import datetime
import networkx as nx
from itertools import combinations

from cuprate.io import (
    iter_cli_assignments, parse_bool_arg, tune_sz,
    build_legacy_result_dirs, find_existing_cluster_report, find_existing_run_dir,
    read_key, write_file, read_fit_metric,
    read_coords_file_Block, create_dict, read_operators,
    write_result_artifact, build_spin_coupling_artifact_from_operator_list,
    write_cluster_points, find_bond_vector, get_all_possible_vectors,
    k2s, k4s, k6s, k8s,
)
from cuprate.clusters import bond_analysis_spin
```

- [ ] **Step 2: 验证可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.lce import parse_arguments; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/lce/__main__.py
git commit -m "refactor: merge lce/ into single __main__.py, remove txt parsers"
```

---

### Task 7: 创建新的 `cuprate/embed/__main__.py`

**Files:**
- Create: `src/cuprate/embed/__main__.py` （覆盖原文件）

- [ ] **Step 1: 合并 embed/ 下所有文件**

合并：
- `embed/cli.py` → `parse_arguments()`
- `embed/utils.py` → 所有 embed 特有函数
- `embed/__main__.py` → `main()`, `if __name__` 块

**删除不搬移：**
- `coupling_foursites_vector_from_dict2`（重复）
- `coupling_sixsites_vector_from_dict_simple`（空实现）

导入重写：
```python
import os, sys, time, numpy as np
from datetime import datetime
from math import sqrt
import networkx as nx

from cuprate.io import (
    iter_cli_assignments, parse_bool_arg, tune_sz,
    build_legacy_result_dirs, find_existing_cluster_report,
    read_coords_file_Block, create_dict, read_operators,
    write_cluster_points, get_all_possible_vectors,
    k2s, k4s, k6s, k8s, k6s_all,
)
from cuprate.clusters import transform, find_connected_sites
```

- [ ] **Step 2: 验证可导入**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -c "from cuprate.embed import parse_arguments; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/cuprate/embed/__main__.py
git commit -m "refactor: merge embed/ into single __main__.py"
```

---

### Task 8: 更新所有测试

**Files:**
- Modify: `tests/test_block_diag.py`
- Modify: `tests/test_embed_lce_cli.py`
- Modify: `tests/test_fixed_sz_s2_gauge.py`
- Modify: `tests/test_mode_eigensystems.py`
- Modify: `tests/test_mode_paths.py`
- Modify: `tests/test_mode_routing.py`
- Modify: `tests/test_sector_matched_selection.py`
- Modify: `tests/test_shared_periphery.py`
- Modify: `tests/test_sz0_output_and_projection.py`

- [ ] **Step 1: 按 Import Rewrite Rules 更新所有测试文件的导入**

具体改写：

**test_block_diag.py:**
```python
# 旧: from cuprate.models.hubbard import Hubbard_SingleBand
# 新:
from cuprate.hubbard import Hubbard_SingleBand
```

**test_embed_lce_cli.py:**
```python
# 旧:
# from cuprate.clusters.square import bond_analysis_spin
# from cuprate.embed.cli import parse_arguments as parse_embed_arguments
# from cuprate.lce.cli import parse_arguments as parse_lce_arguments
# from cuprate.lce.utils import read_coords_file_Block, read_operators, write_operators
# from cuprate.main.monitor import write_memory_status
# from cuprate.main.reporter import cluster_save_results
# 新:
from cuprate.clusters import bond_analysis_spin
from cuprate.embed import parse_arguments as parse_embed_arguments
from cuprate.lce import parse_arguments as parse_lce_arguments
from cuprate.io import read_coords_file_Block, read_operators
from cuprate.lce import write_operators
from cuprate.main import cluster_save_results
# 注意: write_memory_status 已删除。如果测试中有对应 test case，需要删除该测试。
```

**test_fixed_sz_s2_gauge.py:**
```python
# 旧:
# from cuprate.main.modes import resolve_mode_spec
# from cuprate.main.params import Params
# from cuprate.models.hubbard import Hubbard_SingleBand
# 新:
from cuprate.io import resolve_mode_spec, Params
from cuprate.hubbard import Hubbard_SingleBand
```

**test_mode_eigensystems.py:**
```python
# 旧:
# from cuprate.main.modes import resolve_mode_spec
# from cuprate.main.params import Params
# from cuprate.models.hubbard import Hubbard_SingleBand
# 新:
from cuprate.io import resolve_mode_spec, Params
from cuprate.hubbard import Hubbard_SingleBand
```

**test_mode_paths.py:**
```python
# 旧:
# from cuprate.main.params import Params
# from cuprate.shared.io import build_path_spec
# 新:
from cuprate.io import Params, build_path_spec
```

**test_mode_routing.py:**
```python
# 旧:
# from cuprate.main.cli import parse_arguments
# from cuprate.main.params import Params
# from cuprate.shared.io import build_path_spec
# 新:
from cuprate.main import parse_arguments
from cuprate.io import Params, build_path_spec
```

**test_sector_matched_selection.py:**
```python
# 旧:
# from cuprate.core.states import total_mag
# from cuprate.main.modes import resolve_mode_spec
# from cuprate.main.params import Params
# from cuprate.models.hubbard import Hubbard_SingleBand, comb_safe
# 新:
from cuprate.hubbard import total_mag, Hubbard_SingleBand
from cuprate.io import resolve_mode_spec, Params
from cuprate.spin import comb_safe
```

**test_shared_periphery.py:**
```python
# 旧:
# from cuprate.shared.cli import iter_cli_assignments, parse_bool_arg
# from cuprate.shared.legacy_paths import build_legacy_result_dirs
# from cuprate.shared.legacy_paths import find_existing_cluster_report
# from cuprate.shared.legacy_paths import find_existing_run_dir
# 新:
from cuprate.io import (
    iter_cli_assignments, parse_bool_arg,
    build_legacy_result_dirs, find_existing_cluster_report, find_existing_run_dir,
)
```

**test_sz0_output_and_projection.py:**
```python
# 旧:
# from cuprate.clusters.square import Clusters_Square, bond_analysis_spin, canonical_bond_type
# from cuprate.main.modes import resolve_mode_spec
# from cuprate.main.params import Params
# from cuprate.main.solver import cluster_process_work_item
# from cuprate.models.hubbard import Hubbard_SingleBand
# from cuprate.shared.io import PathSpec, build_path_spec, load_array_compat
# 新:
from cuprate.clusters import Clusters_Square, bond_analysis_spin, canonical_bond_type
from cuprate.io import resolve_mode_spec, Params, PathSpec, build_path_spec, load_array_compat
from cuprate.main import cluster_process_work_item
from cuprate.hubbard import Hubbard_SingleBand
```

- [ ] **Step 2: 删除引用 monitor 的测试**

检查 `test_embed_lce_cli.py` 中引用 `write_memory_status` 的测试函数，删除该测试函数（monitor 功能已移除）。

- [ ] **Step 3: 运行测试验证**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -m pytest tests/ -v`
Expected: 全部通过（可能少1个 monitor 相关的测试）

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: update all test imports for new structure"
```

---

### Task 9: 删除旧文件

**Files:**
- Delete: 所有旧目录和文件

- [ ] **Step 1: 删除旧的子包和文件**

```bash
# 删除整个旧子包
rm -rf src/cuprate/core/
rm -rf src/cuprate/shared/
rm -rf src/cuprate/models/
rm -rf src/cuprate/operators/
rm -rf src/cuprate/clusters/
rm -rf src/cuprate/plots/

# 删除 main/ 下的旧文件（保留 __init__.py 和新的 __main__.py）
rm -f src/cuprate/main/cli.py
rm -f src/cuprate/main/modes.py
rm -f src/cuprate/main/params.py
rm -f src/cuprate/main/monitor.py
rm -f src/cuprate/main/solver.py
rm -f src/cuprate/main/reporter.py
rm -f src/cuprate/main/workflow.py
rm -f src/cuprate/main/space.py

# 删除 lce/ 下的旧文件
rm -f src/cuprate/lce/cli.py
rm -f src/cuprate/lce/utils.py

# 删除 embed/ 下的旧文件
rm -f src/cuprate/embed/cli.py
rm -f src/cuprate/embed/utils.py
```

- [ ] **Step 2: 清理 __pycache__**

```bash
find src/cuprate -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
```

- [ ] **Step 3: 运行测试确认一切正常**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete old module structure"
```

---

### Task 10: 最终验证

- [ ] **Step 1: 确认文件结构**

```bash
find src/cuprate -name "*.py" | sort
```

Expected output:
```
src/cuprate/__init__.py
src/cuprate/clusters.py
src/cuprate/embed/__init__.py
src/cuprate/embed/__main__.py
src/cuprate/hubbard.py
src/cuprate/io.py
src/cuprate/lce/__init__.py
src/cuprate/lce/__main__.py
src/cuprate/main/__init__.py
src/cuprate/main/__main__.py
src/cuprate/mpi.py
src/cuprate/spin.py
```

- [ ] **Step 2: 运行全部测试**

Run: `cd /Users/lingzhi/Documents/Code/cuprate && PYTHONPATH=src python -m pytest tests/ -v`
Expected: 全部通过（减去 monitor 相关测试）

- [ ] **Step 3: 验证三个入口可运行**

```bash
PYTHONPATH=src python -m cuprate.main --help 2>&1 || true
PYTHONPATH=src python -m cuprate.lce --help 2>&1 || true
PYTHONPATH=src python -m cuprate.embed --help 2>&1 || true
```

Expected: 各入口正常响应（打印帮助或无报错退出）

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "refactor: complete structure simplification (26 files → 9 files)"
```
