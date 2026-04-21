# 07-测试

回归测试策略、参考数据约定与命名转换规则。

## 1) 测试层级 (MUST)

MUST:
- 测试套件分为三个层级：

| 层级 | 标记 | 范围 | 触发条件 |
|------|------|------|----------|
| 单元 | (无) | 纯函数逻辑：基矢生成、排序、矩阵元、扇区计数 | 每次提交；`pytest tests/` |
| 集成 | `@pytest.mark.integration` | 端到端单团簇流水线：哈密顿量 → 对角化 → 选择 → H_eff → 拟合 | `pytest -m integration` |
| 回归 | `@pytest.mark.regression` | 与 `data_test/` 中重构前代码的参考数据进行数值比对 | `pytest -m regression` |

- 单元测试必须在单核上 30 秒内完成。
- 集成测试最多 5 分钟。
- 回归测试可能更长；需从磁盘读取大量 `.npy` 文件。

## 2) 参考数据布局 (MUST)

MUST:
- 参考数据位于 `data_test/` 下，使用**旧命名约定**（重构前）。
  不得重命名或重新格式化——作为冻结的 ground truth。
- 生产代码不提供对旧命名或格式的向后兼容。
  所有新旧转换由测试侧的 `ReferenceData` 工具负责。
- 目录结构：
  ```
  data_test/
  └── Block_U{U:.4f}_t{t:.4f}/
      ├── N{N}/                                     # 旧 MODE=full (TYPE=all)
      ├── N{N}_sz{Sz:.4f}/                          # 旧 MODE=fixed_sz
      ├── N{N}_adiabatic_restart/                   # 旧 workflow=adiabatic
      ├── N{N}_multi_restart/                       # 旧 workflow=multi
      ├── N{N}_sz{Sz:.4f}_adiabatic_restart/
      ├── N{N}_sz{Sz:.4f}_multi_restart/
      └── log_*.txt
  ```
- 每个运行目录中的逐团簇文件：
  ```
  hole{h}_class{c}_eigvals.npy
  hole{h}_class{c}_states.npy
  hole{h}_class{c}_Heff.npy
  hole{h}_class{c}_T11m1.npy
  hole{h}_class{c}_t11_selected_indices.npy
  hole{h}_class{c}_t11_selected_occupation.npy
  hole{h}_class{c}_double_occupation_expectation.npy
  hole{h}_class{c}_s2_digonal.npy                   # 注意：旧代码拼写错误 "digonal"
  hole{h}_class{c}_s2_selected.npy
  hole{h}_class{c}_cluster{v}_results.txt
  ```

## 3) 命名转换 (MUST)

MUST:
- 测试框架必须在新旧命名约定之间转换。
  参考数据保留旧名称；比较代码将其映射到当前语义。

### 3.1) 目录名

| 旧模式 | 新等价 | 转换规则 |
|--------|--------|----------|
| `Block_U{U}_t{t}` | `U{U}_t{t}` | 去掉 `Block_` 前缀 |
| `N{N}` | `N{N}` (MODE=full) | 不变 |
| `N{N}_sz{Sz:.4f}` | `N{N}_twoSz_{twoSz}` | `twoSz = int(2 * Sz)` |
| `N{N}_multi_restart` | `N{N}_multi_restart` | 工作流后缀不变 |
| `N{N}_adiabatic_restart` | `N{N}_adiabatic_restart` | 工作流后缀不变 |
| `N{N}_sz{Sz}_multi_restart` | `N{N}_twoSz_{twoSz}_multi_restart` | Sz 转换 + 后缀 |

### 3.2) 文件名

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `*_s2_digonal.npy` | `*_S2_diagonal.npy` | 拼写修正 + 大写 |
| `*_s2_selected.npy` | `*_S2_selected.npy` | 大写 |
| 其余 | 相同 | 不变 |

### 3.3) CLI 参数名

| 旧键 | 新键 | 转换 |
|------|------|------|
| `TYPE` | `MODE` | `TYPE=all` → `MODE=full`，`TYPE=sz` → `MODE=fixed_sz` |
| `SZ` | `twoSz` | `twoSz = int(2 * SZ)` |
| `S` | `twoS` | `twoS = int(2 * S)` |
| `WORKFLOW` | `workflow` | 仅大小写变化 |

Code form:
```python
def old_sz_to_twoSz(sz_str: str) -> int:
    """将旧的 'sz0.0000' 或 'sz0.5000' 转为整数 twoSz。"""
    sz_float = float(sz_str.replace("sz", ""))
    return int(round(2 * sz_float))

def old_dirname_to_new(dirname: str) -> str:
    """将旧运行目录名转为新约定。"""
    # N4_sz0.0000_multi_restart → N4_twoSz_0_multi_restart
    ...
```

## 4) 比较规则 (MUST)

MUST:
- 并非所有输出同等确定性。比较策略取决于物理量：

### 4.1) 确定性物理量（严格比较）

| 物理量 | 文件 | 容差 | 说明 |
|--------|------|------|------|
| 本征值 | `eigvals.npy` | `ATOL["tight"]` | 由哈密顿量完全确定 |
| 态 | `states.npy` | 精确（整数） | 基矢排序是规范的 |
| 双占据期望 | `double_occupation_expectation.npy` | `ATOL["tight"]` | 由本征向量导出，确定性 |
| S² 对角 | `s2_digonal.npy` / `S2_diagonal.npy` | `ATOL["loose"]` | 列 0: ⟨S²⟩，列 1: 方差 |

### 4.2) 选择依赖物理量（条件比较）

| 物理量 | 文件 | 条件 | 说明 |
|--------|------|------|------|
| 选定索引 | `t11_selected_indices.npy` | 仅比较 `occ` / `energy` 工作流 | greedy/multi/adiabatic 含随机性 |
| H_eff | `Heff.npy` | 仅当选定索引匹配时 | 给定相同选择，H_eff 确定 |
| T11-I | `T11m1.npy` | 仅当选定索引匹配时 | 同上 |
| 选定占据 | `t11_selected_occupation.npy` | 仅当选定索引匹配时 | 由选择导出 |

### 4.3) 非确定性物理量（仅范数比较）

| 物理量 | 比较方式 | 说明 |
|--------|----------|------|
| `multi` 工作流的 H_eff | `‖T11-I‖` 应 ≤ 旧值 + `ATOL["loose"]` | 随机重启可能找到不同最小值 |
| 绝热重叠 | 应 ≥ 旧值 - `ATOL["loose"]` | 不同选择可能导致差异 |

## 5) 测试用例选择 (MUST)

MUST:
- 回归测试不应比较全部 12.6 万文件。选择代表性用例：

### 5.1) 最小回归集

| 用例 | 参数 | 覆盖 |
|------|------|------|
| N=2, full | U=1, t=0.24 | 最小团簇，完整对角化，基线 |
| N=4, full | U=1, t=0.24 | 4 格点含空穴，同构类 |
| N=4, fixed_sz | U=1, t=0.24, Sz=0 | Sz 分块模式 |
| N=6, fixed_sz | U=1, t=0.24, Sz=0 | 更大团簇，更多扇区 |
| N=4, adiabatic | U=1, t=0.24 | 绝热选择（需 t=0.22 作为前一点） |

### 5.2) 扩展回归集

- t=0.24 下所有 N（2 到 7），所有模式。
- 边界参数点：t=0.02（强耦合）和 t=0.60（弱耦合）。
- N=8 fixed_sz（可用的最大尺寸，测试可扩展性）。

## 6) 测试基础设施 (MUST)

MUST:
- 共享 fixture 提供转换层：

Code form:
```python
# tests/conftest.py

DATA_TEST_DIR = Path(__file__).parent.parent / "data_test"

@pytest.fixture
def ref_data():
    """提供带自动命名转换的参考数据访问器。"""
    return ReferenceData(DATA_TEST_DIR)

class ReferenceData:
    def load_npy(self, U, t, run_dir_new, filename_new) -> np.ndarray:
        """加载参考 .npy 文件，将新名称转为旧路径。"""
        ...

    def has_case(self, U, t, run_dir_new) -> bool:
        """检查该用例是否有参考数据。"""
        ...
```

- 当 `data_test/` 不存在时回归测试优雅跳过（如无数据的 CI 环境）：

Code form:
```python
pytestmark = pytest.mark.regression

@pytest.fixture(autouse=True)
def skip_if_no_data():
    if not DATA_TEST_DIR.exists():
        pytest.skip("data_test/ 不存在")
```

## 7) 比较辅助函数 (MUST)

MUST:
- 提供可复用的断言辅助函数：

Code form:
```python
def assert_eigvals_match(new, ref, label=""):
    """严格本征值比较。"""
    np.testing.assert_allclose(np.sort(new), np.sort(ref),
                                atol=ATOL["tight"], err_msg=f"eigvals 不匹配 {label}")

def assert_heff_match(new, ref, label=""):
    """H_eff 比较（当选择匹配时）。"""
    np.testing.assert_allclose(new, ref, atol=ATOL["loose"], err_msg=f"Heff 不匹配 {label}")

def assert_indices_match(new, ref, label=""):
    """选定索引比较（排序后，仅限确定性方法）。"""
    np.testing.assert_array_equal(np.sort(new), np.sort(ref), err_msg=f"indices 不匹配 {label}")

def assert_t11_no_worse(new_norm, ref_norm, label=""):
    """非确定性选择：新 T11 范数不应显著恶化。"""
    assert new_norm <= ref_norm + ATOL["loose"], \
        f"T11 范数回归 {label}: new={new_norm}, ref={ref_norm}"
```

## 8) 集成测试结构 (MUST)

MUST:
- 集成测试在小团簇（N=2, N=3）上运行完整流水线，使用已知参数，
  无需参考数据。验证内部一致性：
  1. 哈密顿量是厄米的。
  2. 本征值为实数且有序。
  3. 本征向量正交归一。
  4. 扇区维度之和等于完整 Hilbert 空间。
  5. H_eff 是厄米的。
  6. 小 U/t 时自旋耦合拟合 R² > 0.99。
  7. 选定索引数等于 dimspin。

Code form:
```python
@pytest.mark.integration
def test_full_pipeline_N2():
    model = HubbardModel(2, 1.0, 0.3)
    # ... 构建、求解、降维、拟合 ...
    assert model.downfold.t11m1_norm < 1e-6
    assert model.downfold.heff is not None
```

## 9) 不比较的内容 (MUST)

MUST:
- 不比较：
  - `memory_status.txt`、`timing_status.txt` — 机器相关。
  - `log_*.txt` — 包含时间戳、MPI rank 信息。
  - `results.txt` 文本格式 — 格式可能变化；比较 `.npy` 数据。
  - 本征向量本身 — 规范依赖（相位、简并旋转）。
    比较本征向量导出量（本征值、⟨S²⟩、双占据）。
  - 文件存在性或数量 — 新代码可能产生额外文件（如 `.json` 产物）。
