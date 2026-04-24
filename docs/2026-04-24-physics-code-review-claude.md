# Physics Core Code Review — Claude → Codex

**日期**：2026-04-24
**审阅者**：Claude (Opus 4.7)
**请 Codex 做的事**：对本审阅结果做对抗性复审。请指出我的判断中站不住脚的地方、遗漏的问题、以及不同的设计取舍意见。**不要动代码**，只做分析反驳。

---

## 1. 审阅背景

用户最近把 workflow / io / embed / lce / main 子模块移到了 `src/cuprate/back/`，当前 `src/cuprate/` 的"物理核心"只剩 5 个文件：

| 文件 | 行数（约） | 职责 |
|------|-----------|------|
| `src/cuprate/clusters.py` | 330 | 方格子 cluster 枚举、点群同构分类、键分类 |
| `src/cuprate/states.py` | 276 | Fock 态 bit 编码、基生成、Sz/S²/双占据/跳跃原语 |
| `src/cuprate/sectors.py` | 219 | 由 S± 构造 (twoSz, twoS, D) 分块变换矩阵 |
| `src/cuprate/hubbard.py` | 454 | `HubbardModel`：构哈密顿、对角化、缓存、project、fit |
| `src/cuprate/manifold.py` | 522 | `Block` 容器 + 选态 + SVD downfold + spin-operator 拟合 |

依赖图无循环：
```
clusters ─┐
states ──┼─> sectors ─┐
         └─> manifold ┴─> hubbard
```

本次审阅的重点是**代码质量与实现优雅**，不是物理正确性回归。物理公式我逐式核对过（例如 `_lowering_coeff`、`_calc_diagonal_fourS2_element`、`_spin_pair` 的 S⁺S⁻ 系数），看起来都对。

---

## 2. 实现的优点（承认项，供 Codex 核验是否过誉）

1. **`states.py` 的 bit encoding 说明**（文件头 docstring）加 `apply_hop` 的 `(new_state, sign)` / `(None, 0)` 签名：`states.py:81-96`。原语层非常干净。
2. **`_MODE_SCOPES`**（`hubbard.py:41-48`）：把 6 个 solver 模式分解成 `(sz_scope, s2_scope) ∈ {none, one, all}²` 的笛卡尔积。`set_symmetry` 的分支全靠这个表塌缩。这是整份代码最聪明的一处。
3. **`states.py` 的 `spin_matrix = reduce(matmul, ...)`**（`states.py:217-220`）——以 `(bond[::2], bond[1::2])` 配对，然后 reduce matmul 到 S·S 乘积。
4. **`_lowering_coeff`**（`sectors.py:40-41`）：用整数标签写出 SU(2) 系数 `0.5 * sqrt((twoS + twoSz)(twoS - twoSz + 2))`，在 `sectors.py:71-80` 的注释里把量纲来源说透了。
5. **`canonical_form` + 8-op 点群**（`clusters.py:202-227`）。配合 `classify_isomorphic_clusters`（`clusters.py:279-314`）用 networkx 做同构，边界清晰。
6. **`HubbardModel` 的流式 API**：`set_symmetry().build_hamiltonians().solve().project().fit()`。pipeline 的概念分段和方法签名对应得好。

---

## 3. 主要问题（按严重度排序）

### 3.1 `manifold.Block` 是 god class

`src/cuprate/manifold.py:40-483`，单类约 440 行，20+ 方法，同时承担：

- **容器**：`N / nelec / basis_states / ham / eigvals / eigvecs / twoSz / twoS / basis_transform`
- **序列化**：`save / load / exists / label / _label / _fmt_value`（`manifold.py:65-144`）
- **诊断**：`eigenstate_twoSz / eigenstate_twoS / eigenstate_D / double_occ_expectation`（`manifold.py:146-164, 209-212`）
- **Sector 剖分**：`spin_fock_rows / spin_sector_columns / spin_dim`（`manifold.py:186-200`）
- **选态（5 种策略）**：`selected_occ / selected_energy / selected_greedy / selected_greedy_multi / selected_adiabatic / selected`（`manifold.py:214-449`）
- **Downfold / 拟合**：`downfold / t11_norm / _spin_operators`（`manifold.py:202-207, 451-483`）
- **内部 helpers**：`_selection_pool / _greedy_swap`（`manifold.py:230-269`）

概念上可以拆成"已解哈密顿量 + 特征对 + 诊断"的 `Spectrum` 类和真正的容器/分块类 `Block`，现在全挤在一起。

**我的结论**：这是整份代码最明显的非优雅点。请 Codex 评估我这个判断是否成立，或者有没有我没看见的内聚性支撑它"就该这么大"。

### 3.2 MODE_* 常量放错了层

`MODE_SINGLE / MODE_BY_SZ / MODE_BY_SZ_S2 / MODE_ONE_SZ / MODE_ONE_SZ_S2 / MODE_ONE_SZ_BY_S2` 定义在 `manifold.py:22-27`，被 `hubbard.py:12-20` 反过来 import。

这些是 **solver 模式**，不是 downfolding 概念。manifold 理论上不需要知道 solver 怎么分块。把它们移到 `hubbard.py` 或独立的 `modes.py` 更符合分层。请 Codex 判断这个依赖反向是否真的是问题。

### 3.3 `HubbardModel.project()` 产出平行列表

`hubbard.py:383-410`：
```python
self.selected_indices = []   # per block
self.selection_info = []     # per block
self.heff = []               # per block
self.t11m1_norms = []        # per block
```
四个列表一一对应 `self.blocks`，按位置对齐。按领域语义，这些结果属于每个 `Block`，应挂在 `Block.selected / Block.heff / Block.t11m1_norm` 上。现在这样 `fit()`（`hubbard.py:412-453`）在跨 block 拼矩阵时还要索引对齐。

另外构造函数（`hubbard.py:77-83`）一次性赋 7 个 `None` 属性：
```python
self.selected_indices: list[list[int]] | None = None
self.selection_info: list[dict] | None = None
self.heff: list[np.ndarray] | None = None
self.t11m1_norms: list[float] | None = None
self.bond_groups: list[list[Sequence[int]]] | None = None
self.coupling_coeffs: list | None = None
self.fit_metrics: tuple[float, float, float] | None = None
```
7 个"某一 pipeline 阶段之后才非 None"的字段，类型系统无法表达"现在到第几步"。

### 3.4 Pipeline 阶段检查散落

以下方法都以相同前置断言开头：

- `hubbard.py:214` `build_hamiltonians`: `if not getattr(self, "blocks", None): raise RuntimeError("call set_symmetry() first")`
- `hubbard.py:231` `load`
- `hubbard.py:244` `save`
- `hubbard.py:260` `solve`
- `hubbard.py:294` `merge_by_s2`
- `hubbard.py:348` `merge_by_sz`
- `hubbard.py:389` `project`

外加 `merge_by_s2`（`hubbard.py:294-344`）里还要自查 `ham / eigvals / eigvecs / twoSz / twoS / basis_transform` 共 6 项约 10 行断言。

Pipeline stage 如果做成 enum，在一处 `_require_stage(X)` 判断，或者每步返回新类型让 mypy 帮忙，都会更整洁。

### 3.5 `sectors.build_S2_sectors` 返回 5-元组

`sectors.py:113-126`：
```python
sector_blocks.append(
    (twoSz, twoS, D, grouped_states[(twoSz, D)], coeff_block)
)
```
这个 5-元组的解包在 `build_S2_transforms`（`sectors.py:148`）、`write_S2_blocks`（`sectors.py:186`）、`load_S2_blocks`（`sectors.py:215`）里重复 3 次，顺序完全按位置，换顺序就是 bug。升级成 `@dataclass` 更稳。

### 3.6 `Block.set_hamiltonian` 名字骗人

`manifold.py:166-170`：
```python
def set_hamiltonian(self, ham: np.ndarray) -> None:
    if self.basis_transform is None:
        self.ham = ham
    else:
        self.ham = self.basis_transform.conj().T @ ham @ self.basis_transform
```
"set" 有时是纯赋值、有时是基变换（Fock → sector）。`install_fock_hamiltonian` / `set_from_fock` 这类名字更准确地传达"输入是 Fock 坐标，我帮你旋到内部基"。

### 3.7 Selection info schema 不一致

`manifold.py:430-449` 的 `selected` 分发后，info 字段各不相同：

- `selected_occ` / `selected_energy`（`manifold.py:432-437`）：`{"t11m1_norm": ..., "overlap": None}`（临时构造）
- `selected_greedy`（`manifold.py:271-307`）：`{"method", "block", "initial_norm", "best_norm", "improved"}`
- `selected_greedy_multi`（`manifold.py:309-410`）：9 个字段 + `trials` 列表
- `selected_adiabatic`（`manifold.py:412-428`）：`{"t11m1_norm", "overlap": float}`

下游若按统一 schema 消费会出错。另外 `selected_greedy_multi` 里 "append json + flush + callback" 的 3 行模式重复了 4 次（`manifold.py:296-304, 370-375, 402-407` 等），应抽成 `_emit_record`。

---

## 4. 小的一致性问题

1. **`clusters.Cluster`**（`clusters.py:66-68`）：`hole: int = None` / `class_idx: int = None` / `cluster_idx: int = None` 类型注解说 `int` 但默认 `None`，应为 `int | None`。
2. **`ClusterSets.generate`**（`clusters.py:106`）：只取 `bond_classes[0][1]`（最近邻），忽略 `classify_two_site_bonds` 返回的其它距离类——但函数名暗示它能返回所有距离的键分组。
3. **`Cluster.generate_bonds(N=2, is_connected=False)`**（`clusters.py:77-87`）：返回**所有** 2-site pairing，不是 bond。名字把 "bond" 用窄了。
4. **`sectors._build_highest_weight`**（`sectors.py:45-54`）：docstring 写 `hw[(twoS, D)]`，实际 key 是 `(twoSz, D)`。对最高权块这两者数值相等但语义不一致，变量名用 `twoSz` 读起来别扭。
5. **`HubbardModel._s2_transforms`**（`hubbard.py:146-151`）：每次 `set_symmetry` 都重算全套多重态。`(N, nelec)` 定了就是纯函数，扫参时可以 cache。
6. **`Block.downfold` 和 `Block.t11_norm`**（`manifold.py:202-207, 451-468`）：做同一个 SVD 两次——小浪费，但语义独立可接受。
7. **`__init__.py`（`cuprate/__init__.py`）** 只导出 `ATOL`。`states.py` 没用到 `ATOL`，`clusters.py` 和 `hubbard.py` 也没用；实际只有 `sectors.py`（`_null_space` 用 `ATOL["loose"]`）和 `manifold.py`（`spin_sector_columns` 用 `ATOL["tight"]`）消费它。范围对，但从 `__init__` 拉这个常量略绕。

---

## 5. 总体判断

核心物理代码 **实现正确、命名合规、分工清晰**，是能长期维护的起点。不优雅的地方集中在：

- `Block` 太大；
- pipeline 的阶段管理靠 None 标志 + 散落断言；
- MODE 常量放错层；
- 若干输出类型用 tuple 应该升级成 dataclass；
- selection info schema 不统一。

都是"状态管理"和"边界划分"层面的代码债，不是物理错误。

建议动手顺序（**不动代码，仅供讨论**）：

1. (A) `sectors` 5-元组 → `@dataclass`；
2. (B) `MODE_*` 搬去 `hubbard.py` 或 `modes.py`；
3. (C) `Block` 拆出 selection / downfold 自由函数；
4. (D) `HubbardModel.project()` 的结果落回每个 `Block`；
5. (E) pipeline stage 做成 enum 统一校验。

---

## 6. 请 Codex 做对抗性复审

请逐条回答：

1. **上面哪些判断站不住脚？** 例如第 3.2 条（MODE_* 放错层）——如果 `manifold` 真的需要这些常量做 internal dispatch，这个批评就不成立。
2. **我漏了什么更严重的问题？** 尤其欢迎指出：数据流中的潜在竞态、数值稳定性隐患、复数与实数的混用、SVD 的分解使用是否鲁棒、H_eff 的 Hermitian 投影是否可能在 `downfold` 里丢掉小的反厄米分量等。
3. **我承认的"优雅之处"有没有虚夸？** 比如 `_MODE_SCOPES` 是不是其实也只是把复杂度藏在了一张表里，没有真的减少。
4. **你对"动手顺序"A-F 有不同意见吗？** 哪些应该不做、哪些顺序应该调整。

**不要**在这一轮里改代码。也不要围绕"测试有没有覆盖"来下结论——用户已经说过不关心 tests/。请聚焦代码结构、命名、边界、数值健壮性。
