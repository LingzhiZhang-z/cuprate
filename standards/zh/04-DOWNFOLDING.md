# 04-降维

本征态选择、$T_{11}$ 投影度量、SVD 降维至 $H_{\text{eff}}$，
以及自旋耦合拟合。

## 1) 选择目标 (MUST)

MUST:
- 选择在一个已求解的 `Block` 上逐块进行。
- 对每个 block，必须正好选择 `block.spin_dim` 个本征态。
- `block.spin_dim` 是当前 block 坐标框架内 `D = 0` 自旋子空间的维度。
- `HubbardModel.project(method=...)` 对每个当前 block 应用同一种逐块选择方法，
  并在 `HubbardModel` 上保存 `selected_indices`、`selection_info`、`heff`
  和 `t11m1_norms`。

Code form:
```python
selected, info = block.selected(method=method, return_info=True, **kwargs)
heff, t11m1_norm = block.downfold(selected)
```

## 2) $T_{11}$ 投影度量 (MUST)

MUST:
- 设 `selected` 为 block 坐标框架中的本征向量列索引。
- 设 `spin_sector_columns = block.spin_sector_columns()` 为同一 block 框架中的 `D = 0` 列。
- 自旋投影矩阵为：

Code form:
```python
S_BD = block.eigvecs[np.ix_(spin_sector_columns, selected)]
U, sigma, VH = np.linalg.svd(S_BD, full_matrices=False)
T11m1 = U @ np.diag(sigma) @ U.conj().T - np.eye(S_BD.shape[0])
norm = np.linalg.norm(T11m1.flatten())
```

Validation:
- 小的 `norm` 意味着所选流形紧密张成自旋子空间。
- `norm == 0` 表示在数值容差内的完美投影。

## 3) 选择策略 (MUST)

MUST:
- 已实现的方法键为：

| 方法 | 键 | 规则 |
|------|----|------|
| 按占据 | `occ` | 先按最低 `<D>` 排序，再按能量、列索引排序 |
| 按能量 | `energy` | 先按最低能量排序，再按 `<D>`、列索引排序 |
| 贪心交换 | `greedy` | 从按占据排序的候选池开始，执行一次确定性交换扫描以降低 `block.t11_norm` |
| 多次贪心 | `greedy_multi` | 从 greedy 结果开始，随机化高 `D` 尾部，多次重新运行 greedy |
| 绝热 | `adiabatic` | 选择与前一次已选本征向量重叠最大的当前态 |

- `HubbardModel.project()` 默认 `method="occ"`。
- 未知方法必须报错。不存在从 occupation 到 greedy 的 `None` 回退。
- `greedy` 和 `greedy_multi` 的候选池来自按占据排序后的前
  `ratio * block.spin_dim` 个本征态。
- `greedy_multi` 可以通过 `selection_info_path` 或 `info_callback`
  为每个 block/trial 写一条 JSONL 记录。
- `adiabatic` 需要显式传入同一 block 框架的 `eigvecs_previous` 和
  `selected_previous` 数组。从前一参数点加载这些数组是 workchain 的责任，
  不是 `Block` 的责任。
- 如果缺少 adiabatic 种子数据，workchain 必须直接失败，除非用户请求的是其他选择方法。
  不能静默回退到同参数基线运行。

Code form:
```python
selected = block.selected_occ()
selected = block.selected_energy()
selected = block.selected_greedy(ratio=5)
selected = block.selected_greedy_multi(n_trials=40, max_failures=4)
selected = block.selected_adiabatic(eigvecs_previous, selected_previous)
```

## 4) 双占据期望值 (MUST)

MUST:
- 在基于 occupation 的选择前，必须在 Fock 坐标的本征向量中计算双占据算符的对角期望值：

Code form:
```python
dom = calc_double_occupation_matrix(block.basis_states, block.N)
eigvecs = block.eigvecs_fock
double_occ = np.real(np.diag(eigvecs.conj().T @ dom @ eigvecs))
```

## 5) 有效哈密顿量 $H_{\text{eff}}$ (MUST)

MUST:
- 给定规则 2 中的 SVD 和选定本征值
  `Lambda = diag(block.eigvals[selected])`，计算：

Code form:
```python
Lambda = np.diag(block.eigvals[selected])
Heff = U @ VH @ Lambda @ VH.conj().T @ U.conj().T
```

Validation:
- `H_eff` 在容差内必须是厄米的。
- `H_eff` 的谱必须等于选定的本征值。

## 6) 自旋算符基 (MUST)

MUST:
- 用于拟合的自旋算符基包括：
  1. 恒等算符 `I`（常数能量偏移）。
  2. 两格点 Heisenberg：`S_i . S_j`。
  3. 四格点乘积：`(S_i . S_j)(S_k . S_l)`。
  4. 六格点乘积：`(S_i . S_j)(S_k . S_l)(S_m . S_n)`。
- 算符构造在显式 `D = 0` Fock 行 `block.spin_fock_rows()` 上进行。
- 在 `S2` block 中，算符通过 `block.spin_sector_columns()` 和
  `block.basis_transform` 变换到当前 block 的自旋扇区。
- 多格点算符的输出顺序遵循 `08-OPERATOR_OUTPUT.md`。

Code form:
```python
spin_fock_rows = block.spin_fock_rows()
spin_sector_columns = block.spin_sector_columns()
operators = block._spin_operators(bonds)
```

## 7) 最小二乘拟合 (MUST)

MUST:
- `HubbardModel.fit()` 将每个当前 block 的 `H_eff` 展平为目标向量 `b`。
- 它将对应的逐块自旋算符列堆叠成 `A`。
- 它用 `np.linalg.lstsq` 求解 `A x = b`。
- 它报告相对误差、残差范数和 $R^2$。
- 如果没有传入 `bond_groups`，则拟合模型 `Cluster` 生成的所有当前两格点、
  四格点和六格点组。

Code form:
```python
A = np.vstack([block._spin_operators(bonds) for block in model.blocks])
b = np.concatenate([heff.flatten() for heff in model.heff])
x = np.linalg.lstsq(A, b, rcond=None)[0]
```

## 8) 持久化边界 (MUST)

MUST:
- `Block.save()` / `Block.load()` 只持久化已求解的本征系统 cache。
- 选定索引、选择诊断、`H_eff` 和拟合指标是派生输出，必须由 workchain/result-output 层写出。
- 派生 projection 和 fit 结果的规范机器可读形状定义在 `08-OPERATOR_OUTPUT.md`。

## 9) 解释规则 (MUST)

MUST:
- 小的拟合残差本身并不能验证自旋映射。
- `T11` 度量和拟合诊断必须结合解释。
- 若 `|T11-I|` 较大，则无论拟合残差多小，`H_eff` 都可能无法忠实代表低能物理。
